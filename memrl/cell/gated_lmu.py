"""Gated multichannel LMU + Selective LMU — the IGAM thesis variants.

This module hosts two cells that share the same forward pass:
  - `GatedLMU`     : the canonical thesis cell (multichannel u, gated write,
                     dynamic readout, anti-collapse residual). All architectural
                     extension flags default OFF so a bare `GatedLMU(i, h)` is
                     numerically identical to the original thesis port.
  - `SelectiveLMU` : a subclass with every extension turned ON (salience gate,
                     Hadamard calibration, multi-scale theta, layer norm,
                     readout skip, orthogonal W_pre). This is the headline
                     cell for the Phase A paper.

Differences from canonical `LMU` (Voelker 2019, kept in `lmu.py`):
  - **Multichannel u_t** (shape ℝ^C, not scalar) — Legendre memory is a
    (D, C) matrix instead of a (D,) vector. Higher capacity.
    Toggleable via `multichannel: bool` (default True). When False, u
    collapses to (B, 1) via a learned scalar projection and the memory state
    becomes (B, K, D, 1) — matching canonical-LMU scalar bandwidth while
    keeping the gating / readout machinery intact.
  - **Gated write** controlled by `gate_type` ∈ {softsign_sum, tanh_product,
    none}.
  - **Dynamic readout** via W_query — attention pointer over Legendre coefs.
    Toggleable via `dynamic_readout: bool` (default True). When False, the
    readout uses a fixed `W_static: Linear(D, 1)` projection over the
    Legendre dim instead — a static linear pooling, like canonical LMU.
  - **Anti-collapse residual** to keep memory from stagnating once E_h
    predicts u_x perfectly.
  - **Innovation as side output** for Phase B's lifelong intrinsic reward.

  The three "baseline GatedLMU vs LMU" features (multichannel u, gated
  write, dynamic readout) are each toggleable independently, so the
  GatedLMU → LMU gradient can be ablated one feature at a time.

Extensions (each toggleable; all default OFF):

  * **Salience gate** (HiPPO Zoo §3.2, Goffinet et al. 2026). Per-scale
    multiplicative scalar `g_t = sigmoid(W_g · h_prev) ∈ (0, 1)^K` applied
    to the LegT recurrence: `Am ← Am · g_t`. Equivalent to data-dependent
    time-warping of the HiPPO basis. **Bounded above by 1** because `A_d`
    already provides the canonical HiPPO decay — amplification (g > 1)
    breaks ODE stability and explodes over 100+ step episodes (verified
    empirically: a `1 + tanh` formulation grows memory state 265× in a
    single PPO update). Bias initialized to +4 so sigmoid ≈ 0.98 at start
    (gentle decay → near-identity → cell learns selectivity).

  * **Hadamard calibration** (Le et al. 2025, "Stable Hadamard Memory" —
    adapted to bounded (0,1) for LMU compatibility). Per-element
    calibration `C(h_prev) = sigmoid(W_calib · h_prev) ∈ (0, 1)^{K×D×C}`
    applied to `Am ← Am ⊙ C(h_prev)`. More expressive than salience
    (per-coefficient, per-channel) at the cost of more params. Same
    bounded-decay rationale and bias init as the salience gate.

  * **Multi-scale theta** (HiPPO Zoo §3.4, Multiscale HiPPO). Maintain K
    memory banks at thetas `[θ · sf^k for k = -(K//2)..+(K//2)]` evolving
    independently with their own (A_d, B_d). A learned softmax mixer
    `softmax(W_mix · h_prev) ∈ Δ^K` combines them at every read. Solves
    the "what theta?" problem by letting the cell adapt across timescales.

  * **Layer norm** on h_prev before feeding to any of the data-dependent
    projections (`E_h`, `W_query`, `W_g`, `W_calib`, `W_mix`). Modern
    selective-SSM recipe; prevents gate saturation.

  * **Readout skip** of magnitude `readout_skip_scale` from `u_x` to
    `y_internal`. Prevents the memory-readout bottleneck from dominating
    early when `W_query` hasn't learned to point at the right coefs.

  * **Orthogonal `W_pre`** via `torch.nn.utils.parametrizations.orthogonal`.
    Restores the thesis's isometry property (`‖W_pre(innov)‖ = ‖innov‖`)
    without the Cayley optimizer hook. Identity init.

State:        {"h": (B, hidden_size), "m": (B, K, memory_size, input_size)}
              K = n_scales (defaults to 1, so m is (B, 1, D, C) by default;
              storage / einsum cost is negligible vs (B, D, C)).
Side outputs: {"innovation": (B, input_size)}
Output:       h_new (B, hidden_size)

Note on `W_pre` orthogonality:
    With `orthogonal_W_pre=False` (default) we use a plain Linear initialized
    to identity. The isometry property `‖W_pre(innov)‖ = ‖innov‖` holds
    approximately early in training and drifts gradually.
    With `orthogonal_W_pre=True`, the layer is wrapped in PyTorch's orthogonal
    parametrization (Householder-based by default) — exact isometry at every
    forward pass, no custom optimizer hook needed.

Note on spectral_norm:
    The thesis cell wraps E_h and W_h with `spectral_norm`. We don't — PyTorch's
    spectral_norm has mutable power-iteration state that breaks our
    "manual-step equals forward_sequence" test contract. The freebie LayerNorm
    serves a similar stability role.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from memrl.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask
from memrl.cell.lmu import _legt_zoh_matrices


GateType = Literal["softsign_sum", "tanh_product", "none"]


class GatedLMU(RecurrentCell):
    """Multichannel gated LMU with optional selective extensions.

    All extension flags default OFF, so a bare `GatedLMU(i, h)` matches the
    original IGAM thesis cell numerically.

    Args:
        input_size, hidden_size: standard.
        memory_size: Legendre basis dim D.
        theta: LegT memory window. With `n_scales > 1`, this is the *center*
            theta; banks are at `theta · scale_factor^k` for k in a range
            centered on 0.
        gate_type: 'softsign_sum' | 'tanh_product' | 'none' (anti-gating).
        residual_scale: weight of the `u_x.detach()` anti-collapse residual.

        layer_norm: apply LayerNorm to `h_prev` before all data-dependent
            projections (selective-SSM recipe).
        readout_skip_scale: scalar weight of a direct `u_x → y_internal` skip.
        orthogonal_W_pre: wrap `W_pre` in PyTorch's orthogonal parametrization.

        salience_gate: enable per-scale salience gate `g_t = 1 + tanh(W_g h̃)`.
        hadamard_calib: enable per-element Hadamard calibration on memory.
        n_scales: K, number of multi-scale memory banks (1 = no multi-scale).
        scale_factor: theta ratio between adjacent banks (default 2.0).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int = 32,
        theta: float = 100.0,
        gate_type: GateType = "softsign_sum",
        residual_scale: float = 0.05,
        # ── core ablation flags (all default ON for backward compat) ────────
        multichannel: bool = True,
        dynamic_readout: bool = True,
        # ── freebies (all default OFF) ──────────────────────────────────────
        layer_norm: bool = False,
        readout_skip_scale: float = 0.0,
        orthogonal_W_pre: bool = False,
        # ── extensions (all default OFF) ────────────────────────────────────
        salience_gate: bool = False,
        hadamard_calib: bool = False,
        n_scales: int = 1,
        scale_factor: float = 2.0,
    ) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)
        if gate_type not in ("softsign_sum", "tanh_product", "none"):
            raise ValueError(
                f"gate_type must be 'softsign_sum' | 'tanh_product' | 'none'; "
                f"got {gate_type!r}"
            )
        if n_scales < 1:
            raise ValueError(f"n_scales must be ≥ 1; got {n_scales}")
        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be > 0; got {scale_factor}")

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.theta = theta
        self.gate_type = gate_type
        self.residual_scale = residual_scale
        self.n_scales = n_scales
        self.scale_factor = scale_factor
        self.readout_skip_scale = float(readout_skip_scale)
        self.multichannel = multichannel
        self.dynamic_readout = dynamic_readout
        # Effective channel dim — input_size when multichannel, else 1 (scalar u).
        # All shape-bearing internals (e_x, E_h, W_pre, W_m, W_calib, memory state)
        # use _C so the same forward pass handles both regimes.
        self._C = input_size if multichannel else 1

        # ── HiPPO-LegT matrices, one per scale ──────────────────────────────
        # For K=3, sf=2: thetas = [θ/2, θ, 2θ]
        # For K=1:        thetas = [θ]
        offsets = list(range(-(n_scales // 2), n_scales - n_scales // 2))
        thetas = [float(theta) * (scale_factor ** k) for k in offsets]
        self.thetas = thetas
        A_list, B_list = [], []
        for th in thetas:
            A_d, B_d = _legt_zoh_matrices(memory_size, th)
            A_list.append(A_d)
            B_list.append(B_d)
        self.register_buffer("A", torch.stack(A_list, dim=0))           # (K, D, D)
        self.register_buffer("B", torch.stack(B_list, dim=0))           # (K, D, 1)

        # ── Encoder parameters ──────────────────────────────────────────────
        # Multichannel: u_x = x * normalize(e_x), shape (B, C) — per-channel.
        # Scalar:       u_x = e_x_scalar(x),     shape (B, 1) — canonical LMU style.
        if multichannel:
            self.e_x: Optional[nn.Parameter] = nn.Parameter(torch.empty(input_size))
            self.e_x_scalar: Optional[nn.Linear] = None
        else:
            self.e_x = None
            self.e_x_scalar = nn.Linear(input_size, 1, bias=False)
        self.E_h = nn.Linear(hidden_size, self._C, bias=False)
        # Per-scale pooling over Legendre dim: (K, D). Outputs (B, K, _C) via einsum.
        self.e_m = nn.Parameter(torch.zeros(n_scales, memory_size))

        # ── W_pre: pre-write transform on the innovation vector ─────────────
        if gate_type != "none":
            self.W_pre: Optional[nn.Linear] = nn.Linear(self._C, self._C, bias=False)
        else:
            self.W_pre = None
        self._orthogonal_W_pre = orthogonal_W_pre and self.W_pre is not None

        # ── Readout: dynamic (W_query attention) or static (fixed W_static) ─
        if dynamic_readout:
            self.W_query: Optional[nn.Linear] = nn.Linear(hidden_size, memory_size, bias=False)
            self.W_static: Optional[nn.Linear] = None
        else:
            self.W_query = None
            # Static linear pooling over the Legendre dim. Parameterized as
            # Linear(D, 1) so the readout is "look at this fixed combination of
            # Legendre coefficients every step," analogous to canonical LMU's
            # W_m·m read but factored out from the hidden update.
            self.W_static = nn.Linear(memory_size, 1, bias=False)

        # ── Hidden update kernels ───────────────────────────────────────────
        self.W_x = nn.Linear(input_size, hidden_size, bias=True)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_m = nn.Linear(self._C, hidden_size, bias=False)

        # ── LayerNorm (freebie) ─────────────────────────────────────────────
        self.ln_h: nn.Module = (
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity()
        )

        # ── Multi-scale mixer (only if K > 1) ───────────────────────────────
        # Output (B, K) → softmax → weights over scales. Same mixer reused at
        # u_m pooling AND the dynamic readout (same context, same "which
        # timescale matters now" decision).
        if n_scales > 1:
            self.W_mix: Optional[nn.Linear] = nn.Linear(hidden_size, n_scales, bias=True)
        else:
            self.W_mix = None

        # ── Salience gate (extension A) ─────────────────────────────────────
        # Per-scale leak factor `g = sigmoid(W_g h + b)` in (0, 1). Bias
        # init to +4 so sigmoid ≈ 0.98 at start (near-identity decay; cell
        # learns selectivity via gradient on weight + bias).
        # **Bounded above by 1** because A_d already provides the canonical
        # HiPPO decay; amplification breaks ODE stability (verified
        # empirically: a `1 + tanh` formulation grew memory state 265× in
        # a single PPO update).
        if salience_gate:
            self.W_g: Optional[nn.Linear] = nn.Linear(hidden_size, n_scales, bias=True)
        else:
            self.W_g = None

        # ── Hadamard calibration (extension B) ──────────────────────────────
        # Per-element leak factor `calib = sigmoid(W_calib h + b)` in
        # (0, 1)^{K × D × C}. Same bounded-decay rationale and bias init as
        # the salience gate. Param count: hidden_size · (K · D · C). For
        # (H=64, K=3, D=64, C=64) that's ~786k weights + 6144 biases —
        # large but workable.
        if hadamard_calib:
            self.W_calib: Optional[nn.Linear] = nn.Linear(
                hidden_size, n_scales * memory_size * self._C, bias=True,
            )
        else:
            self.W_calib = None

        self.reset_parameters()

        # Apply orthogonal parametrization AFTER reset_parameters: the
        # underlying parameter is initialized to identity (which is already
        # orthogonal), so the parametrization preserves it.
        if self._orthogonal_W_pre:
            torch.nn.utils.parametrizations.orthogonal(self.W_pre)

    def reset_parameters(self) -> None:
        # Encoder
        if self.e_x is not None:
            nn.init.uniform_(self.e_x, -1.0, 1.0)
        if self.e_x_scalar is not None:
            # Canonical LMU uses LeCun uniform on its scalar input projection.
            nn.init.xavier_uniform_(self.e_x_scalar.weight)
        nn.init.xavier_normal_(self.E_h.weight)
        # e_m stays at zero: memory contributes nothing to u at init.

        # W_pre: identity init. Orthogonal parametrization (if used) preserves
        # eye since identity is orthogonal.
        if self.W_pre is not None:
            nn.init.eye_(self.W_pre.weight)

        # Readout: small gain so cell starts with negligible memory
        # influence on h; learns the pointer over training.
        if self.W_query is not None:
            nn.init.orthogonal_(self.W_query.weight, gain=0.01)
        if self.W_static is not None:
            # Same small-gain spirit so the static readout starts near zero.
            nn.init.orthogonal_(self.W_static.weight, gain=0.01)

        # Hidden update kernels: Xavier on weights, zero on bias.
        for layer in (self.W_x, self.W_h, self.W_m):
            nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(self.W_x.bias)

        # Multi-scale mixer: zero-init for near-uniform softmax at start.
        if self.W_mix is not None:
            nn.init.zeros_(self.W_mix.weight)
            nn.init.zeros_(self.W_mix.bias)

        # Salience gate: zero-init weight + bias=+8 → sigmoid(8) ≈ 0.9997 at
        # start. Near-identity decay: cell behaves essentially like vanilla
        # GatedLMU at init, then opens/closes gates only when training signal
        # accumulates. Bias=4 was tried earlier but caused 0.98^155 = 0.044
        # decay per Hard-length episode (vs 0.95 at bias=8) — too aggressive,
        # cell underperformed baseline. Trade-off: sigmoid'(8) ≈ 3e-4 means
        # slower gate learning, but starting near-identity is more important
        # for long-episode RL than fast gate adaptation.
        if self.W_g is not None:
            nn.init.zeros_(self.W_g.weight)
            nn.init.constant_(self.W_g.bias, 8.0)

        # Hadamard calibration: same init scheme as salience gate.
        if self.W_calib is not None:
            nn.init.zeros_(self.W_calib.weight)
            nn.init.constant_(self.W_calib.bias, 8.0)

    # ── gates ──────────────────────────────────────────────────────────────

    @staticmethod
    def _softsign(x: Tensor) -> Tensor:
        return x / (1.0 + x.abs())

    def _compute_write(
        self, u_x: Tensor, u_h: Tensor, u_m: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (u_actual, pred, innov_vec). innov_vec is the side output."""
        pred = u_h + u_m

        if self.gate_type == "softsign_sum":
            gate = self._softsign(u_x)
            innov = self._softsign(pred - u_x)
            innovation_vec = gate + innov
            u_actual = self.W_pre(innovation_vec) + pred

        elif self.gate_type == "tanh_product":
            gate = torch.tanh(u_x)
            innov = torch.tanh(pred - u_x)
            innovation_vec = gate * innov
            u_actual = self.W_pre(innovation_vec) + pred

        else:  # 'none'
            u_actual = u_x + pred
            innovation_vec = torch.zeros_like(u_x)

        if self.gate_type != "none" and self.residual_scale > 0.0:
            u_actual = u_actual + self.residual_scale * u_x.detach()

        return u_actual, pred, innovation_vec

    # ── interface ──────────────────────────────────────────────────────────

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        device = self._resolve_device(device)
        return {
            "h": torch.zeros(
                batch_size, self.hidden_size, device=device, dtype=dtype,
            ),
            "m": torch.zeros(
                batch_size, self.n_scales, self.memory_size, self._C,
                device=device, dtype=dtype,
            ),
        }

    def step(
        self,
        x: Tensor,
        state: State,
        episode_start: Optional[Tensor] = None,
    ) -> tuple[Tensor, State, SideOutputs]:
        if episode_start is not None:
            state = apply_episode_mask(state, episode_start)

        h_prev = state["h"]                                            # (B, H)
        m_prev = state["m"]                                            # (B, K, D, C)

        # Pre-norm h for all data-dependent projections (freebie).
        h_normed = self.ln_h(h_prev)                                   # (B, H)

        # ── 1) encode ──────────────────────────────────────────────────────
        if self.multichannel:
            e_x_n = F.normalize(self.e_x, dim=0)                        # (C,)
            u_x = x * e_x_n                                             # (B, C)
        else:
            u_x = self.e_x_scalar(x)                                    # (B, 1)
        u_h = self.E_h(h_normed)                                        # (B, _C)

        # Per-scale memory pooling, then mix over scales.
        u_m_per_scale = torch.einsum("kd,bkdc->bkc", self.e_m, m_prev)  # (B, K, C)
        if self.W_mix is not None:
            mix = F.softmax(self.W_mix(h_normed), dim=-1)               # (B, K)
            u_m = torch.einsum("bk,bkc->bc", mix, u_m_per_scale)        # (B, C)
        else:
            mix = None  # type: ignore[assignment]
            u_m = u_m_per_scale.squeeze(1)                              # (B, C)

        # ── 2) gated write ──────────────────────────────────────────────────
        u_actual, _pred, innovation_vec = self._compute_write(u_x, u_h, u_m)

        # ── 3) memory update ───────────────────────────────────────────────
        # A · m_prev across all K banks at once.
        Am = torch.einsum("kij,bkjc->bkic", self.A, m_prev)             # (B, K, D, C)

        # Salience gate: per-scale leak factor in (0, 1), broadcast to
        # (B, K, 1, 1). Bounded above by 1 — see module docstring for the
        # ODE-stability rationale.
        if self.W_g is not None:
            g = torch.sigmoid(self.W_g(h_normed))                       # (B, K)
            Am = Am * g.unsqueeze(-1).unsqueeze(-1)

        # Hadamard calibration: per-element leak factor in (0, 1)^{K, D, _C}.
        if self.W_calib is not None:
            calib = torch.sigmoid(self.W_calib(h_normed)).view(
                -1, self.n_scales, self.memory_size, self._C,
            )                                                           # (B, K, D, _C)
            Am = Am * calib

        # Write: B is (K, D, 1), u_actual is (B, C). Broadcast to (B, K, D, C).
        Bu = self.B.unsqueeze(0) * u_actual.unsqueeze(1).unsqueeze(2)   # (B, K, D, C)
        m_new = Am + Bu                                                 # (B, K, D, C)

        # ── 4) readout (per-scale, then mix) ───────────────────────────────
        # Dynamic: C_t = normalize(W_query · h_normed)        — attention pointer over Legendre dim
        # Static:  pool m_new over D with fixed W_static       — data-independent linear readout
        if self.dynamic_readout:
            C_t = F.normalize(self.W_query(h_normed), dim=-1)           # (B, D)
            y_per_scale = torch.einsum("bd,bkdc->bkc", C_t, m_new)      # (B, K, _C)
        else:
            # W_static.weight shape (1, D); contract over D → (B, K, _C).
            y_per_scale = torch.einsum(
                "d,bkdc->bkc", self.W_static.weight.squeeze(0), m_new,
            )                                                           # (B, K, _C)
        if mix is not None:
            y_internal = torch.einsum("bk,bkc->bc", mix, y_per_scale)   # (B, _C)
        else:
            y_internal = y_per_scale.squeeze(1)                         # (B, _C)

        # Readout skip (freebie).
        if self.readout_skip_scale > 0.0:
            y_internal = y_internal + self.readout_skip_scale * u_x

        # ── 5) hidden update ───────────────────────────────────────────────
        h_new = torch.tanh(
            self.W_x(x) + self.W_h(h_prev) + self.W_m(y_internal)
        )                                                               # (B, H)

        return h_new, {"h": h_new, "m": m_new}, {"innovation": innovation_vec}


class SelectiveLMU(GatedLMU):
    """Headline cell for Paper 1 — lean selective multichannel LMU.

    Equivalent to:
        GatedLMU(..., layer_norm=True, readout_skip_scale=0.1,
                 orthogonal_W_pre=True, salience_gate=True,
                 hadamard_calib=False, n_scales=3, scale_factor=2.0)

    Design choices and why:
      - **Multi-scale K=3** banks at thetas {θ/2, θ, 2θ} — principled
        coverage of ~4x dynamic range around the central theta (HiPPO Zoo
        §3.4, Goffinet et al. 2026). The structural extension that lets the
        cell adapt across timescales.
      - **Salience gate** (HiPPO Zoo §3.2) with bias=+8 init — per-scale
        leak factor in (0, 1)^K, near-identity at start so the cell starts
        from baseline GatedLMU behavior and *learns* selectivity only as
        useful. Adds ~K·H ≈ 200 params.
      - **No Hadamard calibration.** The (K, D, C)-shaped Hadamard gate
        from the first SelectiveLMU draft cost ~786k params (35x the
        baseline) AND its compound decay (sigmoid(4)^155 ≈ 4%) caused
        Hard-task underperformance vs baseline. A leaner version (per-D
        instead of per-D-C, or low-rank factorization) is sketched for a
        future ablation. For the headline, salience-only is the cleaner
        story.
      - **Freebies**: LayerNorm on h_prev (modern selective-SSM recipe),
        readout_skip=0.1 (prevents memory-readout bottleneck), orthogonal
        W_pre (exact isometry of innovation projection).

    Param overhead vs baseline GatedLMU at (H=64, mem=32):
      - GatedLMU baseline: ~25k params
      - SelectiveLMU lean: ~130k params (~5x baseline)
      Dominated by W_pre, W_x/W_h/W_m, and the 3x larger memory state.
      Defensible param count for the paper's headline comparison.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int = 32,
        theta: float = 100.0,
        gate_type: GateType = "softsign_sum",
        residual_scale: float = 0.05,
        readout_skip_scale: float = 0.1,
        n_scales: int = 3,
        scale_factor: float = 2.0,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            memory_size=memory_size,
            theta=theta,
            gate_type=gate_type,
            residual_scale=residual_scale,
            # Lean headline: freebies + multi-scale + salience. No Hadamard.
            layer_norm=True,
            readout_skip_scale=readout_skip_scale,
            orthogonal_W_pre=True,
            salience_gate=True,
            hadamard_calib=False,
            n_scales=n_scales,
            scale_factor=scale_factor,
        )
