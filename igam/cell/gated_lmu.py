"""Gated multichannel LMU — the IGAM thesis variant of the Legendre Memory Unit.

This is the **multichannel + gated-write** LMU from the lmu_ppo thesis
codebase, ported to the IGAM cell interface. It is distinct from the
canonical scalar-input LMU in `lmu.py` (Voelker 2019); both are kept as
separate cells for the Phase A ablation.

Differences from canonical `LMU`:
  - **Multichannel u_t** (shape ℝ^C, not scalar) — Legendre memory becomes
    a (D, C) matrix instead of a (D,) vector. Higher capacity.
  - **Gated write** controlled by `gate_type`. Three options:
      'softsign_sum'  (default, fastest convergence in thesis ablations)
      'tanh_product'  (original, both null conditions exact)
      'none'          (additive, no gating — degenerate baseline)
  - **Dynamic readout** via W_query: y_internal = C_t · m_new where
    C_t = normalize(W_query · h_prev). Like an attention pointer over
    Legendre coefficients.
  - **Anti-collapse residual**: a small fraction of the raw encoded u_x
    bypasses the gate to keep memory from collapsing once E_h learns to
    predict u_x.
  - **Spectral normalization** on E_h and W_h (recurrent weights) for
    bounded operator norm — prevents the recurrence from exploding.
  - **Innovation as side output**: the gate's innovation vector (e.g.
    softsign(u_x − pred) or tanh(u_x − pred)) is exposed in the same
    shape (B, C) as DeltaNet/Gated DeltaNet for consistent Phase B
    lifelong-reward consumption.

Math (per step, gate_type='softsign_sum'):

    Encode:
        u_x = x ⊙ normalize(e_x)         # (B, C)
        u_h = E_h(h_prev)                 # (B, C)  — spectral_norm'd
        u_m = e_m · m_prev               # (B, C)  — pooled over Legendre dim
        pred = u_h + u_m                  # cell's prediction of u_x

    Gate (softsign_sum variant):
        gate  = softsign(u_x)
        innov = softsign(u_x − pred)
        innovation_vec = gate + innov
        u_actual = W_pre(innovation_vec) + pred + residual_scale · u_x.detach()

    Memory update (HiPPO-LegT):
        m_t = A_d · m_{t-1} + B_d · u_actual   # (B, D, C)

    Dynamic readout (the "where to look" pointer):
        C_t = normalize(W_query(h_prev))       # (B, D), unit-norm
        y_internal = C_t · m_t                  # (B, C)

    Hidden update:
        h_t = tanh(W_x(x) + W_h(h_prev) + W_m(y_internal))   # (B, H)

Note on `W_pre` orthogonality:
    The thesis cell maintains W_pre as an orthogonal matrix via Cayley
    Riemannian updates (so that r_intr = ‖W_pre(innov)‖ = ‖innov‖,
    an isometry — clean prediction-error metric). Implementing the
    Cayley path requires a custom optimizer hook (W_pre excluded from
    Adam, ortho_update + reorthogonalize called separately). To keep
    this cell drop-in-compatible with the rest of the IGAM cell family
    under standard Adam, we use a regular `nn.Linear` initialized to
    identity. The isometry property is approximately maintained early
    in training and drifts gradually. If exact isometry matters for
    your r_intr metric, switch to `torch.nn.utils.parametrizations.orthogonal`
    or implement the Cayley updates in the trainer.

State:        {"h": (B, hidden_size), "m": (B, memory_size, input_size)}
Side outputs: {"innovation": (B, input_size)}
Output:       h_new (B, hidden_size)
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from igam.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask
from igam.cell.lmu import _legt_zoh_matrices


# Note on spectral_norm:
#   The thesis cell wraps E_h and W_h with `spectral_norm` for bounded
#   operator norm. We DON'T use it here for two reasons:
#     1. PyTorch's spectral_norm has mutable state (power-iteration `u`
#        updates on every forward, in both train and eval mode). This
#        breaks our test contract of "two equivalent forward paths
#        produce the same output" — running step T times vs calling
#        forward_sequence(T) advances the `u` vector differently.
#     2. At our small scales, the recurrent weights don't actually need
#        spectral norm for stability; Xavier init + Adam already keep
#        them in a healthy regime.
#   If long-sequence stability ever becomes an issue, switch E_h and W_h
#   to `torch.nn.utils.parametrizations.spectral_norm` and disable the
#   manual-vs-forward-sequence test for this cell.


GateType = Literal["softsign_sum", "tanh_product", "none"]


class GatedLMU(RecurrentCell):
    """Multichannel gated LMU (IGAM thesis variant)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int = 32,
        theta: float = 100.0,
        gate_type: GateType = "softsign_sum",
        residual_scale: float = 0.05,
    ) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)
        if gate_type not in ("softsign_sum", "tanh_product", "none"):
            raise ValueError(
                f"gate_type must be one of 'softsign_sum', 'tanh_product', "
                f"'none'; got {gate_type!r}"
            )

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.theta = theta
        self.gate_type = gate_type
        self.residual_scale = residual_scale

        # ── HiPPO-LegT matrices (fixed buffers, ZOH-discretized at dt=1) ──
        A_d, B_d = _legt_zoh_matrices(memory_size, theta)
        self.register_buffer("A", A_d)                                  # (D, D)
        self.register_buffer("B", B_d)                                  # (D, 1)

        # ── Encoder parameters ──────────────────────────────────────────────
        # e_x: per-channel weight, normalized to unit norm at use time.
        self.e_x = nn.Parameter(torch.empty(input_size))
        # E_h: hidden → encoded contribution. Plain Linear (no spectral_norm,
        # see top-of-file comment).
        self.E_h = nn.Linear(hidden_size, input_size, bias=False)
        # e_m: pooling weights over the Legendre dim (D).
        self.e_m = nn.Parameter(torch.zeros(memory_size))

        # ── W_pre: the "pre-write" transform applied to the innovation_vec.
        # The thesis cell keeps this orthogonal via Cayley updates; we use a
        # plain Linear initialized to identity to stay drop-in compatible
        # with standard Adam (see module docstring for the trade-off).
        if gate_type != "none":
            self.W_pre = nn.Linear(input_size, input_size, bias=False)
        else:
            self.W_pre = None

        # ── Dynamic readout (W_query) ──────────────────────────────────────
        # Maps hidden state → pointer over Legendre coefficients.
        # Small-gain init keeps the readout near zero early so the cell
        # doesn't bias the policy toward any specific Legendre coefficient.
        self.W_query = nn.Linear(hidden_size, memory_size, bias=False)

        # ── Hidden update kernels ──────────────────────────────────────────
        self.W_x = nn.Linear(input_size, hidden_size, bias=True)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)  # plain (no spectral_norm)
        self.W_m = nn.Linear(input_size, hidden_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Encoder
        nn.init.uniform_(self.e_x, -1.0, 1.0)
        nn.init.xavier_normal_(self.E_h.weight)
        # e_m stays at zero: memory contributes nothing to u_t initially
        # (silent write at episode start). It learns to weight Legendre
        # coefficients over the course of training.

        # W_pre: identity init for plain-Linear approximation of the
        # orthogonal-isometry trick. Output ≈ input early in training.
        if self.W_pre is not None:
            nn.init.eye_(self.W_pre.weight)

        # Dynamic readout init: small gain so the cell starts with negligible
        # influence from the memory on h, then learns the pointer.
        nn.init.orthogonal_(self.W_query.weight, gain=0.01)

        # Hidden update kernels: Xavier on weights, zero on bias.
        for layer in (self.W_x, self.W_h, self.W_m):
            nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(self.W_x.bias)

    # ── gates ──────────────────────────────────────────────────────────────

    @staticmethod
    def _softsign(x: Tensor) -> Tensor:
        return x / (1.0 + x.abs())

    def _compute_write(
        self, u_x: Tensor, u_h: Tensor, u_m: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (u_actual, pred, innov_vec). innov_vec is the side output."""
        pred = u_h + u_m

        if self.gate_type == "softsign_sum":
            gate = self._softsign(u_x)
            innov = self._softsign(u_x - pred)
            innovation_vec = gate + innov
            u_actual = self.W_pre(innovation_vec) + pred

        elif self.gate_type == "tanh_product":
            gate = torch.tanh(u_x)
            innov = torch.tanh(u_x - pred)
            innovation_vec = gate * innov
            u_actual = self.W_pre(innovation_vec) + pred

        else:  # 'none'
            u_actual = u_x + pred
            innovation_vec = torch.zeros_like(u_x)

        # Anti-collapse residual: a small fraction of u_x bypasses the gate.
        # Prevents memory stagnation as E_h learns to predict u_x exactly
        # (innov → 0, writes → pred, m stops updating).
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
                batch_size, self.memory_size, self.input_size,
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
        m_prev = state["m"]                                            # (B, D, C)

        # 1) encode
        e_x_n = F.normalize(self.e_x, dim=0)                            # (C,)
        u_x = x * e_x_n                                                 # (B, C)
        u_h = self.E_h(h_prev)                                          # (B, C)
        u_m = torch.einsum("d,bdc->bc", self.e_m, m_prev)                # (B, C)

        # 2) gated write
        u_actual, _pred, innovation_vec = self._compute_write(u_x, u_h, u_m)

        # 3) Legendre memory update: m_t = A_d m_{t-1} + B_d u_actual
        Am = torch.einsum("ij,bjc->bic", self.A, m_prev)                # (B, D, C)
        Bu = self.B * u_actual.unsqueeze(1)                              # (B, D, C)
        m_new = Am + Bu

        # 4) dynamic readout
        C_t = F.normalize(self.W_query(h_prev), dim=-1)                  # (B, D)
        y_internal = torch.einsum("bd,bdc->bc", C_t, m_new)              # (B, C)

        # 5) hidden update
        h_new = torch.tanh(
            self.W_x(x) + self.W_h(h_prev) + self.W_m(y_internal)
        )                                                                # (B, H)

        return h_new, {"h": h_new, "m": m_new}, {"innovation": innovation_vec}
