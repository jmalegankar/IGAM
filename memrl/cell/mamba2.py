"""Mamba-2 selective SSM block — Dao & Gu 2024 (ICML).

"Transformers are SSMs: Generalized Models and Efficient Algorithms Through
Structured State Space Duality"
https://arxiv.org/abs/2405.21060

The **full Mamba-2 block** as published — selective SSM core wrapped with the
block-level components (`expand=2` up-projection, depthwise causal Conv1d on
[v | B | C], z-gate via RMSNormGated, residual skip). The earlier version of
this file kept only the SSM core; this version adds back the block wrappers
that the docstring previously listed as "deliberate omissions," so the cell
matches the published architecture used in the Mamba-2 results.

Block headline ingredients vs S4D and Linear Transformer:

  1. **Selective B, C, Δ** — input-dependent projections, NOT fixed parameters.
     This is Mamba's headline contribution: the SSM is no longer LTI but
     time-varying through B_t, C_t, Δ_t. (Unchanged from before.)

  2. **Scalar A per head** — the Mamba-2 simplification over Mamba-1's
     diagonal A. Combined with selectivity, this is what gives Mamba-2
     its state-space duality with masked attention. (Unchanged from before.)

  3. **expand = 2** — the SSM operates at d_inner = expand × d_model (256 when
     d_model = 128). Up-projection from d_model → d_inner happens in
     in_proj, the SSM core runs at d_inner, and out_proj brings it back
     to d_model.

  4. **Depthwise causal Conv1d (K=4)** on [v | B | C] post-projection. Adds
     short-range mixing before the SSM. Implemented manually (window ⊙ K).sum
     to avoid the per-call dispatch overhead that nn.Conv1d incurs at L=K=4
     (verified to tank step throughput by ~30×).

  5. **z-gate via RMSNormGated** — a parallel SiLU(z) branch multiplies the
     RMS-normalized SSM output. Fuses gating with normalization, per the
     official Mamba-2 reference.

  6. **Block residual** — out = x_H + out_proj(SSM block). The transformer-
     style residual stream is provided inside the cell so the policy doesn't
     have to wrap it externally.

Math (per step, single block):

    Block input + residual carrier + pre-norm:
        x_H_t = W_in · x_t                                 # (B, d_model)
        u_t   = LN(x_H_t)                                  # pre-norm

    Fused projection into [z | v | B_sel | C_sel | dt_raw]:
        d_inner = expand · d_model;  d_head = d_inner / n_heads
        proj = W_in_proj · u_t                              # (B, 2·d_inner + 2·N + H)

    Depthwise causal Conv1d (K=4) on [v | B_sel | C_sel]:
        buf_t   = state.conv_buf                            # (B, conv_dim, K-1)
        win_t   = [buf_t || conv_in_t]                      # (B, conv_dim, K)
        c_t     = (win_t ⊙ K).sum(-1)                       # depthwise conv
        v_c, B_c, C_c = split(c_t, [d_inner, N, N])
        v_c     = SiLU(v_c)                                 # activation on v only

    Selective scalar SSM at d_inner (per head):
        dt_t = softplus(dt_raw + dt_bias)                   # (B, H), > 0
        A    = −exp(log_A)                                  # (H,), < 0
        A_d  = exp(dt_t · A)                                # (B, H)
        dtB  = dt_t · B_c                                   # (B, H, N) via broadcast
        h_t  = A_d · h_{t-1} + dtB ⊗ v_c                    # (B, H, N, d_head)
        y_t  = C_c^T · h_t + D · v_c                        # (B, H, d_head)

    RMSNormGated (norm_before_gate=False) + out_proj + residual:
        y_flat = flatten_heads(y_t)                         # (B, d_inner)
        y_z    = y_flat · SiLU(z)                           # gate FIRST
        y_rms  = y_z / √(mean(y_z²) + ε) · w_rms            # then RMSNorm
        out    = x_H_t + W_out · y_rms                      # block residual

Initialization (matches mamba2_simple.py reference):
  - **A**: A_init uniform in [1, 16], stored as log_A; forward A = −exp(log_A).
  - **Δ bias**: inverse softplus of Δ ~ log-uniform[1e-3, 1e-1]; gives Δ ≈ Δ_init
    at init when the dt projection is near zero (cell is LTI at init).
  - **D**: ones (Mamba-2 convention; LSTM forget-gate-style "remember by default").
  - **conv kernel**: small normal, std = 1/√K.
  - **in_proj / out_proj / input_linear**: Xavier uniform.
  - **rms gain**: ones (identity at init).

Default kwargs note:
  We use d_state=64 (Mamba-2 paper default). The previous core-only variant
  used d_state=128 to compensate for the missing block-level capacity; with
  expand=2 in place, d_state=64 × d_head_inner=64 gives the same effective
  matrix-memory capacity per head as the old d_state=128 × d_head=32.

Optimizer note (from the Mamba codebase):
  SSM parameters log_A, dt_bias, D are marked `_no_weight_decay` in the
  reference. Weight decay on them pushes the cell toward an unstable
  / degenerate dynamics regime. Manage at the train-time optimizer config;
  not enforced in the cell.

State:        {"h": (B, n_heads, d_state, d_head_inner),
               "conv_buf": (B, d_inner + 2*d_state, K-1)}
Side outputs: {}
Output:       (B, hidden_size)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from memrl.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask


class Mamba2(RecurrentCell):
    """Mamba-2 block (expand + conv + selective SSM + z-gate + residual)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_heads: int = 4,
        d_state: int = 64,
        expand: int = 2,
        d_conv: int = 4,
        d_head: Optional[int] = None,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        A_init_range: tuple[float, float] = (1.0, 16.0),
    ) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)

        d_inner = expand * hidden_size
        if d_head is None:
            if d_inner % n_heads != 0:
                raise ValueError(
                    f"d_inner ({d_inner} = expand×hidden_size) must be divisible "
                    f"by n_heads ({n_heads}) when d_head is not given"
                )
            d_head = d_inner // n_heads
        if d_inner != n_heads * d_head:
            raise ValueError(
                f"d_inner ({d_inner}) must equal n_heads ({n_heads}) × d_head "
                f"({d_head}) = {n_heads * d_head}"
            )
        if not 0.0 < dt_min < dt_max:
            raise ValueError(f"need 0 < dt_min < dt_max, got dt_min={dt_min}, dt_max={dt_max}")
        if not (A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]):
            raise ValueError(
                f"A_init_range must be (lo, hi) with 0 < lo <= hi, got {A_init_range}"
            )
        if d_conv < 1:
            raise ValueError(f"d_conv must be ≥ 1, got {d_conv}")

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_state = d_state
        self.d_head = d_head                       # d_head INNER (post-expand)
        self.d_inner = d_inner
        self.expand = expand
        self.d_conv = d_conv
        self.conv_dim = d_inner + 2 * d_state      # what the conv operates on
        H, N, P, E = n_heads, d_state, d_head, d_inner

        # Input projection (input_size → d_model) — residual carrier.
        self.input_linear = nn.Linear(input_size, hidden_size, bias=False)

        # Pre-LN before the block.
        self.pre_ln = nn.LayerNorm(hidden_size)

        # Fused projection: [z | v | B_sel | C_sel | dt_raw].
        # ngroups = 1 ⇒ B and C are shared across heads (Mamba-2 default).
        d_in_proj = 2 * E + 2 * N + H              # z(E) + v(E) + B(N) + C(N) + dt(H)
        self.in_proj = nn.Linear(hidden_size, d_in_proj, bias=False)

        # Depthwise causal Conv1d (K=d_conv) on [v | B | C], manual to avoid
        # nn.Conv1d's per-call overhead at L=K (cf. mlstm.py for the same fix).
        self.conv_weight = nn.Parameter(torch.empty(self.conv_dim, d_conv))

        # Δ bias: inverse-softplus of Δ ~ log-uniform[dt_min, dt_max] (Mamba init).
        dt_init = torch.exp(
            torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        self.dt_bias = nn.Parameter(inv_dt)

        # A: scalar per head, A < 0 enforced via A = −exp(log_A).
        A_init = torch.empty(H).uniform_(*A_init_range)
        self.log_A = nn.Parameter(torch.log(A_init))

        # D: per-head skip (init ones).
        self.D = nn.Parameter(torch.ones(H))

        # RMSNorm gain on d_inner — the "RMSNormGated" layer (norm then gate).
        self.rms_weight = nn.Parameter(torch.ones(E))
        self.rms_eps = 1e-5

        # Output projection: d_inner → d_model.
        self.out_proj = nn.Linear(d_inner, hidden_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Xavier on the data-path projections. SSM parameters (dt_bias, log_A, D)
        # and rms_weight keep their construction-time inits — those ARE the
        # Mamba-2 defaults.
        nn.init.xavier_uniform_(self.input_linear.weight)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.conv_weight, std=1.0 / math.sqrt(self.d_conv))

    # --- interface ---------------------------------------------------------

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        device = self._resolve_device(device)
        return {
            "h": torch.zeros(
                batch_size, self.n_heads, self.d_state, self.d_head,
                device=device, dtype=dtype,
            ),
            # Rolling K-1 buffer for the depthwise conv on [v | B | C].
            "conv_buf": torch.zeros(
                batch_size, self.conv_dim, self.d_conv - 1,
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

        B = x.shape[0]
        H, N, P, E = self.n_heads, self.d_state, self.d_head, self.d_inner

        # ── Block input + residual carrier + pre-norm ──────────────────────
        x_H = self.input_linear(x)                                       # (B, d_model)
        u = self.pre_ln(x_H)                                             # (B, d_model)

        # ── Fused projection: [z | v | B_sel | C_sel | dt_raw] ─────────────
        proj = self.in_proj(u)                                           # (B, 2E + 2N + H)
        off = 0
        z      = proj[..., off : off + E]; off += E                       # gate arm
        v_pre  = proj[..., off : off + E]; off += E                       # SSM input
        B_pre  = proj[..., off : off + N]; off += N
        C_pre  = proj[..., off : off + N]; off += N
        dt_raw = proj[..., off : off + H]                                  # (B, H)

        # ── Depthwise causal Conv1d (K=d_conv) on [v | B | C] (manual) ─────
        conv_in = torch.cat([v_pre, B_pre, C_pre], dim=-1)               # (B, conv_dim)
        buf_prev = state["conv_buf"]                                      # (B, conv_dim, K-1)
        win = torch.cat([buf_prev, conv_in.unsqueeze(-1)], dim=-1)       # (B, conv_dim, K)
        c = torch.einsum("bck,ck->bc", win, self.conv_weight)            # (B, conv_dim)
        conv_buf_new = win[..., 1:]                                       # (B, conv_dim, K-1)
        # SiLU on the ENTIRE post-conv signal (x, B, C) — matches the
        # official mamba2_simple.py: `xBC = act(conv1d(xBC))` where
        # act = nn.SiLU(). Earlier draft silu'd only v; that was a bug.
        c = F.silu(c)
        v_c = c[..., : E]                                                 # (B, d_inner)
        B_c = c[..., E : E + N]                                           # (B, N)
        C_c = c[..., E + N : E + 2 * N]                                   # (B, N)

        # ── Selective scalar SSM at d_inner ────────────────────────────────
        dt = F.softplus(dt_raw + self.dt_bias)                            # (B, H)
        A = -torch.exp(self.log_A)                                        # (H,)
        A_d = torch.exp(dt * A.unsqueeze(0))                              # (B, H)
        dtB = dt.unsqueeze(-1) * B_c.unsqueeze(1)                         # (B, H, N) via broadcast

        v = v_c.view(B, H, P)                                             # (B, H, d_head_inner)
        h_prev = state["h"]                                               # (B, H, N, P)
        outer_dtB_v = dtB.unsqueeze(-1) * v.unsqueeze(-2)                 # (B, H, N, P)
        h_new = A_d.unsqueeze(-1).unsqueeze(-1) * h_prev + outer_dtB_v    # (B, H, N, P)

        # Readout: y = C^T h. C shared across heads ⇒ einsum 'bn,bhnp->bhp'.
        y = torch.einsum("bn,bhnp->bhp", C_c, h_new)                     # (B, H, P)
        y = y + self.D.view(1, H, 1) * v                                  # per-head skip
        y_flat = y.reshape(B, E)                                          # (B, d_inner)

        # ── RMSNormGated (norm_before_gate=False): gate FIRST, then RMS-norm ─
        # The official Mamba-2 builds RMSNormGated(..., norm_before_gate=False),
        # whose semantics are y = rmsnorm(y · SiLU(z)) · w — the gate enters the
        # RMS statistic. Normalizing first then gating (norm_before_gate=True) is
        # a different, non-default variant.
        y_pre = y_flat * F.silu(z)                                        # (B, d_inner)
        rms = torch.rsqrt(y_pre.pow(2).mean(dim=-1, keepdim=True) + self.rms_eps)
        y_gated = y_pre * rms * self.rms_weight                           # (B, d_inner)

        # ── Out-proj + block residual ──────────────────────────────────────
        y_out = self.out_proj(y_gated)                                    # (B, d_model)
        out = x_H + y_out                                                  # residual

        return out, {"h": h_new, "conv_buf": conv_buf_new}, {}
