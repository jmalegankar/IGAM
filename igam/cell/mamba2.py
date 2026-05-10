"""Mamba-2 selective SSM core — Dao & Gu 2024 (ICML).

"Transformers are SSMs: Generalized Models and Efficient Algorithms Through
Structured State Space Duality"
https://arxiv.org/abs/2405.21060

The selective SSM core from Mamba-2. Three things distinguish it from S4D
and from Linear Transformer:

  1. **Selective B, C, Δ** — input-dependent projections, NOT fixed
     parameters. This is Mamba's headline contribution: the SSM is no
     longer LTI but time-varying through B_t, C_t, Δ_t.

  2. **Scalar A per head** — the Mamba-2 simplification over Mamba-1's
     diagonal A. Combined with selectivity, this is what gives Mamba-2
     its **state-space duality** with masked attention (a scalar-times-
     identity state matrix corresponds to a 1-semiseparable causal mask).

  3. **Multi-head matrix-memory state** — h_t ∈ ℝ^{H × N × P} where
     H = n_heads, N = d_state, P = d_head. Same shape family as
     LinearTransformer / Gated DeltaNet, but with selective dynamics.

Why Mamba-2 specifically for the IGAM ablation table:
  - "Selectivity" is conceptually identical to IGAM's data-dependent
    gates (α_t, β_t, dynamic query): both are input-dependent
    modulations of an otherwise-LTI recurrence.
  - Mamba-2 uses scalar A per head + selectivity to compete with
    attention. IGAM uses matrix-memory state + delta rule + selectivity
    to compete with the same. Direct ablation: "what does the delta
    rule add to a selective matrix-memory cell?"
  - Per the README Phase A baseline list (Week 4–6 ablations).

Math (per step, single head h):
    Continuous SSM:    ẋ_t = A x_t + B_t v_t
                       y_t = C_t^T x_t
    Selectivity:       Δ_t, B_t, C_t  ←  linear projections of x_t
    Discretization (Mamba convention — Δ-scaled Euler for B, exact for A):
                       A_d  =  exp(Δ_t · A)                    # scalar per head
                       B_d  =  Δ_t · B_t                       # selective B
    State update:      h_t  =  A_d · h_{t-1}  +  (B_d ⊗ v_t)   # matrix outer product
    Readout:           y_t  =  C_t^T · h_t  +  D · v_t         # plus per-head skip

Note: Mamba uses the simpler `Δ · B` for the B-discretization (a Euler-
style approximation), not the full ZOH `(exp(Δ·A) − 1)/A · B`. The
official implementation uses this convention; we match it.

Architectural simplifications vs the full Mamba-2 block:
  - **ngroups = 1**: B and C are SHARED across heads. This is the default
    in the official Mamba-2 reference (mamba2_simple.py). It halves the
    parameter count for selective projections relative to per-head B,C.

Deliberate omissions from the full Mamba-2 *block*:
  - **1D depthwise causal conv** (d_conv=4) — adds short-range mixing
    before the SSM. Belongs to the block, not the SSM core. Would also
    require an extra state component (rolling buffer of past K-1 inputs).
  - **Expansion factor** (expand=2) — the block expands d_model → 2*d_model
    before the SSM. Block-level, not SSM-core.
  - **z-gate** (SiLU(z) · y mixing) — the parallel-branch gate that
    multiplies the SSM output. Block-level; analogous to the GLU in S4D
    blocks. Other cells in our family (LinearTransformer, S4D) don't
    include it either, for consistent ablation.
  - **RMSNormGated** — fused gate+norm op from the official kernel.
    Replaced with plain LayerNorm here (no gate to fuse with).

Initialization (matches mamba2_simple.py reference):
  - **A**: A_init uniform in [1, 16], stored as log_A. At forward time
    A = −exp(log_A) ∈ [−16, −1]. Ensures Re(A) < 0 ⇒ |A_d| < 1.
  - **Δ bias**: inverse softplus of Δ_init, where Δ_init is log-uniform
    in [dt_min, dt_max] = [1e-3, 1e-1]. Gives Δ ≈ Δ_init at init when
    the selective dt projection is near zero.
  - **D**: ones (Mamba-2 convention, not random Gaussian like S4D).
  - **in_proj / out_proj**: Xavier uniform.

Optimizer note (from the Mamba codebase):
  SSM parameters log_A, dt_bias, D are marked `_no_weight_decay` in the
  reference. Weight decay on them pushes the cell toward an unstable
  / degenerate dynamics regime. Manage at the train-time optimizer
  config when we get to Phase A benchmarks; not enforced in the cell.

State:        {"h": (B, n_heads, d_state, d_head)}
Side outputs: {}
Output:       (B, hidden_size)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from igam.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask


class Mamba2(RecurrentCell):
    """Mamba-2 selective SSM core in single-step recurrent form."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_heads: int = 4,
        d_state: int = 64,
        d_head: Optional[int] = None,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        A_init_range: tuple[float, float] = (1.0, 16.0),
    ) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)

        if d_head is None:
            if hidden_size % n_heads != 0:
                raise ValueError(
                    f"hidden_size ({hidden_size}) must be divisible by n_heads "
                    f"({n_heads}) when d_head is not given"
                )
            d_head = hidden_size // n_heads
        if hidden_size != n_heads * d_head:
            raise ValueError(
                f"hidden_size ({hidden_size}) must equal n_heads ({n_heads}) "
                f"× d_head ({d_head}) = {n_heads * d_head}"
            )
        if not 0.0 < dt_min < dt_max:
            raise ValueError(f"need 0 < dt_min < dt_max, got dt_min={dt_min}, dt_max={dt_max}")
        if not (A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]):
            raise ValueError(
                f"A_init_range must be (lo, hi) with 0 < lo <= hi, got {A_init_range}"
            )

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_state = d_state
        self.d_head = d_head
        H, N, P = n_heads, d_state, d_head

        # Fused input projection: [v | B_sel | C_sel | dt_raw].
        # ngroups = 1 ⇒ B and C are shared across heads (Mamba-2 default).
        d_in_proj = H * P + 2 * N + H
        self.in_proj = nn.Linear(input_size, d_in_proj, bias=False)

        # Δ bias: inverse-softplus of Δ sampled log-uniform in [dt_min, dt_max].
        # softplus(0 + dt_bias) ≈ Δ_init, so the cell behaves like an LTI SSM
        # at init (dt_raw projection ≈ 0) and learns to modulate selectively.
        dt_init = torch.exp(
            torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        # Numerical-stable inverse softplus: inv_softplus(x) = x + log(-expm1(-x))
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        self.dt_bias = nn.Parameter(inv_dt)

        # A: scalar per head, A < 0 enforced via A = −exp(log_A).
        # Init uniform in A_init_range ⇒ A_used ∈ [−hi, −lo].
        A_init = torch.empty(H).uniform_(*A_init_range)
        self.log_A = nn.Parameter(torch.log(A_init))

        # D: per-head skip. Mamba-2 inits to ones (LSTM forget-gate-style
        # "remember by default"); S4D inits to randn. We match Mamba-2.
        self.D = nn.Parameter(torch.ones(H))

        # Output mixing: linear + LN, consistent with the rest of the cell
        # family. (Mamba-2 block has RMSNormGated + out_proj; we drop the
        # gate per the design notes above and replace RMSNorm with LayerNorm.)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_ln = nn.LayerNorm(hidden_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Xavier on the data-path projections. SSM parameters (dt_bias,
        # log_A, D) keep their construction-time inits — those ARE the
        # Mamba-2 defaults and re-initializing would clobber them.
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

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
        H, N, P = self.n_heads, self.d_state, self.d_head

        # Project: [v | B_sel | C_sel | dt_raw]. ngroups=1 ⇒ B_sel and C_sel
        # are shape (B, N), shared across heads.
        proj = self.in_proj(x)                                          # (B, H*P + 2*N + H)
        v_flat = proj[..., : H * P]                                     # (B, H*P)
        B_sel = proj[..., H * P : H * P + N]                             # (B, N)
        C_sel = proj[..., H * P + N : H * P + 2 * N]                     # (B, N)
        dt_raw = proj[..., -H:]                                          # (B, H)
        v = v_flat.view(B, H, P)                                         # (B, H, P)

        # Selective step size: softplus(raw + bias) > 0 always.
        dt = F.softplus(dt_raw + self.dt_bias)                           # (B, H)

        # Scalar A per head, A < 0.
        A = -torch.exp(self.log_A)                                       # (H,)

        # Discretization. A is scalar per head ⇒ A_d is scalar per head per batch.
        # B uses Δ-scaled Euler (Mamba convention, not full ZOH).
        A_d = torch.exp(dt * A.unsqueeze(0))                             # (B, H)
        dtB = dt.unsqueeze(-1) * B_sel.unsqueeze(1)                       # (B, H, N) via broadcast

        # State update: h_t = A_d · h_{t-1} + (dtB ⊗ v).
        # Outer product expands (B, H, N) and (B, H, P) into (B, H, N, P).
        h_prev = state["h"]                                              # (B, H, N, P)
        outer_dtB_v = dtB.unsqueeze(-1) * v.unsqueeze(-2)                # (B, H, N, P)
        h_new = A_d.unsqueeze(-1).unsqueeze(-1) * h_prev + outer_dtB_v   # (B, H, N, P)

        # Readout: y = C^T h. C shared across heads ⇒ einsum 'bn,bhnp->bhp'.
        y = torch.einsum("bn,bhnp->bhp", C_sel, h_new)                   # (B, H, P)

        # Per-head skip connection.
        y = y + self.D.view(1, H, 1) * v                                 # (B, H, P)

        # Flatten heads, project, LN.
        y_flat = y.reshape(B, self.hidden_size)
        y_out = self.output_ln(self.out_proj(y_flat))                    # (B, hidden_size)

        return y_out, {"h": h_new}, {}
