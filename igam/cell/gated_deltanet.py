"""Gated DeltaNet — Yang et al. 2024 (ICLR 2025). The IGAM headline cell.

"Gated Delta Networks: Improving Mamba2 with Delta Rule"
https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf

The IGAM cell. Sits at the upper-right corner of the Phase A 2×2 ablation
matrix: **matrix memory + delta rule + data-dependent gating**. Extends
the already-built `DeltaNet` cell by adding a per-head decay gate `α_t`
that scales the previous memory down BEFORE each delta-rule update.

Where the cell fits in the ablation gradient (rows = "memory family"):

    LinearTransformer   no decay, no delta rule         ← matrix memory baseline
    RetNet              fixed multi-scale decay         ← + fixed γ
    Mamba2              selective scalar SSM, no delta  ← + selectivity (scalar A per head)
    DeltaNet            delta rule, no decay            ← + delta rule
    Gated DeltaNet      delta rule + selective decay    ← IGAM (this cell)

Math (per step, per head h; k̃ = L2-normalized k, q̃ = L2-normalized q):

    Projections (all from φ(o_t) — per ADR 0004, NO separate hidden state):
        q_t, k_t, v_t  ←  linear(φ(o_t))                # (B, H, d_head)
        α_raw_t, β_raw_t  ←  linear(φ(o_t))             # (B, H) each

    Gates (per-head scalars):
        α_t  =  σ(α_raw_t)        # decay
        β_t  =  σ(β_raw_t)        # write strength

    Innovation (the side output for Phase B lifelong reward):
        v_pred_t  =  W_{t-1}  ·  k̃_t                    # current value at key k̃
        δ_t       =  v_t  −  v_pred_t                   # raw innovation, NOT α-scaled

    State updates (algebraic simplification of the README spec):
        W_t  =  α_t · W_{t-1} · (I − β_t k̃_t k̃_t^T)  +  β_t · v_t k̃_t^T
             =  α_t · W_{t-1}  +  β_t · (v_t − α_t · v_pred_t) · k̃_t^T

        n_t  =  α_t · n_{t-1}  +  β_t · k̃_t            # normalizer

    Readout:
        y_t  =  q̃_t^T · W_t  /  max(|q̃_t^T · n_t|,  ε)

Why the "effective innovation" for the update differs from the side output:
    The pure innovation δ_t = v − W_{t-1} k̃ is the natural prediction-error
    signal — what the Phase B lifelong reward consumes. But the update math
    algebraically simplifies in terms of (v − α · v_pred), not (v − v_pred).
    At α=1 these coincide and the cell reduces exactly to DeltaNet — so the
    ablation "IGAM with α fixed at 1 ≡ DeltaNet" is mathematically clean.

Specializations / ablations (constructor flags below):
    `α_fixed_one=True`:   forces α_t ≡ 1, recovers DeltaNet exactly.
                          Used for the "IGAM with α fixed" Phase A ablation row.
    `β_fixed_one=True`:   forces β_t ≡ 1, full-strength writes every step.
                          Used for the "IGAM with β fixed" Phase A ablation row.

Inheritance from DeltaNet (kept):
    - L2-normalize both q and k (Yang 2024 — "greatly increased performance" over Schlag's L1)
    - Pre-LN on input + Output-LN + Output projection
    - bias=False everywhere; learnable shifts in LN.β
    - Xavier init on projections
    - Side output: `{"innovation": δ_t}` for Phase B compatibility

What's added over DeltaNet:
    - α_t = σ(W_α φ(o_t)) — per-head scalar decay, applied to BOTH W and n.
    - n_t normalizer state (DeltaNet doesn't have one).
    - q̃_t^T n_t denominator clamp at ε in the readout (matches LinearTransformer
      and mLSTM's div-by-normalizer pattern).

State:        {"W": (B, n_heads, d_head, d_head), "n": (B, n_heads, d_head)}
Side outputs: {"innovation": (B, n_heads, d_head)}
Output:       (B, hidden_size)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from igam.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask


class GatedDeltaNet(RecurrentCell):
    """Gated DeltaNet (Yang et al. 2024) — IGAM's headline cell."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_heads: int = 4,
        d_head: Optional[int] = None,
        eps: float = 1e-6,
        alpha_fixed_one: bool = False,
        beta_fixed_one: bool = False,
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

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = d_head
        self.eps = eps
        self.alpha_fixed_one = alpha_fixed_one
        self.beta_fixed_one = beta_fixed_one
        H, D = n_heads, d_head

        # Pre-LN on the input — modern transformer-style stability.
        self.input_ln = nn.LayerNorm(input_size)

        # Fused projection: [q | k | v | α_raw | β_raw]. bias=False; gate
        # shifts and projection shifts come from the input LN's affine.
        d_in_proj = 3 * H * D + 2 * H
        self.qkvab_linear = nn.Linear(input_size, d_in_proj, bias=False)

        # Output mixing — consistent with the rest of the cell family.
        self.out_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_ln = nn.LayerNorm(hidden_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.qkvab_linear.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)

    # --- interface ---------------------------------------------------------

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        device = self._resolve_device(device)
        H, D = self.n_heads, self.d_head
        return {
            "W": torch.zeros(batch_size, H, D, D, device=device, dtype=dtype),
            "n": torch.zeros(batch_size, H, D, device=device, dtype=dtype),
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
        H, D = self.n_heads, self.d_head

        # Pre-LN, then fused projection. All gates and QKV come from φ(o_t)
        # directly — per ADR 0004, no separate hidden state.
        x_normed = self.input_ln(x)                                     # (B, input_size)
        proj = self.qkvab_linear(x_normed)                              # (B, 3*H*D + 2*H)

        off = 0
        q_flat = proj[..., off : off + H * D]; off += H * D
        k_flat = proj[..., off : off + H * D]; off += H * D
        v_flat = proj[..., off : off + H * D]; off += H * D
        alpha_raw = proj[..., off : off + H]; off += H
        beta_raw = proj[..., off : off + H]

        q = q_flat.view(B, H, D)
        k = k_flat.view(B, H, D)
        v = v_flat.view(B, H, D)

        # L2-normalize both q and k. With ||k̃|| = 1 and β ∈ (0, 1), the
        # eigenvalue of (I − β k̃ k̃^T) along k̃ is 1−β ∈ (0, 1) and 1 elsewhere
        # — the contractive stability property IGAM inherits from DeltaNet.
        q_norm = F.normalize(q, p=2, dim=-1, eps=self.eps)              # (B, H, D)
        k_norm = F.normalize(k, p=2, dim=-1, eps=self.eps)

        # Per-head scalar gates. Constructor flags force them to identity
        # for the "IGAM with α fixed" / "IGAM with β fixed" ablation rows.
        if self.alpha_fixed_one:
            alpha = torch.ones(B, H, device=x.device, dtype=x.dtype)
        else:
            alpha = torch.sigmoid(alpha_raw)                            # (B, H)

        if self.beta_fixed_one:
            beta = torch.ones(B, H, device=x.device, dtype=x.dtype)
        else:
            beta = torch.sigmoid(beta_raw)                              # (B, H)

        W_prev = state["W"]                                              # (B, H, D, D)
        n_prev = state["n"]                                              # (B, H, D)

        # Current value prediction: v_pred = W_prev k̃.
        v_pred = torch.einsum("bhij,bhj->bhi", W_prev, k_norm)           # (B, H, D)

        # Pure innovation δ_t = v − W_{t-1} k̃ — the side output, consumed by
        # Phase B's lifelong intrinsic reward (r_life = ||δ||²). Computed
        # WITHOUT α scaling, per the README's definition.
        delta = v - v_pred                                               # (B, H, D)

        # "Effective innovation" for the update math: (v − α · v_pred).
        # When α = 1 this coincides with δ ⇒ exact DeltaNet recovery.
        alpha_b3 = alpha.unsqueeze(-1)                                   # (B, H, 1)
        delta_update = v - alpha_b3 * v_pred                             # (B, H, D)

        # Gated delta-rule update:
        #   W_t = α · W_{t-1} + β · (v − α · v_pred) ⊗ k̃
        beta_b3 = beta.unsqueeze(-1)                                     # (B, H, 1)
        alpha_b4 = alpha.unsqueeze(-1).unsqueeze(-1)                     # (B, H, 1, 1)
        beta_b4 = beta.unsqueeze(-1).unsqueeze(-1)                       # (B, H, 1, 1)
        delta_outer_k = torch.einsum(
            "bhi,bhj->bhij", delta_update, k_norm
        )                                                                # (B, H, D, D)
        W_new = alpha_b4 * W_prev + beta_b4 * delta_outer_k              # (B, H, D, D)

        # Normalizer update — same gates applied to a vector accumulation:
        #   n_t = α · n_{t-1} + β · k̃
        n_new = alpha_b3 * n_prev + beta_b3 * k_norm                     # (B, H, D)

        # Readout: y = q̃^T W / max(|q̃^T n|, ε).
        Wq = torch.einsum("bhij,bhj->bhi", W_new, q_norm)                # (B, H, D)
        nq = torch.einsum("bhi,bhi->bh", q_norm, n_new)                  # (B, H)
        denom = nq.abs().clamp(min=self.eps).unsqueeze(-1)               # (B, H, 1)
        y_per_head = Wq / denom                                          # (B, H, D)

        # Flatten heads, project, LN.
        y_flat = y_per_head.reshape(B, self.hidden_size)
        y_out = self.output_ln(self.out_linear(y_flat))                  # (B, hidden_size)

        return y_out, {"W": W_new, "n": n_new}, {"innovation": delta}
