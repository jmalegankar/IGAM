"""DeltaNet — Schlag, Irie, Schmidhuber 2021 (ICML).

"Linear Transformers Are Secretly Fast Weight Programmers"
https://arxiv.org/abs/2102.11174

Refined parameterization from Yang et al. 2024 (NeurIPS):
"Parallelizing Linear Transformers with the Delta Rule over Sequence Length"
https://arxiv.org/abs/2406.06484

The direct precursor to GatedDeltaNet (Gated DeltaNet). DeltaNet replaces the
Linear Transformer's pure outer-product accumulation with a delta-rule
update that performs error-corrected key→value association. The Gated
DeltaNet paper (Yang et al. 2024) — which GatedDeltaNet extends — defines vanilla
DeltaNet as the α_t ≡ 1 special case: no memory decay before each update.

Why DeltaNet for the GatedDeltaNet ablation table:
  Strips the α_t gate from GatedDeltaNet, leaves the delta rule + dynamic write
  strength β_t intact. The direct ablation question:
      "Does memory-decay gating (α_t) matter, given the delta rule already
       overwrites stale key→value associations through prediction error?"
  GatedDeltaNet's central claim is that α_t adds something orthogonal to the delta
  rule. DeltaNet is the ablation that tests it.

Math (per step, per head):
    Projections:    q_t, k_t, v_t, β_raw_t  ←  linear(x_t)         all (B, H, d_head)
                    β_raw_t  is (B, H)
    Normalize:      q̃_t = l2norm(q_t)
                    k̃_t = l2norm(k_t)
    Write strength: β_t  =  σ(β_raw_t)                              ∈ (0, 1)
    Prediction:     v̄_t  =  W_{t-1} · k̃_t                          # current value at k̃_t
    Innovation:     δ_t  =  v_t − v̄_t                                # error signal
    Update:         W_t  =  W_{t-1}  +  β_t · (δ_t  ⊗  k̃_t)         # delta rule
                       =  W_{t-1} · (I − β_t k̃_t k̃_t^T)  +  β_t · v_t k̃_t^T
    Read:           y_t  =  W_t · q̃_t                                # (B, H, d_head)

Stability:
    With k̃_t L2-normalized (||k̃_t|| = 1) and β_t ∈ (0, 1):
        eigenvalue along k̃_t direction of (I − β_t k̃_t k̃_t^T)  =  1 − β_t  ∈ (0, 1)
        eigenvalues perpendicular to k̃_t                          =  1
    So the recurrence is contractive along the current key direction
    and identity elsewhere — bounded gradient flow under PPO's high-
    variance advantage signal. This is the structural guarantee that
    Yang et al. cite in Gated DeltaNet §2 and that GatedDeltaNet inherits.

Why both q AND k are L2-normalized:
    Schlag 2021 §3.3 normalizes k only ("the magnitude of k controls how
    much memory is overwritten"). Yang et al. 2024 normalize both,
    reporting "greatly increased performance" in their §5 ablations.
    The reason is symmetric: at read time, the magnitude of q controls
    how much memory is queried; in practice this conflates with the
    output projection and helps to constrain it. Modern convention
    (flash-linear-attention, the official Gated DeltaNet repo) is to
    L2-norm both. We match.

No feature map (identity):
    Schlag 2021 discusses kernel feature maps (DPFP, elu+1) as
    alternatives. Vanilla DeltaNet — the form Yang 2024 and the Gated
    DeltaNet paper compare against — uses identity. We match.

Side outputs:
    {"innovation": δ_t ∈ ℝ^{B × H × d_head}}
    The Phase B lifelong intrinsic reward r_life = ||δ_t||² is the same
    quantity for DeltaNet as for GatedDeltaNet. Exposing it from DeltaNet enables
    a clean ablation row: "DeltaNet + lifelong-from-innovation reward"
    vs "GatedDeltaNet + lifelong-from-innovation reward" tests whether GatedDeltaNet's
    architectural gains transfer to the exploration mechanism.

State:        {"W": (B, n_heads, d_head, d_head)}
Side outputs: {"innovation": (B, n_heads, d_head)}
Output:       (B, hidden_size)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from memrl.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask


class DeltaNet(RecurrentCell):
    """Schlag 2021 DeltaNet in single-step recurrent form."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_heads: int = 4,
        d_head: Optional[int] = None,
        eps: float = 1e-6,
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
        H, D = n_heads, d_head

        # Pre-LN on the input — modern transformer-style stability practice.
        self.input_ln = nn.LayerNorm(input_size)

        # Fused projection: [q | k | v | β_raw]. bias=False; the learnable
        # shifts live in LN affines downstream.
        d_in_proj = 3 * H * D + H
        self.qkvbeta_linear = nn.Linear(input_size, d_in_proj, bias=False)

        # Output mixing — linear + LN, consistent with LinearTransformer/S4D/Mamba2.
        self.out_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_ln = nn.LayerNorm(hidden_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Xavier on all projections. No explicit 1/√d_head scaling (no
        # softmax to temper); the L2-normalization on q/k caps the
        # readout magnitude implicitly.
        nn.init.xavier_uniform_(self.qkvbeta_linear.weight)
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

        # Pre-LN, then fused projection to [q, k, v, β_raw].
        x_normed = self.input_ln(x)                                     # (B, input_size)
        proj = self.qkvbeta_linear(x_normed)                            # (B, 3*H*D + H)
        q_flat = proj[..., : H * D]                                     # (B, H*D)
        k_flat = proj[..., H * D : 2 * H * D]                            # (B, H*D)
        v_flat = proj[..., 2 * H * D : 3 * H * D]                        # (B, H*D)
        beta_raw = proj[..., -H:]                                        # (B, H)

        q = q_flat.view(B, H, D)                                         # (B, H, D)
        k = k_flat.view(B, H, D)
        v = v_flat.view(B, H, D)

        # L2-normalize q and k. Yang et al. 2024 — replaces Schlag's L1 norm.
        # Contractive guarantee: with ||k̃||=1 and β ∈ (0,1), the eigenvalue
        # of (I − β k̃ k̃^T) along k̃ is 1−β ∈ (0,1); perpendicular eigenvalues
        # are 1. So the write is contractive in the current key direction
        # and non-disruptive elsewhere.
        q_norm = F.normalize(q, p=2, dim=-1, eps=self.eps)               # (B, H, D)
        k_norm = F.normalize(k, p=2, dim=-1, eps=self.eps)

        # Per-head scalar write strength.
        beta = torch.sigmoid(beta_raw)                                   # (B, H)

        # Delta-rule update.
        W_prev = state["W"]                                              # (B, H, D, D)

        # Current value prediction: v̄ = W_{t-1} k̃.
        # einsum: bhij,bhj->bhi  (matrix-vector per head).
        v_pred = torch.einsum("bhij,bhj->bhi", W_prev, k_norm)           # (B, H, D)

        # Innovation: δ = v − v̄. This is the lifelong-intrinsic-reward signal
        # for Phase B exploration (r_life = ||δ||²), matching GatedDeltaNet's contract.
        delta = v - v_pred                                               # (B, H, D)

        # State update: W_new = W_prev + β · (δ ⊗ k̃).
        # einsum 'bhi,bhj->bhij' gives (B, H, D, D) outer product.
        delta_outer_k = torch.einsum("bhi,bhj->bhij", delta, k_norm)     # (B, H, D, D)
        beta_b = beta.unsqueeze(-1).unsqueeze(-1)                        # (B, H, 1, 1)
        W_new = W_prev + beta_b * delta_outer_k                          # (B, H, D, D)

        # Readout: y = W_new q̃.
        y_per_head = torch.einsum("bhij,bhj->bhi", W_new, q_norm)        # (B, H, D)

        # Flatten heads, project, LN.
        y_flat = y_per_head.reshape(B, self.hidden_size)
        y_out = self.output_ln(self.out_linear(y_flat))                  # (B, hidden_size)

        return y_out, {"W": W_new}, {"innovation": delta}
