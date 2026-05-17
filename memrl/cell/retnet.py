"""RetNet (Multi-Scale Retention) — Sun et al. 2023.

"Retentive Network: A Successor to Transformer for Large Language Models"
https://arxiv.org/abs/2307.08621

The recurrent form of RetNet's Multi-Scale Retention (MSR). Architecturally
this is **Linear Transformer + per-head fixed exponential decay**, where
each head h has a different decay rate γ_h drawn from a multi-scale
schedule. Together with LinearTransformer (no decay) and Mamba-2 (selective
decay), RetNet completes the three-way ablation:

    no decay  →  fixed multi-scale decay  →  selective per-step decay
    LinTrans  →  RetNet                   →  Mamba-2

For GatedDeltaNet specifically, RetNet isolates the question: *does the data-
dependent decay α_t in GatedDeltaNet contribute beyond what a well-chosen fixed
schedule provides?*

Math (per step, per head h):
    Projections:    q_t, k_t, v_t  ←  linear(x_t)              all (B, H, d_head)
    State update:   S^h_t  =  γ_h · S^h_{t-1}  +  k^h_t ⊗ v^h_t   # (d_head, d_head)
    Readout:        y^h_t  =  q^h_t · S^h_t                       # (d_head,)

Multi-scale decay schedule (Sun et al. 2023 Eq. 13):
    γ_h  =  1 − 2^{−5−h}   for h = 0, 1, ..., H − 1

For H = 8 this gives effective decay windows ranging from ≈32 to ≈4096
steps. Lower-index heads forget fast (capture local structure); higher-
index heads forget slowly (capture long-range dependencies). The
exponential spacing in γ_h is the "multi-scale" piece.

  Head 0:   γ = 1 − 2^{−5}  = 0.96875   window ≈ 32
  Head 1:   γ = 1 − 2^{−6}  = 0.98438   window ≈ 64
  Head 2:   γ = 1 − 2^{−7}  = 0.99219   window ≈ 128
  ...
  Head H−1: γ → 1                       window → very long

γ is registered as a buffer (fixed, NOT learned). Matches Sun 2023 §2.2
which treats γ_h as a hyperparameter. A learnable γ_h variant would push
toward "selective decay" (Mamba-2 territory); keep them distinct.

Architectural simplifications vs the full RetNet block:
  - **GroupNorm → LayerNorm**: the official RetNet block applies per-head
    GroupNorm before W_O. We use LayerNorm on the concatenated heads for
    consistency with the rest of the cell family (LinearTransformer,
    DeltaNet, Mamba-2 all use LayerNorm). The empirical difference is
    small for our scales; reviewers can override by subclassing.
  - **No swish-gated branch**: RetNet's full block has
    `out = (swish(XW_G) ⊙ retention(X)) W_O`. The swish gate belongs to
    the *block*, not the retention core itself — we drop it for the same
    reason we dropped Mamba-2's z-gate and S4D's GLU.
  - **No RoPE / xPos positional encoding**: RetNet uses xPos in Q, K to
    encode positions for the parallel/chunkwise forms. In single-step
    recurrent (our setting per ADR 0001), position is implicit in the
    recurrence step count, and episode-boundary resets handle the
    "where am I in the episode" question. RoPE is sequence-modeling
    machinery without an analog in recurrent RL.

Why no normalizer / feature map:
  Unlike LinearTransformer's `φ(q) S / φ(q) z` form, RetNet has no
  denominator normalizer. The exponential decay γ_h provides natural
  magnitude bounding: S_t = Σ_{i≤t} γ^{t-i} (k_i ⊗ v_i), and the
  geometric sum Σ γ^k = 1/(1−γ) is finite. The output projection +
  LayerNorm handle any residual drift.

Deliberate omissions:
  - No dropout, no parallel-scan training (single-step only per ADR 0001).
  - No chunkwise recurrent form (a `forward_sequence` override could add
    it later; the base loop suffices for Phase A).

State:        {"S": (B, n_heads, d_head, d_head)}
Side outputs: {}
Output:       (B, hidden_size)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from memrl.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask


class RetNet(RecurrentCell):
    """Multi-Scale Retention in single-step recurrent form."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_heads: int = 4,
        d_head: Optional[int] = None,
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
        H, D = n_heads, d_head

        # Multi-scale decay rates γ_h = 1 − 2^{−5−h}, h = 0..H-1. FIXED.
        # See Sun et al. 2023 Eq. 13. Stored as buffer — moves with .to(device)
        # but does not receive gradients.
        gamma = 1.0 - torch.pow(2.0, -5.0 - torch.arange(H, dtype=torch.float32))
        self.register_buffer("gamma", gamma)

        # Pre-LN on the input, then fused QKV projection.
        self.input_ln = nn.LayerNorm(input_size)
        self.qkv_linear = nn.Linear(input_size, 3 * hidden_size, bias=False)

        # Output mixing — Linear + LN, consistent with the rest of the cell
        # family (see module docstring for the GroupNorm-vs-LayerNorm choice).
        self.out_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_ln = nn.LayerNorm(hidden_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Xavier on the data-path projections. γ stays at its construction-
        # time multi-scale schedule.
        nn.init.xavier_uniform_(self.qkv_linear.weight)
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
            "S": torch.zeros(batch_size, H, D, D, device=device, dtype=dtype),
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

        # Pre-LN, then fused QKV projection.
        x_normed = self.input_ln(x)                                     # (B, input_size)
        qkv = self.qkv_linear(x_normed)                                 # (B, 3*hidden_size)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, H, D)
        k = k.view(B, H, D)
        v = v.view(B, H, D)

        # Recurrent retention update: S_t = γ_h S_{t-1} + k ⊗ v.
        # γ is per-head scalar (H,); broadcast to (1, H, 1, 1) for the
        # (B, H, D, D) state. Outer product k ⊗ v via einsum.
        S_prev = state["S"]                                              # (B, H, D, D)
        gamma_b = self.gamma.view(1, H, 1, 1)                            # (1, H, 1, 1)
        kv_outer = torch.einsum("bhi,bhj->bhij", k, v)                   # (B, H, D, D)
        S_new = gamma_b * S_prev + kv_outer                              # (B, H, D, D)

        # Retention readout: y = q · S (matrix-vector per head).
        # einsum 'bhi,bhij->bhj' contracts the i (key) axis.
        y_per_head = torch.einsum("bhi,bhij->bhj", q, S_new)             # (B, H, D)

        # Flatten heads, project, LN.
        y_flat = y_per_head.reshape(B, self.hidden_size)
        y_out = self.output_ln(self.out_linear(y_flat))                  # (B, hidden_size)

        return y_out, {"S": S_new}, {}
