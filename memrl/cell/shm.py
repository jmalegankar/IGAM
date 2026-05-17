"""Stable Hadamard Memory (SHM) — Le et al. 2024 (ICLR 2025).

"Stable Hadamard Memory: Revitalizing Memory-Augmented Agents for RL"
https://arxiv.org/abs/2410.10132

A matrix-memory cell where decay is ELEMENT-WISE (Hadamard) rather than
scalar-per-head. Each element of the H×H memory matrix can independently
decay OR amplify per step, via a calibration matrix C ∈ [0, 2]^{H×H}.

Why this completes the 2×2 ablation:
  - LinearTransformer:  W += k v^T                            (no gating)
  - RetNet:             W = γ W + k v^T                        (fixed scalar decay)
  - Mamba-2:            W = A_d W + dt B v^T                   (selective scalar decay)
  - DeltaNet:           W = W (I − β kk^T) + β v k^T            (delta rule, no decay)
  - GatedDeltaNet: W = α W (I − β kk^T) + β v k^T          (delta + selective scalar α)
  - SHM (this cell):    M = M ⊙ C + (η v) k^T                  (element-wise calibration, no delta)

SHM is the matrix-memory cell with the FINEST granularity of decay: a
different multiplier per element of M, rather than one per head/cell.
The trade-off is no delta rule (just additive accumulation, like LinTrans
but with calibrated decay).

Math (per step):

    Projections:
        k = ReLU(W_k x);     q = ReLU(W_q x);     v = W_v x        # ∈ ℝ^H each
        vc = W_vc x          # ∈ ℝ^H — drives the calibration matrix
        η = σ(W_η x)         # ∈ ℝ — scalar write strength
    Normalize:
        k̃ = k / (Σ_i k_i + ε)   # L1 normalize (paper convention)
        q̃ = q / (Σ_i q_i + ε)

    Random θ row (per step, per sample):
        l_t ~ Uniform({0, 1, ..., L-1});   θ_t = θ_matrix[l_t]      # ∈ ℝ^H

    Calibration matrix (per step):
        C_t = 1 + tanh(θ_t ⊗ vc_t)         # (H, H), each elem ∈ [0, 2]

    State update:
        M_t = M_{t-1} ⊙ C_t  +  (η_t · v_t) ⊗ k̃_t                  # (H, H)

    Read:
        y_t = M_t · q̃_t                                            # (H,)

The random θ_t selection is the KEY trick. Prop 4 of the paper proves
E[∏ C_t] = 1 when θ_t are independent across t — so memory products
neither vanish nor explode in expectation. Without the randomness, the
product is correlated and can diverge (this is what mLSTM's max-trick
stabilizer fails at).

Bug in the published reference code:
    Their `retrieve_theta` does `uniform_(0, 1).long()` which always
    truncates to row 0. Clearly a typo for `uniform_(0, L).long()`.
    We implement the corrected version using `torch.randint(0, L, ...)`.

State:        {"M": (B, hidden_size, hidden_size)}
Side outputs: {}                                    # no innovation analog
Output:       (B, hidden_size)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from memrl.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask


class SHM(RecurrentCell):
    """Stable Hadamard Memory (Le et al. 2024 ICLR 2025)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        L: int = 128,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)
        self.mem_size = hidden_size
        self.L = L
        self.eps = eps
        H = hidden_size

        # Pre-LN — consistent with the rest of the cell family.
        self.input_ln = nn.LayerNorm(input_size)

        # Q, K, V, V_C projections (paper: bias=False).
        self.W_k = nn.Linear(input_size, H, bias=False)
        self.W_q = nn.Linear(input_size, H, bias=False)
        self.W_v = nn.Linear(input_size, H, bias=False)
        self.W_vc = nn.Linear(input_size, H, bias=False)
        # Scalar write-strength gate.
        self.W_eta = nn.Linear(input_size, 1, bias=False)

        # θ parameter bank: L rows, each H-dim. One row is randomly sampled
        # per step per sample — the Prop-4 independence-across-t condition.
        self.theta_matrix = nn.Parameter(torch.empty(L, H))

        # Output mixing — Linear + LN, matches the rest of the cell family.
        self.out_linear = nn.Linear(H, H, bias=False)
        self.output_ln = nn.LayerNorm(H)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Paper inits theta_matrix via Xavier uniform. Other projections
        # follow our cell-family convention (Xavier uniform on data-path
        # linears).
        nn.init.xavier_uniform_(self.theta_matrix)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_vc.weight)
        nn.init.xavier_uniform_(self.W_eta.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)

    # --- interface ---------------------------------------------------------

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        device = self._resolve_device(device)
        return {
            "M": torch.zeros(
                batch_size, self.mem_size, self.mem_size,
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
        H = self.mem_size

        # Pre-LN, then all projections.
        x_normed = self.input_ln(x)

        k = F.relu(self.W_k(x_normed))                                    # (B, H)
        q = F.relu(self.W_q(x_normed))                                    # (B, H)
        v = self.W_v(x_normed)                                            # (B, H)
        vc = self.W_vc(x_normed)                                          # (B, H)

        # L1-normalize K, Q. The paper applies this to prevent the additive
        # accumulation from blowing up the state magnitudes (k̃ sums to ~1).
        k_norm = k / (k.sum(dim=-1, keepdim=True) + self.eps)             # (B, H)
        q_norm = q / (q.sum(dim=-1, keepdim=True) + self.eps)             # (B, H)

        # Scalar write strength.
        eta = torch.sigmoid(self.W_eta(x_normed))                         # (B, 1)

        # Random θ row per sample. CORRECTED from the reference repo's bug
        # (their `uniform_(0, 1).long()` always returned 0).
        row_idx = torch.randint(0, self.L, (B,), device=x.device)
        theta = self.theta_matrix[row_idx]                                # (B, H)

        # Calibration matrix C = 1 + tanh(θ_t ⊗ vc_t). Outer product gives
        # an H×H matrix per batch, each element ∈ [0, 2].
        C_logits = torch.einsum("bi,bj->bij", theta, vc)                  # (B, H, H)
        C = 1.0 + torch.tanh(C_logits)                                    # (B, H, H), elem ∈ [0, 2]

        # State update: M_new = M_prev ⊙ C + (η v) ⊗ k̃.
        M_prev = state["M"]                                               # (B, H, H)
        write = eta * v                                                   # (B, H), broadcasts (B,1)·(B,H)
        outer = torch.einsum("bi,bj->bij", write, k_norm)                 # (B, H, H)
        M_new = M_prev * C + outer                                        # (B, H, H)

        # Read: y = M_new · q̃.
        y = torch.einsum("bij,bj->bi", M_new, q_norm)                     # (B, H)

        # Output mixing.
        y_out = self.output_ln(self.out_linear(y))                        # (B, H)

        return y_out, {"M": M_new}, {}
