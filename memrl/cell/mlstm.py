"""mLSTM (matrix LSTM) — Beck et al. 2024 (xLSTM, NeurIPS).

"xLSTM: Extended Long Short-Term Memory"
https://arxiv.org/abs/2405.04517

The matrix-memory variant from the xLSTM paper. mLSTM is what you get when
you take LSTM-style gating (input/forget/output gates) and graft it onto
a matrix-memory state in the linear-attention style. Two innovations
distinguish it from standard LSTM and from LinearTransformer:

  1. **Exponential gating** — i_t = exp(ĩ_t), f_t = exp(f̃_t). Gates are
     strictly positive but UNBOUNDED above, so they can amplify (not just
     attenuate) signal. Standard LSTM's sigmoid gates are bounded to (0,1)
     and can only attenuate.

  2. **Max-trick stabilization** — exponential gates would overflow without
     care. Beck et al. introduce a log-space stabilizer state m_t that
     tracks the running maximum log-magnitude and subtracts it. After
     stabilization, the EFFECTIVE gates are in (0, 1] but their RATIO
     encodes the unbounded amplification.

Why mLSTM for the GatedDeltaNet ablation table:
  Direct competitor in the "gated matrix memory" lane. Same matrix-memory
  commitment as GatedDeltaNet (Gated DeltaNet), but using LSTM-style gating
  instead of the delta rule. The pairwise ablation:

      ┌────────────┬─────────────────┬─────────────────────┐
      │            │  No gating      │  Gated              │
      ├────────────┼─────────────────┼─────────────────────┤
      │ No delta   │ LinearTransformer│ mLSTM (LSTM-style)  │
      │ Delta rule │ DeltaNet        │ GatedDeltaNet (delta + α)    │
      └────────────┴─────────────────┴─────────────────────┘

  GatedDeltaNet and mLSTM are the two "gated matrix memory" cells. The Phase A
  comparison answers: "for gated matrix memory under PPO, does the delta
  rule beat LSTM-style exponential gating?"

Math (per step, per head h):
    Projections:    q_t, k_t, v_t  ←  linear(x_t)             # (B, H, D)
                    i_raw, f_raw, o_raw  ←  linear(x_t)        # (B, H)
    K scaling:      k_t ← k_t / √d_head                        # softmax-style temper
    Output gate:    o_t = σ(o_raw_t)                           # standard sigmoid

    Log-space stabilizer (Beck et al. 2024 Eq. 25):
      m_t  =  max(log(f_t) + m_{t-1},  log(i_t))                # (B, H)
           =  max(f_raw_t + m_{t-1},  i_raw_t)                  # (since log(exp(.)) = identity)
      i'_t =  exp(i_raw_t − m_t)                                 ∈ (0, 1]
      f'_t =  exp(f_raw_t + m_{t-1} − m_t)                       ∈ (0, 1]

    State updates:
      C_t  =  f'_t · C_{t-1}  +  i'_t · (v_t  ⊗  k_t)             # matrix memory
      n_t  =  f'_t · n_{t-1}  +  i'_t · k_t                       # normalizer

    Readout:
      Cq   =  C_t · q_t                                          # (B, H, D)
      h̃_t  =  Cq  /  max(|n_t · q_t|,  1)                        # bounded by normalizer
      h_t  =  o_t  ⊙  h̃_t                                        # output gate

Notes on the stabilization:
  After the max-trick, BOTH i'_t and f'_t are in (0, 1]. Their RATIO
  i'_t / f'_t = exp(i_raw_t − f_raw_t − m_{t-1}) is what encodes "how
  much new info vs. how much old info." Beck et al. prove the stabilized
  recurrence is numerically equivalent to the unstabilized exponential
  one (Theorem in Appendix C of the xLSTM paper) — the stabilization is
  purely a numerical-precision trick.

  At t=0, m_{t-1}=0 (initial state). The first step's stabilizer is
  m_0 = max(f_raw_0, i_raw_0), and i'_0, f'_0 are computed relative to
  it. The recurrence is well-defined from the start.

K-scaling (1/√d_head):
  Same rationale as softmax attention: prevents (v ⊗ k) magnitudes from
  growing with d_head. Standard in linear-attention literature; matches
  the xLSTM reference implementation.

Architectural simplifications vs the full xLSTM block:
  - **No causal conv1d** (the block has a 1D conv before the SSM core,
    same as Mamba-2's block; we skip for the same reason).
  - **No skip connection / residual at the block level** — that's the
    xLSTM *block*, not the mLSTM cell.
  - **No GroupNorm before output** — we use LayerNorm for consistency
    with the rest of the cell family.

Deliberate omissions:
  - No dropout, no learnable initial state.
  - No sLSTM scalar variant — we have GRU and LSTM as RNN baselines;
    sLSTM's contribution (exp gating in a scalar LSTM) is subsumed.

State:        {"C": (B, n_heads, d_head, d_head),
               "n": (B, n_heads, d_head),
               "m": (B, n_heads)}
Side outputs: {}
Output:       (B, hidden_size)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from memrl.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask


class mLSTM(RecurrentCell):
    """Matrix LSTM (xLSTM) in single-step recurrent form."""

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
        self._k_scale = 1.0 / math.sqrt(d_head)
        H, D = n_heads, d_head

        # Pre-LN, then fused projection: [q | k | v | i_raw | f_raw | o_raw].
        # bias=False on QKV projections (consistent with the family); the
        # gate projections inherit no bias either — the LN affine before
        # them handles shifts.
        self.input_ln = nn.LayerNorm(input_size)
        d_in_proj = 3 * H * D + 3 * H
        self.in_proj = nn.Linear(input_size, d_in_proj, bias=False)

        # Output mixing.
        self.out_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_ln = nn.LayerNorm(hidden_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
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
            "C": torch.zeros(batch_size, H, D, D, device=device, dtype=dtype),
            "n": torch.zeros(batch_size, H, D, device=device, dtype=dtype),
            "m": torch.zeros(batch_size, H, device=device, dtype=dtype),
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

        # Pre-LN, fused projection.
        x_normed = self.input_ln(x)                                     # (B, input_size)
        proj = self.in_proj(x_normed)                                   # (B, 3*H*D + 3*H)

        # Slice into Q, K, V, and three gate raws.
        off = 0
        q_flat = proj[..., off : off + H * D]; off += H * D
        k_flat = proj[..., off : off + H * D]; off += H * D
        v_flat = proj[..., off : off + H * D]; off += H * D
        i_raw = proj[..., off : off + H]; off += H                      # log(input gate)
        f_raw = proj[..., off : off + H]; off += H                      # log(forget gate)
        o_raw = proj[..., off : off + H]                                # output gate logits

        q = q_flat.view(B, H, D)
        k = k_flat.view(B, H, D) * self._k_scale                         # 1/√d_head scaling
        v = v_flat.view(B, H, D)

        # Output gate: standard sigmoid in (0, 1).
        o = torch.sigmoid(o_raw)                                         # (B, H)

        # Log-space stabilizer (Beck et al. 2024 Eq. 25).
        # i_raw, f_raw are log(i), log(f) respectively (since exp(raw) = gate).
        m_prev = state["m"]                                              # (B, H)
        m_new = torch.maximum(f_raw + m_prev, i_raw)                     # (B, H)

        # Stabilized gates, both ∈ (0, 1].
        i_stab = torch.exp(i_raw - m_new)                                # (B, H)
        f_stab = torch.exp(f_raw + m_prev - m_new)                       # (B, H)

        # Matrix-memory and normalizer updates.
        C_prev = state["C"]                                              # (B, H, D, D)
        n_prev = state["n"]                                              # (B, H, D)

        # Outer product v ⊗ k → (B, H, D, D).
        vk_outer = torch.einsum("bhi,bhj->bhij", v, k)                   # (B, H, D, D)

        f_b4 = f_stab.unsqueeze(-1).unsqueeze(-1)                        # (B, H, 1, 1)
        i_b4 = i_stab.unsqueeze(-1).unsqueeze(-1)                        # (B, H, 1, 1)
        C_new = f_b4 * C_prev + i_b4 * vk_outer                          # (B, H, D, D)

        f_b3 = f_stab.unsqueeze(-1)                                      # (B, H, 1)
        i_b3 = i_stab.unsqueeze(-1)                                      # (B, H, 1)
        n_new = f_b3 * n_prev + i_b3 * k                                 # (B, H, D)

        # Readout: h̃ = (C q) / max(|n^T q|, 1).
        # The denominator clamp at 1 (not eps!) is intentional per the
        # xLSTM paper — it prevents division by small values from inflating
        # h̃ when the cell is "empty" and ⟨n, q⟩ is near zero. For full
        # memory ⟨n, q⟩ > 1 typically, so the denominator becomes the
        # normalizer; for empty memory the denominator is 1, so h̃ ≈ C q.
        Cq = torch.einsum("bhij,bhj->bhi", C_new, q)                     # (B, H, D)
        nq = (n_new * q).sum(dim=-1, keepdim=True)                       # (B, H, 1)
        denom = torch.clamp(nq.abs(), min=1.0)                           # (B, H, 1)
        h_tilde = Cq / denom                                             # (B, H, D)

        # Output gate.
        h = o.unsqueeze(-1) * h_tilde                                    # (B, H, D)

        # Flatten heads, project, LN.
        h_flat = h.reshape(B, self.hidden_size)
        y_out = self.output_ln(self.out_linear(h_flat))                  # (B, hidden_size)

        return y_out, {"C": C_new, "n": n_new, "m": m_new}, {}
