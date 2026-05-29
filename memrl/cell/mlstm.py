"""mLSTM (matrix LSTM) — Beck et al. 2024 (xLSTM, NeurIPS).

"xLSTM: Extended Long Short-Term Memory"
https://arxiv.org/abs/2405.04517

The matrix-memory variant from the xLSTM paper, **with the block-level
wrappers**: depthwise causal Conv1d (K=4) before the cell core, per-head
GroupNorm on the readout, and an additive block residual on the cell input.
These are exactly the components the original implementation listed as
"Architectural simplifications vs the full xLSTM block" — they are now back in.

mLSTM grafts LSTM-style gating (input/forget/output) onto a matrix-memory
state in the linear-attention style. Two innovations distinguish it from
standard LSTM and from LinearTransformer:

  1. **Exponential gating** — i_t = exp(ĩ_t), f_t = exp(f̃_t). Gates are
     strictly positive but UNBOUNDED above, so they can amplify (not just
     attenuate) signal. Standard LSTM's sigmoid gates are bounded to (0,1).

  2. **Max-trick stabilization** — exponential gates would overflow without
     care. A log-space stabilizer state m_t tracks the running maximum and
     subtracts it. After stabilization, the EFFECTIVE gates are in (0, 1]
     but their RATIO encodes the unbounded amplification.

Math (per step, single block):

    Input residual carrier + pre-norm:
        x_H_t = W_in · x_t                                # (B, H_tot)
        u_t   = LN(x_H_t)                                 # (B, H_tot) pre-norm

    Depthwise causal Conv1d (K=4) — short-range mixing before the recurrence:
        buf   ∈ ℝ^{B × H_tot × (K-1)}                     # rolling buffer state
        win   = [buf_0, buf_1, buf_2, u_t]                 # (B, H_tot, K=4)
        c_t   = conv1d_dw(win)                             # (B, H_tot, 1) → squeeze
        c_t   = SiLU(c_t)                                  # xLSTM activation
        buf_new = win[:, :, 1:]                            # roll forward

    Projections (Q, K, V per head; raw gate logits) from c_t:
        q_t, k_t, v_t  ←  linear(c_t)                     # (B, H_heads, D)
        i_raw, f_raw, o_raw  ←  linear(c_t)                # (B, H_heads)
        k_t ← k_t / √d_head                               # softmax-style temper
        o_t = σ(o_raw)                                     # standard sigmoid

    Log-space stabilizer (Beck et al. 2024 Eq. 25):
        m_t  =  max(f_raw + m_{t-1},  i_raw)               # (B, H_heads)
        i'_t =  exp(i_raw − m_t)                                  ∈ (0, 1]
        f'_t =  exp(f_raw + m_{t-1} − m_t)                        ∈ (0, 1]

    State updates:
        C_t  =  f'_t · C_{t-1}  +  i'_t · (v_t  ⊗  k_t)     # matrix memory
        n_t  =  f'_t · n_{t-1}  +  i'_t · k_t               # normalizer

    Readout:
        Cq    =  C_t · q_t                                  # (B, H_heads, D)
        h̃_t   =  Cq  /  max(|n_t · q_t|,  1)                # bounded by normalizer
        h_t   =  o_t  ⊙  h̃_t                                # output gate, then flatten

    GroupNorm (per-head, num_groups = n_heads) + output projection + residual:
        h_norm = GroupNorm(h_t.flatten_heads)              # (B, H_tot)
        y_t    = W_out · h_norm                            # (B, H_tot)
        out    = x_H_t  +  y_t                              # block residual

Why mLSTM for the GatedDeltaNet ablation table:
  Direct competitor in the "gated matrix memory" lane. Same matrix-memory
  commitment as GatedDeltaNet, but LSTM-style gating instead of the delta rule.
  The Phase A comparison answers: "for gated matrix memory under PPO, does the
  delta rule beat LSTM-style exponential gating?"

K-scaling (1/√d_head) — same rationale as softmax attention; prevents
(v ⊗ k) magnitudes from growing with d_head.

Deliberate omissions (unchanged from before):
  - No dropout, no learnable initial state.
  - No sLSTM scalar variant.
  - No dual-arm up-projection / swish-gate (the very-outer xLSTM block has
    an expanded gate arm that multiplies the cell output; we keep the
    depthwise-conv + cell + GroupNorm + residual core, which is the
    "mLSTM block" most papers reference).

State:        {"C": (B, n_heads, d_head, d_head),
               "n": (B, n_heads, d_head),
               "m": (B, n_heads),
               "conv_buf": (B, hidden_size, K-1)}
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


class mLSTM(RecurrentCell):
    """Matrix LSTM (xLSTM) block in single-step recurrent form."""

    CONV_K = 4   # kernel size of the depthwise causal Conv1d, per xLSTM

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
        Hh, D = n_heads, d_head

        # Input projection (input_size → hidden_size). Also the residual carrier
        # — block skip is on x_H, not on raw x, so the residual lives at H_tot
        # regardless of encoder dim.
        self.input_linear = nn.Linear(input_size, hidden_size, bias=False)

        # Pre-LN on the projected input.
        self.input_ln = nn.LayerNorm(hidden_size)

        # Depthwise causal Conv1d (K=4) before the cell core — per Beck 2024.
        # IMPLEMENTED MANUALLY (Parameter + einsum) rather than nn.Conv1d:
        # at (B, H, K=4), nn.Conv1d's per-call dispatch overhead dominates the
        # trivial elementwise math and tanks step throughput by ~30×. A depthwise
        # conv is mathematically (window ⊙ kernel).sum(-1), which is a single
        # einsum with no extra Python/kernel-dispatch cost. Causal padding lives
        # in the explicit (K-1)-long buffer state.
        self.conv_weight = nn.Parameter(torch.empty(hidden_size, self.CONV_K))

        # Q, K, V projection (fused: 3·hidden_size). Per the xLSTM reference
        # (NX-AI/xlstm), the gates take CONCAT(q, k, v) as their input — not
        # the raw conv'd signal — so they can see the per-head queries/keys
        # /values directly. We keep QKV fused for efficiency.
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        # Gate projections take [q | k | v] (3·hidden_size) and emit per-head
        # logits (n_heads). They have biases — which the special init below
        # exploits: zero-weights ⇒ at init the logits ARE the biases.
        self.igate = nn.Linear(3 * hidden_size, n_heads, bias=True)
        self.fgate = nn.Linear(3 * hidden_size, n_heads, bias=True)
        self.ogate = nn.Linear(3 * hidden_size, n_heads, bias=True)

        # Per-head GroupNorm on the readout (the xLSTM block normalizes each
        # head's output independently; num_groups=n_heads, num_channels=H_tot
        # makes each group exactly one head's d_head channels).
        self.head_gn = nn.GroupNorm(num_groups=n_heads, num_channels=hidden_size)

        # Output projection (within-block, NOT the family-style output_ln).
        self.out_linear = nn.Linear(hidden_size, hidden_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_linear.weight)
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)
        # Conv kernel — small normal; depthwise-only so fan_in is K.
        nn.init.normal_(self.conv_weight, std=1.0 / math.sqrt(self.CONV_K))

        # ── Gate inits per the NX-AI/xlstm reference ──────────────────────
        # Weight=0 + bias only ⇒ at init the gate logit IS the bias, with the
        # data-dependent term turning on as the gate weight learns.
        # Forget gate: bias linspace 3→6 across heads. exp(f_raw) is then
        #   large, but after stabilization the EFFECTIVE forget gate f' →
        #   exp(0) = 1 (perfect retention) while i' → exp(-bias) ≈ 0
        #   (very little new info admitted). That's the "remember by default,
        #   open up to inputs only when learned" prior.
        # Input gate: bias N(0, 0.1) — small noise, no forget-style prior.
        # Output gate: standard (Xavier weight, zero bias).
        for g in (self.igate, self.fgate):
            nn.init.zeros_(g.weight)
        with torch.no_grad():
            self.fgate.bias.copy_(torch.linspace(3.0, 6.0, self.n_heads))
        nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)
        nn.init.xavier_uniform_(self.ogate.weight)
        nn.init.zeros_(self.ogate.bias)

    # --- interface ---------------------------------------------------------

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        device = self._resolve_device(device)
        Hh, D = self.n_heads, self.d_head
        return {
            "C":        torch.zeros(batch_size, Hh, D, D, device=device, dtype=dtype),
            "n":        torch.zeros(batch_size, Hh, D, device=device, dtype=dtype),
            "m":        torch.zeros(batch_size, Hh, device=device, dtype=dtype),
            # Rolling buffer of the past K-1 (post-pre-LN) inputs to the conv.
            # Zero at episode boundaries — the conv sees padded zeros for
            # the first K-1 steps, which is the standard causal-conv convention.
            "conv_buf": torch.zeros(batch_size, self.hidden_size, self.CONV_K - 1,
                                    device=device, dtype=dtype),
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
        Hh, D = self.n_heads, self.d_head

        # ── Block input: project + residual carrier + pre-LN ────────────────
        x_H = self.input_linear(x)                                       # (B, H_tot)
        u = self.input_ln(x_H)                                           # (B, H_tot)

        # ── Depthwise causal Conv1d (K=4) — manual (window ⊙ weight).sum ────
        # Equivalent to nn.Conv1d(C, C, K, groups=C) but avoids the per-call
        # overhead that strangles step throughput at L=K=4 (see __init__ note).
        buf_prev = state["conv_buf"]                                     # (B, H_tot, K-1)
        win = torch.cat([buf_prev, u.unsqueeze(-1)], dim=-1)             # (B, H_tot, K)
        c = torch.einsum("bck,ck->bc", win, self.conv_weight)            # (B, H_tot)
        c = F.silu(c)                                                    # xLSTM activation
        # Roll the buffer one step forward (drop oldest, append current).
        conv_buf_new = win[..., 1:]                                      # (B, H_tot, K-1)

        # ── QKV from the conv'd signal; gates from CONCAT(q,k,v) ────────────
        # Match the official xLSTM cell: q, k, v are projected first, then the
        # gate Linears see [q|k|v] as their input (NOT the raw c).
        proj_qkv = self.qkv_proj(c)                                       # (B, 3·hidden_size)
        H_tot = self.hidden_size
        q_flat = proj_qkv[..., :H_tot]
        k_flat = proj_qkv[..., H_tot : 2 * H_tot]
        v_flat = proj_qkv[..., 2 * H_tot : 3 * H_tot]

        i_raw = self.igate(proj_qkv)                                      # (B, Hh)
        f_raw = self.fgate(proj_qkv)                                      # (B, Hh)
        o_raw = self.ogate(proj_qkv)                                      # (B, Hh)

        q = q_flat.view(B, Hh, D)
        k = k_flat.view(B, Hh, D) * self._k_scale
        v = v_flat.view(B, Hh, D)

        # Output gate (standard sigmoid).
        o = torch.sigmoid(o_raw)                                         # (B, Hh)

        # ── Log-space stabilizer (Beck et al. 2024 Eq. 25) ──────────────────
        m_prev = state["m"]                                              # (B, Hh)
        m_new = torch.maximum(f_raw + m_prev, i_raw)                     # (B, Hh)
        i_stab = torch.exp(i_raw - m_new)                                # (B, Hh)
        f_stab = torch.exp(f_raw + m_prev - m_new)                       # (B, Hh)

        # ── Matrix-memory and normalizer updates ────────────────────────────
        C_prev = state["C"]
        n_prev = state["n"]
        vk_outer = torch.einsum("bhi,bhj->bhij", v, k)                   # (B, Hh, D, D)

        f_b4 = f_stab.unsqueeze(-1).unsqueeze(-1)
        i_b4 = i_stab.unsqueeze(-1).unsqueeze(-1)
        C_new = f_b4 * C_prev + i_b4 * vk_outer

        f_b3 = f_stab.unsqueeze(-1)
        i_b3 = i_stab.unsqueeze(-1)
        n_new = f_b3 * n_prev + i_b3 * k

        # ── Readout: h̃ = Cq / max(|n^T q|, 1), then output-gate ────────────
        Cq = torch.einsum("bhij,bhj->bhi", C_new, q)                     # (B, Hh, D)
        nq = (n_new * q).sum(dim=-1, keepdim=True)                       # (B, Hh, 1)
        denom = torch.clamp(nq.abs(), min=1.0)
        h_tilde = Cq / denom                                              # (B, Hh, D)
        h = o.unsqueeze(-1) * h_tilde                                     # (B, Hh, D)
        h_flat = h.reshape(B, self.hidden_size)                           # (B, H_tot)

        # ── Per-head GroupNorm, output projection, block residual ──────────
        h_norm = self.head_gn(h_flat)                                     # (B, H_tot)
        y = self.out_linear(h_norm)                                       # (B, H_tot)
        out = x_H + y                                                     # block residual

        return out, {"C": C_new, "n": n_new, "m": m_new,
                     "conv_buf": conv_buf_new}, {}
