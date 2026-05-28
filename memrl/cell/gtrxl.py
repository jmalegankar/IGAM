"""Gated Transformer-XL (GTrXL) — Parisotto et al. 2020 (ICML).

"Stabilizing Transformers for Reinforcement Learning"
https://arxiv.org/abs/1910.06764

A Transformer-XL (Dai et al. 2019) adapted to be trainable in the RL setting,
where vanilla post-LN transformers are notoriously unstable. Two changes over
TrXL make it work:

  1. **Identity Map Reordering (pre-LN with an identity path).**  Each submodule
     (attention, MLP) operates on a LayerNorm'd copy of its input; the raw input
     flows to the output through the gate. At init the submodule contributes ~0,
     so the whole layer is the identity map — a Markov reactive policy — which is
     where stable RL training must start.

  2. **GRU gating instead of residual addition.**  Where TrXL does
     `out = x + Sublayer(LN(x))`, GTrXL does `out = GRUGate(x, Sublayer(LN(x)))`.
     The gate's update bias b_g > 0 makes it initially copy the input (identity),
     then learn to admit the submodule. Parisotto found GRU gating beats Highway,
     input, output, and SkipInit gating, and beats the residual baseline.

The Transformer-XL "memory" is the recurrent state here: a rolling cache of the
last `mem_len` layer-input embeddings, used as extended keys/values. Gradients
do not flow into the cache (TrXL stop-gradient) — which matches how the rollout
buffer detaches state at TBPTT chunk boundaries.

Single-step specialization. The classic TrXL processes a length-L segment and
needs the "relative shift" trick to build the L×(M+L) score matrix. With L=1
(one env step), there is a single query attending to M+1 keys at distances
0..M, so the Dai et al. relative-attention decomposition applies directly with
no shift trick:

    score_j = (q + u)·(W_kE h_j)  +  (q + v)·(W_kR R_{d_j})
            └ content term ┘        └ position term ┘

    where h_j are the cached+current embeddings, R_d is the sinusoidal
    relative-position embedding for distance d, and u, v are the global
    content/position bias vectors (per head).

Math (per step):
    e_t   = W_in x_t                                   # (B, d_model)
    ctx   = [mem ; e_t]                                # (B, M+1, d_model)
    # --- attention submodule (identity-map reordered) ---
    a     = MHA_rel(LN1(e_t)  as query,  LN1(ctx) as kv)
    h1    = GRUGate1(e_t, a)
    # --- MLP submodule ---
    f     = W2 ReLU(W1 LN2(h1))
    h2    = GRUGate2(h1, f)
    # --- recurrence: cache the layer input e_t ---
    mem'  = [mem[:, 1:] ; e_t]
    y_t   = h2                                         # (B, d_model = hidden_size)

References:
    Dai et al. 2019, Transformer-XL (arXiv 1901.02860) — segment recurrence + rel pos
    Parisotto et al. 2020, GTrXL (arXiv 1910.06764) — identity reorder + GRU gate
    Vaswani et al. 2017 — multi-head attention

State:        {"mem": (B, mem_len, d_model)}    # rolling embedding cache (detached context)
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


class _GRUGate(nn.Module):
    """GRU-style gating layer (Parisotto 2020 eq. 7–11).

        r = σ(W_r y + U_r x)
        z = σ(W_z y + U_z x − b_g)
        ĥ = tanh(W_g y + U_g (r ⊙ x))
        out = (1 − z) ⊙ x + z ⊙ ĥ

    x = residual/input stream, y = submodule output. b_g > 0 biases the gate
    toward the identity path (out ≈ x) at init — the stability mechanism.
    """

    def __init__(self, d_model: int, gate_bias: float = 2.0) -> None:
        super().__init__()
        self.Wr = nn.Linear(d_model, d_model, bias=False)
        self.Ur = nn.Linear(d_model, d_model, bias=False)
        self.Wz = nn.Linear(d_model, d_model, bias=False)
        self.Uz = nn.Linear(d_model, d_model, bias=False)
        self.Wg = nn.Linear(d_model, d_model, bias=False)
        self.Ug = nn.Linear(d_model, d_model, bias=False)
        self.b_g = nn.Parameter(torch.full((d_model,), float(gate_bias)))
        for m in (self.Wr, self.Ur, self.Wz, self.Uz, self.Wg, self.Ug):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.sigmoid(self.Wr(y) + self.Ur(x))
        z = torch.sigmoid(self.Wz(y) + self.Uz(x) - self.b_g)
        h_hat = torch.tanh(self.Wg(y) + self.Ug(r * x))
        return (1.0 - z) * x + z * h_hat


class GTrXL(RecurrentCell):
    """Single-layer Gated Transformer-XL in single-step recurrent form."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_heads: int = 4,
        mem_len: int = 64,
        d_ff: Optional[int] = None,
        gate_bias: float = 2.0,
    ) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must divide n_heads ({n_heads})")
        self.d_model = hidden_size
        self.n_heads = n_heads
        self.d_head = hidden_size // n_heads
        self.mem_len = mem_len
        # FFN expansion: LLM transformers use 4×, but for a small RL cell a
        # leaner 2× FFN keeps the param count closer to the recurrent baselines
        # without crippling capacity. Override via `d_ff` for the 4× faithful form.
        d_ff = d_ff if d_ff is not None else 2 * hidden_size

        # Input embedding (encoder dim → d_model).
        self.W_in = nn.Linear(input_size, self.d_model, bias=False)

        # Attention projections. W_kE = content keys, W_kR = position keys.
        self.W_q  = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_kE = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v  = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_kR = nn.Linear(self.d_model, self.d_model, bias=False)
        self.attn_out = nn.Linear(self.d_model, self.d_model, bias=False)
        # Global content/position bias vectors (Dai et al. u, v), per head.
        self.u_bias = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.zeros(n_heads, self.d_head))

        # Identity-map-reordering LayerNorms (pre-LN on each submodule input).
        self.ln_attn = nn.LayerNorm(self.d_model)
        self.ln_ff   = nn.LayerNorm(self.d_model)

        # Position-wise MLP submodule.
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, self.d_model),
        )

        # GRU gates replacing the two residual connections.
        self.gate_attn = _GRUGate(self.d_model, gate_bias)
        self.gate_ff   = _GRUGate(self.d_model, gate_bias)

        # Sinusoidal relative-position table for distances 0..mem_len.
        # R[d] is the encoding for "key is d steps before the current query".
        self.register_buffer("rel_pos", self._sinusoid(mem_len + 1, self.d_model),
                             persistent=False)

        self._reset_parameters()

    @staticmethod
    def _sinusoid(n_pos: int, dim: int) -> Tensor:
        pos = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)        # (n_pos, 1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / dim))                      # (dim/2,)
        pe = torch.zeros(n_pos, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe                                                          # (n_pos, dim)

    def _reset_parameters(self) -> None:
        for m in (self.W_in, self.W_q, self.W_kE, self.W_v, self.W_kR, self.attn_out):
            nn.init.xavier_uniform_(m.weight)
        for m in self.ff:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # u_bias, v_bias stay at zero init (standard for TrXL global biases).

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        device = self._resolve_device(device)
        return {
            "mem": torch.zeros(batch_size, self.mem_len, self.d_model,
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
        M, H, nh, dh = self.mem_len, self.d_model, self.n_heads, self.d_head

        e = self.W_in(x)                                       # (B, d_model)
        mem = state["mem"]                                     # (B, M, d_model)

        # Context = [mem ; current]. Length L = M+1, current query is last.
        ctx = torch.cat([mem, e.unsqueeze(1)], dim=1)          # (B, M+1, d_model)
        ctx_ln = self.ln_attn(ctx)                             # pre-LN (identity reorder)
        q_in   = self.ln_attn(e)                               # query from LN'd current

        # ── Relative multi-head attention (single query, M+1 keys) ──────────
        q = self.W_q(q_in).view(B, nh, dh)                     # (B, nh, dh)
        kE = self.W_kE(ctx_ln).view(B, M + 1, nh, dh)          # (B, L, nh, dh)
        v  = self.W_v(ctx_ln).view(B, M + 1, nh, dh)           # (B, L, nh, dh)

        # Relative position keys. Distance of context position j (0..M) from the
        # current query (at index M) is d_j = M − j. rel_pos[d] is the encoding.
        dists = (M - torch.arange(M + 1, device=x.device)).clamp(min=0)    # (L,)
        R = self.rel_pos[dists]                                # (L, d_model)
        kR = self.W_kR(R).view(M + 1, nh, dh)                  # (L, nh, dh)

        # Dai et al. score decomposition:
        #   content:  (q + u) · kE_j     position: (q + v) · kR_j
        q_u = q + self.u_bias                                  # (B, nh, dh)
        q_v = q + self.v_bias                                  # (B, nh, dh)
        content = torch.einsum("bhd,blhd->bhl", q_u, kE)       # (B, nh, L)
        position = torch.einsum("bhd,lhd->bhl", q_v, kR)       # (B, nh, L)
        scores = (content + position) / math.sqrt(dh)          # (B, nh, L)

        attn = torch.softmax(scores, dim=-1)                   # (B, nh, L)
        ctx_vec = torch.einsum("bhl,blhd->bhd", attn, v)       # (B, nh, dh)
        ctx_vec = ctx_vec.reshape(B, H)                        # (B, d_model)
        a = self.attn_out(ctx_vec)                             # (B, d_model)

        # Gate attention output against the residual stream e.
        h1 = self.gate_attn(e, a)                              # (B, d_model)

        # ── MLP submodule (identity reorder + gate) ─────────────────────────
        f = self.ff(self.ln_ff(h1))                            # (B, d_model)
        h2 = self.gate_ff(h1, f)                               # (B, d_model)

        # ── Recurrence: cache the layer-input embedding, drop oldest ────────
        new_mem = torch.cat([mem[:, 1:], e.unsqueeze(1)], dim=1)   # (B, M, d_model)

        return h2, {"mem": new_mem}, {}
