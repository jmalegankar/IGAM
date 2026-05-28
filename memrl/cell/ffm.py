"""Fast and Forgetful Memory (FFM) — Morad, Kortvelesy, Liwicki, Prorok 2023 (NeurIPS).

"Reinforcement Learning with Fast and Forgetful Memory"
https://arxiv.org/abs/2310.04128   (ref impl: github.com/proroklab/ffm)

FFM is the POPGym recurrent SOTA: a drop-in RNN replacement that beats GRU on 17
of 21 POPGym tasks while training orders of magnitude faster (the aggregation is
an associative scan ⇒ O(log T) parallel depth). Its inductive bias comes from
computational-psychology models of human memory: information is stored across a
*bank of fixed exponential-decay traces at multiple timescales*, each modulated
by an oscillation, so the cell natively represents "how long ago" and "with what
periodicity" an event happened.

Core: the **complex multi-scale aggregator.** Project the input to `memory_size`
trace channels, then spread each trace across `context_size` oscillators. State
S ∈ ℂ^{m×c} updates by a per-(trace, context) complex decay:

    γ[k, c] = exp(−softplus(a_k) + i·b_c)        # |γ| = exp(−softplus(a_k)) ∈ (0,1)
    S_t[k, c] = γ[k, c] · S_{t-1}[k, c] + u_t[k]  # u broadcast over the c oscillators

  * a_k (memory_size): log-decay → trace k forgets at rate softplus(a_k). Small
    rate = long memory. Geometric init spans timescales [1, max_timescale].
  * b_c (context_size): angular frequency of oscillator c. Geometric init spans
    periods [1, max_period]. The oscillation lets a single decaying trace encode
    *when* within its window an event occurred (phase), not just *that* it
    decayed — this is what separates FFM from a plain leaky integrator.

Full FFM (vs the ablations FFM-NI / FFM-NO that drop gates):
    u_t = (W_pre x_t) ⊙ σ(W_ig x_t)              # input gate
    S_t = aggregate(u_t)                          # complex multi-scale traces
    z_t = LN(W_mix · vec(ℜ,ℑ of S_t))            # readout over flattened state
    g_t = σ(W_og x_t)                             # output gate
    y_t = g_t ⊙ z_t + (1 − g_t) ⊙ (W_skip x_t)   # gated residual

State is m·c complex numbers — interpretable: every (decay-rate, period) pair has
a known timescale and contextual period (paper §4).

Deliberate omissions (cell-vs-block split, matching s4d.py / lru.py):
  - No parallel associative scan in forward_sequence; the base loop runs `step` T
    times. The scan is the "fast" in FFM; an override is the natural speedup.
  - No CNN/Conv front-end; encoder lives outside the cell.

State:        {"S": (B, memory_size, context_size, 2)}   # complex, real-rep
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


class FFM(RecurrentCell):
    """Fast and Forgetful Memory in single-step recurrent form.

    Args:
        input_size:     C  encoder dim
        hidden_size:    H  hidden / output dim
        memory_size:    m  number of exponential-decay trace channels
        context_size:   c  number of oscillation periods per trace
        min_timescale:  shortest decay timescale (steps) at init
        max_timescale:  longest decay timescale (steps) at init
        max_period:     longest oscillation period (steps) at init
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int = 16,
        context_size: int = 8,
        min_timescale: float = 1.0,
        max_timescale: float = 1024.0,
        max_period: float = 1024.0,
    ) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)
        if not 0.0 < min_timescale < max_timescale:
            raise ValueError(f"need 0 < min_timescale < max_timescale, "
                             f"got ({min_timescale}, {max_timescale})")
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.context_size = context_size
        m, c, H = memory_size, context_size, hidden_size

        # ── Aggregator decay (a) and oscillation (b) parameters ──────────────
        # Decay: softplus(a_k) = α_k = 1/τ_k, τ_k geometric over [min, max] timescale.
        # Store a via the inverse-softplus of the target α so softplus(a)=α at init.
        tau = torch.exp(torch.linspace(math.log(min_timescale),
                                       math.log(max_timescale), m))     # (m,)
        alpha = 1.0 / tau                                               # (m,) target decay rate
        a_init = torch.log(torch.expm1(alpha.clamp(min=1e-6)))          # inverse softplus
        self.a = nn.Parameter(a_init)

        # Oscillation: angular freq b_c = 2π / period_c, period geometric in [1, max_period].
        period = torch.exp(torch.linspace(0.0, math.log(max_period), c))  # (c,)
        b_init = 2.0 * math.pi / period                                 # (c,) angular freq
        self.b = nn.Parameter(b_init)

        # ── Projections + gates (full FFM) ───────────────────────────────────
        self.pre     = nn.Linear(input_size, m, bias=False)     # input → traces
        self.in_gate = nn.Linear(input_size, m, bias=True)      # input gate
        self.mix     = nn.Linear(2 * m * c, H, bias=False)      # readout over flat state
        self.out_ln  = nn.LayerNorm(H)
        self.out_gate = nn.Linear(input_size, H, bias=True)     # output gate
        self.skip     = nn.Linear(input_size, H, bias=False)    # gated-residual skip

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for lin in (self.pre, self.mix, self.skip):
            nn.init.xavier_uniform_(lin.weight)
        for lin in (self.in_gate, self.out_gate):
            nn.init.xavier_uniform_(lin.weight)
            # Bias the gates slightly open at init so signal flows while the
            # cell learns (σ(0.5) ≈ 0.62 input, output gates start mid-range).
            nn.init.zeros_(lin.bias)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        device = self._resolve_device(device)
        return {
            "S": torch.zeros(batch_size, self.memory_size, self.context_size, 2,
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

        # ── Input gate + projection to traces ───────────────────────────────
        u = self.pre(x) * torch.sigmoid(self.in_gate(x))        # (B, m) real

        # ── Complex multi-scale decay: γ[k,c] = exp(−softplus(a_k) + i·b_c) ──
        alpha = F.softplus(self.a)                              # (m,) > 0
        gamma = torch.exp(-alpha.unsqueeze(-1)
                          + 1j * self.b.unsqueeze(0))           # (m, c) cfloat

        S_prev = torch.view_as_complex(state["S"])             # (B, m, c) cfloat
        # u broadcast over the c oscillators; γ broadcast over batch.
        S_new = gamma.unsqueeze(0) * S_prev + u.unsqueeze(-1).to(gamma.dtype)  # (B, m, c)

        # ── Readout over flattened (real, imag) state ───────────────────────
        s_flat = torch.view_as_real(S_new).reshape(B, -1)      # (B, 2*m*c)
        z = self.out_ln(self.mix(s_flat))                      # (B, H)

        # ── Output gate (gated residual) ────────────────────────────────────
        g = torch.sigmoid(self.out_gate(x))                    # (B, H)
        y = g * z + (1.0 - g) * self.skip(x)                   # (B, H)

        return y, {"S": torch.view_as_real(S_new)}, {}
