"""LMU baseline — Voelker, Kajić, Eliasmith 2019 (NeurIPS).

"Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks"
https://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks

The canonical scalar-input LMU — explicitly NOT the gated / multichannel
/ OrthoLayer / dynamic-query variant from the lmu_ppo thesis (`GatedLMU`
/ `SelectiveLMU`, which live on the `gated-lmu` branch). Every gated /
selective enhancement is stripped:
  - No gating (softsign_sum / tanh_product); u_t flows additively into memory.
  - No W_pre OrthoLayer / Cayley updates.
  - No multichannel u (scalar per timestep, as in Voelker 2019 Eq. 3).
  - No dynamic W_query read head; we read m_t directly via W_m.
  - No spectral_norm on E_h or W_h.
  - No anti-collapse residual on the write.

What's kept (the canonical formulation):

Math (per step):
    u_t = e_x · x_t  +  e_h · h_{t-1}  +  e_m · m_{t-1}              # (B, 1) scalar
    m_t = A_d m_{t-1}  +  B_d u_t                                    # (B, d) memory vector
    h_t = tanh(W_x x_t  +  W_h h_{t-1}  +  W_m m_t)                  # (B, n) hidden state

Where:
    A_d, B_d  are the ZOH-discretized HiPPO-LegT matrices (dt=1, theta=window).
              Fixed at construction, registered as buffers. The orthogonal-
              Legendre-polynomial basis is the LMU's defining property; making
              A, B trainable forfeits that interpretation. keras-lmu's default
              is trainable_A=False, trainable_B=False, which we match.

    e_x, e_h, e_m  are 1×in projections (linear, no bias) — they collapse the
                   input/hidden/memory contributions into the scalar u_t.

    W_x, W_h, W_m  are n×in projections (linear, no bias) — they form the
                   hidden update.

Discretization (`torch.linalg.matrix_exp`):
    Continuous LMU:   θ ṁ(t) = A m(t) + B u(t)
    ZOH at dt=1:      m_{k+1} = A_d m_k + B_d u_k
                      A_d = expm(A)
                      B_d = A^{-1} (A_d − I) B
    We compute A, B in float64 for the discretization, then cast to float32
    for storage. scipy.signal.cont2discrete would give the same numerics; we
    use torch.linalg.matrix_exp to avoid the scipy dependency.

Initialization (matches Voelker 2019 / hrshtv reference impl):
    e_x.weight, e_h.weight:  LeCun uniform — U[−√(3/fan_in), √(3/fan_in)]
    e_m.weight:               zeros (silent memory contribution at init)
    W_x, W_h, W_m:           Xavier normal
    A_d, B_d:                fixed Legendre-LegT, not trained

Deliberate omissions:
  - No trainable theta. keras-lmu offers this as `trainable_theta=True`;
    making theta differentiable couples A_d, B_d to the optimizer, costs
    a matrix-exp per forward, and complicates checkpointing. Static theta
    is the standard baseline; add a flag only if a Phase A ablation needs it.
  - No FFT-parallel variant (Chilkuri & Eliasmith 2021). Single-step
    recurrent only, per ADR 0001. The FFT path is a `forward_sequence`
    override we may add later in Phase B.
  - No layer norm. The canonical LMU doesn't use it; bolting it on would
    confuse the baseline-vs-modern comparison story.

State:        {"h": (B, hidden_size), "m": (B, memory_size)}
Side outputs: {}                              # no innovation, no diagnostics
Output:       h_t directly; output_size == hidden_size.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from memrl.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask


# --- Legendre-LegT state-space matrices ------------------------------------


def _legt_zoh_matrices(memory_size: int, theta: float) -> tuple[Tensor, Tensor]:
    """Build continuous HiPPO-LegT (A, B) per Voelker 2019 Eq. 2,
    then ZOH-discretize at dt=1. Returns (A_d, B_d) as float32 tensors.

    A: (d, d)         B: (d, 1)
    """
    d = memory_size
    Q = torch.arange(d, dtype=torch.float64)                            # (d,)
    R = ((2.0 * Q + 1.0) / theta).unsqueeze(1)                          # (d, 1)
    i, j = torch.meshgrid(Q, Q, indexing="ij")                          # both (d, d)

    # (-1)^k via parity check — avoids potentially-complex pow of negative
    # base with float exponent on some torch versions.
    diag_parity = ((i - j + 1).long() % 2 == 0)                         # True if exponent even
    diag_sign = torch.where(diag_parity, 1.0, -1.0).to(torch.float64)   # (d, d)
    A_cont = R * torch.where(i < j, -1.0, diag_sign)                    # (d, d)

    q_parity = (Q.long() % 2 == 0)                                      # (d,)
    sign_Q = torch.where(q_parity, 1.0, -1.0).to(torch.float64).unsqueeze(1)  # (d, 1)
    B_cont = R * sign_Q                                                 # (d, 1)

    # ZOH at dt=1: A_d = expm(A);  B_d = A^{-1}(A_d - I) B
    Id = torch.eye(d, dtype=torch.float64)
    A_d = torch.linalg.matrix_exp(A_cont)                               # (d, d)
    B_d = torch.linalg.solve(A_cont, (A_d - Id) @ B_cont)               # (d, 1)

    return A_d.to(torch.float32), B_d.to(torch.float32)


def _lecun_uniform_(tensor: Tensor) -> None:
    """LeCun uniform initialization: U[−√(3/fan_in), √(3/fan_in)].

    Used for e_x and e_h per the Voelker / hrshtv reference. Equivalent
    to Xavier uniform when fan_out=1 (which is our case for these
    projections), but spelled out explicitly to match the reference.
    """
    fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
    limit = (3.0 / fan_in) ** 0.5
    nn.init.uniform_(tensor, -limit, limit)


# --- cell ------------------------------------------------------------------


class LMU(RecurrentCell):
    """Canonical scalar-input Legendre Memory Unit (Voelker 2019)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,
        theta: float,
    ) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.theta = theta

        # ── 1×in projections that collapse to the scalar u_t ─────────────
        # The "e_*" naming follows Voelker 2019. bias=False — the encoder
        # is a pure linear combination, with no constant offset.
        self.e_x = nn.Linear(input_size, 1, bias=False)
        self.e_h = nn.Linear(hidden_size, 1, bias=False)
        self.e_m = nn.Linear(memory_size, 1, bias=False)

        # ── n×in projections for the hidden update ───────────────────────
        # No bias on any — same as the reference. Adding biases here breaks
        # the equivalence to the canonical equations and gives the cell a
        # learnable offset that the LSTM/GRU baselines don't have.
        self.W_x = nn.Linear(input_size, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_m = nn.Linear(memory_size, hidden_size, bias=False)

        # ── ZOH-discretized HiPPO-LegT matrices, fixed (NOT trained) ─────
        A_d, B_d = _legt_zoh_matrices(memory_size, theta)
        self.register_buffer("A_d", A_d)                                # (d, d)
        self.register_buffer("B_d", B_d)                                # (d, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Encoder projections to u_t: LeCun uniform on x/h, zeros on m.
        # The e_m=0 init is deliberate — at training start, memory does NOT
        # contribute to the next u, so the cell behaves like a vanilla RNN
        # for the first few updates. e_m grows naturally if memory becomes
        # useful for the task.
        _lecun_uniform_(self.e_x.weight)
        _lecun_uniform_(self.e_h.weight)
        nn.init.zeros_(self.e_m.weight)

        # Hidden update: Xavier normal (calibrated for the downstream tanh).
        nn.init.xavier_normal_(self.W_x.weight)
        nn.init.xavier_normal_(self.W_h.weight)
        nn.init.xavier_normal_(self.W_m.weight)

    # --- interface ---------------------------------------------------------

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        device = self._resolve_device(device)
        return {
            "h": torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            "m": torch.zeros(batch_size, self.memory_size, device=device, dtype=dtype),
        }

    def step(
        self,
        x: Tensor,
        state: State,
        episode_start: Optional[Tensor] = None,
    ) -> tuple[Tensor, State, SideOutputs]:
        if episode_start is not None:
            state = apply_episode_mask(state, episode_start)

        h_prev = state["h"]
        m_prev = state["m"]

        # ── Step 1: encode to scalar u_t ─────────────────────────────────
        # Each e_* projection: (B, *) → (B, 1). Their sum is (B, 1).
        u_t = self.e_x(x) + self.e_h(h_prev) + self.e_m(m_prev)         # (B, 1)

        # ── Step 2: Legendre memory update ───────────────────────────────
        # m_t = A_d m_{t-1} + B_d u_t
        # F.linear(input, weight) computes input @ weight.T, so passing
        # A_d (shape (d,d)) gives m_prev @ A_d.T which equals (A_d m_prev^T)^T —
        # the per-batch matrix-vector product expressed row-wise.
        m_new = F.linear(m_prev, self.A_d) + F.linear(u_t, self.B_d)    # (B, d)

        # ── Step 3: hidden state update ──────────────────────────────────
        h_new = torch.tanh(self.W_x(x) + self.W_h(h_prev) + self.W_m(m_new))  # (B, n)

        return h_new, {"h": h_new, "m": m_new}, {}
