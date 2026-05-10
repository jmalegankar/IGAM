"""GRU cell — Cho et al. 2014, with modern best practices.

Modifications over vanilla Cho 2014:
  - Layer normalization on gate and candidate pre-activations (Ba et al. 2016)
  - Per-block orthogonal init for recurrent weights (Saxe et al. 2013)
  - Xavier init for input weights (Glorot & Bengio 2010)
  - Fused input linear (one matmul for r-, z-, candidate-slice contributions)
  - bias=False on linears; learnable shifts come from LN's affine

Math (per step):
    [gates_x_pre | cand_x_pre]  =  W_x x                          # one linear, split
    gates_pre   =  LN_gates(gates_x_pre + W_h_gates h_{t-1})
    r_pre, z_pre  =  chunk(gates_pre)
    r, z   =  σ(r_pre), σ(z_pre)
    cand_pre  =  LN_cand(cand_x_pre + W_h_cand (r ⊙ h_{t-1}))
    h̃   =  tanh(cand_pre)
    h_t   =  z ⊙ h_{t-1}  +  (1 − z) ⊙ h̃                          # Cho/PyTorch convention

Conventions and variants:
  - Update mix matches Cho 2014 eq. 7 and PyTorch's `nn.GRUCell`:
    z=1 RETAINS h_{t-1}, z=0 ADOPTS the candidate. To bias toward
    long-memory at init (analogous to LSTM forget-bias=+1, see Chrono
    Init / Tallec & Ollivier 2018), shift z toward +1 — most easily
    done by setting `gates_ln.bias[H:2H] = +c` for some c > 0. Not
    enabled by default; uncomment in `reset_parameters` if your task
    has known long-range dependencies.

  - Candidate path follows Cho's original ordering: r gates h_{t-1}
    BEFORE the recurrent linear (`tanh(W_x x + W_h(r ⊙ h_{t-1}))`).
    PyTorch's `nn.GRUCell` uses the alternative `tanh(W_x x + r ⊙ W_h h_{t-1})`
    (r gates the OUTPUT of the recurrent linear). The two diverge once
    parameters are non-zero; comparisons against `nn.GRUCell` therefore
    won't match numerically even with identical init seeds.

Deliberate omissions:
  - No recurrent / variational dropout (Gal & Ghahramani 2016).
  - No zoneout (Krueger et al. 2017).
    Regularization is rarely useful for on-policy PPO and can hurt
    sample efficiency; reintroduce only if a downstream benchmark
    demonstrates a need.

State:        {"h": (B, hidden_size)}
Side outputs: {}
Output:       h_t directly; output_size == hidden_size.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from igam.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask


class GRU(RecurrentCell):
    """GRU cell with layer norm and per-block orthogonal recurrent init."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)
        self.hidden_size = hidden_size
        H = hidden_size

        # Fused input linear: one matmul produces input contributions to
        # all of (r, z, h̃). Output layout: [r-slice | z-slice | cand-slice].
        # bias=False — LN's affine carries the learnable shift.
        self.x_linear = nn.Linear(input_size, 3 * H, bias=False)

        # Recurrent linear for gates only (r, z). The candidate's recurrent
        # contribution can't be precomputed because it consumes (r ⊙ h_{t-1}),
        # which depends on r.
        self.h_gates_linear = nn.Linear(H, 2 * H, bias=False)
        self.h_cand_linear = nn.Linear(H, H, bias=False)

        # Layer norms over the pre-activations (before sigmoid / tanh).
        # Default elementwise_affine=True ⇒ learnable γ (init 1) and β (init 0).
        self.gates_ln = nn.LayerNorm(2 * H)
        self.cand_ln = nn.LayerNorm(H)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        H = self.hidden_size
        # Input weights: Xavier uniform — calibrated for the sigmoid/tanh
        # downstream nonlinearities.
        nn.init.xavier_uniform_(self.x_linear.weight)
        # Recurrent weights: per-gate orthogonal init for each H×H block.
        # (Orthogonal_init on the full (2H, H) tensor doesn't give the right
        # structure since you can't have 2H mutually orthogonal vectors in
        # H-dim space — you'd get a semi-orthogonal matrix where one gate's
        # block is correlated with another's.)
        for i in range(2):
            nn.init.orthogonal_(
                self.h_gates_linear.weight[i * H : (i + 1) * H]
            )
        nn.init.orthogonal_(self.h_cand_linear.weight)
        # LN params: γ=1, β=0 by default — no GRU-specific structured init.
        # To enable long-memory bias on the update gate, set
        #     self.gates_ln.bias[H : 2 * H].fill_(c)  # c > 0
        # which biases z toward 1 (retention). Tallec & Ollivier 2018
        # ("Can Recurrent Neural Networks Warp Time?") recommend
        # c = log(T_max - 1) for tasks with a known max time scale T_max.

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
        H = self.hidden_size

        # Input contributions to all three pre-activations, fused matmul.
        x_proj = self.x_linear(x)                                   # (B, 3H)
        gates_x_pre = x_proj[..., : 2 * H]                          # (B, 2H)
        cand_x_pre = x_proj[..., 2 * H :]                           # (B, H)

        # Gates: r, z. LN over the joint 2H pre-activation, then split + sigmoid.
        gates_pre = self.gates_ln(gates_x_pre + self.h_gates_linear(h_prev))
        r_pre, z_pre = gates_pre.chunk(2, dim=-1)
        r = torch.sigmoid(r_pre)
        z = torch.sigmoid(z_pre)

        # Candidate: depends on (r ⊙ h_prev), so its recurrent linear is
        # computed only after r is known. LN over the H pre-activation.
        cand_pre = self.cand_ln(cand_x_pre + self.h_cand_linear(r * h_prev))
        h_tilde = torch.tanh(cand_pre)

        # Mix: Cho 2014 / PyTorch convention — z=1 retains h_prev, z=0 adopts h̃.
        h_new = z * h_prev + (1.0 - z) * h_tilde

        return h_new, {"h": h_new}, {}
