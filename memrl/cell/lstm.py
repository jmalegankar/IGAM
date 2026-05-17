"""LSTM cell — Hochreiter & Schmidhuber (1997), with modern best practices.

Modifications over vanilla LSTM:
  - Layer normalization on gate pre-activations and (optionally) on the
    cell state before the output tanh (Ba et al. 2016)
  - Per-block orthogonal init for recurrent weights (Saxe et al. 2013)
  - Xavier init for input weights (Glorot & Bengio 2010)
  - Forget-gate bias initialized to +1 (Jozefowicz, Zaremba, Sutskever 2015)
  - Fused input and recurrent linears: one matmul each, both 4H wide
  - bias=False on linears; learnable shifts come from LN's affine

Math (per step):
    gates_pre  =  LN_gates(W_x x + W_h h_{t-1})         # one fused matmul each, 4H output
    i_pre, f_pre, g_pre, o_pre  =  chunk(gates_pre, 4)
    i, f, o  =  σ(i_pre), σ(f_pre), σ(o_pre)            # +1 forget bias lives in LN_gates.bias[H:2H]
    g  =  tanh(g_pre)
    c_t  =  f ⊙ c_{t-1}  +  i ⊙ g                       # cell state — the constant error carousel
    h_t  =  o ⊙ tanh(LN_c(c_t))                          # LN'd cell state through output gate

Gate layout: [i, f, g, o], matching PyTorch's nn.LSTMCell convention.
The forget-gate slice is therefore [H : 2H] of every 4H tensor.

Forget bias nuance:
  Standard Jozefowicz 2015 sets the forget gate's *linear* bias to +1.
  We have bias=False on linears (LN's affine carries the shifts), so the
  +1 lives on `gates_ln.bias[H:2H]` instead. Since LayerNorm computes
  `γ * normalize(x) + β`, the +1 is applied POST-normalization. At init
  this gives f_pre ≈ N(0,1) + 1 ⇒ σ(f_pre) ≈ σ(1) ≈ 0.73 on average —
  matching Jozefowicz's effect at init. Dynamics differ slightly during
  training: LN clips pre-activation magnitudes to ~unit variance, so the
  +1 shift is a fixed push in normalized space rather than a learnable
  offset on the raw linear pre-activation. Functionally equivalent for
  the intended purpose (biasing toward "remember" early in training).

Cell-state LN tradeoff:
  `cell_norm=True` (default, Ba 2016) normalizes c_t before the output
  tanh. This stabilizes h_t but strips the cell state's absolute magnitude
  from the output path. The recurrence c_t = f ⊙ c_{t-1} + i ⊙ g is
  itself UN-normalized, so the gradient highway (∂c_t/∂c_{t-1} = diag(f))
  is preserved. Set `cell_norm=False` for tasks that require magnitude
  tracking in the output (counters, sustained-activity).

Deliberate omissions:
  - No recurrent / variational dropout (Gal & Ghahramani 2016).
  - No zoneout (Krueger et al. 2017).
  - No peephole connections (Gers & Schmidhuber 2000).
    Regularization is rarely useful for on-policy PPO; peepholes are
    largely subsumed by modern gating. Reintroduce only if a downstream
    benchmark demonstrates a need.

State:        {"h": (B, hidden_size), "c": (B, hidden_size)}
Side outputs: {}
Output:       h_t directly; output_size == hidden_size.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from memrl.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask


class LSTM(RecurrentCell):
    """LSTM cell with layer norm, per-block orthogonal recurrent init,
    and forget-bias=+1 initialization."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        cell_norm: bool = True,
    ) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)
        self.hidden_size = hidden_size
        self.cell_norm_enabled = cell_norm
        H = hidden_size

        # Fused linears: each maps to 4H, covering all of (i, f, g, o) in
        # layout [i | f | g | o]. bias=False — the learnable shifts (including
        # the +1 forget-bias init) live in LN_gates.bias.
        self.x_linear = nn.Linear(input_size, 4 * H, bias=False)
        self.h_linear = nn.Linear(H, 4 * H, bias=False)

        # Joint LN over the 4H gate pre-activations (one normalization
        # for all four gates). Alternative: per-gate LN (one nn.LayerNorm(H)
        # per gate); functionally similar, marginally more parameters.
        self.gates_ln = nn.LayerNorm(4 * H)

        # Optional cell-state LN before the output tanh. nn.Identity when
        # disabled, so the forward path stays branch-free.
        self.cell_ln: nn.Module = nn.LayerNorm(H) if cell_norm else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        H = self.hidden_size
        # Input weights: Xavier uniform — calibrated for the sigmoid/tanh
        # downstream nonlinearities.
        nn.init.xavier_uniform_(self.x_linear.weight)
        # Recurrent weights: per-gate orthogonal init. Each H×H block of
        # the (4H, H) weight gets its own orthogonal matrix, so each gate's
        # recurrence is norm-preserving in isolation. (orthogonal_ on the
        # full tensor would give column-orthogonality of the joint 4H
        # output, not per-gate orthogonality.)
        for i in range(4):
            nn.init.orthogonal_(
                self.h_linear.weight[i * H : (i + 1) * H]
            )
        # Forget-gate bias = +1 (Jozefowicz, Zaremba, Sutskever 2015).
        # Lives in LN_gates.bias, slice [H:2H], because layout is [i, f, g, o].
        # At init this gives σ(f_pre) ≈ σ(N(1, 1)) ≈ 0.73 on average,
        # biasing the cell toward "remember" rather than "forget."
        # See module docstring for the LN-vs-linear-bias nuance.
        with torch.no_grad():
            self.gates_ln.bias[H : 2 * H].fill_(1.0)
        # All other LN params left at default (γ=1, β=0).

    # --- interface ---------------------------------------------------------

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        device = self._resolve_device(device)
        H = self.hidden_size
        return {
            "h": torch.zeros(batch_size, H, device=device, dtype=dtype),
            "c": torch.zeros(batch_size, H, device=device, dtype=dtype),
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
        c_prev = state["c"]

        # Fused gate pre-activations, LN'd jointly over the 4H dimension.
        # The +1 forget bias is added inside the LN's affine step, so f_pre
        # already carries the shift before chunk + sigmoid.
        gates_pre = self.gates_ln(self.x_linear(x) + self.h_linear(h_prev))
        i_pre, f_pre, g_pre, o_pre = gates_pre.chunk(4, dim=-1)

        i = torch.sigmoid(i_pre)
        f = torch.sigmoid(f_pre)
        g = torch.tanh(g_pre)
        o = torch.sigmoid(o_pre)

        # Cell-state update — gradient highway: ∂c_t/∂c_{t-1} = diag(f),
        # no nonlinearity in this path. f initialized near 1 ⇒ near-identity
        # Jacobian at the start of training.
        c_new = f * c_prev + i * g

        # Output: optionally-LN'd cell state passed through tanh, then gated
        # by o. cell_ln is nn.Identity when disabled — no branch in the path.
        h_new = o * torch.tanh(self.cell_ln(c_new))

        return h_new, {"h": h_new, "c": c_new}, {}
