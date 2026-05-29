"""Memoryless cell — the "no memory" control for the exploration study.

This is NOT a recurrent cell: it carries NO state across timesteps. The output
at step t depends ONLY on the current input x_t (the encoder features), never on
history. It is the architectural realization of a feedforward policy inside the
recurrent-PPO pipeline, used as the "no cell" arm of the memory × exploration
study: it answers "can exploration alone, with no memory, make progress on a
memory task like S13?" (Expected: no — S13 requires recalling the start cue at
the junction, which a memoryless policy structurally cannot do. It is therefore
a NEGATIVE CONTROL, not a performance contender.)

Why a learnable Linear rather than nn.Identity:
  * Keeps the cell well-defined when input_size != output_size (config changes).
  * Gives the "feedforward trunk" one learnable layer so the comparison is a
    fair feedforward policy (encoder → trunk → heads), not encoder → heads with
    the cell slot empty.

State:        {} — empty. The rollout buffer stores nothing; forward_sequence
              receives {} and returns {} (verified safe end-to-end).
Side outputs: {}
Output:       Linear(x); output_size == hidden_size, matching the other cells so
              the policy heads are dimensionally identical across the lineup.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from memrl.cell.base import RecurrentCell, SideOutputs, State


class Memoryless(RecurrentCell):
    """Stateless feedforward "cell": y = Linear(x), no recurrence, empty state."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)
        self.hidden_size = hidden_size
        self.proj = nn.Linear(input_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Orthogonal weight (relu-ish downstream in the heads), zero bias —
        # matches the init convention used elsewhere in the policy stack.
        nn.init.orthogonal_(self.proj.weight, gain=2.0 ** 0.5)
        nn.init.zeros_(self.proj.bias)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        # No recurrent state. Empty dict => buffer allocates no state arrays.
        return {}

    def reset_state(self, state: State, mask: Tensor) -> State:
        # Nothing to reset (no state). Override avoids the base impl's
        # next(iter(state.values())) on an empty dict.
        return state

    def step(
        self,
        x: Tensor,
        state: State,
        episode_start: Optional[Tensor] = None,
    ) -> tuple[Tensor, State, SideOutputs]:
        # episode_start is irrelevant: there is no state to reset.
        return self.proj(x), {}, {}
