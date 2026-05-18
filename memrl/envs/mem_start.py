"""MemoryStartWrapper — forces spawn at corridor entrance in MiniGrid-Memory.

Without this, MiniGrid-Memory places the agent at a random position along
the corridor. If spawned past the hint room the agent never observes the
target object and must guess — making the task unsolvable by any memory
architecture. The wrapper is the Phase 0/1 crutch: it removes the exploration
bottleneck so we can isolate pure memory capacity.

Layout (horizontal MiniGrid-Memory):

    [hint room (x=1)] | [====== corridor ======] | [junction: ball/key]
                      ↑
                 agent spawned here (x=2, facing left toward hint room)

The agent walks left into the hint room, observes the object, turns around,
traverses the corridor, and at the junction chooses the matching side.
"""

import gymnasium as gym
import numpy as np


class MemoryStartWrapper(gym.Wrapper):
    """Reset the agent to the corridor entrance after each env reset.

    Compatible with OneHotPartialObsWrapper → ImgObsWrapper → CastImageFloat32.
    Apply BEFORE those observation wrappers so the forced position is reflected
    in the re-rendered obs.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._force_start_pos()
        obs = self.env.unwrapped.gen_obs()
        return obs, info

    def _force_start_pos(self) -> None:
        env = self.env.unwrapped
        agent_row = env.agent_pos[1]    # corridor row fixed by env layout
        env.agent_pos = np.array([2, agent_row])
        env.agent_dir = 2               # dir=2 = west = facing toward hint room at x=1
