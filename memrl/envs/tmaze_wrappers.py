"""T-Maze environments — FAITHFUL port of Ni, Ma, Eysenbach & Bacon (2023).

"When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment"
https://arxiv.org/abs/2307.03864   ·   code: github.com/twni2016/Memory-RL

This is a direct port of the reference `envs/tmaze.py` (TMazeBase /
TMazeClassicPassive / TMazeClassicActive) to the gymnasium API, so our
recurrent-PPO results are comparable to the paper. An earlier homegrown
simplification diverged in two fatal ways (cue handed over for free at t=0, and
a fixed −0.1 penalty), which created an "always-right, never-turn" local optimum
under PPO. The faithful version below fixes both.

Geometry (a 2-D grid; the agent moves in {right, up, left, down}):

        cue ∈ {G1=up, G2=down}, readable ONLY at the oracle O
          │
          ▼
        [O]──[S]──...──[J]    J = junction (last corridor cell)
                          │ up   → G1
                          │ down → G2

  - Corridor length L runs from oracle O (x=0) to junction J (x = oracle_len+L).
  - **Passive** (oracle_length=0): start S == O, so the cue is observed at t=0.
    Episode length = L+1. Credit-assignment length = 1 (pure MEMORY test).
  - **Active**  (oracle_length=1): start S is ONE STEP RIGHT of O, so the agent
    must first step LEFT to O to read the cue, then traverse right to J. Episode
    length = L+3. Credit-assignment length = L (MEMORY *and* credit assignment).
    This left-fetch is what kills the "always-right" attractor.

Observation — Box(2,) float32, the paper's `ambiguous_position` encoding:
    at oracle O  : [0, cue]   cue = goal_y (±1) on FIRST visit only, else [0, 0]
    in corridor  : [0, 0]
    at junction/ : [1, y]     y = 0 at J, ±1 at the goal cells
      goal cells
Position in the corridor is deliberately unobservable — the agent knows it has
reached the decision zone only via the [1, ·] flag. The ONLY thing that must be
remembered is the cue.

Actions — Discrete(4): 0=right (+x), 1=up (+y), 2=left (−x), 3=down (−y), matching
the reference `action_mapping = [[1,0],[0,1],[-1,0],[0,-1]]`. Moves into walls
are no-ops (agent stays).

Reward (reference `reward_fn`):
    terminal step (t == episode_length): +goal_reward · 𝟙(y == goal_y)
    every earlier step                 : penalty · 𝟙(x < t − oracle_length)
                                         (+ distract_reward if at the oracle)
  penalty defaults to −1/L so the worst case sums to exactly −1.0 (Ni et al.).
  Returns: optimal policy → +1.0 ; memoryless/guessing → +0.5 ; worst → −1.0.

Env-name routing (see ``memrl/envs/__init__.py``):
    "TMaze-Passive-L50-v0"   Passive, corridor_length = 50
    "TMaze-Active-L50-v0"    Active,  corridor_length = 50
    (corridor_length may also be passed as a kwarg; the name wins if both given.)
"""

from __future__ import annotations

import re
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


class TMazeEnv(gym.Env):
    """Passive / Active T-Maze (Ni et al. 2023), faithful port. See module doc."""

    metadata = {"render_modes": []}

    # reference action_mapping = [[1,0],[0,1],[-1,0],[0,-1]] = right, up, left, down
    _ACTION_MAP = ((1, 0), (0, 1), (-1, 0), (0, -1))

    def __init__(
        self,
        corridor_length: int = 10,
        mode: str = "passive",
        penalty: Optional[float] = None,
        goal_reward: float = 1.0,
        distract_reward: float = 0.0,
    ) -> None:
        super().__init__()
        if corridor_length < 1:
            raise ValueError(f"corridor_length must be >= 1, got {corridor_length}")
        if mode not in ("passive", "active"):
            raise ValueError(f"mode must be 'passive' or 'active', got {mode!r}")

        self.corridor_length = int(corridor_length)
        self.mode = mode
        # Passive: oracle == start (cue visible at t=0). Active: oracle one step
        # left of start (must fetch the cue). Episode length per the reference.
        self.oracle_length = 0 if mode == "passive" else 1
        self.episode_length = (
            self.corridor_length + 2 * self.oracle_length + 1
        )  # passive: L+1, active: L+3
        # Horizon-normalized lag penalty: sum_{t} -1/L = -1 worst case (Ni et al.).
        self.penalty = float(penalty) if penalty is not None else -1.0 / self.corridor_length
        self.goal_reward = float(goal_reward)
        self.distract_reward = float(distract_reward)

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Grid + walls (reference construction). bias pads the index math so wall
        # checks never go out of bounds; corridor on row bias_y, goal cells on
        # rows bias_y±1 at the junction column.
        self.bias_x, self.bias_y = 1, 2
        width = self.oracle_length + self.corridor_length + 1 + 2
        self.tmaze_map = np.zeros((3 + 2, width), dtype=bool)
        self.tmaze_map[self.bias_y, self.bias_x : -self.bias_x] = True  # corridor
        self.tmaze_map[[self.bias_y - 1, self.bias_y + 1], -self.bias_x - 1] = True  # goals

        # Episode state (set in reset()).
        self.x = self.oracle_length
        self.y = 0
        self.goal_y = 1
        self.t = 0
        self.oracle_visited = False

    # --- gym interface -----------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.x = self.oracle_length
        self.y = 0
        self.goal_y = 1 if self.np_random.random() < 0.5 else -1
        self.t = 0
        self.oracle_visited = False
        return self._obs(), {"goal_y": self.goal_y}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        self.t += 1

        move_x, move_y = self._ACTION_MAP[action]
        if self.tmaze_map[self.bias_y + self.y + move_y, self.bias_x + self.x + move_x]:
            self.x += move_x
            self.y += move_y

        terminated = self.t >= self.episode_length   # finite-horizon terminal
        reward = self._reward(terminated)

        info: dict[str, Any] = {"goal_y": self.goal_y}
        if terminated:
            info["success"] = bool(self.y == self.goal_y)
        return self._obs(), float(reward), terminated, False, info

    # --- internals ---------------------------------------------------------

    def _obs(self) -> np.ndarray:
        """Ambiguous-position encoding (reference `position_encoding`)."""
        if self.x == 0:
            if not self.oracle_visited:
                exposure = self.goal_y          # cue shown on first oracle visit
                self.oracle_visited = True
            else:
                exposure = 0
            val = (0.0, float(exposure))
        elif self.x < self.oracle_length + self.corridor_length:
            val = (0.0, 0.0)                     # corridor: null
        else:
            val = (1.0, float(self.y))           # junction / goal cells
        return np.array(val, dtype=np.float32)

    def _reward(self, terminated: bool) -> float:
        """Reference `reward_fn`: terminal goal bonus, else lag penalty."""
        if terminated:
            return float(self.y == self.goal_y) * self.goal_reward
        rew = float(self.x < self.t - self.oracle_length) * self.penalty
        if self.x == 0:
            rew += self.distract_reward
        return rew


# ── name parsing + factory ──────────────────────────────────────────────────

_LENGTH_RE = re.compile(r"^L(\d+)$", re.IGNORECASE)


def _parse_tmaze_name(env_name: str) -> tuple[str, Optional[int]]:
    """Parse "TMaze-{Passive|Active}[-L<int>]-v0" → (mode, corridor_length|None)."""
    parts = env_name.split("-")
    if not parts or parts[0] != "TMaze":
        raise ValueError(f"not a TMaze env id: {env_name!r}")
    mode: Optional[str] = None
    length: Optional[int] = None
    for tok in parts[1:]:
        low = tok.lower()
        if low == "passive":
            mode = "passive"
        elif low == "active":
            mode = "active"
        elif low in ("v0", ""):
            continue
        else:
            m = _LENGTH_RE.match(tok)
            if m:
                length = int(m.group(1))
            else:
                raise ValueError(
                    f"unrecognized token {tok!r} in TMaze id {env_name!r}. "
                    f"Expected Passive/Active, optional L<int>, v0."
                )
    if mode is None:
        raise ValueError(
            f"TMaze id {env_name!r} must specify 'Passive' or 'Active' "
            f"(e.g. 'TMaze-Active-L50-v0')."
        )
    return mode, length


def make_tmaze_vec_env(
    env_name: str,
    n_envs: int = 8,
    seed: int = 0,
    corridor_length: Optional[int] = None,
    penalty: Optional[float] = None,
    **env_kwargs: Any,
) -> VecEnv:
    """Build a vectorized Passive/Active T-Maze (faithful Ni et al. port).

    Mode + corridor length come from `env_name` (e.g. "TMaze-Active-L50-v0");
    explicit `corridor_length` overrides the name. `penalty` defaults to −1/L.
    DummyVecEnv because each step is sub-microsecond (IPC would dominate).
    """
    mode, length_from_name = _parse_tmaze_name(env_name)
    L = corridor_length if corridor_length is not None else (
        length_from_name if length_from_name is not None else 10
    )

    def _make_one(rank: int):
        def _init():
            env = TMazeEnv(
                corridor_length=L, mode=mode, penalty=penalty, **env_kwargs
            )
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            return Monitor(env)
        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])
