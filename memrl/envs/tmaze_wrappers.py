"""T-Maze environments — Ni, Ma, Eysenbach & Bacon 2023.

"When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment"
https://arxiv.org/abs/2307.03864   (reference code: github.com/twni2016/Memory-RL)

The T-Maze is the cleanest known probe for *isolating* the two axes that a
recurrent RL agent has to get right on a long-horizon task:

  1. **Memory horizon** — how far back a useful observation lies. Here a single
     binary cue is shown ONLY at t=0 and must be recalled at the junction,
     `corridor_length` steps later. Memory horizon = L for both variants.

  2. **Credit-assignment horizon** — how far the reward-causing action lies from
     the reward. This is exactly what Passive vs Active dials:

       Passive : the agent is carried down the corridor automatically (corridor
                 actions are ignored). Only the final turn affects reward, so the
                 credit-assignment horizon is **1**. Pure memory test.

       Active  : the agent must itself choose "right" at every corridor step to
                 advance, then turn. Every one of the L actions is on the path to
                 reward, so the credit-assignment horizon is **L**. Memory AND
                 credit assignment, at the same length.

Ni et al.'s headline finding: memory length and credit-assignment length are
*separately* hard. A cell that aces Passive-T-Maze-L can still fail Active at
the same L (and vice-versa for some architectures). That decoupling is exactly
what the "memory × exploration" study needs — Passive isolates the memory cell's
raw capacity; Active stresses the long-horizon credit propagation that an
exploration bonus is meant to help with.

Layout (corridor_length L; junction at x = L-1):

        cue ∈ {up, down} shown only at t=0
          │
          ▼
        [x=0] ─ [x=1] ─ ... ─ [x=L-2] ─ [x=L-1]  ← junction
        start                              │ turn up   → +1 if goal is up
                                           │ turn down → +1 if goal is down

Observation — Box(3,) float32, deliberately minimal (matches the paper's
"position is always observable, only the cue must be remembered" design):

    obs = [ x_norm , at_junction , cue ]
        x_norm      = x / (L-1)        ∈ [0, 1]   — where am I in the corridor
        at_junction = 1.0 if x == L-1  ∈ {0, 1}   — "decide now" flag
        cue         = +1 (goal up) / -1 (goal down), shown for the first
                      `cue_steps` steps (default 1 ⇒ only at t=0), else 0.0

Exposing position is intentional: it removes any *incidental* memory burden
(the agent never has to count steps to know it reached the junction), so the
ONLY thing that must be remembered is the cue. That keeps "memory horizon"
clean at exactly L.

Actions — Discrete(4): 0=up, 1=down, 2=left, 3=right. The same 4-way space is
used for both variants so architecture comparisons are apples-to-apples; in
Passive the corridor actions are simply ignored.

Reward (sparse, terminal):
    correct turn at junction → +goal_reward (default +1.0), episode terminates
    wrong turn at junction   → +wrong_penalty (default 0.0), episode terminates
    every step               → +step_penalty (default 0.0)
    Active timeout w/o turn  → truncates, no goal reward

Episode length defaults to `corridor_length` (L): the *tight* Ni et al. budget
where the optimal trajectory (L-1 advances + 1 turn for Active; auto-advance +
turn for Passive) exactly fills the horizon. Pass `episode_length` to add slack
for early "messing around" runs.

Oracle variant (sanity baseline): set `cue_steps` large (or use the "-Oracle"
name token) so the cue is visible every step. Memory is then unnecessary, so a
*failure* on Oracle isolates an RL/credit problem from a memory-capacity one —
the standard Ni et al. upper-bound control.

Env-name routing (see ``memrl/envs/__init__.py``):
    "TMaze-Passive-v0"          Passive, L from `corridor_length` kwarg (def 10)
    "TMaze-Active-v0"           Active,  L from `corridor_length` kwarg (def 10)
    "TMaze-Passive-L20-v0"      Passive, L = 20  (length baked into the id)
    "TMaze-Active-L50-v0"       Active,  L = 50
    "TMaze-Passive-Oracle-L20-v0"  Passive oracle control, L = 20

State:        x (position), goal (±1), t (step count) — all scalar per env.
Output obs:   (n_envs, 3) float32 after vectorization.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

# Action ids (cardinal moves). Only up/down matter at the junction; in Active,
# right advances and left retreats along the corridor.
_UP, _DOWN, _LEFT, _RIGHT = 0, 1, 2, 3


class TMazeEnv(gym.Env):
    """Passive / Active T-Maze (Ni et al. 2023). See module docstring."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        corridor_length: int = 10,
        mode: str = "passive",
        episode_length: Optional[int] = None,
        cue_steps: int = 1,
        goal_reward: float = 1.0,
        wrong_penalty: float = 0.0,
        step_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        if corridor_length < 2:
            raise ValueError(
                f"corridor_length must be >= 2 (need a start and a distinct "
                f"junction), got {corridor_length}"
            )
        if mode not in ("passive", "active"):
            raise ValueError(f"mode must be 'passive' or 'active', got {mode!r}")

        self.L = int(corridor_length)
        self.mode = mode
        # Tight Ni et al. budget by default: exactly enough steps for the
        # optimal trajectory (L-1 advances + 1 turn => L step() calls).
        self.episode_length = int(episode_length) if episode_length is not None else self.L
        self.cue_steps = int(cue_steps)
        self.goal_reward = float(goal_reward)
        self.wrong_penalty = float(wrong_penalty)
        self.step_penalty = float(step_penalty)

        self.action_space = gym.spaces.Discrete(4)
        # obs = [x_norm in [0,1], at_junction in {0,1}, cue in {-1,0,+1}]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

        # Episode state (set in reset()).
        self.x = 0
        self.goal = 1   # +1 = up, -1 = down
        self.t = 0
        self._last_success: Optional[bool] = None

    # --- gym interface -----------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.x = 0
        # self.np_random is seeded by gym.Env.reset(seed=...).
        self.goal = 1 if self.np_random.random() < 0.5 else -1
        self.t = 0
        self._last_success = None
        return self._obs(), {"goal_up": self.goal == 1}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        # Increment first so the NEXT observation (returned below) hides the cue
        # once t >= cue_steps. reset() leaves t=0, so its obs shows the cue.
        self.t += 1

        reward = self.step_penalty
        terminated = False
        truncated = False
        at_junction = self.x == self.L - 1

        if self.mode == "passive":
            if at_junction:
                # The corridor was auto-traversed; this final action is the turn.
                reward += self._evaluate_turn(action)
                terminated = True
            else:
                # Carried one cell forward regardless of action (credit
                # assignment horizon = 1: only the eventual turn matters).
                self.x += 1
        else:  # active
            if action == _RIGHT:
                self.x = min(self.x + 1, self.L - 1)
            elif action == _LEFT:
                self.x = max(self.x - 1, 0)
            elif action in (_UP, _DOWN):
                if at_junction:
                    reward += self._evaluate_turn(action)
                    terminated = True
                # else: up/down off the junction is a wasted step (no-op move).

        if not terminated and self.t >= self.episode_length:
            truncated = True

        info: dict[str, Any] = {"goal_up": self.goal == 1}
        if terminated:
            info["success"] = bool(self._last_success)
        return self._obs(), float(reward), terminated, truncated, info

    # --- internals ---------------------------------------------------------

    def _obs(self) -> np.ndarray:
        x_norm = self.x / (self.L - 1)
        at_junction = 1.0 if self.x == self.L - 1 else 0.0
        cue = float(self.goal) if self.t < self.cue_steps else 0.0
        return np.array([x_norm, at_junction, cue], dtype=np.float32)

    def _evaluate_turn(self, action: int) -> float:
        """Reward for an up/down choice at the junction; records success."""
        if action == _UP:
            correct = self.goal == 1
        elif action == _DOWN:
            correct = self.goal == -1
        else:
            # left/right at the junction is never a correct turn.
            correct = False
        self._last_success = correct
        return self.goal_reward if correct else self.wrong_penalty


# ── name parsing + factory ──────────────────────────────────────────────────

_LENGTH_RE = re.compile(r"^L(\d+)$", re.IGNORECASE)


def _parse_tmaze_name(env_name: str) -> tuple[str, Optional[int], bool]:
    """Parse a "TMaze-..." id into (mode, corridor_length|None, oracle).

    Recognized tokens (order-insensitive, between the "TMaze" prefix and the
    trailing "v0"): "Passive"/"Active" (required), "Oracle" (optional),
    "L<int>" (optional corridor length).

    Examples:
        "TMaze-Passive-v0"            -> ("passive", None, False)
        "TMaze-Active-L50-v0"         -> ("active", 50, False)
        "TMaze-Passive-Oracle-L20-v0"-> ("passive", 20, True)
    """
    parts = env_name.split("-")
    if not parts or parts[0] != "TMaze":
        raise ValueError(f"not a TMaze env id: {env_name!r}")

    mode: Optional[str] = None
    length: Optional[int] = None
    oracle = False
    for tok in parts[1:]:
        low = tok.lower()
        if low == "passive":
            mode = "passive"
        elif low == "active":
            mode = "active"
        elif low == "oracle":
            oracle = True
        elif low in ("v0", ""):
            continue
        else:
            m = _LENGTH_RE.match(tok)
            if m:
                length = int(m.group(1))
            else:
                raise ValueError(
                    f"unrecognized token {tok!r} in TMaze id {env_name!r}. "
                    f"Expected Passive/Active, optional Oracle, optional L<int>, v0."
                )
    if mode is None:
        raise ValueError(
            f"TMaze id {env_name!r} must specify 'Passive' or 'Active' "
            f"(e.g. 'TMaze-Passive-L20-v0')."
        )
    return mode, length, oracle


def make_tmaze_vec_env(
    env_name: str,
    n_envs: int = 8,
    seed: int = 0,
    corridor_length: Optional[int] = None,
    episode_length: Optional[int] = None,
    cue_steps: Optional[int] = None,
    **env_kwargs: Any,
) -> VecEnv:
    """Build a vectorized Passive/Active T-Maze.

    The corridor length / mode / oracle flag come from `env_name` (see
    `_parse_tmaze_name`); explicit kwargs override the name where both are
    given. Each sub-env is wrapped in `Monitor` for SB3 episode bookkeeping.

    Args:
        env_name:        e.g. "TMaze-Passive-v0", "TMaze-Active-L50-v0".
        n_envs:          number of parallel envs.
        seed:            base seed; env i is seeded with `seed + i`.
        corridor_length: overrides the L parsed from the name (or supplies it
                         when the name carries no "L<int>" token). Default 10.
        episode_length:  max steps before truncation (default = corridor_length).
        cue_steps:       #initial steps the cue is visible (default 1). The
                         "-Oracle" name token forces this to span the episode.
        **env_kwargs:    forwarded to TMazeEnv (goal_reward, step_penalty, ...).

    Returns:
        DummyVecEnv (T-Maze steps are sub-microsecond; IPC would dominate).
    """
    mode, length_from_name, oracle = _parse_tmaze_name(env_name)
    L = corridor_length if corridor_length is not None else (
        length_from_name if length_from_name is not None else 10
    )

    if cue_steps is not None:
        resolved_cue_steps = cue_steps
    elif oracle:
        # Cue visible for the whole episode ⇒ memory is unnecessary (control).
        resolved_cue_steps = (episode_length if episode_length is not None else L) + 1
    else:
        resolved_cue_steps = 1

    def _make_one(rank: int):
        def _init():
            env = TMazeEnv(
                corridor_length=L,
                mode=mode,
                episode_length=episode_length,
                cue_steps=resolved_cue_steps,
                **env_kwargs,
            )
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            return Monitor(env)
        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])
