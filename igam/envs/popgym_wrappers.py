"""POPGym environment factory.

POPGym is gym-registered (verified at session start) — every env is
accessible as `gym.make("popgym-<EnvName>-v0")`. For Phase A Tier 1 we
start with `popgym-RepeatPreviousEasy-v0`: Discrete(4) obs, Discrete(4)
action, ~52-step episodes, K=4 lookback.

This module is intentionally thin — it just wraps `gym.make` in a
`DummyVecEnv` with per-env seeding. Subprocess-based parallelism
(`SubprocVecEnv`) is the natural upgrade if Apple-M3 wall-clock becomes
the bottleneck.
"""

from __future__ import annotations

import gymnasium as gym
import popgym  # noqa: F401 — registers POPGym envs with gymnasium
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


def make_popgym_vec_env(
    env_name: str,
    n_envs: int = 8,
    seed: int = 0,
) -> VecEnv:
    """Build a vectorized POPGym environment.

    Each env is wrapped in `Monitor` so SB3 picks up episode rewards and
    lengths into `ep_info_buffer` — this is what populates `rollout/ep_rew_mean`
    and `rollout/ep_len_mean` in the tensorboard logs. Without Monitor those
    fields silently stay empty.

    Args:
        env_name: full gym id, e.g. "popgym-RepeatPreviousEasy-v0".
        n_envs:   number of parallel environments.
        seed:     base seed; env i is seeded with `seed + i`.

    Returns:
        Vectorized environment ready to pass to IGAMPPO.
    """
    def _make_one(rank: int):
        def _init():
            env = gym.make(env_name)
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            return Monitor(env)
        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])
