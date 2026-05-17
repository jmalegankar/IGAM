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

from typing import Any

import gymnasium as gym
import popgym  # noqa: F401 — registers POPGym envs with gymnasium
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


class FlattenTupleDiscrete(gym.ObservationWrapper):
    """Collapse a Tuple-of-Discrete observation space into a single Discrete.

    Several POPGym envs (Autoencode, ...) expose obs as `Tuple(Discrete(a), Discrete(b), ...)`
    where each element encodes one factor (phase, token, ...). SB3's
    RolloutBuffer doesn't natively support Tuple obs; this wrapper flattens
    the tuple into a single integer by treating the tuple as a positional
    number with mixed bases.

    The flattening is lossless: each (a, b, ...) combination gets a unique
    integer id, and `nn.Embedding(total, hidden_dim)` in `FlatEncoder` learns
    a separate embedding for each combination. No structural information is
    lost — the cell sees as much signal as it would from a Dict/Tuple encoder.

    Example: Autoencode's `Tuple(Discrete(2), Discrete(4))` becomes
    `Discrete(8)` with mapping `(phase, token) -> phase * 4 + token`.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Tuple):
            raise TypeError(
                f"FlattenTupleDiscrete expects Tuple obs, got "
                f"{type(env.observation_space).__name__}"
            )
        for i, s in enumerate(env.observation_space.spaces):
            if not isinstance(s, gym.spaces.Discrete):
                raise TypeError(
                    f"FlattenTupleDiscrete expects Tuple-of-Discrete; element "
                    f"{i} is {type(s).__name__}"
                )

        sizes = [s.n for s in env.observation_space.spaces]
        # Mixed-base multipliers: for sizes [2, 4], multipliers = [4, 1] so
        # that obs (phase, token) -> phase*4 + token.
        running = 1
        mults = []
        for n in reversed(sizes):
            mults.append(running)
            running *= n
        self._multipliers = list(reversed(mults))
        self._sizes = sizes
        self.observation_space = gym.spaces.Discrete(running)

    def observation(self, obs: tuple[int, ...]) -> int:
        return int(sum(int(o) * m for o, m in zip(obs, self._multipliers)))


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
        Vectorized environment ready to pass to MemPPO.
    """
    def _make_one(rank: int):
        def _init():
            env = gym.make(env_name)
            # Auto-flatten Tuple-of-Discrete obs (Autoencode, ConcentrationEasy
            # uses MultiDiscrete instead and would need a different wrapper).
            if isinstance(env.observation_space, gym.spaces.Tuple):
                env = FlattenTupleDiscrete(env)
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            return Monitor(env)
        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])
