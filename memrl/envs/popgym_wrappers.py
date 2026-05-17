"""POPGym environment factory.

POPGym is gym-registered (verified at session start) — every env is
accessible as `gym.make("popgym-<EnvName>-v0")`.

Wrappers handle three POPGym obs/action quirks before handing the env to
the trainer:

  - **Tuple-of-Discrete obs** (Autoencode, ...) → `FlattenTupleDiscrete`
    collapses to single `Discrete` via lossless mixed-base encoding.

  - **MultiDiscrete obs** (CountRecall, ...) → `FlattenMultiDiscrete`
    does the same trick on a MultiDiscrete space, again losslessly.

  - **Tuple-obs phase flag** (Autoencode `(phase, card)`) → `ExposePhaseInInfo`
    surfaces `obs[0]` in `info["phase"]` BEFORE flattening, so downstream
    code (eval hooks, callbacks) can group metrics by phase without
    re-decoding the flattened integer.

MultiDiscrete *action* spaces (Battleship) are forwarded as-is —
SB3 PPO handles them natively via MultiCategoricalDistribution.

This module is intentionally thin — it just wraps `gym.make` in a
`DummyVecEnv` with per-env seeding. Subprocess-based parallelism
(`SubprocVecEnv`) is the natural upgrade if wall-clock becomes the
bottleneck.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import popgym  # noqa: F401 — registers POPGym envs with gymnasium
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


def _mixed_base_multipliers(sizes: list[int]) -> tuple[list[int], int]:
    """Compute mixed-base multipliers and total. For sizes [a, b, c]:
    multipliers = [b*c, c, 1]; total = a*b*c. Used by both flatten wrappers.
    """
    running = 1
    mults: list[int] = []
    for n in reversed(sizes):
        mults.append(running)
        running *= int(n)
    return list(reversed(mults)), running


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
        self._multipliers, total = _mixed_base_multipliers(sizes)
        self._sizes = sizes
        self.observation_space = gym.spaces.Discrete(total)

    def observation(self, obs: tuple[int, ...]) -> int:
        return int(sum(int(o) * m for o, m in zip(obs, self._multipliers)))


class FlattenMultiDiscrete(gym.ObservationWrapper):
    """Collapse a MultiDiscrete observation space into a single Discrete.

    Same lossless mixed-base trick as `FlattenTupleDiscrete`, but for
    `MultiDiscrete` spaces. Several POPGym envs (CountRecall, ...) expose
    obs as `MultiDiscrete([a, b, ...])` — semantically a fixed-length array
    of integer indices. SB3's RolloutBuffer prefers a single Discrete; this
    wrapper folds the MultiDiscrete into one int losslessly so the existing
    Embedding-based FlatEncoder works without changes.

    Example: CountRecall's `MultiDiscrete([4, 4])` (dealt_index, queried_index)
    becomes `Discrete(16)` with mapping `(dealt, queried) -> dealt*4 + queried`.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.MultiDiscrete):
            raise TypeError(
                f"FlattenMultiDiscrete expects MultiDiscrete obs, got "
                f"{type(env.observation_space).__name__}"
            )
        nvec = env.observation_space.nvec
        if nvec.ndim != 1:
            raise TypeError(
                f"FlattenMultiDiscrete expects a 1-D nvec; got shape {nvec.shape}"
            )
        sizes = [int(n) for n in nvec]
        self._multipliers, total = _mixed_base_multipliers(sizes)
        self._sizes = sizes
        self.observation_space = gym.spaces.Discrete(total)

    def observation(self, obs: Any) -> int:
        # obs is a numpy array of shape (len(nvec),) — iterate the leading dim.
        return int(sum(int(obs[i]) * m for i, m in enumerate(self._multipliers)))


class ExposePhaseInInfo(gym.Wrapper):
    """For two-phase envs like Autoencode, surface the phase flag in info.

    Must be applied BEFORE `FlattenTupleDiscrete` (or any obs-modifying
    wrapper that collapses the phase index into the flattened obs).
    On every reset/step, sets `info["phase"] = int(obs[phase_index])`.
    Downstream callbacks can group metrics by phase without re-decoding.

    For Autoencode: `obs = (mode_value, card_suit)` → `info["phase"]` is
    0 during WATCH, 1 during PLAY.
    """

    def __init__(self, env: gym.Env, phase_index: int = 0) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Tuple):
            raise TypeError(
                f"ExposePhaseInInfo expects Tuple obs, got "
                f"{type(env.observation_space).__name__}"
            )
        if phase_index < 0 or phase_index >= len(env.observation_space.spaces):
            raise IndexError(
                f"phase_index {phase_index} out of range for Tuple of length "
                f"{len(env.observation_space.spaces)}"
            )
        self._phase_index = phase_index

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["phase"] = int(obs[self._phase_index])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["phase"] = int(obs[self._phase_index])
        return obs, reward, terminated, truncated, info


# Env-id prefixes whose obs[0] is a phase flag worth exposing.
# Currently only Autoencode; add others (e.g. memoroid-style two-phase) here.
_PHASE_ENV_PREFIXES = ("popgym-Autoencode",)


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
            # Expose phase flag in info BEFORE obs-flattening (Autoencode etc.).
            if any(env_name.startswith(p) for p in _PHASE_ENV_PREFIXES):
                env = ExposePhaseInInfo(env, phase_index=0)
            # Auto-flatten factorized discrete obs spaces.
            if isinstance(env.observation_space, gym.spaces.Tuple):
                env = FlattenTupleDiscrete(env)
            elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
                env = FlattenMultiDiscrete(env)
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            return Monitor(env)
        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])
