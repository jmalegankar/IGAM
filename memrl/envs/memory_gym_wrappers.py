"""memory-gym (a.k.a. endless-memory-gym) environment factory.

memory-gym (https://github.com/MarcoMeter/endless-memory-gym) is a suite of
pixel-based memory benchmarks: MortarMayhem, MysteryPath, SearingSpotlights,
and their Endless-* variants. Agent observations are RGB images; the agent
must selectively remember stimuli that appear briefly at episode start and
act on them many steps later — the same selective-write regime that
MiniGrid-Memory exercises, but with longer horizons and richer visuals.

The whole suite installs as ``pip install memory-gym``. Importing the
package registers all gymnasium env IDs.

Registered IDs (memory-gym 1.x):
    MortarMayhem-v0,         MortarMayhem-Grid-v0
    MortarMayhemB-v0,        MortarMayhemB-Grid-v0
    MysteryPath-v0,          MysteryPath-Grid-v0
    SearingSpotlights-v0
    Endless-MortarMayhem-v0
    Endless-MysteryPath-v0
    Endless-SearingSpotlights-v0

Obs handling:
    Raw obs is uint8 ``(H, W, 3)``. We cast to float32 and divide by 255 so
    values lie in [0, 1]. FlatEncoder will flatten H*W*3 — for the default
    84x84 resolution that's 21168 dims (workable but wasteful for an MLP).
    When a CNN encoder lands, swap NormalizeImageObs for a CHW transpose
    that leaves normalization to the conv stack.

Reset options:
    memory-gym envs accept difficulty/scale kwargs via
    ``reset(options=...)``. We accept a ``reset_options`` dict at
    construction time and replay it on every reset via StickyResetOptions.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import memory_gym  # noqa: F401 — registers memory-gym envs with gymnasium
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


MEMORY_GYM_ENV_IDS: frozenset[str] = frozenset({
    "MortarMayhem-v0",
    "MortarMayhem-Grid-v0",
    "MortarMayhemB-v0",
    "MortarMayhemB-Grid-v0",
    "MysteryPath-v0",
    "MysteryPath-Grid-v0",
    "SearingSpotlights-v0",
    "Endless-MortarMayhem-v0",
    "Endless-MysteryPath-v0",
    "Endless-SearingSpotlights-v0",
})


class NormalizeImageObs(gym.ObservationWrapper):
    """Cast a uint8 image obs to float32 in [0, 1] (divide by 255)."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        old = env.observation_space
        if not (isinstance(old, gym.spaces.Box) and old.dtype == np.uint8):
            raise TypeError(
                f"NormalizeImageObs expects uint8 Box obs, got {old}"
            )
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=old.shape, dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32) / 255.0


class SuccessInfoAlias(gym.Wrapper):
    """Mirror memory-gym's terminal ``info["success"]`` to ``info["is_success"]``.

    SB3's EvalCallback auto-logs ``eval/success_rate`` from the ``is_success``
    key (callbacks.py: `info.get("is_success")`); memory-gym uses ``success``.
    With the alias, the deterministic eval reports goal-clearing directly —
    disentangled from shaped reward (penalties) on the dense arms.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "success" in info:
            info["is_success"] = bool(info["success"])
        return obs, reward, terminated, truncated, info


class StickyResetOptions(gym.Wrapper):
    """Replay a fixed ``options`` dict on every ``reset(...)``.

    SB3's VecEnv only threads ``seed`` through auto-reset — env-specific
    difficulty params would otherwise be lost after the first episode.
    """

    def __init__(self, env: gym.Env, options: dict[str, Any]) -> None:
        super().__init__(env)
        self._sticky = dict(options)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        merged = {**self._sticky, **(options or {})}
        return self.env.reset(seed=seed, options=merged)


def make_memory_gym_vec_env(
    env_name: str,
    n_envs: int = 8,
    seed: int = 0,
    reset_options: Optional[dict[str, Any]] = None,
) -> VecEnv:
    """Build a vectorized memory-gym env.

    The wrapper stack:
        gym.make(env_name)         # uint8 (H, W, 3) image obs, Discrete action
        → StickyResetOptions       # (optional) replay difficulty kwargs
        → NormalizeImageObs        # uint8 → float32 / 255
        → Monitor                  # SB3 episode-reward bookkeeping

    Args:
        env_name: full memory-gym id, e.g. ``"Endless-SearingSpotlights-v0"``.
        n_envs:   number of parallel envs.
        seed:     base seed; env i is seeded with ``seed + i``.
        reset_options: passed as ``options`` on every ``env.reset(...)``.
                       Common keys: ``agent_scale``, ``command_count``.
    """
    def _make_one(rank: int):
        def _init():
            env = gym.make(env_name)
            if reset_options:
                env = StickyResetOptions(env, reset_options)
            env = SuccessInfoAlias(env)
            env = NormalizeImageObs(env)
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            # memory-gym populates info only at episode end, but the keys are
            # env-specific (Monitor KeyErrors on absent keywords): MysteryPath
            # adds `num_fails` (off-path falls — the bonus-vs-penalty mechanism
            # metric), MortarMayhem adds `commands_completed` (progress
            # fraction). `success` is common to the suite. Capturing them keeps
            # goal-clearing disentangled from shaped reward on dense arms.
            if "MysteryPath" in env_name:
                keys = ("success", "num_fails")
            elif "MortarMayhem" in env_name:
                keys = ("success", "commands_completed")
            else:
                keys = ("success",)
            return Monitor(env, info_keywords=keys)
        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])
