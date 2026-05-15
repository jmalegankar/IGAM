"""MiniGrid environment factory for selective-memory tasks.

The MG-M-S{7,9,11,13} envs put a key-shaped object in the start room, then
require the agent to traverse a corridor and pick the matching object in the
end room. The corridor steps contain no information about which key was
shown — the agent must *selectively write* only the start-room observation
to memory and *ignore* corridor observations. This is exactly the regime
where a write gate should pay off, in contrast to POPGym RepeatPrevious
where every observation is informative.

Obs handling — important nuance:
    MiniGrid's default `image` observation is a (7, 7, 3) uint8 array, but
    despite the shape it is NOT an image. The three channels are categorical
    codes:
        channel 0: object type    (0..10 — wall, door, key, ball, goal, ...)
        channel 1: color          (0..5  — red, green, blue, purple, ...)
        channel 2: state          (0..2  — open/closed/locked for doors, etc.)
    Treating these as continuous pixel values (e.g., dividing by 255 and
    feeding to an MLP) is wrong — object_type=4 (door) is NOT closer to
    object_type=5 (key) than to object_type=10 (agent).

    The correct treatment is one-hot encoding per channel.
    `OneHotPartialObsWrapper` from minigrid does exactly this: it expands
    the (7, 7, 3) categorical to (7, 7, 20) binary, with 11+6+3=20 hot dims
    per cell (3 bits set per cell). Flattened: 7*7*20 = 980 input dim.

    We then cast uint8 → float32 so FlatEncoder's Box branch handles it.

Action space:
    Default Discrete(7) — left, right, forward, pickup, drop, toggle, done.
    The agent will learn to ignore unused actions. Could restrict via an
    ActionWrapper but the cost is one more abstraction layer; defer until
    baseline performance is known.

Recommended task IDs:
    MiniGrid-MemoryS7-v0       — quick smoke test (short corridor)
    MiniGrid-MemoryS11-v0      — middle difficulty
    MiniGrid-MemoryS13-v0      — canonical memory benchmark
    MiniGrid-MemoryS17Random-v0 — hardest, with random object positions
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from minigrid.wrappers import ImgObsWrapper, OneHotPartialObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


class CastImageFloat32(gym.ObservationWrapper):
    """Cast a uint8 image observation to float32 with the same value range.

    After `OneHotPartialObsWrapper` the image is already 0/1 — no rescaling
    needed, just a dtype cast so FlatEncoder.Box (which expects float input)
    can process it.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        old = env.observation_space
        assert isinstance(old, gym.spaces.Box) and old.dtype == np.uint8
        self.observation_space = gym.spaces.Box(
            low=float(old.low.min()),
            high=float(old.high.max()),
            shape=old.shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32)


def make_minigrid_vec_env(
    env_name: str,
    n_envs: int = 8,
    seed: int = 0,
) -> VecEnv:
    """Build a vectorized MiniGrid environment for memory tasks.

    The wrapper stack:
        gym.make(env_name)             # Dict obs incl. (7,7,3) categorical img
        → OneHotPartialObsWrapper      # (7,7,3) categorical → (7,7,20) one-hot
        → ImgObsWrapper                # strip Dict, keep image only
        → CastImageFloat32             # uint8 → float32 (no rescaling)
        → Monitor                      # SB3 episode-reward bookkeeping

    Args:
        env_name: full gym id, e.g. "MiniGrid-MemoryS13-v0".
        n_envs:   number of parallel environments.
        seed:     base seed; env i is seeded with `seed + i`.

    Returns:
        Vectorized environment ready to pass to IGAMPPO. The encoder will
        see a flat 7*7*20 = 980-d float32 vector per timestep (3 bits set
        per cell out of 20).
    """
    def _make_one(rank: int):
        def _init():
            env = gym.make(env_name)
            env = OneHotPartialObsWrapper(env)
            env = ImgObsWrapper(env)
            env = CastImageFloat32(env)
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            return Monitor(env)
        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])
