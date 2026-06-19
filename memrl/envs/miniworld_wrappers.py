"""MiniWorld-Sign (Farama ``miniworld``) → memory-RL wrapper.

Why a wrapper (the stock env is NOT a memory task)
--------------------------------------------------
``MiniWorld-Sign-v0`` fixes the sign colour and goal shape in ``__init__`` and never
re-randomises them in ``reset`` (``_gen_world`` reads the constant ``self._color_index``).
So across episodes the answer is identical and a feedforward policy memorises it in
its weights — there is nothing to *remember*. We turn it into the both-demand env the
3rd-slot needs by **randomising the sign colour every episode**: the agent must now
NAVIGATE to read the wall sign (coverage), then RETAIN that colour after it leaves the
field of view to reach the matching box (realisation). This is the MiniWorld analogue
of Two-Colors' disappearing cue — done without touching the engine.

What the wrapper does
---------------------
  * ``reset`` → draw a random colour index in [0, n_colors) and ``set_color_index``
    BEFORE the underlying reset so ``_gen_world`` paints that sign;
  * goal shape fixed to box (goal=0) so the only memory item is the sign COLOUR, and
    the Dict's ``goal`` field is uninformative → dropped;
  * obs stripped from ``Dict(obs=image, goal=…)`` to the raw ``(60,80,3)`` uint8 image
    so it routes through the existing channels-last CNN encoder;
  * reward is the stock SPARSE terminal +1 (touch the box whose colour matches the
    sign) → a no-bonus agent stalls (coverage) and a memoryless agent cannot solve it
    (realisation): the entanglement the env is here to demonstrate;
  * ``is_success`` emitted for SB3 Monitor / EvalCallback.

Note: the encoder's CNN is MiniGrid-tuned (three 2×2 convs); it functions on 60×80 but
a Nature-style CNN would be a better frontend for real pixel runs — a later tune.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import miniworld  # noqa: F401 — registers MiniWorld-* ids with gymnasium
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


class SignMemoryWrapper(gym.Wrapper):
    """Randomise the sign colour per episode; expose image-only obs + is_success."""

    def __init__(self, env: gym.Env, n_colors: int = 3, seed: int = 0):
        super().__init__(env)
        self._n_colors = int(n_colors)
        self._rng = np.random.default_rng(seed)
        self._sign_color = 0
        # Drop the Dict; the goal field is fixed (box) → uninformative. Keep the image.
        self.observation_space = env.observation_space["obs"]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Set the colour BEFORE reset so _gen_world paints this episode's sign.
        self._sign_color = int(self._rng.integers(0, self._n_colors))
        self.env.unwrapped.set_color_index(self._sign_color)
        state, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["sign_color"] = self._sign_color   # the episode's RM state = the cue colour
        info["is_success"] = False
        return state["obs"], info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["sign_color"] = self._sign_color   # carried so it's the exact retention target
        info["is_success"] = bool(terminated and reward > 0.5)
        return state["obs"], float(reward), terminated, truncated, info


def make_miniworld_vec_env(
    env_name: str,
    n_envs: int = 8,
    seed: int = 0,
    max_episode_steps: int = 150,
    size: int = 10,
    n_colors: int = 3,
    goal: int = 0,
    **_ignored,
) -> VecEnv:
    """Build a DummyVecEnv of sign-memory MiniWorld envs.

        gym.make(env_name, max_episode_steps, size, goal)   # Dict obs, Discrete(4)
        → SignMemoryWrapper      # per-episode random sign colour + image-only obs
        → Monitor                # SB3 episode bookkeeping (+ is_success)

    Args:
        env_name:          a "MiniWorld-*" id (currently only MiniWorld-Sign-v0).
        n_envs:            number of parallel envs.
        seed:              base seed; env rank is added per worker.
        max_episode_steps: horizon (stock default 20 is too short to separate memory
                           from reactivity; 150 gives a real read→traverse gap).
        size:              maze size passed to the env.
        n_colors:          number of sign colours to sample from (3 = blue/red/green).
        goal:              fixed goal shape (0=box, 1=key).
    """
    if "Sign" not in env_name:
        raise ValueError(f"only MiniWorld-Sign-v0 is wired; got {env_name!r}")

    def _make_one(rank: int):
        def _init():
            env = gym.make(env_name, max_episode_steps=max_episode_steps,
                           size=size, goal=goal)
            env = SignMemoryWrapper(env, n_colors=n_colors, seed=seed + rank)
            env.reset(seed=seed + rank)
            return Monitor(env, info_keywords=("is_success",))
        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])
