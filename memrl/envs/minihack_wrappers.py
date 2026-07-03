"""MiniHack (NetHack Learning Environment) → memory-RL wrapper.

TWO DISTINCT task families — do NOT conflate (an earlier version of this docstring did):

  • Memento (-Short-F2 / -F2 / -F4) = a MiniHack MEMORY task (MiniHack benchmark,
    Samvelyan et al. 2021). NOT an E3B/NovelD/RIDE benchmark — those used the procedural
    MultiRoom / LavaCrossing / Labyrinth COVERAGE tasks; Memento appears in none of them
    (verified against Henaff et al. 2022, E3B). A cue (a sleeping monster of a specific
    type) is shown ONLY at episode start; the agent walks a straight corridor (cue leaves
    the egocentric crop after ~1 step → memory required), then picks a fork from memory
    (wrong fork = invisible trap, native −1). Same cue→retain→choose structure as
    MiniGrid-MemoryS13.
      - A coverage/exploration bonus does NOT help Memento, as a category: the only thing
        to "discover" is a 1-bit memory-gated fork, not new states, so novelty just pulls
        the agent off the single optimal corridor (empirically E3B/NovelD HURT it).
      - Trainability caveat (two stacked horizons): GAE handles the short reward→fork
        credit (~13 steps) fine, but fork→cue is pure BPTT through the recurrence over the
        corridor length, truncated at chunk_len. Short-F2 (~8-step gap) trains to 1.0; the
        long F2/F4 (~57-step gap) floors at chance even at chunk_len 128 — needs chunk_len
        ≫ gap (BPTT-memory-expensive) or an auxiliary cue-preservation loss. Use Short-F2
        as the controlled memory probe.

  • Corridor (-R2/R3/R5) and MazeWalk (-9x9/-15x15/-45x19) = EXPLORATION tasks (no cue, no
    held memory) — the genre exploration bonuses actually target. Hard at small compute
    (E3B used 50M steps + IMPALA + full obs).

We expose the agent-centred **glyph crop** (a small grid of categorical glyph IDs) as
the observation; it routes to `GlyphEncoder` (embedding + CNN). CPU-only, no display/GL
(NLE is a C++ NetHack build, not a renderer).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


class GlyphMemoryWrapper(gym.Wrapper):
    """MiniHack Dict obs → the agent-centred glyph crop only; emit is_success.

    The raw MiniHack obs is a Dict; we keep `glyphs_crop` (an N×N int grid of glyph IDs
    centred on the agent) as the single observation. `is_success` = reached the
    cue-matched target (terminated with positive reward), for SB3 EvalCallback / Monitor.
    """

    def __init__(self, env: gym.Env, obs_key: str = "glyphs_crop",
                 reward_lose: float | None = None):
        super().__init__(env)
        self._obs_key = obs_key
        self.observation_space = env.observation_space[obs_key]
        # reward_lose: remap the wrong-fork death reward (the task's hardcoded −1, NOT the
        # generic MiniHack reward_lose) to this value. reward_lose=0 makes blind commitment
        # +EV so the agent learns to commit instead of collapsing into wander-avoidance.
        self._reward_lose = reward_lose

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["is_success"] = False
        return np.asarray(obs[self._obs_key]), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        correct = bool(terminated and reward > 0.5)        # reached the cue-matched target
        wrong = bool(terminated and reward < -0.5)         # wrong fork / death (task's −1)
        if wrong and self._reward_lose is not None:
            reward = float(self._reward_lose)
        info = dict(info)
        info["is_success"] = correct
        return np.asarray(obs[self._obs_key]), reward, terminated, truncated, info


def make_minihack_vec_env(
    env_name: str,
    n_envs: int = 8,
    seed: int = 0,
    obs_crop: int = 9,
    reward_win: float | None = None,
    reward_lose: float | None = None,
    penalty_step: float | None = None,
    **_ignored,
) -> VecEnv:
    """Build a DummyVecEnv of glyph-crop MiniHack envs.

        gym.make(MiniHack-*, observation_keys=("glyphs_crop",), obs_crop_h/w=obs_crop)
        → GlyphMemoryWrapper   # glyph-crop-only obs + is_success
        → Monitor              # SB3 episode bookkeeping (+ is_success)

    Args:
        env_name:  a "MiniHack-*" id (e.g. MiniHack-Memento-F2-v0, MiniHack-Corridor-R3-v0).
        obs_crop:  side length of the agent-centred glyph crop (default 9 → 9×9).
    """
    import minihack  # noqa: F401 — registers MiniHack-* ids with gymnasium

    # reward_win/penalty_step → MiniHack's reward manager (if it honours them). reward_lose
    # is the TASK's hardcoded wrong-fork −1 (NOT the generic reward_lose), so it is remapped
    # in GlyphMemoryWrapper.step, not via gym.make.
    reward_kw = {k: v for k, v in
                 (("reward_win", reward_win), ("penalty_step", penalty_step)) if v is not None}

    def _make_one(rank: int):
        def _init():
            env = gym.make(env_name, observation_keys=("glyphs_crop",),
                           obs_crop_h=obs_crop, obs_crop_w=obs_crop, **reward_kw)
            env = GlyphMemoryWrapper(env, reward_lose=reward_lose)
            env.reset(seed=seed + rank)
            return Monitor(env, info_keywords=("is_success",))
        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])
