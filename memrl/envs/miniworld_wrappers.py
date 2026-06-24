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

import os
import sys

import gymnasium as gym
import numpy as np

import miniworld  # noqa: F401 — registers MiniWorld-* ids with gymnasium
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

# miniworld pins pyglet<2.0 (the old GL stack), which needs a real X display to make
# a GL context — there is no clean EGL-headless mode. On a headless Linux pod (k8s)
# there is no DISPLAY, so we lazily start a virtual X server (Xvfb via pyvirtualdisplay)
# the first time a MiniWorld env is built. No-op on macOS (Cocoa backend) and no-op
# when a DISPLAY already exists (your local machine, or `xvfb-run`). Kept in a module
# global so the Xvfb process is not garbage-collected mid-run.
_VIRTUAL_DISPLAY = None


def _ensure_headless_display() -> None:
    global _VIRTUAL_DISPLAY
    if (sys.platform != "linux" or os.environ.get("DISPLAY")
            or _VIRTUAL_DISPLAY is not None):
        return
    try:
        from pyvirtualdisplay import Display
        _VIRTUAL_DISPLAY = Display(visible=False, size=(1024, 768))
        _VIRTUAL_DISPLAY.start()                     # sets os.environ["DISPLAY"]
    except Exception as e:                            # pragma: no cover
        import warnings
        warnings.warn(
            f"MiniWorld headless display setup failed ({e}); falling back to any "
            "existing DISPLAY / xvfb-run / EGL. Install `xvfb` + `pyvirtualdisplay`."
        )


class SignMemoryWrapper(gym.Wrapper):
    """Randomise the sign colour per episode; expose image-only obs + is_success."""

    def __init__(self, env: gym.Env, n_colors: int = 3, seed: int = 0,
                 reward_wrong: float = -1.0):
        super().__init__(env)
        self._n_colors = int(n_colors)
        self._rng = np.random.default_rng(seed)
        self._sign_color = 0
        # The stock Sign reward is float(correct)*2 - 1: +1 for touching the matching
        # object, -1 for touching ANY non-matching object, 0 on timeout / end-action.
        # We expose the wrong-object reward as a knob so the env can run two
        # return-matched arms (optimal play = +1 in BOTH; only sub-optimal play differs):
        #   penalty (reward_wrong=-1, stock): blind commitment is -EV (1/6·(+1)+5/6·(-1)≈
        #       -0.67), the timeout/end-action is a 0-cost sanctuary → predicts the FREEZE
        #       (success→0, committed→0) for a no-bonus learner;
        #   sparse  (reward_wrong= 0): blind commitment is +EV (≈+0.17), no avoidable
        #       penalty → the return-matched twin stays ALIVE (keeps committing).
        # This is the MiniWorld-3D analogue of MysteryPath's sparse↔fall-penalty contrast.
        self._reward_wrong = float(reward_wrong)
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
        reward = float(reward)
        correct = bool(terminated and reward > 0.5)    # touched the matching object (+1)
        wrong = bool(terminated and reward < -0.5)     # touched a non-matching object (stock -1)
        if wrong:
            reward = self._reward_wrong                 # penalty: -1 (stock) · sparse twin: 0
        info = dict(info)
        info["sign_color"] = self._sign_color   # carried so it's the exact retention target
        info["is_success"] = correct
        # 'committed' = the agent touched ANY object this episode = the alive / anti-freeze
        # signal (MiniWorld analogue of MysteryPath 'falls'): a frozen agent never commits
        # (committed→0 AND success→0); an alive agent commits often.
        info["committed"] = correct or wrong
        return state["obs"], reward, terminated, truncated, info


def make_miniworld_vec_env(
    env_name: str,
    n_envs: int = 8,
    seed: int = 0,
    max_episode_steps: int = 150,
    size: int = 10,
    n_colors: int = 3,
    goal: int = 0,
    reward_wrong: float = -1.0,
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
        reward_wrong:      reward for touching a non-matching object. -1.0 = stock
                           (the penalty/freeze arm); 0.0 = the return-matched sparse
                           twin (no avoidable penalty). Optimal play (+1) is unchanged.
    """
    if "Sign" not in env_name:
        raise ValueError(f"only MiniWorld-Sign-v0 is wired; got {env_name!r}")

    _ensure_headless_display()   # start a virtual X server on headless Linux (no-op else)

    def _make_one(rank: int):
        def _init():
            env = gym.make(env_name, max_episode_steps=max_episode_steps,
                           size=size, goal=goal)
            env = SignMemoryWrapper(env, n_colors=n_colors, seed=seed + rank,
                                    reward_wrong=reward_wrong)
            env.reset(seed=seed + rank)
            return Monitor(env, info_keywords=("is_success", "committed"))
        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])
