"""popgym-arcade environment factory.

popgym-arcade (https://github.com/bolt-research/popgym-arcade) is a
JAX/gymnax suite of pixel-based POMDP benchmarks (BattleShip, Navigator,
Skittles, CountRecall, Minesweeper, Tetris, ...). Each env supports a
``partial_obs=True`` POMDP variant — the natural setting for memory cells.

Why a custom adapter:
    Upstream uses the gymnax functional API (PRNG keys threaded through
    every reset/step, ``env, env_params = make(...)``), which doesn't match
    the synchronous gymnasium ``Env`` interface that the rest of the
    training stack (SB3 VecEnv, MemPPO) expects. We hide the JAX plumbing
    behind a single-env gymnasium adapter and stack ``DummyVecEnv`` on top.

    This isn't peak JAX throughput — no vmap over n_envs, and obs round-trip
    through numpy every step. If wall-clock becomes the bottleneck, replace
    DummyVecEnv with a native gymnax-vectorized adapter that vmaps step and
    presents the SB3 VecEnv interface directly.

Env naming:
    Configs use the ``popgym-arcade-`` prefix to disambiguate from upstream
    POPGym, e.g. ``popgym-arcade-BattleShipEasy``. The prefix is stripped
    before calling ``popgym_arcade.make``.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


POPGYM_ARCADE_PREFIX = "popgym-arcade-"


def _gymnax_to_gym_space(s: Any) -> gym.Space:
    """Convert a gymnax space to its gymnasium equivalent.

    Gymnax's space classes don't subclass gymnasium.Space, but they expose
    the same shape/dtype/n attributes — a structural conversion is enough.
    """
    cls = type(s).__name__
    if cls == "Discrete":
        return gym.spaces.Discrete(int(s.n))
    if cls == "Box":
        shape = tuple(s.shape)
        # gymnax often gives SCALAR low/high (e.g. 0 and 255 for a pixel Box);
        # gymnasium rejects a 0-d low alongside an explicit multi-dim shape, so
        # broadcast scalar bounds up to the full shape.
        low = np.asarray(s.low, dtype=np.dtype(s.dtype))
        high = np.asarray(s.high, dtype=np.dtype(s.dtype))
        if low.shape != shape:
            low = np.broadcast_to(low, shape).copy()
        if high.shape != shape:
            high = np.broadcast_to(high, shape).copy()
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.dtype(s.dtype))
    raise TypeError(f"Unsupported gymnax space: {cls}")


class GymnaxToGymAdapter(gym.Env):
    """Wrap a single gymnax env so it satisfies the gymnasium.Env interface.

    The adapter holds an internal JAX PRNGKey that's split for each reset
    and step. Observations and rewards are converted to numpy at the
    boundary so downstream code never sees a JAX type.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        env_name: str,
        partial_obs: bool = True,
        obs_size: int = 128,
        seed: int = 0,
        normalize_image: bool = True,
        resize_to: Optional[int] = None,
    ) -> None:
        import os
        # torch owns the GPU; without this, JAX preallocates CUDA memory in
        # GPU pods (fine locally on CPU-only machines, fatal when packed 2-3
        # runs per GPU on the cluster). Env stepping is CPU-bound anyway.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax  # local import — JAX is optional for the rest of memrl
        import popgym_arcade

        self._jax = jax
        self._env, self._env_params = popgym_arcade.make(
            env_name, partial_obs=partial_obs, obs_size=obs_size
        )
        self._key = jax.random.key(int(seed))
        self._state = None
        self._normalize_image = normalize_image
        self._resize_to = resize_to

        # Derive the obs space from an actual reset rather than trusting
        # upstream's observation_space(env_params): some popgym-arcade
        # versions report a fixed default resolution (e.g. 256) regardless of
        # the obs_size actually used to render (e.g. 128) — the declared and
        # emitted shapes disagree and DummyVecEnv's obs buffer breaks.
        probe_obs, _ = self._env.reset(jax.random.key(0), self._env_params)
        probe = self._to_obs(probe_obs)
        self.observation_space = gym.spaces.Box(
            low=0.0 if normalize_image else 0,
            high=1.0 if normalize_image else 255,
            shape=probe.shape,
            dtype=probe.dtype,
        )
        self.action_space = _gymnax_to_gym_space(
            self._env.action_space(self._env_params)
        )

    def _next_key(self):
        self._key, sub = self._jax.random.split(self._key)
        return sub

    def _to_obs(self, obs: Any) -> np.ndarray:
        arr = np.asarray(obs)
        if self._normalize_image and arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        if self._resize_to and arr.ndim == 3 and arr.shape[0] != self._resize_to:
            # Nearest-neighbor resize via index grids — no cv2/PIL dependency.
            # 84x84 gives exact memory/encoder parity with MysteryPath (the
            # validated 2-runs-per-GPU packing profile).
            n = self._resize_to
            idx = (np.arange(n) * (arr.shape[0] / n)).astype(np.intp)
            jdx = (np.arange(n) * (arr.shape[1] / n)).astype(np.intp)
            arr = arr[np.ix_(idx, jdx)]
        return arr

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._key = self._jax.random.key(int(seed))
        key = self._next_key()
        obs, state = self._env.reset(key, self._env_params)
        self._state = state
        return self._to_obs(obs), {}

    def step(self, action):
        key = self._next_key()
        obs, state, reward, done, info = self._env.step(
            key, self._state, int(action), self._env_params
        )
        self._state = state
        # gymnax conflates terminated and truncated into one `done` flag;
        # surface it as terminated and leave truncated False.
        return (
            self._to_obs(obs),
            float(np.asarray(reward)),
            bool(np.asarray(done)),
            False,
            {k: np.asarray(v) for k, v in dict(info).items()} if info else {},
        )


def make_popgym_arcade_vec_env(
    env_name: str,
    n_envs: int = 8,
    seed: int = 0,
    partial_obs: bool = True,
    obs_size: int = 128,
    resize_to: Optional[int] = None,
    defer_reward: bool = False,
) -> VecEnv:
    """Build a vectorized popgym-arcade env.

    Args:
        env_name: id with the ``popgym-arcade-`` prefix, e.g.
                  ``"popgym-arcade-BattleShipEasy"``. The prefix is stripped
                  before passing the remainder to ``popgym_arcade.make``.
        n_envs:   number of parallel envs.
        seed:     base seed; env i is seeded with ``seed + i``.
        partial_obs: True → POMDP variant (the regime memory cells target).
        obs_size: 128 or 256 — pixel resolution of the rendered obs.
        resize_to: optional square size to nearest-neighbor downsample the obs
                  (84 → memory/encoder parity with MysteryPath).
        defer_reward: if True, withhold all per-step reward and pay it as one
                  terminal lump (the SPARSE twin of the natively dense task).
    """
    if env_name.startswith(POPGYM_ARCADE_PREFIX):
        gymnax_name = env_name[len(POPGYM_ARCADE_PREFIX):]
    else:
        gymnax_name = env_name

    def _make_one(rank: int):
        def _init():
            from .popgym_wrappers import DeferredReward
            env = GymnaxToGymAdapter(
                gymnax_name,
                partial_obs=partial_obs,
                obs_size=obs_size,
                seed=seed + rank,
                resize_to=resize_to,
            )
            if defer_reward:
                env = DeferredReward(env)
            return Monitor(env)
        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])
