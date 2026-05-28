"""Classic-control passthrough + velocity-masked POMDP variants.

Two roles:

  1. **Plain passthrough** for fully-observable classic-control envs
     (CartPole-v1, Acrobot-v1, ...). These are MDPs, not memory tasks — their
     job here is a sanity rail: if the PPO + encoder + cell plumbing can't
     solve CartPole-v1, the bug is in the trainer, not the memory architecture.
     A clean, fast pass on a fully-observable task is the first thing to check
     before blaming a flat POPGym curve on the cell.

  2. **Velocity-masked POMDP variant** ("POMDP-<id>"), which is the genuinely
     memory-relevant version. Zeroing the velocity components of the state
     (cart/pole velocities for CartPole) makes the env partially observable:
     the agent can only recover velocity by *integrating position over time*,
     i.e. by remembering past observations. This "P-MDP" / masked-velocity
     construction is a standard minimal memory benchmark (e.g. sb3-contrib's
     MaskVelocityWrapper, and the "Pole" POMDPs in the recurrent-RL literature).
     CartPole with velocities masked is solvable by a competent recurrent cell
     and unsolvable by a memoryless policy — a tight, cheap memory probe that
     runs orders of magnitude faster than POPGym or MiniGrid.

Routing (see ``memrl/envs/__init__.py``):
    "CartPole-v1"          → plain, fully observable
    "POMDP-CartPole-v1"    → CartPole-v1 with cart_vel + pole_vel zeroed
    "Acrobot-v1"           → plain
    "POMDP-Acrobot-v1"     → Acrobot-v1 with the two angular velocities zeroed
    (any id in VELOCITY_INDICES supports the "POMDP-" prefix)

Velocity indices below are ported from sb3-contrib's MaskVelocityWrapper.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

# Which observation indices are velocities (to be zeroed for the POMDP variant).
# Ported from sb3_contrib.common.wrappers.MaskVelocityWrapper. Keyed by the
# base (fully-observable) gym id.
VELOCITY_INDICES: dict[str, np.ndarray] = {
    "CartPole-v1": np.array([1, 3]),            # cart_vel, pole_ang_vel
    "MountainCar-v0": np.array([1]),            # velocity
    "MountainCarContinuous-v0": np.array([1]),  # velocity
    "Pendulum-v1": np.array([2]),               # angular velocity
    "Acrobot-v1": np.array([4, 5]),             # the two joint angular velocities
    "LunarLander-v2": np.array([2, 3, 5]),      # vx, vy, angular velocity
    "LunarLanderContinuous-v2": np.array([2, 3, 5]),
}

# Plain (fully-observable) classic-control ids we pass straight through. Keep
# this an explicit allow-list rather than a catch-all gym.make fallback, so a
# typo'd id still surfaces as "Unknown env id" from the dispatcher instead of a
# confusing deep gym error.
_CLASSIC_IDS = frozenset(VELOCITY_INDICES.keys())

_POMDP_PREFIX = "POMDP-"


def is_classic_env(env_name: str) -> bool:
    """True if `env_name` is a supported classic id or its "POMDP-" variant."""
    if env_name.startswith(_POMDP_PREFIX):
        return env_name[len(_POMDP_PREFIX):] in VELOCITY_INDICES
    return env_name in _CLASSIC_IDS


class MaskVelocityWrapper(gym.ObservationWrapper):
    """Zero out the velocity dimensions of a classic-control observation.

    Turns a fully-observable classic-control MDP into a POMDP whose hidden
    state (velocity) is recoverable only by integrating positions over time —
    i.e. only by an agent with memory. The observation space shape is preserved
    (velocities are set to 0, not removed) so encoders and logging are
    unchanged; the masked dims simply carry no signal.
    """

    def __init__(self, env: gym.Env, env_id: str) -> None:
        super().__init__(env)
        if env_id not in VELOCITY_INDICES:
            raise ValueError(
                f"No velocity-index mask known for {env_id!r}. Supported: "
                f"{sorted(VELOCITY_INDICES)}."
            )
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError(
                f"MaskVelocityWrapper expects a Box observation space, got "
                f"{type(env.observation_space).__name__}."
            )
        self._mask = np.ones(env.observation_space.shape, dtype=np.float32)
        self._mask[VELOCITY_INDICES[env_id]] = 0.0

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return (obs.astype(np.float32) * self._mask)


def make_classic_vec_env(
    env_name: str,
    n_envs: int = 8,
    seed: int = 0,
    **gym_kwargs: Any,
) -> VecEnv:
    """Build a vectorized classic-control env (plain or velocity-masked POMDP).

    Args:
        env_name: "CartPole-v1" (plain) or "POMDP-CartPole-v1" (masked). Must
                  satisfy `is_classic_env`.
        n_envs:   number of parallel envs.
        seed:     base seed; env i is seeded with `seed + i`.
        **gym_kwargs: forwarded to gym.make (e.g. max_episode_steps).

    Returns:
        DummyVecEnv (classic-control steps are cheap; IPC would dominate).
    """
    masked = env_name.startswith(_POMDP_PREFIX)
    base_id = env_name[len(_POMDP_PREFIX):] if masked else env_name
    if base_id not in _CLASSIC_IDS:
        raise ValueError(
            f"Unsupported classic env id {env_name!r}. Supported base ids: "
            f"{sorted(_CLASSIC_IDS)} (optionally with the {_POMDP_PREFIX!r} prefix)."
        )

    def _make_one(rank: int):
        def _init():
            env = gym.make(base_id, **gym_kwargs)
            if masked:
                env = MaskVelocityWrapper(env, base_id)
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            return Monitor(env)
        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])
