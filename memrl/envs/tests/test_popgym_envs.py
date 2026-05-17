"""Smoke tests for the 4-env POPGym suite + new wrappers.

Covers:
  - RepeatPreviousMedium  (Discrete obs, Discrete action, uniform)
  - CountRecallMedium     (MultiDiscrete obs → FlattenMultiDiscrete, Discrete action)
  - AutoencodeMedium      (Tuple obs → ExposePhaseInInfo + FlattenTupleDiscrete)
  - BattleshipEasy        (Discrete obs, MultiDiscrete action — verified through PPO)

Each env is constructed via `make_popgym_vec_env`, smoke-stepped for one rollout
chunk, and the wrapper-specific contracts are checked (info["phase"] for
Autoencode, lossless flattening for MultiDiscrete obs, etc.).

Battleship gets an extra end-to-end pass through MemPPO to verify the
MultiDiscrete action space flows correctly through the rollout buffer and
policy update (this is the integration point most likely to break).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from memrl.envs import make_vec_env
from memrl.envs.popgym_wrappers import (
    ExposePhaseInInfo,
    FlattenMultiDiscrete,
    FlattenTupleDiscrete,
    _mixed_base_multipliers,
)


# ── helpers ────────────────────────────────────────────────────────────────


def _step_one(env, n_envs):
    """Sample a random action per env and step. Returns (obs, reward, dones, infos)."""
    actions = np.stack(
        [env.action_space.sample() for _ in range(n_envs)], axis=0
    ) if hasattr(env.action_space, "sample") else None
    # SB3 VecEnv exposes a vectorized action space at .action_space, but the
    # underlying sub-env sampling above gives shape (n_envs, *act_shape).
    # For VecEnv the natural sampler is env.action_space.sample() returning
    # a single action; loop instead to keep this generic.
    acts = np.stack([env.action_space.sample() for _ in range(n_envs)], axis=0)
    obs, rewards, dones, infos = env.step(acts)
    return obs, rewards, dones, infos


# ── unit tests for the helpers ─────────────────────────────────────────────


def test_mixed_base_multipliers_basic():
    mults, total = _mixed_base_multipliers([2, 4])
    assert mults == [4, 1]
    assert total == 8


def test_mixed_base_multipliers_three_dims():
    mults, total = _mixed_base_multipliers([3, 4, 5])
    assert mults == [20, 5, 1]
    assert total == 60


def test_flatten_multidiscrete_lossless():
    """Round-trip every MultiDiscrete([3, 4]) value through the flattener."""

    class FakeEnv(gym.Env):
        observation_space = gym.spaces.MultiDiscrete([3, 4])
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            return np.array([0, 0]), {}

        def step(self, action):
            return np.array([0, 0]), 0.0, False, False, {}

    wrapped = FlattenMultiDiscrete(FakeEnv())
    assert isinstance(wrapped.observation_space, gym.spaces.Discrete)
    assert wrapped.observation_space.n == 12

    # Every (i, j) maps to a unique int in [0, 12).
    seen = set()
    for i in range(3):
        for j in range(4):
            flat = wrapped.observation(np.array([i, j]))
            assert flat == i * 4 + j
            seen.add(flat)
    assert seen == set(range(12))


def test_expose_phase_in_info():
    """Phase flag from obs[0] appears in info on reset and step."""

    class FakeTupleEnv(gym.Env):
        observation_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(2), gym.spaces.Discrete(4))
        )
        action_space = gym.spaces.Discrete(4)
        _t = 0

        def reset(self, *, seed=None, options=None):
            self._t = 0
            return (0, 2), {}  # phase=0 (WATCH)

        def step(self, action):
            self._t += 1
            phase = 0 if self._t < 3 else 1  # flip to PLAY at t=3
            return (phase, 1), 0.0, self._t > 5, False, {}

    wrapped = ExposePhaseInInfo(FakeTupleEnv())
    obs, info = wrapped.reset()
    assert info["phase"] == 0
    obs, *_, info = wrapped.step(0)
    assert info["phase"] == 0
    # advance to PLAY
    for _ in range(3):
        obs, *_, info = wrapped.step(0)
    assert info["phase"] == 1


# ── end-to-end construction tests for the 4 envs ───────────────────────────


def _construct_and_step(env_name: str, n_envs: int = 2):
    """Factory that builds + 1-step a vec env, returns (env, obs, reward, info)."""
    env = make_vec_env(env_name, n_envs=n_envs, seed=0)
    obs = env.reset()
    acts = np.stack([env.action_space.sample() for _ in range(n_envs)], axis=0)
    obs2, rew, dones, infos = env.step(acts)
    env.close()
    return obs, obs2, rew, dones, infos


def test_repeat_previous_medium_constructs():
    """RepeatPreviousMedium: plain Discrete obs + Discrete action."""
    obs, obs2, rew, dones, infos = _construct_and_step("popgym-RepeatPreviousMedium-v0")
    assert obs.shape == (2,) and obs2.shape == (2,)
    assert rew.shape == (2,)
    assert all(isinstance(i, dict) for i in infos)


def test_count_recall_medium_uses_flatten_multidiscrete():
    """CountRecallMedium: MultiDiscrete obs gets flattened to Discrete."""
    env = make_vec_env("popgym-CountRecallMedium-v0", n_envs=2, seed=0)
    # Underlying obs space (after flattening) must be Discrete.
    assert isinstance(env.observation_space, gym.spaces.Discrete), (
        f"expected Discrete after FlattenMultiDiscrete, got {env.observation_space}"
    )
    obs = env.reset()
    # Vectorized Discrete obs comes back as shape (n_envs,) of ints.
    assert obs.shape == (2,)
    # All obs in valid range for the flattened space.
    assert (obs >= 0).all() and (obs < env.observation_space.n).all()
    env.close()


def test_autoencode_medium_exposes_phase_and_flattens():
    """AutoencodeMedium: Tuple obs gets phase exposed in info, then flattened."""
    env = make_vec_env("popgym-AutoencodeMedium-v0", n_envs=2, seed=0)
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    obs = env.reset()
    # First step should reveal info["phase"] for each env (vec-env aggregates).
    acts = np.stack([env.action_space.sample() for _ in range(2)], axis=0)
    obs2, rew, dones, infos = env.step(acts)
    assert "phase" in infos[0], f"phase missing from info: {infos[0].keys()}"
    assert infos[0]["phase"] in (0, 1)
    env.close()


def test_battleship_easy_constructs_with_multidiscrete_action():
    """BattleshipEasy: Discrete obs + MultiDiscrete action; verify env steps cleanly."""
    env = make_vec_env("popgym-BattleshipEasy-v0", n_envs=2, seed=0)
    assert isinstance(env.action_space, gym.spaces.MultiDiscrete), (
        f"expected MultiDiscrete action, got {env.action_space}"
    )
    # Battleship obs is Discrete(2) — hit/miss — also auto-flattened path
    # (no-op since it's already Discrete).
    obs = env.reset()
    acts = np.stack([env.action_space.sample() for _ in range(2)], axis=0)
    obs2, rew, dones, infos = env.step(acts)
    assert obs2.shape == (2,)
    env.close()


# ── integration test: MultiDiscrete action through MemPPO ──────────────────


@pytest.mark.slow
def test_battleship_trains_one_update_through_mem_ppo():
    """MultiDiscrete action passes through MemPPO's policy + rollout buffer.

    Marked slow because it actually instantiates MemPPO and runs one
    rollout + update. The point is to surface MultiDiscrete-handling bugs in
    the policy / buffer / loss path, which a pure env smoke test wouldn't catch.
    """
    import torch

    from memrl.cell import GRU
    from memrl.ppo import MemPPO

    env = make_vec_env("popgym-BattleshipEasy-v0", n_envs=2, seed=0)
    model = MemPPO(
        env,
        cell_factory=lambda input_size: GRU(input_size, hidden_size=16),
        encoder_dim=16,
        n_steps=64,
        n_epochs=1,
        chunk_len=8,
        n_chunks_per_batch=2,
        verbose=0,
    )
    # Should run rollout + 1 epoch of PPO without crashing.
    model.learn(total_timesteps=128)
    env.close()
