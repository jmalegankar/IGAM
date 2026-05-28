"""Smoke + contract tests for the T-Maze and classic-control wrappers.

T-Maze (Ni et al. 2023) covers the two new env families added for the
memory × exploration study:

  - Passive T-Maze: pure memory (credit-assignment horizon = 1).
  - Active T-Maze:  memory + credit assignment (both horizon = L).

plus the classic-control passthrough and its velocity-masked POMDP variant
(CartPole-v1 sanity rail + POMDP-CartPole-v1 minimal memory probe).

The contract tests assert the *mechanics* that make these valid probes:
optimal-policy reward, cue masking after t=0, the tight episode budget that
makes Active hard, and that masking actually zeros velocity dims.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from memrl.envs import make_vec_env
from memrl.envs.classic_wrappers import (
    VELOCITY_INDICES,
    MaskVelocityWrapper,
    is_classic_env,
)
from memrl.envs.tmaze_wrappers import TMazeEnv, _parse_tmaze_name


# ── T-Maze name parsing ─────────────────────────────────────────────────────


def test_parse_tmaze_name_variants():
    assert _parse_tmaze_name("TMaze-Passive-v0") == ("passive", None, False)
    assert _parse_tmaze_name("TMaze-Active-v0") == ("active", None, False)
    assert _parse_tmaze_name("TMaze-Active-L50-v0") == ("active", 50, False)
    assert _parse_tmaze_name("TMaze-Passive-Oracle-L20-v0") == ("passive", 20, True)
    # token order is insensitive
    assert _parse_tmaze_name("TMaze-L7-Active-v0") == ("active", 7, False)


def test_parse_tmaze_name_requires_mode():
    with pytest.raises(ValueError, match="Passive.*Active"):
        _parse_tmaze_name("TMaze-L10-v0")


def test_parse_tmaze_name_rejects_garbage_token():
    with pytest.raises(ValueError, match="unrecognized token"):
        _parse_tmaze_name("TMaze-Passive-Banana-v0")


# ── T-Maze mechanics ────────────────────────────────────────────────────────


def _run_optimal(mode: str, L: int, oracle: bool = False):
    """Roll an optimal policy: remember the cue, turn that way at the junction."""
    env = TMazeEnv(corridor_length=L, mode=mode, cue_steps=(L + 1 if oracle else 1))
    obs, _ = env.reset(seed=0)
    remembered_cue = obs[2]
    total, done, trunc, steps, info = 0.0, False, False, 0, {}
    while not (done or trunc):
        cue = obs[2] if oracle else remembered_cue
        at_junction = obs[1] > 0.5
        action = (0 if cue > 0 else 1) if at_junction else 3  # turn else go right
        obs, r, done, trunc, info = env.step(action)
        total += r
        steps += 1
    return total, steps, done, trunc, info


@pytest.mark.parametrize("L", [3, 10, 20])
def test_passive_optimal_solves_in_L_steps(L):
    """Passive: memory-only. Optimal policy gets +1 in exactly L steps."""
    total, steps, done, trunc, info = _run_optimal("passive", L)
    assert total == pytest.approx(1.0)
    assert steps == L
    assert done and not trunc
    assert info["success"] is True


@pytest.mark.parametrize("L", [3, 10, 20])
def test_active_optimal_solves_in_L_steps(L):
    """Active: memory + credit assignment. Optimal trajectory just fits L steps."""
    total, steps, done, trunc, info = _run_optimal("active", L)
    assert total == pytest.approx(1.0)
    assert steps == L
    assert done and not trunc
    assert info["success"] is True


def test_active_tight_budget_truncates_on_waste():
    """Active's default budget = L: any wasted step ⇒ can't reach junction ⇒ 0."""
    env = TMazeEnv(corridor_length=10, mode="active")
    env.reset(seed=1)
    total, done, trunc, steps = 0.0, False, False, 0
    while not (done or trunc):
        obs, r, done, trunc, info = env.step(0)  # 'up' is a no-op off the junction
        total += r
        steps += 1
    assert total == pytest.approx(0.0)
    assert trunc and not done
    assert steps == 10


def test_cue_visible_only_at_t0_by_default():
    """Non-oracle: cue is in the reset obs and gone after the first step."""
    env = TMazeEnv(corridor_length=6, mode="active")
    obs, _ = env.reset(seed=2)
    assert obs[2] in (-1.0, 1.0)            # cue present at t=0
    obs, *_ = env.step(3)
    assert obs[2] == 0.0                    # masked thereafter


def test_oracle_keeps_cue_visible_every_step():
    """Oracle control: cue never masked ⇒ memory unnecessary."""
    env = TMazeEnv(corridor_length=6, mode="active", cue_steps=999)
    obs, _ = env.reset(seed=3)
    sign = obs[2]
    assert sign in (-1.0, 1.0)
    for _ in range(4):
        obs, *_ = env.step(3)
        assert obs[2] == sign              # same nonzero cue persists


def test_wrong_turn_terminates_with_no_reward():
    """Turning the wrong way at the junction ends the episode at 0 reward."""
    env = TMazeEnv(corridor_length=4, mode="passive")
    obs, _ = env.reset(seed=0)
    cue = obs[2]
    wrong = 1 if cue > 0 else 0            # opposite of the correct turn
    done = trunc = False
    last_r = None
    while not (done or trunc):
        at_junction = obs[1] > 0.5
        a = wrong if at_junction else 3
        obs, last_r, done, trunc, info = env.step(a)
    assert done and last_r == pytest.approx(0.0)
    assert info["success"] is False


def test_corridor_length_must_be_at_least_2():
    with pytest.raises(ValueError, match="corridor_length"):
        TMazeEnv(corridor_length=1)


# ── T-Maze vectorized construction (via dispatcher) ─────────────────────────


@pytest.mark.parametrize(
    "env_name",
    ["TMaze-Passive-v0", "TMaze-Active-L50-v0", "TMaze-Passive-Oracle-L20-v0"],
)
def test_tmaze_vec_constructs_and_steps(env_name):
    env = make_vec_env(env_name, n_envs=2, seed=0)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (3,)
    assert isinstance(env.action_space, gym.spaces.Discrete) and env.action_space.n == 4
    obs = env.reset()
    assert obs.shape == (2, 3)
    acts = np.stack([env.action_space.sample() for _ in range(2)])
    obs2, rew, dones, infos = env.step(acts)
    assert obs2.shape == (2, 3) and rew.shape == (2,)
    env.close()


def test_tmaze_kwarg_overrides_name_length():
    """corridor_length kwarg supplies L when the name carries no L-token."""
    env = make_vec_env("TMaze-Active-v0", n_envs=1, seed=0, corridor_length=30)
    # x_norm increments by 1/(L-1) per advance; after one 'right' it should be
    # 1/29 for L=30.
    obs = env.reset()
    obs2, *_ = env.step(np.array([3]))
    assert obs2[0, 0] == pytest.approx(1.0 / 29, abs=1e-4)
    env.close()


# ── classic-control + masked POMDP ──────────────────────────────────────────


def test_is_classic_env_recognizes_plain_and_pomdp():
    assert is_classic_env("CartPole-v1")
    assert is_classic_env("POMDP-CartPole-v1")
    assert is_classic_env("Acrobot-v1")
    assert not is_classic_env("popgym-RepeatPreviousMedium-v0")
    assert not is_classic_env("POMDP-NotAnEnv-v9")


def test_cartpole_passthrough_constructs():
    env = make_vec_env("CartPole-v1", n_envs=2, seed=0)
    assert env.observation_space.shape == (4,)
    obs = env.reset()
    assert obs.shape == (2, 4)
    env.close()


def test_pomdp_cartpole_zeros_velocity_dims():
    """POMDP-CartPole-v1 must zero obs indices 1 (cart_vel) and 3 (pole_vel)."""
    env = make_vec_env("POMDP-CartPole-v1", n_envs=4, seed=0)
    obs = env.reset()
    assert np.allclose(obs[:, VELOCITY_INDICES["CartPole-v1"]], 0.0)
    obs2, *_ = env.step(np.array([env.action_space.sample() for _ in range(4)]))
    assert np.allclose(obs2[:, VELOCITY_INDICES["CartPole-v1"]], 0.0)
    # position dims (0, 2) should NOT be forced to zero in general.
    assert not np.allclose(obs[:, [0, 2]], 0.0)
    env.close()


def test_mask_velocity_wrapper_rejects_unknown_env():
    dummy = gym.make("CartPole-v1")
    with pytest.raises(ValueError, match="No velocity-index mask"):
        MaskVelocityWrapper(dummy, "NotARealEnv-v0")
    dummy.close()


def test_unknown_env_id_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown env id"):
        make_vec_env("TotallyBogus-v0", n_envs=1, seed=0)
