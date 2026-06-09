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

Parallelism:
    The factory picks DummyVecEnv (in-process, sequential env steps) or
    SubprocVecEnv (one OS process per env, parallel env steps) based on
    the env var ``MEMRL_VEC_ENV``:
        unset / "auto"  → SubprocVecEnv when n_envs > 1, else Dummy
        "dummy"         → always DummyVecEnv (single-thread; easier to debug)
        "subproc"       → always SubprocVecEnv
    POPGym envs are pure-Python with non-trivial per-step logic, so the
    SubprocVecEnv win is large (3-5x rollout throughput on the i9-11900K)
    once env stepping is the bottleneck — which it always is for PPO on
    a small recurrent cell.
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
import popgym  # noqa: F401 — registers POPGym envs with gymnasium
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


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
    """Collapse a MultiDiscrete observation space into a single Discrete,
    or — when the joint space is too large — into a multi-one-hot Box.

    Small spaces use the lossless mixed-base trick from `FlattenTupleDiscrete`
    (CountRecall's `MultiDiscrete([4, 4])` → `Discrete(16)`), which feeds the
    Embedding-based FlatEncoder. But the joint count is ∏nvec — for wide spaces
    (Concentration: `MultiDiscrete([3]*52)` → 3^52) a single Discrete overflows
    and an Embedding table would be absurd. Above `max_discrete` we instead emit
    the concatenated one-hot encoding, `Box(0, 1, (Σ nvec,))` (Concentration:
    156-dim), which routes to the MLP encoder path. Both encodings are lossless.
    """

    def __init__(self, env: gym.Env, max_discrete: int = 4096) -> None:
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
        self._sizes = sizes
        total = 1
        for n in sizes:
            total *= n
            if total > max_discrete:
                break
        self._one_hot = total > max_discrete
        if self._one_hot:
            self._offsets = np.cumsum([0] + sizes[:-1])
            self._dim = int(sum(sizes))
            self.observation_space = gym.spaces.Box(0.0, 1.0, (self._dim,),
                                                    dtype=np.float32)
        else:
            self._multipliers, total = _mixed_base_multipliers(sizes)
            self.observation_space = gym.spaces.Discrete(total)

    def observation(self, obs: Any):
        if self._one_hot:
            out = np.zeros(self._dim, dtype=np.float32)
            out[self._offsets + np.asarray(obs, dtype=np.int64)] = 1.0
            return out
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
class DeferredReward(gym.Wrapper):
    """Defer all per-step reward to the terminal step (one lump sum).

    Turns a natively DENSE POPGym task into its SPARSE twin: identical
    dynamics/observations/memory demand, identical episode return, but zero
    pre-terminal reward — the density toggle for the revelation/densification
    experiments (the reverse of MysteryPath's sparse→dense toggle).

    Policy-invariance: deferral reweights r_t's contribution from γ^t to γ^T.
    For POPGym prediction tasks (actions don't affect dynamics, r_t depends
    only on a_t) the per-step argmax is unchanged ⇒ π* identical. For tasks
    where actions steer dynamics (e.g. Battleship) it is invariant up to the
    γ^(T−t) reweighting — negligible at γ=0.995 with T≈10²; note in paper.

    Placed BEFORE Monitor so logged episode totals stay comparable across arms.
    """

    def reset(self, **kwargs):
        self._acc = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._acc += float(reward)
        if terminated or truncated:
            lump, self._acc = self._acc, 0.0
            return obs, lump, terminated, truncated, info
        return obs, 0.0, terminated, truncated, info


_PHASE_ENV_PREFIXES = ("popgym-Autoencode",)


def _resolve_vec_env_cls(n_envs: int):
    """Pick DummyVecEnv vs SubprocVecEnv from MEMRL_VEC_ENV.

    Benchmarked on the i9-11900K + RTX 3070 Ti:
        DummyVecEnv:   ~121,000 env-steps/sec   (POPGym AutoencodeMedium, n=8)
        SubprocVecEnv:  ~35,000 env-steps/sec   (same)
    SubprocVecEnv loses because POPGym envs are sub-microsecond per step;
    the IPC overhead from sending obs/actions over a pipe dwarfs the
    parallelism win. SubprocVecEnv is only worth it for envs with heavy
    per-step work (memory-gym, vision-based envs, ...).

    Default is therefore DummyVecEnv. Override with MEMRL_VEC_ENV=subproc.
    """
    mode = os.environ.get("MEMRL_VEC_ENV", "dummy").lower()
    if mode == "subproc":
        return SubprocVecEnv
    return DummyVecEnv


def make_popgym_vec_env(
    env_name: str,
    n_envs: int = 8,
    seed: int = 0,
    defer_reward: bool = False,
) -> VecEnv:
    """Build a vectorized POPGym environment.

    Each env is wrapped in `Monitor` so SB3 picks up episode rewards and
    lengths into `ep_info_buffer` — this is what populates `rollout/ep_rew_mean`
    and `rollout/ep_len_mean` in the tensorboard logs. Without Monitor those
    fields silently stay empty.

    The VecEnv class is chosen via ``MEMRL_VEC_ENV`` (see module docstring).
    On Windows + Python 3.11, SubprocVecEnv uses ``spawn`` semantics — env
    init runs in each subprocess so seeding and POPGym registration happen
    per-worker.

    Args:
        env_name: full gym id, e.g. "popgym-RepeatPreviousEasy-v0".
        n_envs:   number of parallel environments.
        seed:     base seed; env i is seeded with `seed + i`.
        defer_reward: if True, wrap with ``DeferredReward`` — all per-step
                  reward is withheld and paid as one terminal lump sum
                  (the SPARSE twin of the natively dense task).

    Returns:
        Vectorized environment ready to pass to MemPPO.
    """
    def _make_one(rank: int):
        def _init():
            # SubprocVecEnv on Windows uses 'spawn' — popgym registration
            # doesn't survive across the fork, so re-import in the worker.
            import popgym  # noqa: F401
            env = gym.make(env_name)
            # Expose phase flag in info BEFORE obs-flattening (Autoencode etc.).
            if any(env_name.startswith(p) for p in _PHASE_ENV_PREFIXES):
                env = ExposePhaseInInfo(env, phase_index=0)
            # Auto-flatten factorized discrete obs spaces.
            if isinstance(env.observation_space, gym.spaces.Tuple):
                env = FlattenTupleDiscrete(env)
            elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
                env = FlattenMultiDiscrete(env)
            if defer_reward:
                env = DeferredReward(env)
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            return Monitor(env)
        return _init

    vec_cls = _resolve_vec_env_cls(n_envs)
    if vec_cls is SubprocVecEnv:
        # 'spawn' is the only safe start method on Windows. On Linux fork is
        # the default and faster; pass start_method explicitly so behavior is
        # platform-independent.
        return SubprocVecEnv(
            [_make_one(i) for i in range(n_envs)],
            start_method="spawn",
        )
    return DummyVecEnv([_make_one(i) for i in range(n_envs)])
