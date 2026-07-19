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


class OraclePotentialWrapper(gym.Wrapper):
    """PBRS with a task-informed ORACLE state potential (H-POT §3.2) — the PBIM counterexample.

    Φ(s) = −β·d(agent, goal), d = Manhattan grid distance to the (designer-known) goal
    ``end``. Adds the potential-based shaping ``F_t = γ·Φ(s_{t+1}) − Φ(s_t)`` to the reward,
    Φ recomputed each step from the PRIVILEGED env geometry (``normalized_agent_position``,
    ``end``) — information no pixel-obs agent has (W-4: a diagnostic, not a deployable arm).

    Because Φ is a function of state only, PBRS is optimality-preserving (Ng–Harada–Russell
    1999): the discounted per-episode shaping sum ΣₙγⁿFₙ telescopes to γᵀΦ_T − Φ_0 = −Φ_0
    (endpoint-only, since Φ(absorbing)≡0 here), so it cannot change the optimal policy — only
    the critic's learning signal. Verified by the telescoping self-test.

    Why it exists: it is the constructible counterexample to the PBIM null. PBIM's potential
    is the value of the FROZEN agent's own near-constant bonus (degenerate); a TASK-INFORMED
    oracle potential MAY reopen the freeze (that is what PBRS is *for*). Either outcome is a
    locked win (handoff §3.2). ``β`` sets magnitude — tune it so ``info['oracle_absF']`` mean
    matches the e3b arm's delivered bonus within ~2×; ``gamma`` MUST match PPO's γ so the
    telescoping is consistent with the value targets (as PBIM requires).
    """

    def __init__(self, env: gym.Env, beta: float = 0.02, gamma: float = 0.995) -> None:
        super().__init__(env)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self._prev_phi = 0.0

    def _phi(self) -> float:
        u = self.env.unwrapped
        ax, ay = u.normalized_agent_position
        ex, ey = u.end
        d = abs(int(ax) - int(ex)) + abs(int(ay) - int(ey))     # Manhattan on the 7×7 grid
        return -self.beta * d

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self._phi()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Φ(absorbing) ≡ 0 on ANY episode end (goal or time-limit) → the added return is
        # −Φ(s_0), a per-start constant, so the optimal policy is provably unchanged.
        phi_next = 0.0 if (terminated or truncated) else self._phi()
        F = self.gamma * phi_next - self._prev_phi
        self._prev_phi = phi_next
        info = dict(info)
        info["oracle_F"] = float(F)
        info["oracle_absF"] = abs(float(F))
        return obs, float(reward + F), terminated, truncated, info


class DistractorRewardWrapper(gym.Wrapper):
    """Dense DISTRACTOR reward — the Reviewer-2 control: reward DENSITY without
    USEFULNESS. Pays ``ε`` for every step that lands on an ALREADY-VISITED tile
    this episode (farmable by oscillating on the known prefix; discovers nothing).

    Calibration (the load-bearing constraint): ``ε = eps_frac / T_max`` so farming
    the whole episode (≤ T_max revisits) totals ≤ ``eps_frac`` (=0.1) ≪ the unit
    goal reward. The OPTIMAL policy is therefore UNCHANGED — reaching the goal
    (return 1) still dominates farming (return ≤ 0.1) — so this does not "change
    the task"; it only removes reward sparsity and plants a farmable LOCAL optimum
    (a trap). ``ε ≥ 1/T_max`` would flip the optimum to "sit still" and change the
    task, which is exactly what we must not do.

    Falls pay 0: a fall increments the env's ``num_fails`` and resets the agent to
    a visited tile, so without this guard a fall would be rewarded as a revisit —
    the opposite of intent. We zero the reward on any step where ``num_fails``
    increased.

    Mirror of the ``aligned`` arm: aligned pays ``+0.1`` for advancing to a NEW
    frontier tile (useful → makes the bonus redundant); this pays ``ε`` for
    returning to an OLD tile (useless → bonus still needed). Same density axis,
    opposite usefulness — the controlled pair that separates structural sparsity
    from ordinary reward sparsity.
    """

    def __init__(self, env: gym.Env, eps_frac: float = 0.1) -> None:
        super().__init__(env)
        self._T = int(getattr(env.unwrapped, "max_episode_steps", 128))
        self._eps = float(eps_frac) / self._T   # per-revisit; per-episode total ≤ eps_frac
        self._visited: set = set()
        self._prev_fails = 0

    def _tile(self) -> tuple:
        ax, ay = self.env.unwrapped.normalized_agent_position   # integer grid cells
        return (int(ax), int(ay))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._visited = {self._tile()}
        self._prev_fails = int(getattr(self.env.unwrapped, "num_fails", 0))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        nf = int(getattr(self.env.unwrapped, "num_fails", self._prev_fails))
        fell = nf > self._prev_fails
        self._prev_fails = nf
        d = 0.0
        if not fell and not (terminated or truncated):
            t = self._tile()
            if t in self._visited:
                d = self._eps            # revisit an OLD tile → distractor pays
            else:
                self._visited.add(t)      # NEW frontier tile → no distractor
        info = dict(info)
        info["distractor_r"] = float(d)
        return obs, float(reward + d), terminated, truncated, info


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
    oracle_potential: Optional[dict[str, Any]] = None,
    distractor: Optional[dict[str, Any]] = None,
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
        oracle_potential: if given (a dict, e.g. ``{"beta": 0.02, "gamma": 0.995}``),
                       wrap MysteryPath with OraclePotentialWrapper — the task-informed
                       PBRS arm (H-POT §3.2). Set ``gamma`` to PPO's γ. MysteryPath only.
    """
    def _make_one(rank: int):
        def _init():
            env = gym.make(env_name)
            if reset_options:
                env = StickyResetOptions(env, reset_options)
            if oracle_potential is not None and "MysteryPath" in env_name:
                env = OraclePotentialWrapper(env, **oracle_potential)
            if distractor is not None and "MysteryPath" in env_name:
                env = DistractorRewardWrapper(env, **distractor)
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
            elif "SearingSpotlights" in env_name:
                keys = ("success", "coins_collected", "agent_health")
            else:
                keys = ("success",)
            return Monitor(env, info_keywords=keys)
        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])
