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

from typing import Optional

import gymnasium as gym
import numpy as np
from minigrid.wrappers import ImgObsWrapper, OneHotPartialObsWrapper, ViewSizeWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from .mem_start import MemoryStartWrapper


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


class MemoryRewardWrapper(gym.Wrapper):
    """Reward variants for MiniGrid Memory envs, matched to memory-gym MysteryPath.

    MiniGrid's native Memory reward is horizon-DISCOUNTED (``1 - 0.9*t/T`` on reaching the
    matching object, ``0`` on the wrong object or timeout) — it conflates success with speed and
    is *not* sparse. MysteryPath is flat ``+1`` (``reward_step=0``). To compare the two envs
    under a MATCHED reward shape and metric, this wrapper exposes:

      * ``mode="native"`` — pass the env's discounted reward through unchanged (the as-shipped
        MiniGrid reward; what the current S13 runs used).
      * ``mode="flat"``   — ``+1`` on reaching the matching object, ``0`` on the wrong object or
        timeout: the flat-sparse twin, reward- and metric-matched to MysteryPath's sparse arm.
      * ``move_penalty=p>0`` (the FREEZE arm) — additionally subtract ``p`` on every non-``nop``
        action, an UNCONDITIONAL per-move cost the native discount lacks (the discount is booked
        only on success, so it cannot induce a freeze). The ``nop`` action (MiniGrid ``done``,
        idx 6) is free → a do-nothing sanctuary. Set ``p = 1/T_max`` for the horizon-normalized
        freeze matching MysteryPath's ``-1/128`` off-path penalty.

    Always emits ``info["is_success"]`` (reached the matching object), read from env geometry
    (``agent_pos == success_pos``) so it is robust to the reward override — a binary success_rate
    on every arm, matched to MysteryPath.
    """

    def __init__(self, env: gym.Env, mode: str = "native",
                 move_penalty: float = 0.0, nop_action: int = 6) -> None:
        super().__init__(env)
        assert mode in ("native", "flat"), mode
        self.mode = mode
        self.move_penalty = float(move_penalty)
        self.nop_action = int(nop_action)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["is_success"] = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        u = self.env.unwrapped
        success = bool(
            terminated
            and getattr(u, "success_pos", None) is not None
            and tuple(u.agent_pos) == tuple(u.success_pos)
        )
        if self.mode == "flat":
            reward = 1.0 if success else 0.0
        if self.move_penalty and int(action) != self.nop_action:
            reward -= self.move_penalty
        info = dict(info)
        info["is_success"] = success
        return obs, float(reward), terminated, truncated, info


class DistractorRewardWrapper(gym.Wrapper):
    """Dense-but-useless reward for MiniGrid Memory envs (Reviewer-2 control).

    Pays ``ε`` every step the agent steps onto a grid cell it has ALREADY visited
    this episode (a revisit); a step onto a NEW cell pays nothing. The reward is
    *dense* (fires most steps once the agent has moved around) yet *useless* — it
    rewards occupancy of old tiles, never the cue-retention transition the task is
    about, and it does not help find the target object.

    Calibration (the load-bearing constraint): ``ε = eps_frac / T_max`` with
    ``T_max = env.max_steps`` (S13: 5·13² = 845), so farming the whole episode
    (≤ T_max revisits) totals ≤ ``eps_frac`` (=0.1) ≪ the unit ``+1`` goal reward.
    The OPTIMAL policy is UNCHANGED — reaching the matching object (return 1) still
    strictly dominates farming (return ≤ 0.1), and the +1 terminates the episode so
    farming cannot be stacked on top unboundedly. It only removes reward sparsity
    and plants a farmable LOCAL optimum. No DP solver is needed: the per-episode
    bound ≤ eps_frac holds by construction for ANY tile count and geometry.

    S13 vs MysteryPath, one difference worth noting: MysteryPath's distractor is
    density-matched to an ``aligned`` twin (which pays +0.1 for NEW frontier tiles).
    S13 has no ``aligned`` arm (no incremental progress to pay for), so this arm is
    the control against ``sparseV3``: if the bonus STILL equalizes here under a dense
    reward, the equalization is not a reward-frequency / cue-exposure artifact. There
    are no falls in MiniGrid Memory, so (unlike the MysteryPath wrapper) no fall guard
    is needed — every non-terminal step is eligible.
    """

    def __init__(self, env: gym.Env, eps_frac: float = 0.1) -> None:
        super().__init__(env)
        self._T = int(getattr(env.unwrapped, "max_steps", 845))
        self._eps = float(eps_frac) / self._T   # per-revisit; per-episode total ≤ eps_frac
        self._visited: set = set()

    def _tile(self) -> tuple:
        ax, ay = self.env.unwrapped.agent_pos      # integer grid cells (x, y)
        return (int(ax), int(ay))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._visited = {self._tile()}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        d = 0.0
        if not (terminated or truncated):
            t = self._tile()
            if t in self._visited:
                d = self._eps            # revisit an OLD tile → distractor pays
            else:
                self._visited.add(t)      # NEW tile → no distractor
        info = dict(info)
        info["distractor_r"] = float(d)
        return obs, float(reward + d), terminated, truncated, info


def make_minigrid_vec_env(
    env_name: str,
    n_envs: int = 8,
    seed: int = 0,
    use_wrapper: bool = False,
    agent_view_size: Optional[int] = None,
    reward_mode: str = "native",
    move_penalty: float = 0.0,
    distractor: Optional[dict] = None,
) -> VecEnv:
    """Build a vectorized MiniGrid environment for memory tasks.

    The wrapper stack:
        gym.make(env_name)             # Dict obs incl. (7,7,3) categorical img
        → OneHotPartialObsWrapper      # (7,7,3) categorical → (7,7,20) one-hot
        → ImgObsWrapper                # strip Dict, keep image only
        → CastImageFloat32             # uint8 → float32 (no rescaling)
        → Monitor                      # SB3 episode-reward bookkeeping

    Args:
        env_name:    full gym id, e.g. "MiniGrid-MemoryS13-v0".
        n_envs:      number of parallel environments.
        seed:        base seed; env i is seeded with `seed + i`.
        use_wrapper: if True, apply MemoryStartWrapper to force corridor-entrance
                     spawn (Phase 0/1 crutch — removes the exploration bottleneck).
        agent_view_size: if set (odd int ≥3), shrink the egocentric view via
                     ViewSizeWrapper. Smaller views (e.g. 3) increase positional
                     aliasing so the task demands MORE memory (RLBenchNet trick):
                     it turns "solvable without memory" envs like RedBlueDoors
                     into genuine memory tests. Obs becomes (N, N, 20).

    Returns:
        Vectorized environment ready to pass to MemPPO. The encoder will
        see a flat 7*7*20 = 980-d float32 vector per timestep (3 bits set
        per cell out of 20).
    """
    # Reward variants (flat-sparse / freeze) apply only to the Memory-* envs, which expose the
    # success_pos geometry the wrapper reads. For non-memory ids (e.g. RedBlueDoors) the wrapper
    # is skipped. Applied innermost so it sees the raw Discrete(7) action and the unwrapped env.
    is_memory = "Memory" in env_name

    # Distractor arm applies only to the Memory-* envs (needs the flat reward + agent_pos).
    use_distractor = is_memory and distractor is not None
    info_keywords = ("is_success", "distractor_r") if use_distractor else ("is_success",)

    def _make_one(rank: int):
        def _init():
            env = gym.make(env_name)
            if is_memory:
                env = MemoryRewardWrapper(env, mode=reward_mode, move_penalty=move_penalty)
            if use_distractor:
                env = DistractorRewardWrapper(env, **distractor)
            if use_wrapper:
                env = MemoryStartWrapper(env)
            if agent_view_size is not None:
                env = ViewSizeWrapper(env, agent_view_size=agent_view_size)
            env = OneHotPartialObsWrapper(env)
            env = ImgObsWrapper(env)
            env = CastImageFloat32(env)
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            return Monitor(env, info_keywords=info_keywords) if is_memory else Monitor(env)
        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])
