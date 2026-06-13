"""TinyReproduce — the colleague's k-token reproduce task as a tiny RL env.

Show k tokens (vocab size v), then reproduce them (in-order or reverse). This is
Autoencode shrunk to a size where the minimal reward machine is FULLY ENUMERABLE
(vᵏ watch-leaves, merged by remaining-suffix during reproduce), so it is the
clean ground-truth env for the decodability / effective-RM-size probe and for the
P5 "freeze = collapse to the trivial RM" demonstration.

It is a RETENTION task (tokens dealt exogenously → front-loaded revelation → α≈0):
predict the bonus is ≈neutral, like S13 / MortarMayhem / Autoencode. Its value is
(i) exact RM ground truth, (ii) a guaranteed-learnable density toggle (unlike
52-card Autoencode), (iii) it instantiates the colleague's theoretical example.

Density toggle (in-env, exact):
  dense  = +1/k per correct reproduce step, 0 and TERMINATE on first wrong
           (the MortarMayhem rule).
  sparse = the SAME per-episode return paid as one terminal lump sum.
Both terminate on first wrong ⇒ identical dynamics, return-matched at γ=1; the
only difference is reward timing (δ). Optimum = 1.0 on both.

Obs (Box, dim 2+v): [is_watch, is_play, token_onehot(v)]. During watch the shown
token is one-hot; during play the token slot is zero (recall from memory).
Action: Discrete(v). Episode length 2k−1 (1 reset reveal + k−1 watch + k play).

info["rm_state"] = the exact minimal-RM state (remaining-to-reproduce tokens),
consumed by `decode_memory.py --task autoencode` (set --n-suits = v).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TinyReproduce(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, k: int = 4, v: int = 2, order: str = "reverse",
                 density: str = "dense"):
        super().__init__()
        if order not in ("reverse", "inorder"):
            raise ValueError("order must be 'reverse' or 'inorder'")
        if density not in ("sparse", "dense"):
            raise ValueError("density must be 'sparse' or 'dense'")
        self.k, self.v, self.order, self.density = int(k), int(v), order, density
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2 + self.v,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.v)
        self._max_steps = 2 * self.k - 1
        self._seq: list[int] = []
        self._shown = 0
        self._play_idx = 0
        self._acc = 0.0
        self._t = 0

    # ── obs builders ─────────────────────────────────────────────────────────
    def _watch_obs(self, token: int) -> np.ndarray:
        o = np.zeros(2 + self.v, dtype=np.float32)
        o[0] = 1.0                      # is_watch
        o[2 + token] = 1.0
        return o

    def _play_obs(self) -> np.ndarray:
        o = np.zeros(2 + self.v, dtype=np.float32)
        o[1] = 1.0                      # is_play
        return o

    def _target(self) -> int:
        return self._seq[self._play_idx] if self.order == "inorder" \
            else self._seq[self.k - 1 - self._play_idx]

    def _remaining(self) -> tuple:
        if self.order == "inorder":
            return tuple(self._seq[self._play_idx:])
        return tuple(self._seq[: self.k - self._play_idx][::-1])

    # ── gym API ──────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._seq = [int(self.np_random.integers(0, self.v)) for _ in range(self.k)]
        self._shown = 1                 # token 0 revealed in the reset obs
        self._play_idx = 0
        self._acc = 0.0
        self._t = 0
        info = {"phase": "watch", "rm_state": tuple(self._seq[:self._shown])}
        return self._watch_obs(self._seq[0]), info

    def step(self, action):
        self._t += 1
        terminated = truncated = False
        reward = 0.0

        if self._shown < self.k:
            # WATCH: reveal next token; action ignored; no reward.
            tok = self._seq[self._shown]
            self._shown += 1
            obs = self._watch_obs(tok)
            info = {"phase": "watch", "rm_state": tuple(self._seq[:self._shown])}
        else:
            # PLAY: compare to target under the chosen order.
            correct = int(action) == int(self._target())
            self._play_idx += 1
            if correct:
                self._acc += 1.0 / self.k
                done = self._play_idx >= self.k
                if self.density == "dense":
                    reward = 1.0 / self.k
                if done:                       # finished reproducing all
                    if self.density == "sparse":
                        reward = self._acc     # terminal lump sum
                    terminated = True
            else:
                # MortarMayhem rule: no credit, terminate. Sparse pays acc so far.
                if self.density == "sparse":
                    reward = self._acc
                terminated = True
            obs = self._play_obs()
            info = {"phase": "play", "rm_state": self._remaining(),
                    "n_correct": self._play_idx if correct else self._play_idx - 1,
                    "is_correct": correct}

        if self._t >= self._max_steps:
            truncated = True
        return obs, float(reward), terminated, truncated, info


def make_tiny_reproduce_vec_env(env_name: str, n_envs: int = 8, seed: int = 0,
                                k: int = 4, v: int = 2, order: str = "reverse",
                                density: str = "dense", **_ignored):
    """Vectorized TinyReproduce. env_name is accepted for dispatch uniformity
    (e.g. 'TinyReproduce-v0'); params come from env_kwargs (k, v, order, density)."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _make(rank: int):
        def _init():
            env = TinyReproduce(k=k, v=v, order=order, density=density)
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            return Monitor(env)
        return _init

    return DummyVecEnv([_make(i) for i in range(n_envs)])
