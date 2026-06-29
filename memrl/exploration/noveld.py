"""NovelD (Zhang et al. NeurIPS 2021).

Hybrid: RND error difference between consecutive observations, gated by
first-visit-within-episode. Bridges global (RND) and episodic (count-based)
exploration:

    b_t = max(rnd_err(x_t) − α · rnd_err(x_{t-1}), 0) · 1[first_visit_this_episode(x_t)]

Where the first-visit indicator is approximated by an episodic visitation set
(hashed obs). For continuous obs we'd need a learned kNN; for POPGym's
discrete/MultiDiscrete obs, exact hashing works.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import IntrinsicRewardModule
from .e3b import RunningStd
from .rnd import _MLP, _obs_to_tensor


class _EpisodicVisitationSet:
    """Per-env hash set of observation tuples visited this episode."""

    def __init__(self, n_envs: int) -> None:
        self.sets: list[set] = [set() for _ in range(n_envs)]

    def first_visit_then_mark(self, keys: list) -> np.ndarray:
        """Return bool mask (n_envs,): True if the per-env key is a first-visit, then mark.
        `keys` is a list of hashable per-env keys — an obs byte-hash for discrete-grid envs,
        or a discrete pose key (info["novelty_key"]) for continuous-state envs so the gate
        saturates."""
        out = np.zeros(len(self.sets), dtype=bool)
        for i, k in enumerate(keys):
            if k not in self.sets[i]:
                out[i] = True
                self.sets[i].add(k)
        return out

    def reset_envs(self, env_ids) -> None:
        for i in env_ids:
            self.sets[i] = set()


class NovelD(IntrinsicRewardModule):
    """RND error difference + first-visit gating.

    Args:
        n_envs:        number of parallel envs
        obs_dim:       flattened obs dim
        hidden_dim:    RND MLP hidden size
        feature_dim:   RND output dim
        lr:            predictor LR
        alpha_decay:   coefficient α in (rnd(x_t) − α · rnd(x_{t-1}))
        max_bonus:     clip
    """

    def __init__(
        self,
        n_envs: int,
        obs_dim: int,
        hidden_dim: int = 128,
        feature_dim: int = 64,
        lr: float = 1e-4,
        alpha_decay: float = 0.5,
        max_bonus: float = 10.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(n_envs=n_envs, device=device)
        self.obs_dim = obs_dim
        self.alpha_decay = alpha_decay
        self.max_bonus = max_bonus

        self.target    = _MLP(obs_dim, hidden_dim, feature_dim).to(self.device)
        self.predictor = _MLP(obs_dim, hidden_dim, feature_dim).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.rms = RunningStd()

        self.visits = _EpisodicVisitationSet(n_envs)
        # Tell the PPO loop to pass `infos` into compute() so we can read a discrete
        # env-provided novelty key (info["novelty_key"]) for the episodic gate.
        self.wants_infos = True

    @torch.no_grad()
    def _rnd_err(self, x: Tensor) -> Tensor:
        return (self.predictor(x) - self.target(x)).pow(2).mean(dim=-1)

    @torch.no_grad()
    def compute(self, obs, last_obs, action, episode_start, side, cell_state,
                infos=None) -> np.ndarray:
        x_cur  = _obs_to_tensor(obs,      self.device)
        x_prev = _obs_to_tensor(last_obs, self.device)
        err_cur  = self._rnd_err(x_cur).cpu().numpy().astype(np.float32)
        err_prev = self._rnd_err(x_prev).cpu().numpy().astype(np.float32)

        diff = np.maximum(err_cur - self.alpha_decay * err_prev, 0.0)

        # First-visit gate (per-episode). Reset visited set for envs that
        # just ended their previous episode (episode_start=True means the
        # current step is the first of a new episode).
        reset_ids = np.where(episode_start)[0].tolist()
        if reset_ids:
            self.visits.reset_envs(reset_ids)
        # Prefer a DISCRETE env-provided pose key (info["novelty_key"]) so the gate
        # saturates on continuous-state envs; else hash the raw obs (correct
        # for discrete-grid envs S13 / MysteryPath, whose frames are finite).
        n = len(self.visits.sets)
        if (infos is not None and len(infos) == n
                and all(isinstance(inf, dict) and "novelty_key" in inf for inf in infos)):
            keys = [infos[i]["novelty_key"] for i in range(n)]
        else:
            flat = np.asarray(obs).reshape(n, -1)
            keys = [flat[i].tobytes() for i in range(n)]
        first_visit_mask = self.visits.first_visit_then_mark(keys)

        bonus = diff * first_visit_mask.astype(np.float32)

        self.rms.update(bonus)
        bonus = bonus / max(self.rms.std, 1e-8)
        bonus = np.nan_to_num(bonus, nan=0.0, posinf=self.max_bonus, neginf=0.0)
        bonus = np.clip(bonus, 0.0, self.max_bonus)
        self._record_bonus(bonus)
        return bonus

    def reset_envs(self, env_ids) -> None:
        self.visits.reset_envs(env_ids)

    def update(self, rollout: Any) -> dict[str, float]:
        if hasattr(rollout, "observations"):
            obs_buf = rollout.observations
        elif isinstance(rollout, dict) and "observations" in rollout:
            obs_buf = rollout["observations"]
        else:
            return {}
        x = _obs_to_tensor(obs_buf, self.device).reshape(-1, self.obs_dim)
        if x.shape[0] == 0:
            return {}
        with torch.no_grad():
            t = self.target(x)
        p = self.predictor(x)
        loss = F.mse_loss(p, t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {"noveld_loss": float(loss.item()),
                "noveld_running_std": float(self.rms.std)}
