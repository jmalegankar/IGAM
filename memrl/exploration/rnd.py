"""Random Network Distillation (Burda, Edwards, Storkey, Klimov ICLR 2019).

Bonus = prediction error of a learned net against a frozen randomly-initialised
target net, applied to the observation alone:

    r_int_t = ‖f_θ(x_t) − f_target(x_t)‖²

Properties:
  * Lifelong novelty (decreases monotonically as f_θ learns to predict f_target).
  * Robust to noisy-TV (target is deterministic in obs).
  * NOT per-episode — the kill case from the user's MiniGrid Memory result:
    if the hint randomises per episode, RND eventually predicts the global
    distribution and stops rewarding any specific room.

Predictor and target are simple MLPs over the observation. For visual obs you'd
want CNN trunks; we keep it MLP since POPGym observations are vectorised after
the env wrappers (Discrete/MultiDiscrete → flattened ints → embedded).
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


def _obs_to_tensor(obs: Any, device: torch.device, dtype=torch.float32) -> Tensor:
    """Coerce gym obs (dict, ndarray, scalar) to a (B, F) float tensor."""
    if isinstance(obs, dict):
        # Concatenate dict values (uncommon for POPGym but safe).
        parts = [_obs_to_tensor(v, device, dtype) for v in obs.values()]
        return torch.cat(parts, dim=-1)
    arr = np.asarray(obs)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return torch.as_tensor(arr, dtype=dtype, device=device)


class _MLP(nn.Module):
    """Plain MLP used as both RND target and predictor."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RND(IntrinsicRewardModule):
    """RND with running-std normalization on the bonus.

    Args:
        n_envs:      number of parallel envs
        obs_dim:     dimension of (flattened) observation
        hidden_dim:  MLP hidden size (target + predictor share this)
        feature_dim: output dim of both nets (the embedding compared)
        lr:          predictor optimizer LR
        max_bonus:   sanitize: clip returned bonus
    """

    def __init__(
        self,
        n_envs: int,
        obs_dim: int,
        hidden_dim: int = 128,
        feature_dim: int = 64,
        lr: float = 1e-4,
        max_bonus: float = 10.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(n_envs=n_envs, device=device)
        self.obs_dim = obs_dim
        self.max_bonus = max_bonus

        self.target    = _MLP(obs_dim, hidden_dim, feature_dim).to(self.device)
        self.predictor = _MLP(obs_dim, hidden_dim, feature_dim).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False

        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.rms = RunningStd()

    @torch.no_grad()
    def compute(self, obs, last_obs, action, episode_start, side, cell_state) -> np.ndarray:
        x = _obs_to_tensor(obs, self.device)
        t = self.target(x)
        p = self.predictor(x)
        # raw bonus: per-feature L2 distance, summed over dim
        bonus_t = (p - t).pow(2).mean(dim=-1)                  # (n_envs,)
        bonus = bonus_t.detach().cpu().numpy().astype(np.float32)

        # Normalize by running std (RND-standard). Update stats BEFORE dividing.
        self.rms.update(bonus)
        bonus = bonus / max(self.rms.std, 1e-8)
        bonus = np.nan_to_num(bonus, nan=0.0, posinf=self.max_bonus, neginf=0.0)
        bonus = np.clip(bonus, 0.0, self.max_bonus)
        self._record_bonus(bonus)
        return bonus

    def update(self, rollout: Any) -> dict[str, float]:
        """Train predictor on the freshly-collected rollout's observations.

        Expected rollout interface: an iterable of `observations` arrays
        (n_steps × n_envs × obs_dim). We use the entire buffer for one
        pass per rollout — standard RND practice.
        """
        # Try common buffer shapes
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
        return {"rnd_loss": float(loss.item()),
                "rnd_running_std": float(self.rms.std)}
