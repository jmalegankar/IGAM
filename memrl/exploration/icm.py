"""ICM — Intrinsic Curiosity Module (Pathak, Agrawal, Efros, Darrell ICML 2017).

Bonus = forward-model prediction error in an *inverse-dynamics learned*
feature space φ(x):

    φ̂_{t+1} = f_fwd(φ_t, a_t)            (forward model, learned)
    a_hat   = f_inv(φ_t, φ_{t+1})         (inverse model, also learned;
                                            constrains φ to be action-controllable)
    b_t = ‖φ̂_{t+1} − sg[φ_{t+1}]‖²

Loss for training (per the original paper):
    L_fwd = ‖φ̂_{t+1} − sg[φ_{t+1}]‖²
    L_inv = CE(a_hat, a) for discrete OR MSE for continuous
    L_total = β · L_fwd + (1 − β) · L_inv      (β controls inverse/forward balance)

ICM is *lifelong* (no episodic reset), so reset_envs is a no-op.
It's "noisy-TV partially robust" — better than vanilla forward prediction
because φ is constrained to action-controllable features, but not as robust
as RND.
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


class ICM(IntrinsicRewardModule):
    """Intrinsic Curiosity Module with discrete-action inverse loss.

    Args:
        n_envs:       parallel envs
        obs_dim:      flattened obs dim
        n_actions:    action vocab (Discrete)
        feature_dim:  φ dim
        hidden_dim:   internal MLP width
        lr:           Adam LR for all ICM params (φ, f_fwd, f_inv)
        beta:         L_fwd / L_inv balance — paper used 0.2
        max_bonus:    clip
    """

    def __init__(
        self,
        n_envs: int,
        obs_dim: int,
        n_actions: int,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        beta: float = 0.2,
        max_bonus: float = 10.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(n_envs=n_envs, device=device)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.beta = beta
        self.max_bonus = max_bonus

        # φ encoder — trained via inverse dynamics loss
        self.encoder = _MLP(obs_dim, hidden_dim, feature_dim).to(self.device)
        # Forward model: (φ, a) → φ̂_{t+1}.  Action represented as one-hot.
        self.fwd = _MLP(feature_dim + n_actions, hidden_dim, feature_dim).to(self.device)
        # Inverse model: (φ_t, φ_{t+1}) → a_hat (logits over n_actions)
        self.inv = _MLP(feature_dim * 2, hidden_dim, n_actions).to(self.device)

        params = (list(self.encoder.parameters())
                  + list(self.fwd.parameters())
                  + list(self.inv.parameters()))
        self.opt = torch.optim.Adam(params, lr=lr)
        self.rms = RunningStd()

    def _onehot_actions(self, a: np.ndarray) -> Tensor:
        a_t = torch.as_tensor(a.reshape(-1), dtype=torch.long, device=self.device)
        return F.one_hot(a_t, num_classes=self.n_actions).float()

    @torch.no_grad()
    def compute(self, obs, last_obs, action, episode_start, side, cell_state) -> np.ndarray:
        x_prev = _obs_to_tensor(last_obs, self.device)
        x_cur  = _obs_to_tensor(obs,      self.device)
        phi_prev = self.encoder(x_prev)
        phi_cur  = self.encoder(x_cur)
        a_oh = self._onehot_actions(action)
        phi_hat = self.fwd(torch.cat([phi_prev, a_oh], dim=-1))
        bonus_t = (phi_hat - phi_cur).pow(2).mean(dim=-1)
        bonus = bonus_t.cpu().numpy().astype(np.float32)

        self.rms.update(bonus)
        bonus = bonus / max(self.rms.std, 1e-8)
        bonus = np.nan_to_num(bonus, nan=0.0, posinf=self.max_bonus, neginf=0.0)
        bonus = np.clip(bonus, 0.0, self.max_bonus)
        self._record_bonus(bonus)
        return bonus

    def update(self, rollout: Any) -> dict[str, float]:
        """Train φ encoder + fwd + inv models on the rollout.

        Needs: observations (n_steps+1, n_envs, obs_dim) and actions
        (n_steps, n_envs). We extract consecutive (s, a, s') triples.
        """
        if not (hasattr(rollout, "observations") and hasattr(rollout, "actions")):
            return {}
        obs_buf = rollout.observations            # may be (T, B, F) or (T*B, F)
        act_buf = rollout.actions
        obs_t = _obs_to_tensor(obs_buf, self.device)
        if obs_t.dim() == 3:
            T, B = obs_t.shape[0], obs_t.shape[1]
            obs_t = obs_t.reshape(T, B, -1)
            s   = obs_t[:-1].reshape(-1, self.obs_dim)
            s2  = obs_t[1:].reshape(-1, self.obs_dim)
            a   = np.asarray(act_buf)[:-1].reshape(-1)
        else:
            # Already flattened — fall back: pair (i, i+1)
            obs_t = obs_t.reshape(-1, self.obs_dim)
            s, s2 = obs_t[:-1], obs_t[1:]
            a = np.asarray(act_buf).reshape(-1)[:-1]

        if s.shape[0] < 2:
            return {}

        a_oh = self._onehot_actions(a)
        phi_s  = self.encoder(s)
        phi_s2 = self.encoder(s2)
        # Forward loss
        phi_hat = self.fwd(torch.cat([phi_s, a_oh], dim=-1))
        L_fwd = F.mse_loss(phi_hat, phi_s2.detach())
        # Inverse loss
        a_logits = self.inv(torch.cat([phi_s, phi_s2], dim=-1))
        a_t = torch.as_tensor(a, dtype=torch.long, device=self.device)
        L_inv = F.cross_entropy(a_logits, a_t)

        L = self.beta * L_fwd + (1.0 - self.beta) * L_inv
        self.opt.zero_grad()
        L.backward()
        self.opt.step()
        return {"icm_fwd_loss": float(L_fwd.item()),
                "icm_inv_loss": float(L_inv.item()),
                "icm_running_std": float(self.rms.std)}
