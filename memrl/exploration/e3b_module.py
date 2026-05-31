"""E3B wrapper — composes a feature source (φ) with the geometric estimator.

The estimator (Sherman-Morrison + numerical safety) lives in `e3b.py` and was
written for the user's thesis. This wrapper plugs it into the unified
`IntrinsicRewardModule` interface so MemPPO can use it without per-method
branching.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import IntrinsicRewardModule
from .e3b import EllipticalEpisodicBonus, RunningStd
from .phi_sources import IDMPhi, PhiSource


class E3B(IntrinsicRewardModule):
    """Per-episode elliptical bonus over a configurable feature source.

    Args:
        n_envs:      number of parallel envs
        phi_source:  PhiSource instance (RandomPhi / ObsPhi / CellInnovationPhi)
        lambda_reg:  inverse-covariance prior; M_0 = (1/λ)I
        max_bonus:   clip returned bonus
        normalize:   divide bonus by running std before returning
        device:      torch device
    """

    def __init__(
        self,
        n_envs: int,
        phi_source: PhiSource,
        lambda_reg: float = 1.0,
        max_bonus: float = 10.0,
        normalize: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(n_envs=n_envs, device=device)
        self.phi = phi_source.to(self.device)
        self.estimator = EllipticalEpisodicBonus(
            n_envs=n_envs,
            dim=phi_source.dim,
            lambda_reg=lambda_reg,
            device=device,
            max_bonus=max_bonus * 10.0,   # we'll re-clip after normalize
        )
        self.normalize = normalize
        self.rms = RunningStd() if normalize else None
        self.max_bonus = max_bonus

    @torch.no_grad()
    def compute(self, obs, last_obs, action, episode_start, side, cell_state) -> np.ndarray:
        phi = self.phi.encode(obs, side, cell_state)              # (n_envs, dim)
        bonus_t = self.estimator.bonus_and_update(phi)            # (n_envs,)
        bonus = bonus_t.detach().cpu().numpy().astype(np.float32)

        if self.normalize:
            self.rms.update(bonus)
            bonus = bonus / max(self.rms.std, 1e-8)

        bonus = np.nan_to_num(bonus, nan=0.0, posinf=self.max_bonus, neginf=0.0)
        bonus = np.clip(bonus, 0.0, self.max_bonus)
        self._record_bonus(bonus)
        return bonus

    def reset_envs(self, env_ids: np.ndarray | list[int]) -> None:
        """Reset the per-episode ellipsoid for envs that just finished an episode."""
        self.estimator.reset(env_ids)

    def diagnostics(self) -> dict[str, float]:
        base = super().diagnostics()
        est_diag = self.estimator.get_diagnostics()
        # Reset estimator's running counters per rollout for clean per-rollout signal
        self.estimator.reset_diagnostics()
        base.update({
            f"e3b_{k}": float(v) for k, v in est_diag.items()
        })
        if self.rms is not None:
            base["e3b_running_std"] = float(self.rms.std)
        return base

    def update(self, rollout: Any) -> dict[str, float]:
        """E3B has no learned parameters when φ is frozen-random. The IDM-φ
        variant (`E3BIDM`) overrides this to train φ on the rollout."""
        return {}


class E3BIDM(E3B):
    """E3B with an inverse-dynamics-LEARNED φ — the canonical E3B (Henaff et al.
    NeurIPS 2022).

    Identical episodic elliptical machinery as `E3B`, but φ is trained, not
    frozen. An inverse-dynamics head predicts the action aₜ from
    (φ(sₜ), φ(sₜ₊₁)); minimizing that CE loss forces φ to encode the
    *controllable* part of the state. The bonus φᵀC⁻¹φ then measures episodic
    coverage in that controllable-feature space — a far stronger novelty signal
    than e3b_rand's random projection on structured (MiniGrid) observations.

    Shipped as a SEPARATE method (not a replacement for e3b_rand) so the
    random-φ vs learned-φ contrast is a measured ablation: same MLP shape, same
    estimator, same λ — only φ's training differs.

    Args (beyond E3B's):
        n_actions:   Discrete action vocab (inverse-model target).
        feature_dim: φ dimension (also the ellipsoid dimension).
        hidden_dim:  width of φ encoder + IDM head.
        lr:          Adam LR for φ + IDM head.
        idm_epochs:  passes over the rollout's transitions per update().
        idm_batch:   minibatch size for IDM training.
    """

    def __init__(
        self,
        n_envs: int,
        obs_dim: int,
        n_actions: int,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        idm_epochs: int = 1,
        idm_batch: int = 512,
        lambda_reg: float = 1.0,
        max_bonus: float = 10.0,
        normalize: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        phi = IDMPhi(obs_dim=obs_dim, hidden=hidden_dim,
                     feature_dim=feature_dim, device=device)
        super().__init__(
            n_envs=n_envs, phi_source=phi, lambda_reg=lambda_reg,
            max_bonus=max_bonus, normalize=normalize, device=device,
        )
        self.obs_dim = obs_dim
        self.n_actions = int(n_actions)
        self.feature_dim = feature_dim
        self.idm_epochs = int(idm_epochs)
        self.idm_batch = int(idm_batch)
        # Inverse-dynamics head: (φ_t, φ_{t+1}) → action logits.
        self.idm = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.n_actions),
        ).to(self.device)
        for m in self.idm:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        self.opt = torch.optim.Adam(
            list(self.phi.net.parameters()) + list(self.idm.parameters()), lr=lr
        )

    def update(self, rollout: Any) -> dict[str, float]:
        """Train φ + IDM head on within-episode (sₜ, aₜ, sₜ₊₁) transitions.

        Pulls the rollout buffer's (T, n_envs, *) observation / action /
        episode_start arrays, forms consecutive pairs, and DROPS pairs that
        cross an episode boundary (episode_starts[t+1] == 1) so the inverse
        model never sees a (last-obs → reset-obs) transition.
        """
        if not (hasattr(rollout, "observations") and hasattr(rollout, "actions")):
            return {}
        obs = np.asarray(rollout.observations)
        act = np.asarray(rollout.actions)
        if obs.ndim < 2 or obs.shape[0] < 2:
            return {}
        T, B = obs.shape[0], obs.shape[1]
        obs_f = obs.reshape(T, B, -1)
        s   = obs_f[:-1].reshape(-1, self.obs_dim)
        s2  = obs_f[1:].reshape(-1, self.obs_dim)
        a   = act.reshape(T, B)[:-1].reshape(-1)

        eps = getattr(rollout, "episode_starts", None)
        if eps is not None:
            # Valid transition iff the NEXT step is not a new-episode reset.
            keep = np.asarray(eps).reshape(T, B)[1:].reshape(-1) == 0
            s, s2, a = s[keep], s2[keep], a[keep]

        N = s.shape[0]
        if N < 2:
            return {}

        s_t  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        a_t  = torch.as_tensor(a,  dtype=torch.long,    device=self.device)

        last_loss, last_acc = 0.0, 0.0
        for _ in range(max(1, self.idm_epochs)):
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, self.idm_batch):
                idx = perm[i:i + self.idm_batch]
                phi_s  = self.phi(s_t[idx])           # grad path (IDMPhi.forward)
                phi_s2 = self.phi(s2_t[idx])
                logits = self.idm(torch.cat([phi_s, phi_s2], dim=-1))
                loss = F.cross_entropy(logits, a_t[idx])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                last_loss = float(loss.item())
                last_acc = float((logits.argmax(-1) == a_t[idx]).float().mean())
        return {
            "e3b_idm_loss": last_loss,
            "e3b_idm_acc":  last_acc,
            "e3b_idm_n":    float(N),
        }
