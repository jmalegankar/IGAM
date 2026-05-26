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
from torch import Tensor

from .base import IntrinsicRewardModule
from .e3b import EllipticalEpisodicBonus, RunningStd
from .phi_sources import PhiSource


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
        """E3B has no learned parameters when φ is frozen-random. If you want
        an IDM-trained φ, plug that into the PhiSource and train it via a
        separate path (the thesis used an IDM trained on rollouts; we don't
        include that loss here because φ^rand worked better empirically)."""
        return {}
