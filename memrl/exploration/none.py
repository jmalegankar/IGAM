"""No-bonus baseline. The control for every exploration ablation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from .base import IntrinsicRewardModule


class NoBonus(IntrinsicRewardModule):
    """Returns zero bonus every step. Use as the exploration-off baseline."""

    def __init__(self, n_envs: int, device: torch.device | str = "cpu", **_kwargs) -> None:
        super().__init__(n_envs=n_envs, device=device)
        self._zero = np.zeros(n_envs, dtype=np.float32)

    @torch.no_grad()
    def compute(self, obs, last_obs, action, episode_start, side, cell_state) -> np.ndarray:
        return self._zero
