"""Abstract interface for intrinsic-reward modules.

A module:
  * computes per-env bonus at every env step (consumed by `MemPPO.collect_rollouts`),
  * optionally trains learned components on collected rollouts,
  * optionally resets per-episode state when env episodes terminate.

The contract is intentionally minimal so MemPPO doesn't need per-module
branches — every module looks the same from the outside.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, nn


class IntrinsicRewardModule(nn.Module, ABC):
    """Abstract base for all intrinsic-reward methods.

    Subclasses must implement `compute()`. The other hooks are no-ops by default
    so modules without learned params (e.g. E3B+frozen-random) or without
    episodic state (e.g. plain RND) can leave them alone.
    """

    # If True (default), MemPPO zeroes this module's bonus on done steps — the
    # fresh-episodic-state misattribution fix (see collect_rollouts). Modules
    # that emit a MEANINGFUL boundary-step value (PBIM's terminal potential
    # difference −Φ(s_{T−1})) set this False and handle boundaries themselves.
    zero_bonus_on_done: bool = True

    def __init__(self, n_envs: int, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.n_envs = n_envs
        self.device = torch.device(device)
        self._step_buf: list[float] = []   # for per-rollout diagnostics

    # ── required ────────────────────────────────────────────────────────────

    @abstractmethod
    @torch.no_grad()
    def compute(
        self,
        obs: Any,                       # raw obs from env.step()
        last_obs: Any,                  # obs from the previous step
        action: np.ndarray,             # action that produced obs (n_envs, ...)
        episode_start: np.ndarray,      # bool (n_envs,) — True where prev step ended an episode
        side: dict[str, Tensor],        # cell side outputs (innovation, eps_mem, ...)
        cell_state: dict[str, Tensor],  # current cell state (for memory-derived features)
    ) -> np.ndarray:
        """Return per-env bonus (n_envs,) float32, sanitized (finite, ≥0)."""

    # ── optional hooks ──────────────────────────────────────────────────────

    def update(self, rollout: Any) -> dict[str, float]:
        """Train any learned predictors on the freshly-collected rollout.

        Default: no-op (modules with frozen-only components — RND target,
        random CNN — return {} here).

        Returns a dict of scalars to log (e.g. {"rnd_loss": 0.034}).
        """
        return {}

    def reset_envs(self, env_ids: np.ndarray | list[int]) -> None:
        """Reset per-episode state for the given envs.

        Default: no-op (lifelong-only modules like RND don't need this).
        Episodic modules (E3B, NovelD) MUST override.
        """
        pass

    # ── diagnostics helpers ─────────────────────────────────────────────────

    def diagnostics(self) -> dict[str, float]:
        """Return per-rollout scalar diagnostics (logged each rollout).

        Default: bonus mean/max/std over the most recent rollout. Subclasses
        can extend by overriding and merging with super().diagnostics().
        """
        if not self._step_buf:
            return {}
        arr = np.asarray(self._step_buf, dtype=np.float32)
        d = {
            "bonus_mean": float(arr.mean()),
            "bonus_max":  float(arr.max()),
            "bonus_std":  float(arr.std()),
        }
        self._step_buf = []
        return d

    def _record_bonus(self, bonus: np.ndarray) -> None:
        """Helper for subclasses: stash the per-step mean bonus for diagnostics."""
        self._step_buf.append(float(bonus.mean()))
