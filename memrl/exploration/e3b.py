"""Elliptical Episodic Bonus (E3B, Henaff et al. NeurIPS 2022).

Per-episode Mahalanobis coverage signal in feature space φ ∈ R^C:

    b_t = φ_t^T Λ_{t-1}^{-1} φ_t,   Λ_t = λ I + Σ_{i≤t} φ_i φ_i^T

We maintain M_t := Λ_t^{-1} directly via Sherman-Morrison rank-1 updates:

    M_t = M_{t-1} - (M_{t-1} φ φ^T M_{t-1}) / (1 + φ^T M_{t-1} φ)

═══ MATHEMATICAL CORRECTNESS ═══

The Sherman-Morrison identity is exact ONLY when the denominator uses the
TRUE bonus b_true = φ^T M φ, not a clamped version. Clamping the denominator
causes M to over-subtract along φ, eventually making M lose PSD-ness and
producing negative bonuses. See GEX/lmu_ppo/episodic_bonus.py for the
full derivation and bug description.

═══ NUMERICAL SAFETY POLICY ═══

  • SM denominator uses the TRUE, unclamped b_true.
  • Returned bonus is sanitized: NaN/Inf → 0, negative → 0, clipped to max_bonus.
  • Per-env update outcomes:
      SAFE  : b_true finite, ≥ 0, < reject_threshold → SM update applied.
      SKIP  : b_true NaN/Inf or > reject_threshold → M preserved.
      RESET : b_true finite but negative (M lost PSD) → M reset to (1/λ)I.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import torch


class EllipticalEpisodicBonus:
    """Multi-env E3B bonus buffer with Sherman-Morrison updates and numerical safety.

    Args:
        n_envs:           number of parallel envs
        dim:              feature dimension C (matches phi output size)
        lambda_reg:       regularization λ; M_0 = (1/λ)I
        device:           torch device
        normalize_phi:    L2-normalize φ before SM (bounds b ∈ [0, 1/λ]). Default False.
        reject_threshold: skip SM update when b_true exceeds this. Default 1e6.
        max_bonus:        clip returned bonus to this. Default 100.0.
    """

    def __init__(
        self,
        n_envs: int,
        dim: int,
        lambda_reg: float = 1.0,
        device: Union[torch.device, str] = "cpu",
        normalize_phi: bool = False,
        reject_threshold: float = 1e6,
        max_bonus: float = 100.0,
    ) -> None:
        self.n_envs = n_envs
        self.dim = dim
        self.lam = lambda_reg
        self.device = torch.device(device)
        self.normalize_phi = normalize_phi
        self.reject_threshold = reject_threshold
        self.max_bonus = max_bonus

        self.M = self._fresh_M()

        self._cum_skipped = 0
        self._cum_reset = 0
        self._cum_b_max = -float("inf")
        self._cum_calls = 0

    def _fresh_M(self) -> torch.Tensor:
        eye = torch.eye(self.dim, device=self.device) / self.lam
        return eye.unsqueeze(0).expand(self.n_envs, -1, -1).contiguous()

    @torch.no_grad()
    def bonus_and_update(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute b_t = φ^T M_{t-1} φ with true SM update. Sanitize returned bonus.

        Args:
            phi: (n_envs, C) detached float tensor on self.device.
        Returns:
            bonus: (n_envs,) always finite, always in [0, max_bonus].
        """
        assert phi.shape == (self.n_envs, self.dim), \
            f"phi shape {phi.shape} != ({self.n_envs}, {self.dim})"

        if self.normalize_phi:
            phi = torch.nn.functional.normalize(phi, dim=-1, eps=1e-8)

        Mphi = torch.bmm(self.M, phi.unsqueeze(-1)).squeeze(-1)   # (n_envs, C)
        b_true = (phi * Mphi).sum(dim=-1)                         # (n_envs,)

        finite = torch.isfinite(b_true)
        nonneg = b_true >= 0
        small  = b_true < self.reject_threshold

        safe_mask  = finite & nonneg & small
        reset_mask = finite & (~nonneg)
        skip_mask  = (~finite) | (finite & ~small)

        # SM update with TRUE denominator — critical for correctness.
        denom = 1.0 + b_true.clamp(min=0.0)
        outer = Mphi.unsqueeze(-1) * Mphi.unsqueeze(-2)           # (n_envs, C, C)
        delta = outer / denom.view(-1, 1, 1)
        self.M = self.M - delta * safe_mask.float().view(-1, 1, 1)

        if reset_mask.any():
            eye = torch.eye(self.dim, device=self.device) / self.lam
            for i in torch.where(reset_mask)[0].tolist():
                self.M[i] = eye

        self._cum_skipped += int(skip_mask.sum().item())
        self._cum_reset   += int(reset_mask.sum().item())
        if finite.any():
            self._cum_b_max = max(self._cum_b_max, float(b_true[finite].max().item()))
        self._cum_calls += 1

        zeros = torch.zeros_like(b_true)
        b_clean = torch.where(finite, b_true, zeros)
        return b_clean.clamp(min=0.0, max=self.max_bonus)

    @torch.no_grad()
    def reset(self, env_ids: Union[List[int], np.ndarray, torch.Tensor]) -> None:
        """Reset M to (1/λ)I for specified envs. Call at episode boundaries."""
        if isinstance(env_ids, (np.ndarray, torch.Tensor)):
            if hasattr(env_ids, "dtype") and (
                getattr(env_ids, "dtype", None) == torch.bool
                or env_ids.dtype == np.bool_
            ):
                idx = np.where(np.asarray(env_ids))[0].tolist()
            else:
                idx = np.asarray(env_ids).tolist()
        else:
            idx = list(env_ids)

        if not idx:
            return
        eye = torch.eye(self.dim, device=self.device) / self.lam
        for i in idx:
            self.M[i] = eye

    @torch.no_grad()
    def reset_all(self) -> None:
        """Reset all envs' M. Call at training start."""
        self.M = self._fresh_M()

    def reset_diagnostics(self) -> None:
        self._cum_skipped = 0
        self._cum_reset = 0
        self._cum_b_max = -float("inf")
        self._cum_calls = 0

    def get_diagnostics(self) -> dict:
        n_steps = max(self._cum_calls, 1)
        denom = float(n_steps * self.n_envs)
        return {
            "n_skipped":  self._cum_skipped,
            "n_reset":    self._cum_reset,
            "b_max":      self._cum_b_max,
            "n_calls":    self._cum_calls,
            "skip_frac":  self._cum_skipped / denom,
            "reset_frac": self._cum_reset / denom,
        }


class RunningStd:
    """Online running std via Chan's parallel variance (Welford-style, batched).

    Used to normalize the E3B bonus so beta_ep is scale-invariant across
    different envs and phi encoders. Single-stream: all envs contribute to
    one running estimate. Non-finite inputs are dropped defensively.
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: Union[np.ndarray, torch.Tensor]) -> None:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            return

        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.size

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        M2 = (
            self.var * self.count
            + batch_var * batch_count
            + delta ** 2 * self.count * batch_count / tot_count
        )
        self.mean = float(new_mean)
        self.var = float(M2 / tot_count)
        self.count = float(tot_count)

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 1e-8)))
