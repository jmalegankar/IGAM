"""Feature embeddings φ for E3B (and any other ellipsoidal/density-style bonus).

The thesis (Section 8.2, "Agency Principle") established that φ must be
*agency-sensitive from initialization*: deterministic in the current obs and
discriminative under the agent's reachable position distribution. The two
that work are φ^rand (frozen random encoder) and φ^obs (the policy encoder
output u_x). The two that fail are φ^innov and φ^y (memory-derived).

We expose all four here so the runner can test the principle generalises to
the Gated DeltaNet setup, not just the gated multichannel LMU.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def _obs_to_tensor(obs: Any, device: torch.device, dtype=torch.float32) -> Tensor:
    """Coerce gym obs to (B, F) float tensor. Mirrors RND helper."""
    if isinstance(obs, dict):
        parts = [_obs_to_tensor(v, device, dtype) for v in obs.values()]
        return torch.cat(parts, dim=-1)
    arr = np.asarray(obs)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return torch.as_tensor(arr, dtype=dtype, device=device)


class PhiSource(nn.Module, ABC):
    """Abstract: encode `obs` (and optionally side outputs) to feature φ ∈ R^C."""

    @abstractmethod
    @torch.no_grad()
    def encode(
        self,
        obs: Any,
        side: dict[str, Tensor],
        cell_state: dict[str, Tensor],
    ) -> Tensor:
        """Return (n_envs, dim) detached, on self.device, finite."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Feature dimension C."""


# ──────────────────────────────────────────────────────────────────────────
# φ^rand — frozen random MLP on obs.
# The thesis's working choice. Agency-sensitive from step one.
# ──────────────────────────────────────────────────────────────────────────

class RandomPhi(PhiSource):
    """Frozen randomly-initialised MLP encoder on observation."""

    def __init__(
        self,
        obs_dim: int,
        hidden: int = 128,
        feature_dim: int = 64,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self._dim = feature_dim
        self.device = torch.device(device)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        ).to(self.device)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        for p in self.parameters():
            p.requires_grad = False

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode(self, obs, side, cell_state) -> Tensor:
        x = _obs_to_tensor(obs, self.device)
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────
# φ^obs — raw flattened observation (no encoder).
# In the thesis this was the policy encoder output u_x. We use the raw obs
# as a simpler stand-in; both are agency-compatible (deterministic in obs).
# ──────────────────────────────────────────────────────────────────────────

class ObsPhi(PhiSource):
    """Raw flattened observation as the feature."""

    def __init__(self, obs_dim: int, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self._dim = obs_dim
        self.device = torch.device(device)

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode(self, obs, side, cell_state) -> Tensor:
        return _obs_to_tensor(obs, self.device)


# ──────────────────────────────────────────────────────────────────────────
# φ^innov — cell-derived innovation (Agency-Principle counter-example).
# The thesis showed this FAILS for E3B in MiniGrid Memory because:
#   - At t=0 the cell hasn't learned to predict anything, so innovation ≈ 0
#     everywhere — no coverage gradient on the ellipsoid.
#   - By the time the cell does discriminate, the agent has already either
#     stumbled into reward or hasn't.
# We expose this to confirm the principle generalises to Gated DeltaNet's
# eps_mem signal (which is essentially the same thing).
# ──────────────────────────────────────────────────────────────────────────

class CellInnovationPhi(PhiSource):
    """Use the cell's eps_mem (or named innovation key) as a scalar feature.

    Note: this is 1-dimensional. E3B with 1-D φ degenerates to count-based
    novelty on the single scalar — but that's exactly the Agency-Principle
    test: even with a discriminative-on-paper signal, does it drive useful
    exploration? Spoiler from your thesis: no.
    """

    def __init__(
        self,
        side_key: str = "eps_mem",
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self._dim = 1
        self.device = torch.device(device)
        self.side_key = side_key

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode(self, obs, side, cell_state) -> Tensor:
        if self.side_key not in side:
            raise KeyError(
                f"PhiSource={type(self).__name__} requires cell side output "
                f"'{self.side_key}', got {list(side.keys())}"
            )
        x = side[self.side_key].detach().to(self.device).float()
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return x
