"""Encoder: φ: O → ℝ^d_φ.

Dispatches on the observation-space type:
  - Discrete: nn.Embedding → MLP. The "tokens" are scalar integers.
  - Box, 3-D (image, channels-last H×W×C): small CNN → Linear. MiniGrid's
    one-hot grid (7,7,20) and any pixel/grid obs. A conv stack exploits the
    egocentric grid's spatial locality that a flatten-MLP throws away, and
    costs ~20× fewer params than flattening 7*7*20=980 into a Linear (the
    flatten-MLP encoder was 70% of the whole policy, dwarfing the cell that
    the comparison is actually about).
  - Box, other (low-dim symbolic): Linear → MLP. Flattens to (B, prod(shape)).
  - Dict: not supported yet (Phase A POPGym is single-observation). Add when needed.

The CNN path follows the README's "CNN for image obs, MLP for symbolic" and
ports `lmu_ppo/policies.py::MinigridEncoder` — the standard MiniGrid conv stack
that reduces the 7×7 partial view to a 64-d feature before the encoder_dim
projection.

Note: this module is what ADR 0004 calls "φ" — its output is the source of
both the QKV projections AND the data-dependent gates inside any cell. Keep
it intentionally simple so the cell's behavior dominates ablation results.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn


class FlatEncoder(nn.Module):
    """Generic MLP encoder for Discrete or Box observation spaces."""

    def __init__(
        self,
        observation_space: gym.Space,
        encoder_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim

        if isinstance(observation_space, gym.spaces.Discrete):
            self.obs_mode = "discrete"
            # Embedding gives a hidden_dim vector per discrete token; the
            # following MLP collapses to encoder_dim.
            self.embed = nn.Embedding(observation_space.n, hidden_dim)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, encoder_dim),
            )
        elif (isinstance(observation_space, gym.spaces.Box)
              and len(observation_space.shape) == 3
              and min(observation_space.shape[0], observation_space.shape[1]) >= 7):
            # Image obs (H, W, C), channels-last — e.g. MiniGrid's one-hot grid
            # (7, 7, 20). Standard MiniGrid conv stack: three 2×2 convs (with a
            # MaxPool after the first) shrink the 7×7 view to 1×1×64, which a
            # Linear then maps to encoder_dim. The 2×2-conv stack underflows below
            # 7×7, so SMALLER grids (e.g. a 3×3 reduced view from ViewSizeWrapper)
            # fall through to the flat-MLP branch instead — there's negligible
            # spatial structure to exploit at 3×3 anyway, and 3·3·20=180 is a fine
            # MLP input. The flatten dim is computed from a dummy pass.
            self.obs_mode = "image"
            h, w, c = observation_space.shape
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 16, kernel_size=2), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=2), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=2), nn.ReLU(),
            )
            with torch.no_grad():
                n_flat = self.cnn(torch.zeros(1, c, h, w)).reshape(1, -1).shape[1]
            self.mlp = nn.Linear(n_flat, encoder_dim)
        elif isinstance(observation_space, gym.spaces.Box):
            self.obs_mode = "flat"
            obs_dim = int(np.prod(observation_space.shape))
            self.mlp = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, encoder_dim),
            )
        else:
            raise NotImplementedError(
                f"FlatEncoder doesn't support {type(observation_space).__name__} "
                f"observation spaces yet. Add a dispatch branch for Dict/MultiDiscrete "
                f"when Phase A moves past POPGym."
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Orthogonal init for ReLU MLPs / convs (matches the lmu_ppo encoder).
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: Tensor) -> Tensor:
        """obs → (B, encoder_dim).

        Discrete: obs is (B,) or (B, 1) long → embedding lookup → MLP.
        Image:    obs is (B, H, W, C) float → CNN → Linear.
        Box flat: obs is (B, *shape) float → flatten → MLP.
        """
        if self.obs_mode == "discrete":
            # Squeeze any trailing singleton dims (SB3 stores Discrete obs as
            # (B, 1) sometimes; we just want (B,) for embedding lookup).
            obs_long = obs.long().reshape(obs.shape[0])
            return self.mlp(self.embed(obs_long))
        if self.obs_mode == "image":
            # (B, H, W, C) channels-last → (B, C, H, W) for Conv2d.
            feat = self.cnn(obs.float().permute(0, 3, 1, 2))
            return self.mlp(feat.reshape(obs.shape[0], -1))
        obs_flat = obs.float().reshape(obs.shape[0], -1)
        return self.mlp(obs_flat)


class PixelEncoder(nn.Module):
    """Nature-DQN CNN encoder for RGB pixel observations.

    Dedicated to pixel envs (e.g. memory-gym's MysteryPath, 84×84×3) — kept
    SEPARATE from FlatEncoder's MiniGrid path on purpose: a one-hot categorical
    grid and an RGB frame are different enough (input statistics, scale, spatial
    smoothness) that sharing a conv stack is the wrong abstraction. Strided convs
    (8/s4 → 4/s2 → 3/s1) downsample 84×84 to 7×7×64 before the encoder_dim
    projection — a ~0.5M-param encoder rather than the ~12M a no-stride stack
    would produce on inputs this large.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        encoder_dim: int = 64,
        hidden_dim: int = 128,   # unused; kept for a uniform encoder signature
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.obs_mode = "pixel"
        h, w, c = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, c, h, w)).reshape(1, -1).shape[1]
        self.mlp = nn.Linear(n_flat, encoder_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: Tensor) -> Tensor:
        # (B, H, W, C) channels-last → (B, C, H, W) for Conv2d.
        feat = self.cnn(obs.float().permute(0, 3, 1, 2))
        return self.mlp(feat.reshape(obs.shape[0], -1))


def make_encoder(
    observation_space: gym.Space,
    encoder_dim: int = 64,
    hidden_dim: int = 128,
) -> nn.Module:
    """Pick the encoder class for an observation space.

    Large RGB frames (3-D Box, min(H, W) ≥ 40 — e.g. memory-gym's 84×84×3) get
    the dedicated Nature-DQN ``PixelEncoder``; everything else (MiniGrid one-hot
    grids, symbolic Box vectors, Discrete) uses ``FlatEncoder``. The two are
    deliberately distinct classes — pixel and grid envs are not the same problem.
    """
    if (isinstance(observation_space, gym.spaces.Box)
            and len(observation_space.shape) == 3
            and min(observation_space.shape[0], observation_space.shape[1]) >= 40):
        return PixelEncoder(observation_space, encoder_dim, hidden_dim)
    return FlatEncoder(observation_space, encoder_dim, hidden_dim)
