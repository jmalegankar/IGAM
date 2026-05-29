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
        elif isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 3:
            # Image obs (H, W, C), channels-last — e.g. MiniGrid's one-hot grid
            # (7, 7, 20). Standard MiniGrid conv stack: three 2×2 convs (with a
            # MaxPool after the first) shrink the 7×7 view to 1×1×64, which a
            # Linear then maps to encoder_dim. Kernel sizes assume the default
            # 7×7 partial view; the flatten dim is computed from a dummy pass so
            # other view sizes still work as long as they don't underflow.
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
