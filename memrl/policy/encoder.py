"""Encoder: φ: O → ℝ^d_φ.

Dispatches on the observation-space type:
  - Discrete: nn.Embedding → MLP. The "tokens" are scalar integers.
  - Box(low-dim symbolic): Linear → MLP. Flattens to (B, prod(shape)).
  - Dict: not supported yet (Phase A POPGym is single-observation). Add when needed.

CNN encoder for image observations is deliberately deferred — the README says
"CNN for image obs, MLP for symbolic" and Phase A starts on POPGym (symbolic).
The CNN port from `lmu_ppo/policies.py::MinigridEncoder` lands when Phase A
moves to MiniGrid-Memory-S13.

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
            self.is_discrete = True
            # Embedding gives a hidden_dim vector per discrete token; the
            # following MLP collapses to encoder_dim.
            self.embed = nn.Embedding(observation_space.n, hidden_dim)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, encoder_dim),
            )
        elif isinstance(observation_space, gym.spaces.Box):
            self.is_discrete = False
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
        # Orthogonal init for ReLU MLPs (matches the lmu_ppo encoder).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: Tensor) -> Tensor:
        """obs → (B, encoder_dim).

        Discrete: obs is (B,) or (B, 1) long → embedding lookup → MLP.
        Box: obs is (B, *shape) float → flatten → MLP.
        """
        if self.is_discrete:
            # Squeeze any trailing singleton dims (SB3 stores Discrete obs as
            # (B, 1) sometimes; we just want (B,) for embedding lookup).
            obs_long = obs.long().reshape(obs.shape[0])
            return self.mlp(self.embed(obs_long))
        else:
            obs_flat = obs.float().reshape(obs.shape[0], -1)
            return self.mlp(obs_flat)
