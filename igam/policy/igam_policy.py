"""IGAM actor-critic policy.

Generic over any `RecurrentCell`. Per ADR 0004:
  - Encoder φ: O → ℝ^d_φ produces the input to the cell.
  - Cell consumes φ(o_t), returns (y_t, new_state, side_outputs).
  - Actor-critic head input: concat(φ(o_t), y_t).
  - No separate hidden state h_t. The cell state IS the cell's whole state.

Structurally mirrors `lmu_ppo/policies.py::LMUActorCriticPolicy` minus the
LMU-specific machinery (W_pre OrthoLayer + Cayley updates, LegS counter,
6-tuple gate/innov/u_x return, MinigridEncoder hardcoded). The forward /
evaluate_actions / predict_values triplet preserves SB3 PPO's calling
convention.

Phase A is Discrete-action only (POPGym RepeatPrevious, MiniGrid-Memory,
BSuite). Add Box-action support when Phase C moves to MineDojo / Craftax-
continuous.
"""

from __future__ import annotations

from typing import Optional

import torch as th
from gymnasium import spaces
from torch import Tensor, nn
from torch.distributions import Categorical

from igam.cell.base import RecurrentCell
from igam.policy.encoder import FlatEncoder


class IGAMActorCriticPolicy(nn.Module):
    """Encoder φ → RecurrentCell → concat(φ, y) → actor + critic.

    The cell is constructed and passed in by the trainer (`IGAMPPO`); this
    class owns it but doesn't know which cell class it is.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space:      spaces.Space,
        cell:              RecurrentCell,
        encoder_dim:       int = 64,
        encoder_hidden:    int = 128,
        lr:                float = 3e-4,
    ) -> None:
        super().__init__()
        if not isinstance(action_space, spaces.Discrete):
            raise NotImplementedError(
                "IGAMActorCriticPolicy currently supports Discrete action spaces "
                "only. Phase A benchmarks are all Discrete; add Box dispatch "
                "when needed."
            )

        self.encoder_dim = encoder_dim
        self.cell = cell

        # φ: encodes observations.
        self.encoder = FlatEncoder(
            observation_space=observation_space,
            encoder_dim=encoder_dim,
            hidden_dim=encoder_hidden,
        )

        # Head input dim = d_φ + cell.output_size. Per ADR 0004's
        # "actor-critic head receives concat(φ(o_t), y_t)".
        head_dim = encoder_dim + cell.output_size
        n_actions = action_space.n

        self.actor = nn.Linear(head_dim, n_actions)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        # 2-layer critic — matches the lmu_ppo pattern (one tanh hidden layer).
        # Wider hidden than necessary; cheap and stabilizes value learning.
        self.critic = nn.Sequential(
            nn.Linear(head_dim, head_dim // 2),
            nn.Tanh(),
            nn.Linear(head_dim // 2, 1),
        )
        nn.init.orthogonal_(self.critic[0].weight, gain=1.0)
        nn.init.zeros_(self.critic[0].bias)
        nn.init.orthogonal_(self.critic[2].weight, gain=1.0)
        nn.init.zeros_(self.critic[2].bias)

        self.optimizer = th.optim.Adam(self.parameters(), lr=lr, eps=1e-5)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _head_input(phi: Tensor, y: Tensor) -> Tensor:
        """concat(φ(o_t), y_t) — actor-critic head input per ADR 0004."""
        return th.cat([phi, y], dim=-1)

    @staticmethod
    def _episode_start_mask(
        episode_start: Optional[Tensor],
        device: th.device,
    ) -> Optional[Tensor]:
        """Coerce episode-start signal to (B,) bool tensor on the right device.

        SB3 passes episode_start as numpy bool; the cell wants a torch bool tensor.
        """
        if episode_start is None:
            return None
        if isinstance(episode_start, Tensor):
            return episode_start.to(dtype=th.bool, device=device)
        return th.as_tensor(episode_start, dtype=th.bool, device=device)

    # ── rollout (single step) ──────────────────────────────────────────────

    def forward(
        self,
        obs:           Tensor,
        cell_state:    dict[str, Tensor],
        episode_start: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor], dict[str, Tensor]]:
        """Single-step actor-critic forward for rollout collection.

        Returns:
            action       : (B,) long  — argmax if deterministic else sampled
            value        : (B,)       — critic value
            log_prob     : (B,)       — log π(action | obs)
            new_state    : cell state after this step
            side_outputs : whatever the cell emits (e.g., "innovation")
        """
        phi = self.encoder(obs)
        es = self._episode_start_mask(episode_start, phi.device)
        y, new_state, side = self.cell.step(phi, cell_state, episode_start=es)
        head = self._head_input(phi, y)

        logits = self.actor(head)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(head).squeeze(-1)

        return action, value, log_prob, new_state, side

    # ── PPO update (K-step unroll with gradient) ───────────────────────────

    def evaluate_actions(
        self,
        obs_seq:        Tensor,
        state_init:     dict[str, Tensor],
        episode_starts: Tensor,
        actions_seq:    Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Re-run the cell over K timesteps with full gradient.

        Args:
            obs_seq:        (B, K, *obs_shape) — observations
            state_init:     dict of (B, *state_shape) — state at chunk start
            episode_starts: (B, K) float in {0, 1} — 1 ⇒ start of new episode
            actions_seq:    (B, K) long — actions taken at rollout time

        Returns (each flattened over B*K):
            values, log_probs, entropy
        """
        B, K = episode_starts.shape

        # Clone state to avoid mutating the buffer's stored tensors when the
        # cell does in-graph operations. The cell itself returns NEW tensors
        # via apply_episode_mask + step, so cloning is defensive only.
        state = {k: v.clone() for k, v in state_init.items()}

        all_values:    list[Tensor] = []
        all_log_probs: list[Tensor] = []
        all_entropy:   list[Tensor] = []

        for k_step in range(K):
            obs_k = obs_seq[:, k_step]
            phi = self.encoder(obs_k)                                    # (B, d_φ)
            es_k = episode_starts[:, k_step].bool()
            # Cell handles its own episode-start reset internally — we just
            # pass the mask.
            y, state, _ = self.cell.step(phi, state, episode_start=es_k)
            head = self._head_input(phi, y)

            logits = self.actor(head)
            dist = Categorical(logits=logits)
            all_log_probs.append(dist.log_prob(actions_seq[:, k_step]))
            all_entropy.append(dist.entropy())
            all_values.append(self.critic(head).squeeze(-1))

        values    = th.stack(all_values,    dim=1).reshape(B * K)
        log_probs = th.stack(all_log_probs, dim=1).reshape(B * K)
        entropy   = th.stack(all_entropy,   dim=1).reshape(B * K)
        return values, log_probs, entropy

    # ── GAE bootstrap ──────────────────────────────────────────────────────

    def predict_values(
        self,
        obs:           Tensor,
        cell_state:    dict[str, Tensor],
        episode_start: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute V(s_T) for the GAE bootstrap at the rollout boundary."""
        phi = self.encoder(obs)
        es = self._episode_start_mask(episode_start, phi.device)
        y, _, _ = self.cell.step(phi, cell_state, episode_start=es)
        head = self._head_input(phi, y)
        return self.critic(head).squeeze(-1)

    # ── state lifecycle ────────────────────────────────────────────────────

    def initial_state(
        self,
        n_envs: int,
        device: th.device,
    ) -> dict[str, Tensor]:
        return self.cell.init_state(n_envs, device=device)

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)
