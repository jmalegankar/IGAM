"""Actor-critic policy with SEPARATE actor and critic backbones.

Per Ni et al. 2022 ("Recurrent Model-Free RL Can Be a Strong Baseline for
Many POMDPs"), sharing a recurrent encoder between actor and critic causes
the critic-loss gradient to dominate the actor-loss gradient, corrupting
the representation. The 2026-05-11 overnight ablation confirmed this on
POPGym-RepeatPrevious-Easy: stop-gradient on the critic input alone gave
a ~2.4× improvement at 1M steps. Going further — fully separate backbones
(this file) — eliminates the gradient interaction by construction.

Architecture (per ADR 0004 + Ni 2022):
  - encoder_actor:  φ_a: O → ℝ^d_φ
  - cell_actor:     RecurrentCell over φ_a
  - actor head:     Linear over concat(φ_a, y_a)

  - encoder_critic: φ_c: O → ℝ^d_φ    (independent weights)
  - cell_critic:    RecurrentCell over φ_c    (independent weights)
  - critic head:    MLP over concat(φ_c, y_c)

State storage: the cell state is namespaced into `actor_*` and `critic_*`
keys, so a single state dict produced by `init_state` covers both. The
buffer doesn't need to know about the separation; it sees a flat dict
with double the keys.

Side outputs (e.g., `innovation`) flow from the ACTOR cell only. For
Phase B's lifelong intrinsic reward we want δ_t from the cell that
actually shapes behavior; the critic cell's innovation is downstream and
not policy-relevant.
"""

from __future__ import annotations

from typing import Optional

import torch as th
from gymnasium import spaces
from torch import Tensor, nn
from torch.distributions import Categorical

from memrl.cell.base import RecurrentCell
from memrl.policy.encoder import FlatEncoder


ACTOR_PREFIX = "actor_"
CRITIC_PREFIX = "critic_"


def _split_state(state: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Split a namespaced state dict into (actor_state, critic_state)."""
    actor_state = {
        k[len(ACTOR_PREFIX):]: v for k, v in state.items() if k.startswith(ACTOR_PREFIX)
    }
    critic_state = {
        k[len(CRITIC_PREFIX):]: v for k, v in state.items() if k.startswith(CRITIC_PREFIX)
    }
    return actor_state, critic_state


def _merge_state(
    actor_state: dict[str, Tensor],
    critic_state: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Combine actor/critic states into one namespaced dict."""
    return {
        **{f"{ACTOR_PREFIX}{k}": v for k, v in actor_state.items()},
        **{f"{CRITIC_PREFIX}{k}": v for k, v in critic_state.items()},
    }


class MemActorCriticPolicy(nn.Module):
    """Actor-critic with separate recurrent backbones (Ni 2022 / SB3-RecurrentPPO default)."""

    def __init__(
        self,
        observation_space:  spaces.Space,
        action_space:       spaces.Space,
        cell_actor:         RecurrentCell,
        cell_critic:        Optional[RecurrentCell] = None,
        encoder_dim:        int = 64,
        encoder_hidden:     int = 128,
        lr:                 float = 3e-4,
        shared_backbones:   bool = False,
    ) -> None:
        """
        Args:
            shared_backbones: If True, actor and critic share encoder+cell (Option B+).
                The critic input is detached so its gradient does NOT flow into the
                shared backbone (Ni 2022 stop-gradient pattern). 1× compute, fast
                sample efficiency. Best for Tier 1 ablations.
                If False (default), separate per-component backbones (Option A,
                SB3-RecurrentPPO default). 2× compute, slower early, marginally
                better late-game ceiling. Best for final headline numbers.
        """
        super().__init__()
        # Action-space resolution. Both Discrete and MultiDiscrete are supported:
        # Discrete(n)               -> _action_dims = [n],      _is_multi_discrete=False
        # MultiDiscrete([a, b, ..]) -> _action_dims = [a, b, ..], _is_multi_discrete=True
        # The actor outputs sum(_action_dims) logits; sampling / log_prob / entropy
        # treat each dim as an independent Categorical and sum log-probs (per-dim
        # independence is the standard MultiCategorical assumption, matching SB3).
        if isinstance(action_space, spaces.Discrete):
            self._action_dims = [int(action_space.n)]
            self._is_multi_discrete = False
        elif isinstance(action_space, spaces.MultiDiscrete):
            self._action_dims = [int(n) for n in action_space.nvec]
            self._is_multi_discrete = True
        else:
            raise NotImplementedError(
                "MemActorCriticPolicy supports Discrete and MultiDiscrete action "
                f"spaces only; got {type(action_space).__name__}."
            )

        self.shared_backbones = shared_backbones
        self.encoder_dim = encoder_dim

        if shared_backbones:
            # One encoder, one cell, both used by actor AND critic. The
            # critic's input is detached at the head, so critic loss does
            # NOT update the shared encoder/cell.
            self.encoder = FlatEncoder(
                observation_space=observation_space,
                encoder_dim=encoder_dim,
                hidden_dim=encoder_hidden,
            )
            self.cell = cell_actor
            # Aliases so per-component grad-norm logging still works.
            self.encoder_actor = self.encoder
            self.encoder_critic = self.encoder
            self.cell_actor = self.cell
            self.cell_critic = self.cell
            output_size_for_head = cell_actor.output_size
        else:
            # Two independent backbones (Option A).
            if cell_critic is None:
                raise ValueError(
                    "cell_critic must be provided when shared_backbones=False."
                )
            if cell_actor is cell_critic:
                raise ValueError(
                    "cell_actor and cell_critic must be distinct when "
                    "shared_backbones=False. Use shared_backbones=True if you "
                    "want one shared cell."
                )
            if cell_actor.output_size != cell_critic.output_size:
                raise ValueError(
                    f"cell_actor.output_size ({cell_actor.output_size}) must equal "
                    f"cell_critic.output_size ({cell_critic.output_size})."
                )
            self.cell_actor = cell_actor
            self.cell_critic = cell_critic
            self.encoder_actor = FlatEncoder(
                observation_space=observation_space,
                encoder_dim=encoder_dim,
                hidden_dim=encoder_hidden,
            )
            self.encoder_critic = FlatEncoder(
                observation_space=observation_space,
                encoder_dim=encoder_dim,
                hidden_dim=encoder_hidden,
            )
            output_size_for_head = cell_actor.output_size

        head_dim = encoder_dim + output_size_for_head

        # Actor head — single linear over concat(φ_a, y_a). For MultiDiscrete
        # we emit sum(_action_dims) logits in one tensor; downstream helpers
        # split them by dim and build per-dim Categoricals.
        self.actor = nn.Linear(head_dim, sum(self._action_dims))
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        # Critic head — pre-activation LayerNorm + GELU. Replaces the older
        # Tanh-Tanh stack which empirically suffered saturation when the
        # encoder/cell produce large feature magnitudes (the actor cell's
        # state W grew to >70 in the 2026-05-11 tuned run, pushing tanh into
        # the saturated regime; gradient vanished, critic collapsed).
        #
        # Pre-norm (LayerNorm BEFORE linear, then GELU) is the modern
        # canonical fix: LN keeps pre-activations bounded, GELU's smooth
        # nonlinearity has no saturation on the positive side. References:
        #   - Lyle et al. 2024 NeurIPS "Normalization and effective learning
        #     rates in RL" — pre-norm reduces drift, induces implicit LR sched
        #   - Gogianu et al. 2021 NeurIPS "Towards Deeper DRL with Spectral
        #     Normalization" — bounded Lipschitz prevents critic instability
        #   - Henderson et al. 2018 "Deep RL that Matters" — ReLU-family
        #     activations outperform tanh for deep RL hidden layers
        #
        # Init: orthogonal gain=1.0 on hidden layers; SMALL gain=0.01 on
        # the output linear so initial values are near zero (matches the
        # SB3 PPO actor convention; prevents value explosion at t=0).
        self.critic = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim),
            nn.GELU(),
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim // 2),
            nn.GELU(),
            nn.Linear(head_dim // 2, 1),
        )
        # All linears use gain=1.0. The "small gain on output" trick is for
        # POLICY heads (prevents extreme initial actions); for value heads
        # gain=1.0 is the SB3 PPO standard. A small gain on the output
        # linear shrinks the upstream gradient by the same factor, which
        # starves the encoder/cell backbone of learning signal — empirically
        # verified via synthetic gradient-flow probe.
        for mod in self.critic:
            if isinstance(mod, nn.Linear):
                nn.init.orthogonal_(mod.weight, gain=1.0)
                nn.init.zeros_(mod.bias)

        self.optimizer = th.optim.Adam(self.parameters(), lr=lr, eps=1e-5)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _episode_start_mask(
        episode_start: Optional[Tensor],
        device: th.device,
    ) -> Optional[Tensor]:
        if episode_start is None:
            return None
        if isinstance(episode_start, Tensor):
            return episode_start.to(dtype=th.bool, device=device)
        return th.as_tensor(episode_start, dtype=th.bool, device=device)

    # ── action distribution helpers (Discrete + MultiDiscrete) ─────────────

    def _build_dists(self, logits: Tensor) -> list[Categorical]:
        """Split policy logits by per-dim action sizes; return one Categorical per dim.

        For Discrete this is a 1-element list. For MultiDiscrete it's len(nvec)
        Categoricals, treated as independent.
        """
        if self._is_multi_discrete:
            return [Categorical(logits=part) for part in th.split(logits, self._action_dims, dim=-1)]
        return [Categorical(logits=logits)]

    def _sample_action(
        self, dists: list[Categorical], deterministic: bool,
    ) -> Tensor:
        """Sample (or argmax) an action from a list of per-dim Categoricals.

        Returns shape (..., len(dists)) for MultiDiscrete or (...,) for Discrete.
        """
        per_dim = [
            d.logits.argmax(dim=-1) if deterministic else d.sample()
            for d in dists
        ]
        if self._is_multi_discrete:
            return th.stack(per_dim, dim=-1)
        return per_dim[0]

    def _log_prob(self, dists: list[Categorical], actions: Tensor) -> Tensor:
        """Sum per-dim log-probs under the independence assumption.

        For Discrete: `actions` is (...,) long, returns (...,).
        For MultiDiscrete: `actions` is (..., len(dists)) long, returns (...,).
        """
        if self._is_multi_discrete:
            return sum(d.log_prob(actions[..., i]) for i, d in enumerate(dists))
        return dists[0].log_prob(actions)

    def _entropy(self, dists: list[Categorical]) -> Tensor:
        """Sum per-dim entropies (independence)."""
        if self._is_multi_discrete:
            return sum(d.entropy() for d in dists)
        return dists[0].entropy()

    # ── rollout (single step) ──────────────────────────────────────────────

    def forward(
        self,
        obs:           Tensor,
        cell_state:    dict[str, Tensor],
        episode_start: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor], dict[str, Tensor]]:
        """One env step.

        In shared_backbones mode: one encoder + cell forward, critic head sees
        a detached copy of (φ, y) — Ni 2022's stop-gradient pattern.
        In separate-backbones mode: two independent forward passes.

        Returns:
            action       : (B,) long
            value        : (B,)
            log_prob     : (B,)
            new_state    : namespaced dict (actor_*  in shared mode, actor_*+critic_*  in separate mode)
            side_outputs : from the (shared / actor) cell
        """
        actor_state, critic_state = _split_state(cell_state)
        es = self._episode_start_mask(episode_start, next(self.parameters()).device)

        if self.shared_backbones:
            # Single forward through the shared encoder + cell.
            phi = self.encoder(obs)
            y, new_actor_state, side = self.cell.step(phi, actor_state, episode_start=es)

            # Actor — full gradient through encoder + cell.
            actor_head = th.cat([phi, y], dim=-1)
            # Critic — detached, so critic loss does NOT propagate into shared backbone.
            critic_head = th.cat([phi.detach(), y.detach()], dim=-1)

            new_critic_state = {}  # empty — no separate critic state in shared mode
        else:
            # Independent paths.
            phi_a = self.encoder_actor(obs)
            y_a, new_actor_state, side = self.cell_actor.step(phi_a, actor_state, episode_start=es)
            actor_head = th.cat([phi_a, y_a], dim=-1)

            phi_c = self.encoder_critic(obs)
            y_c, new_critic_state, _ = self.cell_critic.step(phi_c, critic_state, episode_start=es)
            critic_head = th.cat([phi_c, y_c], dim=-1)

        logits = self.actor(actor_head)
        dists = self._build_dists(logits)
        action = self._sample_action(dists, deterministic=deterministic)
        log_prob = self._log_prob(dists, action)
        value = self.critic(critic_head).squeeze(-1)

        return action, value, log_prob, _merge_state(new_actor_state, new_critic_state), side

    # ── PPO update (K-step unroll with gradient) ───────────────────────────

    def evaluate_actions(
        self,
        obs_seq:        Tensor,
        state_init:     dict[str, Tensor],
        episode_starts: Tensor,
        actions_seq:    Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Re-run the backbone(s) over K timesteps with full gradient.

        In shared mode: one forward per step, critic head detached.
        In separate mode: two independent forwards per step.
        """
        B, K = episode_starts.shape

        actor_state_init, critic_state_init = _split_state(state_init)
        actor_state = {k: v.clone() for k, v in actor_state_init.items()}
        critic_state = {k: v.clone() for k, v in critic_state_init.items()}

        all_values:    list[Tensor] = []
        all_log_probs: list[Tensor] = []
        all_entropy:   list[Tensor] = []

        for k_step in range(K):
            obs_k = obs_seq[:, k_step]
            es_k = episode_starts[:, k_step].bool()

            if self.shared_backbones:
                phi = self.encoder(obs_k)
                y, actor_state, _ = self.cell.step(phi, actor_state, episode_start=es_k)
                actor_head = th.cat([phi, y], dim=-1)
                critic_head = th.cat([phi.detach(), y.detach()], dim=-1)
            else:
                phi_a = self.encoder_actor(obs_k)
                y_a, actor_state, _ = self.cell_actor.step(phi_a, actor_state, episode_start=es_k)
                actor_head = th.cat([phi_a, y_a], dim=-1)

                phi_c = self.encoder_critic(obs_k)
                y_c, critic_state, _ = self.cell_critic.step(phi_c, critic_state, episode_start=es_k)
                critic_head = th.cat([phi_c, y_c], dim=-1)

            logits = self.actor(actor_head)
            dists = self._build_dists(logits)
            all_log_probs.append(self._log_prob(dists, actions_seq[:, k_step]))
            all_entropy.append(self._entropy(dists))
            all_values.append(self.critic(critic_head).squeeze(-1))

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
        """V(s_T) for the GAE bootstrap."""
        actor_state, critic_state = _split_state(cell_state)
        es = self._episode_start_mask(episode_start, next(self.parameters()).device)

        if self.shared_backbones:
            phi = self.encoder(obs)
            y, _, _ = self.cell.step(phi, actor_state, episode_start=es)
            critic_head = th.cat([phi.detach(), y.detach()], dim=-1)
        else:
            phi_c = self.encoder_critic(obs)
            y_c, _, _ = self.cell_critic.step(phi_c, critic_state, episode_start=es)
            critic_head = th.cat([phi_c, y_c], dim=-1)
        return self.critic(critic_head).squeeze(-1)

    # ── state lifecycle ────────────────────────────────────────────────────

    def initial_state(
        self,
        n_envs: int,
        device: th.device,
    ) -> dict[str, Tensor]:
        if self.shared_backbones:
            # One state object — shared cell. The critic side of the state
            # dict is empty in shared mode.
            return _merge_state(
                self.cell.init_state(n_envs, device=device),
                {},
            )
        return _merge_state(
            self.cell_actor.init_state(n_envs, device=device),
            self.cell_critic.init_state(n_envs, device=device),
        )

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)
