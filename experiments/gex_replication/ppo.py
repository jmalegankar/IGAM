"""MemPPO extended with optional E3B episodic exploration bonus.

Drop-in replacement for memrl/ppo/ppo.py for the GEX-replication experiments.
Adds three constructor params:

    beta_ep     (float, default 0.0)  — E3B bonus scale. 0 = disabled (pure PPO).
    phi_source  (str,   default 'random_encoder') — currently only option;
                frozen random FlatEncoder → phi ∈ R^encoder_dim.
    lambda_reg  (float, default 1.0)  — E3B regularization λ.

When beta_ep > 0, collect_rollouts:
  1. Resets E3B matrix for envs starting a new episode.
  2. Computes phi = frozen_encoder(obs).
  3. Computes Mahalanobis bonus b_t and updates M via Sherman-Morrison.
  4. Normalizes b_t via RunningStd (so beta_ep is scale-invariant).
  5. Adds beta_ep * b_norm * (1 - episode_start) to rewards before buffer.add().

Everything else (TBPTT, per-component grad logging, state-norm logging) is
identical to the base MemPPO.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    obs_as_tensor,
)

from memrl.cell.base import RecurrentCell
from memrl.exploration.e3b import EllipticalEpisodicBonus, RunningStd
from memrl.policy.encoder import FlatEncoder
from .buffer import MemRolloutBuffer
from .policy import MemActorCriticPolicy


CellFactory = Callable[[int], RecurrentCell]


class MemPPO(PPO):
    """PPO + recurrent cell + optional E3B episodic bonus."""

    policy: MemActorCriticPolicy
    rollout_buffer: MemRolloutBuffer

    def __init__(
        self,
        env: GymEnv,
        cell_factory: CellFactory,
        lr: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Optional[float] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        encoder_dim: int = 64,
        encoder_hidden: int = 128,
        shared_backbones: bool = False,
        chunk_len: int = 16,
        n_chunks_per_batch: int = 16,
        # ── E3B params ──────────────────────────────────────────────────────
        beta_ep: float = 0.0,
        phi_source: str = "random_encoder",
        lambda_reg: float = 1.0,
        # ────────────────────────────────────────────────────────────────────
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        self.cell_factory = cell_factory
        self.encoder_dim = encoder_dim
        self.encoder_hidden = encoder_hidden
        self.shared_backbones = shared_backbones
        self.chunk_len = chunk_len
        self.n_chunks_per_batch = n_chunks_per_batch
        self.beta_ep = beta_ep
        self.phi_source = phi_source
        self.lambda_reg = lambda_reg

        super().__init__(
            policy="MlpPolicy",
            env=env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=n_chunks_per_batch * chunk_len,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )
        if _init_setup_model:
            self._setup_model()

    # ── setup ──────────────────────────────────────────────────────────────

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        cell_actor = self.cell_factory(self.encoder_dim).to(self.device)
        cell_critic = (
            None if self.shared_backbones
            else self.cell_factory(self.encoder_dim).to(self.device)
        )

        self.policy = MemActorCriticPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            cell_actor=cell_actor,
            cell_critic=cell_critic,
            encoder_dim=self.encoder_dim,
            encoder_hidden=self.encoder_hidden,
            lr=self.learning_rate if isinstance(self.learning_rate, float)
               else self.learning_rate(1.0),
            shared_backbones=self.shared_backbones,
        ).to(self.device)

        with th.no_grad():
            sample_state = self.policy.initial_state(1, th.device("cpu"))
            state_shapes = {k: tuple(v.shape[1:]) for k, v in sample_state.items()}

        self.rollout_buffer = MemRolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            state_shapes=state_shapes,
            chunk_len=self.chunk_len,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        self._cell_state: Optional[dict[str, th.Tensor]] = None

        _VALID_PHI_SOURCES = {"random_encoder", "encoder_detached", "innovation"}
        if self.beta_ep > 0 and self.phi_source not in _VALID_PHI_SOURCES:
            raise ValueError(
                f"Unknown phi_source {self.phi_source!r}. "
                f"Valid options: {sorted(_VALID_PHI_SOURCES)}"
            )

        # Frozen random encoder — only needed for phi_source='random_encoder'.
        # encoder_detached reuses policy.encoder_actor (no extra module).
        # innovation comes from cell side_outputs (no extra module).
        if self.beta_ep > 0 and self.phi_source == "random_encoder":
            self._phi_encoder = FlatEncoder(
                observation_space=self.observation_space,
                encoder_dim=self.encoder_dim,
                hidden_dim=self.encoder_hidden,
            ).to(self.device)
            for p in self._phi_encoder.parameters():
                p.requires_grad_(False)
        else:
            self._phi_encoder = None

    def _setup_learn(
        self,
        total_timesteps: int,
        callback=None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "ppo",
        progress_bar: bool = False,
    ):
        ret = super()._setup_learn(
            total_timesteps, callback, reset_num_timesteps,
            tb_log_name, progress_bar,
        )
        self._cell_state = self.policy.initial_state(self.n_envs, self.device)

        # innovation phi requires the actor cell to emit "innovation" in side_outputs.
        # GatedLMU does; vanilla LMU does not. Catch the mismatch early.
        if self.beta_ep > 0 and self.phi_source == "innovation":
            probe_state = self.policy.initial_state(1, self.device)
            dummy_obs = th.zeros(
                (1, *self.observation_space.shape), device=self.device
            )
            with th.no_grad():
                _, _, _, _, probe_side = self.policy.forward(dummy_obs, probe_state)
            if "innovation" not in probe_side:
                raise ValueError(
                    "phi_source='innovation' requires the actor cell to emit "
                    "'innovation' in side_outputs. "
                    f"{type(self.policy.cell_actor).__name__} does not."
                )

        if self.beta_ep > 0:
            self._e3b = EllipticalEpisodicBonus(
                n_envs=self.n_envs,
                dim=self.encoder_dim,
                lambda_reg=self.lambda_reg,
                device=self.device,
            )
            self._running_std_e3b = RunningStd()
            self._e3b.reset_all()

        if self.verbose >= 1:
            cell_name = type(self.policy.cell_actor).__name__
            n_params = sum(p.numel() for p in self.policy.parameters())
            e3b_str = f"  beta_ep={self.beta_ep}" if self.beta_ep > 0 else ""
            print(
                f"  cell={cell_name}  params={n_params:,}"
                f"  chunk_len={self.chunk_len}"
                f"  n_chunks_per_batch={self.n_chunks_per_batch}"
                f"{e3b_str}"
            )
        return ret

    # ── rollout collection ─────────────────────────────────────────────────

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()

        if self.beta_ep > 0:
            self._e3b.reset_diagnostics()

        innovation_buf: list[float] = []
        state_norm_buf: list[dict[str, float]] = []
        action_dist_buf: list[np.ndarray] = []
        e3b_bonus_buf: list[float] = []

        n_steps = 0
        while n_steps < n_rollout_steps:
            # ── E3B episode reset — BEFORE policy.forward ──────────────────
            # _last_episode_starts[i]=True: env i just reset, current obs is
            # the first of a new episode. Reset E3B matrix for those envs so
            # coverage is measured within-episode, not across episodes.
            if self.beta_ep > 0:
                ep_starts = self._last_episode_starts          # (n_envs,) bool
                done_ids = np.where(ep_starts)[0]
                if len(done_ids) > 0:
                    self._e3b.reset(done_ids)

            with th.no_grad():
                obs_t = obs_as_tensor(self._last_obs, self.device)
                es = th.as_tensor(
                    self._last_episode_starts, dtype=th.bool, device=self.device,
                )
                actions, values, log_probs, new_state, side = self.policy.forward(
                    obs_t, self._cell_state, episode_start=es,
                )

                # ── phi computation — all sources available post-forward ───
                if self.beta_ep > 0:
                    if self.phi_source == "random_encoder":
                        phi = self._phi_encoder(obs_t).detach()
                    elif self.phi_source == "encoder_detached":
                        phi = self.policy.encoder_actor(obs_t).detach()
                    else:  # innovation
                        phi = side["innovation"].detach()

                if "innovation" in side:
                    innovation_buf.append(side["innovation"].abs().mean().item())
                state_norm_buf.append({k: v.norm().item() for k, v in new_state.items()})

            # ── E3B bonus — outside no_grad, operates on already-detached phi
            if self.beta_ep > 0:
                b_raw = self._e3b.bonus_and_update(phi)        # (n_envs,)
                self._running_std_e3b.update(b_raw)
                b_norm = (b_raw / self._running_std_e3b.std).cpu().numpy()
                # Zero on episode-start steps: E3B just reset so first obs
                # always produces a large bonus regardless of actual novelty.
                e3b_bonus = self.beta_ep * b_norm * (1.0 - ep_starts.astype(np.float32))
                e3b_bonus_buf.append(float(e3b_bonus.mean()))

            actions_np = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions_np)
            self.num_timesteps += env.num_envs

            # Add E3B bonus to environment reward.
            if self.beta_ep > 0:
                rewards = rewards + e3b_bonus

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            action_dist_buf.append(actions_np.copy())

            if isinstance(self.action_space, spaces.Discrete):
                actions_to_store = actions_np.reshape(-1, 1)
            else:
                actions_to_store = actions_np

            rollout_buffer.add(
                self._last_obs,
                actions_to_store,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                self._cell_state,
            )

            self._cell_state = {k: v.clone() for k, v in new_state.items()}
            self._last_obs = new_obs
            self._last_episode_starts = dones
            n_steps += 1

        # GAE bootstrap.
        with th.no_grad():
            obs_t = obs_as_tensor(new_obs, self.device)
            es = th.as_tensor(
                self._last_episode_starts, dtype=th.bool, device=self.device,
            )
            values = self.policy.predict_values(obs_t, self._cell_state, episode_start=es)

        rollout_buffer.compute_returns_and_advantage(values, dones)

        # ── per-rollout diagnostics ────────────────────────────────────────
        if innovation_buf:
            self.logger.record("debug/innovation_mag_mean", float(np.mean(innovation_buf)))
            self.logger.record("debug/innovation_mag_max",  float(np.max(innovation_buf)))

        for key in state_norm_buf[0]:
            vals = [s[key] for s in state_norm_buf]
            self.logger.record(f"debug/state_{key}_norm_mean", float(np.mean(vals)))
            self.logger.record(f"debug/state_{key}_norm_max",  float(np.max(vals)))

        if isinstance(self.action_space, spaces.Discrete):
            all_actions = np.concatenate(action_dist_buf)
            for a in range(self.action_space.n):
                self.logger.record(f"debug/action_frac_{a}", float((all_actions == a).mean()))

        if self.beta_ep > 0:
            self.logger.record("e3b/bonus_mean",   float(np.mean(e3b_bonus_buf)))
            self.logger.record("e3b/bonus_max",    float(np.max(e3b_bonus_buf)))
            self.logger.record("e3b/running_std",  self._running_std_e3b.std)
            diag = self._e3b.get_diagnostics()
            self.logger.record("e3b/skip_frac",    diag["skip_frac"])
            self.logger.record("e3b/reset_frac",   diag["reset_frac"])
            self.logger.record("e3b/b_max_raw",    diag["b_max"])

        callback.on_rollout_end()
        return True

    # ── eval / predict ─────────────────────────────────────────────────────

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self.policy.set_training_mode(False)
        obs_tensor = obs_as_tensor(observation, self.device)

        if isinstance(observation, dict):
            n = next(iter(observation.values())).shape[0]
        else:
            n = observation.shape[0]

        cell_state = self.policy.initial_state(n, self.device) if state is None else state

        with th.no_grad():
            actions, _, _, new_state, _ = self.policy.forward(
                obs_tensor, cell_state,
                episode_start=episode_start,
                deterministic=deterministic,
            )

        return actions.cpu().numpy(), new_state

    # ── PPO update ─────────────────────────────────────────────────────────

    def train(self) -> None:
        self.policy.set_training_mode(True)

        lr = self.lr_schedule(self._current_progress_remaining)
        for pg in self.policy.optimizer.param_groups:
            pg["lr"] = lr
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        pg_losses:      list[float] = []
        value_losses:   list[float] = []
        entropy_losses: list[float] = []
        clip_fractions: list[float] = []
        approx_kl_divs: list[float] = []
        grad_norms:     list[float] = []
        comp_grad_norms: dict[str, list[float]] = defaultdict(list)

        continue_training = True
        for epoch in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.n_chunks_per_batch):
                values, log_prob, entropy = self.policy.evaluate_actions(
                    obs_seq=batch.observations,
                    state_init=batch.cell_state,
                    episode_starts=batch.episode_starts,
                    actions_seq=batch.actions,
                )

                advantages = batch.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = th.exp(log_prob - batch.old_log_prob)
                policy_loss = -th.min(
                    advantages * ratio,
                    advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
                ).mean()

                values_pred = (
                    values if self.clip_range_vf is None
                    else batch.old_values + th.clamp(
                        values - batch.old_values, -clip_range_vf, clip_range_vf,
                    )
                )
                value_loss = F.mse_loss(batch.returns, values_pred)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - batch.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"  early stop epoch {epoch}, KL={approx_kl_div:.3f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()

                per_comp_max = 0.0
                for name, mod in [
                    ("encoder_actor",  self.policy.encoder_actor),
                    ("encoder_critic", self.policy.encoder_critic),
                    ("cell_actor",     self.policy.cell_actor),
                    ("cell_critic",    self.policy.cell_critic),
                    ("actor",          self.policy.actor),
                    ("critic",         self.policy.critic),
                ]:
                    norm = th.nn.utils.clip_grad_norm_(
                        mod.parameters(), self.max_grad_norm,
                    ).item()
                    comp_grad_norms[name].append(norm)
                    per_comp_max = max(per_comp_max, norm)
                grad_norms.append(per_comp_max)

                self.policy.optimizer.step()

                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )
                self._n_updates += 1

            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        self.logger.record("train/policy_loss",        float(np.mean(pg_losses)))
        self.logger.record("train/value_loss",         float(np.mean(value_losses)))
        self.logger.record("train/entropy_loss",       float(np.mean(entropy_losses)))
        self.logger.record("train/approx_kl",          float(np.mean(approx_kl_divs)))
        self.logger.record("train/clip_fraction",      float(np.mean(clip_fractions)))
        self.logger.record("train/grad_norm",          float(np.mean(grad_norms)))
        self.logger.record("train/explained_variance", float(explained_var))
        self.logger.record("train/n_updates",          self._n_updates, exclude="tensorboard")
        self.logger.record("train/learning_rate",      lr)
        self.logger.record("train/clip_range",         clip_range)

        for name, norms in comp_grad_norms.items():
            if norms:
                self.logger.record(f"grad/{name}_norm", float(np.mean(norms)))
