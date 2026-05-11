"""IGAM PPO trainer.

PPO with a generic recurrent policy + TBPTT-chunked rollout buffer.
Cell-agnostic: works with any `RecurrentCell` (GRU, LSTM, LMU, S4D, Mamba2,
LinearTransformer, RetNet, DeltaNet, mLSTM, GatedDeltaNet).

Architectural ancestor: `lmu_ppo/lmu_ppo.py::LMUPPO`. We strip the LMU-
specifics:
  - No W_pre OrthoLayer / Cayley updates (no IGAM cell has W_pre)
  - No LegS step counter
  - No E3B episodic bonus (Phase B work)
  - No phi_source variants (Phase B work)
  - No MiniGrid-specific diagnostics
  - No gate/innov/u_x unpacking from cell.forward

We keep:
  - Chunked TBPTT (chunk_len, n_chunks_per_batch)
  - Per-component grad-norm logging (README discipline)
  - State-norm logging per cell-state component
  - Innovation magnitude logging (when the cell emits one — DeltaNet, IGAM)
  - SB3 PPO scaffolding (callbacks, schedules, env wrappers, eval)

For Phase B (exploration), the lifelong intrinsic reward r_life = ‖δ_t‖²
hooks into this loop by reading `side_outputs["innovation"]` during
`collect_rollouts` and adding `β · r_life · (1 - episode_start)` to the
reward. Stub left in the diagnostic logging today; full Phase B reward
wiring is a separate ADR.
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

from igam.cell.base import RecurrentCell
from igam.policy.buffer import IGAMRolloutBuffer
from igam.policy.igam_policy import IGAMActorCriticPolicy


# A cell factory takes (input_size) and returns a constructed RecurrentCell.
# `input_size` will equal `encoder_dim` (the encoder's output dim).
CellFactory = Callable[[int], RecurrentCell]


class IGAMPPO(PPO):
    """PPO with an IGAM recurrent policy and TBPTT rollout buffer."""

    policy: IGAMActorCriticPolicy
    rollout_buffer: IGAMRolloutBuffer

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
        chunk_len: int = 16,
        n_chunks_per_batch: int = 16,
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        self.cell_factory = cell_factory
        self.encoder_dim = encoder_dim
        self.encoder_hidden = encoder_hidden
        self.chunk_len = chunk_len
        self.n_chunks_per_batch = n_chunks_per_batch

        # SB3 PPO wants a batch_size; we don't use it for sampling (we use
        # n_chunks_per_batch) but it has to be set to something consistent
        # with n_steps * n_envs for SB3's internal checks.
        # batch_size = n_chunks_per_batch * chunk_len gives the effective
        # number of transitions per SGD update.
        super().__init__(
            policy="MlpPolicy",   # placeholder; we replace in _setup_model
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

        # Construct the cell BEFORE the policy/buffer so we can read its
        # state shapes for the buffer.
        cell = self.cell_factory(self.encoder_dim).to(self.device)

        # Discover state shapes from the cell. init_state(batch_size=1)
        # gives (1, *state_shape) per key; the rollout buffer allocates
        # (T, n_envs, *state_shape).
        with th.no_grad():
            sample_state = cell.init_state(batch_size=1, device=th.device("cpu"))
            state_shapes = {k: tuple(v.shape[1:]) for k, v in sample_state.items()}

        self.policy = IGAMActorCriticPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            cell=cell,
            encoder_dim=self.encoder_dim,
            encoder_hidden=self.encoder_hidden,
            lr=self.learning_rate if isinstance(self.learning_rate, float)
               else self.learning_rate(1.0),
        ).to(self.device)

        self.rollout_buffer = IGAMRolloutBuffer(
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

        # Live cell state across rollout collection. Initialized in _setup_learn
        # (when n_envs is known).
        self._cell_state: Optional[dict[str, th.Tensor]] = None

    def _setup_learn(
        self,
        total_timesteps: int,
        callback=None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "igam_ppo",
        progress_bar: bool = False,
    ):
        ret = super()._setup_learn(
            total_timesteps, callback, reset_num_timesteps,
            tb_log_name, progress_bar,
        )
        self._cell_state = self.policy.initial_state(self.n_envs, self.device)
        if self.verbose >= 1:
            cell_name = type(self.policy.cell).__name__
            n_params = sum(p.numel() for p in self.policy.parameters())
            print(f"  cell={cell_name}  params={n_params:,}  "
                  f"chunk_len={self.chunk_len}  n_chunks_per_batch={self.n_chunks_per_batch}")
            print(f"  cell state shapes: " + ", ".join(
                f"{k}={tuple(v.shape[1:])}" for k, v in self._cell_state.items()
            ))
        return ret

    # ── rollout collection ─────────────────────────────────────────────────

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()

        # Per-step diagnostic accumulators.
        innovation_buf: list[float] = []
        state_norm_buf: list[dict[str, float]] = []
        action_dist_buf: list[np.ndarray] = []

        n_steps = 0
        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_t = obs_as_tensor(self._last_obs, self.device)
                # Pass episode_start so the cell zeros state for any envs
                # whose current obs is the first of a new episode. This is
                # the canonical place for episode-boundary handling — the
                # cell's apply_episode_mask helper does it inline.
                es = th.as_tensor(
                    self._last_episode_starts, dtype=th.bool, device=self.device,
                )
                actions, values, log_probs, new_state, side = self.policy.forward(
                    obs_t, self._cell_state, episode_start=es,
                )

                # Diagnostics
                if "innovation" in side:
                    innovation_buf.append(
                        side["innovation"].abs().mean().item()
                    )
                state_norm_buf.append({
                    k: v.norm().item() for k, v in new_state.items()
                })

            actions_np = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions_np)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            action_dist_buf.append(actions_np.copy())

            # Action storage convention (matches SB3 RolloutBuffer for Discrete:
            # (n_envs, 1) float arrays).
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

            # Advance live state — episode-boundary resets happen on the
            # NEXT iteration via `episode_start` passed to policy.forward.
            self._cell_state = {k: v.clone() for k, v in new_state.items()}

            self._last_obs = new_obs
            self._last_episode_starts = dones
            n_steps += 1

        # GAE bootstrap from V(s_T).
        with th.no_grad():
            obs_t = obs_as_tensor(new_obs, self.device)
            es = th.as_tensor(
                self._last_episode_starts, dtype=th.bool, device=self.device,
            )
            values = self.policy.predict_values(obs_t, self._cell_state, episode_start=es)

        rollout_buffer.compute_returns_and_advantage(values, dones)

        # ── per-rollout diagnostics ────────────────────────────────────
        if innovation_buf:
            self.logger.record(
                "debug/innovation_mag_mean", float(np.mean(innovation_buf))
            )
            self.logger.record(
                "debug/innovation_mag_max", float(np.max(innovation_buf))
            )
        for key in state_norm_buf[0]:
            values_per_step = [s[key] for s in state_norm_buf]
            self.logger.record(
                f"debug/state_{key}_norm_mean", float(np.mean(values_per_step))
            )
            self.logger.record(
                f"debug/state_{key}_norm_max", float(np.max(values_per_step))
            )

        # Action histogram diagnostic (entropy is also tracked in train()
        # losses; this gives the raw distribution).
        if isinstance(self.action_space, spaces.Discrete):
            all_actions = np.concatenate(action_dist_buf)
            for a in range(self.action_space.n):
                frac = float((all_actions == a).mean())
                self.logger.record(f"debug/action_frac_{a}", frac)

        callback.on_rollout_end()
        return True

    # ── eval / predict ────────────────────────────────────────────────────

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        """Used by SB3 EvalCallback. Maintains cell state across eval steps.

        `deterministic=True` selects the argmax action — what we want for
        eval-time reward measurement so the policy's epsilon-greedy
        sampling doesn't add variance.
        """
        self.policy.set_training_mode(False)
        obs_tensor = obs_as_tensor(observation, self.device)

        # Discover batch size from obs.
        if isinstance(observation, dict):
            n = next(iter(observation.values())).shape[0]
        else:
            n = observation.shape[0]

        # Build / restore cell state.
        if state is None:
            cell_state = self.policy.initial_state(n, self.device)
        else:
            cell_state = state

        with th.no_grad():
            actions, _, _, new_state, _ = self.policy.forward(
                obs_tensor, cell_state,
                episode_start=episode_start,
                deterministic=deterministic,
            )

        actions = actions.cpu().numpy()
        return actions, new_state

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

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = batch.old_values + th.clamp(
                        values - batch.old_values, -clip_range_vf, clip_range_vf,
                    )
                value_loss = F.mse_loss(batch.returns, values_pred)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - batch.old_log_prob
                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio
                    ).item()
                    approx_kl_divs.append(approx_kl_div)

                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"  early stop epoch {epoch}, KL={approx_kl_div:.3f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()

                # Per-component grad-norm logging (README discipline).
                # Clip per component to max_grad_norm.
                per_comp_max = 0.0
                for name, mod in [
                    ("encoder", self.policy.encoder),
                    ("cell",    self.policy.cell),
                    ("actor",   self.policy.actor),
                    ("critic",  self.policy.critic),
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
        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard",
        )
        self.logger.record("train/learning_rate", lr)
        self.logger.record("train/clip_range",    clip_range)

        for name, norms in comp_grad_norms.items():
            if norms:
                self.logger.record(f"grad/{name}_norm", float(np.mean(norms)))
