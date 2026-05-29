"""Memory-cell PPO trainer.

PPO with a generic recurrent policy + TBPTT-chunked rollout buffer.
Cell-agnostic: works with any `RecurrentCell` (GRU, LSTM, LMU, S4D, Mamba2,
LinearTransformer, RetNet, DeltaNet, mLSTM, ...).

Logging:
  - Chunked TBPTT (chunk_len, n_chunks_per_batch)
  - Per-component grad-norm logging
  - State-norm logging per cell-state component
  - Innovation magnitude logging (when the cell emits one — DeltaNet, etc.)
  - SB3 PPO scaffolding (callbacks, schedules, env wrappers, eval)

Cells emit a `SideOutputs` dict from `step` for things like per-step
innovation signals; this loop forwards them through `collect_rollouts`
so downstream training code (e.g., lifelong intrinsic rewards on the
research branches) can read them.
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
from memrl.policy.buffer import MemRolloutBuffer
from memrl.policy.policy import MemActorCriticPolicy


# A cell factory takes (input_size) and returns a constructed RecurrentCell.
# `input_size` will equal `encoder_dim` (the encoder's output dim).
CellFactory = Callable[[int], RecurrentCell]


class MemPPO(PPO):
    """PPO with a recurrent memory-cell policy and TBPTT rollout buffer."""

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
        lambda_intrinsic: float = 0.0,
        intrinsic_module: Optional["IntrinsicRewardModule"] = None,
        intrinsic_source: str = "eps_mem",         # back-compat, used iff module=None
        encoder_dim: int = 64,
        encoder_hidden: int = 128,
        shared_backbones: bool = False,
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
        self.shared_backbones = shared_backbones
        self.chunk_len = chunk_len
        self.n_chunks_per_batch = n_chunks_per_batch

        # Intrinsic reward — uniform interface via the IntrinsicRewardModule
        # abstraction (see memrl/exploration/). When module is provided AND
        # lambda > 0, per-step bonus is added to extrinsic reward BEFORE the
        # rollout buffer ingests it (stop-grad is automatic — rewards are
        # numpy). Episodic modules (E3B, NovelD) have reset_envs() called on
        # episode boundaries; lifelong modules (RND, ICM) get .update() called
        # per rollout on the buffer.
        #
        # Legacy back-compat: if intrinsic_module=None but lambda > 0, falls
        # back to reading a named side output (default "eps_mem"). This path
        # is preserved for any old code paths but new work should pass an
        # explicit module.
        self.lambda_intrinsic = float(lambda_intrinsic)
        self.intrinsic_module = intrinsic_module
        self.intrinsic_source = intrinsic_source
        if self.lambda_intrinsic > 0 and self.intrinsic_module is None:
            from stable_baselines3.common.running_mean_std import RunningMeanStd
            self.intrinsic_rms = RunningMeanStd(shape=())
        else:
            self.intrinsic_rms = None

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

        # Cell construction depends on backbone mode:
        # - shared_backbones=True (Option B+): one cell, both actor and critic
        #   use it; critic input is detached at the head. 1× compute.
        # - shared_backbones=False (Option A): two cells with independent
        #   weights (Ni 2022 — separate backbones prevent critic gradient
        #   dominance). 2× compute, slightly better late-game ceiling.
        cell_actor = self.cell_factory(self.encoder_dim).to(self.device)
        if self.shared_backbones:
            cell_critic = None
        else:
            cell_critic = self.cell_factory(self.encoder_dim).to(self.device)

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

        # State shape discovery: the policy's initial_state returns a
        # namespaced dict (actor_* + critic_*). The buffer sees a flat
        # dict and allocates arrays for every key — no special handling
        # for the actor/critic split.
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

        # Live cell state across rollout collection. Initialized in _setup_learn
        # (when n_envs is known).
        self._cell_state: Optional[dict[str, th.Tensor]] = None

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
        if self.verbose >= 1:
            cell_name = type(self.policy.cell_actor).__name__
            n_params = sum(p.numel() for p in self.policy.parameters())
            print(f"  cell={cell_name} (×2: separate actor/critic backbones)  "
                  f"params={n_params:,}  chunk_len={self.chunk_len}  "
                  f"n_chunks_per_batch={self.n_chunks_per_batch}")
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
        # PERF: keep these as GPU tensors and stack once at end of rollout
        # rather than .item()-ing each step. A .item() call forces a GPU→CPU
        # sync which serializes the entire pipeline — previously this loop
        # did 14 syncs per env step (12 state norms + 2 innovation), which
        # alone was ~30% of rollout wall time on the 3070 Ti. Now we do one
        # sync per rollout. See PR notes 2026-05-23.
        innovation_buf_t:   list[th.Tensor] = []
        innov_per_env_buf:  list[th.Tensor] = []
        state_keys: tuple[str, ...] | None = None  # set on first iter
        state_norm_rows_t:  list[th.Tensor] = []
        action_dist_buf: list[np.ndarray] = []
        # Per-step phase ids (parallel to innov_per_env_buf), so we can bucket
        # at end-of-rollout in one pass. Phase is small int already on CPU.
        phase_per_step: list[np.ndarray | None] = []

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

                # Diagnostics — accumulate GPU tensors only; drain at rollout end.
                if "innovation" in side:
                    innov_abs = side["innovation"].abs()
                    innovation_buf_t.append(innov_abs.mean())             # scalar GPU
                    innov_per_env_buf.append(innov_abs.mean(dim=-1))      # (n_envs,) GPU
                else:
                    innov_per_env_buf.append(None)  # placeholder for indexing
                if state_keys is None:
                    state_keys = tuple(new_state.keys())
                # Memoryless cells have an empty state dict; th.stack([]) would
                # raise, and there is no state-norm to log, so skip entirely.
                if state_keys:
                    state_norm_rows_t.append(th.stack([
                        new_state[k].norm() for k in state_keys
                    ]))                                                   # (K,) GPU

            actions_np = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions_np)
            self.num_timesteps += env.num_envs

            # Intrinsic reward shaping. Three paths in priority order:
            #   1. `self.intrinsic_module` provided → use the uniform abstraction
            #      (recommended). Module returns sanitized (n_envs,) np.float32.
            #   2. Legacy: `lambda_intrinsic > 0` and a named side output exists
            #      → use built-in running-std normalisation.
            #   3. Otherwise: no intrinsic reward.
            if self.intrinsic_module is not None and self.lambda_intrinsic > 0:
                bonus = self.intrinsic_module.compute(
                    obs=new_obs,
                    last_obs=self._last_obs,
                    action=actions_np,
                    episode_start=self._last_episode_starts,
                    side=side,
                    cell_state=self._cell_state,
                )                                                             # (n_envs,) np.float32
                rewards = rewards.astype(np.float32) + self.lambda_intrinsic * bonus
                # Reset per-episode state for envs that just ended an episode.
                done_ids = [i for i, d in enumerate(dones) if d]
                if done_ids:
                    self.intrinsic_module.reset_envs(done_ids)
                # Lightweight log (per-rollout mean below).
                if not hasattr(self, "_intrinsic_buf"):
                    self._intrinsic_buf = []
                self._intrinsic_buf.append(float(bonus.mean()))
            elif self.lambda_intrinsic > 0 and self.intrinsic_source in side:
                # Legacy path — uses side output directly with running-std norm.
                eps_t = side[self.intrinsic_source].detach()
                if eps_t.dim() == 0:
                    eps_t = eps_t.expand(env.num_envs)
                eps_np = eps_t.cpu().numpy().astype(np.float32)
                self.intrinsic_rms.update(eps_np)
                norm = float(np.sqrt(self.intrinsic_rms.var) + 1e-8)
                intrinsic = (eps_np / norm) * self.lambda_intrinsic
                rewards = rewards.astype(np.float32) + intrinsic
                if not hasattr(self, "_intrinsic_buf"):
                    self._intrinsic_buf = []
                self._intrinsic_buf.append(float(intrinsic.mean()))

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            action_dist_buf.append(actions_np.copy())

            # Cache phase ids per env (cheap; ints on CPU). Bucketing into
            # per-phase mean innovation happens after the rollout when we
            # drain innov_per_env_buf to numpy in a single transfer.
            if innov_per_env_buf[-1] is not None:
                ph = np.full(env.num_envs, -1, dtype=np.int64)
                for env_idx, info in enumerate(infos):
                    if "phase" in info:
                        ph[env_idx] = int(info["phase"])
                phase_per_step.append(ph)
            else:
                phase_per_step.append(None)

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
        # Single GPU→CPU drain at end-of-rollout (vs. 14 per-step syncs
        # previously). The stacks are tiny (~n_steps × <20 scalars) so the
        # transfer cost is negligible compared to the saved sync overhead.
        if innovation_buf_t:
            innovation_arr = th.stack(innovation_buf_t).cpu().numpy()
            self.logger.record("debug/innovation_mag_mean", float(innovation_arr.mean()))
            self.logger.record("debug/innovation_mag_max",  float(innovation_arr.max()))

        # Intrinsic reward drain (per-rollout mean of the per-step bonus).
        if getattr(self, "_intrinsic_buf", None):
            arr = np.array(self._intrinsic_buf, dtype=np.float32)
            self.logger.record("debug/intrinsic_reward_mean", float(arr.mean()))
            self.logger.record("debug/intrinsic_reward_max",  float(arr.max()))
            if self.intrinsic_rms is not None:
                self.logger.record("debug/intrinsic_running_std",
                                   float(np.sqrt(self.intrinsic_rms.var)))
            self._intrinsic_buf = []

        # Module-specific diagnostics + module update on the just-collected rollout.
        if self.intrinsic_module is not None:
            for k, v in self.intrinsic_module.diagnostics().items():
                self.logger.record(f"intrinsic/{k}", v)
            try:
                for k, v in self.intrinsic_module.update(self.rollout_buffer).items():
                    self.logger.record(f"intrinsic/{k}", v)
            except Exception as e:
                # Don't kill training because an exploration module's optional
                # update path threw; just log and skip.
                self.logger.record("intrinsic/update_error", 1.0)
                print(f"[intrinsic] update() failed: {e}")

        # Phase-grouped innovation (only for envs that expose info["phase"]).
        # Autoencode reports phase∈{0:WATCH, 1:PLAY}; the WATCH/PLAY ratio of
        # innovation magnitudes is the interpretability signal we want.
        any_phase = any(p is not None for p in phase_per_step)
        if any_phase and innov_per_env_buf and innov_per_env_buf[0] is not None:
            # Drain per-env innovation in one shot, then bucket on CPU.
            ipe = th.stack([t for t in innov_per_env_buf if t is not None]).cpu().numpy()
            # phase array stacks parallel to ipe (only the steps with a tensor).
            ph_rows = np.stack([p for p in phase_per_step if p is not None])
            per_phase: dict[int, list[float]] = {}
            for step_i in range(ipe.shape[0]):
                for env_i in range(ipe.shape[1]):
                    p = int(ph_rows[step_i, env_i])
                    if p >= 0:
                        per_phase.setdefault(p, []).append(float(ipe[step_i, env_i]))
            for phase, vals in per_phase.items():
                self.logger.record(
                    f"debug/innovation_mag_phase_{phase}_mean", float(np.mean(vals))
                )
                self.logger.record(
                    f"debug/innovation_mag_phase_{phase}_count", int(len(vals))
                )

        if state_norm_rows_t and state_keys is not None:
            # shape: (n_steps, K)
            norms_arr = th.stack(state_norm_rows_t).cpu().numpy()
            for i, key in enumerate(state_keys):
                col = norms_arr[:, i]
                self.logger.record(f"debug/state_{key}_norm_mean", float(col.mean()))
                self.logger.record(f"debug/state_{key}_norm_max",  float(col.max()))

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
                # Clip per component to max_grad_norm. Two cells now —
                # log them separately so we can see if the critic's
                # cell drifts differently from the actor's.
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
        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard",
        )
        self.logger.record("train/learning_rate", lr)
        self.logger.record("train/clip_range",    clip_range)

        for name, norms in comp_grad_norms.items():
            if norms:
                self.logger.record(f"grad/{name}_norm", float(np.mean(norms)))
