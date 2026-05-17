"""TBPTT rollout buffer.

Generic over `RecurrentCell` — works with any cell satisfying the
`memrl.cell.base.RecurrentCell` interface. Cell state shapes are discovered
at construction time via `cell.init_state(batch_size=1)`, so adding a
new cell to the codebase doesn't require touching the buffer.

Architectural ancestor: `lmu_ppo/buffer.py::LMURolloutBuffer`. We keep its
core design (per-step state storage; chunk-based `get()` that yields
`(B, K, ...)` sequences) and strip its LMU-specifics (the hardcoded
(`lmu_h`, `lmu_m`) fields, the `lmu_t` LegS counter).

Why store state at every step, not just chunk starts:
    `add()` is called once per env step; we don't know chunk boundaries
    at add-time. Storing all T positions lets `get()` choose chunk starts
    freely. The extra memory cost is small in absolute terms compared to
    the observation buffer.

Chunk layout:
    Buffer stores (T, n_envs, *) arrays. `get()` enumerates all valid
    non-overlapping chunk starts (t * K, env) for t in [0, T/K) and
    env in [0, n_envs), shuffles them, and yields batches of `n_chunks`
    independent K-step sequences as (B, K, *) tensors. Each chunk's
    cell-state slot is the state at its FIRST timestep — the cell is
    re-run for K steps during `evaluate_actions` to recompute intra-chunk
    states with gradient.
"""

from __future__ import annotations

from typing import Generator, NamedTuple, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


CellStateDict = dict[str, th.Tensor]


class MemRolloutBufferSamples(NamedTuple):
    observations:   th.Tensor              # (B, K, *obs_shape)
    actions:        th.Tensor              # (B, K)  long (Discrete) / (B, K, action_dim) float
    old_values:     th.Tensor              # (B*K,)
    old_log_prob:   th.Tensor              # (B*K,)
    advantages:     th.Tensor              # (B*K,)
    returns:        th.Tensor              # (B*K,)
    cell_state:     CellStateDict          # dict, each (B, *state_shape) — at chunk START
    episode_starts: th.Tensor              # (B, K) float32  1.0 = new episode at this step


class MemRolloutBuffer(RolloutBuffer):
    """RolloutBuffer extended with per-step cell-state storage and chunk-based get().

    State storage is dict-typed: every key returned by the cell's
    `init_state()` gets its own (T, n_envs, *state_shape) numpy array.
    """

    def __init__(
        self,
        buffer_size:       int,
        observation_space: spaces.Space,
        action_space:      spaces.Space,
        state_shapes:      dict[str, tuple[int, ...]],
        chunk_len:         int,
        device:            str = "auto",
        gamma:             float = 0.99,
        gae_lambda:        float = 0.95,
        n_envs:            int = 1,
    ) -> None:
        if buffer_size % chunk_len != 0:
            raise ValueError(
                f"buffer_size ({buffer_size}) must be divisible by "
                f"chunk_len ({chunk_len}) so chunks tile the buffer exactly"
            )
        self.state_shapes = state_shapes
        self.chunk_len = chunk_len

        super().__init__(
            buffer_size, observation_space, action_space,
            device, gamma, gae_lambda, n_envs,
        )

    def reset(self) -> None:
        # Per-key cell-state storage, one array per state component.
        # Shape: (T, n_envs, *state_shape).
        self._state_arrays = {
            k: np.zeros((self.buffer_size, self.n_envs, *shape), dtype=np.float32)
            for k, shape in self.state_shapes.items()
        }
        super().reset()

    # --- add ----------------------------------------------------------------

    def add(  # type: ignore[override]
        self,
        obs:           np.ndarray,
        action:        np.ndarray,
        reward:        np.ndarray,
        episode_start: np.ndarray,
        value:         th.Tensor,
        log_prob:      th.Tensor,
        cell_state:    CellStateDict,    # dict of (n_envs, *state_shape) tensors
    ) -> None:
        """Store one timestep of rollout data plus the cell state going INTO it.

        `cell_state` is the state the cell consumed to produce this step's
        action — i.e., the PRE-step state. This is what we need to restore
        at chunk-start during `evaluate_actions` to replay the chunk with
        gradient.
        """
        for k, v in cell_state.items():
            self._state_arrays[k][self.pos] = v.detach().cpu().numpy()
        super().add(obs, action, reward, episode_start, value, log_prob)

    # --- get ---------------------------------------------------------------

    def get(  # type: ignore[override]
        self,
        n_chunks: Optional[int] = None,
    ) -> Generator[MemRolloutBufferSamples, None, None]:
        """Yield batches of K-step sequence chunks for TBPTT.

        Args:
            n_chunks: how many independent sequences per yielded batch.
                      None = one giant batch containing all chunks.
                      Typical: `n_chunks_per_batch` from the trainer config.

        Each yielded batch carries `n_chunks * K` transitions (flattened
        for the PPO scalar losses) plus `(n_chunks, K, ...)` sequences for
        anything the cell needs to consume in order.
        """
        if not self.full:
            raise RuntimeError("Buffer must be full before sampling.")

        K = self.chunk_len
        n_chunks_per_env = self.buffer_size // K
        all_chunks = [
            (t * K, e)
            for e in range(self.n_envs)
            for t in range(n_chunks_per_env)
        ]
        np.random.shuffle(all_chunks)

        n_chunks = n_chunks or len(all_chunks)
        for start in range(0, len(all_chunks), n_chunks):
            yield self._get_samples(all_chunks[start : start + n_chunks])

    def _get_samples(  # type: ignore[override]
        self,
        chunks: list[tuple[int, int]],
        env: Optional[VecNormalize] = None,
    ) -> MemRolloutBufferSamples:
        K = self.chunk_len
        B = len(chunks)

        t_starts = np.array([c[0] for c in chunks], dtype=np.int64)
        envs = np.array([c[1] for c in chunks], dtype=np.int64)
        # t_idx[b, k] = t_starts[b] + k → (B, K) timestep matrix
        t_idx = t_starts[:, None] + np.arange(K, dtype=np.int64)[None, :]

        # observations: (T, n_envs, *obs_shape) → (B, K, *obs_shape)
        obs = self.to_torch(self.observations[t_idx, envs[:, None]])

        # cell state at chunk start: (B, *state_shape) per key
        cell_state = {
            k: self.to_torch(arr[t_starts, envs])
            for k, arr in self._state_arrays.items()
        }

        # episode_starts: (B, K) — used inside evaluate_actions to reset
        # state at boundaries that occur mid-chunk.
        ep_starts = self.to_torch(
            self.episode_starts[t_idx, envs[:, None]].astype(np.float32)
        )

        # Per-step scalars — fancy-indexed then flattened to (B*K,) for
        # PPO's standard scalar losses.
        actions    = self.to_torch(self.actions[t_idx, envs[:, None]])    # (B,K,1)
        advantages = self.to_torch(self.advantages[t_idx, envs[:, None]])
        returns    = self.to_torch(self.returns[t_idx, envs[:, None]])
        old_vals   = self.to_torch(self.values[t_idx, envs[:, None]])
        old_lp     = self.to_torch(self.log_probs[t_idx, envs[:, None]])

        # Discrete-only for Phase A; Box would skip the .long().squeeze.
        if isinstance(self.action_space, spaces.Discrete):
            actions_out = actions.long().squeeze(-1)                       # (B, K)
        else:
            actions_out = actions                                          # (B, K, action_dim)

        return MemRolloutBufferSamples(
            observations=obs,
            actions=actions_out,
            old_values=old_vals.reshape(B * K),
            old_log_prob=old_lp.reshape(B * K),
            advantages=advantages.reshape(B * K),
            returns=returns.reshape(B * K),
            cell_state=cell_state,
            episode_starts=ep_starts,
        )
