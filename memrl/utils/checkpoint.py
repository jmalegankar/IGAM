"""Resumable checkpointing for MemPPO runs.

Designed so the user (Jai) can interrupt a long ablation run — by Ctrl+C,
or by dropping a stop-file — go play a game, then re-launch the same
command to pick up exactly where it left off.

Why a custom callback (rather than SB3's built-in `CheckpointCallback`):

  * SB3's ``model.save()`` zips ``cloudpickle``-able attributes plus the
    torch state. ``MemPPO`` holds a ``cell_factory`` (closure capturing the
    YAML config) that isn't reliably picklable across sessions, and
    rebuilding the cell from YAML on resume is cleaner than depickling it.
  * We need to capture the LIVE cell state (``model._cell_state``) and
    last obs / episode-starts as well — the cell state is what makes
    resumption seamless mid-rollout boundary.
  * We need a stop-file watch + SIGINT handler that flushes a final save
    before exiting cleanly, returning ``False`` from ``_on_step`` so SB3
    unwinds without an exception.

What's saved (``latest.pt``, atomically renamed from ``latest.pt.tmp``):
    policy_state        nn.Module state_dict for the full policy
    optimizer_state     Adam state (matters for resumed momentum)
    num_timesteps       env-step counter (drives stop condition + LR schedule)
    _n_updates          PPO update counter (for logging only)
    _last_obs           VecEnv obs at the moment of save (n_envs, …)
    _last_episode_starts bool array, same shape
    _cell_state         dict of CPU tensors; copied back to device on load
    torch_rng           CPU RNG state
    torch_cuda_rng      per-device CUDA RNG states (None if no CUDA)
    numpy_rng           np.random state
    py_rng              random state

What's NOT saved (intentionally):
    rollout_buffer      reset every rollout, so a mid-rollout pause loses
                        only the partial rollout's data — harmless.
    ep_info_buffer      stats-only, deque(maxlen=100) of recent ep rewards.
    env RNG state       VecEnv envs continue from their current state
                        because we don't reset on resume. Strict
                        determinism vs. a never-paused run is not a goal.

Markers:
    ``DONE`` file in save_path indicates training completed; the resume
    helpers skip such runs.
    ``STOP`` file in save_path is a per-run pause signal that the callback
    polls each step (cheap: a path existence check).
"""

from __future__ import annotations

import os
import random
import signal
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


# Module-level reference to the active callback so a second Ctrl+C
# fast-quits without dragging through another save attempt.
_ACTIVE_CALLBACK: "Optional[ResumableCheckpointCallback]" = None


class ResumableCheckpointCallback(BaseCallback):
    """Periodically snapshot training state; flush a final save on stop signal.

    Save cadence:
        Saves every ``save_freq_steps`` env-steps. With ``n_envs=8`` and
        ``save_freq_steps=50_000`` the on-disk file gets rewritten roughly
        every 5-10 PPO rollouts, which on a 3070 Ti is ~30s-1min of work
        — the max amount of training you can lose to a hard kill.

    Stop signals (any of):
        - SIGINT (Ctrl+C in the controlling console).
        - A file appears at ``save_path/STOP``.
        - A file appears at any path in ``extra_stop_files``.

    On stop:
        Writes a final checkpoint and returns ``False`` from ``_on_step``,
        which makes ``MemPPO.collect_rollouts`` return ``False``, which
        makes ``learn()`` unwind cleanly. The shell exit code is 0 — the
        runner treats that as "paused, retry later" via the resume helper.

    Args:
        save_path: directory to write ``latest.pt`` into. Created if missing.
        save_freq_steps: env-step interval between auto-saves. Defaults to
            50_000 — small enough to lose <1 min on a 3070 Ti, large enough
            that the I/O isn't measurable.
        extra_stop_files: optional additional paths to poll. Useful when
            the runner wants a single global pause file (e.g. project-root
            ``STOP``) that pauses every concurrent worker at once.
        verbose: 0 silent, 1 prints save / stop notices.
    """

    def __init__(
        self,
        save_path: Path | str,
        save_freq_steps: int = 50_000,
        extra_stop_files: Iterable[Path | str] = (),
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.save_freq_steps = int(save_freq_steps)
        self.stop_files = [self.save_path / "STOP", *(Path(p) for p in extra_stop_files)]

        self._latest_path = self.save_path / "latest.pt"
        self._tmp_path    = self.save_path / "latest.pt.tmp"
        self._done_path   = self.save_path / "DONE"

        self._last_save_step = 0
        self._stop_requested = False
        self._prev_sigint_handler = None

    # ── signal wiring ─────────────────────────────────────────────────────

    def _install_sigint(self) -> None:
        """Install a SIGINT handler that sets a flag.

        Falls back silently if we're not in the main thread (e.g. when SB3
        is driven from a worker thread). A second SIGINT short-circuits to
        the previous handler so the user can always force-quit.
        """
        global _ACTIVE_CALLBACK
        _ACTIVE_CALLBACK = self
        try:
            self._prev_sigint_handler = signal.signal(signal.SIGINT, _sigint_handler)
        except (ValueError, OSError):
            # Not in main thread on this platform; rely on stop-file only.
            self._prev_sigint_handler = None

    def _restore_sigint(self) -> None:
        global _ACTIVE_CALLBACK
        if self._prev_sigint_handler is not None:
            try:
                signal.signal(signal.SIGINT, self._prev_sigint_handler)
            except (ValueError, OSError):
                pass
        if _ACTIVE_CALLBACK is self:
            _ACTIVE_CALLBACK = None

    # ── SB3 callback hooks ────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        self._install_sigint()
        # A pre-existing DONE marker is stale once new training starts.
        if self._done_path.exists():
            try:
                self._done_path.unlink()
            except OSError:
                pass
        # A pre-existing STOP file would immediately halt us — clear it so
        # the user can re-launch without manually deleting it.
        for sf in self.stop_files:
            if sf.exists():
                try:
                    sf.unlink()
                except OSError:
                    pass
        if self.verbose:
            print(f"  checkpoint: save_path={self.save_path}, "
                  f"save_freq_steps={self.save_freq_steps:,}, "
                  f"stop_files={[str(p) for p in self.stop_files]}")

    def _on_step(self) -> bool:
        # 1) stop-file check (cheap path.exists)
        if not self._stop_requested:
            for sf in self.stop_files:
                if sf.exists():
                    if self.verbose:
                        print(f"\n  [checkpoint] stop-file {sf} detected — saving and exiting.")
                    self._stop_requested = True
                    break

        # 2) periodic save
        if self.num_timesteps - self._last_save_step >= self.save_freq_steps:
            self._save()
            self._last_save_step = self.num_timesteps

        # 3) handle stop request: save once more (post-step state) and bail
        if self._stop_requested:
            self._save()
            return False  # tells SB3 to unwind

        return True

    def _on_training_end(self) -> None:
        # Final save + DONE marker (so resume helpers know to skip).
        self._save()
        if not self._stop_requested:
            try:
                self._done_path.touch()
            except OSError:
                pass
            if self.verbose:
                print(f"  [checkpoint] training complete, DONE marker written.")
        self._restore_sigint()

    # ── save / load core ──────────────────────────────────────────────────

    def _save(self) -> None:
        model = self.model
        try:
            cell_state = {
                k: v.detach().cpu().clone() for k, v in (model._cell_state or {}).items()
            }
        except Exception:
            cell_state = {}

        state = {
            "policy_state":        model.policy.state_dict(),
            "optimizer_state":     model.policy.optimizer.state_dict(),
            "num_timesteps":       int(model.num_timesteps),
            "_n_updates":          int(getattr(model, "_n_updates", 0)),
            "_last_obs":           model._last_obs,
            "_last_episode_starts": model._last_episode_starts,
            "_cell_state":         cell_state,
            "torch_rng":           torch.get_rng_state(),
            "torch_cuda_rng":      (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_rng":           np.random.get_state(),
            "py_rng":              random.getstate(),
        }

        # Atomic rename: writing to a sibling temp then os.replace() means a
        # crash mid-save can't corrupt the previous good checkpoint.
        torch.save(state, self._tmp_path)
        os.replace(self._tmp_path, self._latest_path)
        if self.verbose >= 2:
            print(f"  [checkpoint] saved at step {model.num_timesteps:,}")


# Free-standing SIGINT handler — bound to the active callback so the user
# can `kill -INT <pid>` from outside the controlling shell.
def _sigint_handler(signum, frame):
    cb = _ACTIVE_CALLBACK
    if cb is None or cb._stop_requested:
        # Either no callback installed or this is a second Ctrl+C — let the
        # original handler take over for a hard exit.
        sys.stderr.write("\n[checkpoint] second SIGINT — exiting immediately.\n")
        sys.stderr.flush()
        # Restore default and re-raise so the process actually dies.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt
    sys.stderr.write(
        "\n[checkpoint] SIGINT — finishing current step, will checkpoint "
        "and exit. Ctrl+C again to force-quit.\n"
    )
    sys.stderr.flush()
    cb._stop_requested = True


# ── load helper ───────────────────────────────────────────────────────────

def load_checkpoint(run_dir: Path | str, model) -> int:
    """Restore a MemPPO from ``run_dir/latest.pt``. Returns the saved
    ``num_timesteps`` so the caller can shorten ``total_timesteps`` accordingly.
    """
    path = Path(run_dir) / "latest.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    # Load everything to CPU first so the RNG-state ByteTensors land on CPU
    # (torch.set_rng_state requires a CPU ByteTensor). We move policy
    # weights and cell state explicitly below.
    state = torch.load(path, map_location="cpu", weights_only=False)

    model.policy.load_state_dict(state["policy_state"])
    # Send policy params to device — load_state_dict respects the receiving
    # module's device, so this is usually a no-op but explicit is safer.
    model.policy.to(model.device)
    model.policy.optimizer.load_state_dict(state["optimizer_state"])
    # Move optimizer tensors (Adam moments, step) to the target device too —
    # otherwise the first .step() will throw a device-mismatch error.
    for st in model.policy.optimizer.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(model.device)
    model.num_timesteps         = state["num_timesteps"]
    model._n_updates            = state["_n_updates"]
    model._last_obs             = state["_last_obs"]
    model._last_episode_starts  = state["_last_episode_starts"]
    model._cell_state = {
        k: v.to(model.device) for k, v in state["_cell_state"].items()
    }

    # RNG state lives on CPU; torch.set_rng_state requires CPU ByteTensor.
    torch.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda_rng") is not None:
        try:
            cuda_rng = [t.cpu() for t in state["torch_cuda_rng"]]
            torch.cuda.set_rng_state_all(cuda_rng)
        except RuntimeError:
            # Device count changed (e.g. checkpoint made on a 2-GPU box,
            # loading on a 1-GPU box). Skip — RNG state divergence is benign.
            pass
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["py_rng"])

    return int(state["num_timesteps"])


# ── resume-run-dir discovery ──────────────────────────────────────────────

def find_resume_run_dir(
    runs_dir: Path | str,
    benchmark_stem: str,
    cell_name: str,
    seed: int,
) -> Optional[Path]:
    """Find the newest run directory for (benchmark, cell, seed) that has a
    ``latest.pt`` and no ``DONE`` marker. Returns ``None`` if no such
    directory exists (so the caller should start a fresh run).

    Layout matched: ``runs_dir/benchmark_stem/cell_name/seed_<n>_<timestamp>/``
    """
    parent = Path(runs_dir) / benchmark_stem / cell_name
    if not parent.exists():
        return None
    candidates = sorted(
        (p for p in parent.iterdir()
         if p.is_dir() and p.name.startswith(f"seed_{seed}_")),
        key=lambda p: p.name,
        reverse=True,
    )
    for p in candidates:
        if (p / "DONE").exists():
            continue
        if (p / "latest.pt").exists():
            return p
    return None


def is_run_completed(
    runs_dir: Path | str,
    benchmark_stem: str,
    cell_name: str,
    seed: int,
) -> bool:
    """True if any run directory for (benchmark, cell, seed) has a DONE marker."""
    parent = Path(runs_dir) / benchmark_stem / cell_name
    if not parent.exists():
        return False
    for p in parent.iterdir():
        if (
            p.is_dir()
            and p.name.startswith(f"seed_{seed}_")
            and (p / "DONE").exists()
        ):
            return True
    return False
