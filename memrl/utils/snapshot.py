"""Versioned checkpoint snapshots for the decodability probe.

`ResumableCheckpointCallback` overwrites a single `latest.pt` for crash recovery.
The memory-decodability study (docs/decodability_probe_spec.md) instead needs
*frozen snapshots at fixed training milestones* — 0.5M, 2M, 5M, 10M (and 20M for
the headline) env-steps — so we can plot "how decodable is path-memory vs step"
for none / e3b / pbim / penalty arms.

This callback writes one immutable `snapshot_step{N}.pt` per milestone and,
optionally, logs each as a Weights & Biases artifact so probes can run on any
machine (the cluster trains, your laptop probes).

Snapshot contents mirror the resumable checkpoint's policy fields (enough to
rebuild the policy and read its recurrent state); optimizer/RNG are omitted since
snapshots are for analysis, not resumption.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import torch
from stable_baselines3.common.callbacks import BaseCallback


class SnapshotCallback(BaseCallback):
    """Save immutable policy snapshots at fixed env-step milestones.

    Args:
        save_path:      directory for snapshot files (usually the run dir).
        milestones:     env-step counts at which to snapshot. A snapshot fires
                        the first time num_timesteps crosses each milestone.
        config:         the run config dict (saved alongside so the probe can
                        rebuild the cell_factory without the original launch env).
        wandb_run:      optional active wandb run; if given, each snapshot is
                        logged as an artifact named "{run}-snapshots".
        verbose:        0/1.
    """

    def __init__(
        self,
        save_path: str | os.PathLike,
        milestones: Sequence[int],
        config: Optional[dict] = None,
        wandb_run=None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        # sorted, de-duped; we pop as we cross them
        self._todo = sorted(set(int(m) for m in milestones))
        self.config = config
        self.wandb_run = wandb_run

    def _snapshot(self, step: int) -> None:
        fname = self.save_path / f"snapshot_step{step}.pt"
        payload = {
            "policy_state": self.model.policy.state_dict(),
            "num_timesteps": int(self.model.num_timesteps),
            "milestone": int(step),
            "config": self.config,
        }
        # capture live cell state too (handy for warm-start probes), CPU-copied
        cs = getattr(self.model, "_cell_state", None)
        if isinstance(cs, dict):
            payload["cell_state"] = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in cs.items()
            }
        tmp = fname.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, fname)
        if self.verbose:
            print(f"  [snapshot] wrote {fname.name} at {self.model.num_timesteps} steps")

        if self.wandb_run is not None:
            try:
                import wandb
                art = wandb.Artifact(
                    name=f"{self.wandb_run.name}-snapshots",
                    type="model-snapshot",
                    metadata={"milestone": step, "num_timesteps": int(self.model.num_timesteps)},
                )
                art.add_file(str(fname))
                self.wandb_run.log_artifact(art, aliases=[f"step{step}"])
                if self.verbose:
                    print(f"  [snapshot] logged wandb artifact alias step{step}")
            except Exception as e:  # never let logging kill training
                if self.verbose:
                    print(f"  [snapshot] wandb artifact upload skipped: {e}")

    def _on_step(self) -> bool:
        # fire every milestone we've now crossed (handles multi-step jumps)
        while self._todo and self.model.num_timesteps >= self._todo[0]:
            step = self._todo.pop(0)
            self._snapshot(step)
        return True

    def _on_training_end(self) -> None:
        # final snapshot for any milestone never reached exactly (e.g. early stop)
        if self._todo:
            last = self._todo[-1]
            self._snapshot(last)
            self._todo.clear()
