"""B7-theory — truncation × density: the theory's own falsification test.

Proposition T1.1 (docs/theory_memory_density.md): an episodic bonus restores
ONE-HOP credit to memory writes that sparse extrinsic reward cannot reach under
TBPTT truncation. Prediction: **e3b − none widens monotonically as chunk length
k shrinks**; at k ≥ episode length (128 = full BPTT) the truncation channel is
closed and the residual gap isolates the variance-reduction part of
densification.

Design: GRU × k ∈ {1, 2, 4, 8, 16, 32, 64, 128} × {none, e3b_idm} ×
sparse MysteryPath-Grid × 3 seeds = 48 runs. {none, e3b} PAIR per GPU → 24 jobs.

Batch-size control: `n_chunks_per_batch = 2048 / k` holds the PPO minibatch at
a constant 2048 steps for every k, so ONLY the truncation horizon varies
(k=1 → 2048 one-step chunks; k=128 → 16 chunks). n_steps=512 divides evenly by
every k (all powers of two). k=1 is the clean extreme: the recurrent state
still carries forward at inference, but no gradient ever crosses a step — all
temporal credit must arrive through the value bootstrap.

Everything else = the locked winners (lr 1e-4, λ=0.03 for e3b, sparse arm).

Usage:
    python experiments/tbptt_density/generate_configs.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from train import DEFAULT_CELL_KWARGS  # noqa: E402

CELL = "GRU"
CHUNK_LENS = [1, 2, 4, 8, 16, 32, 64, 128]
INTRINSICS = ["none", "e3b_idm"]
SEEDS = [0, 1, 2]
BATCH_STEPS = 2048                      # chunk_len * n_chunks_per_batch, held constant

BASE = {
    "env_name": "MysteryPath-Grid-v0",  # sparse arm: env defaults (goal-only)
    "n_envs": 16,
    "total_timesteps": 10_000_000,
    "encoder_dim": 128,
    "encoder_hidden": 256,
    "lr": 1.0e-4,
    "n_steps": 512,
    "n_epochs": 4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.008,
    "vf_coef": 1.0,
    "max_grad_norm": 0.5,
    "target_kl": 0.05,
    "lambda_intrinsic": 0.03,
    "eval_every_rollouts": 10,
    "n_eval_episodes": 20,
    "wandb": True,
    "wandb_project": "memrl-tbptt",
}


def build_cfg(k: int, intrinsic: str) -> dict:
    ordered = {
        "env_name": BASE["env_name"],
        "n_envs": BASE["n_envs"],
        "total_timesteps": BASE["total_timesteps"],
        "seed": 0,                      # overridden per run via --seed
        "cell": {"name": CELL, "kwargs": dict(DEFAULT_CELL_KWARGS[CELL])},
        "encoder_dim": BASE["encoder_dim"],
        "encoder_hidden": BASE["encoder_hidden"],
    }
    for key in ("lr", "n_steps", "n_epochs", "gamma", "gae_lambda", "clip_range",
                "ent_coef", "vf_coef", "max_grad_norm", "target_kl"):
        ordered[key] = BASE[key]
    ordered["chunk_len"] = k
    ordered["n_chunks_per_batch"] = BATCH_STEPS // k
    ordered["intrinsic"] = intrinsic
    if intrinsic != "none":
        ordered["lambda_intrinsic"] = BASE["lambda_intrinsic"]
    for key in ("eval_every_rollouts", "n_eval_episodes", "wandb", "wandb_project"):
        ordered[key] = BASE[key]
    ordered["run_name"] = f"{CELL}-k{k}-{intrinsic}"
    return ordered


_SCRIPT_TEMPLATE = """\
#!/usr/bin/env bash
# AUTO-GENERATED. B7-theory truncation x density: k=__K__ at seed __SEED__ —
# runs {none, e3b_idm} in PARALLEL on one GPU.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"; RUNS_DIR="${RUNS_DIR:-runs/tbptt_density}"; EXTRA="${EXTRA:-}"
LOGDIR="$HERE/../_logs"; mkdir -p "$LOGDIR"
K="__K__"; SEED="__SEED__"
echo "[s$SEED | k=$K] none + e3b_idm in parallel on one GPU"
pids=()
for intr in none e3b_idm; do
  CFG="$HERE/../configs/tbk_k${K}_${intr}.yaml"
  "$PYTHON" -m train --config "$CFG" --seed "$SEED" \\
      --runs-dir "$RUNS_DIR" --device "$DEVICE" $EXTRA \\
      > "$LOGDIR/k${K}_${intr}_seed${SEED}.log" 2>&1 &
  pids+=("$!"); echo "  launched k=$K/$intr seed=$SEED (pid $!)"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ "$fail" == "0" ]] || { echo "[s$SEED | k=$K] a run FAILED — see $LOGDIR"; exit 1; }
echo "[s$SEED | k=$K] done."
"""

_RUN_ALL_TEMPLATE = """\
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/tbk_*.sh; do
  bn="$(basename "$s")"; [[ "$bn" == "run_all.sh" ]] && continue
  echo "=== $bn ==="; bash "$s"
done
echo "All tbptt-density jobs done."
"""


def main() -> None:
    cfg_dir = HERE / "configs"; scr_dir = HERE / "scripts"
    cfg_dir.mkdir(parents=True, exist_ok=True); scr_dir.mkdir(parents=True, exist_ok=True)
    for old in list(cfg_dir.glob("tbk_*.yaml")) + list(scr_dir.glob("tbk_*.sh")):
        old.unlink()

    for k in CHUNK_LENS:
        for intr in INTRINSICS:
            cfg = build_cfg(k, intr)
            with open(cfg_dir / f"tbk_k{k}_{intr}.yaml", "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

    n_scr = 0
    for seed in SEEDS:
        for k in CHUNK_LENS:
            body = (_SCRIPT_TEMPLATE
                    .replace("__K__", str(k))
                    .replace("__SEED__", str(seed)))
            p = scr_dir / f"tbk_s{seed}_k{k}.sh"
            p.write_text(body); os.chmod(p, 0o755); n_scr += 1
    ra = scr_dir / "run_all.sh"; ra.write_text(_RUN_ALL_TEMPLATE); os.chmod(ra, 0o755)

    n_cfg = len(CHUNK_LENS) * len(INTRINSICS)
    print(f"B7-theory: GRU x k={CHUNK_LENS} x {INTRINSICS} x sparse MysteryPath")
    print(f"batch held at {BATCH_STEPS} steps for all k | project=memrl-tbptt | 10M")
    print(f"wrote {n_cfg} configs + {n_scr} scripts (+ run_all.sh)")
    print(f"= {n_scr} GPU jobs (none+e3b pair each) = {n_cfg * len(SEEDS)} runs")


if __name__ == "__main__":
    main()
