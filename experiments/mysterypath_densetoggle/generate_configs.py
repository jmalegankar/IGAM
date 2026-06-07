"""B2 — the dense/sparse toggle: the experiment that decides the paper.

Same MysteryPath maze, same memory demand (invisible path), reward density TOGGLED:
  sparse = goal-only (default, reward_path_progress=0)
  dense  = +0.1 per newly-stepped correct path tile (reward_path_progress=0.1),
           via env_kwargs reset_options (verified to activate).

Crossed with the exploration bonus, holding the winning HPs fixed:
  {GRU, RetNet, GatedDeltaNet} × {sparse, dense} × {none, e3b_idm} × 5 seeds
  = 12 conditions × 5 = 60 runs.

HYPOTHESIS (K3): the e3b bonus HELPS under sparse reward but goes neutral/HARMFUL
under dense reward (the densification benefit vanishes; the policy-bias cost
remains). The effect flips sign, memory held fixed. This is what converts the
paper from "exploration helps sparse memory (trivial)" to "exploration bonuses
are reward densifiers, harmful on dense memory."

Locked HPs from the HP-sweep winner: e3b_idm, λ=0.03, chunk_len=64, lr=1e-4.
Packing: per (cell, density, seed) one GPU runs the {none, e3b} PAIR in parallel
(keeps the key bonus-vs-none comparison on one device/seed) → 30 jobs.

Usage:
    python experiments/mysterypath_densetoggle/generate_configs.py
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

CELLS = ["GRU", "RetNet", "GatedDeltaNet"]
DENSITIES = ["sparse", "dense"]
INTRINSICS = ["none", "e3b_idm"]
SEEDS = [0, 1, 2, 3, 4]
DENSE_ENV_KWARGS = {"reset_options": {"reward_path_progress": 0.1}}

BASE = {
    "env_name": "MysteryPath-Grid-v0",
    "n_envs": 16,
    "total_timesteps": 10_000_000,   # gate budget; extend to 20M for camera-ready
    "encoder_dim": 128,
    "encoder_hidden": 256,
    "lr": 1.0e-4,                     # WINNER
    "n_steps": 512,
    "n_epochs": 4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.008,
    "vf_coef": 1.0,
    "max_grad_norm": 0.5,
    "target_kl": 0.05,
    "chunk_len": 64,                 # WINNER
    "n_chunks_per_batch": 32,
    "lambda_intrinsic": 0.03,        # WINNER (e3b only)
    "eval_every_rollouts": 10,
    "n_eval_episodes": 20,
    "wandb": True,
    "wandb_project": "memrl-mpg-densetoggle",
}


def build_cfg(cell: str, density: str, intrinsic: str) -> dict:
    ordered = {
        "env_name": BASE["env_name"],
    }
    if density == "dense":
        ordered["env_kwargs"] = DENSE_ENV_KWARGS
    ordered.update({
        "n_envs": BASE["n_envs"],
        "total_timesteps": BASE["total_timesteps"],
        "seed": 0,                      # overridden per run via --seed
        "cell": {"name": cell, "kwargs": dict(DEFAULT_CELL_KWARGS[cell])},
        "encoder_dim": BASE["encoder_dim"],
        "encoder_hidden": BASE["encoder_hidden"],
    })
    for k in ("lr", "n_steps", "n_epochs", "gamma", "gae_lambda", "clip_range",
              "ent_coef", "vf_coef", "max_grad_norm", "target_kl",
              "chunk_len", "n_chunks_per_batch"):
        ordered[k] = BASE[k]
    ordered["intrinsic"] = intrinsic
    if intrinsic != "none":
        ordered["lambda_intrinsic"] = BASE["lambda_intrinsic"]
    for k in ("eval_every_rollouts", "n_eval_episodes", "wandb", "wandb_project"):
        ordered[k] = BASE[k]
    ordered["run_name"] = f"{cell}-{density}-{intrinsic}"
    return ordered


_SCRIPT_TEMPLATE = """\
#!/usr/bin/env bash
# AUTO-GENERATED. B2 dense-toggle: __CELL__ / __DENSITY__ at seed __SEED__ —
# runs {none, e3b_idm} in PARALLEL on one GPU (the bonus-vs-none comparison).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"; RUNS_DIR="${RUNS_DIR:-runs/mysterypath_densetoggle}"; EXTRA="${EXTRA:-}"
LOGDIR="$HERE/../_logs"; mkdir -p "$LOGDIR"
CELL="__CELL__"; DENSITY="__DENSITY__"; SEED="__SEED__"
echo "[s$SEED | $CELL $DENSITY] none + e3b_idm in parallel on one GPU"
pids=()
for intr in none e3b_idm; do
  CFG="$HERE/../configs/mpgdt_${CELL}_${DENSITY}_${intr}.yaml"
  "$PYTHON" -m train --config "$CFG" --seed "$SEED" \\
      --runs-dir "$RUNS_DIR" --device "$DEVICE" $EXTRA \\
      > "$LOGDIR/${CELL}_${DENSITY}_${intr}_seed${SEED}.log" 2>&1 &
  pids+=("$!"); echo "  launched ${CELL}/${DENSITY}/${intr} seed=$SEED (pid $!)"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ "$fail" == "0" ]] || { echo "[s$SEED | $CELL $DENSITY] a run FAILED — see $LOGDIR"; exit 1; }
echo "[s$SEED | $CELL $DENSITY] done."
"""

_RUN_ALL_TEMPLATE = """\
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/mpgdt_*.sh; do
  bn="$(basename "$s")"; [[ "$bn" == "run_all.sh" ]] && continue
  echo "=== $bn ==="; bash "$s"
done
echo "All dense-toggle jobs done."
"""


def main() -> None:
    cfg_dir = HERE / "configs"; scr_dir = HERE / "scripts"
    cfg_dir.mkdir(parents=True, exist_ok=True); scr_dir.mkdir(parents=True, exist_ok=True)

    for cell in CELLS:
        for density in DENSITIES:
            for intr in INTRINSICS:
                cfg = build_cfg(cell, density, intr)
                with open(cfg_dir / f"mpgdt_{cell}_{density}_{intr}.yaml", "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)

    n_scr = 0
    for seed in SEEDS:
        for cell in CELLS:
            for density in DENSITIES:
                body = (_SCRIPT_TEMPLATE
                        .replace("__CELL__", cell)
                        .replace("__DENSITY__", density)
                        .replace("__SEED__", str(seed)))
                p = scr_dir / f"mpgdt_s{seed}_{cell}_{density}.sh"
                p.write_text(body); os.chmod(p, 0o755); n_scr += 1
    ra = scr_dir / "run_all.sh"; ra.write_text(_RUN_ALL_TEMPLATE); os.chmod(ra, 0o755)

    n_cfg = len(CELLS) * len(DENSITIES) * len(INTRINSICS)
    print("env=MysteryPath-Grid  project=memrl-mpg-densetoggle  budget=10M")
    print(f"HPs: e3b_idm λ=0.03 ck=64 lr=1e-4 | cells={CELLS} | density={DENSITIES}")
    print(f"wrote {n_cfg} configs + {n_scr} scripts (+ run_all.sh)")
    print(f"= {n_scr} GPU jobs (none+e3b each) = {len(CELLS)*len(DENSITIES)*len(INTRINSICS)*len(SEEDS)} runs")


if __name__ == "__main__":
    main()
