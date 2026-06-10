"""MortarMayhem-Grid density toggle — the second-env CORE FLIP, natively return-matched.

The paper's third task (after MiniGrid-S13 and MysteryPath-Grid), same memory-gym
pipeline, but a DIFFERENT memory type: sequence working-memory (memorize a command
sequence during a show phase, then execute it blind) vs MysteryPath's spatial
trace memory.

Why MortarMayhem carries the cleanest flip we have:
  * NATIVE density knobs (no wrapper): `reward_episode_success` (terminal) vs
    `reward_command_success` (per-command, aligned with progress).
  * With a FIXED command count cc, aligned-dense max return = cc * (1/cc) = 1.0
    = sparse max return — **exactly return-matched by construction** (the
    magnitude confound that forced MysteryPath onto the fall penalty, solved
    natively here; empirically verified: both arms max_ret = 1.0).
  * Command failure TERMINATES the episode -> no useful per-step penalty arm;
    MM hosts the core flip {sparse, aligned-dense} only. The anti-dense /
    freeze-rescue cell stays MysteryPath's.

Difficulty: command_count=[4] -> random-policy success 0.0017 (measured, 600
episodes) — hard-but-learnable, ~6x sparser than sparse MysteryPath (~1%).
Fallback if both intrinsic arms stall at the seed-0 stage: cc=3 (2.2% random).

Design: {sparse, dense} x {none, e3b_idm} x {GRU, RetNet, GatedDeltaNet} x
5 seeds = 60 runs. Packing: {none, e3b} PAIR per GPU (84x84 pixel profile)
-> 30 jobs. HPs: the MysteryPath winner (e3b_idm, lambda=0.03, ck=64, lr=1e-4),
transferred not retuned (global-HP fair-comparison story).

Predictions (docs/revelation_and_densification.md):
  sparse:        e3b > none  (unpaid controllable revelation -> monetizer)
  aligned-dense: e3b ~ none  (revelation already paid -> redundant)

Usage:
    python experiments/mortarmayhem_densetoggle/generate_configs.py
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

COMMAND_COUNT = 4                       # measured: 0.0017 random success
_BASE_OPTS = {"command_count": [COMMAND_COUNT]}
ARM_OPTS = {
    # terminal-only: +1.0 iff the whole sequence is executed
    "sparse": {**_BASE_OPTS, "reward_command_success": 0.0,
               "reward_episode_success": 1.0},
    # aligned per-command: +1/cc per correct command, max = exactly 1.0
    "dense": {**_BASE_OPTS, "reward_command_success": round(1.0 / COMMAND_COUNT, 4),
              "reward_episode_success": 0.0},
}

BASE = {
    "env_name": "MortarMayhem-Grid-v0",
    "n_envs": 16,
    "total_timesteps": 10_000_000,
    "encoder_dim": 128,
    "encoder_hidden": 256,
    "lr": 1.0e-4,                     # WINNER (transferred)
    "n_steps": 512,
    "n_epochs": 4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.008,
    "vf_coef": 1.0,
    "max_grad_norm": 0.5,
    "target_kl": 0.05,
    "chunk_len": 64,                  # WINNER (transferred)
    "n_chunks_per_batch": 32,
    "lambda_intrinsic": 0.03,         # WINNER (transferred, e3b only)
    "eval_every_rollouts": 10,
    "n_eval_episodes": 20,
    "wandb": True,
    "wandb_project": "memrl-mm-toggle",
}


def build_cfg(cell: str, density: str, intrinsic: str) -> dict:
    ordered = {
        "env_name": BASE["env_name"],
        "env_kwargs": {"reset_options": dict(ARM_OPTS[density])},
        "n_envs": BASE["n_envs"],
        "total_timesteps": BASE["total_timesteps"],
        "seed": 0,                      # overridden per run via --seed
        "cell": {"name": cell, "kwargs": dict(DEFAULT_CELL_KWARGS[cell])},
        "encoder_dim": BASE["encoder_dim"],
        "encoder_hidden": BASE["encoder_hidden"],
    }
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
# AUTO-GENERATED. MortarMayhem density toggle: __CELL__ / __DENSITY__ at seed
# __SEED__ — runs {none, e3b_idm} in PARALLEL on one GPU (84x84 pixel pair).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"; RUNS_DIR="${RUNS_DIR:-runs/mortarmayhem_densetoggle}"; EXTRA="${EXTRA:-}"
LOGDIR="$HERE/../_logs"; mkdir -p "$LOGDIR"
CELL="__CELL__"; DENSITY="__DENSITY__"; SEED="__SEED__"
echo "[s$SEED | $CELL $DENSITY] none + e3b_idm in parallel on one GPU"
pids=()
for intr in none e3b_idm; do
  CFG="$HERE/../configs/mmdt_${CELL}_${DENSITY}_${intr}.yaml"
  "$PYTHON" -m train --config "$CFG" --seed "$SEED" \\
      --runs-dir "$RUNS_DIR" --device "$DEVICE" $EXTRA \\
      > "$LOGDIR/${CELL}_${DENSITY}_${intr}_seed${SEED}.log" 2>&1 &
  pids+=("$!"); echo "  launched $CELL/$DENSITY/$intr seed=$SEED (pid $!)"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ "$fail" == "0" ]] || { echo "[s$SEED | $CELL $DENSITY] a run FAILED — see $LOGDIR"; exit 1; }
echo "[s$SEED | $CELL $DENSITY] done."
"""

_RUN_ALL_TEMPLATE = """\
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/mmdt_*.sh; do
  bn="$(basename "$s")"; [[ "$bn" == "run_all.sh" ]] && continue
  echo "=== $bn ==="; bash "$s"
done
echo "All MortarMayhem-toggle jobs done."
"""


def main() -> None:
    cfg_dir = HERE / "configs"; scr_dir = HERE / "scripts"
    cfg_dir.mkdir(parents=True, exist_ok=True); scr_dir.mkdir(parents=True, exist_ok=True)
    for old in list(cfg_dir.glob("mmdt_*.yaml")) + list(scr_dir.glob("mmdt_*.sh")):
        old.unlink()

    for cell in CELLS:
        for density in DENSITIES:
            for intr in INTRINSICS:
                cfg = build_cfg(cell, density, intr)
                with open(cfg_dir / f"mmdt_{cell}_{density}_{intr}.yaml", "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)

    n_scr = 0
    for seed in SEEDS:
        for cell in CELLS:
            for density in DENSITIES:
                body = (_SCRIPT_TEMPLATE
                        .replace("__CELL__", cell)
                        .replace("__DENSITY__", density)
                        .replace("__SEED__", str(seed)))
                p = scr_dir / f"mmdt_s{seed}_{cell}_{density}.sh"
                p.write_text(body); os.chmod(p, 0o755); n_scr += 1
    ra = scr_dir / "run_all.sh"; ra.write_text(_RUN_ALL_TEMPLATE); os.chmod(ra, 0o755)

    n_cfg = len(CELLS) * len(DENSITIES) * len(INTRINSICS)
    print(f"env=MortarMayhem-Grid (cc={COMMAND_COUNT}, return-matched 1.0)  "
          f"project=memrl-mm-toggle  budget=10M")
    print(f"HPs transferred: e3b_idm λ=0.03 ck=64 lr=1e-4 | cells={CELLS}")
    print(f"wrote {n_cfg} configs + {n_scr} scripts (+ run_all.sh)")
    print(f"= {n_scr} GPU jobs (none+e3b pair each) = {n_cfg * len(SEEDS)} runs")


if __name__ == "__main__":
    main()
