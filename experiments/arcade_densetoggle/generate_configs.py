"""POPGym-Arcade density toggle — the REVERSE flip + the controllability control.

MysteryPath toggles sparse→dense; here we toggle pixel POPGym-Arcade tasks
dense→SPARSE via DeferredReward (all per-step reward withheld, paid as one
terminal lump: identical dynamics/observations/memory demand and episode
return — only the payment schedule changes).

Why Arcade over vector POPGym (docs/revelation_and_densification.md):
  * obs-richness is MATCHED across the controllability contrast (both tasks
    are 84x84x3 screens), so "novelty is uncontrollable" is not confounded
    with "obs space is tiny" (vector RepeatPrevious's Discrete(4) starved
    e3b's ellipsoid for a degenerate reason);
  * e3b operates in its native pixel regime, same PixelEncoder + e3b config
    as the MysteryPath headline — only the task family changes;
  * the suite's partial_obs flag gives the B5 observability counterfactual
    on the same tasks later, for free.

Two tasks, two predictions:
  * BattleShipEasy  — CONTROLLABLE revelation (you steer which cells get
    probed; ~8% of steps pay natively ≈ 10 events/ep). Prediction: the
    reverse flip — e3b > none on deferred-sparse, e3b ≲ none on native.
  * CountRecallEasy — UNCONTROLLABLE revelation (the dealt stream is
    action-independent; natively already quasi-sparse ~2%). Prediction:
    e3b ≈ none in BOTH arms — sparsity alone is not sufficient for a bonus
    to help; controllable revelation is (alignment α ≈ 0).

Design: 2 tasks × {dense=native, sparse=deferred} × {none, e3b_idm} ×
{GRU, GatedDeltaNet} × 5 seeds = 80 runs.
Packing: per (task, cell, density, seed) one GPU runs the {none, e3b} PAIR
in parallel — obs resized to 84x84 for exact memory/encoder parity with the
validated MysteryPath 2-per-GPU profile → 40 jobs.

HPs: transferred from the MysteryPath winner (e3b_idm, λ=0.03, ck=64, lr=1e-4)
— NOT retuned, consistent with the paper's global-HP fair-comparison story.

Prereq: the cluster image must bake jax + popgym-arcade (see Dockerfile).

Usage:
    python experiments/arcade_densetoggle/generate_configs.py
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

TASKS = {
    # short label -> full env id (popgym-arcade- prefix routes the factory)
    "BattleShip": "popgym-arcade-BattleShipEasy",
    "CountRecall": "popgym-arcade-CountRecallEasy",
}
CELLS = ["GRU", "GatedDeltaNet"]
DENSITIES = ["dense", "sparse"]          # dense = native; sparse = deferred
INTRINSICS = ["none", "e3b_idm"]
SEEDS = [0, 1, 2, 3, 4]
# partial_obs=True is the memory-demanding POMDP variant; resize_to=84 gives
# exact parity with the MysteryPath encoder + GPU-packing profile.
BASE_ENV_KWARGS = {"partial_obs": True, "resize_to": 84}

BASE = {
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
    "wandb_project": "memrl-arcade-toggle",
}


def build_cfg(task_label: str, cell: str, density: str, intrinsic: str) -> dict:
    env_kwargs = dict(BASE_ENV_KWARGS)
    if density == "sparse":
        env_kwargs["defer_reward"] = True
    ordered = {
        "env_name": TASKS[task_label],
        "env_kwargs": env_kwargs,
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
    ordered["run_name"] = f"{task_label}-{cell}-{density}-{intrinsic}"
    return ordered


_SCRIPT_TEMPLATE = """\
#!/usr/bin/env bash
# AUTO-GENERATED. Arcade density toggle: __TASK__ / __CELL__ / __DENSITY__ at
# seed __SEED__ — runs {none, e3b_idm} in PARALLEL on one GPU (84x84 parity
# with the validated MysteryPath 2-per-GPU packing).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"; RUNS_DIR="${RUNS_DIR:-runs/arcade_densetoggle}"; EXTRA="${EXTRA:-}"
LOGDIR="$HERE/../_logs"; mkdir -p "$LOGDIR"
TASK="__TASK__"; CELL="__CELL__"; DENSITY="__DENSITY__"; SEED="__SEED__"
echo "[s$SEED | $TASK $CELL $DENSITY] none + e3b_idm in parallel on one GPU"
pids=()
for intr in none e3b_idm; do
  CFG="$HERE/../configs/arcdt_${TASK}_${CELL}_${DENSITY}_${intr}.yaml"
  "$PYTHON" -m train --config "$CFG" --seed "$SEED" \\
      --runs-dir "$RUNS_DIR" --device "$DEVICE" $EXTRA \\
      > "$LOGDIR/${TASK}_${CELL}_${DENSITY}_${intr}_seed${SEED}.log" 2>&1 &
  pids+=("$!"); echo "  launched $TASK/$CELL/$DENSITY/$intr seed=$SEED (pid $!)"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ "$fail" == "0" ]] || { echo "[s$SEED | $TASK $CELL $DENSITY] a run FAILED — see $LOGDIR"; exit 1; }
echo "[s$SEED | $TASK $CELL $DENSITY] done."
"""

_RUN_ALL_TEMPLATE = """\
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/arcdt_*.sh; do
  bn="$(basename "$s")"; [[ "$bn" == "run_all.sh" ]] && continue
  echo "=== $bn ==="; bash "$s"
done
echo "All arcade-toggle jobs done."
"""


def main() -> None:
    cfg_dir = HERE / "configs"; scr_dir = HERE / "scripts"
    cfg_dir.mkdir(parents=True, exist_ok=True); scr_dir.mkdir(parents=True, exist_ok=True)
    for old in list(cfg_dir.glob("arcdt_*.yaml")) + list(scr_dir.glob("arcdt_*.sh")):
        old.unlink()

    for task in TASKS:
        for cell in CELLS:
            for density in DENSITIES:
                for intr in INTRINSICS:
                    cfg = build_cfg(task, cell, density, intr)
                    with open(cfg_dir / f"arcdt_{task}_{cell}_{density}_{intr}.yaml", "w") as f:
                        yaml.safe_dump(cfg, f, sort_keys=False)

    n_scr = 0
    for seed in SEEDS:
        for task in TASKS:
            for cell in CELLS:
                for density in DENSITIES:
                    body = (_SCRIPT_TEMPLATE
                            .replace("__TASK__", task)
                            .replace("__CELL__", cell)
                            .replace("__DENSITY__", density)
                            .replace("__SEED__", str(seed)))
                    p = scr_dir / f"arcdt_s{seed}_{cell}_{task}_{density}.sh"
                    p.write_text(body); os.chmod(p, 0o755); n_scr += 1
    ra = scr_dir / "run_all.sh"; ra.write_text(_RUN_ALL_TEMPLATE); os.chmod(ra, 0o755)

    n_cfg = len(TASKS) * len(CELLS) * len(DENSITIES) * len(INTRINSICS)
    print("envs=BattleShipEasy+CountRecallEasy (arcade, partial_obs, 84x84)")
    print("project=memrl-arcade-toggle  budget=10M")
    print(f"HPs transferred: e3b_idm λ=0.03 ck=64 lr=1e-4 | cells={CELLS}")
    print(f"wrote {n_cfg} configs + {n_scr} scripts (+ run_all.sh)")
    print(f"= {n_scr} GPU jobs (none+e3b pair each) = {n_cfg * len(SEEDS)} runs")


if __name__ == "__main__":
    main()
