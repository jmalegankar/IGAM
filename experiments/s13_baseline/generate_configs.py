"""Generate the MiniGrid-MemoryS13 *memory-only* baseline (configs + run scripts).

One YAML per cell + one launch script per cell. This is the "all cells + PPO"
baseline — the x-axis of the memory × exploration study: how far does each
cell get on S13 from memory alone, with NO exploration bonus. The exploration
columns (rnd / e3b_rand / noveld / icm / e3b_obs) are a separate follow-up.

Emits, under experiments/s13_baseline/:
  configs/s13_<cell>_none_<N>M.yaml   one per cell (seed set at launch)
  scripts/run_<cell>.sh               one per cell — runs its 3 seeds AT ONCE
                                       (in parallel); honors PYTHON/DEVICE/SEEDS
  scripts/run_all.sh                  runs every per-cell script (cells serial,
                                       seeds parallel within each)

Design choices (and why):
  - HPs follow the thesis S13 reference (gex full_system.yaml), reviewed for this
    baseline. γ=0.999 (effective horizon ~1000 ≥ max_steps 845, so the terminal
    reward propagates), GAE λ=0.98, vf_coef=1.0 (separate critic backbone ⇒ no
    policy/value gradient fight + hard sparse value target), ent 0.008 (modest on
    purpose: entropy is the baseline's ONLY exploration driver — kept low so the
    "memory-only" reading stays clean). Two deliberate changes vs the thesis:
      * TBPTT chunk_len 16 → 32 — covers S13's ~13–25-step cue→decision recall
        span so the gradient actually connects cue-encoding to the decision (16
        straddles it). Full-episode BPTT is infeasible (max_steps=845).
      * total_timesteps 5M → 10M — memory-only on an exploration-bottlenecked
        task converges slowly; 10M avoids under-training confounding the cell
        comparison. eval cadence widened (every 5 rollouts, 20 eps) so eval cost
        stays ~5% rather than ballooning to ~1200 evals.
  - Per-cell kwargs imported from train.py's DEFAULT_CELL_KWARGS — the SAME table
    a normal `train.py --cell <X>` run uses (Mamba2 d_state=128 for param parity,
    etc.). No drift possible.
  - NO exploration / wrapper keys are written. Current train.py drives exploration
    solely via `intrinsic`; the MiniGrid MemoryStartWrapper is not wired, so S13
    runs with RANDOM SPAWN = exploration bottleneck ON — the regime where a
    baseline cell must both explore back to the hint room AND remember the cue.

Usage:
    python experiments/s13_baseline/generate_configs.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from train import DEFAULT_CELL_KWARGS  # noqa: E402  (canonical per-cell kwargs)

# The 10 PUBLISHED baseline cells (no cells we made — DTHLMU/GatedLMU/SelectiveLMU
# are excluded; this batch is the "established architectures" baseline). Weak→
# strong memory, param-matched (~0.88–1.5M params), all verified to build + run
# on S13. Edit to add more from train.py's CELL_REGISTRY (e.g. S4D, DeltaNet,
# RetNet, LinearTransformer).
CELLS = [
    "GRU", "LSTM", "mLSTM", "LMU", "LRU", "Mamba2", "FFM",
    "GatedDeltaNet", "SHM", "GTrXL",
]

# Per-cell hyperparameter overrides — applied AFTER the BASE HPs.
# Mamba-2 gets a halved learning rate per RLBenchNet (arXiv 2505.15040) and the
# Mamba-2 codebase's recommendation; at the default 3e-4 the SSM tends to be
# unstable in PPO. The rest of the cells use BASE['lr'].
PER_CELL_OVERRIDES: dict[str, dict] = {
    "Mamba2": {"lr": 1.5e-4},
}

# S13 reference HPs (thesis full_system.yaml, exploration-key-free).
BASE = {
    "env_name": "MiniGrid-MemoryS13-v0",
    "n_envs": 16,
    "total_timesteps": 10_000_000,
    "seed": 0,                      # overridden per run via train.py --seed
    "encoder_dim": 128,
    "encoder_hidden": 256,
    "lr": 3.0e-4,
    "n_steps": 512,
    "n_epochs": 4,
    "gamma": 0.999,
    "gae_lambda": 0.98,
    "clip_range": 0.2,
    "ent_coef": 0.008,
    "vf_coef": 1.0,
    "max_grad_norm": 0.5,
    "target_kl": 0.05,
    # TBPTT window. Raised from the thesis's 16 → 32: S13's cue→decision recall
    # span is the corridor traversal (~13–25 steps on a 13×13 grid), so a 16-step
    # gradient window often straddles the cue-encoding and the decision, starving
    # the credit path that teaches the cell to STORE the cue. 32 covers the span
    # with margin; full-episode BPTT is infeasible (max_steps=845). The hidden
    # state still carries across chunks at inference; only the gradient truncates.
    "chunk_len": 32,
    # 8192 transitions/rollout ÷ 32 = 256 chunks → 8 minibatches of 32 chunks
    # (1024 transitions each), a healthy PPO minibatch.
    "n_chunks_per_batch": 32,
    "intrinsic": "none",            # memory-only baseline
    # Eval cadence: every 5 rollouts (~40k env-steps) × 20 episodes. At 10M that's
    # ~244 evals — good curve resolution at ~5% overhead (vs ~1200 evals/quadratic
    # cost if left at every-rollout). 20 eps tames S13's random-spawn eval variance.
    "eval_every_rollouts": 5,
    "n_eval_episodes": 20,
    # Weights & Biases: on by default for these cloud-handoff configs. All TB
    # scalars are mirrored (sync_tensorboard). Disable per-run with --no-wandb.
    "wandb": True,
    "wandb_project": "memrl-s13-baseline",
}


def build_cfg(cell: str) -> dict:
    if cell not in DEFAULT_CELL_KWARGS:
        raise KeyError(f"{cell} not in train.py DEFAULT_CELL_KWARGS")
    cfg = dict(BASE)
    cfg.update(PER_CELL_OVERRIDES.get(cell, {}))      # e.g. Mamba-2 lr=1.5e-4
    cfg["cell"] = {"name": cell, "kwargs": dict(DEFAULT_CELL_KWARGS[cell])}
    # Order keys so the file reads cleanly: identity, then cell, then HPs.
    ordered = {
        "env_name": cfg["env_name"],
        "n_envs": cfg["n_envs"],
        "total_timesteps": cfg["total_timesteps"],
        "seed": cfg["seed"],
        "cell": cfg["cell"],
        "encoder_dim": cfg["encoder_dim"],
        "encoder_hidden": cfg["encoder_hidden"],
    }
    for k in ("lr", "n_steps", "n_epochs", "gamma", "gae_lambda", "clip_range",
              "ent_coef", "vf_coef", "max_grad_norm", "target_kl",
              "chunk_len", "n_chunks_per_batch", "intrinsic",
              "eval_every_rollouts", "n_eval_episodes",
              "wandb", "wandb_project"):
        ordered[k] = cfg[k]
    return ordered


# Per-cell launch script. Placeholders __CELL__ / __SUFFIX__ filled by .replace()
# (avoids clashing python braces with bash ${...}).
_SCRIPT_TEMPLATE = """\
#!/usr/bin/env bash
# AUTO-GENERATED by generate_configs.py — regenerate, don't hand-edit.
# Run __CELL__ on MiniGrid-MemoryS13 for its seeds, ALL AT ONCE (parallel).
# Knobs (env vars):
#   PYTHON=.venv/bin/python   interpreter (default: repo .venv if present)
#   DEVICE=cuda               torch device (default cuda)
#   RUNS_DIR=runs/s13_baseline   output root
#   SEEDS="0 1 2"             seeds to run
#   PARALLEL=1                1 = seeds concurrently (default); 0 = sequential
#   EXTRA="--no-wandb"        extra train.py flags (e.g. --total-timesteps 8192)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"
RUNS_DIR="${RUNS_DIR:-runs/s13_baseline}"
SEEDS="${SEEDS:-0 1 2}"
PARALLEL="${PARALLEL:-1}"
EXTRA="${EXTRA:-}"
CFG="$HERE/../configs/s13___CELL___none___SUFFIX__.yaml"
LOGDIR="$HERE/../_logs"; mkdir -p "$LOGDIR"

echo "[__CELL__] python=$PYTHON device=$DEVICE seeds=[$SEEDS] parallel=$PARALLEL"
pids=()
for s in $SEEDS; do
  if [[ "$PARALLEL" == "1" ]]; then
    "$PYTHON" -m train --config "$CFG" --seed "$s" \\
        --runs-dir "$RUNS_DIR" --device "$DEVICE" $EXTRA \\
        > "$LOGDIR/__CELL___seed${s}.log" 2>&1 &
    pids+=("$!")
    echo "  launched __CELL__ seed=$s (pid $!) → $LOGDIR/__CELL___seed${s}.log"
  else
    echo "  __CELL__ seed=$s (sequential)"
    "$PYTHON" -m train --config "$CFG" --seed "$s" \\
        --runs-dir "$RUNS_DIR" --device "$DEVICE" $EXTRA
  fi
done
if [[ "$PARALLEL" == "1" ]]; then
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ "$fail" == "0" ]] || { echo "[__CELL__] a seed FAILED — see $LOGDIR"; exit 1; }
fi
echo "[__CELL__] all seeds done."
"""

_RUN_ALL_TEMPLATE = """\
#!/usr/bin/env bash
# AUTO-GENERATED. Run every per-cell script: cells serial, seeds parallel within.
# All env knobs (PYTHON/DEVICE/RUNS_DIR/SEEDS/PARALLEL) propagate to each script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/run_*.sh; do
  [[ "$(basename "$s")" == "run_all.sh" ]] && continue
  echo "=== $(basename "$s") ==="
  bash "$s"
done
echo "All cells done."
"""


def write_scripts(cells: list[str], suffix: str) -> list[Path]:
    scripts_dir = HERE / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for cell in cells:
        body = _SCRIPT_TEMPLATE.replace("__CELL__", cell).replace("__SUFFIX__", suffix)
        path = scripts_dir / f"run_{cell}.sh"
        path.write_text(body)
        os.chmod(path, 0o755)
        written.append(path)
    run_all = scripts_dir / "run_all.sh"
    run_all.write_text(_RUN_ALL_TEMPLATE)
    os.chmod(run_all, 0o755)
    written.append(run_all)
    return written


def main() -> None:
    suffix = f"{BASE['total_timesteps'] // 1_000_000}M"

    out_dir = HERE / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfgs = []
    for cell in CELLS:
        cfg = build_cfg(cell)
        path = out_dir / f"s13_{cell}_none_{suffix}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        cfgs.append(path)

    scripts = write_scripts(CELLS, suffix)

    print(f"Wrote {len(cfgs)} configs → {out_dir} (suffix {suffix}):")
    for p in cfgs:
        print(f"  {p.name}")
    print(f"\nWrote {len(scripts)} scripts → {HERE / 'scripts'}:")
    for p in scripts:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
