"""Generate the MiniGrid-MemoryS13 baseline study (configs + run scripts).

Two axes of the memory × exploration study:

  1. MEMORY-ONLY baseline (x-axis): 11 cells × intrinsic=none × 3 seeds = 33
     runs. How far does each cell get on S13 from memory alone, NO exploration.

  2. EXPLORATION arm (y-axis): {GRU, Memoryless} × {rnd, noveld, e3b_rand} × 3
     seeds = 18 runs. Does an intrinsic bonus help on this exploration-
     bottlenecked task? GRU = does exploration help a policy that CAN remember;
     Memoryless = negative control (no memory ⇒ can't solve S13 regardless).

Emits, under experiments/s13_baseline/:
  configs/s13_<cell>_none_<N>M.yaml        memory baseline, one per cell
  configs/s13_<cell>_<method>_<N>M.yaml    exploration arm, one per condition
  scripts/run_<cell>.sh                    memory baseline: a cell's 3 seeds AT
                                           ONCE (parallel); honors PYTHON/DEVICE/SEEDS
  scripts/run_all.sh                       runs every memory-baseline script
  scripts/explore_<cell>_<method>.sh       exploration arm: a condition's 3 seeds
  scripts/run_exploration.sh               runs every exploration-arm script

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
    "GRU", "LSTM", "mLSTM", "LRU", "Mamba2", "FFM",
    "GatedDeltaNet", "SHM", "GTrXL", "RetNet", "LinearTransformer",
]

# ── Exploration arm ─────────────────────────────────────────────────────────
# The SECOND axis of the study: does an intrinsic-reward exploration bonus help
# on S13 (which is exploration-bottlenecked)? We cross two backbones with three
# bonuses, 3 seeds each = 18 runs, on top of the 11-cell memory-only baseline.
#
#   Backbones (EXPLORATION_CELLS):
#     - GRU        : a competent memory cell. Tests whether exploration helps a
#                    policy that CAN remember overcome the exploration bottleneck
#                    (the real performance comparison).
#     - Memoryless : the stateless feedforward control. S13 REQUIRES recalling
#                    the start cue at the junction, which a memoryless policy
#                    structurally cannot do — so this is a NEGATIVE CONTROL that
#                    shows exploration alone (no memory) is insufficient.
#
#   Bonuses (EXPLORATION_METHODS) — verified against their papers (2026-05-29):
#     - rnd      : lifelong novelty = ‖predictor(x) − frozen_target(x)‖²
#                  (Burda et al. ICLR 2019).
#     - noveld   : hybrid = max(rnd(sₜ) − α·rnd(sₜ₋₁), 0) gated by first-visit
#                  (Zhang et al. NeurIPS 2021; α=0.5 default).
#     - e3b_rand : episodic elliptical/Mahalanobis bonus φᵀΛ⁻¹φ over a FROZEN
#                  random φ — cell-agnostic, runs identically on any backbone
#                  (Henaff et al. NeurIPS 2022; random-φ per the thesis finding).
EXPLORATION_CELLS = ["GRU", "Memoryless"]
EXPLORATION_METHODS = ["rnd", "noveld", "e3b_rand"]

# Intrinsic-reward weight for the exploration arm. train.py adds the bonus in a
# SINGLE reward stream: reward ← reward + λ · bonus (ppo.py:294). All three
# bonuses are running-std-normalized to ≈O(1) per step, and S13's extrinsic
# reward is a single sparse terminal ≈1.0, so with γ=0.999 over ~400–800-step
# episodes the discounted intrinsic return dominates the extrinsic for any
# λ ≳ 0.003. λ=0.01 keeps exploration as a strong-but-not-overwhelming early
# driver. NOTE: std-normalization means the normalized bonus does NOT auto-
# anneal to zero as novelty is learned — this λ is the primary knob to retune
# if the agent over-explores (lower it) or never explores (raise it).
LAMBDA_INTRINSIC = 0.01

# Per-cell hyperparameter overrides — applied AFTER the BASE HPs.
# Mamba-2 gets a halved learning rate per RLBenchNet (arXiv 2505.15040) and the
# Mamba-2 codebase's recommendation; at the default 3e-4 the SSM tends to be
# unstable in PPO. The rest of the cells use BASE['lr']. Shared PPO HPs are
# otherwise identical across cells — this is a param-matched comparison, so we
# do NOT per-cell-tune the optimizer; only the optimizer-stability exception
# above (Mamba-2) deviates, and it does so for a documented architecture reason.
PER_CELL_OVERRIDES: dict[str, dict] = {
    "Mamba2": {"lr": 1.5e-4},
}

# Per-cell CELL-KWARGS overrides — merged onto train.py's DEFAULT_CELL_KWARGS for
# THIS experiment only (the global defaults are left untouched so other configs
# don't drift). Used to widen the memory horizon of the cells with a HARD memory
# window so S13 can exercise their real capability:
#   - GTrXL  mem_len 64 → 128: the attention cache is an explicit window. Under
#     S13's random spawn, early-training traversals from cue-sighting to the
#     decision are long and erratic; with a 64-step cache the cue can fall OUT of
#     the window entirely, so attention never sees it and the credit signal that
#     teaches "remember the cue" never forms. 128 gives ~5× margin over the
#     ~13–25-step nominal recall span. Param-neutral: mem_len sizes only the
#     rel_pos buffer + detached KV cache, NOT any trainable weight.
# The decay/gated cells (LRU r_max=0.999 ≈ 1000-step, FFM max_timescale=1024,
# mLSTM/Mamba2/GatedDeltaNet/SHM/RetNet) have NO hard window — their horizon
# already far exceeds S13's recall span — so they keep their defaults.
# LinearTransformer is intentionally left ungated/decay-free (its defining
# property): no forgetting, so state can saturate on long episodes. That is the
# baseline reading we want from it, not something to "fix" via overrides.
PER_CELL_KWARGS_OVERRIDES: dict[str, dict] = {
    "GTrXL": {"mem_len": 128},
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


def build_cfg(cell: str, intrinsic: str = "none",
              lambda_intrinsic: float = 0.0) -> dict:
    if cell not in DEFAULT_CELL_KWARGS:
        raise KeyError(f"{cell} not in train.py DEFAULT_CELL_KWARGS")
    cfg = dict(BASE)
    cfg.update(PER_CELL_OVERRIDES.get(cell, {}))      # e.g. Mamba-2 lr=1.5e-4
    kwargs = dict(DEFAULT_CELL_KWARGS[cell])
    kwargs.update(PER_CELL_KWARGS_OVERRIDES.get(cell, {}))  # e.g. GTrXL mem_len, LMU theta
    cfg["cell"] = {"name": cell, "kwargs": kwargs}
    cfg["intrinsic"] = intrinsic
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
              "chunk_len", "n_chunks_per_batch", "intrinsic"):
        ordered[k] = cfg[k]
    # Intrinsic-reward weight: ONLY meaningful (and only written) for the
    # exploration arm. train.py defaults lambda_intrinsic=0.0, so an absent key
    # == bonus has zero weight == pure memory-only baseline. Writing it for the
    # "none" configs would be a silent no-op, so we omit it there.
    if intrinsic != "none" and lambda_intrinsic > 0:
        ordered["lambda_intrinsic"] = lambda_intrinsic
    for k in ("eval_every_rollouts", "n_eval_episodes", "wandb", "wandb_project"):
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
# AUTO-GENERATED. Run every per-cell MEMORY-BASELINE script (intrinsic=none):
# cells serial, seeds parallel within. Does NOT run the exploration arm — use
# run_exploration.sh for that. All env knobs (PYTHON/DEVICE/RUNS_DIR/SEEDS/
# PARALLEL) propagate to each script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/run_*.sh; do
  bn="$(basename "$s")"
  # Skip the aggregators themselves (run_*.sh also matches run_exploration.sh).
  [[ "$bn" == "run_all.sh" || "$bn" == "run_exploration.sh" ]] && continue
  echo "=== $bn ==="
  bash "$s"
done
echo "All cells done."
"""

# Exploration-arm per-condition launcher. Mirrors _SCRIPT_TEMPLATE but the
# config path carries the intrinsic method and the file is named
# explore_<cell>_<method>.sh so run_all.sh's run_*.sh glob never picks it up.
_EXPLORE_SCRIPT_TEMPLATE = """\
#!/usr/bin/env bash
# AUTO-GENERATED by generate_configs.py — regenerate, don't hand-edit.
# Exploration arm: __CELL__ + __INTRINSIC__ on MiniGrid-MemoryS13, all seeds
# AT ONCE (parallel). Same env-var knobs as the memory-baseline run_*.sh.
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
CFG="$HERE/../configs/s13___CELL_____INTRINSIC_____SUFFIX__.yaml"
LOGDIR="$HERE/../_logs"; mkdir -p "$LOGDIR"
TAG="__CELL_____INTRINSIC__"

echo "[$TAG] python=$PYTHON device=$DEVICE seeds=[$SEEDS] parallel=$PARALLEL"
pids=()
for s in $SEEDS; do
  if [[ "$PARALLEL" == "1" ]]; then
    "$PYTHON" -m train --config "$CFG" --seed "$s" \\
        --runs-dir "$RUNS_DIR" --device "$DEVICE" $EXTRA \\
        > "$LOGDIR/${TAG}_seed${s}.log" 2>&1 &
    pids+=("$!")
    echo "  launched $TAG seed=$s (pid $!) → $LOGDIR/${TAG}_seed${s}.log"
  else
    echo "  $TAG seed=$s (sequential)"
    "$PYTHON" -m train --config "$CFG" --seed "$s" \\
        --runs-dir "$RUNS_DIR" --device "$DEVICE" $EXTRA
  fi
done
if [[ "$PARALLEL" == "1" ]]; then
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ "$fail" == "0" ]] || { echo "[$TAG] a seed FAILED — see $LOGDIR"; exit 1; }
fi
echo "[$TAG] all seeds done."
"""

_RUN_EXPLORATION_TEMPLATE = """\
#!/usr/bin/env bash
# AUTO-GENERATED. Run every exploration-arm script (explore_*.sh): conditions
# serial, seeds parallel within. All env knobs propagate to each script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/explore_*.sh; do
  echo "=== $(basename "$s") ==="
  bash "$s"
done
echo "All exploration conditions done."
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


def write_explore_scripts(conditions: list[tuple[str, str]], suffix: str) -> list[Path]:
    """One explore_<cell>_<method>.sh per exploration condition + run_exploration.sh."""
    scripts_dir = HERE / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for cell, method in conditions:
        body = (_EXPLORE_SCRIPT_TEMPLATE
                .replace("__CELL__", cell)
                .replace("__INTRINSIC__", method)
                .replace("__SUFFIX__", suffix))
        path = scripts_dir / f"explore_{cell}_{method}.sh"
        path.write_text(body)
        os.chmod(path, 0o755)
        written.append(path)
    run_expl = scripts_dir / "run_exploration.sh"
    run_expl.write_text(_RUN_EXPLORATION_TEMPLATE)
    os.chmod(run_expl, 0o755)
    written.append(run_expl)
    return written


def main() -> None:
    suffix = f"{BASE['total_timesteps'] // 1_000_000}M"

    out_dir = HERE / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Memory-only baseline: 11 cells × intrinsic=none ─────────────────────
    cfgs = []
    for cell in CELLS:
        cfg = build_cfg(cell)
        path = out_dir / f"s13_{cell}_none_{suffix}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        cfgs.append(path)
    scripts = write_scripts(CELLS, suffix)

    # ── Exploration arm: {GRU, Memoryless} × {rnd, noveld, e3b_rand} ────────
    explore_conditions = [
        (cell, method)
        for cell in EXPLORATION_CELLS
        for method in EXPLORATION_METHODS
    ]
    expl_cfgs = []
    for cell, method in explore_conditions:
        cfg = build_cfg(cell, intrinsic=method, lambda_intrinsic=LAMBDA_INTRINSIC)
        path = out_dir / f"s13_{cell}_{method}_{suffix}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        expl_cfgs.append(path)
    expl_scripts = write_explore_scripts(explore_conditions, suffix)

    print(f"Wrote {len(cfgs)} memory-baseline configs → {out_dir} (suffix {suffix}):")
    for p in cfgs:
        print(f"  {p.name}")
    print(f"\nWrote {len(expl_cfgs)} exploration-arm configs:")
    for p in expl_cfgs:
        print(f"  {p.name}")
    print(f"\nWrote {len(scripts)} memory-baseline scripts → {HERE / 'scripts'}:")
    for p in scripts:
        print(f"  {p.name}")
    print(f"\nWrote {len(expl_scripts)} exploration-arm scripts:")
    for p in expl_scripts:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
