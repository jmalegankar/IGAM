#!/usr/bin/env bash
# Memento-F2 e3b λ-sweep, in the LEARNABLE regime (chunk_len = n_steps = 2048, ~3%
# straddle). The ck=ns=2048 "none" run (the discriminator, succ climbs >0.5) is the
# λ=0 anchor; this script runs the e3b arms at λ ∈ {0.003, 0.01, 0.03}, bracketing the
# ~0.01 march/diffuse crossover. Question: at a λ low enough to keep the agent marching
# (ep_len ~72), does e3b CLIMB like the none anchor (bonus neutral, harm was pure
# traversal/scale) or PLATEAU below it (bonus interferes with cue-learning even when
# traversal is fine)?  Dense (penalty_step=-0.01) so λ competes with the hurry gradient.
#
# Runs SEQUENTIALLY (one ck=2048 BPTT job at a time) so it's light next to the
# discriminator on a single GPU. lam0.003 is first — the decisive arm — so an early
# stop still yields the key result.
#
# Usage (from the IGAM repo root, env active). Detach the whole sequence with nohup:
#   nohup bash experiments/memF2_lambda_sweep/run_sweep.sh > runs/spark/sweep.log 2>&1 &
# Override device / output dir / interpreter:
#   DEVICE=cpu RUNS_DIR=runs/sweep PYTHON=.venv/bin/python bash experiments/memF2_lambda_sweep/run_sweep.sh
set -uo pipefail

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
RUNS_DIR=${RUNS_DIR:-runs/spark}
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RUNS_DIR"

# The MARCHING arms (total 20M) — these reach the fork and sit at the cue-blind 0.5
# plateau; the question now is whether they GROK the cue with much longer training (on S13,
# GDN-none sat at 0.545 until ~17M then finished 0.98). Resume to keep the ~3M already done:
#   train.py --resume-from runs/spark/<run_name>/<cell>/<seed_dir> --total-timesteps 20000000 ...
CONFIGS=(
  memF2_dense_gdn_noveld_ck2048_lam0.03   # marches (ep_len ~195) at standard λ — best grok candidate
  memF2_dense_gdn_e3b_ck2048_lam0.001     # marches (escapes basin, crossover ~0.002) — grok?
  memF2_dense_gdn_noveld_ck2048_lam0.003  # marches — grok?
  # DIFFUSED (collapsed to ep_len ~1037, succ 0): e3b lam0.003/0.01/0.03 — done, NOT re-run.
)

echo "running ${#CONFIGS[@]} ck=ns e3b arms SEQUENTIALLY → $RUNS_DIR"
echo "(λ=0 anchor = the running ck=ns 'none' discriminator)"
for f in "${CONFIGS[@]}"; do
  echo "=== [$(date +%H:%M:%S)] START $f ==="
  if OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} "$PYTHON" train.py \
        --config "$DIR/$f.yaml" --device "$DEVICE" --no-wandb --runs-dir "$RUNS_DIR" \
        > "$RUNS_DIR/log_$f.txt" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] DONE  $f ==="
  else
    echo "=== [$(date +%H:%M:%S)] FAILED $f (rc=$?) — continuing to next ==="
  fi
done
echo "sweep complete."
