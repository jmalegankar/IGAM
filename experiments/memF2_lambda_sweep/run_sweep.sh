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

CONFIGS=(
  memF2_dense_gdn_noveld_ck2048_lam0.03   # NovelD standard — does its first-visit/RND-diff align
                                          # forward (march+learn) or collapse to diffusion like e3b?
  memF2_dense_gdn_e3b_ck2048_lam0.001     # e3b DECISIVE: ~33x below the hurry — any λ escape the basin?
  memF2_dense_gdn_noveld_ck2048_lam0.003  # NovelD low λ
  memF2_dense_gdn_e3b_ck2048_lam0.003     # e3b clean re-confirm (parallel run showed collapse)
  # e3b lam0.01 / lam0.03 already shown to collapse (parallel) — files kept, not re-run by default
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
