#!/usr/bin/env bash
# Memento-F2 e3b λ-sweep, in the LEARNABLE regime (chunk_len = n_steps = 2048, ~3%
# straddle). The ck=ns=2048 "none" run (the discriminator, succ climbs >0.5) is the
# λ=0 anchor; this script launches the e3b arms at λ ∈ {0.003, 0.01, 0.03}, bracketing
# the ~0.01 march/diffuse crossover. Question: at a λ low enough to keep the agent
# marching (ep_len ~72), does e3b CLIMB like the none anchor (bonus neutral, harm was
# pure traversal/scale) or PLATEAU below it (bonus interferes with cue-learning even
# when traversal is fine)?  Dense (penalty_step=-0.01) so λ competes with the hurry
# gradient. NOTE: at ck128 every arm caps at the horizon wall ~0.5 — that's why this
# runs at ck=ns, matched to the discriminator, not 128.
#
# Usage (from the IGAM repo root, env active):
#   bash experiments/memF2_lambda_sweep/run_sweep.sh
# Override device / output dir / interpreter:
#   DEVICE=cpu RUNS_DIR=runs/sweep PYTHON=.venv/bin/python bash experiments/memF2_lambda_sweep/run_sweep.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
RUNS_DIR=${RUNS_DIR:-runs/spark}
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RUNS_DIR"

CONFIGS=(
  memF2_dense_gdn_e3b_ck2048_lam0.003   # below crossover — should march; does it learn?
  memF2_dense_gdn_e3b_ck2048_lam0.01    # at the crossover
  memF2_dense_gdn_e3b_ck2048_lam0.03    # above crossover — expected to diffuse → 0
)

for f in "${CONFIGS[@]}"; do
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} nohup "$PYTHON" train.py \
      --config "$DIR/$f.yaml" --device "$DEVICE" --no-wandb --runs-dir "$RUNS_DIR" \
      > "$RUNS_DIR/log_$f.txt" 2>&1 &
  echo "launched $f (pid $!)"
done
echo "launched ${#CONFIGS[@]} ck=ns e3b arms → $RUNS_DIR (λ=0 anchor = the running ck=ns 'none' discriminator)"
echo "NOTE: 3 more ck=2048 BPTT jobs on top of the discriminator is heavy GPU; run just lam0.003 first if contended."
