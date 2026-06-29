#!/usr/bin/env bash
# Memento-F2 e3b λ-sweep: does the e3b bonus HURT by overpowering corridor traversal
# (a scale/tuning effect, recoverable by lowering λ) or by a genuine task-level
# interaction? Dense regime (penalty_step=-0.01 "hurry" gradient is present so λ has
# something to lose to), GDN, chunk_len 128, varying ONLY lambda_intrinsic.
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
  memF2_dense_gdn_none_ck128          # λ=0 anchor
  memF2_dense_gdn_e3b_ck128_lam0.001
  memF2_dense_gdn_e3b_ck128_lam0.003
  memF2_dense_gdn_e3b_ck128_lam0.01
  memF2_dense_gdn_e3b_ck128_lam0.03
)

for f in "${CONFIGS[@]}"; do
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} nohup "$PYTHON" train.py \
      --config "$DIR/$f.yaml" --device "$DEVICE" --no-wandb --runs-dir "$RUNS_DIR" \
      > "$RUNS_DIR/log_$f.txt" 2>&1 &
  echo "launched $f (pid $!)"
done
echo "all ${#CONFIGS[@]} λ-sweep runs launched → $RUNS_DIR (tensorboard --logdir $RUNS_DIR)"
