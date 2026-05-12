#!/usr/bin/env bash
# Tier 1 4-cell ablation on POPGym-RepeatPrevious-Easy.
#
# Configuration: Option A (separate backbones, per Ni 2022 / SB3 default)
# Cells: LSTM, Mamba2, mLSTM, IGAM (== GatedDeltaNet)
# Steps:  3M per cell × 4 cells = ~6 hours wall-clock sequential on Apple M3
#
# Run sequentially (M3 CPU saturates if parallel; sequential is faster total).
# Each run writes to runs/popgym_repeat_previous_easy/<cell>/seed_<n>_<timestamp>/
# with config.yaml + tensorboard logs + eval/evaluations.npz + best_model.zip.
#
# Usage:
#   bash scripts/run_tier1.sh                  # default: seed 0, all 4 cells
#   SEED=1 bash scripts/run_tier1.sh           # override seed
#   CELLS="LSTM IGAM" bash scripts/run_tier1.sh   # only run a subset

set -euo pipefail

SEED="${SEED:-0}"
CELLS="${CELLS:-LSTM Mamba2 mLSTM IGAM}"
CONFIG="${CONFIG:-benchmarks/phase_a/popgym_repeat_previous_easy.yaml}"
LOG_DIR="${LOG_DIR:-/tmp/igam_tier1}"
mkdir -p "$LOG_DIR"

echo "=== Tier 1 sweep ==="
echo "Cells:       $CELLS"
echo "Seed:        $SEED"
echo "Config:      $CONFIG"
echo "Log dir:     $LOG_DIR"
echo "Start:       $(date)"
echo

START_TIME=$(date +%s)

for CELL in $CELLS; do
  CELL_START=$(date +%s)
  LOG_FILE="$LOG_DIR/${CELL}_seed${SEED}.log"
  echo ">>> [$(date +%H:%M:%S)] Launching $CELL ..."
  .venv/bin/python train.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --cell "$CELL" \
    > "$LOG_FILE" 2>&1
  CELL_END=$(date +%s)
  ELAPSED=$((CELL_END - CELL_START))
  echo "    done in $((ELAPSED / 60)) min  (log: $LOG_FILE)"
done

END_TIME=$(date +%s)
TOTAL=$((END_TIME - START_TIME))
echo
echo "=== Tier 1 sweep complete in $((TOTAL / 60)) min ==="
echo "Final eval rewards:"
.venv/bin/python -c "
import numpy as np, glob, os
for cell in '$CELLS'.split():
    pat = f'runs/popgym_repeat_previous_easy/{cell}/seed_${SEED}_*/'
    matches = sorted(glob.glob(pat), key=lambda p: os.path.getmtime(p))
    if not matches:
        print(f'  {cell}: no run found')
        continue
    d = matches[-1]
    z = np.load(d + 'eval/evaluations.npz')
    means = z['results'].mean(axis=1)
    best = means.max(); final = means[-1]
    print(f'  {cell:<14s}  best={best:+.3f}  final={final:+.3f}')
"
