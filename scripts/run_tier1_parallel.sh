#!/usr/bin/env bash
# Tier 1 sweep — PARALLEL variant.
#
# Launches all cells simultaneously, each capped to OMP_NUM_THREADS=2 so
# the M3's ~8 performance cores aren't oversubscribed (4 procs × 2 threads
# = 8 threads, leaving room for the OS + the DummyVecEnv worker threads).
#
# Wall-clock: roughly the same as one sequential 3M run (~85-100 min on M3)
# if the threading caps work out; up to 4× the sequential single-run time
# if there's contention (worst case still no slower than sequential 4-run).
#
# Per-process logs in /tmp/igam_tier1/<cell>_seed<n>.log.
# Run dirs in runs/popgym_repeat_previous_easy/<cell>/seed_<n>_<timestamp>/.

set -euo pipefail

SEED="${SEED:-0}"
CELLS="${CELLS:-LSTM Mamba2 mLSTM IGAM}"
CONFIG="${CONFIG:-benchmarks/phase_a/popgym_repeat_previous_easy.yaml}"
# Log dir defaults to /tmp/igam_tier1 but on Windows /tmp/ may not exist.
# Override with: LOG_DIR=./runs/_logs bash scripts/run_tier1_parallel.sh
LOG_DIR="${LOG_DIR:-/tmp/igam_tier1}"
mkdir -p "$LOG_DIR"

# Thread caps — applied per-process via env vars. PyTorch / NumPy / BLAS all
# honor these. 2 threads per process × 4 processes = 8 threads, fits M3.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
# Tell PyTorch explicitly too (belt-and-suspenders)
PY_THREAD_ENV="--no-stdin"  # placeholder, actual cap via env above

echo "=== Tier 1 PARALLEL sweep ==="
echo "Cells:   $CELLS"
echo "Seed:    $SEED"
echo "Config:  $CONFIG"
echo "Threads: $OMP_NUM_THREADS per process"
echo "Start:   $(date)"
echo

START_TIME=$(date +%s)

PIDS=()
for CELL in $CELLS; do
  LOG_FILE="$LOG_DIR/${CELL}_seed${SEED}.log"
  echo ">>> [$(date +%H:%M:%S)] Launching $CELL (background) ..."
  .venv/bin/python train.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --cell "$CELL" \
    > "$LOG_FILE" 2>&1 &
  PIDS+=($!)
done

echo
echo "All 4 launched. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."

# Wait for ALL processes to finish.
FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAILED=$((FAILED + 1))
  fi
done

END_TIME=$(date +%s)
TOTAL=$((END_TIME - START_TIME))
echo
echo "=== Tier 1 PARALLEL complete in $((TOTAL / 60)) min  ($FAILED failed) ==="
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
    eval_path = d + 'eval/evaluations.npz'
    if not os.path.exists(eval_path):
        print(f'  {cell}: incomplete (no eval/evaluations.npz)')
        continue
    z = np.load(eval_path)
    means = z['results'].mean(axis=1)
    best = means.max(); final = means[-1]
    print(f'  {cell:<14s}  best={best:+.3f}  final={final:+.3f}')
"
