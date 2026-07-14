#!/usr/bin/env bash
# AUTO-GENERATED (PBIM3MPG). GatedDeltaNet/aligned/pbim_e3b_idm seed 0 — ONE run / one GPU, snapshots ON.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"; RUNS_DIR="${RUNS_DIR:-runs/memtrain_PBIM3MPG}"; EXTRA="${EXTRA:-}"
SNAP="${SNAPSHOT_STEPS:-500000,2000000,5000000,10000000,20000000}"; SNAP_WANDB="${SNAPSHOT_TO_WANDB:---snapshot-to-wandb}"
LOGDIR="$HERE/../../_logs/PBIM3MPG"; mkdir -p "$LOGDIR"
CFG="$HERE/../../configs/PBIM3MPG/GatedDeltaNet_aligned_pbim_e3b_idm.yaml"
echo "[PBIM3MPG] GatedDeltaNet/aligned/pbim_e3b_idm seed=0"
exec "$PYTHON" -m train --config "$CFG" --seed 0 \
    --runs-dir "$RUNS_DIR" --device "$DEVICE" \
    --wandb --snapshot-steps "$SNAP" $SNAP_WANDB $EXTRA \
    2>&1 | tee "$LOGDIR/GatedDeltaNet_aligned_pbim_e3b_idm_seed0.log"
