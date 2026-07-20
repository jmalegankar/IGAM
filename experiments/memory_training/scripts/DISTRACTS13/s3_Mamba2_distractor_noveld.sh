#!/usr/bin/env bash
# AUTO-GENERATED (DISTRACTS13). Mamba2/distractor/noveld seed 3 — ONE run / one GPU, snapshots ON.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"; RUNS_DIR="${RUNS_DIR:-runs/memtrain_DISTRACTS13}"; EXTRA="${EXTRA:-}"
SNAP="${SNAPSHOT_STEPS:-500000,2000000,5000000,10000000,20000000}"; SNAP_WANDB="${SNAPSHOT_TO_WANDB:---snapshot-to-wandb}"
LOGDIR="$HERE/../../_logs/DISTRACTS13"; mkdir -p "$LOGDIR"
CFG="$HERE/../../configs/DISTRACTS13/Mamba2_distractor_noveld.yaml"
echo "[DISTRACTS13] Mamba2/distractor/noveld seed=3"
exec "$PYTHON" -m train --config "$CFG" --seed 3 \
    --runs-dir "$RUNS_DIR" --device "$DEVICE" \
    --wandb --snapshot-steps "$SNAP" $SNAP_WANDB $EXTRA \
    2>&1 | tee "$LOGDIR/Mamba2_distractor_noveld_seed3.log"
