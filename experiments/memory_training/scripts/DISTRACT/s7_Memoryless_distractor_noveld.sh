#!/usr/bin/env bash
# AUTO-GENERATED (DISTRACT). Memoryless/distractor/noveld seed 7 — ONE run / one GPU, snapshots ON.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"; RUNS_DIR="${RUNS_DIR:-runs/memtrain_DISTRACT}"; EXTRA="${EXTRA:-}"
SNAP="${SNAPSHOT_STEPS:-500000,2000000,5000000,10000000,20000000}"; SNAP_WANDB="${SNAPSHOT_TO_WANDB:---snapshot-to-wandb}"
LOGDIR="$HERE/../../_logs/DISTRACT"; mkdir -p "$LOGDIR"
CFG="$HERE/../../configs/DISTRACT/Memoryless_distractor_noveld.yaml"
echo "[DISTRACT] Memoryless/distractor/noveld seed=7"
exec "$PYTHON" -m train --config "$CFG" --seed 7 \
    --runs-dir "$RUNS_DIR" --device "$DEVICE" \
    --wandb --snapshot-steps "$SNAP" $SNAP_WANDB $EXTRA \
    2>&1 | tee "$LOGDIR/Memoryless_distractor_noveld_seed7.log"
