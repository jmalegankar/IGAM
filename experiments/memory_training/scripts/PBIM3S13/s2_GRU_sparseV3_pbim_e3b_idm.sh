#!/usr/bin/env bash
# AUTO-GENERATED (PBIM3S13). GRU/sparseV3/pbim_e3b_idm seed 2 — ONE run / one GPU, snapshots ON.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"; RUNS_DIR="${RUNS_DIR:-runs/memtrain_PBIM3S13}"; EXTRA="${EXTRA:-}"
SNAP="${SNAPSHOT_STEPS:-500000,2000000,5000000,10000000,20000000}"; SNAP_WANDB="${SNAPSHOT_TO_WANDB:---snapshot-to-wandb}"
LOGDIR="$HERE/../../_logs/PBIM3S13"; mkdir -p "$LOGDIR"
CFG="$HERE/../../configs/PBIM3S13/GRU_sparseV3_pbim_e3b_idm.yaml"
echo "[PBIM3S13] GRU/sparseV3/pbim_e3b_idm seed=2"
exec "$PYTHON" -m train --config "$CFG" --seed 2 \
    --runs-dir "$RUNS_DIR" --device "$DEVICE" \
    --wandb --snapshot-steps "$SNAP" $SNAP_WANDB $EXTRA \
    2>&1 | tee "$LOGDIR/GRU_sparseV3_pbim_e3b_idm_seed2.log"
