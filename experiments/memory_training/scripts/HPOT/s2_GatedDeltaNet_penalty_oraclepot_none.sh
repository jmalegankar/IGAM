#!/usr/bin/env bash
# AUTO-GENERATED (HPOT). GatedDeltaNet/penalty_oraclepot/none seed 2 — ONE run / one GPU, snapshots ON.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"; RUNS_DIR="${RUNS_DIR:-runs/memtrain_HPOT}"; EXTRA="${EXTRA:-}"
SNAP="${SNAPSHOT_STEPS:-500000,2000000,5000000,10000000,20000000}"; SNAP_WANDB="${SNAPSHOT_TO_WANDB:---snapshot-to-wandb}"
LOGDIR="$HERE/../../_logs/HPOT"; mkdir -p "$LOGDIR"
CFG="$HERE/../../configs/HPOT/GatedDeltaNet_penalty_oraclepot_none.yaml"
echo "[HPOT] GatedDeltaNet/penalty_oraclepot/none seed=2"
exec "$PYTHON" -m train --config "$CFG" --seed 2 \
    --runs-dir "$RUNS_DIR" --device "$DEVICE" \
    --wandb --snapshot-steps "$SNAP" $SNAP_WANDB $EXTRA \
    2>&1 | tee "$LOGDIR/GatedDeltaNet_penalty_oraclepot_none_seed2.log"
