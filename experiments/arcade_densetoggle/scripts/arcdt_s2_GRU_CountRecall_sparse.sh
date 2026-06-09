#!/usr/bin/env bash
# AUTO-GENERATED. Arcade density toggle: CountRecall / GRU / sparse at
# seed 2 — runs {none, e3b_idm} in PARALLEL on one GPU (84x84 parity
# with the validated MysteryPath 2-per-GPU packing).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"; RUNS_DIR="${RUNS_DIR:-runs/arcade_densetoggle}"; EXTRA="${EXTRA:-}"
LOGDIR="$HERE/../_logs"; mkdir -p "$LOGDIR"
TASK="CountRecall"; CELL="GRU"; DENSITY="sparse"; SEED="2"
echo "[s$SEED | $TASK $CELL $DENSITY] none + e3b_idm in parallel on one GPU"
pids=()
for intr in none e3b_idm; do
  CFG="$HERE/../configs/arcdt_${TASK}_${CELL}_${DENSITY}_${intr}.yaml"
  "$PYTHON" -m train --config "$CFG" --seed "$SEED" \
      --runs-dir "$RUNS_DIR" --device "$DEVICE" $EXTRA \
      > "$LOGDIR/${TASK}_${CELL}_${DENSITY}_${intr}_seed${SEED}.log" 2>&1 &
  pids+=("$!"); echo "  launched $TASK/$CELL/$DENSITY/$intr seed=$SEED (pid $!)"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ "$fail" == "0" ]] || { echo "[s$SEED | $TASK $CELL $DENSITY] a run FAILED — see $LOGDIR"; exit 1; }
echo "[s$SEED | $TASK $CELL $DENSITY] done."
