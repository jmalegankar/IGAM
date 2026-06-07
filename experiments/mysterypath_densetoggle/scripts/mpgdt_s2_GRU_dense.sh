#!/usr/bin/env bash
# AUTO-GENERATED. B2 dense-toggle: GRU / dense at seed 2 —
# runs {none, e3b_idm} in PARALLEL on one GPU (the bonus-vs-none comparison).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"; RUNS_DIR="${RUNS_DIR:-runs/mysterypath_densetoggle}"; EXTRA="${EXTRA:-}"
LOGDIR="$HERE/../_logs"; mkdir -p "$LOGDIR"
CELL="GRU"; DENSITY="dense"; SEED="2"
echo "[s$SEED | $CELL $DENSITY] none + e3b_idm in parallel on one GPU"
pids=()
for intr in none e3b_idm; do
  CFG="$HERE/../configs/mpgdt_${CELL}_${DENSITY}_${intr}.yaml"
  "$PYTHON" -m train --config "$CFG" --seed "$SEED" \
      --runs-dir "$RUNS_DIR" --device "$DEVICE" $EXTRA \
      > "$LOGDIR/${CELL}_${DENSITY}_${intr}_seed${SEED}.log" 2>&1 &
  pids+=("$!"); echo "  launched ${CELL}/${DENSITY}/${intr} seed=$SEED (pid $!)"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ "$fail" == "0" ]] || { echo "[s$SEED | $CELL $DENSITY] a run FAILED — see $LOGDIR"; exit 1; }
echo "[s$SEED | $CELL $DENSITY] done."
