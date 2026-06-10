#!/usr/bin/env bash
# AUTO-GENERATED. Arcade density toggle: BattleShip / GatedDeltaNet / dense /
# e3b_idm at seeds {0 1 2} — one condition, seeds in PARALLEL on one GPU
# (3/GPU at ck64 + 84x84; homogeneous job, no tail-idle).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"; RUNS_DIR="${RUNS_DIR:-runs/arcade_densetoggle}"; EXTRA="${EXTRA:-}"
LOGDIR="$HERE/../_logs"; mkdir -p "$LOGDIR"
TASK="BattleShip"; CELL="GatedDeltaNet"; DENSITY="dense"; INTR="e3b_idm"
CFG="$HERE/../configs/arcdt_${TASK}_${CELL}_${DENSITY}_${INTR}.yaml"
echo "[$TASK $CELL $DENSITY $INTR] seeds 0 1 2 in parallel on one GPU"
pids=()
for seed in 0 1 2; do
  "$PYTHON" -m train --config "$CFG" --seed "$seed" \
      --runs-dir "$RUNS_DIR" --device "$DEVICE" $EXTRA \
      > "$LOGDIR/${TASK}_${CELL}_${DENSITY}_${INTR}_seed${seed}.log" 2>&1 &
  pids+=("$!"); echo "  launched seed=$seed (pid $!)"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ "$fail" == "0" ]] || { echo "[$TASK $CELL $DENSITY $INTR] a seed FAILED — see $LOGDIR"; exit 1; }
echo "[$TASK $CELL $DENSITY $INTR] seeds 0 1 2 done."
