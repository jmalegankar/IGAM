#!/usr/bin/env bash
# RERUN of failed HP combo lam0p01_ck32_lr0p0001 — cells [GRU RetNet] in parallel on ONE GPU
# (2-cells-per-GPU packing for OOM headroom). Knobs: PYTHON DEVICE RUNS_DIR SEED EXTRA.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"; cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"; RUNS_DIR="${RUNS_DIR:-runs/mysterypath_hpsweep}"
SEED="${SEED:-0}"; EXTRA="${EXTRA:-}"
LOGDIR="$HERE/../_logs"; mkdir -p "$LOGDIR"
CELLS="GRU RetNet"
echo "[rerun lam0p01_ck32_lr0p0001 | GRU-RetNet] cells=[$CELLS] on one GPU"
pids=()
for cell in $CELLS; do
  CFG="$REPO_ROOT/experiments/mysterypath_hpsweep/configs/mpg_hp_${cell}_lam0p01_ck32_lr0p0001.yaml"
  "$PYTHON" -m train --config "$CFG" --seed "$SEED" \
      --runs-dir "$RUNS_DIR" --device "$DEVICE" $EXTRA \
      > "$LOGDIR/${cell}_lam0p01_ck32_lr0p0001_seed${SEED}_rerun.log" 2>&1 &
  pids+=("$!"); echo "  launched $cell (pid $!)"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ "$fail" == "0" ]] || { echo "[rerun lam0p01_ck32_lr0p0001 | GRU-RetNet] a cell FAILED"; exit 1; }
echo "[rerun lam0p01_ck32_lr0p0001 | GRU-RetNet] done."
