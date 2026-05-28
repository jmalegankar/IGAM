#!/usr/bin/env bash
# AUTO-GENERATED. Run every per-cell script: cells serial, seeds parallel within.
# All env knobs (PYTHON/DEVICE/RUNS_DIR/SEEDS/PARALLEL) propagate to each script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/run_*.sh; do
  [[ "$(basename "$s")" == "run_all.sh" ]] && continue
  echo "=== $(basename "$s") ==="
  bash "$s"
done
echo "All cells done."
