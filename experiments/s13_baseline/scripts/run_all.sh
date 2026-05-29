#!/usr/bin/env bash
# AUTO-GENERATED. Run every per-cell MEMORY-BASELINE script (intrinsic=none):
# cells serial, seeds parallel within. Does NOT run the exploration arm — use
# run_exploration.sh for that. All env knobs (PYTHON/DEVICE/RUNS_DIR/SEEDS/
# PARALLEL) propagate to each script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/run_*.sh; do
  bn="$(basename "$s")"
  # Skip the aggregators themselves (run_*.sh also matches run_exploration.sh).
  [[ "$bn" == "run_all.sh" || "$bn" == "run_exploration.sh" ]] && continue
  echo "=== $bn ==="
  bash "$s"
done
echo "All cells done."
