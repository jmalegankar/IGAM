#!/usr/bin/env bash
# AUTO-GENERATED. Run every exploration-arm script (explore_*.sh): conditions
# serial, seeds parallel within. All env knobs propagate to each script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/explore_*.sh; do
  echo "=== $(basename "$s") ==="
  bash "$s"
done
echo "All exploration conditions done."
