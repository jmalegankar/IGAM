#!/usr/bin/env bash
# AUTO-GENERATED. Run every HP combo (serial; the 3 cells run parallel within each).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/hp_*.sh; do echo "=== $(basename "$s") ==="; bash "$s"; done
echo "All HP-sweep combos done."
