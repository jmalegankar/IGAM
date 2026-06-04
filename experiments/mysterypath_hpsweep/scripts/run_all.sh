#!/usr/bin/env bash
# AUTO-GENERATED. Run every HP-sweep condition (serial; seeds parallel within).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/hp_*.sh; do echo "=== $(basename "$s") ==="; bash "$s"; done
echo "All HP-sweep conditions done."
