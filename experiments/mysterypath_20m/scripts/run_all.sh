#!/usr/bin/env bash
# AUTO-GENERATED. Run every 20M (seed,pair) job serially (2 cells parallel within).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/mpg20m_*.sh; do
  bn="$(basename "$s")"; [[ "$bn" == "run_all.sh" ]] && continue
  echo "=== $bn ==="; bash "$s"
done
echo "All 20M jobs done."
