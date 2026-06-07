#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/mpgdt_*.sh; do
  bn="$(basename "$s")"; [[ "$bn" == "run_all.sh" ]] && continue
  echo "=== $bn ==="; bash "$s"
done
echo "All dense-toggle jobs done."
