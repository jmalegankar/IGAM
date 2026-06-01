#!/usr/bin/env bash
# AUTO-GENERATED. Run every MysteryPath-Grid leaf script (conditions serial,
# seeds parallel within). All env knobs propagate.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/run_*.sh "$HERE"/explore_*.sh; do
  bn="$(basename "$s")"
  [[ "$bn" == "run_all.sh" ]] && continue
  echo "=== $bn ==="
  bash "$s"
done
echo "All MysteryPath-Grid conditions done."
