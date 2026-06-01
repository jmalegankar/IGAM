#!/usr/bin/env bash
# Launch one Kubernetes Job (one GPU) per RedBlueDoors-8x8 leaf script under
# experiments/redbluedoors_8x8/scripts/, skipping the aggregator run_all.sh.
#
# Dedicated launcher (separate from k8s/launch-jobs.sh) — uses its own template
# k8s/rbd-job-template.yaml and app label memrl-rbd, so the S13 and RedBlueDoors
# sweeps never collide.
#
# Result: 15 Jobs = 11 cells (intrinsic=none) + Memoryless × {none, e3b_idm, rnd,
# noveld}.
#
# Usage:
#   k8s/launch-rbd-jobs.sh                       # apply ALL 15 Jobs
#   DRY_RUN=1 k8s/launch-rbd-jobs.sh             # print manifests, apply nothing
#   ONLY='<ERE>' k8s/launch-rbd-jobs.sh          # only basenames matching ERE
#
# Examples:
#   ONLY='^explore_'  k8s/launch-rbd-jobs.sh     # just the Memoryless bonus arm
#   ONLY='^run_'      k8s/launch-rbd-jobs.sh     # just the cell memory ranking
#
# Watch / clean up:
#   kubectl get jobs -l app=memrl-rbd
#   kubectl logs -l app=memrl-rbd --tail=20 --prefix
#   kubectl delete jobs -l app=memrl-rbd
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
TEMPLATE="$HERE/rbd-job-template.yaml"
SCRIPTS_REL="experiments/redbluedoors_8x8/scripts"
SCRIPTS_DIR="$REPO_ROOT/$SCRIPTS_REL"
JOB_PREFIX="memrl-rbd"

[[ -f "$TEMPLATE" ]] || { echo "missing template: $TEMPLATE" >&2; exit 1; }

shopt -s nullglob
count=0
for f in "$SCRIPTS_DIR"/run_*.sh "$SCRIPTS_DIR"/explore_*.sh; do
  bn="$(basename "$f")"
  # Skip the aggregator — it invokes leaf scripts, not a single training run.
  case "$bn" in
    run_all.sh) continue ;;
  esac

  # Optional subset filter: ONLY is an extended regex matched against basename.
  if [[ -n "${ONLY:-}" && ! "$bn" =~ ${ONLY} ]]; then
    continue
  fi

  base="${bn%.sh}"            # run_GRU | explore_Memoryless_rnd
  stem="${base#run_}"         # GRU     | explore_Memoryless_rnd
  # DNS-1123 job name: lowercase, '_' -> '-'.
  suffix="$(printf '%s' "$stem" | tr 'A-Z' 'a-z' | tr '_' '-')"
  jobname="${JOB_PREFIX}-${suffix}"
  script_path="$SCRIPTS_REL/$bn"

  manifest="$(sed -e "s|__JOBNAME__|$jobname|g" -e "s|__SCRIPT__|$script_path|g" "$TEMPLATE")"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '# %s -> %s\n%s\n---\n' "$bn" "$jobname" "$manifest"
  else
    printf '%s\n' "$manifest" | kubectl apply -f -
  fi
  count=$((count + 1))
done

echo "Processed $count Jobs (DRY_RUN=${DRY_RUN:-0})."
