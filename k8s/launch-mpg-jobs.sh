#!/usr/bin/env bash
# Launch one Kubernetes Job (one GPU) per MysteryPath-Grid leaf script under
# experiments/mysterypath_grid/scripts/, skipping the aggregator run_all.sh.
#
# Dedicated launcher (separate from launch-jobs.sh / launch-rbd-jobs.sh) — uses
# its own template k8s/mpg-job-template.yaml and app label memrl-mpg.
#
# Result: 8 Jobs = {GRU, Memoryless} × {none, e3b_idm, rnd, noveld}.
#
# Usage:
#   k8s/launch-mpg-jobs.sh                       # apply ALL 8 Jobs
#   DRY_RUN=1 k8s/launch-mpg-jobs.sh             # print manifests, apply nothing
#   ONLY='<ERE>' k8s/launch-mpg-jobs.sh          # only basenames matching ERE
#
# Examples:
#   ONLY='_e3b_idm' k8s/launch-mpg-jobs.sh       # just the e3b_idm arm
#   ONLY='^run_'    k8s/launch-mpg-jobs.sh       # just the none baselines
#
# NOTE: requires the image rebuilt with memory-gym baked in (see Dockerfile).
#
# Watch / clean up:
#   kubectl get jobs -l app=memrl-mpg
#   kubectl logs -l app=memrl-mpg --tail=20 --prefix
#   kubectl delete jobs -l app=memrl-mpg
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
TEMPLATE="$HERE/mpg-job-template.yaml"
SCRIPTS_REL="experiments/mysterypath_grid/scripts"
SCRIPTS_DIR="$REPO_ROOT/$SCRIPTS_REL"
JOB_PREFIX="memrl-mpg"

[[ -f "$TEMPLATE" ]] || { echo "missing template: $TEMPLATE" >&2; exit 1; }

shopt -s nullglob
count=0
for f in "$SCRIPTS_DIR"/run_*.sh "$SCRIPTS_DIR"/explore_*.sh; do
  bn="$(basename "$f")"
  case "$bn" in
    run_all.sh) continue ;;
  esac

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
