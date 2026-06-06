#!/usr/bin/env bash
# Launch the MysteryPath-Grid 20M × 5-seed headline sweep — one Kubernetes Job
# (one GPU) per (seed, cell-pair) script under experiments/mysterypath_20m/
# scripts/, each running 2 cells in parallel. 30 Jobs = 12 cells × 5 seeds / 2.
#
# Usage:
#   k8s/launch-mpg20m-jobs.sh                # apply all 30 Jobs
#   DRY_RUN=1 k8s/launch-mpg20m-jobs.sh      # print manifests, apply nothing
#   ONLY='_s0_' k8s/launch-mpg20m-jobs.sh    # just seed 0 (6 jobs) — stage by seed
#
# NB: requires the image rebuilt with these scripts baked in.
#
# Watch / clean up:
#   kubectl get jobs -l app=memrl-mpg20m
#   kubectl delete jobs -l app=memrl-mpg20m
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
TEMPLATE="$HERE/mpg20m-job-template.yaml"
SCRIPTS_REL="experiments/mysterypath_20m/scripts"
SCRIPTS_DIR="$REPO_ROOT/$SCRIPTS_REL"
JOB_PREFIX="memrl-mpg20m"

[[ -f "$TEMPLATE" ]] || { echo "missing template: $TEMPLATE" >&2; exit 1; }

shopt -s nullglob
count=0
for f in "$SCRIPTS_DIR"/mpg20m_*.sh; do
  bn="$(basename "$f")"
  case "$bn" in run_all.sh) continue ;; esac
  if [[ -n "${ONLY:-}" && ! "$bn" =~ ${ONLY} ]]; then
    continue
  fi
  stem="${bn%.sh}"                       # mpg20m_s0_GRU-LSTM
  suffix="$(printf '%s' "${stem#mpg20m_}" | tr 'A-Z' 'a-z' | tr '_' '-')"
  jobname="${JOB_PREFIX}-${suffix}"
  jobname="${jobname:0:63}"; jobname="${jobname%-}"
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
