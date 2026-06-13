#!/usr/bin/env bash
# Launch the B7 tbptt-density experiment — one Kubernetes Job
# (one GPU) per (seed, cell, density) script under
# experiments/tbptt_density/scripts/, each running {none, e3b_idm}
# in parallel. 30 Jobs = 3 cells × 2 densities × 5 seeds (× 2 runs = 60).
#
# Usage:
#   k8s/launch-tbk-jobs.sh                   # apply all 30 Jobs
#   DRY_RUN=1 k8s/launch-tbk-jobs.sh         # print manifests, apply nothing
#   ONLY='_s0_' k8s/launch-tbk-jobs.sh       # just seed 0 (6 jobs) — stage first
#   ONLY='_sparse' k8s/launch-tbk-jobs.sh    # just the sparse arm (15 jobs)
#
# NB: requires the image rebuilt with these scripts baked in (push GIT_REF first).
#
# Watch / clean up:
#   kubectl get jobs -l app=memrl-tbk
#   kubectl delete jobs -l app=memrl-tbk
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
TEMPLATE="$HERE/tbk-job-template.yaml"
SCRIPTS_REL="experiments/tbptt_density/scripts"
SCRIPTS_DIR="$REPO_ROOT/$SCRIPTS_REL"
JOB_PREFIX="memrl-tbk"

[[ -f "$TEMPLATE" ]] || { echo "missing template: $TEMPLATE" >&2; exit 1; }

shopt -s nullglob
count=0
for f in "$SCRIPTS_DIR"/tbk_*.sh; do
  bn="$(basename "$f")"
  case "$bn" in run_all.sh) continue ;; esac
  if [[ -n "${ONLY:-}" && ! "$bn" =~ ${ONLY} ]]; then
    continue
  fi
  stem="${bn%.sh}"                       # tbk_s0_GRU_sparse
  suffix="$(printf '%s' "${stem#tbk_}" | tr 'A-Z' 'a-z' | tr '_' '-')"
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
