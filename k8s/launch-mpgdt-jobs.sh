#!/usr/bin/env bash
# Launch the B2 dense/sparse-toggle experiment — one Kubernetes Job (one GPU)
# per (seed, cell, density) script under experiments/mysterypath_densetoggle/
# scripts/, each running {none, e3b_idm} in parallel.
# 30 Jobs = 3 cells × 2 densities × 5 seeds.
#
# Usage:
#   k8s/launch-mpgdt-jobs.sh                  # apply all 30 Jobs
#   DRY_RUN=1 k8s/launch-mpgdt-jobs.sh        # print manifests, apply nothing
#   ONLY='_s0_' k8s/launch-mpgdt-jobs.sh      # just seed 0 (6 jobs) — stage by seed
#   ONLY='_dense' k8s/launch-mpgdt-jobs.sh    # just the dense arm (15 jobs)
#
# NB: requires the image rebuilt with these scripts baked in (push GIT_REF first).
#
# Watch / clean up:
#   kubectl get jobs -l app=memrl-mpgdt
#   kubectl delete jobs -l app=memrl-mpgdt
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
TEMPLATE="$HERE/mpgdt-job-template.yaml"
SCRIPTS_REL="experiments/mysterypath_densetoggle/scripts"
SCRIPTS_DIR="$REPO_ROOT/$SCRIPTS_REL"
JOB_PREFIX="memrl-mpgdt"

[[ -f "$TEMPLATE" ]] || { echo "missing template: $TEMPLATE" >&2; exit 1; }

shopt -s nullglob
count=0
for f in "$SCRIPTS_DIR"/mpgdt_*.sh; do
  bn="$(basename "$f")"
  case "$bn" in run_all.sh) continue ;; esac
  if [[ -n "${ONLY:-}" && ! "$bn" =~ ${ONLY} ]]; then
    continue
  fi
  stem="${bn%.sh}"                       # mpgdt_s0_GRU_dense
  suffix="$(printf '%s' "${stem#mpgdt_}" | tr 'A-Z' 'a-z' | tr '_' '-')"
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
