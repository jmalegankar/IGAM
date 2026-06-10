#!/usr/bin/env bash
# Launch the POPGym-Arcade density-toggle experiment — one Kubernetes Job
# (one GPU) per (task, cell, density, intrinsic, seed-group) script under
# experiments/arcade_densetoggle/scripts/, each running ONE condition at 3
# (or 2) seeds in parallel. 32 Jobs = 16 conditions × 2 seed-groups (80 runs).
#
# Usage:
#   k8s/launch-arcdt-jobs.sh                      # apply all 32 Jobs
#   DRY_RUN=1 k8s/launch-arcdt-jobs.sh            # print manifests, apply nothing
#   ONLY='_sg0' k8s/launch-arcdt-jobs.sh          # seed-group 0,1,2 (16 jobs) — stage first
#   ONLY='BattleShip' k8s/launch-arcdt-jobs.sh    # just BattleShip (16 jobs)
#   ONLY='_sparse' k8s/launch-arcdt-jobs.sh       # just the deferred arm (16 jobs)
#
# NB: requires the image rebuilt WITH the popgym-arcade extra (jax) and these
# scripts baked in (push GIT_REF first).
#
# Watch / clean up:
#   kubectl get jobs -l app=memrl-arcdt
#   kubectl delete jobs -l app=memrl-arcdt
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
TEMPLATE="$HERE/arcdt-job-template.yaml"
SCRIPTS_REL="experiments/arcade_densetoggle/scripts"
SCRIPTS_DIR="$REPO_ROOT/$SCRIPTS_REL"
JOB_PREFIX="memrl-arcdt"

[[ -f "$TEMPLATE" ]] || { echo "missing template: $TEMPLATE" >&2; exit 1; }

shopt -s nullglob
count=0
for f in "$SCRIPTS_DIR"/arcdt_*.sh; do
  bn="$(basename "$f")"
  case "$bn" in run_all.sh) continue ;; esac
  if [[ -n "${ONLY:-}" && ! "$bn" =~ ${ONLY} ]]; then
    continue
  fi
  stem="${bn%.sh}"                       # arcdt_BattleShip_GRU_dense_none_sg0
  suffix="$(printf '%s' "${stem#arcdt_}" | tr 'A-Z' 'a-z' | tr '_' '-')"
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
