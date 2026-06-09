#!/usr/bin/env bash
# Launch the POPGym-Arcade density-toggle experiment — one Kubernetes Job
# (one GPU) per (seed, cell, task, density) script under
# experiments/arcade_densetoggle/scripts/, each running {none, e3b_idm} in
# parallel. 40 Jobs = 2 tasks × 2 cells × 2 densities × 5 seeds.
#
# Usage:
#   k8s/launch-arcdt-jobs.sh                      # apply all 40 Jobs
#   DRY_RUN=1 k8s/launch-arcdt-jobs.sh            # print manifests, apply nothing
#   ONLY='_s0_' k8s/launch-arcdt-jobs.sh          # just seed 0 (8 jobs) — stage by seed
#   ONLY='BattleShip' k8s/launch-arcdt-jobs.sh    # just BattleShip (20 jobs)
#   ONLY='_sparse' k8s/launch-arcdt-jobs.sh       # just the deferred arm (20 jobs)
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
  stem="${bn%.sh}"                       # arcdt_s0_GRU_BattleShip_dense
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
