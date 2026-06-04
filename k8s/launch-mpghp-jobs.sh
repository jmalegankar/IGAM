#!/usr/bin/env bash
# Launch one Kubernetes Job (one GPU) per MysteryPath HP-sweep condition under
# experiments/mysterypath_hpsweep/scripts/ (the hp_*.sh leaves), skipping run_all.sh.
#
# Dedicated launcher (template k8s/mpghp-job-template.yaml, app=memrl-mpghp).
# 36 Jobs = 3 cells × {λ:3} × {chunk_len:2} × {lr:2}, intrinsic=e3b_idm, 5M, 1 seed.
#
# Usage:
#   k8s/launch-mpghp-jobs.sh                  # apply all 36
#   DRY_RUN=1 k8s/launch-mpghp-jobs.sh        # preview
#   ONLY='GRU'   k8s/launch-mpghp-jobs.sh     # just one cell's 12 combos
#   ONLY='ck64'  k8s/launch-mpghp-jobs.sh     # just the chunk_len=64 half
#
# Watch / clean: kubectl get jobs -l app=memrl-mpghp ; kubectl delete jobs -l app=memrl-mpghp
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
TEMPLATE="$HERE/mpghp-job-template.yaml"
SCRIPTS_REL="experiments/mysterypath_hpsweep/scripts"
SCRIPTS_DIR="$REPO_ROOT/$SCRIPTS_REL"
JOB_PREFIX="memrl-mpghp"

[[ -f "$TEMPLATE" ]] || { echo "missing template: $TEMPLATE" >&2; exit 1; }

shopt -s nullglob
count=0
for f in "$SCRIPTS_DIR"/hp_*.sh; do
  bn="$(basename "$f")"
  [[ "$bn" == "run_all.sh" ]] && continue
  if [[ -n "${ONLY:-}" && ! "$bn" =~ ${ONLY} ]]; then continue; fi

  base="${bn%.sh}"            # hp_GRU_lam0p003_ck32_lr0p0001
  stem="${base#hp_}"          # GRU_lam0p003_ck32_lr0p0001
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
