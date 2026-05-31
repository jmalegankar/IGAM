#!/usr/bin/env bash
# Launch one Kubernetes Job (CPU-only) per Active T-Maze leaf script under
# experiments/tmaze_active/scripts/, skipping the aggregator run_all.sh.
#
# SEPARATE from k8s/launch-jobs.sh (the S13 launcher) on purpose — uses its own
# template k8s/maze-job-template.yaml and its own app label (memrl-tmaze), so the
# two experiments never collide.
#
# Usage:
#   k8s/launch-maze-jobs.sh                          # apply ALL tmaze leaf Jobs
#   DRY_RUN=1 k8s/launch-maze-jobs.sh                # print manifests, apply nothing
#   ONLY='<ERE>' k8s/launch-maze-jobs.sh             # only basenames matching ERE
#
# First arm — the `none` baseline at corridor length L=50 (11 cells × 3 seeds):
#   ONLY='^run_.*_L50\.sh$' k8s/launch-maze-jobs.sh
#
# Watch / clean up:
#   kubectl get jobs -l app=memrl-tmaze
#   kubectl logs -l app=memrl-tmaze --tail=20 --prefix
#   kubectl delete jobs -l app=memrl-tmaze
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
TEMPLATE="$HERE/maze-job-template.yaml"
SCRIPTS_REL="experiments/tmaze_active/scripts"
SCRIPTS_DIR="$REPO_ROOT/$SCRIPTS_REL"
JOB_PREFIX="memrl-tmaze"

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

  base="${bn%.sh}"            # run_GRU_L50 | explore_GRU_noveld_L50
  stem="${base#run_}"         # GRU_L50     | explore_GRU_noveld_L50
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
