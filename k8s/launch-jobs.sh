#!/usr/bin/env bash
# Launch one Kubernetes Job (one GPU) per training script under
# experiments/s13_baseline/scripts/, EXCLUDING the aggregators run_all.sh and
# run_exploration.sh (those just invoke the leaf scripts serially — running one
# as a single pod would pin the whole sweep onto one GPU, defeating the fan-out).
#
# Result: 17 Jobs = 11 memory-baseline cells + 6 exploration conditions.
#
# Image is pinned in k8s/job-template.yaml (r0hanpat1l/memory:latest); the
# scripts are baked into that image, so nothing local is uploaded — kubectl just
# creates the Jobs and the cluster pulls the image.
#
# Prereqs in the cluster (see k8s/job-template.yaml header):
#   * Secret `wandb` (key `api-key`)   — k8s/wandb-secret.example.yaml
#   * ReadWriteMany PVC `memrl-runs`
#
# Usage:
#   k8s/launch-jobs.sh               # apply ALL leaf Jobs
#   DRY_RUN=1 k8s/launch-jobs.sh     # print the manifests, apply nothing
#   ONLY='<ERE>' k8s/launch-jobs.sh  # apply only scripts whose basename matches
#                                    # the extended regex ONLY. e.g. launch just
#                                    # the phase-2 NovelD sweep on the cells that
#                                    # haven't run yet (GRU/Memoryless already did):
#   ONLY='explore_(LSTM|mLSTM|LRU|Mamba2|FFM|GatedDeltaNet|SHM|GTrXL|RetNet|LinearTransformer)_noveld' \
#       k8s/launch-jobs.sh
#
# Watch / clean up:
#   kubectl get jobs -l app=memrl-s13
#   kubectl logs -l app=memrl-s13 --tail=20 --prefix
#   kubectl delete jobs -l app=memrl-s13
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
TEMPLATE="$HERE/job-template.yaml"
SCRIPTS_REL="experiments/s13_baseline/scripts"
SCRIPTS_DIR="$REPO_ROOT/$SCRIPTS_REL"

[[ -f "$TEMPLATE" ]] || { echo "missing template: $TEMPLATE" >&2; exit 1; }

shopt -s nullglob
count=0
for f in "$SCRIPTS_DIR"/run_*.sh "$SCRIPTS_DIR"/explore_*.sh; do
  bn="$(basename "$f")"
  # Skip aggregators — they invoke the leaf scripts, not a single training run.
  case "$bn" in
    run_all.sh | run_exploration.sh) continue ;;
  esac

  # Optional subset filter: ONLY is an extended regex matched against the
  # basename. Lets you launch a slice (e.g. just the new NovelD cells) without
  # re-applying Jobs that already ran. Unset ONLY ⇒ launch everything.
  if [[ -n "${ONLY:-}" && ! "$bn" =~ ${ONLY} ]]; then
    continue
  fi

  base="${bn%.sh}"            # run_GRU            | explore_GRU_rnd
  stem="${base#run_}"         # GRU                | explore_GRU_rnd
  # DNS-1123 job name: lowercase, '_' -> '-'.
  suffix="$(printf '%s' "$stem" | tr 'A-Z' 'a-z' | tr '_' '-')"
  jobname="memrl-s13-${suffix}"
  script_path="$SCRIPTS_REL/$bn"

  # '|' delimiter so the script path's '/' doesn't clash with sed.
  manifest="$(sed -e "s|__JOBNAME__|$jobname|g" -e "s|__SCRIPT__|$script_path|g" "$TEMPLATE")"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '# %s -> %s\n%s\n---\n' "$bn" "$jobname" "$manifest"
  else
    printf '%s\n' "$manifest" | kubectl apply -f -
  fi
  count=$((count + 1))
done

echo "Processed $count Jobs (DRY_RUN=${DRY_RUN:-0})."
