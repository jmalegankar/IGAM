#!/usr/bin/env bash
# Launch the MysteryPath HP-sweep RERUNS — one Kubernetes Job (one GPU) per
# rerun script under experiments/mysterypath_hpsweep/rerun/. These are the
# combos that crashed under the 3-cells-per-GPU packing, re-split to <=2 cells
# per GPU for OOM headroom (GRU+RetNet share a GPU; GatedDeltaNet runs solo).
#
# Reuses k8s/mpghp-job-template.yaml (same image / wandb secret / RUNS_DIR /
# SEED=0 / 1 GPU). 6 scripts -> 6 Jobs.
#
# Usage:
#   k8s/launch-mpghp-rerun.sh                 # apply all 6 rerun Jobs
#   DRY_RUN=1 k8s/launch-mpghp-rerun.sh       # print manifests, apply nothing
#   ONLY='<ERE>' k8s/launch-mpghp-rerun.sh    # only basenames matching ERE
#
# NB: requires the image rebuilt with these rerun scripts baked in (git clone).
#
# Watch / clean up:
#   kubectl get jobs -l app=memrl-mpghp
#   kubectl delete jobs -l app=memrl-mpghp
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
TEMPLATE="$HERE/mpghp-job-template.yaml"
SCRIPTS_REL="experiments/mysterypath_hpsweep/rerun"
SCRIPTS_DIR="$REPO_ROOT/$SCRIPTS_REL"
JOB_PREFIX="memrl-mpghp-rerun"

[[ -f "$TEMPLATE" ]] || { echo "missing template: $TEMPLATE" >&2; exit 1; }

shopt -s nullglob
count=0
for f in "$SCRIPTS_DIR"/rerun_*.sh; do
  bn="$(basename "$f")"
  if [[ -n "${ONLY:-}" && ! "$bn" =~ ${ONLY} ]]; then
    continue
  fi
  stem="${bn%.sh}"               # rerun_lam0p01_ck32_lr0p0001_GRU-RetNet
  suffix="$(printf '%s' "${stem#rerun_}" | tr 'A-Z' 'a-z' | tr '_' '-')"
  jobname="${JOB_PREFIX}-${suffix}"
  # DNS-1123 names cap at 63 chars; truncate defensively (still unique by combo).
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

echo "Processed $count rerun Jobs (DRY_RUN=${DRY_RUN:-0})."
