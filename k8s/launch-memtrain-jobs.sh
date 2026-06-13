#!/usr/bin/env bash
# Launch the memory-training redo — one Kubernetes Job (one GPU) per RUN script
# under experiments/memory_training/scripts/<EID>[/<env>]/. The registry emits one
# script per run (one intrinsic), so each Job is a single `python -m train`.
#
# Layout (after `registry.py --emit <EID>`):
#   scripts/E1/<env>/E1_s<seed>_<cell>_<density>_<bonus>.sh    (core grid, per env-slice)
#   scripts/E3/E3_s<seed>_GRU_<lLABEL>_<bonus>.sh              (method studies)
#
# Select what to launch:
#   EID    experiment id (default E1)                  e.g. EID=E3
#   ENV    core env-slice filter (E1 only)             e.g. ENV=MysteryPath
#   ONLY   regex on the script basename (stage waves)  e.g. ONLY='s0_'  (seed-0 wave)
#   DRY_RUN=1  print manifests, apply nothing
#
# Examples:
#   DRY_RUN=1 k8s/launch-memtrain-jobs.sh                       # preview ALL of E1 (910)
#   ENV=MysteryPath ONLY='s0_' k8s/launch-memtrain-jobs.sh     # E1 MysteryPath seed-0 wave
#   ENV=MysteryPath k8s/launch-memtrain-jobs.sh                 # E1 MysteryPath (all seeds)
#   EID=E3 k8s/launch-memtrain-jobs.sh                          # the λ-sweep
#
# Stage safely: DRY_RUN=1 first → ONLY='s0_' (one seed wave) → inspect wandb → rest.
#
# Watch / clean up:
#   kubectl get jobs -l app=memrl-memtrain
#   kubectl delete jobs -l app=memrl-memtrain
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
TEMPLATE="$HERE/memtrain-job-template.yaml"
SCRIPTS_REL="experiments/memory_training/scripts"
JOB_PREFIX="memrl"
EID="${EID:-E1}"

[[ -f "$TEMPLATE" ]] || { echo "missing template: $TEMPLATE" >&2; exit 1; }
ROOT="$REPO_ROOT/$SCRIPTS_REL/$EID"
[[ -d "$ROOT" ]] || { echo "no emitted scripts for $EID — run: python -m experiments.memory_training.registry --emit $EID" >&2; exit 1; }

count=0
# recurse: scripts/<EID>/**/*.sh  (env subdirs for E1; flat for methods)
while IFS= read -r -d '' f; do
  rel="${f#"$REPO_ROOT/"}"                          # repo-relative path → container path
  bn="$(basename "$f")"
  # ENV filter applies to the env subdir (E1 only)
  if [[ -n "${ENV:-}" && "$rel" != *"/$EID/$ENV/"* ]]; then continue; fi
  if [[ -n "${ONLY:-}" && ! "$bn" =~ ${ONLY} ]]; then continue; fi
  # jobname from the path below scripts/ : E1/MysteryPath/E1_s0_GRU_sparse_none.sh
  stem="${rel#"$SCRIPTS_REL/"}"; stem="${stem%.sh}"
  jobname="${JOB_PREFIX}-$(printf '%s' "$stem" | tr 'A-Z/_' 'a-z--')"
  jobname="$(printf '%s' "$jobname" | tr -s '-')"
  jobname="${jobname:0:63}"; jobname="${jobname%-}"
  manifest="$(sed -e "s|__JOBNAME__|$jobname|g" -e "s|__SCRIPT__|$rel|g" "$TEMPLATE")"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '# %s -> %s\n%s\n---\n' "$bn" "$jobname" "$manifest"
  else
    printf '%s\n' "$manifest" | kubectl apply -f -
  fi
  count=$((count + 1))
done < <(find "$ROOT" -type f -name '*.sh' -print0 | sort -z)

echo "Processed $count Jobs (EID=$EID ENV=${ENV:-*} ONLY=${ONLY:-*} DRY_RUN=${DRY_RUN:-0})."
