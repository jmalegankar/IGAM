#!/usr/bin/env bash
# Launch the memory-training redo — Kubernetes Jobs (one GPU each) over the RUN
# scripts under experiments/memory_training/scripts/<EID>[/<env>]/. The registry
# emits one script per run (one intrinsic); a Job runs PACK of them in parallel
# on its single GPU (PACK=1 default → one run/Job).
#
# Layout (after `registry.py --emit <EID>`):
#   scripts/E1/<env>/s<seed>_<cell>_<density>_<bonus>.sh    (core grid, per env-slice)
#   scripts/E3/s<seed>_GRU_<lLABEL>_<bonus>.sh              (method studies)
#
# Knobs:
#   EID    experiment id (default E1)                  e.g. EID=E3
#   ENV    core env-slice filter (E1 only)             e.g. ENV=MysteryPath
#   ONLY   regex on the script basename (stage waves)  e.g. ONLY='s0_'  (seed-0 wave)
#   PACK   runs per Job/GPU (default 1; try 2-3 after a GPU-mem check)
#   DRY_RUN=1  print manifests, apply nothing
#
# Examples:
#   DRY_RUN=1 ENV=MysteryPath ONLY='s0_' k8s/launch-memtrain-jobs.sh   # preview seed-0 wave
#   ENV=MysteryPath ONLY='s0_' k8s/launch-memtrain-jobs.sh             # apply it
#   PACK=2 ENV=Autoencode k8s/launch-memtrain-jobs.sh                  # 2 runs/GPU (vector → cheap)
#   EID=E3 k8s/launch-memtrain-jobs.sh                                 # the λ-sweep
#
# Stage safely: DRY_RUN=1 → seed-0 wave → inspect wandb (mem/util, e3b_idm_acc,
# success, freeze falls→0) → raise PACK and/or launch the rest.
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
PACK="${PACK:-1}"
CPU_PER=5; MEM_PER=20                               # per-run requests; scale by PACK

[[ -f "$TEMPLATE" ]] || { echo "missing template: $TEMPLATE" >&2; exit 1; }
ROOT="$REPO_ROOT/$SCRIPTS_REL/$EID"
[[ -d "$ROOT" ]] || { echo "no emitted scripts for $EID — run: python -m experiments.memory_training.registry --emit $EID" >&2; exit 1; }

# Collect matching scripts (repo-relative paths), sorted & filtered.
# (Portable to bash 3.2 — no mapfile; script paths have no newlines/spaces.)
FILES=()
while IFS= read -r rel; do
  [[ -n "$rel" ]] && FILES+=("$rel")
done < <(
  find "$ROOT" -type f -name '*.sh' -print0 | sort -z | while IFS= read -r -d '' f; do
    rel="${f#"$REPO_ROOT/"}"; bn="$(basename "$f")"
    [[ -n "${ENV:-}"  && "$rel" != *"/$EID/$ENV/"* ]] && continue
    [[ -n "${ONLY:-}" && ! "$bn" =~ ${ONLY} ]] && continue
    printf '%s\n' "$rel"
  done
)
[[ ${#FILES[@]} -gt 0 ]] || { echo "no scripts matched (EID=$EID ENV=${ENV:-*} ONLY=${ONLY:-*})"; exit 0; }

cpu=$(( CPU_PER * PACK )); mem=$(( MEM_PER * PACK ))
TEMPLATE_BODY="$(cat "$TEMPLATE")"
jobs=0
for ((i = 0; i < ${#FILES[@]}; i += PACK)); do
  chunk=("${FILES[@]:i:PACK}")
  # jobname from the first script's path below scripts/
  stem="${chunk[0]#"$SCRIPTS_REL/"}"; stem="${stem%.sh}"
  jobname="${JOB_PREFIX}-$(printf '%s' "$stem" | tr 'A-Z/_' 'a-z--')"
  [[ "$PACK" -gt 1 ]] && jobname="${jobname}-x${PACK}"
  jobname="$(printf '%s' "$jobname" | tr -s '-')"; jobname="${jobname:0:63}"; jobname="${jobname%-}"
  # command: PACK=1 → ["bash","<script>"]; PACK>1 → ["bash","-c","bash a & bash b & wait"]
  if [[ "$PACK" -eq 1 ]]; then
    cmd="[\"bash\", \"${chunk[0]}\"]"
  else
    inner=""; for s in "${chunk[@]}"; do inner+="bash $s & "; done; inner+="wait"
    cmd="[\"bash\", \"-c\", \"$inner\"]"
  fi
  # bash string replacement (NOT sed): cmd contains '&' which sed treats as the
  # matched text — bash ${//} replaces literally, so backgrounding survives.
  manifest="${TEMPLATE_BODY//__JOBNAME__/$jobname}"
  manifest="${manifest//__COMMAND__/$cmd}"
  manifest="${manifest//__CPU__/$cpu}"
  manifest="${manifest//__MEM__/$mem}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '# %s (%d run/s) -> %s\n%s\n---\n' "$(basename "${chunk[0]}")" "${#chunk[@]}" "$jobname" "$manifest"
  else
    printf '%s\n' "$manifest" | kubectl apply -f -
  fi
  jobs=$((jobs + 1))
done

echo "Processed $jobs Jobs over ${#FILES[@]} runs (EID=$EID ENV=${ENV:-*} ONLY=${ONLY:-*} PACK=$PACK CPU=$cpu MEM=${mem}Gi DRY_RUN=${DRY_RUN:-0})."
