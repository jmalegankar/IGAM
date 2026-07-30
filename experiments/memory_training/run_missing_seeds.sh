#!/usr/bin/env bash
# run_missing_seeds.sh — complete the reported matrix to n=10.
#
# Three cells finished 9 of 10 seeds. This launches exactly the three missing
# runs, in parallel, on one local GPU box (intended: DGX Spark, overnight).
#
#   MPG      / distractor / GatedDeltaNet / none    seed 9
#   MPG      / distractor / Mamba2        / none    seed 8
#   S13      / distractor / GatedDeltaNet / noveld  seed 2
#
# These are PRODUCTION runs: they log to wandb (projects memrl-mpg-distractor /
# memrl-s13-distractor, group "distractor"), because make_tables.py and
# appendix_stats.py pull from there. Do NOT add --no-wandb.
#
# On the box:
#   git pull && wandb login          # once
#   nohup bash experiments/memory_training/run_missing_seeds.sh > ~/missing.log 2>&1 &
#   tail -f ~/missing.log
#
# Knobs:
#   P=3           concurrent runs (default 3 — all of them)
#   DEVICE=cuda   torch device
#   SKIP_SMOKE=1  skip the GatedDeltaNet/Mamba2 CUDA-kernel preflight
#   NO_RESUME=1   don't query wandb to skip already-finished runs
#   NO_SNAP=1     skip --snapshot-to-wandb artifact uploads (saves upload time)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; cd "$REPO"
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$REPO/.venv/bin/python" ]]; then PY="$REPO/.venv/bin/python"; else PY="python"; fi
fi
P="${P:-3}"
export DEVICE="${DEVICE:-cuda}"
[[ "${NO_SNAP:-0}" == 1 ]] && export SNAPSHOT_TO_WANDB=""

# script path                                              wandb project           run name
JOBS=(
  "DISTRACT/s9_GatedDeltaNet_distractor_none|memrl-mpg-distractor|GatedDeltaNet-distractor-none-seed9"
  "DISTRACT/s8_Mamba2_distractor_none|memrl-mpg-distractor|Mamba2-distractor-none-seed8"
  "DISTRACTS13/s2_GatedDeltaNet_distractor_noveld|memrl-s13-distractor|GatedDeltaNet-distractor-noveld-seed2"
)

echo "=============================================================="
echo " completing the reported matrix to n=10 — 3 runs, 20M steps each"
echo " device=$DEVICE  P=$P  python=$PY"
echo "=============================================================="

# ── sanity: every script and config must exist before we start ───────────────
for j in "${JOBS[@]}"; do
  f="$HERE/scripts/${j%%|*}.sh"
  [[ -f "$f" ]] || { echo "FATAL: missing $f"; exit 1; }
done

# ── CUDA-kernel preflight: GDN + Mamba2 must actually start on this GPU ──────
# Both cells here need mamba-ssm / causal-conv1d. Failing fast beats discovering
# it eight hours in.
if [[ "${SKIP_SMOKE:-0}" != 1 ]]; then
  for j in "${JOBS[@]}"; do
    f="$HERE/scripts/${j%%|*}.sh"
    n="$(basename "${j%%|*}")"
    echo "[smoke] $n (50k steps, no-wandb) …"
    if EXTRA="--total-timesteps 50000 --no-wandb" DEVICE="$DEVICE" \
       bash "$f" >"/tmp/smoke_${n}.log" 2>&1; then
      echo "[smoke] $n ok"
    else
      echo "[smoke] $n FAILED — see /tmp/smoke_${n}.log"
      echo "        Likely missing CUDA kernels (mamba-ssm / causal-conv1d)."
      echo "        Install them, or SKIP_SMOKE=1 to bypass this check."
      exit 2
    fi
  done
fi

# ── skip anything already FINISHED in wandb (safe to re-run this script) ─────
RUN=()
for j in "${JOBS[@]}"; do
  path="${j%%|*}"; rest="${j#*|}"; proj="${rest%%|*}"; name="${rest##*|}"
  if [[ "${NO_RESUME:-0}" != 1 ]]; then
    state="$("$PY" - "$proj" "$name" <<'PYEOF' 2>/dev/null || true
import sys, wandb
proj, name = sys.argv[1], sys.argv[2]
try:
    for r in wandb.Api().runs(f"jai-malegaonkar/{proj}", filters={"displayName": name}):
        print(r.state); break
except Exception:
    pass
PYEOF
)"
    if [[ "$state" == "finished" ]]; then
      echo "[resume] $name already finished — skipping"
      continue
    fi
    [[ -n "$state" ]] && echo "[resume] $name exists in state '$state' — will re-run"
  fi
  RUN+=("$HERE/scripts/${path}.sh")
done

echo "launching ${#RUN[@]} of ${#JOBS[@]} runs"
[[ ${#RUN[@]} -gt 0 ]] || { echo "nothing to do — matrix already complete."; exit 0; }

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "[dry-run] would launch:"; printf '  %s\n' "${RUN[@]}"; exit 0
fi
printf '%s\n' "${RUN[@]}" | PYTHON="$PY" xargs -P "$P" -I{} bash {}
rc=$?
echo "=============================================================="
echo "done (exit $rc). Verify with:"
echo "  python -m experiments.analysis.appendix_stats"
echo "then re-pull per-seed data and drop the n=9 caveat in appendix.tex"
echo "  (Table 1 caption + the 'Seed completeness' paragraph)."
echo "=============================================================="
exit $rc
