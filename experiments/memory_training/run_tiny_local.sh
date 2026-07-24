#!/usr/bin/env bash
# run_tiny_local.sh — run the unified TinyReproduce sweep (lr=1e-3, n=10) on a
# local GPU box, SHARDED BY SEED so two machines can split it with no wandb
# run-name collisions. These are PRODUCTION runs: they log to wandb project
# memrl-tiny-1e3 (NOT --no-wandb), because make_tables.py pulls from there.
#
# Per box (git pull + `wandb login` first):
#   SHARD=0-4 P=4  bash experiments/memory_training/run_tiny_local.sh   # 3070 Ti (PC)
#   SHARD=5-9 P=9  bash experiments/memory_training/run_tiny_local.sh   # DGX Spark
#
# Knobs (env vars):
#   SHARD   inclusive seed range "lo-hi"  (default 0-9 = everything)
#   P       concurrent runs               (default 4; Spark can go ~8-10)
#   DEVICE  torch device                  (default cuda)
#   NO_SNAP=1     skip --snapshot-to-wandb artifact uploads
#   SKIP_SMOKE=1  skip the GatedDeltaNet/Mamba2 CUDA-kernel preflight
#   NO_RESUME=1   don't query wandb to skip already-finished runs
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; cd "$REPO"
PY="${PYTHON:-python}"
SHARD="${SHARD:-0-9}"; P="${P:-4}"
export DEVICE="${DEVICE:-cuda}"
export RUNS_DIR="${RUNS_DIR:-runs/tiny1e3}"
[[ "${NO_SNAP:-0}" == 1 ]] && export SNAPSHOT_TO_WANDB=""
LO="${SHARD%-*}"; HI="${SHARD#*-}"
SDIR="$HERE/scripts/E1/TinyReproduce"
CDIR="$HERE/configs/E1/TinyReproduce"

# ── seed-filtered script list ────────────────────────────────────────────────
ALL=()
for s in $(seq "$LO" "$HI"); do
  for f in "$SDIR"/s${s}_*.sh; do [[ -e "$f" ]] && ALL+=("$f"); done
done
[[ ${#ALL[@]} -gt 0 ]] || { echo "no scripts for SHARD=$SHARD in $SDIR"; exit 1; }
echo "TinyReproduce sweep — SHARD=$SHARD (${#ALL[@]} scripts), P=$P, DEVICE=$DEVICE, project=memrl-tiny-1e3"

# ── CUDA-kernel preflight: GDN + Mamba2 must start on this GPU (50k steps, no wandb) ──
if [[ "${SKIP_SMOKE:-0}" != 1 ]]; then
  for C in GatedDeltaNet Mamba2; do
    f="$SDIR/s${LO}_${C}_sparse_none.sh"
    [[ -f "$f" ]] || continue
    echo "[smoke] $C (50k steps, no-wandb) on $DEVICE …"
    if EXTRA="--total-timesteps 50000 --no-wandb" DEVICE="$DEVICE" bash "$f" >"/tmp/smoke_${C}.log" 2>&1; then
      echo "[smoke] $C ok"
    else
      echo "[smoke] $C FAILED on $DEVICE — likely missing CUDA kernels (mamba-ssm / causal-conv1d)."
      echo "        See /tmp/smoke_${C}.log ; install the kernels or SKIP_SMOKE=1 to bypass."
      exit 2
    fi
  done
fi

# ── skip runs already FINISHED in wandb (clean resume) ───────────────────────
SKIP="$(mktemp)"; : > "$SKIP"
if [[ "${NO_RESUME:-0}" != 1 ]]; then
  echo "[resume] querying memrl-tiny-1e3 for finished runs …"
  "$PY" - <<'PYEOF' > "$SKIP" 2>/dev/null || true
import wandb
try:
    for r in wandb.Api().runs("jai-malegaonkar/memrl-tiny-1e3"):
        if r.state == "finished":
            print(r.name)
except Exception:
    pass
PYEOF
  echo "[resume] $(wc -l < "$SKIP" | tr -d ' ') finished runs will be skipped"
fi

# ── filter out finished, then run P-at-a-time ────────────────────────────────
RUN=()
for f in "${ALL[@]}"; do
  bn="$(basename "$f" .sh)"                    # s5_GatedDeltaNet_sparse_e3b_idm
  seed="${bn%%_*}"; seed="${seed#s}"           # 5
  stem="${bn#*_}"                              # GatedDeltaNet_sparse_e3b_idm
  cfg="$CDIR/${stem}.yaml"
  rn="$(grep -E '^run_name:' "$cfg" 2>/dev/null | awk '{print $2}')"
  full="${rn}-seed${seed}"                     # GatedDeltaNet-sparse-e3b_idm-seed5
  grep -qxF "$full" "$SKIP" 2>/dev/null && continue
  RUN+=("$f")
done
rm -f "$SKIP"

echo "launching ${#RUN[@]} runs (${#ALL[@]} in shard, $(( ${#ALL[@]} - ${#RUN[@]} )) already done)"
[[ ${#RUN[@]} -gt 0 ]] || { echo "nothing to do — shard complete."; exit 0; }
printf '%s\n' "${RUN[@]}" | xargs -P "$P" -I{} bash {}
echo "shard $SHARD done."
