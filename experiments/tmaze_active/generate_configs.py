"""Generate the Active T-Maze sweep (configs + run scripts).

Extends Ni et al. (NeurIPS 2023, "When Do Transformers Shine in RL?") from their
LSTM-vs-Transformer comparison to the full modern memory-cell zoo, and adds an
intrinsic-bonus arm.

Three axes:
  1. CELL          — the 11-cell lineup (same as the S13 study).
  2. INTRINSIC     — {none, noveld, e3b_idm}. On Active T-Maze a bonus is NOT an
                     exploration aid (the agent reaches the junction fine); it is
                     REWARD DENSIFICATION that shortens the effective credit-
                     assignment horizon. Hypothesis: helps MORE as L grows.
  3. L (corridor)  — the memory AND credit-assignment horizon, dialed together.

  11 cells × 3 intrinsics × |L| lengths × 3 seeds.

Why Active (not Passive): Passive auto-traverses the corridor (credit-assignment
horizon = 1), so a NovelD/E3B bonus is a literal no-op. Active makes every one of
the L corridor actions on-path to the reward, so the bonus can actually act.

T-Maze-specific design choices (vs the S13 configs):
  * obs is Box(3,) ⇒ the FlatEncoder uses its MLP path (no CNN). Cell-agnostic.
  * FULL-EPISODE BPTT: episodes are exactly L steps, so chunk_len = n_steps = L.
    Gradient connects the t=0 cue to the t=L decision — the whole point. Cheap
    here (2-3 scalar obs), unlike S13 where full BPTT over 845 steps is infeasible.
  * γ scaled to L: γ = 1 − 1/(2L) (effective horizon ≈ 2L ≥ L), so the sparse
    terminal reward survives discounting over the corridor. (γ^L ≈ 0.6 at every L.)
  * GTrXL mem_len = L so its attention window actually spans the corridor (else
    the cue falls out of cache and the memory test is unwinnable for it).
  * total_timesteps grows with L (credit assignment gets harder): 2M + 16k·L.
  * Mamba-2 keeps its halved LR (PPO-stability exception, same as S13).

Usage:
    python experiments/tmaze_active/generate_configs.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from train import DEFAULT_CELL_KWARGS  # noqa: E402

CELLS = [
    "GRU", "LSTM", "mLSTM", "LRU", "Mamba2", "FFM",
    "GatedDeltaNet", "SHM", "GTrXL", "RetNet", "LinearTransformer",
]
INTRINSICS = ["none", "noveld", "e3b_idm"]
L_VALUES = [50, 100, 250, 500]
LAMBDA_INTRINSIC = 0.01

# NOTE: the reward shaping lives in the env now. TMazeEnv is a faithful port of
# Ni et al., with a horizon-normalized lag penalty (−1/L per step behind pace, so
# the worst case sums to −1.0) and the Active oracle-fetch mechanic. Nothing to
# set here — memoryless floor = 0.5, optimal = 1.0.

# Mamba-2 keeps the halved LR (optimizer-stability exception). Everything else
# uses BASE['lr'] — param-matched comparison, no per-cell optimizer tuning.
PER_CELL_LR = {"Mamba2": 3.0e-4}

BASE = {
    "n_envs": 32,                 # toy env ⇒ cheap; 32 episodes / rollout
    "encoder_dim": 128,
    "encoder_hidden": 256,
    "lr": 6.0e-4,
    "n_epochs": 4,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 1.0,
    "max_grad_norm": 0.5,
    "target_kl": 0.05,
    "n_chunks_per_batch": 16,     # 32 chunks/rollout → 2 minibatches of 16
    "eval_every_rollouts": 20,
    "n_eval_episodes": 50,        # cheap toy eval; tames the ±cue variance
    "wandb": True,
    # wandb_project is set PER-L in build_cfg (memrl-tmaze-active-L<L>).
}


def _episode_length(L: int) -> int:
    # Active T-Maze: corridor_length + 2*oracle_length + 1, oracle_length=1.
    return L + 3


def _gamma_for_ep(ep_len: int) -> float:
    # Effective horizon ~ 2·ep_len so the terminal goal survives discounting.
    return round(1.0 - 1.0 / (2 * ep_len), 6)


def _timesteps_for_L(L: int) -> int:
    return int(2_000_000 + 16_000 * L)


def build_cfg(cell: str, intrinsic: str, L: int) -> dict:
    if cell not in DEFAULT_CELL_KWARGS:
        raise KeyError(f"{cell} not in train.py DEFAULT_CELL_KWARGS")
    ep_len = _episode_length(L)
    kwargs = dict(DEFAULT_CELL_KWARGS[cell])
    if cell == "GTrXL":
        kwargs["mem_len"] = ep_len     # attention window must span the full episode

    ordered = {
        "env_name": f"TMaze-Active-L{L}-v0",
        "n_envs": BASE["n_envs"],
        "total_timesteps": _timesteps_for_L(L),
        "seed": 0,                      # overridden per run via --seed
        "cell": {"name": cell, "kwargs": kwargs},
        "encoder_dim": BASE["encoder_dim"],
        "encoder_hidden": BASE["encoder_hidden"],
        "lr": PER_CELL_LR.get(cell, BASE["lr"]),
        "n_steps": ep_len,              # one full episode per env per rollout
        "n_epochs": BASE["n_epochs"],
        "gamma": _gamma_for_ep(ep_len),
        "gae_lambda": BASE["gae_lambda"],
        "clip_range": BASE["clip_range"],
        "ent_coef": BASE["ent_coef"],
        "vf_coef": BASE["vf_coef"],
        "max_grad_norm": BASE["max_grad_norm"],
        "target_kl": BASE["target_kl"],
        "chunk_len": ep_len,            # FULL-episode BPTT
        "n_chunks_per_batch": BASE["n_chunks_per_batch"],
        "intrinsic": intrinsic,
    }
    if intrinsic != "none":
        ordered["lambda_intrinsic"] = LAMBDA_INTRINSIC
    for k in ("eval_every_rollouts", "n_eval_episodes", "wandb"):
        ordered[k] = BASE[k]
    # Each corridor length gets its OWN wandb project so the cell × intrinsic
    # comparison at a fixed L lives on one dashboard (lengths aren't comparable
    # on the same axes — reward scale, horizon, and difficulty all shift with L).
    ordered["wandb_project"] = f"memrl-tmaze-active-L{L}"
    return ordered


# none → run_<cell>_L<L>.sh ; bonus → explore_<cell>_<intrinsic>_L<L>.sh
# (mirrors the S13 run_/explore_ split so the k8s launcher globs both.)
_SCRIPT_TEMPLATE = """\
#!/usr/bin/env bash
# AUTO-GENERATED by generate_configs.py — regenerate, don't hand-edit.
# Active T-Maze: __CELL__ + __INTRINSIC__ at corridor length L=__L__, all seeds.
# Env knobs: PYTHON DEVICE RUNS_DIR SEEDS PARALLEL EXTRA (same as S13 scripts).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"; RUNS_DIR="${RUNS_DIR:-runs/tmaze_active}"
SEEDS="${SEEDS:-0 1 2}"; PARALLEL="${PARALLEL:-1}"; EXTRA="${EXTRA:-}"
CFG="$HERE/../configs/__CFG__"
LOGDIR="$HERE/../_logs"; mkdir -p "$LOGDIR"
TAG="__TAG__"
echo "[$TAG] python=$PYTHON device=$DEVICE seeds=[$SEEDS] parallel=$PARALLEL"
pids=()
for s in $SEEDS; do
  if [[ "$PARALLEL" == "1" ]]; then
    "$PYTHON" -m train --config "$CFG" --seed "$s" \\
        --runs-dir "$RUNS_DIR" --device "$DEVICE" $EXTRA \\
        > "$LOGDIR/${TAG}_seed${s}.log" 2>&1 &
    pids+=("$!")
    echo "  launched $TAG seed=$s (pid $!) → $LOGDIR/${TAG}_seed${s}.log"
  else
    echo "  $TAG seed=$s (sequential)"
    "$PYTHON" -m train --config "$CFG" --seed "$s" \\
        --runs-dir "$RUNS_DIR" --device "$DEVICE" $EXTRA
  fi
done
if [[ "$PARALLEL" == "1" ]]; then
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ "$fail" == "0" ]] || { echo "[$TAG] a seed FAILED — see $LOGDIR"; exit 1; }
fi
echo "[$TAG] all seeds done."
"""

_RUN_ALL_TEMPLATE = """\
#!/usr/bin/env bash
# AUTO-GENERATED. Run every Active T-Maze leaf script (cells serial, seeds
# parallel within). All env knobs propagate.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/run_*.sh "$HERE"/explore_*.sh; do
  bn="$(basename "$s")"
  [[ "$bn" == "run_all.sh" ]] && continue
  echo "=== $bn ==="
  bash "$s"
done
echo "All Active T-Maze conditions done."
"""


def main() -> None:
    cfg_dir = HERE / "configs"
    scr_dir = HERE / "scripts"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    scr_dir.mkdir(parents=True, exist_ok=True)

    n_cfg = n_scr = 0
    for cell in CELLS:
        for intrinsic in INTRINSICS:
            for L in L_VALUES:
                cfg = build_cfg(cell, intrinsic, L)
                cfg_name = f"tmaze_active_{cell}_{intrinsic}_L{L}.yaml"
                with open(cfg_dir / cfg_name, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
                n_cfg += 1

                if intrinsic == "none":
                    scr_name = f"run_{cell}_L{L}.sh"
                    tag = f"{cell}_none_L{L}"
                else:
                    scr_name = f"explore_{cell}_{intrinsic}_L{L}.sh"
                    tag = f"{cell}_{intrinsic}_L{L}"
                body = (_SCRIPT_TEMPLATE
                        .replace("__CELL__", cell)
                        .replace("__INTRINSIC__", intrinsic)
                        .replace("__L__", str(L))
                        .replace("__CFG__", cfg_name)
                        .replace("__TAG__", tag))
                p = scr_dir / scr_name
                p.write_text(body)
                os.chmod(p, 0o755)
                n_scr += 1

    run_all = scr_dir / "run_all.sh"
    run_all.write_text(_RUN_ALL_TEMPLATE)
    os.chmod(run_all, 0o755)

    print(f"cells={len(CELLS)} intrinsics={len(INTRINSICS)} L={L_VALUES}")
    print(f"wrote {n_cfg} configs → {cfg_dir}")
    print(f"wrote {n_scr} leaf scripts (+ run_all.sh) → {scr_dir}")
    print(f"total conditions = {n_cfg}  (× 3 seeds = {n_cfg*3} runs)")


if __name__ == "__main__":
    main()
