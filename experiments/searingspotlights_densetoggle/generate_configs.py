"""SearingSpotlights density toggle — the ANTI-DENSE / freeze-rescue replication.

Third memory-gym task, third memory type: DEAD-RECKONING — the arena's global
light dims off after a brief glimpse; the agent must localize itself, the coin,
and the exit from memory while dodging moving, damaging spotlights. The most
robotics-flavored memory demand in the suite (localization under intermittent
observation + dynamic hazard avoidance).

Division of labor across the three memory-gym tasks:
  MysteryPath  — all three arms (sparse / aligned+0.1 / anti-fall-penalty)
  MortarMayhem — sparse vs ALIGNED-dense (exactly return-matched, native)
  SearingSpotlights — sparse vs ANTI-dense (native per-step penalty knob)

Arms (native knobs only):
  sparse = env DEFAULTS (exit +1.0, coin +0.25, all per-step terms 0;
           measured: random success 0.005, max return 1.25)
  anti   = defaults + reward_inside_spotlight = -0.008 (per lit step,
           horizon-consistent with MysteryPath's fall penalty)
Return-matching is EXACT: an optimal agent is never lit, so both arms max out
at 1.25 — the same optimum-preservation lemma as the fall penalty.

Registered predictions (docs/revelation_and_densification.md):
  * sparse: e3b > none (monetizer; random succ 0.5% -> hard-sparse).
  * anti, none-arm: tests the FREEZE BOUNDARY CONDITION — unlike MysteryPath,
    spotlights WANDER, so standing still does not avoid the penalty (no
    obvious absorbing zero-cost region). Full freeze would EXTEND the
    lazy-robot result; no-freeze localizes it to avoidable penalties. Either
    outcome is informative; we do not pre-commit a sign for none-anti.
  * anti: e3b > none (rescue/robustness — the bonus out-bids the fine and
    keeps exploration alive, as measured on MysteryPath at n=15).

Design: {sparse, anti} x {none, e3b_idm} x {GRU, RetNet, GatedDeltaNet} x
5 seeds = 60 runs; {none, e3b} PAIR per GPU -> 30 jobs. HPs transferred from
the MysteryPath winner (global-HP fair-comparison story). max_steps=256, so
horizon-consistent penalty would be 1/256; we keep -0.008 (=1/128) for
cross-env comparability with MysteryPath's penalty — noted in the doc.

Usage:
    python experiments/searingspotlights_densetoggle/generate_configs.py
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

CELLS = ["GRU", "RetNet", "GatedDeltaNet"]
DENSITIES = ["sparse", "anti"]
INTRINSICS = ["none", "e3b_idm"]
SEEDS = [0, 1, 2, 3, 4]

ARM_OPTS = {
    # env defaults ARE sparse (exit 1.0 + coin 0.25, no per-step terms) —
    # pass nothing and let defaults apply.
    "sparse": {},
    # native per-step penalty while inside a spotlight; optimal agent never
    # lit -> max return identical to sparse (1.25).
    "anti": {"reward_inside_spotlight": -0.008},
}

BASE = {
    "env_name": "SearingSpotlights-v0",
    "n_envs": 16,
    "total_timesteps": 10_000_000,
    "encoder_dim": 128,
    "encoder_hidden": 256,
    "lr": 1.0e-4,                     # WINNER (transferred)
    "n_steps": 512,
    "n_epochs": 4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.008,
    "vf_coef": 1.0,
    "max_grad_norm": 0.5,
    "target_kl": 0.05,
    "chunk_len": 64,                  # WINNER (transferred)
    "n_chunks_per_batch": 32,
    "lambda_intrinsic": 0.03,         # WINNER (transferred, e3b only)
    "eval_every_rollouts": 10,
    "n_eval_episodes": 20,
    "wandb": True,
    "wandb_project": "memrl-ss-toggle",
}


def build_cfg(cell: str, density: str, intrinsic: str) -> dict:
    ordered = {"env_name": BASE["env_name"]}
    if ARM_OPTS[density]:
        ordered["env_kwargs"] = {"reset_options": dict(ARM_OPTS[density])}
    ordered.update({
        "n_envs": BASE["n_envs"],
        "total_timesteps": BASE["total_timesteps"],
        "seed": 0,                      # overridden per run via --seed
        "cell": {"name": cell, "kwargs": dict(DEFAULT_CELL_KWARGS[cell])},
        "encoder_dim": BASE["encoder_dim"],
        "encoder_hidden": BASE["encoder_hidden"],
    })
    for k in ("lr", "n_steps", "n_epochs", "gamma", "gae_lambda", "clip_range",
              "ent_coef", "vf_coef", "max_grad_norm", "target_kl",
              "chunk_len", "n_chunks_per_batch"):
        ordered[k] = BASE[k]
    ordered["intrinsic"] = intrinsic
    if intrinsic != "none":
        ordered["lambda_intrinsic"] = BASE["lambda_intrinsic"]
    for k in ("eval_every_rollouts", "n_eval_episodes", "wandb", "wandb_project"):
        ordered[k] = BASE[k]
    ordered["run_name"] = f"{cell}-{density}-{intrinsic}"
    return ordered


_SCRIPT_TEMPLATE = """\
#!/usr/bin/env bash
# AUTO-GENERATED. SearingSpotlights density toggle: __CELL__ / __DENSITY__ at
# seed __SEED__ — runs {none, e3b_idm} in PARALLEL on one GPU (84x84 pair).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${DEVICE:-cuda}"; RUNS_DIR="${RUNS_DIR:-runs/searingspotlights_densetoggle}"; EXTRA="${EXTRA:-}"
LOGDIR="$HERE/../_logs"; mkdir -p "$LOGDIR"
CELL="__CELL__"; DENSITY="__DENSITY__"; SEED="__SEED__"
echo "[s$SEED | $CELL $DENSITY] none + e3b_idm in parallel on one GPU"
pids=()
for intr in none e3b_idm; do
  CFG="$HERE/../configs/ssdt_${CELL}_${DENSITY}_${intr}.yaml"
  "$PYTHON" -m train --config "$CFG" --seed "$SEED" \\
      --runs-dir "$RUNS_DIR" --device "$DEVICE" $EXTRA \\
      > "$LOGDIR/${CELL}_${DENSITY}_${intr}_seed${SEED}.log" 2>&1 &
  pids+=("$!"); echo "  launched $CELL/$DENSITY/$intr seed=$SEED (pid $!)"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ "$fail" == "0" ]] || { echo "[s$SEED | $CELL $DENSITY] a run FAILED — see $LOGDIR"; exit 1; }
echo "[s$SEED | $CELL $DENSITY] done."
"""

_RUN_ALL_TEMPLATE = """\
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$HERE"/ssdt_*.sh; do
  bn="$(basename "$s")"; [[ "$bn" == "run_all.sh" ]] && continue
  echo "=== $bn ==="; bash "$s"
done
echo "All SearingSpotlights-toggle jobs done."
"""


def main() -> None:
    cfg_dir = HERE / "configs"; scr_dir = HERE / "scripts"
    cfg_dir.mkdir(parents=True, exist_ok=True); scr_dir.mkdir(parents=True, exist_ok=True)
    for old in list(cfg_dir.glob("ssdt_*.yaml")) + list(scr_dir.glob("ssdt_*.sh")):
        old.unlink()

    for cell in CELLS:
        for density in DENSITIES:
            for intr in INTRINSICS:
                cfg = build_cfg(cell, density, intr)
                with open(cfg_dir / f"ssdt_{cell}_{density}_{intr}.yaml", "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)

    n_scr = 0
    for seed in SEEDS:
        for cell in CELLS:
            for density in DENSITIES:
                body = (_SCRIPT_TEMPLATE
                        .replace("__CELL__", cell)
                        .replace("__DENSITY__", density)
                        .replace("__SEED__", str(seed)))
                p = scr_dir / f"ssdt_s{seed}_{cell}_{density}.sh"
                p.write_text(body); os.chmod(p, 0o755); n_scr += 1
    ra = scr_dir / "run_all.sh"; ra.write_text(_RUN_ALL_TEMPLATE); os.chmod(ra, 0o755)

    n_cfg = len(CELLS) * len(DENSITIES) * len(INTRINSICS)
    print("env=SearingSpotlights (sparse=defaults 1.25max | anti=-0.008/lit, return-matched)")
    print("project=memrl-ss-toggle  budget=10M")
    print(f"HPs transferred: e3b_idm λ=0.03 ck=64 lr=1e-4 | cells={CELLS}")
    print(f"wrote {n_cfg} configs + {n_scr} scripts (+ run_all.sh)")
    print(f"= {n_scr} GPU jobs (none+e3b pair each) = {n_cfg * len(SEEDS)} runs")


if __name__ == "__main__":
    main()
