"""only_missing — diff a registry grid against FINISHED wandb runs → emit a re-launch plan.

The cluster oversubscription problem (2026-07): Bridge blind-re-launches the whole grid,
finished cells included, so hundreds of jobs contend for GPUs and the excess instant-fail
(0 steps / 0 min). This tool closes that loop: it resolves the full planned grid from the
registry (the single source of truth), asks wandb which (cell, density, bonus, seed) cells
are already DONE (finished at ≥done-frac of budget) or IN-FLIGHT (running), and emits ONLY
the real gaps — grouped by seed so Bridge launches small, schedulable waves.

Usage:
    python -m experiments.memory_training.only_missing E1:S13     # staged per-seed commands
    python -m experiments.memory_training.only_missing HPOT --mode regex   # one ONLY= regex
    python -m experiments.memory_training.only_missing E5 --mode list       # basenames

Modes:
    staged (default) : ready-to-paste `EID=.. ENV=.. ONLY='<regex>' k8s/launch-…` per seed
    regex            : a single ONLY= alternation regex over all missing basenames
    list             : missing script basenames, one per line

A missing (cell,density,bonus,seed) is anything NOT covered by a finished run at
≥ done-frac·budget steps AND not currently running. A wandb project that doesn't exist yet
(e.g. HPOT/E5 never launched) → every cell is missing (launch the whole arm).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.memory_training.registry import CORE, METHOD  # noqa: E402


def _arms_for(emit_id: str):
    """Resolve an emit-id (E1 | E1:<env> | <method-eid>) to registry arm objects."""
    if emit_id == "E1" or emit_id.startswith("E1:"):
        env = emit_id.split(":", 1)[1] if ":" in emit_id else None
        arms = [a for a in CORE if env is None or a.env == env]
        if not arms:
            raise SystemExit(f"no core env-arm matches {emit_id!r}")
        return "E1", arms
    if emit_id in METHOD:
        return emit_id, [METHOD[emit_id]]
    raise SystemExit(f"unknown emit-id {emit_id!r} (expected E1 | E1:<env> | a method eid)")


def _cellname(cfg: dict):
    c = cfg.get("cell")
    return c.get("name") if isinstance(c, dict) else c


def _basename(cell, dens, intr, seed) -> str:
    return f"s{seed}_{cell}_{dens}_{intr}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("emit_id", help="E1 | E1:<env> | <method-eid> (e.g. E1:S13, HPOT, E5)")
    ap.add_argument("--entity", default="jai-malegaonkar")
    ap.add_argument("--mode", choices=["staged", "regex", "list"], default="staged")
    ap.add_argument("--done-frac", type=float, default=0.9,
                    help="a run counts as DONE at ≥ this fraction of its budget (default 0.9)")
    args = ap.parse_args()

    import wandb
    api = wandb.Api()
    launch_eid, arms = _arms_for(args.emit_id)

    total_missing = 0
    for arm in arms:
        project = arm.project
        budget = arm.budget
        env_tag = getattr(arm, "env", None)                 # EnvArm has .env; Method doesn't
        dens_labels = [d.label for d in arm.densities]
        planned = {(c, d, i, s)
                   for c in arm.cells for d in dens_labels
                   for i in arm.intrinsics for s in arm.seeds}

        covered = set()
        exists = True
        try:
            runs = api.runs(f"{args.entity}/{project}")
            for r in runs:
                cfg = r.config
                key = (_cellname(cfg), cfg.get("wandb_group"),
                       cfg.get("intrinsic"), cfg.get("seed"))
                if key[0] is None or key[1] is None:
                    continue
                gs = r.summary.get("global_step", 0) or 0
                if r.state == "running" or (r.state == "finished" and gs >= args.done_frac * budget):
                    covered.add(key)
        except Exception as e:
            exists = False
            print(f"# [{project}] not found / error ({type(e).__name__}) → all {len(planned)} "
                  f"cells missing", file=sys.stderr)

        missing = sorted(planned - covered)
        total_missing += len(missing)
        tag = env_tag or launch_eid
        print(f"# [{tag}→{project}] planned {len(planned)} · covered "
              f"{len(covered) if exists else 0} · MISSING {len(missing)}"
              f"{'  (project absent)' if not exists else ''}", file=sys.stderr)
        if not missing:
            continue

        env_part = f"ENV={env_tag} " if env_tag else ""
        if args.mode == "list":
            for k in missing:
                print(f"{_basename(*k)}.sh")
        elif args.mode == "regex":
            rx = "^(" + "|".join(_basename(*k) for k in missing) + r")\.sh$"
            print(f"# {tag}: {len(missing)} missing")
            print(rx)
        else:  # staged: one throttled command per seed
            by_seed: dict[int, list] = defaultdict(list)
            for k in missing:
                by_seed[k[3]].append(k)
            for s in sorted(by_seed):
                ks = by_seed[s]
                rx = "^(" + "|".join(_basename(*k) for k in ks) + r")\.sh$"
                print(f"{env_part}EID={launch_eid} ONLY='{rx}' "
                      f"k8s/launch-memtrain-jobs.sh   # {tag} seed {s}: {len(ks)} runs")

    print(f"# TOTAL MISSING across arm(s): {total_missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
