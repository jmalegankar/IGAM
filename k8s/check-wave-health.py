#!/usr/bin/env python
"""Wave go/no-go health check for the memory-training launch.

Reports, for a wandb project (optionally filtered to one seed wave), how many runs
are alive vs disconnected and how fresh their heartbeats are — so you can decide
whether to launch the next seed wave or stop and diagnose.

Usage:
    python k8s/check-wave-health.py                       # whole memrl-memtrain-mpg
    python k8s/check-wave-health.py --seed 0              # just the s0 wave
    python k8s/check-wave-health.py --project memrl-memtrain-mpg --seed 0 --stale-min 5

Verdict: GO if (almost) every started run has a fresh heartbeat; HOLD otherwise.
Pairs naturally with: kubectl get jobs -l app=memrl-memtrain
"""
from __future__ import annotations

import argparse
import datetime
import time

import wandb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="jai-malegaonkar")
    ap.add_argument("--project", default="memrl-memtrain-mpg")
    ap.add_argument("--seed", type=int, default=None, help="filter to one seed wave")
    ap.add_argument("--stale-min", type=float, default=5.0,
                    help="heartbeat older than this = disconnected")
    args = ap.parse_args()

    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{args.entity}/{args.project}"))
    if args.seed is not None:
        runs = [r for r in runs if r.name.endswith(f"seed{args.seed}")]

    now = time.time()
    fresh = stale = nohb = 0
    states: dict[str, int] = {}
    for r in runs:
        states[r.state] = states.get(r.state, 0) + 1
        hb = r._attrs.get("heartbeatAt")
        if not hb:
            nohb += 1
            continue
        age = (now - datetime.datetime.fromisoformat(hb.replace("Z", "+00:00")).timestamp()) / 60
        if age <= args.stale_min:
            fresh += 1
        else:
            stale += 1

    total = len(runs)
    wave = f" seed{args.seed} wave" if args.seed is not None else ""
    print(f"{args.project}{wave}: {total} runs")
    print(f"  states: {states}")
    print(f"  heartbeat: {fresh} fresh (≤{args.stale_min:g}m) · {stale} stale/disconnected · {nohb} never-started")
    # GO if no started run is disconnected (allow a few not-yet-started)
    if total == 0:
        print("  → no runs yet (wave not launched or not registered)")
    elif stale == 0:
        print(f"  → ✅ GO — all started runs are alive; safe to launch the next wave")
    else:
        frac = stale / max(1, fresh + stale)
        print(f"  → ⛔ HOLD — {stale} disconnected ({frac:.0%}); the cluster is choking. "
              f"Do NOT add more load; investigate (egress, wandb backend, CPU) before scaling.")


if __name__ == "__main__":
    main()
