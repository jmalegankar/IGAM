"""MQAR sweep: compare hebbian_mode × difficulty in one CLI.

Trains the cell on a grid of (mode, n_pairs, extra_pad) configurations,
writes a single JSON results file, and prints a summary table.

Designed to run on CPU in 1-2h total (vs days for the equivalent PPO ablation).

Usage:
    python -m experiments.mqar.run_sweep
    python -m experiments.mqar.run_sweep --quick    # fewer configs, ~20 min
    python -m experiments.mqar.run_sweep --modes additive gated_delta_eps
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from experiments.mqar.mqar import MQARConfig, TrainConfig, train_one


RESULTS_DIR = Path(__file__).parent / "_results"

MODES = ["additive", "delta", "gated_delta", "gated_delta_eps"]

# Difficulty grid — each entry is (n_pairs, extra_pad).
# Easy:    few pairs, no gap.
# Medium:  more pairs, small gap.
# Hard:    many pairs, big gap.
DIFFICULTY_GRID = [
    ("easy",   {"n_pairs":  4, "extra_pad":   0}),
    ("medium", {"n_pairs": 16, "extra_pad":  16}),
    ("hard",   {"n_pairs": 32, "extra_pad":  64}),
]

QUICK_GRID = [
    ("easy",   {"n_pairs":  4, "extra_pad":   0}),
    ("medium", {"n_pairs": 16, "extra_pad":  16}),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--quick", action="store_true",
                        help="2 difficulties × N modes × 1000 steps. ~20 min on CPU.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override training steps per config (default 2000 full, 1000 quick)")
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--n-queries", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default _results/sweep_<timestamp>.json)")
    args = parser.parse_args()

    device = torch.device(args.device)
    grid = QUICK_GRID if args.quick else DIFFICULTY_GRID
    default_steps = 1000 if args.quick else 2000
    n_steps = args.steps if args.steps is not None else default_steps

    total = len(args.modes) * len(grid)
    print(f"Sweep: {len(args.modes)} modes × {len(grid)} difficulties = {total} runs")
    print(f"  steps/run: {n_steps},  device: {args.device}\n")

    results = []
    t0 = time.perf_counter()
    n = 0
    for diff_name, diff in grid:
        for mode in args.modes:
            n += 1
            print(f"[{n}/{total}] mode={mode}  difficulty={diff_name}  {diff}")
            mcfg = MQARConfig(vocab_size=args.vocab, n_queries=args.n_queries, **diff)
            tcfg = TrainConfig(hebbian_mode=mode, batch_size=args.batch_size,
                               n_steps=n_steps, seed=args.seed)
            r = train_one(mcfg, tcfg, device, verbose=False)
            r["difficulty"] = diff_name
            results.append(r)
            print(f"   → acc={r['final_acc']:.3f}  "
                  f"(chance={r['chance']:.4f}, {r['wallclock_s']:.1f}s)\n")

    elapsed = time.perf_counter() - t0

    # Pretty summary table
    print("=" * 72)
    print(f"SWEEP DONE in {elapsed/60:.1f} min")
    print("=" * 72)
    header = f"{'mode':<18}" + "".join(f"{d[0]:>15}" for d in grid)
    print(header)
    print("-" * len(header))
    for mode in args.modes:
        row = f"{mode:<18}"
        for diff_name, diff in grid:
            acc = next(r["final_acc"] for r in results
                       if r["hebbian_mode"] == mode and r["difficulty"] == diff_name)
            row += f"{acc:>14.3f} "
        print(row)
    chance = 1.0 / args.vocab
    print(f"\n(chance accuracy = {chance:.4f})")

    # Persist
    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "elapsed_s": elapsed,
        "grid": [{"name": n, **d} for n, d in grid],
        "modes": args.modes,
        "vocab": args.vocab,
        "n_steps": n_steps,
        "results": results,
    }, indent=2))
    print(f"\nresults: {out_path}")


if __name__ == "__main__":
    main()
