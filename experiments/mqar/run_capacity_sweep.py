"""Capacity sweep: does bigger assoc_size unlock medium MQAR?

Focused experiment: medium difficulty (16 pairs, gap=16), 3 modes,
3 assoc_sizes. Tests whether the collapse-at-medium failure mode in
the first sweep was a capacity bottleneck.

Compute estimate (CPU): 9 configs × ~8 min = ~70 min.

Usage:
    python -m experiments.mqar.run_capacity_sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.mqar.mqar import MQARConfig, TrainConfig, train_one


RESULTS_DIR = Path(__file__).parent / "_results"

# Hold task constant at MEDIUM — that's where prior sweep saw all-mode collapse.
TASK = {"n_pairs": 16, "extra_pad": 16}

MODES = ["additive", "gated_delta", "gated_delta_eps"]
ASSOC_SIZES = [64, 128, 256]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--vocab", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--assoc-sizes", nargs="+", type=int, default=ASSOC_SIZES)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    total = len(args.modes) * len(args.assoc_sizes)
    print(f"Capacity sweep: {len(args.modes)} modes × {len(args.assoc_sizes)} assoc_sizes "
          f"= {total} runs on MEDIUM difficulty {TASK}")
    print(f"  steps/run: {args.steps},  device: {args.device}\n")

    results = []
    t0 = time.perf_counter()
    n = 0
    for asize in args.assoc_sizes:
        for mode in args.modes:
            n += 1
            print(f"[{n}/{total}] mode={mode}  assoc_size={asize}")
            mcfg = MQARConfig(vocab_size=args.vocab, n_queries=4, **TASK)
            tcfg = TrainConfig(
                hebbian_mode=mode,
                batch_size=args.batch_size,
                n_steps=args.steps,
                seed=args.seed,
                assoc_size=asize,
            )
            r = train_one(mcfg, tcfg, device, verbose=False)
            r["assoc_size"] = asize
            results.append(r)
            print(f"   → acc={r['final_acc']:.3f}  "
                  f"(chance={r['chance']:.4f}, {r['wallclock_s']:.1f}s)\n")

    elapsed = time.perf_counter() - t0
    print("=" * 72)
    print(f"DONE in {elapsed/60:.1f} min")
    print("=" * 72)
    print(f"{'mode':<18}" + "".join(f"{f'A={a}':>12}" for a in args.assoc_sizes))
    print("-" * (18 + 12 * len(args.assoc_sizes)))
    for mode in args.modes:
        row = f"{mode:<18}"
        for asize in args.assoc_sizes:
            acc = next(r["final_acc"] for r in results
                       if r["hebbian_mode"] == mode and r["assoc_size"] == asize)
            row += f"{acc:>11.3f} "
        print(row)
    print(f"\n(chance = {1.0/args.vocab:.4f})")

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"capacity_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "elapsed_s": elapsed,
        "task": TASK,
        "vocab": args.vocab,
        "n_steps": args.steps,
        "modes": args.modes,
        "assoc_sizes": args.assoc_sizes,
        "results": results,
    }, indent=2))
    print(f"\nresults: {out_path}")


if __name__ == "__main__":
    main()
