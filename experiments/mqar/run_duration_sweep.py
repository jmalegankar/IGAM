"""Duration sweep: does longer training change the mode ranking on EASY?

The first sweep used 1000 steps and showed gated_delta > gated_delta_eps > delta > additive
on easy. But MQAR convention is 5000-10000 steps. If 5000 steps shows a totally different
ranking (e.g., all modes reach 90%+), our 1000-step result was just noise.

Holds the task at EASY (4 pairs, no gap), assoc_size=64, runs all 4 modes for 5000 steps.

Usage:
    python -m experiments.mqar.run_duration_sweep
    python -m experiments.mqar.run_duration_sweep --steps 10000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.mqar.mqar import MQARConfig, TrainConfig, train_one


RESULTS_DIR = Path(__file__).parent / "_results"

TASK = {"n_pairs": 4, "extra_pad": 0}   # EASY
MODES = ["additive", "delta", "gated_delta", "gated_delta_eps"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--vocab", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--assoc-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    total = len(args.modes)
    print(f"Duration sweep: {total} modes × {args.steps} steps on EASY task {TASK}")
    print(f"  vocab={args.vocab}  assoc_size={args.assoc_size}  device={args.device}\n")

    results = []
    t0 = time.perf_counter()
    for i, mode in enumerate(args.modes, 1):
        print(f"[{i}/{total}] mode={mode}")
        mcfg = MQARConfig(vocab_size=args.vocab, n_queries=4, **TASK)
        tcfg = TrainConfig(
            hebbian_mode=mode,
            batch_size=args.batch_size,
            n_steps=args.steps,
            seed=args.seed,
            assoc_size=args.assoc_size,
        )
        r = train_one(mcfg, tcfg, device, verbose=False)
        results.append(r)
        # Also report intermediate trajectory so we see if it's still climbing
        last_hist = r.get("history", [])
        if len(last_hist) >= 3:
            traj = "  ".join(f"@{h['step']}={h['eval_acc']:.3f}" for h in last_hist[-5:])
        else:
            traj = ""
        print(f"   → final_acc={r['final_acc']:.3f}  "
              f"(chance={r['chance']:.4f}, {r['wallclock_s']:.1f}s)")
        if traj:
            print(f"     trajectory: {traj}\n")
        else:
            print()

    elapsed = time.perf_counter() - t0
    print("=" * 72)
    print(f"DONE in {elapsed/60:.1f} min")
    print("=" * 72)
    print(f"{'mode':<20}  {'final_acc':>10}  {'vs chance':>10}")
    print("-" * 46)
    chance = 1.0 / args.vocab
    for r in results:
        ratio = r["final_acc"] / chance
        print(f"{r['hebbian_mode']:<20}  {r['final_acc']:>10.3f}  {ratio:>9.1f}×")
    print(f"\n(chance = {chance:.4f})")

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"duration_sweep_{args.steps}steps_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "elapsed_s": elapsed,
        "task": TASK,
        "vocab": args.vocab,
        "assoc_size": args.assoc_size,
        "n_steps": args.steps,
        "modes": args.modes,
        "results": results,
    }, indent=2))
    print(f"\nresults: {out_path}")


if __name__ == "__main__":
    main()
