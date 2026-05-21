"""Run all architectural ablations sequentially across multiple seeds.

Each ablation is: no-wrapper MemS13 + phiRand E3B, 5M steps.
Configs live in experiments/gex_replication/configs/ablation/.

Usage:
    # All ablations, seeds 0-2
    python -m experiments.gex_replication.run_ablations

    # Single ablation, single seed
    python -m experiments.gex_replication.run_ablations --variants full_system no_gate --seeds 0

    # Dry run (print commands, don't execute)
    python -m experiments.gex_replication.run_ablations --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ABLATION_DIR = Path(__file__).parent / "configs" / "ablation"

# Ordered list — full_system first so it's the visual reference in TensorBoard.
VARIANTS: list[str] = [
    "full_system",
    "no_gate",
    "no_multichannel",
    "no_dynamic_wquery",
    "no_ortho_wpre",
    "no_residual",
    "vanilla_lmu",
]

DEFAULT_SEEDS: list[int] = [0, 1, 2]


def build_cmd(variant: str, seed: int, runs_dir: str) -> list[str]:
    config = ABLATION_DIR / f"{variant}.yaml"
    if not config.exists():
        raise FileNotFoundError(f"Ablation config not found: {config}")
    return [
        sys.executable, "-m", "experiments.gex_replication.train",
        "--config", str(config),
        "--seed", str(seed),
        "--runs-dir", runs_dir,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants", nargs="+", default=VARIANTS,
        choices=VARIANTS, metavar="VARIANT",
        help=f"Which ablations to run (default: all). Choices: {VARIANTS}",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Seeds to run each variant with (default: 0 1 2)",
    )
    parser.add_argument(
        "--runs-dir", default="runs/ablations",
        help="Root directory for run outputs",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    total = len(args.variants) * len(args.seeds)
    print(f"Ablation plan: {len(args.variants)} variants × {len(args.seeds)} seeds = {total} runs")
    print(f"Variants: {args.variants}")
    print(f"Seeds:    {args.seeds}")
    print(f"Runs dir: {args.runs_dir}\n")

    for i, variant in enumerate(args.variants):
        for j, seed in enumerate(args.seeds):
            run_num = i * len(args.seeds) + j + 1
            cmd = build_cmd(variant, seed, args.runs_dir)
            label = f"[{run_num}/{total}] {variant} seed={seed}"
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"  {' '.join(cmd)}")
            print(f"{'='*60}")

            if not args.dry_run:
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"\nERROR: {label} exited with code {result.returncode}")
                    print("Continuing with next run...\n")

    print("\nAll ablation runs complete.")


if __name__ == "__main__":
    main()
