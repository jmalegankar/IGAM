"""DTH-LMU Hebbian-rule ablation runner.

Compares the four Hebbian update modes on the four diagnostic POPGym tasks
to validate the synthesized Doc-1 + Doc-2 architecture decisions:

  additive          baseline — pure Hebbian outer-product writes.
                    Establishes the O(T/A) interference floor.
  delta             DeltaNet rule (β = α = 1).
                    Tests: does targeted overwrite alone fix the failures?
  gated_delta       Gated DeltaNet (α, β from hidden state).
                    Tests: do learned scalar gates add value over plain delta?
  gated_delta_eps   ε-gated delta — β = σ(a·ε_mem − b), α from h.
                    Tests: does memory-conditioned curiosity help?

Each task config in benchmarks/phase_a/ablation/ is read, the kwarg
`hebbian_mode` is overridden, and the resulting cfg is dumped to a
per-variant yaml under `_generated/`. Runs land under `runs/dth_lmu_ablations/`.

Usage:
    # Full grid (4 tasks × 4 modes × 2 seeds = 32 runs)
    python -m experiments.dth_lmu_ablations.run_ablations

    # Subset, short, 3 concurrent on a single GPU (good for a 3070)
    python -m experiments.dth_lmu_ablations.run_ablations \
        --tasks autoencode_medium battleship_easy \
        --modes additive gated_delta_eps \
        --seeds 0 1 \
        --timesteps 5_000_000 \
        --parallel 3

    # Print commands without launching
    python -m experiments.dth_lmu_ablations.run_ablations --dry-run
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO_ROOT     = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "phase_a" / "ablation"
GENERATED_DIR = Path(__file__).parent / "_generated"

TASKS = [
    "autoencode_medium",      # content recall — delta should help most
    "countrecall_medium",     # cumulative stats — LegS-fixed already; ablations should be neutral
    "battleship_easy",        # spatial key→value — delta + ε-gating should help most
    "repeat_previous_medium", # recency — fast LMU dominates; ablations should be neutral
]

MODES = ["additive", "delta", "gated_delta", "gated_delta_eps"]

DEFAULT_SEEDS = [0, 1]


def write_variant_config(
    task: str,
    mode: str,
    timesteps: int | None = None,
    n_epochs: int | None = None,
) -> Path:
    """Read the base task config, override hebbian_mode + optional knobs,
    write a per-variant yaml.  Returns the path of the generated file."""
    base = BENCHMARK_DIR / f"dth_lmu_{task}_15M.yaml"
    if not base.exists():
        raise FileNotFoundError(f"Base config not found: {base}")
    with open(base) as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    cfg.setdefault("cell", {}).setdefault("kwargs", {})["hebbian_mode"] = mode
    if timesteps is not None:
        cfg["total_timesteps"] = int(timesteps)
    if n_epochs is not None:
        cfg["n_epochs"] = int(n_epochs)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{int(cfg['total_timesteps']) // 1_000_000}M"
    out = GENERATED_DIR / f"dth_lmu_{task}_{mode}_{suffix}.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def build_cmd(config_path: Path, seed: int, runs_dir: str) -> list[str]:
    return [
        sys.executable, "-m", "train",
        "--config", str(config_path),
        "--seed", str(seed),
        "--runs-dir", runs_dir,
    ]


def _run_one(label: str, cmd: list[str]) -> tuple[str, int]:
    """Launch one training run; capture stdout/stderr to a per-run log file
    so parallel runs don't interleave on the terminal."""
    log_dir = GENERATED_DIR / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label.replace(' ', '_').replace('/', '_')}.log"
    with open(log_path, "w") as logf:
        res = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return label, res.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", nargs="+", default=TASKS, choices=TASKS, metavar="TASK",
        help=f"Diagnostic tasks to ablate over. Choices: {TASKS}",
    )
    parser.add_argument(
        "--modes", nargs="+", default=MODES, choices=MODES, metavar="MODE",
        help=f"Hebbian update modes to test. Choices: {MODES}",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Seeds to run each (task, mode) pair with",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total_timesteps in all generated configs (e.g. 5_000_000)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=None,
        help="Override PPO n_epochs (default 10 in the base configs; 4 is faster)",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Concurrent runs (3070: 3-4 fits in 8GB VRAM; CPU env-stepping usually bottlenecks first)",
    )
    parser.add_argument(
        "--runs-dir", default="runs/dth_lmu_ablations",
        help="Root directory for run outputs",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    # Build the full job list up front
    jobs: list[tuple[str, list[str]]] = []
    for task in args.tasks:
        for mode in args.modes:
            config = write_variant_config(task, mode,
                                          timesteps=args.timesteps,
                                          n_epochs=args.n_epochs)
            for seed in args.seeds:
                label = f"task={task} mode={mode} seed={seed}"
                cmd   = build_cmd(config, seed, args.runs_dir)
                jobs.append((label, cmd))

    total = len(jobs)
    print(f"Plan: {len(args.tasks)} tasks × {len(args.modes)} modes × {len(args.seeds)} seeds = {total} runs")
    print(f"  tasks:     {args.tasks}")
    print(f"  modes:     {args.modes}")
    print(f"  seeds:     {args.seeds}")
    print(f"  timesteps: {args.timesteps or '(base config)'}")
    print(f"  n_epochs:  {args.n_epochs or '(base config)'}")
    print(f"  parallel:  {args.parallel}")
    print(f"  runs:      {args.runs_dir}\n")

    if args.dry_run:
        for i, (label, cmd) in enumerate(jobs, 1):
            print(f"[{i}/{total}] {label}\n    {' '.join(cmd)}")
        return

    if args.parallel <= 1:
        for i, (label, cmd) in enumerate(jobs, 1):
            print(f"\n[{i}/{total}] {label}")
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"  ERROR: exit code {res.returncode}")
    else:
        print(f"Launching up to {args.parallel} concurrent runs. "
              f"Per-run logs in {GENERATED_DIR / '_logs'}/\n")
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = {ex.submit(_run_one, label, cmd): label
                       for label, cmd in jobs}
            done = 0
            for fut in as_completed(futures):
                label, rc = fut.result()
                done += 1
                status = "OK" if rc == 0 else f"FAIL (rc={rc})"
                print(f"[{done}/{total}] {status}  {label}")

    print("\nAll ablation runs complete.")


if __name__ == "__main__":
    main()
