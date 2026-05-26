"""Cell × Intrinsic-Reward sweep on POPGym diagnostic tasks.

Cross-product of memory cells and intrinsic-reward modules. Directly tests:
  (a) which cell can solve POPGym AR-class tasks (Autoencode, CountRecall),
  (b) which exploration mechanism makes the cell actually use its memory,
  (c) whether the thesis's Agency Principle generalises to a Gated DeltaNet
      memory cell (compare e3b_rand vs e3b_innov).

Default plan:
  cells:         GatedDeltaNet, LSTM (control)
  intrinsics:    none, rnd, e3b_rand, e3b_obs, e3b_innov, noveld, icm
  tasks:         autoencode_medium, countrecall_medium, battleship_easy,
                 repeat_previous_medium
  seeds:         0, 1

Generates per-variant YAMLs in _generated/ and launches train.py.

Usage:
    # Full grid (~88 runs)
    python -m experiments.cell_intrinsic_matrix.run_matrix

    # Smaller targeted sweep
    python -m experiments.cell_intrinsic_matrix.run_matrix \\
        --cells GatedDeltaNet --intrinsics none e3b_rand \\
        --tasks autoencode_medium --seeds 0 \\
        --timesteps 2_000_000 --parallel 2
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO_ROOT     = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "phase_a" / "ablation"
GENERATED_DIR = Path(__file__).parent / "_generated"

CELLS = ["GatedDeltaNet", "LSTM"]
INTRINSICS = ["none", "rnd", "e3b_rand", "e3b_obs", "e3b_innov", "noveld", "icm"]
TASKS = [
    "autoencode_medium",
    "countrecall_medium",
    "battleship_easy",
    "repeat_previous_medium",
]
DEFAULT_SEEDS = [0, 1]

# Per-cell default kwargs. Keep in sync with CELL_REGISTRY defaults.
CELL_DEFAULTS = {
    "GatedDeltaNet": {"assoc_size": 64},
    "LSTM":          {},
    "GRU":           {},
    "DTHLMU":        {"memory_size": 32, "theta": 100.0, "n_scales": 3,
                       "scale_factor": 2.0, "assoc_size": 64,
                       "hebbian_mode": "gated_delta_eps"},
    "SelectiveLMU":  {"memory_size": 32, "theta": 100.0, "gate_type": "softsign_sum",
                       "n_scales": 3, "scale_factor": 2.0, "readout_skip_scale": 0.1},
}


def _base_config_path(task: str) -> Path:
    """The DTH-LMU 15M configs already encode the right PPO HPs per task."""
    return BENCHMARK_DIR / f"dth_lmu_{task}_15M.yaml"


def write_variant_config(
    task: str,
    cell_name: str,
    intrinsic_name: str,
    timesteps: int | None = None,
    n_epochs: int | None = None,
    lambda_intrinsic: float = 0.1,
) -> Path:
    """Read the task config, swap cell + add intrinsic, write a per-variant yaml."""
    base = _base_config_path(task)
    if not base.exists():
        raise FileNotFoundError(f"Base config not found: {base}")
    with open(base) as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    # Replace the cell
    cfg["cell"] = {
        "name": cell_name,
        "kwargs": copy.deepcopy(CELL_DEFAULTS.get(cell_name, {})),
    }
    # Add intrinsic
    cfg["intrinsic"] = intrinsic_name
    if intrinsic_name != "none":
        cfg["lambda_intrinsic"] = float(lambda_intrinsic)

    if timesteps is not None:
        cfg["total_timesteps"] = int(timesteps)
    if n_epochs is not None:
        cfg["n_epochs"] = int(n_epochs)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{int(cfg['total_timesteps']) // 1_000_000}M"
    fname = f"{cell_name}_{task}_{intrinsic_name}_{suffix}.yaml"
    out = GENERATED_DIR / fname
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
    log_dir = GENERATED_DIR / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label.replace(' ', '_').replace('/', '_')}.log"
    with open(log_path, "w") as logf:
        res = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return label, res.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="+", default=CELLS, metavar="CELL",
                        help=f"Cell classes (default {CELLS}).")
    parser.add_argument("--intrinsics", nargs="+", default=INTRINSICS, metavar="NAME",
                        help=f"Intrinsic modules (default {INTRINSICS}).")
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS, metavar="TASK",
                        help=f"POPGym tasks (default {TASKS}).")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Seeds per (cell, intrinsic, task).")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total_timesteps (e.g. 2_000_000 for a fast pass).")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Override PPO n_epochs.")
    parser.add_argument("--lambda-intrinsic", type=float, default=0.1,
                        help="Scale for intrinsic reward (default 0.1).")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Concurrent train.py workers.")
    parser.add_argument("--runs-dir", default="runs/cell_intrinsic_matrix",
                        help="Root for run outputs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without launching.")
    args = parser.parse_args()

    jobs: list[tuple[str, list[str]]] = []
    for task in args.tasks:
        for cell in args.cells:
            for intrinsic in args.intrinsics:
                cfg_path = write_variant_config(
                    task, cell, intrinsic,
                    timesteps=args.timesteps,
                    n_epochs=args.n_epochs,
                    lambda_intrinsic=args.lambda_intrinsic,
                )
                for seed in args.seeds:
                    label = f"task={task} cell={cell} intrinsic={intrinsic} seed={seed}"
                    cmd   = build_cmd(cfg_path, seed, args.runs_dir)
                    jobs.append((label, cmd))

    total = len(jobs)
    print(f"Plan: {len(args.cells)} cells × {len(args.intrinsics)} intrinsics × "
          f"{len(args.tasks)} tasks × {len(args.seeds)} seeds = {total} runs")
    print(f"  cells:       {args.cells}")
    print(f"  intrinsics:  {args.intrinsics}")
    print(f"  tasks:       {args.tasks}")
    print(f"  seeds:       {args.seeds}")
    print(f"  timesteps:   {args.timesteps or '(per-config default)'}")
    print(f"  n_epochs:    {args.n_epochs or '(per-config default)'}")
    print(f"  λ_intrinsic: {args.lambda_intrinsic}")
    print(f"  parallel:    {args.parallel}")
    print(f"  runs_dir:    {args.runs_dir}\n")

    if args.dry_run:
        for i, (label, cmd) in enumerate(jobs, 1):
            print(f"[{i}/{total}] {label}\n    {' '.join(cmd)}")
        return

    t0 = time.perf_counter()
    if args.parallel <= 1:
        for i, (label, cmd) in enumerate(jobs, 1):
            print(f"\n[{i}/{total}] {label}")
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print(f"  ERROR: exit code {rc}")
    else:
        print(f"Launching up to {args.parallel} concurrent runs. "
              f"Per-run logs in {GENERATED_DIR / '_logs'}/\n")
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = {ex.submit(_run_one, lbl, cmd): lbl for lbl, cmd in jobs}
            done = 0
            for fut in as_completed(futures):
                lbl, rc = fut.result()
                done += 1
                tag = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"[{done}/{total}] {tag}  {lbl}")

    elapsed = time.perf_counter() - t0
    print(f"\nAll runs complete in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
