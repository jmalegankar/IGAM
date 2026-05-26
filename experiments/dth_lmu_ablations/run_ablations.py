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

Pause & resume:
    - Drop a STOP file (default: <repo>/STOP) and every running worker will
      save its checkpoint and exit cleanly. The runner then quits too.
    - Re-run the same `python -m experiments.dth_lmu_ablations.run_ablations`
      command and it'll auto-detect: skip runs with DONE markers, resume
      the rest from their latest.pt.
    - Ctrl+C the runner has the same effect (workers receive SIGINT and
      flush a final checkpoint before exiting).
    - --no-resume forces a fresh start (ignores all latest.pt files).

CPU threading:
    The runner sets MEMRL_TORCH_THREADS = max(1, 16 // parallel) per worker
    so N concurrent runs on the 16-thread i9-11900K don't oversubscribe.
    Override with --torch-threads if you want to tune.

Usage:
    # Full grid (4 tasks × 4 modes × 2 seeds = 32 runs)
    python -m experiments.dth_lmu_ablations.run_ablations

    # Recommended starting command for the 3070 Ti (8 GB).
    # 18 runs × 5M steps × parallel=2 (safer than 3 with bigger Hebbian M):
    python -m experiments.dth_lmu_ablations.run_ablations \
        --tasks autoencode_medium battleship_easy countrecall_medium \
        --modes additive delta gated_delta_eps \
        --seeds 0 1 \
        --timesteps 5_000_000 \
        --n-epochs 4 \
        --parallel 2

    # Pause everything to play games:
    echo "" > STOP
    # (workers flush latest.pt and exit; delete STOP and re-run to resume)
    rm STOP

    # Print commands without launching
    python -m experiments.dth_lmu_ablations.run_ablations --dry-run
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO_ROOT     = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "phase_a" / "ablation"
GENERATED_DIR = Path(__file__).parent / "_generated"

# So `from memrl.utils ...` works when this script is run as -m.
sys.path.insert(0, str(REPO_ROOT))
from memrl.utils.checkpoint import find_resume_run_dir, is_run_completed  # noqa: E402

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
    ent_coef: float | None = None,
    lambda_intrinsic: float | None = None,
) -> Path:
    """Read the base task config, override hebbian_mode + optional knobs,
    write a per-variant yaml.  Returns the path of the generated file.

    Non-default knobs (ent_coef, lambda_intrinsic) get reflected in the
    filename suffix so sweeps don't overwrite each other's run dirs.
    """
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
    if ent_coef is not None:
        cfg["ent_coef"] = float(ent_coef)
    if lambda_intrinsic is not None:
        cfg["lambda_intrinsic"] = float(lambda_intrinsic)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    parts = [f"dth_lmu_{task}_{mode}", f"{int(cfg['total_timesteps']) // 1_000_000}M"]
    if ent_coef is not None:
        parts.append(f"ent{ent_coef:g}".replace(".", "p"))         # 0.05 → "ent0p05"
    if lambda_intrinsic is not None:
        parts.append(f"int{lambda_intrinsic:g}".replace(".", "p")) # 0.1  → "int0p1"
    out = GENERATED_DIR / ("_".join(parts) + ".yaml")
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def build_cmd(
    config_path: Path,
    seed: int,
    runs_dir: str,
    resume_dir: Path | None = None,
    save_freq_steps: int | None = None,
    stop_file: Path | None = None,
) -> list[str]:
    cmd = [
        sys.executable, "-m", "train",
        "--config", str(config_path),
        "--seed", str(seed),
        "--runs-dir", runs_dir,
    ]
    if resume_dir is not None:
        cmd += ["--resume-from", str(resume_dir)]
    if save_freq_steps is not None:
        cmd += ["--save-freq-steps", str(save_freq_steps)]
    if stop_file is not None:
        cmd += ["--stop-file", str(stop_file)]
    return cmd


def _run_one(
    label: str,
    cmd: list[str],
    env_overrides: dict[str, str] | None = None,
) -> tuple[str, int]:
    """Launch one training run; capture stdout/stderr to a per-run log file
    so parallel runs don't interleave on the terminal."""
    log_dir = GENERATED_DIR / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label.replace(' ', '_').replace('/', '_')}.log"
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    with open(log_path, "w") as logf:
        res = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
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
        "--ent-coef", type=float, default=None,
        help="Override PPO entropy coefficient (default 0.01 in the base configs; "
             "0.05 forces more exploration when policy collapses to random behavior).",
    )
    parser.add_argument(
        "--lambda-intrinsic", type=float, default=None,
        help="Intrinsic-reward scale λ (default 0 = off). The cell's `eps_mem` "
             "side output is RND-normalised and added to extrinsic reward as "
             "λ · (eps_mem / σ_running). Try 0.1 to start.",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Concurrent runs (3070 Ti: 2-3 fits in 8GB VRAM; CPU env-stepping usually bottlenecks first)",
    )
    parser.add_argument(
        "--runs-dir", default="runs/dth_lmu_ablations",
        help="Root directory for run outputs",
    )
    parser.add_argument(
        "--save-freq-steps", type=int, default=50_000,
        help="Forwarded to train.py: env-step interval between resumable saves.",
    )
    parser.add_argument(
        "--stop-file", default=str(REPO_ROOT / "STOP"),
        help="Global pause-file path. Every worker polls this; touching it "
             "(`echo > STOP`) makes all running workers save and exit cleanly. "
             "Re-run the same command to resume. Default: <repo>/STOP.",
    )
    parser.add_argument(
        "--torch-threads", type=int, default=None,
        help="Override per-worker MEMRL_TORCH_THREADS. Default: max(1, 16 // parallel) "
             "so the 16-thread CPU is shared without oversubscription.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Force-start fresh runs (ignore any existing latest.pt). "
             "Default: auto-resume incomplete runs and skip completed ones.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    # Decide per-worker torch thread budget. On the i9-11900K (8 cores / 16
    # threads), three parallel runs each capped at 4 intra-op threads leaves
    # ~4 logical cores free for env stepping and the OS.
    if args.torch_threads is not None:
        per_worker_threads = args.torch_threads
    else:
        per_worker_threads = max(1, 16 // max(args.parallel, 1))

    stop_file_path = Path(args.stop_file).resolve() if args.stop_file else None
    # Stale STOP from a previous pause would prevent us starting; clear it.
    if stop_file_path is not None and stop_file_path.exists() and not args.dry_run:
        try:
            stop_file_path.unlink()
            print(f"  cleared stale stop-file: {stop_file_path}")
        except OSError:
            pass

    # Build the full job list up front, applying resume / skip logic.
    jobs: list[tuple[str, list[str]]] = []
    skipped: list[str] = []
    resumed: list[str] = []
    for task in args.tasks:
        for mode in args.modes:
            config = write_variant_config(task, mode,
                                          timesteps=args.timesteps,
                                          n_epochs=args.n_epochs,
                                          ent_coef=args.ent_coef,
                                          lambda_intrinsic=args.lambda_intrinsic)
            for seed in args.seeds:
                label = f"task={task} mode={mode} seed={seed}"

                # Auto-skip already-completed runs (DONE marker present).
                if not args.no_resume and is_run_completed(
                    args.runs_dir, config.stem, "DTHLMU", seed,
                ):
                    skipped.append(label)
                    continue

                # Auto-resume: reuse the most recent run dir with latest.pt
                # but no DONE.
                resume_dir = None
                if not args.no_resume:
                    resume_dir = find_resume_run_dir(
                        args.runs_dir, config.stem, "DTHLMU", seed,
                    )
                    if resume_dir is not None:
                        resumed.append(f"{label} -> {resume_dir.name}")

                cmd = build_cmd(
                    config, seed, args.runs_dir,
                    resume_dir=resume_dir,
                    save_freq_steps=args.save_freq_steps,
                    stop_file=stop_file_path,
                )
                jobs.append((label, cmd))

    total = len(jobs)
    grid_total = len(args.tasks) * len(args.modes) * len(args.seeds)
    print(f"Plan: {len(args.tasks)} tasks × {len(args.modes)} modes × "
          f"{len(args.seeds)} seeds = {grid_total} runs total "
          f"({total} to launch, {len(skipped)} done, {len(resumed)} to resume)")
    print(f"  tasks:        {args.tasks}")
    print(f"  modes:        {args.modes}")
    print(f"  seeds:        {args.seeds}")
    print(f"  timesteps:    {args.timesteps or '(base config)'}")
    print(f"  n_epochs:     {args.n_epochs or '(base config)'}")
    print(f"  parallel:     {args.parallel}  (torch threads/worker: {per_worker_threads})")
    print(f"  runs:         {args.runs_dir}")
    print(f"  stop-file:    {stop_file_path or '(none)'}  "
          f"— `echo > {stop_file_path.name if stop_file_path else 'STOP'}` "
          f"to pause all workers")
    print(f"  save-freq:    {args.save_freq_steps:,} env steps")
    if skipped:
        print(f"  skipping ({len(skipped)} already DONE):")
        for s in skipped:
            print(f"    - {s}")
    if resumed:
        print(f"  resuming ({len(resumed)}):")
        for r in resumed:
            print(f"    - {r}")
    print()

    if args.dry_run:
        for i, (label, cmd) in enumerate(jobs, 1):
            print(f"[{i}/{total}] {label}\n    {' '.join(cmd)}")
        return

    if total == 0:
        print("Nothing to launch — every run is already DONE.")
        return

    # Per-worker env overrides: cap torch threads.
    env_overrides = {"MEMRL_TORCH_THREADS": str(per_worker_threads)}

    if args.parallel <= 1:
        for i, (label, cmd) in enumerate(jobs, 1):
            print(f"\n[{i}/{total}] {label}")
            run_env = os.environ.copy()
            run_env.update(env_overrides)
            res = subprocess.run(cmd, env=run_env)
            if res.returncode != 0:
                print(f"  ERROR: exit code {res.returncode}")
            # If a global stop-file appeared, don't keep launching the
            # remaining jobs — the user wants the box free.
            if stop_file_path is not None and stop_file_path.exists():
                print(f"  stop-file {stop_file_path} present — halting runner.")
                print(f"  Re-run the same command to resume.")
                return
    else:
        print(f"Launching up to {args.parallel} concurrent runs. "
              f"Per-run logs in {GENERATED_DIR / '_logs'}/\n")
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = {ex.submit(_run_one, label, cmd, env_overrides): label
                       for label, cmd in jobs}
            done = 0
            for fut in as_completed(futures):
                label, rc = fut.result()
                done += 1
                status = "OK" if rc == 0 else f"FAIL (rc={rc})"
                print(f"[{done}/{total}] {status}  {label}")

    if stop_file_path is not None and stop_file_path.exists():
        print(f"\nPaused: stop-file {stop_file_path} is set. "
              f"Delete it and re-run to resume.")
    else:
        print("\nAll ablation runs complete.")


if __name__ == "__main__":
    main()
