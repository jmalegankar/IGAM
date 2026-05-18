"""Cross-platform launcher for the full POPGym 15M suite.

Runs LMU + GatedLMU pairs on 4 POPGym envs at 15M steps each. Hyperparams
per env are calibrated from the POPGym/SHM literature (see results.md).

Modes:
    --sequential  : one process at a time, full CPU/GPU to each (safer
                    when contention hurts more than parallelism helps)
    --parallel    : 2 processes at a time (a single env-pair concurrent)
    --all-parallel: 8 processes at once (only if you have heavy compute)
    --env <name>  : run only one env-pair (e.g. autoencode_medium)

Default: --parallel (env-pair concurrent, env-pairs sequential).

Output layout:
    runs/gate/F1/<env>_15M/
        <config_stem>/<cell>/seed_0_<ts>/
        <config_stem>.log
        _suite.log
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
F1_DIR = REPO_ROOT / "runs" / "gate" / "F1"

# Each entry: (env_short, lmu_config_stem, gated_config_stem)
ENV_PAIRS = [
    ("repeat_previous_medium_15M",
     "lmu_medium_tuned_15M",
     "gated_lmu_medium_tuned_15M"),
    ("autoencode_medium_15M",
     "lmu_autoencode_medium_15M",
     "gated_lmu_autoencode_medium_15M"),
    ("countrecall_medium_15M",
     "lmu_countrecall_medium_15M",
     "gated_lmu_countrecall_medium_15M"),
    ("battleship_easy_15M",
     "lmu_battleship_easy_15M",
     "gated_lmu_battleship_easy_15M"),
]


def log(msg: str, file=None) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    if file is not None:
        with file.open("a") as f:
            f.write(line + "\n")


def device_check(suite_log: Path) -> None:
    cmd = [
        sys.executable, "-c",
        "import torch; "
        "print(f'torch={torch.__version__}'); "
        "print(f'cuda_available={torch.cuda.is_available()}'); "
        "print('device_name=' + (torch.cuda.get_device_name(0) "
        "  if torch.cuda.is_available() else 'CPU'));"
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    for ln in out.strip().splitlines():
        log("  " + ln, file=suite_log)


def launch(stem: str, run_dir: Path, suite_log: Path) -> subprocess.Popen:
    """Spawn one train.py process. Returns Popen handle."""
    cfg_path = REPO_ROOT / "benchmarks" / "phase_a" / "ablation" / f"{stem}.yaml"
    log_path = run_dir / f"{stem}.log"
    cmd = [
        sys.executable, str(REPO_ROOT / "train.py"),
        "--config", str(cfg_path),
        "--seed", "0",
        "--runs-dir", str(run_dir),
    ]
    log(f"  starting {stem}", file=suite_log)
    log_file = log_path.open("w")
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)


def run_pair(env_short: str, lmu_stem: str, gated_stem: str,
             parallel: bool, suite_log: Path) -> None:
    """Run one env-pair (LMU + GatedLMU). Optionally in parallel."""
    run_dir = F1_DIR / env_short
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"==== {env_short} ({'parallel' if parallel else 'sequential'}) ====",
        file=suite_log)

    if parallel:
        procs = [launch(lmu_stem, run_dir, suite_log),
                 launch(gated_stem, run_dir, suite_log)]
        for stem, p in zip([lmu_stem, gated_stem], procs):
            rc = p.wait()
            log(f"  exited {rc} : {stem}", file=suite_log)
    else:
        for stem in (lmu_stem, gated_stem):
            p = launch(stem, run_dir, suite_log)
            rc = p.wait()
            log(f"  exited {rc} : {stem}", file=suite_log)


def run_all_parallel(suite_log: Path) -> None:
    """All 8 arms at once. Heavy contention; only do this if you have lots of cores."""
    log("==== ALL 8 ARMS PARALLEL (heavy contention) ====", file=suite_log)
    procs = []
    names = []
    for env_short, lmu_stem, gated_stem in ENV_PAIRS:
        run_dir = F1_DIR / env_short
        run_dir.mkdir(parents=True, exist_ok=True)
        for stem in (lmu_stem, gated_stem):
            procs.append(launch(stem, run_dir, suite_log))
            names.append(stem)
    for stem, p in zip(names, procs):
        rc = p.wait()
        log(f"  exited {rc} : {stem}", file=suite_log)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sequential", action="store_true",
                      help="one process at a time (8 total, full resource)")
    mode.add_argument("--parallel", action="store_true",
                      help="2 processes concurrent (env-pair), env-pairs serial (default)")
    mode.add_argument("--all-parallel", action="store_true",
                      help="8 processes at once (heavy contention)")
    parser.add_argument("--env", default=None,
                        help="run only this env (e.g. autoencode_medium_15M)")
    args = parser.parse_args()

    # Default is --parallel if no mode flag given
    if not (args.sequential or args.parallel or args.all_parallel):
        args.parallel = True

    F1_DIR.mkdir(parents=True, exist_ok=True)
    suite_log = F1_DIR / "_15M_suite.log"
    suite_log.touch(exist_ok=True)

    log("=" * 80, file=suite_log)
    log(f"POPGym 15M suite launcher", file=suite_log)
    log(f"  mode: {'sequential' if args.sequential else 'all-parallel' if args.all_parallel else 'parallel (env-pair concurrent)'}",
        file=suite_log)
    log(f"  env filter: {args.env or 'all 4 envs'}", file=suite_log)
    device_check(suite_log)
    log("=" * 80, file=suite_log)

    pairs = ENV_PAIRS
    if args.env:
        pairs = [p for p in ENV_PAIRS if p[0] == args.env]
        if not pairs:
            log(f"ERROR: unknown env {args.env!r}. valid: {[p[0] for p in ENV_PAIRS]}",
                file=suite_log)
            return 1

    if args.all_parallel:
        run_all_parallel(suite_log)
    else:
        for env_short, lmu_stem, gated_stem in pairs:
            run_pair(env_short, lmu_stem, gated_stem,
                     parallel=args.parallel, suite_log=suite_log)

    log("=" * 80, file=suite_log)
    log("POPGym 15M suite COMPLETE.", file=suite_log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
