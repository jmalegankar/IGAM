"""Cross-platform launcher for the RP-Medium 15M three-way.

Equivalent to scripts/run_rp_medium_15M.sh but works on Windows, macOS, and Linux.

Usage:
    python scripts/run_rp_medium_15M.py            # parallel (default)
    python scripts/run_rp_medium_15M.py --sequential

Output:
    runs/gate/F1/repeat_previous_medium_15M/
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
RUN_DIR = REPO_ROOT / "runs" / "gate" / "F1" / "repeat_previous_medium_15M"
SUITE_LOG = RUN_DIR / "_suite.log"

CONFIGS = [
    "lmu_medium_tuned_15M",
    "gated_lmu_medium_tuned_15M",
    "gated_lmu_medium_tuned_no_gate_15M",
]


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with SUITE_LOG.open("a") as f:
        f.write(line + "\n")


def device_check() -> None:
    """Log torch + CUDA availability."""
    cmd = [
        sys.executable, "-c",
        "import torch; "
        "print(f'torch={torch.__version__}'); "
        "print(f'cuda_available={torch.cuda.is_available()}'); "
        "print(f'device_count={torch.cuda.device_count()}'); "
        "print('device_name=' + (torch.cuda.get_device_name(0) "
        "  if torch.cuda.is_available() else 'CPU'));"
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    for ln in out.strip().splitlines():
        log("  " + ln)


def run_arm(config_stem: str, blocking: bool = False) -> subprocess.Popen | None:
    """Launch one train.py process; return Popen handle (or None if blocking)."""
    cfg_path = REPO_ROOT / "benchmarks" / "phase_a" / "ablation" / f"{config_stem}.yaml"
    log_path = RUN_DIR / f"{config_stem}.log"
    cmd = [
        sys.executable, str(REPO_ROOT / "train.py"),
        "--config", str(cfg_path),
        "--seed", "0",
        "--runs-dir", str(RUN_DIR),
    ]
    log(f"Starting {config_stem} → {log_path.name}")
    log_file = log_path.open("w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    if blocking:
        rc = proc.wait()
        log_file.close()
        log(f"  exited {rc} : {config_stem}")
        return None
    return proc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run arms one at a time instead of all parallel.",
    )
    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    SUITE_LOG.touch(exist_ok=True)

    log("=" * 70)
    log(f"RP-Medium 15M three-way ({'sequential' if args.sequential else 'parallel'})")
    log(f"  configs:  {CONFIGS}")
    log(f"  run dir:  {RUN_DIR}")
    device_check()
    log("=" * 70)

    if args.sequential:
        for stem in CONFIGS:
            run_arm(stem, blocking=True)
    else:
        procs = [run_arm(stem) for stem in CONFIGS]
        for stem, proc in zip(CONFIGS, procs):
            rc = proc.wait()
            log(f"  exited {rc} : {stem}")

    log("=" * 70)
    log("RP-Medium 15M suite complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
