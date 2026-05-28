"""Aggregate the S13 baseline runs into a cell × (seed-averaged) table.

Walks a runs directory, reads each run's snapshot `config.yaml` (cell, intrinsic,
env) and `eval/evaluations.npz` (eval reward vs steps), and writes:

  - `_results/runs_long.csv`  — one row per run: cell, intrinsic, seed,
    final_rew, best_rew, last_timestep
  - `_results/grid_final.csv` — cell × intrinsic grid (mean over seeds) of the
    final eval reward (also printed to stdout)

"Final" = mean eval reward over the last `--final-window` eval points (default 5),
steadier than the single last point on high-variance MemoryS13. "Best" = max over
eval points. For this batch every run is intrinsic=none, so the grid is one
column — but the script also handles the later exploration sweep unchanged.

Usage:
    python experiments/s13_baseline/aggregate.py --runs-dir runs/s13_baseline
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


def _final_best(eval_file: Path, final_window: int) -> tuple[float, float, int, int]:
    z = np.load(eval_file)
    per_eval = z["results"].mean(axis=1)        # (n_evals, n_eps) → per-eval mean
    ts = z["timesteps"].astype(int)
    w = min(final_window, len(per_eval))
    return float(per_eval[-w:].mean()), float(per_eval.max()), int(ts[-1]), len(per_eval)


def collect(runs_dir: Path, final_window: int) -> list[dict]:
    records: list[dict] = []
    for cfg_path in sorted(runs_dir.rglob("config.yaml")):
        run_dir = cfg_path.parent
        eval_file = run_dir / "eval" / "evaluations.npz"
        if not eval_file.exists():
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        try:
            final_mean, best_mean, last_ts, n_pts = _final_best(eval_file, final_window)
        except Exception as e:
            print(f"[skip] {run_dir}: {type(e).__name__}: {e}")
            continue
        records.append({
            "cell": cfg.get("cell", {}).get("name", "?"),
            "intrinsic": cfg.get("intrinsic", "none"),
            "env": cfg.get("env_name", "?"),
            "seed": cfg.get("seed", -1),
            "final_rew": final_mean,
            "best_rew": best_mean,
            "last_timestep": last_ts,
            "n_eval_points": n_pts,
            "run_dir": str(run_dir),
        })
    return records


def write_long_csv(records: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["cell", "intrinsic", "env", "seed", "final_rew", "best_rew",
            "last_timestep", "n_eval_points", "run_dir"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(records)


def build_grid(records, metric):
    cells = sorted({r["cell"] for r in records})
    intrinsics = sorted({r["intrinsic"] for r in records})
    grid = {}
    for c in cells:
        for it in intrinsics:
            vals = [r[metric] for r in records if r["cell"] == c and r["intrinsic"] == it]
            if vals:
                grid[(c, it)] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    return cells, intrinsics, grid


def print_grid(cells, intrinsics, grid, metric):
    colw = 16
    print(f"\n=== {metric}: cell × intrinsic (mean ± std over seeds, n) ===")
    header = f"{'cell':16s}" + "".join(f"{it:>{colw}s}" for it in intrinsics)
    print(header); print("-" * len(header))
    for c in cells:
        row = f"{c:16s}"
        for it in intrinsics:
            if (c, it) in grid:
                m, s, n = grid[(c, it)]
                row += f"{m:.2f}±{s:.2f}(n{n})".rjust(colw)
            else:
                row += f"{'·':>{colw}s}"
        print(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs/s13_baseline")
    p.add_argument("--out", default="experiments/s13_baseline/_results")
    p.add_argument("--final-window", type=int, default=5)
    p.add_argument("--metric", default="final_rew", choices=["final_rew", "best_rew"])
    args = p.parse_args()

    records = collect(Path(args.runs_dir), args.final_window)
    if not records:
        print(f"No completed runs with eval files under {args.runs_dir}.")
        return

    out_dir = Path(args.out)
    write_long_csv(records, out_dir / "runs_long.csv")
    print(f"{len(records)} runs aggregated → {out_dir / 'runs_long.csv'}")

    cells, intrinsics, grid = build_grid(records, args.metric)
    print_grid(cells, intrinsics, grid, args.metric)

    with open(out_dir / "grid_final.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell"] + intrinsics)
        for c in cells:
            w.writerow([c] + [
                (f"{grid[(c, it)][0]:.4f}" if (c, it) in grid else "")
                for it in intrinsics
            ])
    print(f"grid → {out_dir / 'grid_final.csv'}")


if __name__ == "__main__":
    main()
