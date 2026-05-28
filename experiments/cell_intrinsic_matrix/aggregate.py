"""Aggregate the cell × intrinsic S13 sweep into the substitutability grid.

Walks a runs directory, reads each run's snapshot `config.yaml` (cell name,
intrinsic, env) and `eval/evaluations.npz` (eval reward vs timesteps), and
produces:

  - a long-form CSV (one row per run): cell, intrinsic, seed, final_rew,
    best_rew, last_timestep
  - a pivoted cell × intrinsic grid (mean over seeds) of the final eval reward,
    printed to stdout and written to <out>/grid_final.csv

"Final" = mean eval reward over the last `--final-window` eval points (default
5), which is steadier than the single last point on a high-variance task like
MemoryS13. "Best" = max over eval points (the EvalCallback-style best model).

The substitutability read: scan a row (fixed cell) across intrinsic columns —
how much does exploration lift a weak cell? Scan a column (fixed intrinsic)
down cells — how much does memory strength matter at fixed exploration? Iso-
performance cells across the off-diagonal are the substitutability evidence.

Usage:
    python -m experiments.cell_intrinsic_matrix.aggregate \\
        --runs-dir runs/cell_intrinsic_matrix --out experiments/cell_intrinsic_matrix/_results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


def _final_best(eval_file: Path, final_window: int) -> tuple[float, float, int, int]:
    """Return (final_mean, best_mean, last_timestep, n_eval_points)."""
    z = np.load(eval_file)
    # results: (n_evals, n_episodes) → mean over episodes per eval point.
    per_eval = z["results"].mean(axis=1)
    ts = z["timesteps"].astype(int)
    w = min(final_window, len(per_eval))
    final_mean = float(per_eval[-w:].mean())
    best_mean = float(per_eval.max())
    return final_mean, best_mean, int(ts[-1]), len(per_eval)


def collect(runs_dir: Path, final_window: int) -> list[dict]:
    """One record per run that has both a config snapshot and an eval file."""
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
        except Exception as e:  # corrupt/empty npz mid-run
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
        for r in records:
            w.writerow(r)


def build_grid(records: list[dict], metric: str) -> tuple[list[str], list[str], dict]:
    """Return (cells, intrinsics, {(cell,intr): (mean, std, n)}) over seeds."""
    cells = sorted({r["cell"] for r in records})
    intrinsics = sorted({r["intrinsic"] for r in records})
    grid: dict[tuple[str, str], tuple[float, float, int]] = {}
    for c in cells:
        for it in intrinsics:
            vals = [r[metric] for r in records if r["cell"] == c and r["intrinsic"] == it]
            if vals:
                grid[(c, it)] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    return cells, intrinsics, grid


def print_grid(cells, intrinsics, grid, metric: str) -> None:
    colw = 14
    print(f"\n=== {metric}: cell × intrinsic (mean ± std over seeds) ===")
    header = f"{'cell':16s}" + "".join(f"{it:>{colw}s}" for it in intrinsics)
    print(header)
    print("-" * len(header))
    for c in cells:
        row = f"{c:16s}"
        for it in intrinsics:
            if (c, it) in grid:
                m, s, n = grid[(c, it)]
                row += f"{m:>7.2f}±{s:<5.2f}"[:colw].rjust(colw)
            else:
                row += f"{'·':>{colw}s}"
        print(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs/cell_intrinsic_matrix")
    p.add_argument("--out", default="experiments/cell_intrinsic_matrix/_results")
    p.add_argument("--final-window", type=int, default=5)
    p.add_argument("--metric", default="final_rew", choices=["final_rew", "best_rew"])
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out)
    records = collect(runs_dir, args.final_window)
    if not records:
        print(f"No completed runs with eval files under {runs_dir}.")
        return

    write_long_csv(records, out_dir / "runs_long.csv")
    print(f"{len(records)} runs aggregated → {out_dir / 'runs_long.csv'}")

    cells, intrinsics, grid = build_grid(records, args.metric)
    print_grid(cells, intrinsics, grid, args.metric)

    # Pivot CSV (mean only).
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
