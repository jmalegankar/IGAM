"""Export wandb run histories for the S13 study into CSVs for offline analysis.

Pulls every run in the project and writes:
  _wandb_export/summary.csv          one row/run: cell, intrinsic, seed, final &
                                     max eval/mean_reward, step count, state
  _wandb_export/curves/<run>.csv     full eval curve (global_step, eval/mean_reward,
                                     eval/mean_ep_length, + any intrinsic/* metrics)

This is the analysis-friendly alternative to downloading raw TensorBoard event
files (which are protobuf and need parsing). If you specifically want the .tfevents
files, set DOWNLOAD_TB=1 and they land under _wandb_export/tb/<run>/.

Usage:
    .venv/bin/python experiments/s13_baseline/export_wandb.py
    DOWNLOAD_TB=1 .venv/bin/python experiments/s13_baseline/export_wandb.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import wandb

PROJECT = os.environ.get("WANDB_PROJECT_PATH", "jai-malegaonkar/memrl-s13-baseline")
OUT = Path(__file__).resolve().parent / "_wandb_export"
DOWNLOAD_TB = os.environ.get("DOWNLOAD_TB", "0") == "1"

CURVE_BASE = ["eval/mean_reward", "eval/mean_ep_length"]


def main() -> None:
    (OUT / "curves").mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    runs = list(api.runs(PROJECT))
    print(f"{PROJECT}: {len(runs)} runs")

    rows = []
    for i, r in enumerate(runs, 1):
        cfg, summ = r.config, r.summary
        cell = cfg.get("cell", {})
        cellname = cell.get("name") if isinstance(cell, dict) else cell

        # Eval curve. With sync_tensorboard each scalar is logged on its OWN
        # history row (its own _step), so scan_history(keys=[A,B]) — which only
        # returns rows where BOTH are present — comes back empty. Pull the eval
        # reward by itself; each row then carries _step + the value.
        try:
            hist = pd.DataFrame(r.scan_history(keys=["eval/mean_reward"]))
        except Exception:
            hist = pd.DataFrame()
        if not hist.empty and "_step" in hist:
            hist = hist.sort_values("_step")
        if not hist.empty:
            hist.to_csv(OUT / "curves" / f"{r.name}__{r.id}.csv", index=False)

        ev = hist["eval/mean_reward"].dropna() if "eval/mean_reward" in hist else pd.Series([], dtype=float)
        rows.append(
            dict(
                run=r.name,
                run_id=r.id,
                state=r.state,
                cell=cellname,
                intrinsic=cfg.get("intrinsic"),
                seed=cfg.get("seed"),
                lr=cfg.get("lr"),
                lambda_intrinsic=cfg.get("lambda_intrinsic"),
                final_step=summ.get("global_step"),
                final_eval=summ.get("eval/mean_reward"),
                max_eval=float(ev.max()) if len(ev) else None,
                last5_eval=float(ev.tail(5).mean()) if len(ev) else None,
                n_eval_pts=int(len(ev)),
            )
        )

        if DOWNLOAD_TB:
            tb_dir = OUT / "tb" / r.name
            tb_dir.mkdir(parents=True, exist_ok=True)
            for f in r.files():
                if "tfevents" in f.name:
                    f.download(root=str(tb_dir), replace=True)

        if i % 20 == 0:
            print(f"  ...{i}/{len(runs)}")

    df = pd.DataFrame(rows).sort_values(
        ["intrinsic", "cell", "seed"], na_position="last"
    )
    df.to_csv(OUT / "summary.csv", index=False)
    print(f"\nwrote {OUT/'summary.csv'} ({len(df)} runs) + {len(df)} curve CSVs")
    if DOWNLOAD_TB:
        print(f"raw .tfevents under {OUT/'tb'}/")


if __name__ == "__main__":
    main()
