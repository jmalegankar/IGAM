"""Plot eval-reward learning curves for the canonical-baselines comparison.

Reads each run's `eval/evaluations.npz` and plots mean eval reward vs
timesteps, one line per cell, with a legend.

Usage:
    python scripts/plot_baselines.py [--out plots/baselines.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Display name, run-directory path (relative to repo root).
RUNS = [
    ("LMU",                 "runs/lmu_medium/LMU/seed_0_20260516_131752"),
    ("LSTM",                "runs/lstm_medium/LSTM/seed_0_20260516_131752"),
    ("GRU",                 "runs/gru_medium/GRU/seed_0_20260516_161407"),
    ("S4D",                 "runs/s4d_medium/S4D/seed_0_20260516_161407"),
    ("GatedLMU (tuned)",    "runs/gated_lmu_medium_tuned/GatedLMU/seed_0_20260516_175919"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="plots/baselines_repeat_prev_medium.png")
    parser.add_argument("--smooth", type=int, default=5,
                        help="moving-average window in eval points (default 5)")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, run_dir in RUNS:
        eval_file = Path(run_dir) / "eval" / "evaluations.npz"
        if not eval_file.exists():
            print(f"[skip] {label}: no eval file at {eval_file}")
            continue
        z = np.load(eval_file)
        ts = z["timesteps"].astype(float) / 1e6           # to millions
        # results shape (n_evals, n_episodes); take mean over episodes
        rew = z["results"].mean(axis=1)

        # Moving-average smoothing
        if args.smooth > 1 and len(rew) > args.smooth:
            kernel = np.ones(args.smooth) / args.smooth
            rew_smooth = np.convolve(rew, kernel, mode="valid")
            ts_smooth = ts[args.smooth - 1:]
        else:
            rew_smooth = rew
            ts_smooth = ts

        ax.plot(ts_smooth, rew_smooth, label=label, linewidth=2)

    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("Eval episode reward")
    ax.set_title("POPGym RepeatPrevious-Medium — canonical recurrent baselines vs GatedLMU")
    ax.axhline(0.0, color="grey", linestyle=":", alpha=0.5, linewidth=1)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path.resolve()}")


if __name__ == "__main__":
    main()
