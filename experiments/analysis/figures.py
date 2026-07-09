"""Publication figures for the memory×exploration paper. Regenerate as seeds land.

Emits (as vector PDF + PNG):
  fig1_interaction  — the cross-env cell×bonus sign flip (dumbbell): MPG amplifies, S13 equalizes.
                      This is the ANOVA made visual; F/p/η² go in the caption, not the plot.
  fig2_freeze       — MysteryPath penalty arm: none freezes (→0), e3b/noveld rescue, pbim frozen
                      (ρ=0); idle (no-op) fraction as the freeze signature.
  fig3_curves       — learning curves that kill "it catches up later": MPG penalty none(flat 0)
                      vs e3b(climbs); S13-v3 none(flat chance) vs e3b(late grok).
  fig4_axis         — I-5 operational axis: P(trigger under random policy) per env, exogenous vs
                      agent-contingent (results-free; grounds §axis).

Data: converged tail-mean success is pulled from wandb and cached per (project,density) CSV, so
reruns are instant; pass --refresh to re-pull as new seeds finish. Fig 4 is results-free constants
from `memrl.probes.axis_classifier`. Colorblind-safe (Okabe–Ito), dark text, no chartjunk.

Usage:
    python -m experiments.analysis.figures --which all --out-dir experiments/analysis/figs
    python -m experiments.analysis.figures --which interaction --refresh
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── style ───────────────────────────────────────────────────────────────────
INK = "#1a1a1a"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.facecolor": "white", "axes.facecolor": "white",
})
# Okabe–Ito colorblind-safe
C = {"none": "#666666", "e3b_idm": "#0072B2", "noveld": "#009E73", "pbim_e3b_idm": "#D55E00"}
LBL = {"none": "none", "e3b_idm": "E3B", "noveld": "NovelD", "pbim_e3b_idm": "PBIM"}
CELLS = ["GRU", "LSTM", "RetNet", "GatedDeltaNet", "Mamba2", "Memoryless"]
CELL_SHORT = {"GatedDeltaNet": "GDN", "Mamba2": "Mamba-2"}


# ── data (wandb pull + per-(project,density) CSV cache) ──────────────────────
def _cellname(cfg):
    c = cfg.get("cell")
    return c.get("name") if isinstance(c, dict) else c


def _tail(run, key, frac=0.2):
    try:
        h = run.history(keys=[key], samples=120, pandas=False)
        v = [x[key] for x in h if x.get(key) is not None]
    except Exception:
        v = []
    if not v:
        return None
    return float(np.mean(v[-max(3, int(len(v) * frac)):]))


def load_grid(entity, project, density, cache_dir, extra=None, min_step=15e6, refresh=False):
    """Tidy grid (cell,bonus,seed,success[,extra]) for one (project,density), cached to CSV."""
    cache = Path(cache_dir) / f"{project}__{density}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)
    import wandb
    api = wandb.Api()
    best = defaultdict(lambda: (-1, None))
    for r in api.runs(f"{entity}/{project}"):
        if r.state not in ("finished", "running"):
            continue
        c = r.config
        if c.get("wandb_group") != density:
            continue
        k = (_cellname(c), c.get("intrinsic"), c.get("seed"))
        gs = r.summary.get("global_step", 0) or 0
        if gs > best[k][0]:
            best[k] = (gs, r)
    rows = []
    for (cell, bonus, seed), (gs, r) in best.items():
        if gs < min_step or cell is None:
            continue
        s = _tail(r, "eval/success_rate")
        row = {"cell": cell, "bonus": bonus, "seed": seed, "success": s}
        if extra:
            row["extra"] = _tail(r, extra)
        if s is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def _mean(df, cell, bonus, col="success"):
    v = df[(df.cell == cell) & (df.bonus == bonus)][col].to_numpy(float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else np.nan


def pull_curves(entity, specs, cache_dir, refresh=False):
    """specs: list of (project, density, cell, bonus). Returns {label:(steps,succ)} downsampled."""
    cache = Path(cache_dir) / "curves.csv"
    if cache.exists() and not refresh:
        d = pd.read_csv(cache)
        return {lbl: (g["step"].to_numpy(), g["succ"].to_numpy())
                for lbl, g in d.groupby("label")}
    import wandb
    api = wandb.Api()
    out, rows = {}, []
    for project, density, cell, bonus in specs:
        best = (-1, None)
        for r in api.runs(f"{entity}/{project}"):
            c = r.config
            if (c.get("wandb_group") == density and _cellname(c) == cell
                    and c.get("intrinsic") == bonus and r.state in ("finished", "running")):
                gs = r.summary.get("global_step", 0) or 0
                if gs > best[0]:
                    best = (gs, r)
        if best[1] is None:
            continue
        try:
            h = best[1].history(keys=["eval/success_rate", "global_step"], samples=300, pandas=False)
            pts = [(x.get("global_step"), x["eval/success_rate"]) for x in h
                   if x.get("eval/success_rate") is not None and x.get("global_step") is not None]
        except Exception:
            pts = []
        if not pts:
            continue
        env = "mpg" if "mpg" in project else ("s13" if "s13" in project else project)
        lbl = f"{env}:{CELL_SHORT.get(cell, cell)}:{LBL[bonus]}"
        st = np.array([p[0] for p in pts]); sc = np.array([p[1] for p in pts])
        out[lbl] = (st, sc)
        rows += [{"label": lbl, "step": a, "succ": b} for a, b in zip(st, sc)]
    if rows:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(cache, index=False)
    return out


# ── figures ──────────────────────────────────────────────────────────────────
def fig_interaction(mpg, s13, bonus="e3b_idm"):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    order = CELLS[::-1]  # Memoryless at bottom
    for ax, df, title, sub in (
        (axes[0], mpg, "MysteryPath (sparse)", "bonus widens the gap → amplify"),
        (axes[1], s13, "MemoryS13 (view 3×3)", "bonus closes the gap → equalize")):
        for i, cell in enumerate(order):
            n, b = _mean(df, cell, "none"), _mean(df, cell, bonus)
            if np.isnan(n) or np.isnan(b):
                continue
            ax.plot([n, b], [i, i], color="#c3c2b7", lw=2, zorder=1, solid_capstyle="round")
            ax.scatter([n], [i], s=42, facecolor="white", edgecolor="#888780",
                       linewidth=1.6, zorder=2)
            ax.scatter([b], [i], s=52, color=C[bonus], zorder=3)
        ax.axvline(1.0, color="#e1e0d9", lw=1, ls=(0, (3, 3)), zorder=0)
        ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_xlabel("success rate")
        ax.set_title(title, loc="left", pad=12)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color="#52514e")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([CELL_SHORT.get(c, c) for c in order])
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#888780", mew=1.6, ms=7, label="none"),
         plt.Line2D([], [], marker="o", ls="", color=C[bonus], ms=8, label=f"+ {LBL[bonus]}")]
    axes[1].legend(handles=h, loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def fig_freeze(penalty, sparse):
    bonuses = ["none", "e3b_idm", "noveld", "pbim_e3b_idm"]
    fig, (ax, axi) = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    x = np.arange(len(CELLS)); w = 0.2
    for j, bon in enumerate(bonuses):
        vals = [_mean(penalty, c, bon) for c in CELLS]
        xs = x + (j - 1.5) * w
        ax.bar(xs, vals, w, color=C[bon], label=LBL[bon], edgecolor="white", linewidth=0.5)
        for xi, v in zip(xs, vals):
            if not np.isnan(v) and v < 0.05:     # label the frozen arms so 0-bars are legible
                ax.text(xi, 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#52514e")
    ref = [_mean(sparse, c, "none") for c in CELLS]
    ax.plot(x, ref, "k_", ms=14, mew=1.4, label="sparse-none (alive)")
    ax.set_ylim(0, 1.0); ax.set_ylabel("success rate")
    ax.set_title("MysteryPath penalty arm: none freezes, e3b/noveld rescue, PBIM stays frozen (ρ=0)",
                 loc="left", pad=8)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8.5)
    idle = [_mean(penalty, c, "none", "extra") for c in CELLS]
    axi.bar(x, idle, 0.5, color="#888780", edgecolor="white", linewidth=0.5)
    axi.set_ylim(0, 0.6); axi.set_ylabel("idle frac\n(penalty-none)", fontsize=8.5)
    axi.set_xticks(x); axi.set_xticklabels([CELL_SHORT.get(c, c) for c in CELLS])
    fig.tight_layout()
    return fig


def fig_curves(curves):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    panels = [("MysteryPath penalty (GatedDeltaNet)", "mpg", 0.0),
              ("MemoryS13 3×3 (LSTM)", "s13", 0.5)]
    for ax, (title, tag, chance) in zip(axes, panels):
        for lbl, (st, sc) in curves.items():
            if tag not in lbl.lower():
                continue
            bon = "e3b_idm" if "E3B" in lbl else "none"
            o = np.argsort(st)
            xs_ = np.array(st)[o] / 1e6
            ys_ = pd.Series(np.array(sc)[o]).rolling(
                max(3, len(sc) // 30), center=True, min_periods=1).mean().to_numpy()
            ax.plot(xs_, ys_, color=C[bon], lw=1.8, label=LBL[bon])
        ax.axhline(chance, color="#c3c2b7", lw=1, ls=(0, (3, 3)))
        ax.set_ylim(0, 1.02); ax.set_xlabel("env steps (M)"); ax.set_title(title, loc="left", pad=8)
        ax.legend(loc="upper left", frameon=False)
    axes[0].set_ylabel("success rate")
    fig.tight_layout()
    return fig


# I-5 axis constants (from memrl.probes.axis_classifier, uniform-random policy; results-free).
# class: exo (P→1), agent (P→0), mid (intermediate = S13's view-dependent residual acquisition)
AXIS = [
    ("MysteryPath", "goal (via hidden path)", 0.013, "agent"),
    ("MysteryPath", "fall / advance", 1.00, "exo"),
    ("S13 3×3", "cue enters view", 0.61, "mid"),
    ("S13 5×5", "cue enters view", 0.70, "mid"),
    ("S13 7×7", "cue enters view", 0.76, "mid"),
    ("S13", "reach match", 0.31, "mid"),
    ("Autoencode-Tiny", "reproduce all", 0.00, "agent"),
    ("Autoencode-Tiny", "token reveal", 1.00, "exo"),
]
AXIS_COL = {"agent": "#D55E00", "exo": "#0072B2", "mid": "#E69F00"}


def fig_axis():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels = [f"{e} — {t}" for e, t, _, _ in AXIS]
    vals = [v for _, _, v, _ in AXIS]
    cols = [AXIS_COL[k] for *_, k in AXIS]
    y = np.arange(len(AXIS))[::-1]
    ax.barh(y, vals, 0.62, color=cols, edgecolor="white", linewidth=0.5)
    for yi, v in zip(y, vals):
        ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 1.18); ax.set_xlabel("P(trigger within budget, uniform-random policy)")
    ax.set_title("Operational acquisition/retention axis (I-5)", loc="left", pad=8)
    h = [plt.Line2D([], [], marker="s", ls="", color=AXIS_COL["exo"], ms=8, label="exogenous"),
         plt.Line2D([], [], marker="s", ls="", color=AXIS_COL["mid"], ms=8, label="intermediate (residual acq.)"),
         plt.Line2D([], [], marker="s", ls="", color=AXIS_COL["agent"], ms=8, label="agent-contingent")]
    ax.legend(handles=h, loc="center right", bbox_to_anchor=(0.99, 0.30), frameon=False, fontsize=8.5)
    fig.tight_layout()
    return fig


def save(fig, out_dir, name):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {out/name}.pdf (+.png)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", default="all",
                    choices=["all", "interaction", "freeze", "curves", "axis"])
    ap.add_argument("--entity", default="jai-malegaonkar")
    ap.add_argument("--out-dir", default="experiments/analysis/figs")
    ap.add_argument("--cache-dir", default="experiments/analysis/figs/_cache")
    ap.add_argument("--refresh", action="store_true", help="re-pull wandb (as seeds land)")
    args = ap.parse_args()
    W = args.which

    if W in ("all", "interaction"):
        print("fig1 interaction …")
        mpg = load_grid(args.entity, "memrl-memtrain-mpg", "sparse", args.cache_dir, refresh=args.refresh)
        s13 = load_grid(args.entity, "memrl-s13-matched", "sparseV3", args.cache_dir, refresh=args.refresh)
        save(fig_interaction(mpg, s13), args.out_dir, "fig1_interaction")
    if W in ("all", "freeze"):
        print("fig2 freeze …")
        pen = load_grid(args.entity, "memrl-memtrain-mpg", "penalty", args.cache_dir,
                        extra="debug/action_frac_0", refresh=args.refresh)
        spa = load_grid(args.entity, "memrl-memtrain-mpg", "sparse", args.cache_dir, refresh=args.refresh)
        save(fig_freeze(pen, spa), args.out_dir, "fig2_freeze")
    if W in ("all", "curves"):
        print("fig3 curves …")
        specs = [("memrl-memtrain-mpg", "penalty", "GatedDeltaNet", "none"),
                 ("memrl-memtrain-mpg", "penalty", "GatedDeltaNet", "e3b_idm"),
                 ("memrl-s13-matched", "sparseV3", "LSTM", "none"),
                 ("memrl-s13-matched", "sparseV3", "LSTM", "e3b_idm")]
        curves = pull_curves(args.entity, specs, args.cache_dir, refresh=args.refresh)
        if curves:
            save(fig_curves(curves), args.out_dir, "fig3_curves")
        else:
            print("  (no curve data yet — skipped)")
    if W in ("all", "axis"):
        print("fig4 axis …")
        save(fig_axis(), args.out_dir, "fig4_axis")


if __name__ == "__main__":
    main()
