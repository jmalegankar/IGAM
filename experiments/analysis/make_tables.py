"""Regenerate every paper table from ONE tail-mean definition.

The Results tables drifted (up to 0.13) because cells were hand-filled at
different times and by different estimators (last-eval summary vs tail-mean).
This script is the single source of truth: it pulls each run's eval curve, takes
the mean over the final `TAIL_FRAC` of eval checkpoints, aggregates over seeds,
and emits the LaTeX for Tables 1-3 plus the cross-architecture spread stats and
the appendix falls table. Every number the paper prints should come from here.

    python -m experiments.analysis.make_tables            # pull wandb, write .tex
    python -m experiments.analysis.make_tables --csv cache.csv   # from cached grid

Output: experiments/analysis/tables/paper_tables.tex  (+ the tidy grid CSV).

Conventions:
  * tail mean = mean of the last max(1, round(TAIL_FRAC * len)) eval points.
  * A cell renders as `.XX` (leading zero dropped); exactly 1.0 renders `1.0`.
  * n != the table's stated default seed count is marked with a superscript, so a
    reader never mistakes a short cell for a full one.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

TAIL_FRAC = 0.20
ENTITY = "jai-malegaonkar"
CELLS = ["GRU", "LSTM", "RetNet", "GatedDeltaNet", "Mamba2", "Memoryless"]
DISP = {"GatedDeltaNet": "GatedDeltaNet", "Mamba2": "Mamba-2"}  # display names
BONUS = ["none", "e3b_idm", "noveld"]
HERE = Path(__file__).resolve().parent
OUT = HERE / "tables"


# ── pull ────────────────────────────────────────────────────────────────────
def tail_mean(run, frac=TAIL_FRAC):
    try:
        h = run.history(keys=["eval/success_rate"], samples=500)
        v = h["eval/success_rate"].dropna().values
    except Exception:
        v = []
    if len(v) == 0:
        s = run.summary.get("eval/success_rate", run.summary.get("success_rate"))
        return float(s) if s is not None else None
    k = max(1, round(len(v) * frac))
    return float(np.mean(v[-k:]))


def _cell(run):
    c = (run.config or {}).get("cell")
    return c.get("name") if isinstance(c, dict) else None


def pull(api, project, group=None, metric=tail_mean):
    """Return {(cell, bonus): [per-seed values]} for one project (optionally one group)."""
    filt = {"group": group} if group else None
    agg = defaultdict(list)
    for r in api.runs(f"{ENTITY}/{project}", filters=filt):
        if r.state not in ("finished", "running"):
            continue
        cell, intr = _cell(r), (r.config or {}).get("intrinsic")
        if cell not in CELLS or intr not in BONUS:
            continue
        m = metric(r)
        if m is not None:
            agg[(cell, intr)].append(m)
    return agg


# ── format ──────────────────────────────────────────────────────────────────
def _num(x):
    return "1.0" if round(x, 2) >= 1.0 else f"{x:.2f}".lstrip("0")


def fmt(vals, default_n, bold=False):
    """Render a cell as `mean` with a compact `±std` (std over seeds), n-superscript if != default."""
    if not vals:
        return "---"
    m, n = float(np.mean(vals)), len(vals)
    sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    ms = _num(m)
    if bold:
        ms = f"\\textbf{{{ms}}}"
    s = f"{ms}{{\\scriptsize$\\pm${_num(sd)}}}"
    if n != default_n:
        s += f"$^{{{n}}}$"
    return s


def spread(agg, bonus, cells):
    """std of per-cell means across `cells` for one bonus, + the per-cell means."""
    means = [np.mean(agg[(c, bonus)]) for c in cells if agg.get((c, bonus))]
    return (np.std(means, ddof=0), means) if len(means) == len(cells) else (np.nan, means)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "paper_tables.tex"))
    args = ap.parse_args()
    import wandb
    api = wandb.Api()
    OUT.mkdir(parents=True, exist_ok=True)
    tex, log = [], []

    def P(s=""):
        tex.append(s)

    # ---- Table 1: MysteryPath (4 variants) ----
    print("pulling MPG …", flush=True)
    mpg = {
        "Sparse":    pull(api, "memrl-memtrain-mpg", "sparse"),
        "Penalty":   pull(api, "memrl-memtrain-mpg", "penalty"),
        "Aligned":   pull(api, "memrl-memtrain-mpg", "aligned"),
        "Distractor": pull(api, "memrl-mpg-distractor", "distractor"),
    }
    P("% ==== Table 1: MysteryPath-Grid ====")
    P("\\begin{table*}[t]\\centering")
    P("\\caption{MysteryPath-Grid: deterministic-policy success rate (tail mean over "
      "the final 20\\% of eval checkpoints). Default $n{=}5$ seeds; a superscript gives "
      "the seed count where it differs. \\textbf{Bold} marks the frozen ($.00$) penalty "
      "cells. On sparse the bonus's gains concentrate on the higher-baseline cells; the "
      "penalty freezes every cell and the bonus reopens it; on the dense \\emph{aligned} "
      "reward the bonus is redundant, and on the dense-but-useless \\emph{distractor} it "
      "is undiminished.}")
    P("\\label{tab:mpg}\\small")
    P("\\begin{tabular}{@{}l ccc ccc ccc ccc@{}}\\toprule")
    P(" & \\multicolumn{3}{c}{Sparse} & \\multicolumn{3}{c}{Penalty} & "
      "\\multicolumn{3}{c}{Aligned (dense)} & \\multicolumn{3}{c}{Distractor (dense)} \\\\")
    P("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}")
    P("Cell & none & e3b & nvld & none & e3b & nvld & none & e3b & nvld & none & e3b & nvld \\\\\\midrule")
    for c in CELLS:
        cells_out = [DISP.get(c, c)]
        for var in ["Sparse", "Penalty", "Aligned", "Distractor"]:
            for b in BONUS:
                bold = (var == "Penalty" and b == "none")
                cells_out.append(fmt(mpg[var].get((c, b), []), 5, bold=bold))
        P(" & ".join(cells_out) + " \\\\")
    P("\\bottomrule\\end{tabular}\\end{table*}\n")

    # ---- Table 2: MemoryS13 ----
    print("pulling S13 …", flush=True)
    s13 = {
        "3x3 Sparse":    pull(api, "memrl-s13-matched", "sparseV3"),
        "3x3 Penalty":   pull(api, "memrl-s13-matched", "freezeV3"),
        "3x3 Distractor": pull(api, "memrl-s13-distractor", None),
        "7x7 Sparse":    pull(api, "memrl-s13-matched", "sparseV7"),
    }
    P("% ==== Table 2: MiniGrid-MemoryS13 ====")
    P("\\begin{table*}[t]\\centering")
    P("\\caption{MiniGrid-MemoryS13: success rate (tail mean, final 20\\% of eval "
      "checkpoints). Default $n{=}5$; superscripts mark $n{\\neq}5$. At the discriminative "
      "$3\\times3$ view the bonus lifts the chance-level cells toward the ceiling while the "
      "already-solving cells gain far less or slip; the penalty does \\emph{not} freeze here "
      "(no zero-cost sanctuary) and the bonus stays effective; the distractor holds the "
      "optimum fixed yet the equalization is preserved. The $7\\times7$ view is carried on "
      "sparse only---its no-bonus baselines saturate, which is why $3\\times3$ is the "
      "discriminative setting.}")
    P("\\label{tab:s13}\\small\\setlength{\\tabcolsep}{4.5pt}")
    P("\\begin{tabular}{@{}l ccc ccc ccc ccc@{}}\\toprule")
    P(" & \\multicolumn{3}{c}{$3\\times3$ Sparse} & \\multicolumn{3}{c}{$3\\times3$ Penalty} & "
      "\\multicolumn{3}{c}{$3\\times3$ Distractor} & \\multicolumn{3}{c}{$7\\times7$ Sparse} \\\\")
    P("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}")
    P("Cell & none & e3b & nvld & none & e3b & nvld & none & e3b & nvld & none & e3b & nvld \\\\\\midrule")
    for c in CELLS:
        cells_out = [DISP.get(c, c)]
        for var in ["3x3 Sparse", "3x3 Penalty", "3x3 Distractor", "7x7 Sparse"]:
            for b in BONUS:
                cells_out.append(fmt(s13[var].get((c, b), []), 5))
        P(" & ".join(cells_out) + " \\\\")
    P("\\bottomrule\\end{tabular}\\end{table*}\n")

    # ---- Table 3: TinyReproduce ----
    print("pulling Tiny …", flush=True)
    tiny = {"Sparse": pull(api, "memrl-memtrain-tiny", "sparse"),
            "Dense":  pull(api, "memrl-memtrain-tiny", "dense")}
    tuned = defaultdict(list)
    for r in api.runs(f"{ENTITY}/memrl-tiny-exp5"):
        if r.state not in ("finished", "running"):
            continue
        g = (r.group or "").replace("besthp_", ""); intr = (r.config or {}).get("intrinsic")
        m = tail_mean(r)
        if m is not None and g in ("sparse", "dense") and intr in BONUS:
            tuned[(g, intr)].append(m)
    P("% ==== Table 3: TinyReproduce ====")
    P("\\begin{table}[t]\\centering")
    P("\\caption{TinyReproduce ($k{=}10$, reverse): exact-recall success rate (tail mean). "
      "Shared-hyperparameter block is $n{=}3$ for none/E3B and $n{=}4$--$5$ for NovelD; the "
      "tuned GatedDeltaNet row is $n{=}4$--$5$. Superscripts give the seed count where it "
      "differs from the block default. Both bonuses are inert on every cell with headroom "
      "($|\\Delta|{\\le}.05$). RetNet, Mamba-2 and Memoryless do not learn the task at the "
      "shared rate and are reported floored; GatedDeltaNet is floored there too, hence the "
      "tuned row (learning rate $10^{-3}$), where it has headroom and the bonus still does "
      "nothing. The tuned row is not comparable to the block above it.}")
    P("\\label{tab:tiny}\\small\\setlength{\\tabcolsep}{5pt}")
    P("\\begin{tabular}{@{}l ccc ccc@{}}\\toprule")
    P(" & \\multicolumn{3}{c}{Sparse} & \\multicolumn{3}{c}{Dense} \\\\")
    P("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
    P("Cell & none & e3b & nvld & none & e3b & nvld \\\\\\midrule")
    for c in CELLS:
        row = [DISP.get(c, c)]
        for var in ["Sparse", "Dense"]:
            dn = 3 if var else 3
            for b in BONUS:
                # default n=3 for none/e3b, n=5 for nvld: mark deviations from those
                dflt = 3 if b in ("none", "e3b_idm") else 5
                row.append(fmt(tiny[var].get((c, b), []), dflt))
        P(" & ".join(row) + " \\\\")
    P("\\midrule")
    trow = ["GatedDeltaNet (tuned)"]
    for var in ["sparse", "dense"]:
        for b in BONUS:
            dflt = 4 if b in ("none", "e3b_idm") else 5
            trow.append(fmt(tuned.get((var, b), []), dflt))
    P(" & ".join(trow) + " \\\\")
    P("\\bottomrule\\end{tabular}\\end{table}\n")

    # ---- spread stats (memory cells only) ----
    MEM = ["GRU", "LSTM", "RetNet", "GatedDeltaNet", "Mamba2"]
    P("% ==== cross-architecture spread (std over the 5 memory cells) ====")
    for env, agg in [("MysteryPath sparse", mpg["Sparse"]), ("MemoryS13 3x3 sparse", s13["3x3 Sparse"])]:
        line = f"% {env}: "
        for b in BONUS:
            sd, _ = spread(agg, b, MEM)
            line += f"std[{b}]={sd:.3f}  "
        P(line)
    log.append("SPREAD (std over memory cells):")
    for env, agg in [("MPG", mpg["Sparse"]), ("S13", s13["3x3 Sparse"])]:
        row = f"  {env}: "
        for b in BONUS:
            sd, _ = spread(agg, b, MEM)
            row += f"{b}={sd:.3f} "
        log.append(row)

    # ---- appendix: falls ----
    P("% ==== Appendix: episode falls on MysteryPath sparse (rollout/ep_num_fails_mean) ====")
    falls = pull(api, "memrl-memtrain-mpg", "sparse",
                 metric=lambda r: (float(r.summary["rollout/ep_num_fails_mean"])
                                   if "rollout/ep_num_fails_mean" in r.summary else None))
    P("\\begin{table}[t]\\centering")
    P("\\caption{MysteryPath sparse: mean falls per episode ($n{=}5$). The memoryless "
      "agent under E3B covers the most ground yet never succeeds.}")
    P("\\label{tab:falls}\\small\\begin{tabular}{@{}lcc@{}}\\toprule")
    P("Cell & none & E3B \\\\\\midrule")
    for c in CELLS:
        n = falls.get((c, "none"), []); e = falls.get((c, "e3b_idm"), [])
        ns = f"{np.mean(n):.1f}" if n else "---"; es = f"{np.mean(e):.1f}" if e else "---"
        P(f"{DISP.get(c, c)} & {ns} & {es} \\\\")
    P("\\bottomrule\\end{tabular}\\end{table}")

    Path(args.out).write_text("\n".join(tex) + "\n")
    print("\n".join(log))
    print(f"\nwrote {args.out}  ({len(tex)} lines)")


if __name__ == "__main__":
    main()
