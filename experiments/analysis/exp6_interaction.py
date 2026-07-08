"""EXP-6 — the cell×bonus interaction and its cross-env sign reversal (MPG amplify ↔ S13 equalize).

Two complementary reads, both over the SAME landed grids (MysteryPath sparse + S13 matched
sparse), no new training:

  1. R2 (pre-registered, the inference): per-env (Δstrong − Δweak) with seed-bootstrap CIs,
     and the cross-env sign-flip verdict. This is `registered_stats.r2_interaction`. It is the
     claim; the big grids are descriptive.
  2. ART-ANOVA (reviewer-facing backstop, answers the A11 multiple-comparisons attack): the
     non-parametric factorial test that the env×cell×bonus 3-way interaction is real — i.e. the
     cell:bonus interaction genuinely differs by env — with the F, p, and partial η². Aligned
     Rank Transform (Wobbrock et al. 2011): align the response to isolate one effect, rank, run
     a standard factorial ANOVA on the ranks, read only the aligned term. Run per-bonus
     ({none,e3b} and {none,noveld}) to mirror R2, plus the per-env 2-way cell:bonus.

The response is per-run CONVERGED success (tail-mean of the last 20% of eval points). Excludes
pbim (its S13 arm is a separate training pathology) and Memoryless (the floor control).

Usage:
    python -m experiments.analysis.exp6_interaction                      # pull wandb, report
    python -m experiments.analysis.exp6_interaction --out-csv exp6.csv   # + cache the tidy grid
    python -m experiments.analysis.exp6_interaction --csv exp6.csv       # re-run from cache (fast)
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.analysis.registered_stats import CellData, r2_interaction  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# wandb pull → tidy grid (env, cell, bonus, seed, success)
# ──────────────────────────────────────────────────────────────────────────
def _cellname(cfg):
    c = cfg.get("cell")
    return c.get("name") if isinstance(c, dict) else c


def _tail_success(run, frac=0.2):
    try:
        h = run.history(keys=["eval/success_rate"], samples=120, pandas=False)
        v = [x["eval/success_rate"] for x in h if x.get("eval/success_rate") is not None]
    except Exception:
        v = []
    if not v:
        return None
    k = max(3, int(len(v) * frac))
    return float(np.mean(v[-k:]))


def pull_grid(api, entity, project, density, env_label, min_step) -> pd.DataFrame:
    """One row per (cell,bonus,seed): converged success, from the best finished/running run."""
    best = defaultdict(lambda: (-1, None))
    for r in api.runs(f"{entity}/{project}"):
        if r.state not in ("finished", "running"):
            continue
        c = r.config
        if c.get("wandb_group") != density:
            continue
        key = (_cellname(c), c.get("intrinsic"), c.get("seed"))
        gs = r.summary.get("global_step", 0) or 0
        if gs > best[key][0]:
            best[key] = (gs, r)
    rows = []
    for (cell, bonus, seed), (gs, r) in best.items():
        if gs < min_step or cell is None:
            continue
        s = _tail_success(r)
        if s is not None:
            rows.append({"env": env_label, "cell": cell, "bonus": bonus,
                         "seed": seed, "success": s, "global_step": gs})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# ART-ANOVA (Aligned Rank Transform)
# ──────────────────────────────────────────────────────────────────────────
def _art_align(df: pd.DataFrame, response: str, all_factors: list[str], target: list[str]):
    """Align `response` to isolate the `target` effect: residual (from the full-cell mean)
    plus the pure target-interaction estimate (inclusion–exclusion over marginal means)."""
    resid = df[response] - df.groupby(all_factors)[response].transform("mean")
    grand = df[response].mean()
    est = np.zeros(len(df))
    S = list(target)
    for k in range(len(S) + 1):
        for T in itertools.combinations(S, k):
            sign = (-1) ** (len(S) - len(T))
            marg = grand if not T else df.groupby(list(T))[response].transform("mean").to_numpy()
            est = est + sign * (marg if np.ndim(marg) else np.full(len(df), marg))
    return resid.to_numpy() + est


def art_anova(df: pd.DataFrame, response: str, factors: list[str], target: list[str]) -> dict:
    """Aligned Rank Transform ANOVA F-test for the `target` term over `factors`.
    Returns F, p, partial η², and the max NON-target F (should be ≈0 if alignment worked).
    Requires statsmodels (`pip install statsmodels`); returns an error dict if absent."""
    from scipy.stats import rankdata
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
    except ModuleNotFoundError:
        return {"error": "statsmodels not installed (pip install statsmodels for the ART omnibus)"}

    d = df.copy()
    d["_aligned"] = _art_align(d, response, factors, target)
    d["_rank"] = rankdata(d["_aligned"].to_numpy())
    formula = "_rank ~ " + "*".join(f"C({f})" for f in factors)
    model = smf.ols(formula, data=d).fit()
    tbl = anova_lm(model, typ=2)
    term = ":".join(f"C({f})" for f in target)
    if term not in tbl.index:
        return {"error": f"term {term} not in ANOVA table {list(tbl.index)}"}
    ss_e = float(tbl.loc[term, "sum_sq"]); ss_r = float(tbl.loc["Residual", "sum_sq"])
    other = [t for t in tbl.index if t not in (term, "Residual")]
    max_other_F = max((float(tbl.loc[t, "F"]) for t in other), default=0.0)
    return {"F": float(tbl.loc[term, "F"]), "p": float(tbl.loc[term, "PR(>F)"]),
            "df": (float(tbl.loc[term, "df"]), float(tbl.loc["Residual", "df"])),
            "partial_eta2": ss_e / (ss_e + ss_r) if (ss_e + ss_r) > 0 else float("nan"),
            "alignment_ok": max_other_F < 1.0, "max_nontarget_F": max_other_F}


# ──────────────────────────────────────────────────────────────────────────
def _cells_for(df, env, bonus, cell_order):
    out = []
    sub = df[(df.env == env)]
    for cell in cell_order:
        none = sub[(sub.cell == cell) & (sub.bonus == "none")]["success"].to_numpy(float)
        bon = sub[(sub.cell == cell) & (sub.bonus == bonus)]["success"].to_numpy(float)
        if none.size and bon.size:
            out.append(CellData(cell, none, bon))
    return out


def report(df: pd.DataFrame, strong, weak, cells, amplify_env, equalize_env):
    order = [c for c in cells]
    print(f"\ncells={order}  strong={strong}  weak={weak}")
    print(f"envs: amplify={amplify_env}  equalize={equalize_env}")
    print("coverage (rows per env×bonus):")
    print("  " + df.groupby(["env", "bonus"]).size().to_string().replace("\n", "\n  "))

    for bonus in ("e3b_idm", "noveld"):
        amp = _cells_for(df, amplify_env, bonus, order)
        eq = _cells_for(df, equalize_env, bonus, order)
        print(f"\n### bonus = {bonus}")
        if amp:
            print("  " + amplify_env + " Δ:",
                  {cd.cell: round(cd.bonus.mean() - cd.none.mean(), 2) for cd in amp})
        if eq:
            print("  " + equalize_env + " Δ:",
                  {cd.cell: round(cd.bonus.mean() - cd.none.mean(), 2) for cd in eq})
        # R2 (pre-registered)
        amp_s = [cd for cd in amp if cd.cell in strong + weak]
        eq_s = [cd for cd in eq if cd.cell in strong + weak]
        if amp_s and eq_s:
            r = r2_interaction(amp_s, eq_s, strong, weak)
            a, e = r["amplify_leg"], r["equalize_leg"]
            print(f"  R2 : {amplify_env} {a['interaction']:+.3f} CI[{a['lo']:+.2f},{a['hi']:+.2f}]"
                  f" | {equalize_env} {e['interaction']:+.3f} CI[{e['lo']:+.2f},{e['hi']:+.2f}]"
                  f"  → sign_flip={r['sign_flip_confirmed']}")
        # ART-ANOVA 3-way (env×cell×bonus) on {none, bonus}
        sub = df[df.bonus.isin(["none", bonus]) & df.cell.isin(order)]
        if sub.env.nunique() == 2 and sub.cell.nunique() >= 2:
            a3 = art_anova(sub, "success", ["env", "cell", "bonus"], ["env", "cell", "bonus"])
            if "error" in a3:
                print(f"  ART: skipped — {a3['error']}")
            else:
                print(f"  ART: env×cell×bonus  F({a3['df'][0]:.0f},{a3['df'][1]:.0f})="
                      f"{a3['F']:.2f}  p={a3['p']:.4f}  partial_η²={a3['partial_eta2']:.3f}"
                      f"  [align_ok={a3['alignment_ok']}]")
        # per-env 2-way cell:bonus (amplify vs equalize magnitude)
        for env in (amplify_env, equalize_env):
            s2 = df[(df.env == env) & df.bonus.isin(["none", bonus]) & df.cell.isin(order)]
            if s2.cell.nunique() >= 2 and s2.bonus.nunique() == 2:
                a2 = art_anova(s2, "success", ["cell", "bonus"], ["cell", "bonus"])
                if "error" not in a2:
                    print(f"       {env} cell×bonus  F={a2['F']:.2f} p={a2['p']:.4f} "
                          f"partial_η²={a2['partial_eta2']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="jai-malegaonkar")
    ap.add_argument("--mpg-project", default="memrl-memtrain-mpg")
    ap.add_argument("--mpg-density", default="sparse")
    ap.add_argument("--s13-project", default="memrl-s13-matched")
    ap.add_argument("--s13-density", default="sparseV3")
    ap.add_argument("--min-step", type=float, default=15e6)
    ap.add_argument("--cells", default="GRU,LSTM,RetNet,GatedDeltaNet,Mamba2")
    ap.add_argument("--strong", default="RetNet,GatedDeltaNet")
    ap.add_argument("--weak", default="GRU,LSTM")
    ap.add_argument("--csv", default=None, help="load the tidy grid from CSV (skip wandb)")
    ap.add_argument("--out-csv", default=None, help="cache the pulled tidy grid to CSV")
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        import wandb
        api = wandb.Api()
        print("pulling MPG …", flush=True)
        mpg = pull_grid(api, args.entity, args.mpg_project, args.mpg_density, "MPG", args.min_step)
        print("pulling S13 …", flush=True)
        s13 = pull_grid(api, args.entity, args.s13_project, args.s13_density, "S13", args.min_step)
        df = pd.concat([mpg, s13], ignore_index=True)
        if args.out_csv:
            df.to_csv(args.out_csv, index=False)
            print(f"wrote {args.out_csv} ({len(df)} rows)")

    cells = [c.strip() for c in args.cells.split(",")]
    strong = [c.strip() for c in args.strong.split(",")]
    weak = [c.strip() for c in args.weak.split(",")]
    report(df, strong, weak, cells, amplify_env="MPG", equalize_env="S13")


if __name__ == "__main__":
    main()
