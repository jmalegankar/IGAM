"""Registered-contrast statistics for the memory×exploration paper (handoff §1.2, §4, §15).

Inference in the paper is confined to a handful of PRE-REGISTERED contrasts; the big
grids are descriptive (CIs only). This module implements exactly those contrasts as
pure functions so a landed grid becomes a claim with one call, and the decision rules
in the handoff are executable, not prose:

  bootstrap_ci        seed-level bootstrap CI of any statistic (the descriptive read)
  paired_diff_ci      CI of a within-cell paired delta (bonus − none), the HP-fair unit
  tost_equivalent     TOST for the α≈0 nulls (±0.05 margin) — the Tiny / MortarMayhem null
  headroom_lift       η = (bonus − none)/(ceiling − none), per cell (R1)
  r1_equalization     mean η(weak) − mean η(strong) with bootstrap CI (R1: >0 ⇒ equalize)
  r2_interaction      (Δstrong − Δweak) per env; R2: +on amplify-env, −on equalize-env
  rho_pbim            ρ = (pbim − none)/(e3b − none) with CI + the alive-guard (R4/EXP-9)
  freeze_signature    per-cell freeze booleans: succ CI-upper<τ AND idle elevated (R4)

Design choices (locked):
  * Bootstrap over SEEDS (the unit of replication), not steps/episodes.
  * Deltas are PAIRED by seed where the seeds are matched (bonus and none share seeds) —
    the within-cell paired delta is the HP-fair comparison (handoff A12). Falls back to
    unpaired when seed sets differ.
  * TOST at α ⇔ the (1−2α) CI of the difference lies within (−margin, +margin). We use
    the 90% bootstrap CI (α=0.05) so equivalence uses the SAME bootstrap machinery as
    everything else; a scipy t-based TOST is offered as a cross-check.
  * No magic censoring: floor/ceiling handling is explicit (headroom normalization),
    never a silent clip.

Tidy input schema (one row per run): columns
    env, cell, arm, bonus, seed, success[, auc]
`arm` is the density label (sparse/penalty/aligned or sparseV3/freezeV3/…). Load with
`load_grid(csv)`; the contrast functions take grouped numpy arrays so they are unit-
testable without a CSV (see `--selftest`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd
except Exception:  # pandas is only needed for the CSV/CLI layer
    pd = None


# ──────────────────────────────────────────────────────────────────────────
# Bootstrap primitives (seed-level)
# ──────────────────────────────────────────────────────────────────────────
def bootstrap_ci(values, statistic=np.mean, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(point, lo, hi) — percentile bootstrap CI of `statistic` over `values`.

    `values` are the per-seed measurements for one (env, cell, arm, bonus) cell.
    Returns NaNs if empty.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"),) * 3
    if v.size == 1:
        return (float(statistic(v)), float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot = statistic(v[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(v)), float(lo), float(hi)


def _resample_diff(a, b, paired, rng, n_boot):
    """Bootstrap distribution of mean(a) − mean(b). Paired resamples a shared seed
    index (variance-reducing, correct when a[i]/b[i] are the same seed)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if paired and a.size == b.size and a.size > 0:
        n = a.size
        idx = rng.integers(0, n, size=(n_boot, n))
        return (a[idx].mean(1) - b[idx].mean(1))
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    return a[ia].mean(1) - b[ib].mean(1)


def paired_diff_ci(bonus, none, n_boot: int = 10000, alpha: float = 0.05,
                   seed: int = 0, paired: bool = True) -> dict:
    """CI of the within-cell delta mean(bonus) − mean(none). The HP-fair unit of the
    amplification/equalization reads. `paired` pairs by seed order when sizes match."""
    a = np.asarray(bonus, float); a = a[~np.isnan(a)]
    b = np.asarray(none, float); b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_bonus": int(a.size), "n_none": int(b.size), "paired": False}
    use_paired = paired and (a.size == b.size)
    rng = np.random.default_rng(seed)
    boot = _resample_diff(a, b, use_paired, rng, n_boot)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"delta": float(a.mean() - b.mean()), "lo": float(lo), "hi": float(hi),
            "n_bonus": int(a.size), "n_none": int(b.size), "paired": bool(use_paired),
            "sig": bool(lo > 0 or hi < 0)}


# ──────────────────────────────────────────────────────────────────────────
# TOST — equivalence for the α≈0 nulls (handoff §4)
# ──────────────────────────────────────────────────────────────────────────
def tost_equivalent(bonus, none, margin: float = 0.05, alpha: float = 0.05,
                    n_boot: int = 10000, seed: int = 0, paired: bool = True) -> dict:
    """Two-one-sided-test equivalence of (bonus − none) within ±margin.

    Bootstrap form: equivalent at level α ⇔ the (1−2α) CI of the difference lies
    inside (−margin, +margin). Reports that CI and the verdict; also a scipy t-based
    cross-check when scipy is available. Use for: Tiny (E3B−none), MortarMayhem-aligned.
    """
    a = np.asarray(bonus, float); a = a[~np.isnan(a)]
    b = np.asarray(none, float); b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return {"equivalent": None, "reason": "n<2", "margin": margin}
    use_paired = paired and (a.size == b.size)
    rng = np.random.default_rng(seed)
    boot = _resample_diff(a, b, use_paired, rng, n_boot)
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])  # (1−2α) CI
    out = {"delta": float(a.mean() - b.mean()), "ci90_lo": float(lo), "ci90_hi": float(hi),
           "margin": float(margin), "equivalent": bool(lo > -margin and hi < margin),
           "paired": bool(use_paired), "n_bonus": int(a.size), "n_none": int(b.size)}
    try:  # scipy t-based TOST cross-check (paired if matched, else Welch)
        from scipy import stats
        d = a - b if use_paired else None
        if use_paired:
            p_lo = stats.ttest_1samp(d, -margin, alternative="greater").pvalue
            p_hi = stats.ttest_1samp(d, margin, alternative="less").pvalue
        else:
            p_lo = stats.ttest_ind(a, b, equal_var=False, alternative="greater").pvalue
            # shift for the +margin bound: test (a-b) < margin
            p_hi = stats.ttest_ind(a - margin, b, equal_var=False, alternative="less").pvalue
        out["tost_p"] = float(max(p_lo, p_hi))
        out["tost_t_equivalent"] = bool(out["tost_p"] < alpha)
    except Exception:
        pass
    return out


# ──────────────────────────────────────────────────────────────────────────
# R1 — equalization (headroom-normalized lift)
# ──────────────────────────────────────────────────────────────────────────
def headroom_lift(bonus, none, ceiling: float) -> float:
    """η = (mean(bonus) − mean(none)) / (ceiling − mean(none)); NaN if no headroom."""
    a = float(np.nanmean(bonus)); b = float(np.nanmean(none))
    denom = ceiling - b
    if denom <= 1e-9:
        return float("nan")
    return (a - b) / denom


@dataclass
class CellData:
    """Per-cell measurements for one arm: bonus vs none arrays (per-seed success)."""
    cell: str
    none: np.ndarray
    bonus: np.ndarray


def r1_equalization(cells: list[CellData], weak: list[str], strong: list[str],
                    ceiling: float | None = None, n_boot: int = 10000,
                    alpha: float = 0.05, seed: int = 0) -> dict:
    """R1: mean η(weak) − mean η(strong) with bootstrap CI over cells.

    η per cell is headroom-normalized to `ceiling` (default: best per-seed success
    observed across all cells' bonus+none arms, or 1.0 if that exceeds 0.9). Equalization
    is confirmed iff the CI of (mean η_weak − mean η_strong) EXCLUDES 0 on the positive
    side. Bootstrap resamples cells within each class (cells are the unit here).
    """
    by = {c.cell: c for c in cells}
    if ceiling is None:
        allv = np.concatenate([np.concatenate([c.none, c.bonus]) for c in cells])
        m = float(np.nanmax(allv)) if allv.size else 1.0
        ceiling = 1.0 if m > 0.9 else m
    eta = {c.cell: headroom_lift(c.bonus, c.none, ceiling) for c in cells}
    weak_eta = np.array([eta[w] for w in weak if w in by and not np.isnan(eta[w])])
    strong_eta = np.array([eta[s] for s in strong if s in by and not np.isnan(eta[s])])
    if weak_eta.size == 0 or strong_eta.size == 0:
        return {"gap": float("nan"), "reason": "empty class after headroom filter",
                "ceiling": ceiling, "eta": eta}
    rng = np.random.default_rng(seed)
    wi = rng.integers(0, weak_eta.size, size=(n_boot, weak_eta.size))
    si = rng.integers(0, strong_eta.size, size=(n_boot, strong_eta.size))
    boot = weak_eta[wi].mean(1) - strong_eta[si].mean(1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    gap = float(weak_eta.mean() - strong_eta.mean())
    return {"gap": gap, "lo": float(lo), "hi": float(hi), "ceiling": float(ceiling),
            "eta": {k: round(v, 4) for k, v in eta.items()},
            "equalize_confirmed": bool(lo > 0),
            "mean_eta_weak": float(weak_eta.mean()), "mean_eta_strong": float(strong_eta.mean())}


# ──────────────────────────────────────────────────────────────────────────
# R2 — the same-bonus sign flip (cross-env interaction)
# ──────────────────────────────────────────────────────────────────────────
def _interaction_leg(cells: list[CellData], strong: list[str], weak: list[str],
                     n_boot: int, alpha: float, rng) -> dict:
    """(mean Δstrong − mean Δweak) with CI, where Δ = mean(bonus) − mean(none) per cell."""
    d = {c.cell: float(np.nanmean(c.bonus) - np.nanmean(c.none)) for c in cells}
    ds = np.array([d[s] for s in strong if s in d])
    dw = np.array([d[w] for w in weak if w in d])
    si = rng.integers(0, ds.size, size=(n_boot, ds.size))
    wi = rng.integers(0, dw.size, size=(n_boot, dw.size))
    boot = ds[si].mean(1) - dw[wi].mean(1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"interaction": float(ds.mean() - dw.mean()), "lo": float(lo), "hi": float(hi),
            "delta_strong": float(ds.mean()), "delta_weak": float(dw.mean()),
            "per_cell_delta": {k: round(v, 4) for k, v in d.items()}}


def r2_interaction(amplify_env: list[CellData], equalize_env: list[CellData],
                   strong: list[str], weak: list[str], n_boot: int = 10000,
                   alpha: float = 0.05, seed: int = 0) -> dict:
    """R2 headline: the SAME bonus's (Δstrong − Δweak) is POSITIVE on the amplify env
    (MysteryPath) and NEGATIVE on the equalize env (S13 matched), each CI-separated
    from 0. Confirmed iff both legs hold."""
    rng = np.random.default_rng(seed)
    amp = _interaction_leg(amplify_env, strong, weak, n_boot, alpha, rng)
    eq = _interaction_leg(equalize_env, strong, weak, n_boot, alpha, rng)
    confirmed = bool(amp["lo"] > 0 and eq["hi"] < 0)
    return {"amplify_leg": amp, "equalize_leg": eq, "sign_flip_confirmed": confirmed}


# ──────────────────────────────────────────────────────────────────────────
# ρ — PBIM cannot rescue (the potential-delivery null); with the alive-guard
# ──────────────────────────────────────────────────────────────────────────
def rho_pbim(none, e3b, pbim, n_boot: int = 10000, alpha: float = 0.05,
             seed: int = 0, ci_upper_threshold: float = 0.25) -> dict:
    """ρ = (mean(pbim) − mean(none)) / (mean(e3b) − mean(none)); the fraction of E3B's
    rescue that the potential delivery recovers. Guarded: undefined unless the E3B
    rescue itself is real (e3b − none CI-lower > 0). Confirmed PBIM-null iff ρ CI-upper
    < `ci_upper_threshold`. Use on the FROZEN arm (rescue) and the ALIVE arm (W-5)."""
    b0 = np.asarray(none, float); b0 = b0[~np.isnan(b0)]
    e = np.asarray(e3b, float); e = e[~np.isnan(e)]
    p = np.asarray(pbim, float); p = p[~np.isnan(p)]
    if min(b0.size, e.size, p.size) < 2:
        return {"rho": float("nan"), "reason": "n<2"}
    rng = np.random.default_rng(seed)
    n = min(b0.size, e.size, p.size)
    # paired-by-seed resample (arms share seeds); truncate to common n
    idx = rng.integers(0, n, size=(n_boot, n))
    denom = e[:n][idx].mean(1) - b0[:n][idx].mean(1)
    numer = p[:n][idx].mean(1) - b0[:n][idx].mean(1)
    e3b_rescue = float(e[:n].mean() - b0[:n].mean())
    e3b_lo = float(np.percentile(denom, 100 * alpha / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_boot = np.where(np.abs(denom) > 1e-6, numer / denom, np.nan)
    rho_boot = rho_boot[~np.isnan(rho_boot)]
    point = (float(np.nanmean(p[:n]) - np.nanmean(b0[:n]))
             / e3b_rescue) if abs(e3b_rescue) > 1e-6 else float("nan")
    lo, hi = (np.percentile(rho_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if rho_boot.size else (float("nan"), float("nan")))
    guard_ok = e3b_lo > 0
    return {"rho": point, "lo": float(lo), "hi": float(hi),
            "e3b_rescue": e3b_rescue, "e3b_rescue_ci_lo": e3b_lo,
            "alive_guard_ok": bool(guard_ok),
            "pbim_null_confirmed": bool(guard_ok and hi < ci_upper_threshold)}


# ──────────────────────────────────────────────────────────────────────────
# R4 — the freeze signature (per cell)
# ──────────────────────────────────────────────────────────────────────────
def freeze_signature(success, idle_frac, alive_idle_frac, succ_ceiling: float = 0.05,
                     idle_ratio: float = 2.0, n_boot: int = 10000, alpha: float = 0.05,
                     seed: int = 0) -> dict:
    """Per-cell freeze test (R4): success CI-UPPER < `succ_ceiling` AND the idle (no-op)
    fraction is ≥ `idle_ratio`× the return-matched alive-arm idle level. `success` and
    `idle_frac` are the penalty-none per-seed arrays; `alive_idle_frac` the sparse-none
    per-seed idle array. Returns the booleans + the CIs behind them."""
    s_pt, s_lo, s_hi = bootstrap_ci(success, n_boot=n_boot, alpha=alpha, seed=seed)
    i_pt, i_lo, i_hi = bootstrap_ci(idle_frac, n_boot=n_boot, alpha=alpha, seed=seed)
    alive_i = float(np.nanmean(alive_idle_frac)) if np.asarray(alive_idle_frac).size else float("nan")
    frozen = bool(s_hi < succ_ceiling and (not np.isnan(alive_i)) and i_lo > idle_ratio * alive_i)
    return {"frozen": frozen, "success": s_pt, "success_ci_hi": s_hi,
            "idle_frac": i_pt, "idle_ci_lo": i_lo, "alive_idle_frac": alive_i,
            "idle_threshold": idle_ratio * alive_i}


# ──────────────────────────────────────────────────────────────────────────
# CSV / CLI layer
# ──────────────────────────────────────────────────────────────────────────
def load_grid(csv_path: str):
    if pd is None:
        raise SystemExit("pandas required for the CSV layer")
    return pd.read_csv(csv_path)


def cells_from_df(df, env: str, arm: str, bonus: str, none_bonus: str = "none",
                  metric: str = "success") -> list[CellData]:
    """Build per-cell (none vs bonus) arrays from a tidy grid dataframe."""
    sub = df[(df["env"] == env) & (df["arm"] == arm)]
    out = []
    for cell, g in sub.groupby("cell"):
        none = g[g["bonus"] == none_bonus][metric].to_numpy(float)
        bon = g[g["bonus"] == bonus][metric].to_numpy(float)
        if none.size and bon.size:
            out.append(CellData(cell=str(cell), none=none, bonus=bon))
    return out


def _selftest() -> None:
    """Synthetic data with KNOWN structure → the rules must fire the right way."""
    rng = np.random.default_rng(0)

    def arm(mean, n=5, sd=0.05):
        return np.clip(rng.normal(mean, sd, n), 0, 1)

    # R1/R2 amplify env (MysteryPath): strong cells benefit MORE than weak (Δstrong>Δweak)
    amp = [CellData("RetNet", arm(0.30), arm(0.75)), CellData("GatedDeltaNet", arm(0.28), arm(0.72)),
           CellData("GRU", arm(0.30), arm(0.45)), CellData("LSTM", arm(0.29), arm(0.44))]
    # equalize env (S13 matched): weak cells benefit MORE (Δweak>Δstrong) → η(weak)>η(strong)
    eq = [CellData("RetNet", arm(0.80), arm(0.85)), CellData("GatedDeltaNet", arm(0.82), arm(0.86)),
          CellData("GRU", arm(0.45), arm(0.80)), CellData("LSTM", arm(0.44), arm(0.78))]
    strong, weak = ["RetNet", "GatedDeltaNet"], ["GRU", "LSTM"]

    r1 = r1_equalization(eq, weak, strong)
    r2 = r2_interaction(amp, eq, strong, weak)
    assert r1["equalize_confirmed"], r1
    assert r2["sign_flip_confirmed"], r2
    assert r2["amplify_leg"]["interaction"] > 0 and r2["equalize_leg"]["interaction"] < 0

    # TOST: e3b ≈ none within ±0.05 (Tiny null, tight sd) vs a real effect
    null = tost_equivalent(arm(0.40, 8, 0.015), arm(0.40, 8, 0.015), margin=0.05)
    effect = tost_equivalent(arm(0.60, 6), arm(0.40, 6), margin=0.05)
    assert null["equivalent"] is True, null
    assert effect["equivalent"] is False, effect

    # ρ: PBIM recovers ~0 of E3B's rescue (frozen arm), guard passes
    r = rho_pbim(none=arm(0.02, 5), e3b=arm(0.55, 5), pbim=arm(0.03, 5))
    assert r["alive_guard_ok"] and r["pbim_null_confirmed"], r

    # freeze signature: penalty-none frozen (succ≈0, idle high) vs sparse-none alive
    fz = freeze_signature(success=arm(0.01, 5, 0.01), idle_frac=arm(0.85, 5, 0.03),
                          alive_idle_frac=arm(0.20, 5, 0.03))
    assert fz["frozen"], fz
    print("registered_stats selftest: PASS")
    print(f"  R1 gap={r1['gap']:.3f} CI[{r1['lo']:.3f},{r1['hi']:.3f}] eta={r1['eta']}")
    print(f"  R2 amplify int={r2['amplify_leg']['interaction']:.3f} "
          f"equalize int={r2['equalize_leg']['interaction']:.3f}")
    print(f"  TOST null equiv={null['equivalent']} (CI90 "
          f"[{null['ci90_lo']:.3f},{null['ci90_hi']:.3f}]); effect equiv={effect['equivalent']}")
    print(f"  rho={r['rho']:.3f} CI[{r['lo']:.3f},{r['hi']:.3f}] guard={r['alive_guard_ok']}")
    print(f"  freeze frozen={fz['frozen']} succ_hi={fz['success_ci_hi']:.3f} "
          f"idle_lo={fz['idle_ci_lo']:.3f} thr={fz['idle_threshold']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run synthetic-data checks")
    ap.add_argument("--csv", help="tidy grid CSV (env,cell,arm,bonus,seed,success[,auc])")
    ap.add_argument("--metric", default="success")
    args = ap.parse_args()
    if args.selftest or not args.csv:
        _selftest(); return
    df = load_grid(args.csv)
    print(df.groupby(["env", "arm", "bonus", "cell"])[args.metric]
          .agg(["mean", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()
