"""Telescoping-validity + magnitude check for the OraclePotentialWrapper (H-POT §3.2).

The oracle-Φ arm is only a legitimate PBRS arm if its shaping is return-neutral: over any
episode the discounted shaping sum ΣₙγⁿFₙ must telescope to γᵀΦ_T − Φ_0 = −Φ_0 (Φ(absorbing)≡0),
i.e. it depends ONLY on the start state, never the trajectory → the optimal policy is unchanged
(Ng–Harada–Russell 1999). This is the same test PBIM's potential must pass.

Also reports mean|F| so β can be tuned to match the e3b arm's delivered bonus (~2×).

Run:  python -m experiments.analysis.oracle_potential_check --episodes 50 --beta 0.02
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import memory_gym  # noqa: F401 — registers the ids
import numpy as np

from memrl.envs.memory_gym_wrappers import OraclePotentialWrapper, StickyResetOptions


def check(episodes: int = 50, beta: float = 0.02, gamma: float = 0.995,
          seed: int = 0, tol: float = 1e-6) -> dict:
    env = gym.make("MysteryPath-Grid-v0")
    env = StickyResetOptions(env, {"reward_fall_off": -0.008})   # the penalty arm
    oracle = OraclePotentialWrapper(env, beta=beta, gamma=gamma)
    rng = np.random.default_rng(seed)
    n_act = oracle.action_space.n

    max_abs_resid = 0.0
    all_absF, ep_disc, ep_negphi0 = [], [], []
    for ep in range(episodes):
        oracle.reset(seed=seed + ep)
        phi0 = oracle._prev_phi                     # Φ(s_0) captured right after reset
        disc, gp, done = 0.0, 1.0, False
        while not done:
            a = int(rng.integers(0, n_act))
            _, _, term, trunc, info = oracle.step(a)
            disc += gp * info["oracle_F"]
            gp *= gamma
            all_absF.append(info["oracle_absF"])
            done = term or trunc
        resid = abs(disc - (-phi0))                 # must telescope to −Φ_0
        max_abs_resid = max(max_abs_resid, resid)
        ep_disc.append(disc); ep_negphi0.append(-phi0)

    absF = np.asarray(all_absF)
    out = {
        "episodes": episodes, "beta": beta, "gamma": gamma,
        "max_abs_telescoping_residual": max_abs_resid,
        "telescoping_ok": bool(max_abs_resid < tol),
        "mean_absF": float(absF.mean()), "p90_absF": float(np.percentile(absF, 90)),
        "mean_disc_sum": float(np.mean(ep_disc)),
        "mean_neg_phi0": float(np.mean(ep_negphi0)),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = check(args.episodes, args.beta, args.gamma, args.seed)
    print(f"telescoping residual (max |ΣγⁿFₙ − (−Φ₀)|) = {res['max_abs_telescoping_residual']:.2e}"
          f"  → {'PASS ✓' if res['telescoping_ok'] else 'FAIL ✗'}")
    print(f"magnitude: mean|F| = {res['mean_absF']:.4f}  p90|F| = {res['p90_absF']:.4f}  "
          f"(tune β to match the e3b delivered-bonus mean within ~2×)")
    print(f"return-neutrality: mean ΣγⁿFₙ = {res['mean_disc_sum']:.4f} ≈ "
          f"mean(−Φ₀) = {res['mean_neg_phi0']:.4f}")
    if not res["telescoping_ok"]:
        raise SystemExit("oracle potential telescoping check FAILED")


if __name__ == "__main__":
    main()
