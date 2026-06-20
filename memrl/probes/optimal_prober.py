"""Optimal-prober DP — the K4 falsifier for the freeze fixed point.

The freeze claim is only a *pathology* if doing nothing is STRICTLY sub-optimal under
the SPECIFIED reward at the −0.008 fall penalty. We prove that here with an exact DP on
a tractable corridor-POMDP abstraction of a MysteryPath-style hidden-path task, and we
locate the penalty p̄ at which freezing first becomes optimal — showing −0.008 ≪ p̄.

Model (corridor POMDP). The hidden path is a sequence of `L` stages; at each stage the
agent faces `b` candidate moves, exactly one of which is on-path. The agent cannot see
which until it tries: a wrong move is a FALL (reward = `fall_penalty`, stays at the
stage with that branch eliminated), a right move advances. Reaching stage L pays
`goal_reward`. "Do nothing" stays forever for a return of 0 (the absorbing
zero-penalty sanctuary). The optimal policy probes (it will always reach the goal,
incurring some falls), so it is the right reference for "is freezing rational?".

Exact DP over states (stage, r = #candidate branches still untried at this stage):
    V(L)        = goal_reward
    V(stage,r)  = max( 0,                                   # freeze (sanctuary)
                       (1/r)·γ·V(stage+1,b)                  # try → right (prob 1/r)
                     + ((r−1)/r)·(fall_penalty + γ·V(stage,r−1)) )  # try → fall
The optimal return is V* = V(0,b). Freezing is strictly sub-optimal iff V* > 0.
E[falls|π*] is read off the same recursion; the P1 bound is ε = |fall_penalty|·E[falls].
"""

from __future__ import annotations

import argparse
import json


def solve(L: int, b: int, fall_penalty: float, goal_reward: float = 1.0,
          gamma: float = 1.0) -> dict:
    """Exact DP. Returns V*, the freeze-optimal flag, E[falls|π*], and the P1 ε."""
    # V[stage][r]: r in 1..b ; memoized bottom-up.
    Vgoal = goal_reward
    # value + expected-falls tables, indexed [stage][r]
    V = [[0.0] * (b + 1) for _ in range(L + 1)]
    F = [[0.0] * (b + 1) for _ in range(L + 1)]      # E[falls | act optimally from here]
    froze = [[False] * (b + 1) for _ in range(L + 1)]
    for r in range(1, b + 1):
        V[L][r] = Vgoal                               # already at goal
    for stage in range(L - 1, -1, -1):
        for r in range(1, b + 1):
            v_next = V[stage + 1][b]
            f_next = F[stage + 1][b]
            v_fall_branch = V[stage][r - 1] if r >= 2 else 0.0   # r=1 → must be right
            f_fall_branch = F[stage][r - 1] if r >= 2 else 0.0
            p_right = 1.0 / r
            v_try = (p_right * gamma * v_next
                     + (1 - p_right) * (fall_penalty + gamma * v_fall_branch))
            f_try = (p_right * f_next
                     + (1 - p_right) * (1.0 + f_fall_branch))
            if v_try >= 0.0:                           # probe beats freezing
                V[stage][r] = v_try; F[stage][r] = f_try; froze[stage][r] = False
            else:
                V[stage][r] = 0.0; F[stage][r] = 0.0; froze[stage][r] = True
    vstar = V[0][b]
    return {"L": L, "b": b, "fall_penalty": fall_penalty, "goal_reward": goal_reward,
            "gamma": gamma, "V_star": vstar, "do_nothing_return": 0.0,
            "freeze_is_optimal": vstar <= 0.0, "E_falls_opt": F[0][b],
            "P1_epsilon": abs(fall_penalty) * F[0][b],
            "optimality_gap": vstar - 0.0}


def freeze_threshold(L: int, b: int, goal_reward: float = 1.0, gamma: float = 1.0,
                     lo: float = -2.0, hi: float = 0.0, tol: float = 1e-6) -> float:
    """Penalty p̄ (negative) at which V* first hits 0 — freezing becomes optimal at p ≤ p̄."""
    # V* increases with fall_penalty (less negative → higher V*). Invariant: hi is a
    # penalty where probing still wins (V*>0), lo is one where freezing wins (V*≤0);
    # the boundary p̄ is between. hi=0 (V*=goal>0), lo=-2 (V*=0). Converge to p̄.
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if solve(L, b, mid, goal_reward, gamma)["V_star"] > 0.0:
            hi = mid                       # probing still wins here → boundary is below
        else:
            lo = mid                       # freezing wins here → boundary is above
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--L", type=int, default=20, help="path length (MysteryPath ~20)")
    ap.add_argument("--b", type=int, default=4, help="branches per stage")
    ap.add_argument("--fall-penalty", type=float, default=-0.008)
    ap.add_argument("--goal-reward", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    args = ap.parse_args()
    res = solve(args.L, args.b, args.fall_penalty, args.goal_reward, args.gamma)
    res["freeze_threshold_pbar"] = freeze_threshold(args.L, args.b, args.goal_reward, args.gamma)
    res["penalty_vs_threshold"] = f"{args.fall_penalty} vs p̄={res['freeze_threshold_pbar']:.4f}"
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
