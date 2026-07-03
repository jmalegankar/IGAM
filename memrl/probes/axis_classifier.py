"""I-5 — the random-policy PRM-advancement classifier (the AXIS definition, results-free).

This is the paper's answer to the circularity attack (A2, handoff §13) and the
operational counterpart of the RM-1 formal definition (§12): the acquisition/retention
label is assigned from ENVIRONMENT STRUCTURE, measured under a uniform-random policy,
BEFORE any results — never post-hoc from which cell won.

Method (no policy, no checkpoint — just the env + uniform-random actions):
  For each env we fix a natural labeling (the PRM-transition triggers) and, under the
  uniform-random policy, measure for each labeled transition class the probability its
  trigger event occurs within the episode budget, plus the time-to-first-occurrence.

  Definition (RM-1). A PRM transition is EXOGENOUS if its trigger fires w.h.p. under any
  policy (P → 1); AGENT-CONTINGENT if it requires the agent to cause it (P → 0 under
  random). The separation is measured, not assumed — it comes out orders of magnitude
  apart, so the exogenous≥0.9 / contingent≤0.1 cut is reported alongside the raw P.

  ACQUISITION regime: the PRM's path to ACCEPT needs agent-contingent transitions whose
  post-transition state is not predictable at t=0 (MysteryPath: the invisible path is
  re-drawn each episode → advancing it is agent-contingent AND unpredictable).
  RETENTION regime: the PRM state is set by exogenous events and merely carried
  (TinyReproduce: the register is loaded by exogenous token reveals; the agent only has
  to hold it — the play-phase emits are agent-contingent for ACCEPT but do not set state).

Validation leg (the classifier must not be trusted un-checked): TinyReproduce exposes the
EXACT minimal-RM state in info["rm_state"]. We reconstruct the PRM trajectory purely from
the observable labeled events (the shown-token one-hots + the emit actions under the known
order) and assert it equals info["rm_state"] step-for-step across many random episodes.
Any mismatch is a labeling bug — fix before trusting MysteryPath/S13 (which have no
exposed ground truth).

Output: one row per env × view for the Setup-section agent-contingency table —
fraction of PRM transition classes that are agent-contingent + the random-policy
time-to-event distributions.

Usage:
    python -m memrl.probes.axis_classifier --env both --n-episodes 300
    python -m memrl.probes.axis_classifier --env tiny --k 10 --v 4   # validation leg only
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from memrl.envs import make_vec_env
from memrl.probes.decode_memory import _unwrap, _path_set, _agent_xy


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────
def _n_actions(vec_env) -> int:
    sp = vec_env.action_space
    return int(sp.n) if hasattr(sp, "n") else int(np.prod(sp.nvec))


def _classify(p: float, exo: float = 0.9, contingent: float = 0.1) -> str:
    if p >= exo:
        return "exogenous"
    if p <= contingent:
        return "agent-contingent"
    return "intermediate"


def _summ_event(occurred: list[bool], first_t: list[float], budget: int) -> dict:
    """Summarize one labeled-event class over episodes: P(within budget) + TTFE dist."""
    occ = np.asarray(occurred, dtype=float)
    p = float(occ.mean()) if occ.size else float("nan")
    tt = np.asarray([t for t in first_t if t is not None], dtype=float)
    return {"p_within_budget": round(p, 4), "classification": _classify(p),
            "n": int(occ.size), "budget": budget,
            "ttfe_mean": (round(float(tt.mean()), 2) if tt.size else None),
            "ttfe_p10": (round(float(np.percentile(tt, 10)), 2) if tt.size else None),
            "ttfe_p90": (round(float(np.percentile(tt, 90)), 2) if tt.size else None)}


# ──────────────────────────────────────────────────────────────────────────
# TinyReproduce — retention regime + the EXACT validation leg
# ──────────────────────────────────────────────────────────────────────────
def classify_tiny(k: int = 10, v: int = 4, order: str = "reverse",
                  n_episodes: int = 300, seed: int = 0) -> dict:
    env = make_vec_env("TinyReproduce-v0", n_envs=1, seed=seed,
                       k=k, v=v, order=order, density="sparse")
    rng = np.random.default_rng(seed)
    budget = 2 * k - 1

    def target(seq, play_idx):
        return seq[play_idx] if order == "inorder" else seq[k - 1 - play_idx]

    def remaining(seq, play_idx):        # env._remaining(): suffix still to reproduce
        return (tuple(seq[play_idx:]) if order == "inorder"
                else tuple(seq[: k - play_idx][::-1]))

    # Phase is fixed by the step index t (1-based within the episode), NOT the obs —
    # the obs lags the env's internal `shown` by one step. Watch = steps 1..k-1
    # (reveal seq[t], exogenous); play = steps k..2k-1 (emit seq position t-k). The env's
    # info["rm_state"] means the SHOWN PREFIX during watch and the REMAINING SUFFIX during
    # play; we reconstruct the matching object per phase and assert it equals ground truth.
    shown_all, accept_all, first_accept_t = [], [], []
    correct_emits, total_emits = 0, 0
    mismatches = checked_steps = 0

    obs = env.reset()
    seq = [int(np.argmax(np.asarray(obs)[0][2:]))]      # seq[0] from the reset watch obs
    t = 0
    eps = 0
    guard = 0
    while eps < n_episodes and guard < n_episodes * (budget + 4):
        guard += 1
        t += 1
        is_watch_step = t <= k - 1
        a = int(rng.integers(0, v))
        if not is_watch_step:                           # PLAY: emit position (t-k)
            pos = t - k
            correct = (a == target(seq, pos))
            total_emits += 1
            correct_emits += int(correct)
        obs, _, dones, infos = env.step(np.asarray([a]))
        info = infos[0]
        o1 = np.asarray(obs)[0]
        done = bool(dones[0])

        if is_watch_step and not done:                  # env revealed seq[t] → o1 shows it
            seq.append(int(np.argmax(o1[2:])))
        if not done:                                    # compare per-phase reconstruction
            recon = tuple(seq) if is_watch_step else remaining(seq, (t - k) + 1)
            gt = tuple(info.get("rm_state", ()))
            checked_steps += 1
            mismatches += int(recon != gt)

        if done:
            shown_all.append(len(seq) == k)             # register loaded fully (exogenous)
            accepted = bool(info.get("is_success", 0.0))
            accept_all.append(accepted)
            first_accept_t.append(t if accepted else None)
            eps += 1
            seq = [int(np.argmax(o1[2:]))]              # new episode's reset watch obs
            t = 0

    p_emit_correct = correct_emits / max(1, total_emits)
    transitions = {
        "reveal(token)  [watch → load register]": _summ_event(
            shown_all, [1.0] * len(shown_all), budget),
        "accept  [reproduce all k]": _summ_event(
            accept_all, first_accept_t, budget),
    }
    n_contingent = sum(1 for v_ in transitions.values()
                       if v_["classification"] == "agent-contingent")
    return {
        "env": "TinyReproduce", "view": f"k{k}v{v}", "regime": "retention",
        "transitions": transitions,
        "frac_agent_contingent": round(n_contingent / len(transitions), 3),
        "p_random_correct_emit": round(p_emit_correct, 4),
        "p_random_correct_emit_expected": round(1.0 / v, 4),
        "validation": {"checked_steps": checked_steps, "rm_state_mismatches": mismatches,
                       "exact_match": mismatches == 0},
    }


# ──────────────────────────────────────────────────────────────────────────
# MysteryPath — acquisition regime
# ──────────────────────────────────────────────────────────────────────────
def classify_mysterypath(n_episodes: int = 300, seed: int = 0,
                         max_steps: int = 128, grid_dim: int = 7) -> dict:
    env = make_vec_env("MysteryPath-Grid-v0", n_envs=1, seed=seed)
    rng = np.random.default_rng(seed)
    n_act = _n_actions(env)

    goal_occ, goal_t = [], []
    fall_occ, fall_t = [], []          # "at least one fall" event
    advance_occ, advance_t = [], []    # "advance to a NEW on-path tile" (beyond start)
    path_frac = []                     # fraction of the hidden path random discovers

    obs = env.reset()
    mp = _unwrap(env)
    path = _path_set(mp)
    start = _agent_xy(mp)
    on_path_seen = {start}
    prev_fails = int(mp.num_fails)
    first_fall = first_adv = None
    t = 0
    eps = 0
    guard = 0
    while eps < n_episodes and guard < n_episodes * (max_steps + 4):
        guard += 1
        a = int(rng.integers(0, n_act))
        obs, _, dones, infos = env.step(np.asarray([a]))
        info = infos[0]
        t += 1
        done = bool(dones[0])
        reached_goal = bool(info.get("success", 0.0)) if done else False
        if not done:
            mp = _unwrap(env)
            cur_fails = int(mp.num_fails)
            if cur_fails > prev_fails and first_fall is None:
                first_fall = t
            prev_fails = cur_fails
            xy = _agent_xy(mp)
            if xy in path and xy not in on_path_seen and xy != start:
                on_path_seen.add(xy)
                if first_adv is None:
                    first_adv = t

        if done:
            goal_occ.append(reached_goal); goal_t.append(t if reached_goal else None)
            fall_occ.append(first_fall is not None); fall_t.append(first_fall)
            advance_occ.append(first_adv is not None); advance_t.append(first_adv)
            # discovered on-path tiles beyond start / total non-start path tiles
            denom = max(1, len(path) - 1)
            path_frac.append((len(on_path_seen) - 1) / denom)
            eps += 1
            mp = _unwrap(env)                # new episode (auto-reset)
            path = _path_set(mp); start = _agent_xy(mp)
            on_path_seen = {start}; prev_fails = int(mp.num_fails)
            first_fall = first_adv = None
            t = 0

    transitions = {
        "fall(tile)  [step off path]": _summ_event(fall_occ, fall_t, max_steps),
        "advance(tile)  [reach a NEW on-path tile]": _summ_event(advance_occ, advance_t, max_steps),
        "goal  [reach end via the hidden path]": _summ_event(goal_occ, goal_t, max_steps),
    }
    n_contingent = sum(1 for v_ in transitions.values()
                       if v_["classification"] == "agent-contingent")
    return {
        "env": "MysteryPath", "view": "grid7", "regime": "acquisition",
        "transitions": transitions,
        "frac_agent_contingent": round(n_contingent / len(transitions), 3),
        "path_fraction_discovered_random": round(float(np.mean(path_frac)), 4),
        "mean_path_len": round(float(np.mean([len(_path_set(mp))])), 1),
    }


# ──────────────────────────────────────────────────────────────────────────
# MiniGrid-MemoryS13 — retention regime with a VIEW-DEPENDENT residual acquisition
# demand (the cue-exposure dose-response; also the geometry I-4 reuses under a
# trained policy). The cue (start-room object) sits at (1, h//2-1); the agent must
# ORIENT/move to bring it into the egocentric view. Smaller view ⇒ more agent-
# contingent cue exposure ⇒ more residual acquisition on an otherwise-retention task.
# ──────────────────────────────────────────────────────────────────────────
def _cue_visible(u, cue_xy: tuple[int, int], view: int) -> bool:
    """Is the cue cell inside the agent's `view`×`view` egocentric window (with wall
    occlusion), computed at an EXPLICIT view size — ViewSizeWrapper keeps the reduced
    size on the wrapper, not on `u` (unwrapped), so we must not rely on u.agent_view_size."""
    ax, ay = u.agent_pos
    dx, dy = u.dir_vec
    rx, ry = u.right_vec
    hs = view // 2
    tx = ax + dx * (view - 1) - rx * hs          # top-left view corner (mirrors get_view_coords)
    ty = ay + dy * (view - 1) - ry * hs
    lx, ly = cue_xy[0] - tx, cue_xy[1] - ty
    vx = rx * lx + ry * ly
    vy = -(dx * lx + dy * ly)
    if not (0 <= vx < view and 0 <= vy < view):
        return False
    try:                                          # wall occlusion via the obs vis-mask at size V
        _, vis = u.gen_obs_grid(view)
        return bool(vis[vx, vy])
    except Exception:
        return True                               # geometric in-view (no occlusion info)


def classify_s13(view: int = 3, n_episodes: int = 200, seed: int = 0) -> dict:
    env = make_vec_env("MiniGrid-MemoryS13-v0", n_envs=1, seed=seed,
                       agent_view_size=view, reward_mode="flat")
    rng = np.random.default_rng(seed)
    n_act = _n_actions(env)
    env.reset()
    u = _unwrap(env)
    max_steps = int(getattr(u, "max_steps", 5 * 13 ** 2))
    cue = (1, int(u.height) // 2 - 1)

    cue_occ, cue_t = [], []                        # cue-enters-view: the state-SETTING event
    succ_occ, succ_t = [], []                      # reach the matching object (accept)
    seen_this_ep = False
    first_seen = None
    t = 0
    eps = 0
    guard = 0
    # check visibility at t=0 too (agent may spawn already seeing the cue)
    if _cue_visible(u, cue, view):
        seen_this_ep, first_seen = True, 0
    while eps < n_episodes and guard < n_episodes * (max_steps + 4):
        guard += 1
        a = int(rng.integers(0, n_act))
        _, _, dones, infos = env.step(np.asarray([a]))
        info = infos[0]
        t += 1
        done = bool(dones[0])
        u = _unwrap(env)
        if not done:
            if not seen_this_ep and _cue_visible(u, cue, view):
                seen_this_ep, first_seen = True, t
        if done:
            cue_occ.append(seen_this_ep); cue_t.append(first_seen)
            accepted = bool(info.get("is_success", 0.0))
            succ_occ.append(accepted); succ_t.append(t if accepted else None)
            eps += 1
            u = _unwrap(env)                       # new episode (auto-reset)
            cue = (1, int(u.height) // 2 - 1)
            seen_this_ep = _cue_visible(u, cue, view)
            first_seen = 0 if seen_this_ep else None
            t = 0
    transitions = {
        f"cue_seen(view{view})  [enters egocentric view]": _summ_event(cue_occ, cue_t, max_steps),
        "accept  [reach matching object]": _summ_event(succ_occ, succ_t, max_steps),
    }
    n_contingent = sum(1 for v_ in transitions.values()
                       if v_["classification"] == "agent-contingent")
    return {"env": "MiniGrid-MemoryS13", "view": f"view{view}", "regime": "retention",
            "transitions": transitions,
            "frac_agent_contingent": round(n_contingent / len(transitions), 3),
            "note": "cue_seen exogeneity DECREASES with view size = the residual "
                    "acquisition demand (H-KNOB-B dose-response)"}


# ──────────────────────────────────────────────────────────────────────────
def _print_table(rows: list[dict]) -> None:
    print("\n=== I-5 agent-contingency table (uniform-random policy) ===")
    for r in rows:
        print(f"\n{r['env']} [{r['view']}]  regime={r['regime']}  "
              f"frac_agent_contingent={r['frac_agent_contingent']}")
        for name, s in r["transitions"].items():
            print(f"    {name:<44} P={s['p_within_budget']:<7} {s['classification']:<17} "
                  f"TTFE(mean/p10/p90)={s['ttfe_mean']}/{s['ttfe_p10']}/{s['ttfe_p90']}")
        if "validation" in r:
            val = r["validation"]
            print(f"    VALIDATION vs info['rm_state']: {val['checked_steps']} steps, "
                  f"{val['rm_state_mismatches']} mismatches → "
                  f"{'EXACT MATCH ✓' if val['exact_match'] else 'MISMATCH ✗'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="both", choices=["both", "tiny", "mysterypath", "s13"])
    ap.add_argument("--views", default="7,5,3", help="s13: comma view sizes to sweep")
    ap.add_argument("--n-episodes", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--v", type=int, default=4)
    ap.add_argument("--order", default="reverse", choices=["reverse", "inorder"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the JSON rows here")
    args = ap.parse_args()

    rows = []
    if args.env in ("both", "tiny"):
        rows.append(classify_tiny(k=args.k, v=args.v, order=args.order,
                                  n_episodes=args.n_episodes, seed=args.seed))
    if args.env in ("both", "mysterypath"):
        rows.append(classify_mysterypath(n_episodes=args.n_episodes, seed=args.seed))
    if args.env == "s13":
        for v in (int(x) for x in args.views.split(",")):
            rows.append(classify_s13(view=v, n_episodes=args.n_episodes, seed=args.seed))

    _print_table(rows)
    for r in rows:
        print("\n" + json.dumps(r))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.out}")
    # the validation leg is a hard gate — surface a nonzero exit if it fails
    for r in rows:
        if r.get("validation") and not r["validation"]["exact_match"]:
            raise SystemExit("I-5 validation FAILED: rm_state reconstruction mismatch")


if __name__ == "__main__":
    main()
