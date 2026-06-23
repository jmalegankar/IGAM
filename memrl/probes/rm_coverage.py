"""Behavioral effective-RM-size + idle-fraction — the confound-free instrument.

This is the measurement the reward-machine paper routes through, and the deliberate
opposite of `decode_memory` in one respect: it does NOT teacher-force and it does NOT
decode a frozen state (that turned out to measure copy-capacity — see the random-init
floor in decode_memory). Instead it rolls the agent's OWN (deterministic) policy and
counts what it DOES:

  * effective RM-size = number of DISTINCT realized reward-machine states the policy
    drives through per episode. Freeze → ~1 (the do-nothing absorbing state); a
    re-inflated / working policy → grows. This is purely behavioral.
  * idle-fraction = fraction of steps the policy emits the no-op / "stay" action
    (only meaningful where such an action exists, e.g. MysteryPath/S13).
  * success — for reference, so we can show RM-size co-moves with (but is a distinct
    axis from) task success.

Realized RM state per env:
  * register envs (TinyReproduce / Autoencode): the EXACT minimal-RM state is
    `info['rm_state']` (remaining-to-reproduce suffix) — distinct tuples are exact.
  * MysteryPath-Grid: no emitted RM state, so we use the behavioral proxy that
    theory_v2 names as the MysteryPath minimal-RM state — the confirmed
    path/off-path KNOWLEDGE grid; distinct knowledge configurations the rollout
    induces. Still purely behavioral (built from the agent's own trajectory).

Usage:
    python -m memrl.probes.rm_coverage --run-dir <run with snapshot_step*.pt> \
        --n-episodes 100 [--idle-action 0]
Emits one JSON line per snapshot: {step, eff_rm_size, idle_frac, success, ...}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from memrl.probes.decode_memory import (
    build_policy_from_snapshot, _unwrap, _path_set, _agent_xy, _flatten_actor_state,
)


def _det_action(policy, o, cell_state, es):
    """Deterministic (argmax) action from the policy; returns (action_np, new_state)."""
    try:
        a, _, _, cell_state, _ = policy.forward(o, cell_state, es, deterministic=True)
    except TypeError:
        a, _, _, cell_state, _ = policy.forward(o, cell_state, es)
    return a.cpu().numpy(), cell_state


@torch.no_grad()
def rollout_coverage(policy, env, n_episodes: int, idle_action: int = 0,
                     max_steps: int = 256, grid_dim: int = 7, device: str = "cpu"):
    """Roll the policy on-policy; per episode collect distinct realized RM states,
    idle-fraction, and success. Returns aggregated means + per-episode arrays."""
    raw = _unwrap(env)
    is_mp = hasattr(raw, "mystery_path")
    # SearingSpotlights has no emitted rm_state and no knowledge grid; its freeze
    # signal is MOVEMENT — a frozen agent stays at spawn, an active one roams. We use
    # distinct discretized agent positions (agent.rect.center) as the realized-coverage
    # proxy: frozen → ~1 position, active → many. (NOT the literal RM, which for SS is
    # the dead-reckoning belief; it is the behavioral did-it-freeze read.)
    is_ss = (not is_mp) and hasattr(getattr(raw, "agent", None), "rect")
    ss_bucket = max(1, int(getattr(raw, "screen_dim", 96)) // 12) if is_ss else 1
    coverage_kind = "mp_knowledge" if is_mp else ("ss_position" if is_ss else "exact_rm")

    rm_sizes, idle_fracs, succs, ep_lens = [], [], [], []
    obs = env.reset()
    cell_state = policy.initial_state(1, torch.device(device))
    first = True
    rm_states = set()
    knowledge = np.zeros((grid_dim, grid_dim), dtype=np.int8) if is_mp else None
    idle = steps = 0
    succ = 0.0
    eps_done = 0
    guard = 0
    while eps_done < n_episodes and guard < n_episodes * (max_steps + 4):
        guard += 1
        if is_mp:
            mp = _unwrap(env)
            ax, ay = _agent_xy(mp)
            path = _path_set(mp)
            if 0 <= ax < grid_dim and 0 <= ay < grid_dim:
                knowledge[ax, ay] = 1 if (ax, ay) in path else -1
            rm_states.add(knowledge.tobytes())               # realized RM-state proxy
        elif is_ss:
            cx, cy = _unwrap(env).agent.rect.center           # movement-coverage proxy
            rm_states.add((int(cx) // ss_bucket, int(cy) // ss_bucket))

        o = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
        es = torch.as_tensor([first], device=device)
        act, cell_state = _det_action(policy, o, cell_state, es)
        first = False
        # idle = the no-op: a scalar Discrete action == idle_action, or ALL components
        # of a MultiDiscrete action == idle_action (no movement/rotation).
        idle += int(bool(np.all(np.asarray(act[0]) == idle_action)))
        steps += 1

        obs, _, dones, infos = env.step(act)
        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
        if not is_mp and not is_ss:
            rm_states.add(tuple(info0.get("rm_state", ())))   # exact RM state
        succ = max(succ, float(info0.get("is_success", 0.0)) if "is_success" in info0
                   else float(info0.get("success", 0.0)))

        if bool(dones[0]) or steps >= max_steps:
            rm_sizes.append(len(rm_states - {b""} if is_mp else rm_states))
            idle_fracs.append(idle / max(1, steps))
            succs.append(succ); ep_lens.append(steps)
            # reset per-episode accumulators
            rm_states = set(); idle = steps = 0; succ = 0.0; first = True
            if knowledge is not None:
                knowledge = np.zeros((grid_dim, grid_dim), dtype=np.int8)
            cell_state = policy.initial_state(1, torch.device(device))
            eps_done += 1

    def m(a): return float(np.mean(a)) if a else None
    return {"eff_rm_size": m(rm_sizes), "eff_rm_size_std": (float(np.std(rm_sizes)) if rm_sizes else None),
            "idle_frac": m(idle_fracs), "success": m(succs), "ep_len": m(ep_lens),
            "n_episodes": len(rm_sizes), "coverage_kind": coverage_kind,
            "is_mysterypath": bool(is_mp)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--snapshots", default=None, help="comma-separated milestones (default: all)")
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--idle-action", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run = Path(args.run_dir)
    snaps = sorted(run.glob("snapshot_step*.pt"), key=lambda p: int(p.stem.split("step")[1]))
    if args.snapshots:
        want = {int(s) for s in args.snapshots.split(",")}
        snaps = [p for p in snaps if int(p.stem.split("step")[1]) in want]
    if not snaps:
        raise SystemExit(f"no snapshots in {run}")

    for sp in snaps:
        step = int(sp.stem.split("step")[1])
        policy, env, cfg = build_policy_from_snapshot(str(sp), device=args.device)
        res = rollout_coverage(policy, env, n_episodes=args.n_episodes,
                               idle_action=args.idle_action, device=args.device)
        print(json.dumps({"run": run.name, "step": step, "metric": "rm_coverage", **res}))


if __name__ == "__main__":
    main()
