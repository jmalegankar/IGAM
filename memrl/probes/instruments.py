"""H-INST — behavioral instrument batch for MysteryPath checkpoints (no retraining).

Where `decode_memory.py` asks *"is the task-memory decodable from the frozen
recurrent state?"* (a representational read, matched-coverage random bank), this
module asks *"what does the trained policy actually DO?"* (a behavioral read): it
rolls each checkpoint's OWN policy and measures the mechanism + freeze signature.

Instruments (handoff §2), all from re-rolled checkpoints, ≥N episodes, BOTH eval
modes (I-6):

  I-1  repeat_fall_fraction  — of the localized falls in an episode, the fraction
        landing on a tile the agent had ALREADY fallen on earlier this episode.
        High = the memory is not retaining which tiles are off-path (the mechanism
        e3b is predicted to fix; Memoryless+e3b predicted highest).
  I-2  frontier_probe_rate   — of position-changing moves, the fraction onto a tile
        not yet occupied this episode (frontier) vs re-traversal of known tiles.
        Coverage → frontier conversion; complements I-1.
  I-3  distinct_tiles        — distinct tiles occupied in the episode. The FREEZE
        signature that replaces "do-nothing": frozen runs collapse distinct_tiles
        (falls→0, success 0, elevated no-op) — quotable against the residual idle.
  falls / success / action histogram / frac_a0 (the literal no-op) — logged for
        every episode so the freeze signature is one row, not a re-derivation.

MysteryPath-Grid semantics this relies on (verified against the installed env):
  * Discrete(4): 0=no-op, 1=rotate-left, 2=rotate-right, 3=move-forward. Position
    changes ONLY on action 3. So frac_a0 (and non-move fraction) is a real idle read.
  * A FALL = moving onto an off-path tile: env increments `num_fails`, sets
    `is_off_path=True`, leaves the agent ON the off-path tile THIS step; the NEXT
    step force-teleports it to `start` (action ignored). So a per-step `num_fails`
    increment localizes the fall tile as `normalized_agent_position` — except on the
    terminal step (SB3 auto-reset hides it), where we fall back to info["num_fails"].
  * success on reaching `end`; a fall never ends the episode (only goal / max_steps).

Usage:
    python -m memrl.probes.instruments --run-dir runs/MysteryPath/RetNet-sparse-e3b_idm-seed0 \
        --snapshots 500000,2000000,5000000,10000000,20000000 --n-episodes 100 --out inst.csv
    python -m memrl.probes.instruments --smoke        # no checkpoint: validate the loggers

Emits one tidy CSV row per (snapshot, eval_mode, episode); aggregation lives in
notebooks. Metadata (cell/arm/bonus/seed) is read from the snapshot's embedded config.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from memrl.probes.decode_memory import (
    build_policy_from_snapshot, _unwrap, _path_set, _agent_xy,
)


# ──────────────────────────────────────────────────────────────────────────
# Metadata (cell / arm / bonus / seed) from the run config
# ──────────────────────────────────────────────────────────────────────────
def _arm_from_cfg(cfg: dict) -> str:
    """Infer the MysteryPath reward arm from the reset_options (the density label
    is not stored in the config, but the reward keys that define the arm are)."""
    ro = (cfg.get("env_kwargs", {}) or {}).get("reset_options", {}) or {}
    if float(ro.get("reward_fall_off", 0.0)) < 0.0:
        return "penalty"
    if float(ro.get("reward_path_progress", 0.0)) > 0.0:
        return "aligned"
    return "sparse"


def _meta(cfg: dict) -> dict:
    return {"cell": cfg["cell"]["name"], "arm": _arm_from_cfg(cfg),
            "bonus": cfg.get("intrinsic", "none"), "seed": int(cfg.get("seed", 0))}


# ──────────────────────────────────────────────────────────────────────────
# Behavioral rollout of the checkpoint's OWN policy
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def rollout(policy, vec_env, n_episodes: int, deterministic: bool,
            device: str = "cpu", max_steps: int = 128) -> list[dict]:
    """Roll `policy` on a single-env MysteryPath vec_env for n_episodes; return one
    trace dict per episode with the raw counts the instruments need. Env state is
    introspected each NON-terminal step (num_fails / position / is_off_path); the
    terminal step falls back to info (SB3 auto-reset hides its post-step state)."""
    episodes: list[dict] = []
    obs = vec_env.reset()
    mp = _unwrap(vec_env)
    guard, guard_max = 0, n_episodes * (max_steps + 8) + 200

    while len(episodes) < n_episodes and guard < guard_max:
        path = _path_set(mp)
        start = _agent_xy(mp)
        occupied = {start}
        fallen: set = set()
        localized_falls = repeat_falls = 0
        moves = frontier_moves = 0
        action_counts = [0, 0, 0, 0]
        prev_fails = int(mp.num_fails)
        length = 0
        success = 0
        num_fails_info = None
        cell_state = policy.initial_state(1, torch.device(device))
        first = True
        done = False

        while not done and guard < guard_max:
            guard += 1
            # is the agent entering this step off-path? then the env IGNORES the
            # action and force-resets to start — exclude it from move/probe accounting.
            is_reset_step = bool(getattr(mp, "is_off_path", False))
            o = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
            es = torch.as_tensor(np.asarray([first], dtype=bool), device=device)
            action_t, _, _, cell_state, _ = policy.forward(
                o, cell_state, es, deterministic=deterministic)
            a = int(np.asarray(action_t.cpu()).reshape(-1)[0])
            first = False
            if 0 <= a < 4:
                action_counts[a] += 1

            pos_before = _agent_xy(mp)
            obs, _, dones, infos = vec_env.step(np.asarray([a]))
            length += 1
            done = bool(dones[0])
            if done:
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                success = int(info.get("success", 0))
                num_fails_info = int(info.get("num_fails", prev_fails))
                break

            mp = _unwrap(vec_env)
            cur_fails = int(mp.num_fails)
            pos_after = _agent_xy(mp)
            if cur_fails > prev_fails:                 # a fall happened THIS step
                localized_falls += 1
                if pos_after in fallen:
                    repeat_falls += 1
                fallen.add(pos_after)
            if not is_reset_step and pos_after != pos_before:
                moves += 1
                if pos_after not in occupied:
                    frontier_moves += 1
            occupied.add(pos_after)
            prev_fails = cur_fails

        num_fails = num_fails_info if num_fails_info is not None else prev_fails
        episodes.append({
            "length": length, "success": success, "num_fails": num_fails,
            "distinct_tiles": len(occupied),               # I-3
            "localized_falls": localized_falls,
            "repeat_falls": repeat_falls,                  # I-1 numerator
            "moves": moves, "frontier_moves": frontier_moves,  # I-2
            "action_counts": action_counts,
            "path_len": len(path),
        })
        # next episode: obs already holds the auto-reset initial obs; re-unwrap env
        mp = _unwrap(vec_env)

    return episodes


def _episode_row(ep: dict) -> dict:
    """Derive the per-episode instrument metrics from a trace dict."""
    L = max(1, ep["length"])
    ac = ep["action_counts"]
    tot_a = max(1, sum(ac))
    lf = ep["localized_falls"]
    mv = ep["moves"]
    return {
        "length": ep["length"],
        "success": ep["success"],
        "num_fails": ep["num_fails"],
        "distinct_tiles": ep["distinct_tiles"],                      # I-3
        "repeat_fall_fraction": (ep["repeat_falls"] / lf) if lf > 0 else "",  # I-1
        "frontier_probe_rate": (ep["frontier_moves"] / mv) if mv > 0 else "",  # I-2
        "frac_a0": ac[0] / tot_a,                                    # literal no-op
        "frac_a1": ac[1] / tot_a,
        "frac_a2": ac[2] / tot_a,
        "frac_a3": ac[3] / tot_a,                                    # move-forward
        "moves": mv, "frontier_moves": ep["frontier_moves"],
        "localized_falls": lf, "repeat_falls": ep["repeat_falls"],
    }


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────
EVAL_MODES = {"deterministic": True, "stochastic": False}
FIELDS = ["run", "step", "eval_mode", "episode", "cell", "arm", "bonus", "seed",
          "length", "success", "num_fails", "distinct_tiles",
          "repeat_fall_fraction", "frontier_probe_rate",
          "frac_a0", "frac_a1", "frac_a2", "frac_a3",
          "moves", "frontier_moves", "localized_falls", "repeat_falls"]


def run_snapshot(snapshot_path: str, run_name: str, n_episodes: int,
                 eval_modes: list[str], device: str, writer) -> list[dict]:
    policy, env, cfg = build_policy_from_snapshot(snapshot_path, device=device)
    meta = _meta(cfg)
    step = int(Path(snapshot_path).stem.split("step")[1]) if "step" in snapshot_path else -1
    max_steps = int((cfg.get("env_kwargs", {}) or {}).get(
        "reset_options", {}).get("max_steps", 128))
    summaries = []
    for mode in eval_modes:
        eps = rollout(policy, env, n_episodes, deterministic=EVAL_MODES[mode],
                      device=device, max_steps=max_steps)
        succ = np.mean([e["success"] for e in eps]) if eps else float("nan")
        dt = np.mean([e["distinct_tiles"] for e in eps]) if eps else float("nan")
        nf = np.mean([e["num_fails"] for e in eps]) if eps else float("nan")
        a0 = np.mean([e["action_counts"][0] / max(1, sum(e["action_counts"]))
                      for e in eps]) if eps else float("nan")
        summaries.append({"step": step, "mode": mode, "success": succ,
                          "distinct_tiles": dt, "num_fails": nf, "frac_a0": a0})
        for i, ep in enumerate(eps):
            row = {"run": run_name, "step": step, "eval_mode": mode, "episode": i, **meta}
            row.update(_episode_row(ep))
            writer.writerow(row)
    return summaries


def _build_fresh_policy(env_name: str, cell_name: str, device: str = "cpu"):
    """Build an UNTRAINED policy (no snapshot) — for --smoke validation of the loggers."""
    from train import make_cell_factory
    from memrl.envs import make_vec_env
    from memrl.ppo import MemPPO
    from train import DEFAULT_CELL_KWARGS
    env = make_vec_env(env_name=env_name, n_envs=1, seed=123)
    factory = make_cell_factory(cell_name=cell_name,
                                cell_kwargs=dict(DEFAULT_CELL_KWARGS[cell_name]),
                                hidden_size=128)
    model = MemPPO(env=env, cell_factory=factory, lr=1e-4, n_steps=128, n_epochs=1,
                   encoder_dim=128, encoder_hidden=256, chunk_len=32,
                   n_chunks_per_batch=8, intrinsic_module=None, lambda_intrinsic=0.0,
                   verbose=0, seed=0, device=device)
    model.policy.set_training_mode(False)
    return model.policy, env


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", help="dir containing snapshot_step*.pt")
    ap.add_argument("--snapshots", default=None,
                    help="comma-separated milestones to eval (default: all found)")
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--eval-modes", default="deterministic,stochastic",
                    help="comma list from {deterministic,stochastic} (I-6)")
    ap.add_argument("--out", default="instruments.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="no checkpoint: build a fresh policy, run 3 episodes, print rows")
    args = ap.parse_args()
    eval_modes = [m.strip() for m in args.eval_modes.split(",") if m.strip() in EVAL_MODES]

    if args.smoke:
        policy, env = _build_fresh_policy("MysteryPath-Grid-v0", "GRU", args.device)
        eps = rollout(policy, env, n_episodes=3, deterministic=True,
                      device=args.device, max_steps=128)
        print(f"[smoke] rolled {len(eps)} episodes on MysteryPath-Grid (untrained GRU)")
        for i, ep in enumerate(eps):
            print(f"  ep{i}: {_episode_row(ep)}")
        eps_s = rollout(policy, env, n_episodes=2, deterministic=False,
                        device=args.device, max_steps=128)
        print(f"[smoke] stochastic mode OK ({len(eps_s)} eps); action_counts "
              f"ep0={eps_s[0]['action_counts'] if eps_s else 'n/a'}")
        return

    run = Path(args.run_dir)
    snaps = sorted(run.glob("snapshot_step*.pt"),
                   key=lambda p: int(p.stem.split("step")[1]))
    if args.snapshots:
        want = {int(s) for s in args.snapshots.split(",")}
        snaps = [p for p in snaps if int(p.stem.split("step")[1]) in want]
    if not snaps:
        raise SystemExit(f"no snapshots in {run} (train with --snapshot-steps ...)")

    out = Path(args.out)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for sp in snaps:
            summ = run_snapshot(str(sp), run.name, args.n_episodes, eval_modes,
                                args.device, writer)
            for s in summ:
                print(f"[{run.name}] step={s['step']:>9} {s['mode']:<13} "
                      f"succ={s['success']:.3f} distinct_tiles={s['distinct_tiles']:.1f} "
                      f"num_fails={s['num_fails']:.1f} frac_a0={s['frac_a0']:.3f}",
                      file=sys.stderr)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
