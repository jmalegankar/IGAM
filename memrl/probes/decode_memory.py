"""Memory-decodability probe for MysteryPath-Grid (paper claim C1).

Question
--------
Does an exploration bonus *train the recurrent memory*? We answer it directly:
freeze a policy checkpoint, and ask how linearly-decodable the task-relevant
memory — the agent's episodic knowledge of which tiles are path vs off-path — is
from its frozen recurrent state, as a function of training step and arm.

Predicted curves (the paper's centerpiece figure)
-------------------------------------------------
    sparse-none      : decodability rises slowly
    sparse-e3b       : decodability rises fast        (the 3×, mechanistically)
    penalty-none     : decodability FLAT at chance    (empty statistic = freeze)
    penalty-e3b      : decodability recovers          (rescue = re-trained memory)
    S13 (retention)  : none ≈ e3b                     (front-loaded revelation:
                                                        no behavioral training work)

The matched-coverage control (critical)
---------------------------------------
Decodability could rise simply because a better policy *visits more tiles*. To
remove that confound we score every arm on the SAME held-out trajectories,
generated once by a single fixed BEHAVIOR policy (default: uniform-random). Each
checkpoint's recurrent net merely *replays* those observation sequences to
produce h_t; the latent labels and state coverage are identical across arms, so
any decodability gap is representational, not behavioral. (Use --behavior policy
with a reference checkpoint for a realistic-coverage sensitivity check.)

What is decoded
---------------
For each step we build the agent's *confirmable* knowledge grid (7×7):
    +1  tile known to be ON path   (agent stepped there without falling)
    -1  tile known to be OFF path  (agent fell there)
     0  tile not yet visited this episode
We probe two targets:
    (K) the running KNOWLEDGE grid  — what the memory has actually observed
    (P) the full ground-truth PATH  — tests extrapolation beyond what's seen
Per-cell balanced accuracy + macro-AUC are reported; the headline metric is the
mean over visited cells of (K) — "how much of what it has seen does the memory
still hold?"

Usage
-----
    python -m memrl.probes.decode_memory \
        --run-dir runs/MysteryPath/RetNet-sparse-e3b_idm-seed0 \
        --snapshots 500000,2000000,5000000,10000000 \
        --behavior random --n-episodes 200 --probe both

Point --run-dir at each arm's run; collect the printed JSON lines and plot
decodability vs milestone per arm. Snapshots come from SnapshotCallback
(train.py --snapshot-steps ...) or wandb artifacts (download first).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────
# Model reconstruction from a snapshot
# ──────────────────────────────────────────────────────────────────────────
def build_policy_from_snapshot(snapshot_path: str, device: str = "cpu"):
    """Rebuild the MemActorCriticPolicy and load a snapshot's weights.

    Mirrors train.py's model construction (cell_factory + encoder dims). The
    snapshot carries the run `config`; we reuse make_cell_factory / make_vec_env
    from train.py so the architecture matches bit-for-bit.

    Returns (policy, vec_env, cfg). The vec_env is a single-env DummyVecEnv whose
    unwrapped env exposes `.mystery_path` and `.normalized_agent_position`.
    """
    from train import make_cell_factory          # module-level helpers in train.py
    from memrl.envs import make_vec_env
    from memrl.ppo import MemPPO

    snap = torch.load(snapshot_path, map_location=device, weights_only=False)
    cfg = snap["config"]
    if cfg is None:
        raise ValueError(
            f"{snapshot_path} has no embedded config; re-run training with the "
            "updated SnapshotCallback (it stores cfg) or pass --config explicitly."
        )

    env = make_vec_env(env_name=cfg["env_name"], n_envs=1, seed=cfg["seed"] + 999,
                       **cfg.get("env_kwargs", {}))
    factory = make_cell_factory(
        cell_name=cfg["cell"]["name"],
        cell_kwargs=cfg["cell"].get("kwargs", {}),
        hidden_size=cfg["encoder_dim"],
    )
    model = MemPPO(
        env=env, cell_factory=factory, lr=cfg["lr"],
        n_steps=cfg["n_steps"], n_epochs=cfg["n_epochs"],
        gamma=cfg.get("gamma", 0.99), gae_lambda=cfg.get("gae_lambda", 0.95),
        encoder_dim=cfg["encoder_dim"], encoder_hidden=cfg.get("encoder_hidden", 128),
        shared_backbones=cfg.get("shared_backbones", False),
        chunk_len=cfg["chunk_len"], n_chunks_per_batch=cfg["n_chunks_per_batch"],
        intrinsic_module=None, lambda_intrinsic=0.0,
        tensorboard_log=None, verbose=0, seed=cfg["seed"], device=device,
    )
    model.policy.load_state_dict(snap["policy_state"])
    model.policy.set_training_mode(False)
    return model.policy, env, cfg


# ──────────────────────────────────────────────────────────────────────────
# Ground-truth latent extraction from the MysteryPath env
# ──────────────────────────────────────────────────────────────────────────
def _unwrap(vec_env):
    e = vec_env.envs[0]
    return e.unwrapped


def _path_set(mp_env) -> set[tuple[int, int]]:
    """Set of (x,y) grid coords on the hidden path (excluding nothing)."""
    return {(int(n.x), int(n.y)) for n in mp_env.mystery_path.path}


def _agent_xy(mp_env) -> tuple[int, int]:
    return (int(mp_env.normalized_agent_position[0]),
            int(mp_env.normalized_agent_position[1]))


# ──────────────────────────────────────────────────────────────────────────
# Trajectory bank (fixed behavior policy → matched coverage across arms)
# ──────────────────────────────────────────────────────────────────────────
def collect_bank(
    vec_env,
    behavior: Callable[[Any], np.ndarray] | None,
    n_episodes: int,
    grid_dim: int = 7,
    max_steps: int = 128,
    seed: int = 0,
) -> list[dict]:
    """Roll a fixed behavior policy; record obs, episode_start, and per-step
    ground-truth (knowledge grid + full path). Returns a list of episode dicts."""
    rng = np.random.default_rng(seed)
    n_actions = vec_env.action_space.n if hasattr(vec_env.action_space, "n") else None
    bank, ep = [], None

    obs = vec_env.reset()
    mp = _unwrap(vec_env)
    for _ in range(n_episodes):
        path = _path_set(mp)
        knowledge = np.zeros((grid_dim, grid_dim), dtype=np.int8)
        full = np.zeros((grid_dim, grid_dim), dtype=np.int8)
        for (x, y) in path:
            if 0 <= x < grid_dim and 0 <= y < grid_dim:
                full[x, y] = 1
        ep = {"obs": [], "episode_start": [], "knowledge": [], "full": [], "visited": []}
        done = False
        first = True
        steps = 0
        while not done and steps < max_steps + 1:
            ax, ay = _agent_xy(mp)
            on_path = (ax, ay) in path
            if 0 <= ax < grid_dim and 0 <= ay < grid_dim:
                knowledge[ax, ay] = 1 if on_path else -1
            ep["obs"].append(np.asarray(obs, dtype=np.float32)[0])
            ep["episode_start"].append(bool(first))
            ep["knowledge"].append(knowledge.copy().reshape(-1))
            ep["full"].append(full.reshape(-1).copy())
            ep["visited"].append((knowledge.reshape(-1) != 0).astype(np.int8))
            first = False

            if behavior is None:
                act = np.array([rng.integers(0, n_actions)])
            else:
                act = behavior(obs)
            obs, _, dones, _ = vec_env.step(act)
            done = bool(dones[0])
            steps += 1
            mp = _unwrap(vec_env)  # post-reset env on done (auto-reset)
        for k in ep:
            ep[k] = np.asarray(ep[k])
        bank.append(ep)
    return bank


# ──────────────────────────────────────────────────────────────────────────
# Recurrent-state extraction (replay bank obs through a checkpoint)
# ──────────────────────────────────────────────────────────────────────────
def _flatten_actor_state(cell_state: dict, device) -> torch.Tensor:
    """Concatenate all actor-side state tensors into (n_envs, D)."""
    from memrl.policy.policy import _split_state
    actor_state, _ = _split_state(cell_state)
    parts = []
    for v in actor_state.values():
        if torch.is_tensor(v):
            parts.append(v.reshape(v.shape[0], -1))
    return torch.cat(parts, dim=-1) if parts else torch.zeros((1, 1), device=device)


@torch.no_grad()
def extract_states(policy, bank: list[dict], device: str = "cpu") -> dict:
    """Replay each episode's observations through `policy`, collecting the actor
    recurrent state aligned with the latent labels. Coverage is fixed by the bank,
    so this isolates the representation."""
    H, K, F, VIS = [], [], [], []
    for ep in bank:
        cell_state = policy.initial_state(1, torch.device(device))
        T = ep["obs"].shape[0]
        for t in range(T):
            o = torch.as_tensor(ep["obs"][t:t+1], device=device)
            es = torch.as_tensor(ep["episode_start"][t:t+1], device=device)
            # forward returns (action, value, log_prob, new_state, side)
            _, _, _, cell_state, _ = policy.forward(o, cell_state, es)
            h = _flatten_actor_state(cell_state, device)[0].cpu().numpy()
            H.append(h); K.append(ep["knowledge"][t]); F.append(ep["full"][t])
            VIS.append(ep["visited"][t])
    return {"H": np.asarray(H, np.float32), "K": np.asarray(K, np.int8),
            "F": np.asarray(F, np.int8), "VIS": np.asarray(VIS, np.int8)}


# ──────────────────────────────────────────────────────────────────────────
# Probe training (linear + MLP), per-cell balanced accuracy
# ──────────────────────────────────────────────────────────────────────────
def _fit_probe_torch(X, y, kind="linear", epochs=150, device="cpu"):
    """Binary probe; returns test balanced-accuracy. y in {0,1}."""
    n = X.shape[0]
    idx = np.random.default_rng(0).permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    Xtr = torch.as_tensor(X[tr], device=device); ytr = torch.as_tensor(y[tr], dtype=torch.float32, device=device)
    Xte = torch.as_tensor(X[te], device=device); yte = y[te]
    d = X.shape[1]
    if kind == "linear":
        net = torch.nn.Linear(d, 1).to(device)
    else:
        net = torch.nn.Sequential(torch.nn.Linear(d, 128), torch.nn.ReLU(),
                                  torch.nn.Linear(128, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    # class-balanced BCE
    pos = float(ytr.mean()) + 1e-6
    w = torch.where(ytr > 0.5, torch.tensor(0.5 / pos, device=device),
                    torch.tensor(0.5 / (1 - pos), device=device))
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logit = net(Xtr).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, ytr, weight=w)
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = (torch.sigmoid(net(Xte).squeeze(-1)).cpu().numpy() > 0.5).astype(int)
    # balanced accuracy
    out = []
    for c in (0, 1):
        m = yte == c
        if m.sum() > 0:
            out.append((pred[m] == c).mean())
    return float(np.mean(out)) if out else 0.5


def decodability(states: dict, target="K", kind="linear", grid_dim=7, device="cpu") -> dict:
    """Mean per-cell balanced accuracy of decoding the latent grid from h_t.

    target="K": knowledge grid (was this cell confirmed on/off path?) — we decode
                the binary "is this cell a CONFIRMED PATH tile" over VISITED cells.
    target="F": full path map (is this cell on the path), all cells.

    Also returns per-cell accuracies (for the effective-RM-size readout) and the
    summed "resolved bits" = Σ_cell max(0, 2·(bal_acc − 0.5)) — a 0..1 resolved
    fraction per binary cell, summed to bits of minimal-RM state the memory holds.
    """
    H = states["H"]
    accs = []
    for cell in range(grid_dim * grid_dim):
        if target == "K":
            # restrict to rows where this cell has been visited; label = on-path(1)/off(0)
            vis = states["VIS"][:, cell] == 1
            if vis.sum() < 50:
                continue
            y = (states["K"][vis, cell] == 1).astype(int)
            X = H[vis]
        else:
            y = (states["F"][:, cell] == 1).astype(int)
            X = H
        if len(np.unique(y)) < 2:
            continue
        accs.append(_fit_probe_torch(X, y, kind=kind, device=device))
    resolved_bits = float(np.sum([max(0.0, 2.0 * (a - 0.5)) for a in accs]))
    return {"mean_bal_acc": float(np.mean(accs)) if accs else 0.5,
            "n_cells": len(accs), "target": target, "probe": kind,
            "resolved_bits": resolved_bits,
            "eff_rm_size_bits": float(2.0 ** resolved_bits)}


# ──────────────────────────────────────────────────────────────────────────
# Autoencode: EXACT minimal-RM state = the remaining-to-reproduce suit sequence.
# The toggle wrapper exposes it in info["rm_state"]; we decode it from h_t. This
# is the one env where the probe's target is the *exact* minimal-RM state, not a
# proxy — so it validates the decodability / eff-RM-size methodology end-to-end.
# ──────────────────────────────────────────────────────────────────────────
def collect_bank_autoencode(vec_env, n_episodes: int, max_pos: int = 12,
                            seed: int = 0) -> list[dict]:
    """Roll a uniform-random policy on the Autoencode-toggle env; record per step
    the obs, episode_start, and the remaining-to-reproduce suits (padded to
    max_pos with -1). info["rm_state"] is the exact minimal-RM state."""
    rng = np.random.default_rng(seed)
    asp = vec_env.action_space
    n_act = asp.n if hasattr(asp, "n") else int(np.prod(asp.nvec))
    bank, ep = [], None
    obs = vec_env.reset()
    cur_rm = ()                       # unknown until first step's info
    first = True
    eps_done = 0
    ep = {"obs": [], "episode_start": [], "remaining": []}
    guard = 0
    while eps_done < n_episodes and guard < n_episodes * 1000:
        guard += 1
        rem = np.full(max_pos, -1, dtype=np.int64)
        for i, s in enumerate(cur_rm[:max_pos]):
            rem[i] = int(s)
        ep["obs"].append(np.asarray(obs, dtype=np.float32)[0])
        ep["episode_start"].append(bool(first))
        ep["remaining"].append(rem)
        first = False
        act = np.array([rng.integers(0, n_act)])
        obs, _, dones, infos = vec_env.step(act)
        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
        cur_rm = tuple(info0.get("rm_state", ()))
        if bool(dones[0]):
            for k in ep:
                ep[k] = np.asarray(ep[k])
            bank.append(ep)
            ep = {"obs": [], "episode_start": [], "remaining": []}
            eps_done += 1
            first = True
            cur_rm = ()
    return bank


@torch.no_grad()
def extract_states_autoencode(policy, bank: list[dict], device: str = "cpu") -> dict:
    """Replay each episode's obs through `policy`; collect actor state + the
    padded remaining-suit latent (the exact minimal-RM state)."""
    H, R = [], []
    for ep in bank:
        cell_state = policy.initial_state(1, torch.device(device))
        T = ep["obs"].shape[0]
        for t in range(T):
            o = torch.as_tensor(ep["obs"][t:t + 1], device=device)
            es = torch.as_tensor(ep["episode_start"][t:t + 1], device=device)
            _, _, _, cell_state, _ = policy.forward(o, cell_state, es)
            h = _flatten_actor_state(cell_state, device)[0].cpu().numpy()
            H.append(h); R.append(ep["remaining"][t])
    return {"H": np.asarray(H, np.float32), "R": np.asarray(R, np.int64)}


def _fit_probe_multiclass(X, y, n_classes, kind="linear", epochs=150, device="cpu"):
    """Multiclass probe; returns test accuracy. y in {0..n_classes-1}."""
    n = X.shape[0]
    idx = np.random.default_rng(0).permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    Xtr = torch.as_tensor(X[tr], device=device); ytr = torch.as_tensor(y[tr], dtype=torch.long, device=device)
    Xte = torch.as_tensor(X[te], device=device); yte = y[te]
    d = X.shape[1]
    if kind == "linear":
        net = torch.nn.Linear(d, n_classes).to(device)
    else:
        net = torch.nn.Sequential(torch.nn.Linear(d, 128), torch.nn.ReLU(),
                                  torch.nn.Linear(128, n_classes)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(net(Xtr), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(Xte).argmax(-1).cpu().numpy()
    return float((pred == yte).mean())


def decodability_autoencode(states: dict, n_suits=4, kind="linear", device="cpu") -> dict:
    """Decode the remaining-to-reproduce suit at each relative position from h_t.

    resolved_bits = Σ_pos 2·max(0, (acc − chance)/(1 − chance)) — bits of the
    exact minimal-RM state the memory holds (2 bits/position for 4 suits).
    eff_rm_size_bits = 2^resolved_bits. Freeze ⇒ ≈chance everywhere ⇒ ≈1 state.
    """
    H, R = states["H"], states["R"]
    max_pos = R.shape[1]
    chance = 1.0 / n_suits
    accs, per_pos = [], []
    for p in range(max_pos):
        mask = R[:, p] >= 0                       # this relative position has a card
        if mask.sum() < 50:
            continue
        y = R[mask, p]; X = H[mask]
        if len(np.unique(y)) < 2:
            continue
        a = _fit_probe_multiclass(X, y, n_suits, kind=kind, device=device)
        accs.append(a); per_pos.append((p, round(a, 3)))
    resolved_bits = float(np.sum([2.0 * max(0.0, (a - chance) / (1 - chance)) for a in accs]))
    return {"mean_acc": float(np.mean(accs)) if accs else chance,
            "n_pos": len(accs), "probe": kind, "resolved_bits": resolved_bits,
            "eff_rm_size_bits": float(2.0 ** resolved_bits), "per_pos": per_pos}


def effective_rm_size(states: dict, grid_dim=7, ks=(1, 2, 4, 8, 16, 32),
                      device="cpu") -> dict:
    """Effective reward-machine size of the learned memory (P5 readout).

    Two complementary estimates, both from the SAME probe data (no new rollouts):

    (1) bits-based: `2^{resolved_bits}` from the per-cell knowledge-grid
        decodability — how many distinguishable minimal-RM states the memory
        linearly resolves. The headline scalar: freeze ⇒ ≈1 (trivial machine);
        e3b grows it. (Computed by decodability(..., target="K").)

    (2) clustering cross-check: k-means h_t into k groups for increasing k; report
        the adjusted mutual information between cluster id and the (quantized)
        knowledge-grid state. The k at which AMI plateaus ≈ #RM states the memory
        actually separates. Uses sklearn if available; else skipped.
    """
    out = decodability(states, target="K", kind="linear", grid_dim=grid_dim, device=device)
    result = {"resolved_bits": out["resolved_bits"],
              "eff_rm_size_bits": out["eff_rm_size_bits"],
              "mean_bal_acc": out["mean_bal_acc"]}
    # clustering cross-check (optional)
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_mutual_info_score
        H = states["H"]
        # quantize the knowledge grid to a state id (sign per cell → base-3-ish hash)
        K = states["K"]  # (N, 49) in {-1,0,1}
        state_id = np.zeros(K.shape[0], dtype=np.int64)
        for c in range(K.shape[1]):
            state_id = state_id * 3 + (K[:, c] + 1)
        # collapse to a manageable label set (rank-encode)
        _, state_lab = np.unique(state_id, return_inverse=True)
        amis = {}
        for k in ks:
            if k > len(H):
                continue
            km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(H)
            amis[k] = float(adjusted_mutual_info_score(state_lab, km.labels_))
        result["kmeans_ami"] = amis
        result["n_true_rm_states"] = int(len(np.unique(state_lab)))
    except Exception as e:
        result["kmeans_ami"] = None
        result["cluster_note"] = f"sklearn unavailable or failed: {type(e).__name__}"
    return result


# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="run dir containing snapshot_step*.pt")
    ap.add_argument("--snapshots", default=None,
                    help="comma-separated milestones to probe (default: all found)")
    ap.add_argument("--behavior", default="random", choices=["random"],
                    help="fixed behavior policy generating the matched-coverage bank")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--probe", default="both", choices=["linear", "mlp", "both"])
    ap.add_argument("--target", default="K", choices=["K", "F", "both"])
    ap.add_argument("--task", default="auto", choices=["auto", "mysterypath", "autoencode"],
                    help="latent/RM-state type. 'auto' infers from the run's env_name.")
    ap.add_argument("--max-pos", type=int, default=12,
                    help="autoencode: # leading remaining-suit positions to decode")
    ap.add_argument("--n-suits", type=int, default=4,
                    help="autoencode/tiny: vocab size (suits=4; TinyReproduce v=2)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run = Path(args.run_dir)
    snaps = sorted(run.glob("snapshot_step*.pt"),
                   key=lambda p: int(p.stem.split("step")[1]))
    if args.snapshots:
        want = {int(s) for s in args.snapshots.split(",")}
        snaps = [p for p in snaps if int(p.stem.split("step")[1]) in want]
    if not snaps:
        raise SystemExit(f"no snapshots in {run} (train with --snapshot-steps ...)")

    # Build the bank ONCE from the final checkpoint's env; reuse for every
    # milestone → identical matched-coverage bank across arms.
    _, env, cfg = build_policy_from_snapshot(str(snaps[-1]), device=args.device)
    task = args.task
    ekw = cfg.get("env_kwargs", {})
    is_reproduce = (cfg["env_name"].startswith("popgym-Autoencode")
                    or cfg["env_name"].startswith("TinyReproduce"))
    if task == "auto":
        task = "autoencode" if is_reproduce else "mysterypath"
    # Auto-set vocab/length from the run config so the lag-Δ retention probe is
    # right per env (TinyReproduce carries k,v; Autoencode is suits=4, len 52).
    if task == "autoencode" and cfg["env_name"].startswith("TinyReproduce"):
        if "--n-suits" not in sys.argv: args.n_suits = int(ekw.get("v", args.n_suits))
        if "--max-pos" not in sys.argv: args.max_pos = int(ekw.get("k", args.max_pos))

    probes = ["linear", "mlp"] if args.probe == "both" else [args.probe]
    if task == "autoencode":
        bank = collect_bank_autoencode(env, n_episodes=args.n_episodes, max_pos=args.max_pos)
        for sp in snaps:
            step = int(sp.stem.split("step")[1])
            policy, _, _ = build_policy_from_snapshot(str(sp), device=args.device)
            states = extract_states_autoencode(policy, bank, device=args.device)
            for pk in probes:
                res = decodability_autoencode(states, n_suits=args.n_suits,
                                              kind=pk, device=args.device)
                # res["per_pos"] = [(relative-position, acc), ...] = the lag-Δ
                # RETENTION curve: position r is "decode the token due r steps from
                # now" — how well the memory RETAINS each held token. Decay with r
                # (and e3b>none) = the bonus aids retention; flat = it doesn't.
                print(json.dumps({"run": run.name, "step": step,
                                  "metric": "decodability_retention",
                                  "lag_retention_curve": res.get("per_pos"), **res}))
        return

    bank = collect_bank(env, behavior=None, n_episodes=args.n_episodes,
                        max_steps=cfg.get("env_kwargs", {}).get("max_steps", 128))
    targets = ["K", "F"] if args.target == "both" else [args.target]
    for sp in snaps:
        step = int(sp.stem.split("step")[1])
        policy, _, _ = build_policy_from_snapshot(str(sp), device=args.device)
        states = extract_states(policy, bank, device=args.device)
        for tgt in targets:
            for pk in probes:
                res = decodability(states, target=tgt, kind=pk, device=args.device)
                rec = {"run": run.name, "step": step, "metric": "decodability", **res}
                print(json.dumps(rec))
        # P5 effective-reward-machine-size readout (freeze ⇒ ~1; e3b re-inflates)
        rm = effective_rm_size(states, device=args.device)
        print(json.dumps({"run": run.name, "step": step, "metric": "eff_rm_size", **rm}))


if __name__ == "__main__":
    main()
