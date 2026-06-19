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

Calibration (so a NULL is real, not a blind/under-fit probe)
------------------------------------------------------------
A "the bonus didn't help realization" claim is only valid if the probe could have
SEEN a difference. We make every read false-negative-safe and false-positive-gated:
  * features standardized on train stats + trained to convergence — an under-fit
    probe reports "not decodable" on clean signal (the worst failure mode);
  * report BOTH linear (Moore-separability = headline) AND MLP (information-present);
    linear-flat + MLP-high = "stored-not-separable", never "no realization";
  * a shuffled-label floor (refit on permuted y, n_shuffles); a cell is credited
    only if its accuracy clears floor_mean + 2.5·floor_std (kills the max(0,·) clamp
    bias) — see `_credit`. `resolved_bits` sums only significant cells;
  * an OBS-ONLY baseline (source="O"): if the latent decodes from the raw obs, the
    hidden-state read is a ceiling artifact (the MysteryPath K-saturation mode) — on
    register envs the PLAY phase masks the token so this should sit at the floor;
  * register envs (Tiny/Autoencode) decode the EXACT minimal-RM state PLAY-phase only;
  * stateless cells (Memoryless) return resolved_bits=None — realization UNDEFINED,
    not a measured null (the actor state is a constant, which would read as chance).

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
import sys
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


def _actor_state_is_empty(cell_state: dict) -> bool:
    """True if the actor side carries NO recurrent tensor (e.g. Memoryless).

    Without this guard `_flatten_actor_state` returns a constant zeros((1,1)), so a
    stateless cell would decode at the empirical floor BY CONSTRUCTION and masquerade
    as a measured 'realization = NO'. Callers must report stateless cells as
    realization-UNDEFINED, never as a null with content.
    """
    from memrl.policy.policy import _split_state
    actor_state, _ = _split_state(cell_state)
    return not any(torch.is_tensor(v) for v in actor_state.values())


@torch.no_grad()
def extract_states(policy, bank: list[dict], device: str = "cpu") -> dict:
    """Replay each episode's observations through `policy`, collecting the actor
    recurrent state aligned with the latent labels. Coverage is fixed by the bank,
    so this isolates the representation."""
    H, O, K, F, VIS = [], [], [], [], []
    stateless = False
    for ep in bank:
        cell_state = policy.initial_state(1, torch.device(device))
        T = ep["obs"].shape[0]
        for t in range(T):
            o = torch.as_tensor(ep["obs"][t:t+1], device=device)
            es = torch.as_tensor(ep["episode_start"][t:t+1], device=device)
            # forward returns (action, value, log_prob, new_state, side)
            _, _, _, cell_state, _ = policy.forward(o, cell_state, es)
            stateless = stateless or _actor_state_is_empty(cell_state)
            h = _flatten_actor_state(cell_state, device)[0].cpu().numpy()
            H.append(h); O.append(ep["obs"][t].reshape(-1))
            K.append(ep["knowledge"][t]); F.append(ep["full"][t])
            VIS.append(ep["visited"][t])
    return {"H": np.asarray(H, np.float32), "O": np.asarray(O, np.float32),
            "K": np.asarray(K, np.int8), "F": np.asarray(F, np.int8),
            "VIS": np.asarray(VIS, np.int8), "stateless": stateless}


# ──────────────────────────────────────────────────────────────────────────
# Probe training (linear + MLP), per-cell balanced accuracy
# ──────────────────────────────────────────────────────────────────────────
def _fit_probe_torch(X, y, kind="linear", epochs=300, device="cpu", n_shuffles=10):
    """Binary probe. Returns (test balanced-accuracy, floor_mean, floor_std).

    The floor is the SAME probe refit on permuted training labels and scored on the
    real test labels — the empirical false-positive baseline, repeated `n_shuffles`
    times so callers can credit only accuracy that is SIGNIFICANTLY above the floor
    distribution (mean + 2·std), not merely above its mean. Without the std gate the
    per-cell max(0,·) clamp biases resolved-bits upward on pure noise.
    """
    n = X.shape[0]
    idx = np.random.default_rng(0).permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    d = X.shape[1]
    # Standardize features on TRAIN stats — recurrent states have wildly different
    # per-dim scales; without this a linear probe under-fits genuine signal (a
    # false-negative machine). Then train enough to actually converge.
    mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
    Xz = (X - mu) / sd

    def _fit(ytr_arr, yte_arr):
        Xtr = torch.as_tensor(Xz[tr], device=device)
        ytr = torch.as_tensor(ytr_arr, dtype=torch.float32, device=device)
        Xte = torch.as_tensor(Xz[te], device=device)
        if kind == "linear":
            net = torch.nn.Linear(d, 1).to(device)
        else:
            net = torch.nn.Sequential(torch.nn.Linear(d, 128), torch.nn.ReLU(),
                                      torch.nn.Linear(128, 1)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
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
        out = [(pred[yte_arr == c] == c).mean() for c in (0, 1) if (yte_arr == c).sum() > 0]
        return float(np.mean(out)) if out else 0.5

    acc = _fit(y[tr], y[te])
    floors = [_fit(y[tr][np.random.default_rng(1000 + s).permutation(len(tr))], y[te])
              for s in range(max(1, n_shuffles))]
    return acc, float(np.mean(floors)), float(np.std(floors))


def _credit(acc, floor_mean, floor_std, z=2.5):
    """Resolved fraction 0..1 of one classifier, gated for significance: 0 unless the
    real accuracy clears the shuffled-floor distribution by z·std (kills the clamp
    bias on noise), then (acc − floor_mean)/(1 − floor_mean)."""
    if acc <= floor_mean + z * floor_std:
        return 0.0
    return max(0.0, (acc - floor_mean) / (1.0 - floor_mean + 1e-9))


def decodability(states: dict, target="K", kind="linear", grid_dim=7, device="cpu",
                 source="H") -> dict:
    """Mean per-cell balanced accuracy of decoding the latent grid from h_t.

    target="K": knowledge grid (was this cell confirmed on/off path?) — we decode
                the binary "is this cell a CONFIRMED PATH tile" over VISITED cells.
    target="F": full path map (is this cell on the path), all cells.
    source="H": decode from the recurrent hidden state (the realization read).
    source="O": decode from the raw OBSERVATION (ceiling/leakage control — if the
                latent is decodable from the obs alone, the hidden-state read is
                uninterpretable, which is the MysteryPath K-saturation failure mode).

    `resolved_bits` credits, per binary cell, only accuracy ABOVE the empirical
    shuffled-label floor: Σ_cell max(0, (acc − floor)/(1 − floor)).  Stateless cells
    (Memoryless) return resolved_bits=None — realization is UNDEFINED, not a null.
    """
    if states.get("stateless") and source == "H":
        return {"mean_bal_acc": None, "mean_chance_floor": None, "n_cells": 0,
                "target": target, "probe": kind, "source": source,
                "resolved_bits": None, "resolved_frac": None, "eff_rm_size_bits": None,
                "stateless": True,
                "note": "no recurrent state; realization undefined (not a measured null)"}
    X_all = states[source]
    accs, floors, credits = [], [], []
    for cell in range(grid_dim * grid_dim):
        if target == "K":
            # restrict to rows where this cell has been visited; label = on-path(1)/off(0)
            vis = states["VIS"][:, cell] == 1
            if vis.sum() < 50:
                continue
            y = (states["K"][vis, cell] == 1).astype(int)
            X = X_all[vis]
        else:
            y = (states["F"][:, cell] == 1).astype(int)
            X = X_all
        if len(np.unique(y)) < 2:
            continue
        a, fmean, fstd = _fit_probe_torch(X, y, kind=kind, device=device)
        accs.append(a); floors.append(fmean); credits.append(_credit(a, fmean, fstd))
    resolved_bits = float(np.sum(credits))
    return {"mean_bal_acc": float(np.mean(accs)) if accs else None,
            "mean_chance_floor": float(np.mean(floors)) if floors else None,
            "n_cells": len(accs), "n_significant": int(np.sum(np.asarray(credits) > 0)),
            "target": target, "probe": kind, "source": source,
            "resolved_bits": resolved_bits, "resolved_bits_max": float(len(accs)),
            "resolved_frac": float(resolved_bits / len(accs)) if accs else None,
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
    raw = _unwrap(vec_env)
    # TEACHER-FORCING: reproduce envs (TinyReproduce) TERMINATE on the first wrong
    # play token, so a uniform-random bank dies before the play phase → no retention
    # data. Drive the env with the CORRECT token (env._target()) instead: this both
    # traverses the full play phase AND gives policy-independent MATCHED coverage —
    # every checkpoint replays the identical correct trajectory, so a decodability
    # gap is representational. Watch-phase actions are ignored by the env. Envs with
    # no `_target` oracle fall back to random (warned if play coverage is thin).
    teacher = hasattr(raw, "_target")

    def _reset_rm():
        return (tuple(getattr(raw, "_seq", ())[:getattr(raw, "_shown", 0)])
                if teacher else ())

    bank = []
    obs = vec_env.reset()
    raw = _unwrap(vec_env)
    cur_rm = _reset_rm()                          # SB3 drops reset info → read it here
    first = True
    eps_done = 0
    ep = {"obs": [], "episode_start": [], "remaining": [], "is_play": []}
    guard = 0
    while eps_done < n_episodes and guard < n_episodes * 4000:
        guard += 1
        rem = np.full(max_pos, -1, dtype=np.int64)
        for i, s in enumerate(cur_rm[:max_pos]):
            rem[i] = int(s)
        o0 = np.asarray(obs, dtype=np.float32)[0]
        ep["obs"].append(o0)
        ep["episode_start"].append(bool(first))
        ep["remaining"].append(rem)
        ep["is_play"].append(bool(o0[1] > 0.5))   # obs = [is_watch, is_play, onehot]
        first = False
        raw = _unwrap(vec_env)
        if teacher:
            act = np.array([int(raw._target())])   # correct token (ignored in watch)
        else:
            act = np.array([rng.integers(0, n_act)])
        obs, _, dones, infos = vec_env.step(act)
        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
        cur_rm = tuple(info0.get("rm_state", ()))
        if bool(dones[0]):
            for k in ep:
                ep[k] = np.asarray(ep[k])
            bank.append(ep)
            ep = {"obs": [], "episode_start": [], "remaining": [], "is_play": []}
            eps_done += 1
            first = True
            raw = _unwrap(vec_env)
            cur_rm = _reset_rm()
    n_play = int(sum(int(np.asarray(e["is_play"]).sum()) for e in bank))
    if n_play < 50:
        print(f"  [probe] WARNING: only {n_play} play-phase steps (teacher={teacher}); "
              f"realization read will be empty.", file=sys.stderr)
    return bank


@torch.no_grad()
def extract_states_autoencode(policy, bank: list[dict], device: str = "cpu") -> dict:
    """Replay each episode's obs through `policy`; collect actor state + the
    padded remaining-suit latent (the exact minimal-RM state)."""
    H, O, R, PLAY = [], [], [], []
    stateless = False
    for ep in bank:
        cell_state = policy.initial_state(1, torch.device(device))
        T = ep["obs"].shape[0]
        for t in range(T):
            o = torch.as_tensor(ep["obs"][t:t + 1], device=device)
            es = torch.as_tensor(ep["episode_start"][t:t + 1], device=device)
            _, _, _, cell_state, _ = policy.forward(o, cell_state, es)
            stateless = stateless or _actor_state_is_empty(cell_state)
            h = _flatten_actor_state(cell_state, device)[0].cpu().numpy()
            H.append(h); O.append(ep["obs"][t].reshape(-1))
            R.append(ep["remaining"][t]); PLAY.append(bool(ep["is_play"][t]))
    return {"H": np.asarray(H, np.float32), "O": np.asarray(O, np.float32),
            "R": np.asarray(R, np.int64), "is_play": np.asarray(PLAY, bool),
            "stateless": stateless}


def _fit_probe_multiclass(X, y, n_classes, kind="linear", epochs=300, device="cpu",
                          n_shuffles=10):
    """Multiclass probe. Returns (test accuracy, floor_mean, floor_std).

    The empirical floor (probe refit on permuted labels) beats the theoretical 1/K
    chance when classes are imbalanced; the std over shuffles lets callers gate for
    significance (mean + 2·std) rather than crediting any accuracy above the mean.
    """
    n = X.shape[0]
    idx = np.random.default_rng(0).permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    d = X.shape[1]
    mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
    Xz = (X - mu) / sd

    def _fit(ytr_arr, yte_arr):
        Xtr = torch.as_tensor(Xz[tr], device=device)
        ytr = torch.as_tensor(ytr_arr, dtype=torch.long, device=device)
        Xte = torch.as_tensor(Xz[te], device=device)
        if kind == "linear":
            net = torch.nn.Linear(d, n_classes).to(device)
        else:
            net = torch.nn.Sequential(torch.nn.Linear(d, 128), torch.nn.ReLU(),
                                      torch.nn.Linear(128, n_classes)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(net(Xtr), ytr)
            loss.backward(); opt.step()
        with torch.no_grad():
            pred = net(Xte).argmax(-1).cpu().numpy()
        return float((pred == yte_arr).mean())

    acc = _fit(y[tr], y[te])
    floors = [_fit(y[tr][np.random.default_rng(1000 + s).permutation(len(tr))], y[te])
              for s in range(max(1, n_shuffles))]
    return acc, float(np.mean(floors)), float(np.std(floors))


def decodability_autoencode(states: dict, n_suits=4, kind="linear", device="cpu",
                            source="H", play_only=True) -> dict:
    """Decode the remaining-to-reproduce suit at each relative position from h_t.

    Restricted to PLAY-phase steps (play_only=True): during WATCH the shown token is
    one-hot in the obs, so decoding it is leakage, not retention. Play-phase steps
    mask the token → a clean forced-retention read of the exact minimal-RM state.

    source="H": recurrent hidden state (the realization read).
    source="O": raw obs (leakage control — should be ≈floor in the play phase).

    resolved_bits = Σ_pos log2(n_suits)·max(0, (acc − floor)/(1 − floor)) against the
    EMPIRICAL shuffled-label floor (not theoretical 1/K). Stateless cells return None.
    """
    if states.get("stateless") and source == "H":
        return {"mean_acc": None, "mean_chance_floor": None, "n_pos": 0, "probe": kind,
                "source": source, "play_only": bool(play_only), "resolved_bits": None,
                "resolved_frac": None, "eff_rm_size_bits": None, "per_pos": [],
                "stateless": True,
                "note": "no recurrent state; realization undefined (not a measured null)"}
    X_all, R = states[source], states["R"]
    max_pos = R.shape[1]
    bits_per = float(np.log2(n_suits))
    rowmask = (states["is_play"].astype(bool) if (play_only and "is_play" in states)
               else np.ones(R.shape[0], dtype=bool))
    accs, floors, credits, per_pos = [], [], [], []
    for p in range(max_pos):
        mask = (R[:, p] >= 0) & rowmask           # this position has a card AND play-phase
        if mask.sum() < 50:
            continue
        y = R[mask, p]; X = X_all[mask]
        if len(np.unique(y)) < 2:
            continue
        a, fmean, fstd = _fit_probe_multiclass(X, y, n_suits, kind=kind, device=device)
        cr = _credit(a, fmean, fstd)
        accs.append(a); floors.append(fmean); credits.append(cr)
        per_pos.append((p, round(a, 3), round(fmean, 3), round(bits_per * cr, 3)))
    resolved_bits = float(np.sum([bits_per * c for c in credits]))
    return {"mean_acc": float(np.mean(accs)) if accs else None,
            "mean_chance_floor": float(np.mean(floors)) if floors else None,
            "n_pos": len(accs), "n_significant": int(np.sum(np.asarray(credits) > 0)),
            "probe": kind, "source": source, "play_only": bool(play_only),
            "resolved_bits": resolved_bits,
            "resolved_bits_max": float(len(accs) * bits_per),
            "resolved_frac": float(resolved_bits / (len(accs) * bits_per)) if accs else None,
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
    if states.get("stateless"):
        return {"stateless": True, "resolved_bits": None, "eff_rm_size_bits": None,
                "note": "no recurrent state; effective RM size undefined"}
    # Report BOTH probes: linear = Moore-separability (the headline); MLP = information
    # present at all. linear-flat + MLP-high = "stored-but-not-separable" (a finding),
    # never "no realization". Reporting only linear conflates absence with nonlinearity.
    result = {}
    for kind in ("linear", "mlp"):
        out = decodability(states, target="K", kind=kind, grid_dim=grid_dim, device=device)
        result[f"resolved_bits_{kind}"] = out["resolved_bits"]
        result[f"eff_rm_size_bits_{kind}"] = out["eff_rm_size_bits"]
        result[f"mean_bal_acc_{kind}"] = out["mean_bal_acc"]
        result[f"mean_chance_floor_{kind}"] = out["mean_chance_floor"]
    # back-compat headline aliases (linear)
    result["resolved_bits"] = result["resolved_bits_linear"]
    result["eff_rm_size_bits"] = result["eff_rm_size_bits_linear"]
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
        obs_baseline_done = False
        for sp in snaps:
            step = int(sp.stem.split("step")[1])
            policy, _, _ = build_policy_from_snapshot(str(sp), device=args.device)
            states = extract_states_autoencode(policy, bank, device=args.device)
            # Obs-only ceiling/leakage control (snapshot-independent → emit once). If
            # this is well above floor in the PLAY phase, the token leaks from the obs
            # and the hidden-state read is uninterpretable.
            if not obs_baseline_done:
                for pk in probes:
                    ob = decodability_autoencode(states, n_suits=args.n_suits, kind=pk,
                                                 device=args.device, source="O")
                    print(json.dumps({"run": run.name, "metric": "obs_baseline_retention",
                                      "control": "ceiling/leakage", **ob}))
                obs_baseline_done = True
            for pk in probes:
                res = decodability_autoencode(states, n_suits=args.n_suits, kind=pk,
                                              device=args.device, source="H")
                # res["per_pos"] = [(rel-position, acc, shuffled_floor), ...] = the lag-Δ
                # RETENTION curve over PLAY-phase steps: position r is "decode the token
                # due r steps from now" — how well the memory RETAINS each held token.
                # Decay with r (and e3b>none) = the bonus aids retention; flat = it doesn't.
                print(json.dumps({"run": run.name, "step": step,
                                  "metric": "decodability_retention",
                                  "lag_retention_curve": res.get("per_pos"), **res}))
        return

    bank = collect_bank(env, behavior=None, n_episodes=args.n_episodes,
                        max_steps=cfg.get("env_kwargs", {}).get("max_steps", 128))
    targets = ["K", "F"] if args.target == "both" else [args.target]
    obs_baseline_done = False
    for sp in snaps:
        step = int(sp.stem.split("step")[1])
        policy, _, _ = build_policy_from_snapshot(str(sp), device=args.device)
        states = extract_states(policy, bank, device=args.device)
        # Obs-only ceiling/leakage control (snapshot-independent → emit once). On
        # MysteryPath the path marks stay observable, so this is expected HIGH — it
        # is the witness that a flat hidden-state K-read is a ceiling artifact, not a
        # representational null (relabel MysteryPath as belief-, not RM-, realization).
        if not obs_baseline_done:
            for tgt in targets:
                for pk in probes:
                    ob = decodability(states, target=tgt, kind=pk, device=args.device,
                                      source="O")
                    print(json.dumps({"run": run.name, "metric": "obs_baseline",
                                      "control": "ceiling/leakage", **ob}))
            obs_baseline_done = True
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
