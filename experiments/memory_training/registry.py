"""Experiment registry — the single declarative source of truth for the redo.

Structure (per the PI): there is ONE core experiment and a few method studies.

  E1  CORE GRID = env × density × bonus × cell × seed, replicated across the env
      suite. MortarMayhem / Autoencode / S13 / Battleship / TinyReproduce /
      SearingSpotlights are NOT separate experiments — they are E1 *on another
      env* (a slice). The 20M HEADLINE (3× sparse none-vs-e3b) is likewise just
      the MysteryPath-sparse slice of E1. Each env-arm carries the same design;
      what differs is the memory type it stresses and (for the return-matched
      envs) how clean the δ-toggle is.

  E2  PROBE (analysis, offline) — decodability + eff-RM-size over E1 snapshots.

  Method studies vary a NON-env axis, so they are distinct experiments:
    E3  GAE-λ sweep            (P2 mechanism)
    E6  Tiny maze sanctuary+DP (P1 bound, P4 deconfound)        [BLOCKED]
    E9  TBPTT k-sweep          (P2 secondary)                   [BLOCKED]
    E10 φ-ablation             (which bonus trains memory)
    E11 entangled IDM-aux arm  (P3 causal + recipe)             [BLOCKED]

Usage (does NOT auto-run — this file is setup, not launch):
    python -m experiments.memory_training.registry --list
    python -m experiments.memory_training.registry --emit E1     # or E1:MysteryPath
    python -m experiments.memory_training.registry --emit-all

Emission lays out configs/<EID>[/<env>]/<cell>_<density>_<bonus>.yaml and matching
scripts, and routes wandb so the UI hierarchy is project→env, group→reward/density,
job_type→bonus, tags=[EID,env,cell,density,bonus], name={cell}-{density}-{bonus}-seedN.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from train import DEFAULT_CELL_KWARGS  # noqa: E402

SIX = ["GRU", "LSTM", "RetNet", "GatedDeltaNet", "Mamba2", "Memoryless"]
# Full memory-cell zoo (the 11 published cells + Memoryless floor) — for the
# cell × memory-type comparison on the embodied retention env (S13), where the
# gated-RNN-vs-SSM recall inversion is most relevant. GTrXL carries a param
# confound (report counts). Excludes branch/LMU variants + S4D.
ZOO12 = ["GRU", "LSTM", "RetNet", "GatedDeltaNet", "Mamba2", "LRU", "FFM",
         "LinearTransformer", "SHM", "mLSTM", "GTrXL", "Memoryless"]
FOUR = ["GRU", "RetNet", "GatedDeltaNet", "Memoryless"]
THREE_STRONG = ["GRU", "RetNet", "GatedDeltaNet"]
TINY3 = ["GRU", "RetNet", "Memoryless"]

HP = dict(n_envs=16, encoder_dim=128, encoder_hidden=256, lr=1.0e-4, n_steps=512,
          n_epochs=4, gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.008,
          vf_coef=1.0, max_grad_norm=0.5, target_kl=0.05, chunk_len=64,
          n_chunks_per_batch=32, lambda_intrinsic=0.03,
          eval_every_rollouts=10, n_eval_episodes=20)
SNAP = "500000,2000000,5000000,10000000"


@dataclass
class Density:
    label: str
    env_kwargs: dict = field(default_factory=dict)
    cfg: dict = field(default_factory=dict)


@dataclass
class EnvArm:
    """One env-slice of the CORE grid (E1)."""
    env: str                          # short tag (also wandb-project suffix + folder)
    env_name: str
    project: str
    cells: list
    intrinsics: list
    densities: list                   # list[Density]
    budget: int
    role: str                         # what this env-slice uniquely contributes
    seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    blocked_on: str = ""


@dataclass
class Method:
    """A study that varies a NON-env axis (sweep / ablation / analysis)."""
    eid: str
    title: str
    env_name: str
    project: str
    cells: list
    intrinsics: list
    densities: list
    seeds: list
    budget: int
    serves: str
    blocked_on: str = ""
    analysis: bool = False
    notes: str = ""


def _mpg(label, **rk):
    return Density(label, env_kwargs=({"reset_options": rk} if rk else {}))


# ── E1 CORE GRID: one design, replicated across the env suite ────────────────
CORE = [
    EnvArm("MysteryPath", "MysteryPath-Grid-v0", "memrl-memtrain-mpg", SIX,
           ["none", "e3b_idm", "pbim_e3b_idm", "noveld"],
           [_mpg("sparse"), _mpg("penalty", reward_fall_off=-0.008),
            _mpg("aligned", reward_path_progress=0.1)],
           20_000_000,
           "α>0 probing · F1 decodability · F2 freeze/rescue · F3 PBIM · "
           "entanglement · P5 · 20M HEADLINE (sparse none-vs-e3b slice)"),
    EnvArm("MortarMayhem", "MortarMayhem-Grid-v0", "memrl-memtrain-mm", FOUR,
           ["none", "e3b_idm", "pbim_e3b_idm"],
           [Density("sparse", env_kwargs={"reset_options": {"command_count": [4],
                    "reward_command_success": 0.0, "reward_episode_success": 1.0}}),
            Density("aligned", env_kwargs={"reset_options": {"command_count": [4],
                    "reward_command_success": 0.25, "reward_episode_success": 0.0}})],
           20_000_000,
           "α≈0 sequence-WM · EXACTLY return-matched (4×0.25=1.0) → clean redundancy flip"),
    EnvArm("Autoencode", "popgym-AutoencodeEasy-v0", "memrl-memtrain-autoencode", SIX,
           ["none", "e3b_idm"],
           [Density("sparse", env_kwargs={"autoencode_density": "sparse", "autoencode_order": "reverse"}),
            Density("dense", env_kwargs={"autoencode_density": "dense", "autoencode_order": "reverse"})],
           20_000_000,
           "α≈0 reproduce · EXACT δ-toggle · KNOWN minimal RM → exact probe ground truth"),
    EnvArm("S13", "MiniGrid-MemoryS13-v0", "memrl-memtrain-s13", ZOO12,
           ["none", "e3b_idm", "noveld"], [Density("sparse")],
           10_000_000,
           "EMBODIED PO retention (egocentric 7×7) · α≈0 bonus-neutral contrast · "
           "noveld≥e3b reversal · FULL 12-cell zoo → cell×memory-type recall comparison"),
    EnvArm("Battleship", "popgym-BattleshipEasy-v0", "memrl-memtrain-battleship", FOUR,
           ["none", "e3b_idm"],
           [Density("dense", env_kwargs={"expose_action_coords": True}),
            Density("sparse", env_kwargs={"expose_action_coords": True, "defer_reward": True})],
           10_000_000,
           "α>0 probing (2nd positive env) → generalizes 'bonus helps' beyond MysteryPath"),
    EnvArm("TinyReproduce", "TinyReproduce-v0", "memrl-memtrain-tiny", SIX,
           ["none", "e3b_idm"],
           [Density("sparse", env_kwargs={"k": 10, "v": 4, "order": "reverse", "density": "sparse"}),
            Density("dense", env_kwargs={"k": 10, "v": 4, "order": "reverse", "density": "dense"})],
           10_000_000,
           "α≈0 retention (k=10,v=4, reverse) · known RM · CLEAN lag-Δ retention probe "
           "(Autoencode-52 is unlearnable; this is the tractable tunable version)"),
    EnvArm("SearingSpotlights", "SearingSpotlights-v0", "memrl-memtrain-ss", FOUR,
           ["none", "e3b_idm"],
           [Density("sparse"),
            Density("anti", env_kwargs={"reset_options": {"reward_inside_spotlight": -0.008}})],
           10_000_000,
           "α<0 hazard · no-sanctuary freeze boundary (anti-none should NOT freeze)"),
]

# ── method studies (vary a non-env axis) ────────────────────────────────────
METHODS = [
    Method("E2", "Decodability + eff-RM-size probe (OFFLINE over E1 snapshots)",
           "—", "—", [], [], [], [], 0,
           "F1 decodability · P5 eff-RM-size — analysis, no training runs",
           analysis=True,
           notes="python -m memrl.probes.decode_memory --run-dir <E1 run> "
                 "--snapshots 500000,2000000,5000000,10000000,20000000 ; "
                 "Autoencode/TinyReproduce slices: add --task autoencode (--n-suits 2 for Tiny)."),
    Method("E3", "GAE-λ sweep (P2 mechanism)",
           "MysteryPath-Grid-v0", "memrl-memtrain-lambda", ["GRU"], ["none", "e3b_idm"],
           [Density(f"l{l}", cfg={"gae_lambda": l}) for l in (0.8, 0.9, 0.95, 0.99, 1.0)],
           [0, 1, 2], 10_000_000,
           "sparse deficit & e3b−none shrink as λ→1 — can falsify our own P2"),
    Method("E6", "Tiny maze: sanctuary toggle + DP bound (P1/P4)",
           "TinyMaze-v0", "memrl-memtrain-tinymaze", ["GRU", "RetNet"], ["none", "e3b_idm"],
           [Density("sanctuary"), Density("no_sanctuary")], [0, 1, 2, 3, 4], 3_000_000,
           "P1 bound ε=p·E[falls] (DP) · P4 freeze-needs-sanctuary (within-env, deconfounds SS)",
           blocked_on="build TinyMaze env + belief-MDP DP solver "
                      "(memrl/envs/tiny_maze.py, memrl/probes/optimal_prober.py)"),
    Method("E9", "TBPTT k-sweep (P2 secondary)",
           "MysteryPath-Grid-v0", "memrl-memtrain-ksweep", ["GRU"], ["none", "e3b_idm"],
           [Density(f"k{k}", cfg={"chunk_len": k, "n_chunks_per_batch": max(1, 2048 // k)})
            for k in (1, 2, 4, 8, 16, 32, 64, 128)], [0, 1, 2], 10_000_000,
           "e3b−none non-monotone in k, collapses at k=1 (both arms fail)",
           blocked_on="episode-aligned chunking in buffer.py + per-epoch state-refresh control"),
    Method("E10", "φ-ablation: which bonus trains memory",
           "MysteryPath-Grid-v0", "memrl-memtrain-phi", THREE_STRONG,
           ["e3b_idm", "e3b_rand", "e3b_obs"], [_mpg("sparse")], [0, 1, 2, 3, 4], 10_000_000,
           "IDM-φ ≫ random-φ ⇒ controllable features matter; ≈ ⇒ any episodic novelty"),
    Method("E11", "Entangled IDM-aux arm (P3 causal + recipe)",
           "MysteryPath-Grid-v0", "memrl-memtrain-entangle", THREE_STRONG,
           ["none", "e3b_idm", "e3b_idm_shared"], [_mpg("sparse")], [0, 1, 2, 3, 4], 10_000_000,
           "shared-IDM-aux raises decodability+success w/o policy-bias cost ⇒ credit-to-memory",
           blocked_on="build e3b_idm_shared arm (IDM φ = policy encoder, aux CE into backbone)"),
]
METHOD = {m.eid: m for m in METHODS}


# ── config / script emission (NOT auto-run) ─────────────────────────────────
def _cfg(env_name, project, cell, dens, intrinsic, budget, eid, env_tag):
    cfg = {"env_name": env_name}
    if dens.env_kwargs:
        cfg["env_kwargs"] = dens.env_kwargs
    cfg.update({"n_envs": HP["n_envs"], "total_timesteps": budget, "seed": 0,
                "cell": {"name": cell, "kwargs": dict(DEFAULT_CELL_KWARGS[cell])},
                "encoder_dim": HP["encoder_dim"], "encoder_hidden": HP["encoder_hidden"]})
    for k in ("lr", "n_steps", "n_epochs", "gamma", "gae_lambda", "clip_range",
              "ent_coef", "vf_coef", "max_grad_norm", "target_kl",
              "chunk_len", "n_chunks_per_batch"):
        cfg[k] = HP[k]
    cfg.update(dens.cfg)
    cfg["intrinsic"] = intrinsic
    if intrinsic != "none":
        cfg["lambda_intrinsic"] = HP["lambda_intrinsic"]
    cfg.update({"eval_every_rollouts": HP["eval_every_rollouts"],
                "n_eval_episodes": HP["n_eval_episodes"], "wandb": True,
                "wandb_project": project, "wandb_group": dens.label,
                "wandb_job_type": intrinsic,
                "wandb_tags": [eid, env_tag, cell, dens.label, intrinsic],
                "run_name": f"{cell}-{dens.label}-{intrinsic}"})
    return cfg


_SCRIPT = """\
#!/usr/bin/env bash
# AUTO-GENERATED ({eid}). {cell}/{dens}/{intr} seed {seed} — ONE run / one GPU, snapshots ON.
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="$(cd "$HERE/{up}" && pwd)"
cd "$REPO_ROOT"
if [[ -z "${{PYTHON:-}}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python"; fi
fi
DEVICE="${{DEVICE:-cuda}}"; RUNS_DIR="${{RUNS_DIR:-runs/memtrain_{eid}}}"; EXTRA="${{EXTRA:-}}"
SNAP="${{SNAPSHOT_STEPS:-{snap}}}"; SNAP_WANDB="${{SNAPSHOT_TO_WANDB:---snapshot-to-wandb}}"
LOGDIR="$HERE/{loglvl}/_logs/{eid}"; mkdir -p "$LOGDIR"
CFG="$HERE/{cfglvl}/configs/{subdir}/{cell}_{dens}_{intr}.yaml"
echo "[{eid}] {cell}/{dens}/{intr} seed={seed}"
exec "$PYTHON" -m train --config "$CFG" --seed {seed} \\
    --runs-dir "$RUNS_DIR" --device "$DEVICE" \\
    --wandb --snapshot-steps "$SNAP" $SNAP_WANDB $EXTRA \\
    2>&1 | tee "$LOGDIR/{cell}_{dens}_{intr}_seed{seed}.log"
"""


def _emit_grid(eid, env_name, project, cells, intrinsics, densities, seeds, budget,
               env_tag, subdir):
    """Emit one env×density×bonus×cell×seed grid → configs/<subdir>/ scripts/<subdir>/.
    ONE launch script per run (one GPU each): no intrinsic-packing → no OOM, and
    bonus-vs-none stays fair because seeds are matched across jobs."""
    cdir = HERE / "configs" / subdir; sdir = HERE / "scripts" / subdir
    cdir.mkdir(parents=True, exist_ok=True); sdir.mkdir(parents=True, exist_ok=True)
    snap = SNAP + (",20000000" if budget >= 20_000_000 else "")
    # HERE = experiments/memory_training/scripts/<subdir>. cfglvl/loglvl reach
    # memory_training (depth ups); REPO_ROOT is two more (…/experiments/repo).
    depth = subdir.count("/") + 2                     # HERE → memory_training
    up = "/".join([".."] * (depth + 2))               # HERE → repo root
    cfglvl = "/".join([".."] * depth)
    loglvl = "/".join([".."] * depth)
    ncfg = nsh = 0
    for cell in cells:
        for dens in densities:
            for intr in intrinsics:
                with open(cdir / f"{cell}_{dens.label}_{intr}.yaml", "w") as f:
                    yaml.safe_dump(_cfg(env_name, project, cell, dens, intr, budget,
                                        eid, env_tag), f, sort_keys=False)
                ncfg += 1
                for seed in seeds:
                    # filename has NO eid prefix — the dir (scripts/<eid>/…) already
                    # encodes it; this keeps k8s jobnames short (<63) and dedup'd.
                    p = sdir / f"s{seed}_{cell}_{dens.label}_{intr}.sh"
                    p.write_text(_SCRIPT.format(eid=eid, cell=cell, dens=dens.label,
                                 intr=intr, seed=seed, snap=snap, up=up,
                                 cfglvl=cfglvl, loglvl=loglvl, subdir=subdir))
                    os.chmod(p, 0o755); nsh += 1
    return ncfg, nsh


def emit_core(only_env: str | None = None):
    tot_c = tot_s = 0
    for arm in CORE:
        if only_env and arm.env != only_env:
            continue
        if arm.blocked_on:
            print(f"  [E1:{arm.env}] BLOCKED: {arm.blocked_on}"); continue
        c, s = _emit_grid("E1", arm.env_name, arm.project, arm.cells, arm.intrinsics,
                          arm.densities, arm.seeds, arm.budget, arm.env, f"E1/{arm.env}")
        print(f"  [E1:{arm.env}] {c} configs, {s} scripts → configs/E1/{arm.env}/")
        tot_c += c; tot_s += s
    print(f"  E1 total: {tot_c} configs, {tot_s} scripts")


def emit_method(m: Method):
    if m.analysis:
        print(f"  [{m.eid}] analysis-only (no run grid) — see notes"); return
    if m.blocked_on:
        print(f"  [{m.eid}] BLOCKED: {m.blocked_on}"); return
    c, s = _emit_grid(m.eid, m.env_name, m.project, m.cells, m.intrinsics,
                      m.densities, m.seeds, m.budget, m.env_name, m.eid)
    print(f"  [{m.eid}] {c} configs, {s} scripts → configs/{m.eid}/")


def summarize():
    print("E1  CORE GRID (env × density × bonus × cell × seed) — one experiment, per-env slices:")
    core_runs = 0
    for a in CORE:
        r = len(a.cells) * len(a.densities) * len(a.intrinsics) * len(a.seeds)
        core_runs += r
        print(f"    {a.env:17} {r:>4} runs  {a.budget//10**6:>3}M  "
              f"[{'/'.join(a.intrinsics)}] × [{'/'.join(d.label for d in a.densities)}] "
              f"× {len(a.cells)}c × {len(a.seeds)}s")
        print(f"        └ {a.role}")
    print(f"  E1 TOTAL: {core_runs} runs   (headline = MysteryPath sparse none-vs-e3b slice)\n")
    print("Method studies (vary a non-env axis):")
    mr = 0
    for m in METHODS:
        r = len(m.cells) * len(m.densities) * len(m.intrinsics) * len(m.seeds)
        mr += 0 if m.blocked_on else r
        tag = "analysis" if m.analysis else (f"⛔ {m.blocked_on[:30]}" if m.blocked_on else f"{r} runs")
        print(f"    {m.eid:4} {m.title[:46]:46} {tag}")
    print(f"\n  grand total runnable: {core_runs + mr} runs  "
          f"(blocked: {[m.eid for m in METHODS if m.blocked_on]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--emit", metavar="ID", help="E1 | E1:<env> | E3 | E10 …")
    ap.add_argument("--emit-all", action="store_true")
    args = ap.parse_args()
    if args.emit:
        if args.emit.startswith("E1"):
            env = args.emit.split(":", 1)[1] if ":" in args.emit else None
            emit_core(env)
        else:
            emit_method(METHOD[args.emit])
    elif args.emit_all:
        emit_core()
        for m in METHODS:
            emit_method(m)
    else:
        summarize()


if __name__ == "__main__":
    main()
