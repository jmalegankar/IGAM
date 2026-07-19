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
    hp: dict = field(default_factory=dict)   # per-env HP overrides (the global HP
                                             # is the MysteryPath winner; other envs
                                             # were tuned differently)


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
           # RIDE replaces PBIM as the 3rd non-potential episodic bonus (2026-07-19,
           # advisor call): impact-family (Δφ / √N_ep) alongside E3B (coverage) and
           # NovelD (prediction-error). RIDE uses λ=0.005 (see _cfg) not 0.03 — its
           # dense every-step impact bonus swamps the sparse task at 0.03 (validated
           # on S13: λ0.005 → succ 1.0, λ0.03 → dawdle/0). PBIM data retained in
           # appendix; RIDE is the headline 3rd family.
           ["none", "e3b_idm", "ride", "noveld"],
           [_mpg("sparse"), _mpg("penalty", reward_fall_off=-0.008),
            _mpg("aligned", reward_path_progress=0.1)],
           20_000_000,
           "α>0 probing · F1 decodability · F2 freeze/rescue · F3 RIDE (3rd family) · "
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
    EnvArm("S13", "MiniGrid-MemoryS13-v0", "memrl-s13-matched", SIX,
           # RIDE replaces PBIM (advisor call 2026-07-19); λ=0.005 via _cfg. Validated
           # locally: GDN RIDE λ0.005 → succ 1.0 (equalizes, matches e3b/noveld refs).
           ["none", "e3b_idm", "noveld", "ride"],
           # EXP-7 / EXP-1. Reward- AND metric-matched to MysteryPath via MemoryRewardWrapper
           # (minigrid_wrappers.py). MiniGrid's NATIVE S13 reward is horizon-DISCOUNTED
           # (1 - 0.9 t/T, T_max = 5*13^2 = 845) — not sparse — which confounds the
           # amplify-vs-equalize sign with a reward-shape/metric difference. reward_mode=flat
           # gives a clean +1/0 with a binary success_rate (matched to MPG's sparse arm). The
           # freeze arm adds an UNCONDITIONAL -1/T_max per movement action with a free no-op
           # sanctuary — the always-paid cost the success-discount lacks (so it can freeze).
           [Density("sparseV3", env_kwargs={"agent_view_size": 3, "reward_mode": "flat"}),
            Density("sparseV7", env_kwargs={"agent_view_size": 7, "reward_mode": "flat"}),
            Density("freezeV3", env_kwargs={"agent_view_size": 3, "reward_mode": "flat",
                                            "move_penalty": 1.0 / 845})],
           20_000_000,
           "EXP-7 amplify-vs-equalize at n=5 (flat-sparse, both views, success_rate) + EXP-1 "
           "the freeze on a cue-retention task · common 6-cell zoo + pbim (ρ keystone, 2nd env). "
           "12-cell native (discounted) runs are appendix-only, archived in memrl-memtrain-s13.",
           # S13's OWN tuned HP (the values that solved it in memrl-s13-baseline) — NOT the
           # MysteryPath global HP. Delayed cross-corridor reward needs the longer credit
           # horizon (gamma/λ) and lr 3e-4.
           hp={"gamma": 0.999, "gae_lambda": 0.98, "chunk_len": 32, "lr": 3.0e-4}),
    EnvArm("Battleship", "popgym-BattleshipEasy-v0", "memrl-memtrain-battleship", FOUR,
           ["none", "e3b_idm"],
           [Density("dense", env_kwargs={"expose_action_coords": True}),
            Density("sparse", env_kwargs={"expose_action_coords": True, "defer_reward": True})],
           10_000_000,
           "α>0 probing (2nd positive env) → generalizes 'bonus helps' beyond MysteryPath"),
    EnvArm("TinyReproduce", "TinyReproduce-v0", "memrl-memtrain-tiny",
           SIX + ["GTrXL", "LinearTransformer"],   # +full-attn + linear-attn: does
           # attention beat the SSM exact-recall floor? (Jelassi "Repeat After Me")
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
    EnvArm("MiniHack", "MiniHack-Memento-F2-v0", "memrl-memtrain-minihack", SIX,
           ["none", "e3b_idm", "noveld", "rnd"],
           # Memento default reward: sparse +1 on the cue-matched target, -0.01/step,
           # wrong fork = trap/death. cue (sleeping monster) shown ONLY at start → leaves
           # view → memory REQUIRED (memoryless feedforward cannot learn). α>0 (exploration
           # helps reach the distant fork). Corridor-R2/R3/R5 = exploration-difficulty knob.
           [Density("sparse")],
           10_000_000,
           "EXPLORATION-COMMUNITY memory benchmark (E3B/NovelD/RIDE's home) · cue→retain→"
           "choose, memory required · the RND ('global, mismatched') vs E3B/NovelD arm for C3 "
           "· glyph-crop obs → GlyphEncoder (embedding+CNN). CPU/no-display.",
           seeds=[0, 1, 2]),
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
    Method("E15", "GatedDeltaNet LR×state-dim sweep on Tiny (rigor: tuning, not capacity)",
           "TinyReproduce-v0", "memrl-memtrain-gdnsweep", ["GatedDeltaNet"], ["none"],
           # each Density = one (lr, assoc_size) cell: cfg overrides lr AND the cell
           # kwargs (assoc_size); env_kwargs pins k=10/v=4 sparse (the exact-recall task
           # where GDN floors at the registry lr=1e-4). labels avoid dots (k8s-valid).
           [Density(f"lr{lt}_a{a}",
                    env_kwargs={"k": 10, "v": 4, "order": "reverse", "density": "sparse"},
                    cfg={"lr": lv, "cell": {"name": "GatedDeltaNet",
                                            "kwargs": {"assoc_size": a}}})
            for (lt, lv) in (("1e4", 1e-4), ("3e4", 3e-4), ("1e3", 1e-3), ("3e3", 3e-3))
            for a in (64, 256)],
           [0, 1], 10_000_000,
           "C4/F8: does GDN close the gated-RNN exact-recall gap at its best HP? "
           "report cell rankings at per-cell-best HP (k10 floor at lr1e-4 is tuning, not capacity)"),
    Method("HPOT", "Oracle-potential rescue arm (PBIM counterexample, MysteryPath penalty)",
           "MysteryPath-Grid-v0", "memrl-mpg-oraclepot", THREE_STRONG, ["none"],
           # H-POT §3.2: penalty arm + a TASK-INFORMED state potential Φ=−β·d(agent,goal)
           # delivered as PBRS on r_ext via OraclePotentialWrapper (NOT an intrinsic module —
           # so intrinsic=none). Tests whether an oracle potential reopens the freeze where
           # PBIM's bonus-derived potential provably cannot (ρ=0). gamma MUST match the HP γ
           # (0.995) for telescoping consistency; β tuned to the e3b delivered-bonus scale.
           # Both outcomes are locked wins (rescue ⇒ "bonus value is its non-potential
           # residual"; no-rescue ⇒ "neither potential reopens the freeze").
           [Density("penalty_oraclepot",
                    env_kwargs={"reset_options": {"reward_fall_off": -0.008},
                                "oracle_potential": {"beta": 0.02, "gamma": 0.995}})],
           [0, 1, 2], 20_000_000,
           "PBIM-section survival kit (A5): task-informed extrinsic potential vs the freeze; "
           "n=3 → backfill to 5 if contested. Compare vs penalty-none freeze in memrl-memtrain-mpg."),
    Method("E5", "GatedDeltaNet best-HP α≈0 closure (Tiny)",
           "TinyReproduce-v0", "memrl-tiny-exp5", ["GatedDeltaNet"], ["none", "e3b_idm"],
           # GDN at its per-cell-best HP (lr 1e-3, assoc 64 — the E15 sweet spot where it
           # reaches ~0.35-0.40, so the cell has HEADROOM). Does E3B still do nothing there?
           # Closes the floor-confound on the α≈0 control (E15 was intrinsic=none only, so
           # "E3B≈none" was only shown while GDN was floored).
           [Density("besthp_sparse",
                    env_kwargs={"k": 10, "v": 4, "order": "reverse", "density": "sparse"},
                    cfg={"lr": 1e-3, "cell": {"name": "GatedDeltaNet", "kwargs": {"assoc_size": 64}}}),
            Density("besthp_dense",
                    env_kwargs={"k": 10, "v": 4, "order": "reverse", "density": "dense"},
                    cfg={"lr": 1e-3, "cell": {"name": "GatedDeltaNet", "kwargs": {"assoc_size": 64}}})],
           [0, 1, 2, 3, 4], 10_000_000,
           "α≈0 at GDN best HP: E3B−none within margin even with headroom ⇒ bonus genuinely "
           "inert, not a floor artifact. Pairs with E15."),
    Method("DISTRACT", "Distractor-reward ablation — structural vs reward sparsity (MysteryPath)",
           "MysteryPath-Grid-v0", "memrl-mpg-distractor", SIX, ["none", "e3b_idm", "noveld"],
           # Reviewer-2 control (DistractorRewardWrapper): +ε for revisiting an already-seen
           # tile this episode (dense, USELESS), ε=eps_frac/T_max so per-episode total ≤ 0.1 ≪
           # the unit goal reward → optimum UNCHANGED (still solve), only reward-sparsity removed
           # + a farmable local optimum planted; fall=0. DENSITY-MATCHED to `aligned` (which pays
           # +0.1 for NEW frontier tiles = useful), opposite usefulness. Prediction: amplification
           # PERSISTS here (Δstrong−Δweak under e3b ≈ sparse's) while it VANISHES on aligned ⇒
           # structural sparsity ≠ reward sparsity. No RIDE arm (RIDE fall-farms on MPG regardless).
           [Density("distractor", env_kwargs={"distractor": {"eps_frac": 0.1}})],
           [0, 1, 2, 3, 4], 20_000_000,
           "Kills 'your effect is just reward sparsity' — dense-but-useless reward, same density "
           "as aligned, opposite usefulness; bonus still amplifies ⇒ it's the acquisition structure."),
    # ── NORMALIZED-PBIM RELAUNCH — pbim3 (2026-07-13) ─────────────────────────
    # Two prior PBIM attempts, both superseded:
    #   • original (memrl-memtrain-mpg / memrl-s13-matched): no terminal anchor →
    #     S13 V_int→3e6 runaway at γ=0.999·T=845. DROPPED.
    #   • pbim2 (memrl-mpg-pbim2 / memrl-s13-pbim2): anchored + Φ=+V_int. Bounded,
    #     but delivered −(b) (consumption sign) with an UN-centered ~50-scale
    #     terminal spike (λ·Φ≈1.6 > +1 task reward) → agents ran out the clock to
    #     dodge the terminal kick (ep_len 8→845, success→0 on every arm with a
    #     baseline). A real magnitude/sign pathology, kept as the "why normalize"
    #     ablation.
    # pbim3 = faithful NORMALIZED PBIM (Forbes et al. 2024, Eq. 34): Φ=−V_int of
    # the running-mean-CENTERED bonus, so delivery is +（b−b̄) (raw-e3b direction),
    # tiny magnitude (no spike/stall), exact telescoping to +V_int(s₀)≈0. Validated
    # on the bench by experiments/analysis/pbim_anchor_check.py (sign corr=1.000,
    # telescoping resid<1e-3, γ=0.999 max|Φ| 56→0.02) + a trainer smoke (bonus_mean
    # −0.5→~0). FRESH projects so analysis never mixes normalized with pbim2/orig;
    # none/e3b/noveld baselines in the original projects are untouched.
    Method("PBIM3MPG", "Normalized-PBIM relaunch (MysteryPath, all 3 densities)",
           "MysteryPath-Grid-v0", "memrl-mpg-pbim3", SIX, ["pbim_e3b_idm"],
           [_mpg("sparse"), _mpg("penalty", reward_fall_off=-0.008),
            _mpg("aligned", reward_path_progress=0.1)],
           [0, 1, 2, 3, 4], 20_000_000,
           "F3/ρ keystone on the faithful normalized potential (+（b−b̄) guidance, exact "
           "telescoping): compare vs none/e3b_idm baselines in memrl-memtrain-mpg (unchanged) "
           "and vs the un-normalized stall in memrl-mpg-pbim2."),
    Method("PBIM3S13", "Normalized-PBIM relaunch (S13, fast-EMA b̄, all 3 densities)",
           "MiniGrid-MemoryS13-v0", "memrl-s13-pbim3b", SIX, ["pbim_e3b_idm"],
           # memrl-s13-pbim3 (ema_momentum=0.99 default) showed a RESIDUAL dawdle:
           # ep_len 500–845 (none≈8–17), succ 0.06–0.29, V_int~O(10) NOT O(0.01).
           # Cause: E3B bonus decays over training; a slow b̄ lags it → centered
           # bonus stays negative → V_int accumulates over S13's 845·γ=0.999 horizon.
           # Fix = faster b̄ (ema_momentum 0.99→0.95 via intrinsic_kwargs) so b̄
           # tracks the decay and keeps centered≈0. MPG (memrl-mpg-pbim3) is fine at
           # 0.99 and UNCHANGED. S13's tuned HP + the ema override ride in Density.cfg.
           [Density("sparseV3", env_kwargs={"agent_view_size": 3, "reward_mode": "flat"},
                    cfg={"gamma": 0.999, "gae_lambda": 0.98, "chunk_len": 32, "lr": 3.0e-4,
                         "intrinsic_kwargs": {"ema_momentum": 0.95}}),
            Density("sparseV7", env_kwargs={"agent_view_size": 7, "reward_mode": "flat"},
                    cfg={"gamma": 0.999, "gae_lambda": 0.98, "chunk_len": 32, "lr": 3.0e-4,
                         "intrinsic_kwargs": {"ema_momentum": 0.95}}),
            Density("freezeV3", env_kwargs={"agent_view_size": 3, "reward_mode": "flat",
                                            "move_penalty": 1.0 / 845},
                    cfg={"gamma": 0.999, "gae_lambda": 0.98, "chunk_len": 32, "lr": 3.0e-4,
                         "intrinsic_kwargs": {"ema_momentum": 0.95}})],
           [0, 1, 2, 3, 4], 20_000_000,
           "S13-PBIM the fair way + fast-EMA fix: the ep_len 8→845 stall must be GONE "
           "(watch eval/mean_ep_length ≈ none's, pbim_V_int_mean O(1) not O(10), disc_sum small). "
           "If dawdle persists the within-episode novelty structure is inherent → S13-PBIM stays "
           "MPG-only in the paper (no loss, keystone is MPG)."),
]
METHOD = {m.eid: m for m in METHODS}


# ── config / script emission (NOT auto-run) ─────────────────────────────────
def _cfg(env_name, project, cell, dens, intrinsic, budget, eid, env_tag, hp_extra=None):
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
    if hp_extra:                       # per-env HP override (e.g. S13's tuned HP)
        cfg.update(hp_extra)
    cfg.update(dens.cfg)               # density-specific (e.g. λ-sweep) wins last
    cfg["intrinsic"] = intrinsic
    if intrinsic != "none":
        # RIDE's dense, non-decaying impact bonus swamps the sparse task at the
        # shared 0.03 (dawdle/0 success); 0.005 is the validated value (S13 GDN
        # λ0.005 → 1.0). E3B/NovelD keep the shared HP λ.
        cfg["lambda_intrinsic"] = 0.005 if intrinsic == "ride" else HP["lambda_intrinsic"]
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
               env_tag, subdir, hp_extra=None):
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
                                        eid, env_tag, hp_extra), f, sort_keys=False)
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
                          arm.densities, arm.seeds, arm.budget, arm.env, f"E1/{arm.env}",
                          hp_extra=arm.hp)
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
        # exact "E1" or "E1:<env>" → core grid; everything else (incl. methods that
        # start with "E1" like E10/E11/E15) → method dispatch.
        if args.emit == "E1" or args.emit.startswith("E1:"):
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
