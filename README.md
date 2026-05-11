# IGAM: Innovation-Gated Associative Memory
## Implementation Plan

**Author:** Jai
**Last updated:** May 2026
**Status:** Phase A Week 1 in progress — cell interface, 9 baseline cells, and test harness built; Gated DeltaNet (headline) and PPO integration pending.

---

## What this document is

This is the canonical design and execution plan for IGAM, the memory + exploration architecture that succeeds the lmu_ppo thesis work. It exists to:

1. Lock in the architectural commitments before implementation, so we don't relitigate decisions mid-debugging.
2. Define exit criteria for each phase, so we know when to advance, when to debug, and when to bail.
3. Record the rationale for non-obvious choices, so future-Jai (or a collaborator) can audit the design without reconstructing six months of context.

It is **not** a research proposal. The research case is in the deep-research output (`docs/igam-deep-research.md`); that document is the *why*. This is the *how*.

---

## Phase A progress (as of 2026-05-12)

We expanded Week 1 beyond the original "implement Gated DeltaNet + MQAR test" scope to build the full baseline lineup against a common interface before the ablation sweep. This costs ~1 week of extra time up front but means the entire Phase A ablation table is buildable from one codebase — no interface refactors during the ablation runs themselves.

### Built

- **Project scaffolding**: `pyproject.toml`, `.gitignore`, `igam/` package, `igam/cell/` subpackage
- **Cell interface** (`igam/cell/base.py`): `RecurrentCell` ABC with `init_state`, `step`, `forward_sequence`, `reset_state` + free helpers `apply_episode_mask`, `detach_state`, `run_sequence`. The dict-state convention lets the rollout buffer iterate state components generically. Side outputs (e.g., `innovation` for the lifelong intrinsic reward) flow through `step` as a `SideOutputs` dict.
- **9 baseline cells** (all in `igam/cell/`, all conforming to the interface):
  - **RNN family**: `GRU`, `LSTM` — modern LN-equipped, per-block orthogonal init, forget-bias=+1
  - **Polynomial memory**: `LMU` (Voelker 2019, canonical scalar-input LegT; not the thesis's gated variant)
  - **Structured SSM**: `S4D` (Gu 2022, LTI diagonal), `Mamba2` (Dao & Gu 2024, selective)
  - **Matrix-memory family**: `LinearTransformer` (Katharopoulos 2020), `RetNet` (Sun 2023, fixed multi-scale γ), `DeltaNet` (Schlag 2021 + Yang 2024), `mLSTM` (Beck 2024 / xLSTM with max-trick stabilization)
- **Test harness** (`igam/cell/tests/`):
  - `conftest.py` — `cell_name_and_factory` parametrize fixture covering all 9 cells
  - `test_interface.py` — 144 contract tests covering shape discovery, device inference, dtype propagation, episode-start reset, `forward_sequence`-vs-manual-loop equivalence (regression guard), helpers, gradient flow
  - `test_synthetic_recall.py` — fast tier (composability/finite outputs) and slow tier (per-cell MQAR thresholds, opt-in via `pytest -m slow`)
  - 162 fast tests pass in ~4s; 9 slow tests pass in ~85s

### The 2×2 ablation framing

The cells map onto a 2×2 design matrix that the Phase A ablation can exploit:

```
                  No gating              Gated
No delta rule     LinearTransformer      mLSTM, RetNet
Delta rule        DeltaNet               IGAM
```

Each arrow isolates one design choice:
- `LinearTransformer → DeltaNet`: does the delta rule help?
- `LinearTransformer → mLSTM/RetNet`: does gating help (without delta)?
- `DeltaNet → IGAM`: does gating help (given the delta rule)?
- `mLSTM → IGAM`: is the delta + α-gating scheme better than LSTM-style exp gating?

Orthogonal structured-SSM family sub-figure: `S4D (LTI) → RetNet (fixed decay) → Mamba2 (selective)`. Side story.

### Tiered experiment plan (replaces the flat ablation list in Weeks 4–6 below)

- **Tier 1 (headline, 4 cells)**: `LSTM, Mamba2, mLSTM, IGAM`. Page-1 figure. If IGAM doesn't win here, the paper has no thesis.
- **Tier 2 (internal 2×2, 4 cells)**: `LinearTransformer, mLSTM, DeltaNet, IGAM`. Methods-section centerpiece.
- **Tier 3 (full appendix, all 9 cells)**: Everything for completeness.

Run cheapest-first: Tier 1 with 1 seed on POPGym-easy before committing to Tier 2/3. Hyperparameter sweep only on IGAM + Tier 1 cells; use published defaults for Tier 3.

### Not yet built (next sessions)

- **Tier 1 benchmark sweep** — POPGym easy + MiniGrid-Memory + BSuite. PPO integration is in place; just need to run.
- **MiniGrid-Memory port** — `lmu_ppo`'s `MemoryStartWrapper` integration for the Phase A regression check at Week 2.

### Running benchmarks

The PPO trainer (`igam/ppo/igam_ppo.py`) is generic over any cell in `igam.cell`. Configs live in `benchmarks/phase_a/`. Quick start:

```bash
# Tier 1 canary — IGAM on POPGym-RepeatPrevious-Easy, 1M steps
.venv/bin/python train.py --config benchmarks/phase_a/popgym_repeat_previous_easy.yaml --seed 0
```

Each run writes to `runs/<benchmark>/<cell>/seed_<n>_<timestamp>/`:
- `config.yaml` — exact config used
- `git_hash.txt` — commit hash at run start
- `env.txt` — `pip freeze` output
- `PPO_*/` — tensorboard event files
- `eval/` — periodic eval episode results (npz)
- `best_model/best_model.zip` — best-by-eval checkpoint
- `final_model.zip` — last checkpoint

### Monitoring (TensorBoard)

```bash
.venv/bin/python -m tensorboard --logdir runs/ --port 6006
# open http://localhost:6006/
```

What to watch:

| Metric | Meaning | Healthy range |
|---|---|---|
| `rollout/ep_rew_mean` | Training-time episode reward | trending up |
| `eval/mean_reward` | Held-out env reward, deterministic policy | trending up; less noisy than rollout |
| `train/policy_loss` | PPO surrogate loss | trending toward 0 |
| `train/value_loss` | Critic MSE | trending toward 0 |
| `train/approx_kl` | KL between old/new policy | <0.02; if it spikes, lower LR or clip_range |
| `train/explained_variance` | 1 − Var(returns − values) / Var(returns) | → 1 |
| `train/entropy_loss` | Negative policy entropy | trending toward 0 (less exploration over time) |
| `grad/{encoder,cell,actor,critic}_norm` | Per-component grad norms (README discipline) | <max_grad_norm (default 0.5) |
| `debug/innovation_mag_mean` | `‖δ_t‖` for DeltaNet/IGAM cells | non-zero, bounded |
| `debug/state_*_norm_*` | Per-state-component magnitudes | bounded; watch for explosion |
| `debug/action_frac_*` | Action distribution | shouldn't collapse to one action early |

---

## Relationship to the thesis codebase

`lmu_ppo` (the thesis codebase) is **frozen**. It is the experimental vehicle for the four-failures empirical story and should be preserved as a reproducible thesis artifact. Cherry-pick from it; do not modify it.

What we steal from `lmu_ppo`:

- The TBPTT rollout buffer pattern (`LMURolloutBuffer`'s chunked layout). This is genuinely good infrastructure and we'd rebuild it identically.
- The episode-boundary masking convention (zero memory state at episode_starts).
- The diagnostic logging discipline (per-component grad norms, per-channel activations, eigenvalue tracking from `episodic_bonus.py`).
- The PPO integration scaffolding (`LMUPPO` collect_rollouts → train loop).

What we leave behind:

- `LMUCell`, `LegSCell`, `lmu_t.py`, `lmu_s.py` — the Legendre memory cells. IGAM uses Gated DeltaNet; LegT/LegS do not survive.
- `OrthoLayer` and Cayley updates — unnecessary, the delta rule's contraction bound replaces them.
- `EllipticalEpisodicBonus` and Sherman-Morrison machinery — replaced by FSQ + count-min sketch.
- The HSWVIME stack (`hswvime_ppo/`, `models/wyner.py`, `models/wyner_lmu.py`, `models/vae.py`, `models/qa_module.py`) — the failure-mode that motivated the entire trajectory; cited in the paper's Background, not used.

---

## Architectural commitments (frozen)

These are the design decisions that define IGAM. Changing any of these is a research decision, not an implementation decision; if you find yourself wanting to change one mid-implementation, escalate to a design review.

### Memory cell

- **Gated DeltaNet** (Yang et al. 2024, ICLR 2025).
- Matrix state W_t ∈ R^(d_k × d_v), head-factored into H heads. Normalizer n_t maintained alongside.
- **State is (W_t, n_t)** — no separate recurrent hidden state (per ADR 0004). The cell is a pure key-value associative memory.
- Update rule: W_t = α_t W_{t-1} (I − β_t k_t k_t^T / ||k_t||^2) + β_t v_t k_t^T
- Normalizer update: n_t = α_t n_{t-1} + β_t k_t
- Read: y_t = q̃_t^T W_t / (q̃_t^T n_t + ε), with L2-normalized q̃_t = L2norm(q_t) and k̃_t = L2norm(k_t).
- **Gates and projections all come from φ(o_t)** (per ADR 0004 — Design A, the canonical Gated DeltaNet form):
  - α_t = σ(W_α φ(o_t)) — per-head scalar decay
  - β_t = σ(W_β φ(o_t)) — per-head scalar write strength
  - q_t = W_Q φ(o_t),  k_t = W_K φ(o_t),  v_t = W_V φ(o_t) — dynamic query preserves the lmu_ppo state-dependent readout insight.
- Innovation δ_t = v_t − W_{t-1} k̃_t — exposed as a side output for the lifelong intrinsic reward (Phase B).
- Actor-critic head input: concat(φ(o_t), y_t) — same structure as lmu_ppo's `cat([h, m_pooled])`, with φ(o_t) replacing h and y_t replacing m_pooled.
- **Training mode: single-step recurrent** (per ADR 0001). Parallel-scan path stubbed but not implemented in Phase A; revisited in Phase B.

### Encoder and auxiliary loss

- φ: O → R^d_φ. CNN for image obs, MLP for symbolic.
- Auxiliary loss is **hybrid IDM + JEPA-style latent prediction**:
  - IDM: -log p(a_t | φ(o_t), φ(o_{t+1}))
  - JEPA: ||P(φ(o_t), a_t) − φ_EMA(o_{t+1})||^2, with EMA target (τ=0.99)
  - L_φ = λ_idm L_idm + λ_jepa L_jepa, both 1.0 by default.
- φ is **detached** before entering the memory cell. The auxiliary optimizer is separate from PPO's optimizer.
- Phase A uses IDM only (POPGym is symbolic, MiniGrid is spatial bottleneck).
- Phase B adds JEPA prediction. Phase C adds I-JEPA-style block masking.

### Exploration

- Lifelong intrinsic reward: r_life_t = ||δ_t||^2, normalized by per-env Welford running mean/std.
- Episodic intrinsic reward: r_epi_t = 1 / sqrt(N̂_e(c_t)) where c_t = FSQ(φ(o_t)) is the discrete code and N̂ is a per-env count-min sketch (H=4 hashes, W=2^16 width).
- Combined: r_int_t = r_epi_t · clip(r_life_t, 1, L)  (NGU multiplicative form).
- Total reward: r_t = r_ext_t + β_int · r_int_t · (1 − episode_start_t).
- β_int = 0.01 default; ablate.

### FSQ bottleneck

- Project φ output to d_fsq ∈ [3, 8] dims, bound (tanh), round to L_i levels.
- Phase A/B: d_fsq=6, L=5 (codebook ~15,625).
- Phase C (Craftax-pixel, Montezuma, MineDojo): d_fsq=8, L=8 (codebook ~16M), applied per spatial location for pixel obs.
- No commitment loss, no learned codebook (FSQ is a stop-gradient quantization).

### PPO integration

- TBPTT chunks of length K=16, identical to lmu_ppo.
- Episode_starts mask zeros (W, n) at boundaries within a chunk.
- Memory state stored at chunk *start* in the rollout buffer; intra-chunk states are recomputed during evaluate_actions.
- Episodic state (count-min sketch, FSQ) reset at episode boundaries.

---

## Repository layout

```
igam/
├── README.md
├── plan.md                    # this document
├── docs/
│   ├── igam-deep-research.md  # the research synthesis
│   ├── thesis-retrospective.md # the four-failures story
│   └── adr/                   # architecture decision records (one per major choice)
├── igam/
│   ├── __init__.py
│   ├── cell/
│   │   ├── __init__.py
│   │   ├── gated_deltanet.py  # the matrix-memory cell
│   │   └── tests/
│   │       └── test_synthetic_recall.py  # Week 1 sanity tests
│   ├── encoder/
│   │   ├── __init__.py
│   │   ├── cnn.py             # image encoder
│   │   ├── mlp.py             # symbolic encoder
│   │   └── fsq.py             # finite scalar quantization
│   ├── auxiliary/
│   │   ├── __init__.py
│   │   ├── idm.py             # inverse dynamics model
│   │   ├── jepa.py            # latent prediction with EMA target
│   │   └── ema.py             # EMA target maintenance
│   ├── exploration/
│   │   ├── __init__.py
│   │   ├── count_min_sketch.py
│   │   ├── lifelong.py        # innovation-magnitude bonus
│   │   └── episodic.py        # FSQ + CMS episodic bonus
│   ├── policy/
│   │   ├── __init__.py
│   │   ├── igam_policy.py     # actor-critic with IGAM cell
│   │   └── buffer.py          # TBPTT rollout buffer (ported from lmu_ppo)
│   ├── ppo/
│   │   ├── __init__.py
│   │   └── igam_ppo.py        # PPO with IGAM-aware rollout collection
│   └── envs/
│       ├── __init__.py
│       ├── popgym_wrappers.py
│       ├── minigrid_wrappers.py # ported from lmu_ppo
│       ├── obstructed_maze.py
│       ├── craftax_wrappers.py
│       └── memory_start_wrapper.py # for Phase A regression check
├── train.py                   # main entry point
├── benchmarks/
│   ├── phase_a/               # POPGym + MiniGrid-Memory configs
│   ├── phase_b/               # ObstructedMaze + Craftax-symbolic
│   └── phase_c/               # pixel domains
├── scripts/
│   ├── run_synthetic_recall.py # Week 1 standalone test
│   ├── ablate_cell.py
│   └── ablate_auxiliary.py
├── tests/
│   ├── test_cell.py
│   ├── test_encoder.py
│   ├── test_buffer.py
│   └── test_integration.py
├── notebooks/
│   └── (analysis notebooks; not for running experiments)
├── pyproject.toml
└── .github/
    └── workflows/
        └── ci.yml             # unit tests + linting on push
```

A few things worth noting about this layout:

- **The cell, encoder, auxiliary, and exploration modules are independent packages.** This is deliberate. The cell should be testable on synthetic recall without any PPO scaffolding. The encoder + auxiliary should be testable on offline trajectories. The exploration module should be testable with mock features. We do not want the kind of monolith where you can only test the cell by running 100k PPO steps.

- **`docs/adr/` for architecture decision records.** Every time a non-obvious decision gets made (cell hyperparameter, auxiliary loss weight, FSQ configuration), drop a 1-page ADR explaining what we tried, what worked, what we picked, and why. By Week 12 you'll have ~15 of these and they'll be the source of truth when writing the paper's methods section.

- **`benchmarks/` separate from `scripts/`.** Benchmarks are the experimental configurations we run for the paper. Scripts are dev-time utilities. Keep them apart so the paper's reproducibility section can point at `benchmarks/` cleanly.

- **`tests/` matters.** This is RL code; bugs are silent. The Sherman-Morrison bug from lmu_ppo is the cautionary tale. Unit tests on the cell, encoder, buffer, and FSQ are non-optional. Aim for ≥70% coverage of the core modules.

---

## Phase 0 — Pre-implementation derisking (COMPLETE)

Both research questions resolved before Week 1. Findings drove two ADRs.

### 0.1 Linear-attention cells under on-policy PPO
**Resolved:** Yes, training instability is widely documented for Gated DeltaNet, mLSTM, and RWKV-7 under on-policy PPO with parallel-scan TBPTT. Failure modes are structural: matrix-memory states accumulate via covariance/outer-product updates that lack the bounded saturation of LSTM cells; high-variance PPO advantage signals propagate multiplicatively through these dense matrices; parallel prefix scans compound numerical errors logarithmically across chunks. Documented community workarounds (EGGROLL evolution strategies for RWKV-7, CISPO importance-weight clipping for linear attention) confirm the incompatibility is real, not folkloric.

**Decision:** ADR 0001 — implement single-step recurrent training fallback. Accept ~3× wall-clock penalty for Phase A in exchange for stability, bounded gradient flow, and per-step gradient clipping discipline. **Phase B will revisit parallel-scan once the Phase A configuration is stable**; some of the documented instabilities are cold-start pathologies that may not recur with a warm-started encoder and converged cell.

### 0.2 Craftax-symbolic model-free PPO baseline
**Resolved:** The published PPO-RNN baseline on Craftax-symbolic at 1M steps is **2.3%** of maximum reward. PPO-stateless lands at 2.2%; the recurrent policy adds ~0.1pp. Standard intrinsic-motivation baselines (RND, ICM, E3B) all also sit at ~2.2%. Source: Matthews et al. 2024 (original Craftax paper) and the official Craftax-Symbolic-v1 leaderboard.

The very low baseline tells a critical story for Phase B planning: **at 1M steps on symbolic obs, the bottleneck is exploration, not memory.** The PPO-RNN-vs-PPO gap of 0.1pp confirms that no recurrent policy is encoding useful long-horizon state at this sample budget — the agent hasn't reached the deeper achievements (mining, smelting, dungeons) that create memory-relevant dependencies. Memory becomes the dominant constraint only after exploration cracks past roughly 10% reward.

**Decision:** ADR 0002 — Phase B baseline is 2.3%. Success thresholds recalibrated below.

---

## Phase A — Memory cell validation (Weeks 1–8)

**Goal:** Establish that Gated DeltaNet under on-policy PPO with single-step recurrent TBPTT works, matches or exceeds the lmu_ppo LMU on memory-only tasks, and is the right foundation for Phases B and C.

**No exploration mechanism in Phase A.** Episodic bonus, lifelong bonus, FSQ — all disabled. Only the cell + encoder + PPO. This is the critical isolation discipline from the thesis (Failure 2: "stage isolation as methodology").

### Week 1 — Cell prototype + synthetic test

**Tasks:**
1. ~~Implement `igam/cell/gated_deltanet.py` in single-step recurrent mode (per ADR 0001).~~ **Pending — extends the already-built `DeltaNet` cell with the `α_t` decay gate.** Reference impl: `flash-linear-attention` library on GitHub. Port only the recurrent step path; leave parallel-scan signature in place but raise `NotImplementedError`.
2. **[done]** Write `igam/cell/tests/test_synthetic_recall.py` (MQAR). Selective copy deferred — at the test-scale config used here (hidden_size=16) cells bunch in 0.30–0.40 MQAR accuracy regardless, so adding selective copy would not differentiate further until Phase A POPGym-scale runs.

**Exit criterion (HARD):** MQAR accuracy ≥0.95 on length-256 sequences with 8-token vocabulary, training in <30 minutes on a single GPU under single-step recurrent mode. **If this fails, the cell has a bug. Do not proceed to Week 2 until it passes.** *(Headline cell still pending; check applies once `Gated DeltaNet` is implemented.)*

**Deliverable:** A working cell, a passing synthetic test, and a commit tagged `cell-v0.1`.

**Extended scope completed this week (beyond original plan):** the cell interface (`base.py`) and 8 *baseline* cells were built alongside, so the Phase A 2×2 ablation table is buildable from one codebase. See "Phase A progress" section at the top of this document.

### Week 2 — Wire cell into PPO, regression test on lmu_ppo task

**Tasks:**
1. Port the rollout buffer from `lmu_ppo/buffer.py` to `igam/policy/buffer.py`. Replace LMU state shape (h, m) with IGAM state shape (W, n) — per ADR 0004, no separate `h`. The chunked TBPTT layout is unchanged.
2. Implement `igam/policy/igam_policy.py`: actor-critic with the IGAM cell between encoder and policy/value heads. Mirror `lmu_ppo/policies.py` structurally; only the cell call differs.
3. Implement `igam/ppo/igam_ppo.py`: PPO with IGAM rollout collection. Mirror `lmu_ppo/lmu_ppo.py`; remove the E3B and lifelong-bonus code.
4. Run on **MiniGrid-Memory-S13 with the MemoryStartWrapper** (the regression benchmark from the thesis). No exploration bonus. Just memory.

**Exit criterion (SOFT):** IGAM matches or beats lmu_ppo's LMU baseline on MiniGrid-Memory-S13-with-wrapper within ±20%. If IGAM underperforms by more than 20%, debug the integration before Week 3 — most likely culprit is the cell init or the actor-critic head dimensions.

**Likely failure modes to watch for:**
- Cell initialization too aggressive; W_t saturates in first 100 steps. Fix: smaller init for W_K, W_V, W_Q.
- Normalizer n_t numerically unstable. Fix: layer-norm on y_t before policy head.
- TBPTT chunk boundaries not respected. Fix: verify episode_starts masking zeros both W and n.

**Deliverable:** End-to-end MiniGrid-Memory-S13 + wrapper run, training curves checked into `benchmarks/phase_a/results/`. Commit tagged `e2e-v0.1`.

### Week 3 — POPGym sanity check

**Tasks:**
1. Wire POPGym environments via `igam/envs/popgym_wrappers.py`. Port the lmu_ppo POPGym integration if there is one.
2. Run on the easier POPGym tasks: RepeatPrevious-Easy, Concentration-Easy, Battleship-Easy. These are the canary tests — if IGAM fails here, it won't work on the harder benchmarks.

**Exit criterion (SOFT):** Match or exceed published GRU/LSTM baselines on the three easy tasks within 1M steps. Numbers are in the POPGym paper (Morad et al. 2023).

**Deliverable:** POPGym-easy results in `benchmarks/phase_a/results/`.

### Weeks 4–6 — Phase A full benchmark sweep

**Tasks:**
1. Run on the full Phase A benchmark set:
   - POPGym: RepeatPrevious (Easy/Medium/Hard), Concentration, Battleship, Autoencode, MultiArmedBandit. Publication-quality (3 seeds, full curves).
   - MiniGrid-Memory: S5, S7, S9, S11, S13, all *with* the wrapper. Memory-only.
   - BSuite: memory_length, discounting_chain.
2. Run **architectural ablations** vs. baselines, structured per the tiered experiment plan (see "Phase A progress" section above). Internal IGAM ablations (α_t fixed, β_t fixed, fixed query) map onto cells we've already built:
   - **IGAM with α_t fixed** ≡ `DeltaNet` (already implemented)
   - **IGAM with β_t fixed** ≡ a `Gated DeltaNet` variant (add as constructor flag)
   - **IGAM with fixed query** ≡ closest analog is `LinearTransformer` (no dynamic W_Q)
   - **LSTM, GRU baselines** — already implemented
   - **Vanilla LMU baseline** ≡ `LMU` (canonical Voelker 2019, already implemented; the gated/W_pre thesis variant is left in `lmu_ppo` and not ported here)
   - **Mamba-2 baseline** — already implemented as `Mamba2`
   - **Plus**: `LinearTransformer`, `RetNet`, `S4D`, `mLSTM` for the broader 2×2 / SSM-family comparisons

3. **Decision point at end of Week 6:** Does IGAM beat LMU on ≥4/6 POPGym memory tasks?
   - **Yes:** advance to Phase B. Phase A is a success, write up the ablation table for the paper.
   - **No:** stop and diagnose. Two failure modes are most likely:
     - The matrix memory is over-fitting to recent keys (decay too aggressive). Fix: lower the α_t floor, ablate β_t.
     - The Legendre-basis prior of LMU is doing genuine work. Fix: hybrid — initialize W_V projection with HiPPO-LegS basis, recovering LMU as a strict special case. This is the principled fallback hybrid.
   - If after 2 weeks of debugging IGAM still doesn't match LMU, **escalate to a research decision**: maybe the cell choice is wrong and we should try mLSTM or RWKV-7 instead. Don't burn months on a wrong cell.

**Deliverable:** Phase A ablation table (the format that goes in the paper), all curves, all logs. Commit tagged `phase-a-complete`.

### Weeks 7–8 — Phase A buffer / write-up

Two weeks of margin. Use them for:
- Hyperparameter sensitivity sweeps on the headline tasks (3 seeds × 3 LR × 3 chunk lengths) so we know the result isn't a knife-edge.
- Writing the Phase A section of the paper's experimental ablation. This is good rest-of-research hygiene; if you wait until Week 22 to write up Week 6 results you'll have forgotten the details.
- Refactoring any module that the integration revealed needed cleanup.

---

## Phase B — Exploration validation (Weeks 9–18)

**Goal:** Add the encoder auxiliary, FSQ bottleneck, count-min sketch, and combined intrinsic bonus. Validate on memory + sparse-reward tasks. Beat the E3B-IDM baseline on ObstructedMaze. Crack the Craftax-symbolic exploration barrier.

### Week 9 — Encoder auxiliary, IDM only + leaderboard re-verification

**Tasks:**
1. **Re-verify the Craftax-symbolic 1M leaderboard.** Confirm the 2.3% PPO-RNN baseline is still the published ceiling for the strict MFRL+symbolic+1M topology. If a stronger baseline has been published in the intervening weeks, update ADR 0002 and recalibrate Phase B thresholds.
2. Implement `igam/encoder/cnn.py`, `igam/encoder/mlp.py`, `igam/encoder/fsq.py`.
3. Implement `igam/auxiliary/idm.py`: small MLP that predicts a_t from (φ(o_t), φ(o_{t+1})). Cross-entropy loss for discrete actions.
4. Implement `igam/auxiliary/ema.py`: EMA target maintenance for any φ-related target. (Used by JEPA in Week 11; placeholder for now.)
5. Wire IDM auxiliary into the training loop with a separate optimizer. φ is detached before entering the memory cell.

**Exit criterion:** IDM accuracy ≥0.6 on MiniGrid-Memory-S13 within 100k steps. Sanity check that the auxiliary loop runs.

### Week 10 — Episodic mechanism (FSQ + CMS)

**Tasks:**
1. Implement `igam/exploration/count_min_sketch.py`. Per-env CMS, H=4 hashes, W=2^16 width. Reset at episode boundaries.
2. Implement `igam/exploration/episodic.py`: r_epi_t = 1 / sqrt(N̂(c_t)), c_t = FSQ(φ(o_t)).
3. Implement `igam/exploration/lifelong.py`: r_life_t = ||δ_t||^2, with per-env Welford normalization.
4. Combine into r_int_t = r_epi_t · clip(r_life_t, 1, L) (NGU form).
5. Add r_int_t to the PPO reward stream with mask for episode_start.

**Exit criterion:** No-wrapper MiniGrid-Memory-S13 solved (eval reward ≥0.9 within 5M steps). This is the IGAM analog of lmu_ppo's random-φ E3B success on the same task. **If this fails, the issue is almost certainly in the exploration mechanism, not the cell — debug before continuing.**

**Diagnostic logs to add (non-optional):**
- FSQ codebook utilization per episode (fraction of cells touched).
- CMS bucket distribution histogram.
- Lifelong bonus normalization stats.
- Per-step compute time of episodic vs. lifelong vs. cell forward.

### Week 11 — Add JEPA latent prediction

**Tasks:**
1. Implement `igam/auxiliary/jepa.py`: predictor MLP P(φ(o_t), a_t) → φ_EMA(o_{t+1}); MSE loss with EMA target.
2. Combine with IDM: L_φ = λ_idm L_idm + λ_jepa L_jepa, both 1.0.
3. Re-run no-wrapper MiniGrid-Memory-S13 with hybrid auxiliary. Should match Week 10's result (this is a regression check, not an upgrade).

**Exit criterion:** No-wrapper Memory-S13 still solved. JEPA loss converges (no collapse to constant predictions).

### Weeks 12–14 — ObstructedMaze full sweep

**Tasks:**
1. Run on ObstructedMaze-2Dl, 2Dlh, 2Dlhb. 3 seeds each, 25M steps each.
2. Ablations:
   - Full IGAM (cell + IDM + JEPA + episodic + lifelong)
   - − JEPA (IDM only — Phase A-like auxiliary)
   - − episodic (lifelong only)
   - − lifelong (episodic only)
   - − FSQ (continuous + kNN episodic on raw φ)
   - − IDM + JEPA (random encoder, like lmu_ppo random-φ baseline)
3. Compare to E3B-IDM published numbers.

**Exit criterion:** ObstructedMaze-2Dlhb success ≥0.9 by 10M steps (E3B-IDM published is ~0.9 by 25M; we want to match it earlier).

### Weeks 15–17 — Craftax-symbolic

**Tasks:**
1. Wire Craftax via `igam/envs/craftax_wrappers.py`. Use Craftax-Classic-1M symbolic environment (8268-dim observation vector).
2. Run with full IGAM, 3 seeds, 5M steps (we run further than the 1M benchmark target to characterize convergence behavior beyond the published baselines).
3. Evaluate at the canonical 1M checkpoint and at 5M for comparison.
4. Compare to:
   - The 2.3% PPO-RNN baseline at 1M (ADR 0002).
   - The 2.2% PPO baselines (stateless, ICM, RND, E3B all at this tier).
   - Vanilla PPO + LSTM with our IDM+JEPA auxiliary (run ourselves; this isolates the cell's contribution from the auxiliary's contribution).
   - Vanilla PPO + IGAM cell *without* auxiliary or episodic mechanism (run ourselves; this isolates the auxiliary+episodic contribution from the cell's contribution).

**Success criteria, tiered:**
- **Tier 1 (publication-worthy minimum):** ≥5% reward at 1M steps. This is >2× the published PPO-RNN baseline and clears all standard intrinsic-motivation methods. Sufficient for paper inclusion as "matches model-based methods on symbolic exploration without a world model."
- **Tier 2 (strong result):** ≥10% reward at 1M steps. This crosses the threshold where memory starts mattering — at this reward level the agent is reaching mining/smelting/inventory-dependent decisions. Position the paper as "first model-free symbolic-PPO method to reach the memory-relevant achievement tier."
- **Tier 3 (headline result):** ≥20% reward at 1M steps. This puts IGAM in conversation with model-based methods (Simulus 6.6%, Efficient MBRL 5.4% on the same metric) and substantially exceeds them, despite operating without world-model machinery.
- **Tier 4 (stretch):** Beyond 20%, we're approaching DreamerV3-class pixel results without pixels. Highly publishable.

**Diagnostic interpretation thresholds:**
- If reward < 3%: the architecture isn't beating PPO-RNN. Investigate whether the auxiliary is poisoning φ or the episodic mechanism is not firing on inventory transitions. Likely two weeks of debugging.
- If reward 3-5%: marginal improvement, paper exists but is shaky. Run isolation ablations to identify which of {cell, IDM, JEPA, FSQ, count-min} is doing the work. The paper's narrative depends on which component the ablation isolates.
- If reward 5-10%: solid Tier 1 result. Run a Tier 2 stretch with 3 additional seeds at 5M steps to characterize convergence.
- If reward 10%+: Tier 2 hit. The exploration barrier is cracked. Add a "memory contribution" ablation (IGAM cell vs. LSTM, both with full auxiliary+episodic) to demonstrate that memory matters *now* in this regime.

**Deep diagnostic if reward < 5%:**
- Dump FSQ codes per state. Verify that "crafted sword" produces different code than "no sword" with otherwise identical observation. If not, the bottleneck is in the encoder; check IDM and JEPA training signal on inventory features.
- Check lifelong bonus on long episodes. If it decays to zero by episode-end (the matrix memory is "full"), data-dependent decay α_t is not aggressive enough.
- Iterate.

**Decision point at end of Week 17:**
- **Tier 2 or higher hit:** strong paper. Advance to Phase C with confidence. Begin paper writing for the IGAM contribution.
- **Tier 1 only (5-10%):** publishable but not headline. Decision: more debugging to push toward Tier 2 (2-4 weeks), or write the paper at Tier 1 and bank the result. Conservative: write the paper, submit, move on.
- **Below Tier 1 (<5%):** do not advance to Phase C. Spend 2-4 weeks diagnosing and either fix the gap or pivot the paper's framing toward "negative result on Craftax exploration via X mechanism" — which is itself publishable as a workshop note given how poorly all model-free methods do at this benchmark.

### Week 18 — Phase B buffer / write-up + parallel-scan revisit experiment

Margin week. Three uses:

**1. Parallel-scan revisit (per ADR 0001 footnote).** Now that the cell, encoder, and policy are converged on Phase A and B tasks, retry parallel-scan training on a small Phase A benchmark (MiniGrid-Memory-S13 with wrapper). If parallel-scan is stable in this warm-started configuration, we recover throughput for Phase C scaling. If it remains unstable, we have concrete evidence for the paper that the instability is not just a cold-start phenomenon.

**2. Phase B write-up.** Same purpose as Weeks 7–8.

**3. Refactor for Phase C readiness.** Pixel encoder stubs, MineCLIP wrapper plumbing, etc.

---

## Phase C — Scale (Weeks 19+, post-thesis)

**Goal:** Pixel domains. Stochasticity. Long horizons. The "does this generalize" phase.

This phase is the **conference paper's experimental section**, not the thesis. It happens after the master's thesis is submitted.

### Targets

- **Craftax-Full 1B**: full version, 1 billion frames. Stretch goal: match ≥10% reward (current SOTA 27.91% with TWM; model-free target is to clear PPO-RNN ≪5% by a wide margin).
- **Montezuma's Revenge with sticky actions**: stochasticity test. Add Curiosity-in-Hindsight head. Target ≥10k average return matching BYOL-Hindsight.
- **MineDojo programmatic tasks**: replace encoder with frozen MineCLIP. Run IDM + JEPA on MineCLIP features. Target: harvest-milk and shear-sheep ≥0.5 success matching MineCLIP baseline.

### New components needed in Phase C

- `igam/auxiliary/jepa_masked.py`: I-JEPA-style block-masked prediction for pixel domains.
- `igam/auxiliary/curiosity_hindsight.py`: Jarrett 2023 hindsight head for stochastic environments.
- `igam/encoder/mineclip.py`: MineCLIP wrapper, frozen.

These are not implemented in Phases A or B. Don't pre-build them. Build when needed.

---

## Engineering discipline (non-optional)

These are the practices that turn a research codebase into one that survives review and publication. Skipping them is the kind of decision that pays short-term and costs months later.

### Reproducibility

- **All experiments configured via YAML in `benchmarks/`.** No hyperparameters in code. No magic numbers in scripts. The benchmark configs are what we cite in the paper's reproducibility appendix.
- **Seed every run.** SB3-style: pass seed to env, network, and replay buffer. Log it.
- **Pin dependencies in `pyproject.toml`.** Use poetry or uv. Lock the file. RL bugs that come from dependency drift are nightmare bugs.
- **Each run produces a directory in `runs/{benchmark}/{config}/{seed}/`** with:
  - `config.yaml` (the exact config used)
  - `tensorboard/` logs
  - `checkpoints/`
  - `git_hash.txt` (commit hash at run start)
  - `env.txt` (`pip freeze` output)

### Testing

- **Unit tests for every module in `igam/`.** Aim for 70% coverage minimum. No exceptions for "research code."
- **Integration test that runs Phase A's MiniGrid-Memory-S13-wrapper to completion at small scale (100k steps) and verifies eval reward >0.5.** This is the smoke test. Run it on every commit to main via CI.
- **Determinism tests.** Set seed, run cell forward 100 steps, hash the output, commit the hash. If a refactor changes the hash, you've changed semantics. Investigate before merging.
- **The recurrent and parallel-scan training paths must produce identical outputs on the same input.** Test this when the parallel-scan path is implemented in Week 18. The Sherman-Morrison bug from lmu_ppo is what happens when this kind of invariant goes unchecked.

### Logging

- **Per-component gradient norms** logged every PPO update. From lmu_ppo; this caught the W_pre stability issue and will catch its IGAM analogs.
- **Memory state norms** logged every rollout. Watch for collapse (||W||→0) or explosion (||W||→∞).
- **Innovation magnitude distribution** logged every rollout. The lifelong bonus's input.
- **Cell-internal eigenvalues / singular values** sampled occasionally (every 100k steps). The IGAM analog of M_min_eigval; if W_t loses contraction, we want to know.
- **FSQ codebook utilization** logged every rollout. If <30% of codes are ever touched, something's wrong with the encoder.
- **CMS bucket distribution** logged every rollout. Watch for hash collisions saturating the sketch.

### Code review and ADRs

- **All non-trivial design decisions get an ADR in `docs/adr/`.** ADR format: one page, sections "Context / Decision / Consequences / Alternatives considered". By Week 18 you'll have ~20 ADRs and they're the source of truth for the paper's methods writeup.
- **Self-review every PR before merging to main.** Even solo: open a PR, read the diff, check it against the design spec. This is the single highest-leverage discipline in solo research.

### When to break the plan

The plan is a contract with future-Jai. Break it when:

- **Phase A reveals a fundamental cell issue** that requires switching from Gated DeltaNet to mLSTM or RWKV-7. Update plan, document the decision in an ADR, continue.
- **Phase B reveals that the JEPA auxiliary is doing nothing** (ablation shows no improvement over IDM-only). Drop it. Simpler is better.
- **A new paper publishes** that solves the problem we're working on. Read it. If it dominates IGAM, the contribution may shift; if it's complementary, cite it. Either way, the plan adjusts.

Don't break it for:

- **A clever idea you had on Tuesday.** If it's still clever on Friday, write an ADR and decide deliberately.
- **A failure mode that's two weeks of debugging away from resolution.** Debug first; rearchitect only if debugging fails.
- **Reviewer fatigue or impatience.** The plan exists because research moves slower than enthusiasm.

---

## What success looks like

**Phase A success:** IGAM matches or beats LMU on POPGym, MiniGrid-Memory, BSuite. Ablation table shows that data-dependent gating, dynamic query, and the matrix-memory write rule each contribute. Paper has its memory-cell contribution.

**Phase B success:** ObstructedMaze-2Dlhb solved at ≥0.9 by 10M steps. Craftax-symbolic-1M reward ≥5% (Tier 1) — a >2× improvement on the published 2.3% PPO-RNN baseline. Ablations show the lifelong-from-innovation, the FSQ episodic, and the IDM+JEPA auxiliary each contribute. Paper has its exploration contribution and the unified-objective story.

**Phase C success (post-thesis):** Craftax-Full ≥10% reward, Montezuma sticky ≥10k return, MineDojo-harvest-milk ≥0.5. Paper has cross-task validation and is conference-ready.

**The thesis success:** Phases A and B done; the four-failures empirical story (from lmu_ppo) presented as the methodological foundation; the IGAM architecture presented as the synthesis. Defended.

---

## Risks I'm tracking

- **Cell training instability under PPO** (resolved by ADR 0001 with single-step recurrent fallback). Residual risk: instability in single-step mode itself, which would indicate a deeper architectural problem requiring cell-class change.
- **Encoder collapse** despite EMA target. Mitigation: VICReg regularization is the planned-but-deferred fix.
- **FSQ codebook under-utilization in Craftax**. Mitigation: VICReg on φ to spread codes; or train FSQ on inverse-dynamics features specifically rather than on full φ.
- **The unification framing being rejected by reviewers** as "you just glued three modules together and called it unified." Mitigation: lean on the shared-objective math (innovation = write magnitude = surprisal); acknowledge in the paper that this is an *architectural* unification, not a fully-derived information-theoretic one.
- **Time. Master's thesis defense is a fixed deadline.** Phases A and B are the thesis; Phase C is post-thesis. Don't let Phase C ambition delay thesis submission.
- **The 2.3% baseline is so low that any moderate gain looks impressive, but the absolute bar for "this matters scientifically" is also low.** Mitigation: lean on the *ablation table* rather than the headline number. A 5% headline + clean ablations isolating cell, auxiliary, and episodic contributions is more publishable than a 15% headline without ablations.
- **Wall-clock cost of single-step recurrent training is real.** Phase A benchmarks at 5M-25M steps × 3 seeds × multiple ablations may run 3× longer than originally estimated. Mitigation: invest in smaller-scale sanity sweeps (1M steps, 1 seed) before committing 3-seed full runs; use the parallel-scan-revisit experiment in Week 18 to recover throughput before Phase C scaling.
- **Phase B's exploration-bottleneck framing depends on the 2.3% baseline being correct.** If a recent paper has published a stronger MFRL+symbolic+1M result we missed, the framing shifts. Mitigation: re-verify the leaderboard at the start of Week 9 (Phase B start) before committing to Tier 1/2/3 thresholds.

---

## Notation reference

| Symbol | Meaning |
|---|---|
| φ | Encoder, observations → R^d_φ |
| φ_EMA | EMA target encoder |
| W_t | Matrix memory state at time t |
| n_t | Normalizer state, maintained alongside W_t (see Memory cell) |
| k_t, v_t, q_t | Key, value, query at time t |
| δ_t | Innovation: v_t − W_{t-1} k_t / ‖k_t‖² |
| α_t, β_t | Data-dependent decay and learning rate gates |
| c_t | FSQ code: discretized φ(o_t) |
| N̂_e(c) | Per-env count-min estimate of code c's frequency this episode |
| r^life, r^epi, r^int | Lifelong, episodic, combined intrinsic reward |
| L_φ, L_idm, L_jepa | Encoder auxiliary losses |
| λ_idm, λ_jepa, β_int | Loss/reward weights |

---

End of plan. Updates to this document go through git history; don't edit in place without committing.