# IGAM: Innovation-Gated Associative Memory
## Implementation Plan

**Author:** Jai
**Last updated:** May 2026
**Status:** Pre-implementation — design frozen, ready for Week 1.

---

## What this document is

This is the canonical design and execution plan for IGAM, the memory + exploration
architecture that succeeds the lmu_ppo thesis work. It exists to:

1. Lock in the architectural commitments before implementation, so we don't
   relitigate decisions mid-debugging.
2. Define exit criteria for each phase, so we know when to advance, when to
   debug, and when to bail.
3. Record the rationale for non-obvious choices, so future-Jai (or a collaborator)
   can audit the design without reconstructing six months of context.

It is **not** a research proposal. The research case is in the deep-research
output (`docs/igam-deep-research.md`); that document is the *why*. This is
the *how*.

---

## Relationship to the thesis codebase

`lmu_ppo` (the thesis codebase) is **frozen**. It is the experimental vehicle
for the four-failures empirical story and should be preserved as a reproducible
thesis artifact. Cherry-pick from it; do not modify it.

What we steal from `lmu_ppo`:

- The TBPTT rollout buffer pattern (`LMURolloutBuffer`'s chunked layout). This
  is genuinely good infrastructure and we'd rebuild it identically.
- The episode-boundary masking convention (zero memory state at episode_starts).
- The diagnostic logging discipline (per-component grad norms, per-channel
  activations, eigenvalue tracking from `episodic_bonus.py`).
- The PPO integration scaffolding (`LMUPPO` collect_rollouts → train loop).

What we leave behind:

- `LMUCell`, `LegSCell`, `lmu_t.py`, `lmu_s.py` — the Legendre memory cells.
  IGAM uses Gated DeltaNet; LegT/LegS do not survive.
- `OrthoLayer` and Cayley updates — unnecessary, the delta rule's contraction
  bound replaces them.
- `EllipticalEpisodicBonus` and Sherman-Morrison machinery — replaced by
  FSQ + count-min sketch.
- The HSWVIME stack (`hswvime_ppo/`, `models/wyner.py`, `models/wyner_lmu.py`,
  `models/vae.py`, `models/qa_module.py`) — the failure-mode that motivated
  the entire trajectory; cited in the paper's Background, not used.

---

## Architectural commitments (frozen)

These are the design decisions that define IGAM. Changing any of these is a
research decision, not an implementation decision; if you find yourself wanting
to change one mid-implementation, escalate to a design review.

### Memory cell

- **Gated DeltaNet** with RWKV-7-style vector-valued in-context learning rate
  and decay gates.
- Matrix state W_t ∈ R^(d_k × d_v), head-factored into H heads.
- Update rule: W_t = α_t W_{t-1} (I − β_t k_t k_t^T / ||k_t||^2) + β_t v_t k_t^T
- Read: y_t = q_t^T W_t / (q_t^T n_t + ε), with normalizer n_t maintained alongside.
- α_t = σ(W_α h_t), β_t = σ(W_β h_t), data-dependent gates.
- Innovation δ_t = v_t − W_{t-1} k_t / ||k_t||^2 — exposed as a side output
  for the lifelong intrinsic reward.

### Encoder and auxiliary loss

- φ: O → R^d_φ. CNN for image obs, MLP for symbolic.
- Auxiliary loss is **hybrid IDM + JEPA-style latent prediction**:
  - IDM: -log p(a_t | φ(o_t), φ(o_{t+1}))
  - JEPA: ||P(φ(o_t), a_t) − φ_EMA(o_{t+1})||^2, with EMA target (τ=0.99)
  - L_φ = λ_idm L_idm + λ_jepa L_jepa, both 1.0 by default.
- φ is **detached** before entering the memory cell. The auxiliary optimizer
  is separate from PPO's optimizer.
- Phase A uses IDM only (POPGym is symbolic, MiniGrid is spatial bottleneck).
- Phase B adds JEPA prediction. Phase C adds I-JEPA-style block masking.

### Exploration

- Lifelong intrinsic reward: r_life_t = ||δ_t||^2, normalized by per-env
  Welford running mean/std.
- Episodic intrinsic reward: r_epi_t = 1 / sqrt(N̂_e(c_t)) where c_t = FSQ(φ(o_t))
  is the discrete code and N̂ is a per-env count-min sketch (H=4 hashes,
  W=2^16 width).
- Combined: r_int_t = r_epi_t · clip(r_life_t, 1, L)  (NGU multiplicative form).
- Total reward: r_t = r_ext_t + β_int · r_int_t · (1 − episode_start_t).
- β_int = 0.01 default; ablate.

### FSQ bottleneck

- Project φ output to d_fsq ∈ [3, 8] dims, bound (tanh), round to L_i levels.
- Phase A/B: d_fsq=6, L=5 (codebook ~15,625).
- Phase C (Craftax-pixel, Montezuma, MineDojo): d_fsq=8, L=8 (codebook ~16M),
  applied per spatial location for pixel obs.
- No commitment loss, no learned codebook (FSQ is a stop-gradient quantization).

### PPO integration

- TBPTT chunks of length K=16, identical to lmu_ppo.
- Episode_starts mask zeros (h, W, n) at boundaries within a chunk.
- Memory state stored at chunk *start* in the rollout buffer; intra-chunk
  states are recomputed during evaluate_actions.
- Episodic state (count-min sketch, FSQ) reset at episode boundaries.

---

## Repository layout
igam/
├── README.md
├── plan.md                    # this document
├── docs/
│   ├── igam-deep-research.md  # the research synthesis
│   ├── thesis-retrospective.md # the four-failures story
│   └── adr/                   # architecture decision records (one per major choice)
├── igam/
│   ├── init.py
│   ├── cell/
│   │   ├── init.py
│   │   ├── gated_deltanet.py  # the matrix-memory cell
│   │   └── tests/
│   │       └── test_synthetic_recall.py  # Week 1 sanity tests
│   ├── encoder/
│   │   ├── init.py
│   │   ├── cnn.py             # image encoder
│   │   ├── mlp.py             # symbolic encoder
│   │   └── fsq.py             # finite scalar quantization
│   ├── auxiliary/
│   │   ├── init.py
│   │   ├── idm.py             # inverse dynamics model
│   │   ├── jepa.py            # latent prediction with EMA target
│   │   └── ema.py             # EMA target maintenance
│   ├── exploration/
│   │   ├── init.py
│   │   ├── count_min_sketch.py
│   │   ├── lifelong.py        # innovation-magnitude bonus
│   │   └── episodic.py        # FSQ + CMS episodic bonus
│   ├── policy/
│   │   ├── init.py
│   │   ├── igam_policy.py     # actor-critic with IGAM cell
│   │   └── buffer.py          # TBPTT rollout buffer (ported from lmu_ppo)
│   ├── ppo/
│   │   ├── init.py
│   │   └── igam_ppo.py        # PPO with IGAM-aware rollout collection
│   └── envs/
│       ├── init.py
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



A few things worth noting about this layout:

- **The cell, encoder, auxiliary, and exploration modules are independent
  packages.** This is deliberate. The cell should be testable on synthetic
  recall without any PPO scaffolding. The encoder + auxiliary should be
  testable on offline trajectories. The exploration module should be testable
  with mock features. We do not want the kind of monolith where you can only
  test the cell by running 100k PPO steps.

- **`docs/adr/` for architecture decision records.** Every time a non-obvious
  decision gets made (cell hyperparameter, auxiliary loss weight, FSQ
  configuration), drop a 1-page ADR explaining what we tried, what worked,
  what we picked, and why. By Week 12 you'll have ~15 of these and they'll
  be the source of truth when writing the paper's methods section.

- **`benchmarks/` separate from `scripts/`.** Benchmarks are the experimental
  configurations we run for the paper. Scripts are dev-time utilities. Keep
  them apart so the paper's reproducibility section can point at
  `benchmarks/` cleanly.

- **`tests/` matters.** This is RL code; bugs are silent. The Sherman-Morrison
  bug from lmu_ppo is the cautionary tale. Unit tests on the cell, encoder,
  buffer, and FSQ are non-optional. Aim for ≥70% coverage of the core modules.

---

## Phase 0: Pre-implementation (Week 0)

Before writing any IGAM code, two focused research sessions to derisk Phase A.
These were flagged in the JEPA discussion. **Both should resolve to a written
paragraph in `docs/` before Week 1 begins.**

### 0.1 Linear-attention cells under on-policy PPO

**Question:** Are there published reports of training instability when running
Gated DeltaNet, mLSTM, or RWKV-7 under on-policy PPO with TBPTT chunks?

**Why:** If yes, the contingency (fall back to single-step recurrent training,
accept ~3× wall-clock cost) is a real Phase A engineering decision and should
be planned for, not discovered. If no, we use the parallel-scan training path.

**How:** 1 day of literature review. Search terms: "DeltaNet PPO", "linear
attention reinforcement learning", "Mamba on-policy", "RWKV reinforcement
learning". Also check OpenReview for ICLR/NeurIPS 2024–2025 RL submissions
using these cells.

**Deliverable:** `docs/adr/0001-cell-training-mode.md` documenting findings
and the chosen training path (parallel scan vs. recurrent fallback).

### 0.2 Craftax-symbolic model-free PPO baseline

**Question:** What is the strongest published number on Craftax-symbolic 1M
for a model-free PPO + recurrent policy (no world model, no transformer,
no auxiliary world-model loss)?

**Why:** Phase B's success criterion needs to be set against a comparable
baseline. The Cohen 2025 TWM number (69.66%) is not comparable — it's a
transformer world model with a Dyna loop. The right comparator is the
strongest model-free recurrent PPO published.

**How:** 1 day. Read the original Craftax paper's baseline table; check the
Craftax leaderboard if one exists; search recent Craftax papers for "PPO
baseline" or "PPO-RNN baseline".

**Deliverable:** `docs/adr/0002-craftax-baseline.md` with the chosen target
number for Phase B.

---

## Phase A — Memory cell validation (Weeks 1–8)

**Goal:** Establish that Gated DeltaNet under on-policy PPO with TBPTT works,
matches or exceeds the lmu_ppo LMU on memory-only tasks, and is the right
foundation for Phases B and C.

**No exploration mechanism in Phase A.** Episodic bonus, lifelong bonus, FSQ —
all disabled. Only the cell + encoder + PPO. This is the critical isolation
discipline from the thesis (Failure 2: "stage isolation as methodology").

### Week 1 — Cell prototype + synthetic test

**Tasks:**
1. Implement `igam/cell/gated_deltanet.py`. Reference impl:
   `flash-linear-attention` library on GitHub. Port the chunkwise parallel
   form (WY-representation Householder products) and the recurrent
   single-step form. Both must produce identical outputs (test this).
2. Implement `igam/cell/tests/test_synthetic_recall.py`:
   - Multi-Query Associative Recall (MQAR) task: input is a sequence of
     (key, value) pairs followed by query keys; predict the matching values.
     Standard benchmark for associative memory.
   - Selective copy task: copy a subset of marked tokens from input to output.
   - Both tasks have known scaling laws; the cell should solve both at
     small scale (sequence length ≤ 256, vocab ≤ 64).

**Exit criterion (HARD):** MQAR accuracy ≥0.95 on length-256 sequences with
8-token vocabulary, training in <10 minutes on a single GPU. **If this fails,
the cell has a bug. Do not proceed to Week 2 until it passes.**

**Deliverable:** A working cell, a passing synthetic test, and a commit
tagged `cell-v0.1`.

### Week 2 — Wire cell into PPO, regression test on lmu_ppo task

**Tasks:**
1. Port the rollout buffer from `lmu_ppo/buffer.py` to `igam/policy/buffer.py`.
   Replace LMU state shape (h, m) with IGAM state shape (h, W, n). The
   chunked TBPTT layout is unchanged.
2. Implement `igam/policy/igam_policy.py`: actor-critic with the IGAM cell
   between encoder and policy/value heads. Mirror `lmu_ppo/policies.py`
   structurally; only the cell call differs.
3. Implement `igam/ppo/igam_ppo.py`: PPO with IGAM rollout collection.
   Mirror `lmu_ppo/lmu_ppo.py`; remove the E3B and lifelong-bonus code.
4. Run on **MiniGrid-Memory-S13 with the MemoryStartWrapper** (the regression
   benchmark from the thesis). No exploration bonus. Just memory.

**Exit criterion (SOFT):** IGAM matches or beats lmu_ppo's LMU baseline on
MiniGrid-Memory-S13-with-wrapper within ±20%. If IGAM underperforms by more
than 20%, debug the integration before Week 3 — most likely culprit is the
cell init or the actor-critic head dimensions.

**Likely failure modes to watch for:**
- Cell initialization too aggressive; W_t saturates in first 100 steps.
  Fix: smaller init for W_K, W_V, W_Q.
- Normalizer n_t numerically unstable. Fix: layer-norm on y_t before policy head.
- TBPTT chunk boundaries not respected. Fix: verify episode_starts masking
  zeros all of (h, W, n), not just (h, W).

**Deliverable:** End-to-end MiniGrid-Memory-S13 + wrapper run, training curves
checked into `benchmarks/phase_a/results/`. Commit tagged `e2e-v0.1`.

### Week 3 — POPGym sanity check

**Tasks:**
1. Wire POPGym environments via `igam/envs/popgym_wrappers.py`. Port the
   lmu_ppo POPGym integration if there is one.
2. Run on the easier POPGym tasks: RepeatPrevious-Easy, Concentration-Easy,
   Battleship-Easy. These are the canary tests — if IGAM fails here, it
   won't work on the harder benchmarks.

**Exit criterion (SOFT):** Match or exceed published GRU/LSTM baselines on
the three easy tasks within 1M steps. Numbers are in the POPGym paper
(Morad et al. 2023).

**Deliverable:** POPGym-easy results in `benchmarks/phase_a/results/`.

### Weeks 4–6 — Phase A full benchmark sweep

**Tasks:**
1. Run on the full Phase A benchmark set:
   - POPGym: RepeatPrevious (Easy/Medium/Hard), Concentration, Battleship,
     Autoencode, MultiArmedBandit. Publication-quality (3 seeds, full curves).
   - MiniGrid-Memory: S5, S7, S9, S11, S13, all *with* the wrapper. Memory-only.
   - BSuite: memory_length, discounting_chain.
2. Run **architectural ablations** vs. baselines:
   - IGAM (full)
   - IGAM with α_t fixed (no data-dependent decay)
   - IGAM with β_t fixed (no data-dependent learning rate)
   - IGAM with fixed query (no dynamic W_Q) — this is the closest analog
     to vanilla LMU; the comparison is critical for the paper's narrative
     about W_Q.
   - LSTM baseline
   - GRU baseline
   - Vanilla LMU baseline (port from lmu_ppo)
   - Mamba-2 baseline (use a published reference impl)

3. **Decision point at end of Week 6:** Does IGAM beat LMU on ≥4/6 POPGym
   memory tasks?
   - **Yes:** advance to Phase B. Phase A is a success, write up the
     ablation table for the paper.
   - **No:** stop and diagnose. Two failure modes are most likely:
     - The matrix memory is over-fitting to recent keys (decay too aggressive).
       Fix: lower the α_t floor, ablate β_t.
     - The Legendre-basis prior of LMU is doing genuine work. Fix: hybrid —
       initialize W_V projection with HiPPO-LegS basis, recovering LMU as
       a strict special case. This is the principled fallback hybrid.
   - If after 2 weeks of debugging IGAM still doesn't match LMU, **escalate
     to a research decision**: maybe the cell choice is wrong and we should
     try mLSTM or RWKV-7 instead. Don't burn months on a wrong cell.

**Deliverable:** Phase A ablation table (the format that goes in the paper),
all curves, all logs. Commit tagged `phase-a-complete`.

### Weeks 7–8 — Phase A buffer / write-up

Two weeks of margin. Use them for:
- Hyperparameter sensitivity sweeps on the headline tasks (3 seeds × 3 LR ×
  3 chunk lengths) so we know the result isn't a knife-edge.
- Writing the Phase A section of the paper's experimental ablation. This is
  good rest-of-research hygiene; if you wait until Week 22 to write up Week 6
  results you'll have forgotten the details.
- Refactoring any module that the integration revealed needed cleanup.

---

## Phase B — Exploration validation (Weeks 9–18)

**Goal:** Add the encoder auxiliary, FSQ bottleneck, count-min sketch, and
combined intrinsic bonus. Validate on memory + sparse-reward tasks. Beat the
E3B-IDM baseline on ObstructedMaze. Make Craftax-symbolic work.

### Week 9 — Encoder auxiliary, IDM only

**Tasks:**
1. Implement `igam/encoder/cnn.py`, `igam/encoder/mlp.py`, `igam/encoder/fsq.py`.
2. Implement `igam/auxiliary/idm.py`: small MLP that predicts a_t from
   (φ(o_t), φ(o_{t+1})). Cross-entropy loss for discrete actions.
3. Implement `igam/auxiliary/ema.py`: EMA target maintenance for any φ-related
   target. (Used by JEPA in Week 11; placeholder for now.)
4. Wire IDM auxiliary into the training loop with a separate optimizer.
   φ is detached before entering the memory cell.

**Exit criterion:** IDM accuracy ≥0.6 on MiniGrid-Memory-S13 within 100k
steps. Sanity check that the auxiliary loop runs.

### Week 10 — Episodic mechanism (FSQ + CMS)

**Tasks:**
1. Implement `igam/exploration/count_min_sketch.py`. Per-env CMS, H=4
   hashes, W=2^16 width. Reset at episode boundaries.
2. Implement `igam/exploration/episodic.py`: r_epi_t = 1 / sqrt(N̂(c_t)),
   c_t = FSQ(φ(o_t)).
3. Implement `igam/exploration/lifelong.py`: r_life_t = ||δ_t||^2, with
   per-env Welford normalization.
4. Combine into r_int_t = r_epi_t · clip(r_life_t, 1, L) (NGU form).
5. Add r_int_t to the PPO reward stream with mask for episode_start.

**Exit criterion:** No-wrapper MiniGrid-Memory-S13 solved (eval reward ≥0.9
within 5M steps). This is the IGAM analog of lmu_ppo's random-φ E3B success
on the same task. **If this fails, the issue is almost certainly in the
exploration mechanism, not the cell — debug before continuing.**

**Diagnostic logs to add (non-optional):**
- FSQ codebook utilization per episode (fraction of cells touched).
- CMS bucket distribution histogram.
- Lifelong bonus normalization stats.
- Per-step compute time of episodic vs. lifelong vs. cell forward.

### Week 11 — Add JEPA latent prediction

**Tasks:**
1. Implement `igam/auxiliary/jepa.py`: predictor MLP P(φ(o_t), a_t) →
   φ_EMA(o_{t+1}); MSE loss with EMA target.
2. Combine with IDM: L_φ = λ_idm L_idm + λ_jepa L_jepa, both 1.0.
3. Re-run no-wrapper MiniGrid-Memory-S13 with hybrid auxiliary. Should
   match Week 10's result (this is a regression check, not an upgrade).

**Exit criterion:** No-wrapper Memory-S13 still solved. JEPA loss converges
(no collapse to constant predictions).

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

**Exit criterion:** ObstructedMaze-2Dlhb success ≥0.9 by 10M steps (E3B-IDM
published is ~0.9 by 25M; we want to match it earlier).

### Weeks 15–17 — Craftax-symbolic

**Tasks:**
1. Wire Craftax via `igam/envs/craftax_wrappers.py`. Use Craftax-Classic-1M
   (the symbolic variant).
2. Run with full IGAM, 3 seeds, 5M steps.
3. Compare to:
   - The Phase B baseline established in Phase 0.2.
   - Vanilla PPO + LSTM (run this ourselves for apples-to-apples).
4. **Deep diagnostic if reward < 60%:**
   - Dump FSQ codes per state. Verify that "crafted sword" produces
     different code than "no sword" with otherwise identical observation.
     If not, the bottleneck is in the encoder; check IDM and JEPA training
     signal on inventory features.
   - Check lifelong bonus on long episodes. If it decays to zero by
     episode-end (the matrix memory is "full"), data-dependent decay α_t
     is not aggressive enough.
5. Iterate.

**Exit criterion (SOFT):** Craftax-Classic-1M reward ≥60%. Stretch: ≥70%
(approaches Cohen 2025's TWM SOTA without a world model).

**Decision point at end of Week 17:** Does IGAM beat the Phase B baseline
on Craftax?
- **Yes by a wide margin (≥10pp):** strong paper. Advance to Phase C
  with confidence.
- **Yes by a small margin or matches (within ±5pp):** publishable, but the
  paper's headline is shaky. Consider running additional ablations to
  isolate which component is doing the work, and consider whether to
  position as "matches SOTA without a world model" rather than "beats SOTA".
- **No:** stop and diagnose. The most likely issues are encoder capacity
  (CNN too small for symbolic obs structure) or auxiliary balance
  (λ_jepa needs tuning for Craftax specifically). Budget 2–4 weeks of
  debugging before deciding whether Phase C is realistic.

### Week 18 — Phase B buffer / write-up

Margin week. Same purpose as Week 7–8.

---

## Phase C — Scale (Weeks 19+, post-thesis)

**Goal:** Pixel domains. Stochasticity. Long horizons. The "does this generalize"
phase.

This phase is the **conference paper's experimental section**, not the thesis.
It happens after the master's thesis is submitted.

### Targets

- **Craftax-Full 1B**: full version, 1 billion frames. Stretch goal: match
  ≥10% reward (current SOTA 27.91% with TWM; model-free target is to clear
  PPO-RNN ≪5% by a wide margin).
- **Montezuma's Revenge with sticky actions**: stochasticity test. Add
  Curiosity-in-Hindsight head. Target ≥10k average return matching
  BYOL-Hindsight.
- **MineDojo programmatic tasks**: replace encoder with frozen MineCLIP.
  Run IDM + JEPA on MineCLIP features. Target: harvest-milk and shear-sheep
  ≥0.5 success matching MineCLIP baseline.

### New components needed in Phase C

- `igam/auxiliary/jepa_masked.py`: I-JEPA-style block-masked prediction
  for pixel domains.
- `igam/auxiliary/curiosity_hindsight.py`: Jarrett 2023 hindsight head
  for stochastic environments.
- `igam/encoder/mineclip.py`: MineCLIP wrapper, frozen.

These are not implemented in Phases A or B. Don't pre-build them. Build
when needed.

---

## Engineering discipline (non-optional)

These are the practices that turn a research codebase into one that survives
review and publication. Skipping them is the kind of decision that pays
short-term and costs months later.

### Reproducibility

- **All experiments configured via YAML in `benchmarks/`.** No hyperparameters
  in code. No magic numbers in scripts. The benchmark configs are what we
  cite in the paper's reproducibility appendix.
- **Seed every run.** SB3-style: pass seed to env, network, and replay buffer.
  Log it.
- **Pin dependencies in `pyproject.toml`.** Use poetry or uv. Lock the file.
  RL bugs that come from dependency drift are nightmare bugs.
- **Each run produces a directory in `runs/{benchmark}/{config}/{seed}/`** with:
  - `config.yaml` (the exact config used)
  - `tensorboard/` logs
  - `checkpoints/`
  - `git_hash.txt` (commit hash at run start)
  - `env.txt` (`pip freeze` output)

### Testing

- **Unit tests for every module in `igam/`.** Aim for 70% coverage minimum.
  No exceptions for "research code."
- **Integration test that runs Phase A's MiniGrid-Memory-S13-wrapper to
  completion at small scale (100k steps) and verifies eval reward >0.5.**
  This is the smoke test. Run it on every commit to main via CI.
- **Determinism tests.** Set seed, run cell forward 100 steps, hash the
  output, commit the hash. If a refactor changes the hash, you've changed
  semantics. Investigate before merging.
- **The recurrent and parallel-scan training paths must produce identical
  outputs on the same input.** Test this. The Sherman-Morrison bug from
  lmu_ppo is what happens when this kind of invariant goes unchecked.

### Logging

- **Per-component gradient norms** logged every PPO update. From lmu_ppo;
  this caught the W_pre stability issue and will catch its IGAM analogs.
- **Memory state norms** logged every rollout. Watch for collapse (||W||→0)
  or explosion (||W||→∞).
- **Innovation magnitude distribution** logged every rollout. The lifelong
  bonus's input.
- **Cell-internal eigenvalues / singular values** sampled occasionally
  (every 100k steps). The IGAM analog of M_min_eigval; if W_t loses
  contraction, we want to know.
- **FSQ codebook utilization** logged every rollout. If <30% of codes are
  ever touched, something's wrong with the encoder.
- **CMS bucket distribution** logged every rollout. Watch for hash collisions
  saturating the sketch.

### Code review and ADRs

- **All non-trivial design decisions get an ADR in `docs/adr/`.** ADR format:
  one page, sections "Context / Decision / Consequences / Alternatives
  considered". By Week 18 you'll have ~20 ADRs and they're the source of
  truth for the paper's methods writeup.
- **Self-review every PR before merging to main.** Even solo: open a PR,
  read the diff, check it against the design spec. This is the single
  highest-leverage discipline in solo research.

### When to break the plan

The plan is a contract with future-Jai. Break it when:

- **Phase A reveals a fundamental cell issue** that requires switching
  from Gated DeltaNet to mLSTM or RWKV-7. Update plan, document the
  decision in an ADR, continue.
- **Phase B reveals that the JEPA auxiliary is doing nothing** (ablation
  shows no improvement over IDM-only). Drop it. Simpler is better.
- **A new paper publishes** that solves the problem we're working on. Read
  it. If it dominates IGAM, the contribution may shift; if it's
  complementary, cite it. Either way, the plan adjusts.

Don't break it for:

- **A clever idea you had on Tuesday.** If it's still clever on Friday,
  write an ADR and decide deliberately.
- **A failure mode that's two weeks of debugging away from resolution.**
  Debug first; rearchitect only if debugging fails.
- **Reviewer fatigue or impatience.** The plan exists because research
  moves slower than enthusiasm.

---

## What success looks like

**Phase A success:** IGAM matches or beats LMU on POPGym, MiniGrid-Memory,
BSuite. Ablation table shows that data-dependent gating, dynamic query, and
the matrix-memory write rule each contribute. Paper has its memory-cell
contribution.

**Phase B success:** ObstructedMaze-2Dlhb solved at ≥0.9 by 10M steps.
Craftax-Classic-1M reward ≥60%. Ablations show the lifelong-from-innovation,
the FSQ episodic, and the IDM+JEPA auxiliary each contribute. Paper has
its exploration contribution and the unified-objective story.

**Phase C success (post-thesis):** Craftax-Full ≥10% reward, Montezuma sticky
≥10k return, MineDojo-harvest-milk ≥0.5. Paper has cross-task validation
and is conference-ready.

**The thesis success:** Phases A and B done; the four-failures empirical
story (from lmu_ppo) presented as the methodological foundation; the IGAM
architecture presented as the synthesis. Defended.

---

## Risks I'm tracking

- **Cell training instability under PPO** (Phase 0.1 question). Mitigation:
  fall back to recurrent single-step training if parallel scan is unstable.
- **Encoder collapse** despite EMA target. Mitigation: VICReg regularization
  is the planned-but-deferred fix.
- **FSQ codebook under-utilization in Craftax**. Mitigation: VICReg on φ
  to spread codes; or train FSQ on inverse-dynamics features specifically
  rather than on full φ.
- **The unification framing being rejected by reviewers** as "you just glued
  three modules together and called it unified." Mitigation: lean on the
  shared-objective math (innovation = write magnitude = surprisal);
  acknowledge in the paper that this is an *architectural* unification,
  not a fully-derived information-theoretic one.
- **Time. Master's thesis defense is a fixed deadline.** Phases A and B are
  the thesis; Phase C is post-thesis. Don't let Phase C ambition delay
  thesis submission.

---

## Notation reference

| Symbol | Meaning |
|---|---|
| $\phi$ | Encoder, observations → R^d_φ |
| $\phi_{\text{EMA}}$ | EMA target encoder |
| $W_t$ | Matrix memory state at time t |
| $h_t$ | Small recurrent hidden state alongside W_t |
| $k_t, v_t, q_t$ | Key, value, query at time t |
| $\delta_t$ | Innovation: v_t − W_{t-1} k_t / ‖k_t‖² |
| $\alpha_t, \beta_t$ | Data-dependent decay and learning rate gates |
| $c_t$ | FSQ code: discretized φ(o_t) |
| $\hat{N}_e(c)$ | Per-env count-min estimate of code c's frequency this episode |
| $r^{\text{life}}, r^{\text{epi}}, r^{\text{int}}$ | Lifelong, episodic, combined intrinsic reward |
| $L_\phi, L_{\text{idm}}, L_{\text{jepa}}$ | Encoder auxiliary losses |
| $\lambda_{\text{idm}}, \lambda_{\text{jepa}}, \beta_{\text{int}}$ | Loss/reward weights |

---

End of plan. Updates to this document go through git history; don't edit in
place without committing.
