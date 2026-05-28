# Memory × Exploration in Deep RL — Study Plan v2

**Status:** Learning project → paper. Thesis already complete; this systematizes its
central observation. No hard deadline, but scoped to ~6–8 weeks of focused work.

**Compute:** 1× B100, 1× TitanX, 1× RTX 3070 Ti. ~30–120 GPU-days.
**Algorithm:** PPO (`MemPPO` + cell registry + `IntrinsicRewardModule`).

---

## 0. The thesis (the novel spine)

> **Memory-architecture strength and exploration bonuses are partially substitutable
> for solving sparse-reward memory POMDPs.** A weaker memory cell paired with the right
> exploration head can match or beat a stronger cell with no exploration.

Concrete hook (already in hand from the thesis): **LMU + E3B solves MiniGrid Memory-S13
in ~3M steps.** RLBenchNet (2025) shows that *without* exploration, only Transformer-XL
and Mamba-2 cross the learning threshold on Memory-S11 — every weaker cell fails outright.
So an *inferior* cell + exploration solved what a *superior* cell needs its full
representational edge to do. Nobody has mapped this trade-off.

**Why this is novel.** The literature studies one axis at a time:
- Cell zoos with plain PPO/PQN, no bonus: RLBenchNet, POPGym, POPGym Arcade.
- Exploration bonuses with a fixed architecture (LSTM / frame-stack): RND, E3B, NovelD, NGU.
- Memory-vs-credit with transformers, no bonus: Ni et al. 2023.

The systematic **{6 memory cells} × {4 exploration heads}** interaction matrix on POMDP
memory tasks is, to our knowledge, unstudied. The substitutability finding reframes the
field's "find the best cell" project: maybe the bottleneck is exploration, not memory
capacity.

---

## 1. What changed from v1 (and why)

| v1 | v2 | Reason |
|---|---|---|
| 8 cells | **6 cells** (drop S4D, mLSTM) | S4D≈LRU (both diagonal SSM); mLSTM≈SHM (both matrix-state). Keep one representative each. Every cell is a validation liability. |
| 12 hypotheses | **5 core + 3 free side-analyses** | The marginal ones (H6, H11, H12-as-finding) dilute focus. Reorganize the strong ones around the substitutability thesis. |
| 18 envs / 21 configs | **~9 envs** spanning 5 categories | Cut to the minimum that tests the thesis. Expand only if a result demands it. |
| "4 hrs/run mean" | **GPU-starvation-aware budget** | Small-batch recurrent PPO is latency-bound; a B100 is underutilized at n_envs=8. Pack many configs per GPU; the per-run estimate is 1.5–2× v1's. |
| H9 ranked #2 | **substitutability = the spine** | It IS the contribution, not a side-audit. |

---

## 2. Cell zoo (6)

| Cell | Family | Role in the substitutability story |
|---|---|---|
| **GRU** | gated RNN, vector state | The "inferior" cell. Field default. If GRU+exploration ≈ GTrXL-alone, that's the headline. |
| **LSTM** | gated RNN, vector state | Second weak baseline; legibility / cross-paper comparability. |
| **Mamba-2** | selective SSM, vector state | "Superior" cell #1 (RLBenchNet Memory-S11 winner). The bar to beat with a weaker cell + exploration. |
| **LRU** | linear RU, vector state | Cheap SSM (POPGym Arcade favorite). Tests whether the SSM edge survives without selectivity. |
| **GTrXL** | gated transformer, attention state | "Superior" cell #2; the strongest memory-only baseline. The substitutability claim is most striking measured against this. |
| **SHM** | Hadamard matrix memory | Credit-assignment-targeted design; the only cell built for the H-E length bound. |

**Build/validate status:** GRU, LSTM, Mamba-2, SHM, LRU already in registry (verify LRU).
**GTrXL must be built + validated** against RLBenchNet's Memory-S11 number — treat as a
named sub-task with its own pre-flight.

Memoryless controls (required, not optional): **PPO-1** (MLP, current obs) and **PPO-4**
(MLP, 4-frame stack). These make "memory matters" measurable rather than asserted, and
they're the backbone of H-B.

---

## 3. Exploration heads (4)

| Head | Decay | Role |
|---|---|---|
| **NoBonus** | — | Pure-memory baseline (what the cell does alone) |
| **RND** | lifelong | Cheapest bonus; field default |
| **E3B** (φ_obs) | episodic | Per-episode novelty; the thesis's winner on memory tasks |
| **NovelD** | lifelong + first-visit | Combined; the "both signals" point |

No ICM (dominated by RND on action-controllable noisy-TV, Mavor-Parker 2022).

---

## 4. Environments (~9, by category)

| Cat | Env | Memory? | Explore? | Why |
|---|---|---|---|---|
| A pure memory | Passive T-Maze (50, 250) | yes | no | Cleanest memory-length probe (Ni et al.) |
| A | MiniGrid-Memory-S11 | yes | no | RLBenchNet's hard memory env; the substitutability target |
| A | POPGym RepeatPrevious-Medium | yes | no | Symbolic, fast, canonical |
| B memory+credit | Active T-Maze (50, 250) | yes | no | Pure credit-assignment probe |
| C memory+explore | MiniGrid-DoorKey-8x8 | yes | yes | The interaction's primary battleground |
| C | MiniGrid-Memory-S11 (no-hint-room variant) | yes | yes | Where "inferior cell + E3B" should shine |
| D hard explore | MiniGrid-MultiRoom-N4-S5 | light | yes | Exploration-dominated; bonus should clearly help |
| F control | CartPole-v1 (full obs) | no | no | Sanity: memory shouldn't hurt |

T-Maze lengths count as separate configs → ~11 env configs. Pixel envs (Memory Gym,
POPGym Arcade) deferred to a confirmation pass — see §8.

---

## 5. Hypotheses

### Core (the contribution)

**H-A — Exploration's sign flips with task category.**
Pure-memory (Cat A): bonus *hurts* (injects advantage noise that competes with credit to
the memorable obs). Hard-explore (Cat D): helps. Memory+explore (Cat C): U-shaped in β.
*Falsify:* bonus is benign-or-helpful on all Cat-A envs across all cells/heads.

**H-B — Memory strength and exploration are substitutes. [HEADLINE]**
On Cat-A and Cat-C envs: (weak cell + best exploration) ≥ (strong cell + NoBonus), and
(PPO-4 + RND) comes within 10% of (best recurrent + NoBonus) on ≥1 Cat-A env.
*Falsify:* strong-cell-alone strictly dominates weak-cell+exploration everywhere; PPO-4+RND
>30% behind on every env. (= "memory is essential and irreplaceable.")

**H-C — The "best cell" ranking is not exploration-invariant.**
Cell ranking under NoBonus differs from ranking under E3B. Specifically, GRU/LSTM climb
relative to Mamba-2/GTrXL once exploration is added.
*Falsify:* identical rankings with and without exploration → cell choice and exploration
choice are independent and memory quality is destiny.

### Supporting (rigor + mechanism)

**H-D — Reset hygiene rivals cell choice. [cheap, protects everything]**
Gap between (correct episode-reset) and (leaky state) exceeds the gap between two
well-implemented cells. RLBenchNet blamed their Mamba-v1 Memory-S11 failure on cross-episode
leakage — test whether published cell rankings are partly reset-correctness rankings.
*Falsify:* reset mode barely moves performance.

**H-E — Memory-length ⟂ credit-length, across the whole zoo (extends Ni et al.).**
Every cell scales further on Passive T-Maze (memory) than Active T-Maze (credit), and the
credit-length cliff is ≈ the same across cells — because PPO's GAE horizon bounds credit
propagation regardless of cell capacity.
*Falsify:* some cell's credit-length matches its memory-length (that cell is interesting).

### Free side-analyses (fall out of the H-A grid + logs)

- **S1 (was H3):** two-way ANOVA (cell × head) per env — are interaction terms small?
- **S2 (was H10):** gate-activation KL between bonus-on/off on Cat-A — mechanism for H-A.
- **S3 (was H12):** return-vs-wall-clock alongside return-vs-steps — SSM advantage shrinks/inverts.

### Dropped from v1
H6 (10× compute), H11 (collapses into H-E given Ni et al.'s prior), H8 (kept only as a free
correlation if pixel envs get run). H2 absorbed into the sharper H-C.

---

## 6. Run grid + compute

**Primary (H-A/B/C):** 6 cells × 4 heads × 11 env-configs × 3 seeds = **792 runs.**
**Controls (H-B):** 2 memoryless × 2 heads (NoBonus, RND) × 11 × 3 = **132 runs.**
**H-D:** 2 cells × 2 reset-modes × 3 envs × 5 seeds = **60 runs.**
**H-E:** 6 cells × (4 passive + 2 active lengths) × 3 seeds = **108 runs.**

3-seed first pass total: **~1,090 runs.** Backfill the statistically-tight cells/envs to
5 seeds (expect ~25%) → ~1,350 runs.

**Compute reality:** symbolic recurrent ~2–3 h/run; GTrXL + pixel up to ~10 h/run;
memoryless ~1 h. **Critical: pack 3–6 configs per GPU** — a B100 at n_envs=8 single-batch
is <20% utilized, so co-locating runs is free throughput. Honest estimate: **~70–90 GPU-days**
for the full thing; ~22–30 wall-clock days across 3 GPUs.

---

## 7. Pre-flight checks (before the 800-run grid)

Keep all of v1's — they're the strongest part. Non-negotiable ones:

1. **Baseline reproduction:** GRU+PPO on DoorKey-8x8 → >0.9 in <2M (matches RLBenchNet). If off, stop.
2. **GTrXL validation:** must reproduce RLBenchNet's Memory-S11 solve. This is the cell whose
   memory-only strength the substitutability claim is measured against — if our GTrXL is weak,
   H-B is unfalsifiable.
3. **RND wiring:** RND+PPO-1 must beat NoBonus+PPO-1 on MultiRoom (cleanest "exploration
   regularizes a memoryless policy" test — and directly H-B-relevant).
4. **Reset hygiene:** every cell >0.95 on Passive T-Maze-50; a failure there is a reset bug.
5. **TBPTT-length control:** GRU on Passive T-Maze-250 across TBPTT ∈ {16,32,64,128,full}.
   If credit-success rises monotonically with TBPTT, set TBPTT=full for all H-E runs.
   (Fixed-32 would make H-E measure a TBPTT artifact, not the cell.)
6. **Mamba-2 / SSM LR separation (RESeL):** separate LR for SSM core vs heads. Without it a
   "Mamba is bad" result is a hyperparameter bug.
7. **Wall-clock benchmark on heaviest combo** (GTrXL + E3B + pixel) → real FPS / GPU mem /
   util → recompute grid time before committing.

---

## 8. What to build

1. **GTrXL cell** (not in registry) — build + validate. *Largest single build risk.*
2. **LRU cell** — verify it's correct in registry.
3. **CNN encoder** (`NatureCNN`, vendored) for pixel envs + fix the MiniGrid wrapper that
   currently flattens categorical channels. *We confirmed this is missing.*
4. **Passive/Active T-Maze** port (Ni et al., github.com/twni2016/Memory-RL) → ~1 day.
5. **Reset-leak harness** (state not reset at episode boundary) for H-D → ~0.5 day.
6. **TBPTT chunk-boundary audit** of `MemPPO.update()` — confirm gradient propagates across
   chunks; fix the `state.detach()`-at-chunk-start bug if present → 0.5–1 day.
7. **SSM LR-separation flag** in `MemPPO` → 0.5 day.
8. **Analysis notebooks** (one per hypothesis) + wall-clock plotting → ~2 days.

Realistic build budget given this codebase's track record: **~2 weeks**, not the v1 "few days."

---

## 9. Decision points / off-ramps

- **After pre-flight:** GRU baseline doesn't match RLBenchNet → stop and debug. GTrXL doesn't
  solve Memory-S11 → H-B is on hold until it does.
- **After H-D:** reset hygiene dominates everything → pivot to an "audit of recurrent-RL
  implementations" paper (a real, citable contribution on its own).
- **After H-A/B first pass (the substitutability grid):** if H-B is strongly positive, that's
  the paper — write it before grinding H-E/H-D to completion.
- **Pixel confirmation pass:** only if a symbolic result needs validating on pixels. Memory Gym
  (finite mode) first; POPGym Arcade only if we migrate to JAX.

---

## 10. The honest framing for the writeup

Contribution, in one sentence: *we map the memory-cell × exploration-head interaction that
the field has only ever studied one-axis-at-a-time, and show the two axes are partially
substitutable — a result that reframes "which memory cell is best" as conditional on whether
you're also exploring.* H-A/B/C are the contribution; H-D/E are the rigor that makes the
interaction claims trustworthy; the controls (PPO-1/PPO-4) are what separate this from an
assertion. Replication-flavored pieces (H-E extends Ni et al.) are *supporting evidence for a
new thesis*, not the thesis itself.

*v2 — incorporates: substitutability as the spine, 6-cell zoo (LRU+GTrXL kept), 5 core
hypotheses, GPU-starvation-aware compute, GTrXL build flagged as the key risk.*
