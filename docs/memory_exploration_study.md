# Memory × Exploration in Deep RL — Study Plan

**Status:** Learning project → paper. Thesis complete; this systematizes its central
observation. *Exploratory phase first* (mess around, build intuition), rigorous grid later.
**Supersedes:** `memory_exploration_study_v2.md`.

---

## 0. The thesis (the novel spine)

> **Memory-architecture strength and exploration bonuses are partially substitutable
> for solving sparse-reward memory POMDPs.** A weaker memory cell paired with the right
> exploration head can match or beat a stronger cell with no exploration.

Concrete hook (already in hand): **LMU + E3B solved MiniGrid Memory-S13 in ~3M steps**
(thesis), and in this codebase **GatedDeltaNet + PPO solved MiniGrid Memory-S13 with no
exploration at all** (seed 0; seeds 1-2 verifying). Meanwhile RLBenchNet (2025) shows that
*without* exploration only Transformer-XL and Mamba-2 cross the threshold on Memory-S11.
The trade-off between "better cell" and "add exploration" has never been mapped.

---

## 1. Novelty — verified, not assumed

The user asked to confirm this rather than assert it. Literature check (Nov 2025) from both
sides of the question:

**Memory-cell papers fix the algorithm (plain PPO/PQN, no bonus) and vary the cell:**
- POPGym (Morad ICLR 2023), POPGym Arcade (Wang 2025), RLBenchNet (Smirnov & Gu 2025),
  Influence-aware Memory (Springer 2022), RATE / Recurrent Action Transformer (2023),
  "Memory-Improvable Domains" (Tao 2025). None add an intrinsic-reward head.

**Exploration-bonus papers fix the architecture (a single LSTM / small encoder) and vary the bonus:**
- RND (Burda 2019), ICM (Pathak 2017), NovelD (Zhang 2021), E3B (Henaff 2022),
  "Global vs Episodic Bonuses in Contextual MDPs" (Henaff 2023). All use one fixed recurrent
  policy. NGU/Agent57 (Badia 2020) is the closest — but it's *one* cell (R2D2 LSTM) +
  episodic+lifelong exploration, not a zoo.

**Conclusion:** the systematic **{memory cell} × {exploration head}** interaction matrix on
POMDP memory tasks is, to our knowledge, **unstudied**. The closest adjacent work (Henaff 2023)
varies the exploration axis while holding the architecture fixed; nobody varies both. That
gap — plus the substitutability finding — is the contribution.

(Caveat for the writeup: this is a "to our knowledge" claim. Before submission, do a focused
related-work pass on RL-exploration surveys and any POMDP-exploration papers post-2024.)

---

## 2. Cell zoo (8) — "what's the right memory substrate, and does it matter once you explore?"

Spans three state-representation families: **vector** (GRU, LSTM, Mamba-2, LRU),
**matrix** (SHM, FFM, GatedDeltaNet), **attention** (GTrXL).

| Cell | Family | Built? | Role |
|---|---|---|---|
| **GRU** | gated RNN, vector | ✅ | The "inferior" cell; field default; the one we want to rescue with exploration |
| **LSTM** | gated RNN, vector | ✅ | Second weak baseline; legibility / cross-paper comparability |
| **Mamba-2** | selective SSM, vector | ✅ | "Superior" cell #1 (RLBenchNet Memory-S11 winner); bar to beat with weak+explore |
| **LRU** | linear RU, vector | ✅ | Cheap SSM; tests whether the SSM edge survives without selectivity |
| **GTrXL** | gated transformer, attention | ✅ | "Superior" cell #2; strongest memory-only baseline; substitutability measured vs this |
| **SHM** | Hadamard matrix memory | ✅ | Matrix-state, credit-assignment-targeted design |
| **FFM** | multi-decay aggregator, matrix | **⬜ build** | **POPGym recurrent SOTA (Morad NeurIPS 2023).** The strongest published general-purpose recurrent cell — the most important baseline to include. |
| **GatedDeltaNet** | delta-rule matrix | ✅ | Delta-rule cell that solved MemoryS13 with no exploration; the substitutability hook's protagonist |

**Memoryless controls** (required): **PPO-1** (MLP, current obs), **PPO-4** (MLP, 4-frame stack).
These make "memory matters" measurable instead of asserted.

**To build:** FFM (Morad et al. 2023, github.com/proroklab/ffm). Multi-bank exponential-decay
aggregation; `S_t` has per-dim decay rate + contextual period. ~half day to port faithfully.

**Param parity:** match *width* (hidden_size), not param count — the cell-comparison
standard. Tune Mamba `d_state` to land at parity (it's a non-distorting knob). Report counts
transparently: LSTM ≈ 1.3×, GTrXL ≈ 3.6× GRU are intrinsic to those architectures. GTrXL being
bigger *strengthens* the substitutability story (beat a 3.6× model with a smaller one + a bonus).

---

## 3. Exploration heads (5)

| Head | Decay | Built? | Tests |
|---|---|---|---|
| **NoBonus** | — | ✅ | Pure-memory baseline |
| **RND** | lifelong | ✅ | Cheapest bonus; field default; predicted to fail on per-episode-hint tasks |
| **E3B** (φ_rand / φ_obs) | episodic | ✅ | Per-episode novelty; thesis winner on memory tasks; φ-source is the Agency-Principle knob |
| **NovelD** | lifelong + first-visit | ✅ | Combined-signal point |
| **ICM** | lifelong | ✅ | Classic baseline (kept for completeness; dominated by RND on noisy-TV) |

---

## 4. Environments (by category)

| Cat | Env(s) | Memory? | Explore? |
|---|---|---|---|
| A pure memory | Passive T-Maze (50/250/750/1500), MiniGrid-Memory-S11/S13, POPGym RepeatPrevious-Medium | ✅ | — |
| B memory+credit | Active T-Maze (50/250) | ✅ | — |
| C memory+explore | MiniGrid-DoorKey-8×8, RedBlueDoors-6×6, Memory Gym Mortar Mayhem | ✅ | ✅ |
| D hard explore | MiniGrid-MultiRoom-N4-S5, ObstructedMaze-1Q | light | ✅ |
| E noisy POMDP | Memory Gym Searing Spotlights | ✅ | ✅ |
| F full-obs control | CartPole-v1 | — | — |

To build: CNN encoder (NatureCNN; current `FlatEncoder` flattens pixels) + Passive/Active
T-Maze port (Ni et al., github.com/twni2016/Memory-RL) + Memory Gym integration.

---

## 5. Phase 0 — Playground (do this first; forget compute)

The point right now is to *mess around and build intuition*, not run a 1000-job grid.
Minimal, fast, single-seed, short runs:

1. **Pick 1 env that already works:** MiniGrid-Memory-S13 (we have it; GatedDeltaNet solves it).
2. **Run a small cross-slice:** {GRU, GatedDeltaNet, GTrXL} × {NoBonus, E3B(φ_rand)} on it,
   single seed, ~3-5M steps. 6 runs.
3. **Look at the learning curves together.** The first thing to see: does GRU+E3B catch
   GTrXL-NoBonus? That's the substitutability picture in miniature.
4. **Then add one harder env** (DoorKey-8×8 or RepeatPrevious-Medium) and repeat.
5. **Build FFM**, drop it into the same slice — it's the SOTA recurrent baseline, so it
   anchors "how good is good."

No pre-registration, no falsification criteria yet — just play, watch curves, develop
hunches about which interactions are real. The rigorous grid (Phase 1+) is designed *after*
Phase 0 tells us which cells/heads/envs are worth the full sweep.

---

## 6. Phase 1+ — The rigorous study (after Phase 0)

Hypotheses (committed-to before the big grid, with falsification criteria):

- **H-A** Exploration's sign flips by task category: hurts pure-memory (Cat A), helps hard-explore
  (Cat D), U-shaped on memory+explore (Cat C). *Falsify:* benign/helpful on all Cat-A.
- **H-B [HEADLINE]** Memory↔exploration substitutability: (weak cell + best exploration) ≥
  (strong cell + NoBonus); PPO-4+RND within 10% of best-recurrent+NoBonus on ≥1 Cat-A env.
  *Falsify:* strong-cell-alone dominates everywhere.
- **H-C** "Best cell" ranking changes once exploration is added. *Falsify:* identical rankings
  with/without bonus.
- **H-D** Reset hygiene rivals cell choice (cheap audit; protects every other claim).
- **H-E** Memory-length ⟂ credit-length across the whole zoo (extends Ni et al. 2023 beyond
  transformers). Supporting evidence, not the contribution.

Free side-analyses from the same logs: cell×head ANOVA, gate-activation KL (mechanism for H-A),
return-vs-wall-clock.

(Compute budgeting deferred — see v2 for the GPU-starvation-aware grid math when we get there.)

---

## 7. Build status

**Done this session:** GatedDeltaNet, MultiLayerGatedDeltaNet, LRU, GTrXL cells;
IntrinsicRewardModule abstraction + NoBonus/RND/E3B/NovelD/ICM; cell×intrinsic runner;
MemPPO intrinsic integration; train.py wiring.

**To build (in priority order):**
1. **FFM cell** (Morad 2023) — the SOTA recurrent baseline; highest-value addition.
2. **CNN encoder** (NatureCNN) + fix MiniGrid wrapper that flattens categorical channels.
3. **Passive/Active T-Maze** env port (Ni et al.).
4. **GTrXL validation** — reproduce RLBenchNet Memory-S11 solve before H-B is falsifiable.
5. Memory Gym integration; reset-leak harness; TBPTT-boundary audit.

---

## 8. References

**Cells:** GRU (Cho 2014), LSTM (Hochreiter 1997), Mamba-2 (Dao & Gu ICML 2024), LRU (Orvieto
ICML 2023, arXiv 2303.06349), GTrXL (Parisotto ICML 2020, arXiv 1910.06764), SHM (Le ICLR 2025,
arXiv 2410.10132), **FFM (Morad NeurIPS 2023, arXiv 2310.04128)**, GatedDeltaNet (Yang ICLR 2025).

**Exploration:** RND (Burda ICLR 2019), ICM (Pathak ICML 2017), NovelD (Zhang NeurIPS 2021),
E3B (Henaff NeurIPS 2022, arXiv 2210.05805), Global-vs-Episodic (Henaff ICML 2023, arXiv 2306.03236),
NGU (Badia ICLR 2020).

**Benchmarks:** POPGym (Morad ICLR 2023), Memory Gym (Pleines JMLR 2025), RLBenchNet
(Smirnov & Gu 2025, arXiv 2505.15040), Memory-RL / T-Maze (Ni et al. NeurIPS 2023, arXiv 2307.03864).

**Framing:** Ni, Eysenbach, Salakhutdinov ICML 2022 (recurrent baselines, arXiv 2110.05038);
Ni, Ma, Eysenbach, Bacon NeurIPS 2023 Oral (memory vs credit assignment — key for H-E).

---

*v3 — adds verified-novelty section, FFM + GatedDeltaNet to the 8-cell zoo, and a Phase-0
playground phase (mess-around-first). Compute budgeting deferred to v2's grid math.*
