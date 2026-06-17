# Master experiment plan v2 — "Exploration Bonuses Train Memory"

*The full redo. Drafted 2026-06-12. All prior wandb projects are DEPRECATED
(they predate the bug fixes, lack a clean baseline, and mix HARKed designs).
Everything below is re-run under new `memrl-memtrain-*` projects with the fixes
baked in. Companion to `theory_v2_memory_training.md`, `paper_plan_memory_training.md`,
`project_aaai_review_findings`.*

---

## 0. Invariants every run now satisfies (the fixes)

1. **E3B MultiDiscrete fix** — multi-head IDM (`e3b_module.py`); SS trains instead
   of silently running a frozen-random bonus. Verify `e3b_idm_acc` is logged & > chance.
2. **Done-step misattribution fix** — bonus zeroed on terminated envs (`ppo.py`);
   no terminal-reward kick, no "reward for dying."
3. **Snapshots** — `--snapshot-steps 500000,2000000,5000000,10000000 --snapshot-to-wandb`
   on every run (so the decodability probe can run anywhere).
4. **Clean baseline** — every comparison has a within-project `none` arm at the
   **same budget and code version**. No cross-project / cross-budget headline.
5. **HPs not bonus-selected** — re-confirm the winning HPs include a `none`-arm
   sweep, or report the e3b-selection and show robustness. (Reuse the locked
   winner λ_int=0.03, ck=64, lr=1e-4 only after this check.)
6. **n=5, significance, full disclosure** — Mann-Whitney / Welch per cell; report
   seed counts, crashed/excluded runs, and metric definitions (eval success vs
   train return) in every table.
7. **Two metrics, never mixed across arms** — compare on **success/eval** (max 1.0
   on all arms) for the headline; report shaped return separately. Penalty/aligned
   arms are return-matched only up to the measured bound (P1).

## 1. Vocabulary

- **Cells (6):** GRU, LSTM (weak); RetNet, GatedDeltaNet, Mamba2 (strong);
  Memoryless (floor). GTrXL → appendix only (param + cache confounds). Report params.
- **Arms (intrinsic):** `none` / `e3b_idm` (raw bonus) / `pbim_e3b_idm` (potential
  delivery). Optional: `e3b_rand`, `noveld` for ablations.
- **Density:** `sparse` (terminal only) / `penalty` (faithful fall fine −0.008) /
  `aligned-dense` (per-use positive, MortarMayhem only — exactly return-matched).

---

## 2. Experiment table

**Restructured (2026-06-13):** the env×density×bonus grid is ONE experiment, **E1**,
replicated across the env suite. MortarMayhem / Autoencode / S13 / Battleship /
TinyReproduce / SearingSpotlights are **E1 on another env** (a slice), not separate
experiments. The 20M headline (3× sparse none-vs-e3b) is the **MysteryPath-sparse
slice** of E1 — no standalone headline experiment. Only studies that vary a
*non-env* axis (λ, k, φ, architecture) are distinct. Source of truth:
`experiments/memory_training/registry.py` (`--list`). Cost is not a constraint (cluster).

### E1 — CORE GRID (env × density × bonus × cell × seed)

| env-slice | mem-type / α | cells | bonuses | densities | budget | runs | unique role |
|-----------|--------------|-------|---------|-----------|--------|------|-------------|
| MysteryPath | spatial trace, α>0 | 6 | none/e3b/pbim/noveld | sparse/penalty/aligned | 20M | 360 | **headline** + F1/F2/F3/entangle/P5 |
| MortarMayhem | sequence WM, α≈0 | 4* | none/e3b/pbim | sparse/aligned | 20M | 120 | **exact** return-match (4×0.25=1) → clean redundancy flip |
| Autoencode | reproduce, α≈0 | 6 | none/e3b | sparse/dense | 20M | 120 | exact δ-toggle + **known RM** (exact probe GT) |
| S13 | recall, α≈0 | 6 | none/e3b/noveld | sparse | 20M | 90 | retention contrast + noveld≥e3b reversal |
| Battleship | grid-probe, α>0 | 4* | none/e3b | sparse/dense | 10M | 80 | **2nd α>0** env → generalizes "bonus helps" |
| TinyReproduce | reproduce, α≈0 | 3 | none/e3b | sparse/dense | 2M | 60 | **enumerable RM** (colleague's example) → exact eff-RM-size |
| SearingSpotlights | dead-reckon, α<0 | 4* | none/e3b | sparse/anti | 10M | 80 | α<0 hazard + no-sanctuary freeze boundary |

**E1 total: 910 runs.** Headline = MysteryPath-sparse none-vs-e3b slice (no separate run).

### Method studies (vary a non-env axis → distinct experiments)

| id | study | env | grid | runs | serves | deps |
|----|-------|-----|------|------|--------|------|
| **E2** | decodability + eff-RM-size probe | — (E1 snapshots) | offline | 0 | F1, P5 | E1 snapshots |
| **E3** | GAE-λ sweep | MysteryPath sparse | GRU × λ{.8,.9,.95,.99,1} × {none,e3b} × 3 | 30 | P2 mechanism (can falsify our own theory) | ✓ |
| **E6** | tiny maze: sanctuary + DP | TinyMaze | 2c × {sanc,no-sanc} × {none,e3b} × 5 | 40 | P1 bound, P4 deconfound | ⛔ build env + DP |
| **E9** | TBPTT k-sweep | MysteryPath sparse | GRU × k{1..128} × {none,e3b} × 3 | 48 | P2 secondary (non-monotone, k=1 collapse) | ⛔ episode-aligned chunks |
| **E10** | φ-ablation | MysteryPath sparse | 3c × {idm,rand,obs} × 5 | 45 | which bonus trains memory | ✓ |
| **E11** | entangled IDM-aux | MysteryPath sparse | 3c × {none,e3b,shared} × 5 | 45 | P3 causal + recipe | ⛔ build shared arm |

\* MortarMayhem / Battleship / SS use 4 cells (GRU, RetNet, GatedDeltaNet, Memoryless;
weak/strong/floor preserved). **Grand total runnable: 985** (E6/E9/E11 blocked on code).

---

## 3. What each experiment proves (and the pre-registered decision rule)

**E1 + E2 (the paper).** The core grid + probes.
- *Decodability (F1):* on E1 snapshots, decode the minimal-RM state (knowledge grid)
  from the frozen recurrent state, matched-coverage bank.
  **Rule:** confirmed iff sparse-e3b > sparse-none (resolved bits, bootstrap CI)
  AND penalty-none ≈ chance (eff_rm_size ≈ 1) AND penalty-e3b > chance.
- *Freeze/rescue (F2):* eval success + `ep_num_fails`. **Rule:** freeze iff
  penalty-none success≈0 AND falls→0 (the diagnostic), distinct from sparse-none
  (falls > 0, success > 0); rescue iff penalty-e3b success ≈ sparse-e3b.
- *PBIM (F3):* **Rule:** if pbim-e3b ≈ raw-e3b → densification (P3a) dominant; if
  pbim-e3b < raw-e3b (CI-separated) → distribution/curriculum (P3b) load-bearing.
  Either is the result; state which before unblinding is *not* required, but the
  interpretation map is fixed.
- *Entanglement:* Memoryless+e3b ≤ Memoryless+none; capacity gradient = benefit
  vs cell param-count slope negative.
- *Effective RM size (P5):* eff_rm_size(penalty-none) → 1; e3b re-inflates.
- *PBIM stability control (optional, ~free):* add a `raw-φ-PBIM` arm (potential
  fit to raw φ-features instead of `V_int`) on GRU+RetNet sparse. **Rule:**
  raw-φ-PBIM shifts the success optimum vs `none` (beyond the P1 bound) while
  `V_int-PBIM` does not — *demonstrates* (not asserts) that PBIM's potential sits
  on a future-reward-sufficient statistic (theory_v2 P5 PBIM caveat). Tier-2,
  skip if compute-bound.

**E3 (P2 mechanism).** **Rule:** sparse-none deficit and e3b−none both shrink as
gae_lambda → 1. If they don't, P2 (GAE-horizon) is wrong and the credit story
needs revisiting. This is the test that can falsify our own corrected theory.

**E4 (redundancy, clean).** MortarMayhem is *exactly* return-matched
(4×0.25 = 1.0). **Rule:** e3b−none > 0 on sparse; e3b−none ≈ 0 on aligned-dense
(the redundancy flip), on **success**. Also the within-MM density main effect
(dense-none ≫ sparse-none) is the cleanest "density governs credit-to-memory."

**E5 (retention contrast).** S13 cue is observed → front-loaded revelation → no
behavioral memory-training work for the bonus. **Rule:** e3b ≈ none on S13 (and
the noveld≥e3b reversal is *expected*, reported as a feature, not hidden).
Decodability probe on S13 (cue decodability): predict none ≈ e3b.

**E6 (P1 + P4, cheap, high-theory-value).**
- DP on a fixed small maze → `E[falls | π*]` → validates P1's bound ε = p·E[falls].
- Sanctuary toggle: **Rule:** freeze (penalty-none → trivial RM) appears WITH a
  passive sanctuary, absent WITHOUT it — separating "sanctuary" from "penalty
  information-content" (confounded in MysteryPath-vs-SS). Known minimal RM here →
  validate decodability→RM-state and eff_rm_size end-to-end.

**E7 (honest headline).** 20M with a within-project none baseline → the real 3×
number on success, with per-cell seed counts.

**E8 (no-sanctuary boundary; stretch).** With the MultiDiscrete fix, SS e3b now
trains (verify `e3b_idm_acc`). **Rule:** anti-none does NOT freeze (no sanctuary);
report e3b sign honestly (the prior α<0 result was a bug — re-open the question).

**E9 (k-sweep secondary; stretch).** Needs episode-aligned chunking first.
**Rule:** e3b−none non-monotone in k, peaking at intermediate k, collapsing at
k=1 (both arms fail). Report `n_updates`/`grad_norm`/`clip_frac` per k; exempt
GTrXL.

**E10 (which bonus; tier 3).** IDM-φ vs random-φ vs obs-φ. **Rule:** if IDM-φ ≫
random-φ, controllable features matter; if comparable, any episodic novelty trains
memory. (Could be folded into E1 by adding `e3b_rand`.)

**E12 (Autoencode — known-RM retention + clean δ-toggle).** POPGym Autoencode
(reverse/LIFO; card 1 held the full 2N−1 steps = maximal Δ). `dense` = +1/N per
correct, 0+terminate on wrong (MortarMayhem rule); `sparse` = same paid as a
terminal lump sum (`DeferredReward`) → **exactly return-matched at γ=1, pure δ
manipulation**. **Rules:** (i) density main effect — dense-none learns earlier than
sparse-none (credit-to-memory); (ii) retention contrast — e3b ≈ none (front-loaded
revelation, α≈0), reinforcing S13/MM; (iii) **probe ground truth** — the
remaining-to-reproduce suit sequence is the *exact* minimal-RM state
(`info["rm_state"]`), so `decode_memory.py --task autoencode` validates the
decodability/eff-RM-size methodology against truth, not a proxy. Caveats: Autoencode
is HARD at 52 cards (may be near-flat at 10M like MM-sparse — the probe GT + density
main-effect are the load-bearing deliverables, not high success); report a
shuffled-label control to set the resolved-bits chance floor.

**E13 (Battleship — the 2nd α>0 env; closes the "single positive env" gap).**
Recoverable probing (a miss is −small, no death) → novelty (fire a new cell) =
coverage = progress, **cleanly α>0** (unlike MineSweeper, where clicking a mine =
death makes novelty dangerous). Memory-essential (remember fired cells);
MultiDiscrete([8,8]) probing exercises the E3B fix. Coord-augmented obs
(`expose_action_coords`) makes φ position-aware so episodic novelty means
"new cell" (without it, hit/miss obs is too poor — the obs-richness confound).
**Rules:** (i) e3b > none on sparse (the positive effect *generalizes* off
MysteryPath — different task family: search vs path-follow); (ii) e3b ≈ none on
dense (redundancy); (iii) Memoryless ≪ memory cells. Sparse = dense deferred.

**E14 (TinyReproduce — enumerable-RM probe/P5 validation).** k=6,v=2 → 64
enumerable minimal-RM states (the colleague's k-token reproduce example).
RETENTION (α≈0); its value is **exact RM ground truth**, not a bonus result.
**Rules:** (i) `decode_memory --task autoencode --n-suits 2` recovers the
remaining-token RM state with high accuracy on a learned memory (validates the
probe methodology against truth); (ii) eff_rm_size tracks toward the true 2^bits;
(iii) under a (hypothetical) anti arm the memory would collapse — here it's the
clean P5 demonstrator. Cheap (2M, episode ~11). Pairs with E6 tiny-maze.

**E11 (entangle, confirmation + recipe; stretch).** Direct IDM-aux-loss into the
policy encoder. **Rule:** if entangled-aux raises decodability + success without
the policy-bias cost, it confirms credit-to-memory is the operative quantity and
offers a recipe. Clearly the *contrast*, never the headline.

---

## 4. Tiers & compute (be ready to cut from the bottom)

Unit = one 10M pixel-recurrent run (20M = 2 units).

| Tier | experiments | run-units | what you get |
|------|-------------|-----------|--------------|
| **1 (must)** | E1, E2, E3 | ~210 | the whole thesis: F1 decodability, F2 freeze/rescue, F3 PBIM, entanglement, P5, P2 mechanism |
| **2 (should)** | E4, E5, E6 | ~230 | generality across 3 memory types + P1/P4 causal |
| **3 (nice)** | E7, E10 | ~150 | honest 20M headline + φ ablation |
| **4 (stretch)** | E8, E9, E11 | ~150 | SS boundary, k-sweep, entangle recipe |

**Minimum publishable paper = Tier 1 + one Tier-2 generality env (E4 preferred,
because it is the only exactly-return-matched density toggle).** If compute-bound,
ship Tier 1 + E4 + E6 (E6 is cheap and carries P1/P4). Everything else strengthens.

**Critical-path note:** E1 is the bottleneck and unlocks E2 and E7. Launch E1
first, in seed-grouped waves; start E3/E4/E5 in parallel as GPUs free.

---

## 5. Execution order (dependencies)

```
Phase A (launch immediately, parallel):
  E1 core grid (60 jobs)  ─┬─► E2 probes (CPU, as snapshots land)
  E3 λ-sweep (15 jobs)     │
  E4 MortarMayhem (40)     │
  E5 S13 (30)              │
  E6 tiny-maze DP (CPU) + sanctuary runs (small)
Phase B (after E1 lands):
  E7 20M (extend learners + none baseline)
  E10 φ-ablation (or fold e3b_rand into E1 from the start)
Phase C (stretch, needs code):
  E8 SS (MultiDiscrete fix done; just launch)     ← actually launch in A if GPUs
  E9 k-sweep (after episode-aligned-chunk code)
  E11 entangle (after aux-loss arm code)
```

Code prerequisites still owed:
- **E9:** episode-aligned chunking in `buffer.py` (chunks start at episode
  boundaries, not the rollout grid) + an optional per-epoch state-refresh control.
- **E11:** an `e3b_idm_shared` arm whose IDM φ *is* the policy encoder (aux CE loss
  into the shared backbone) — a new flag, ~1 day.
- **E6:** a tiny fixed-maze env + belief-MDP DP solver (`memrl/envs/tiny_maze.py`,
  `memrl/probes/optimal_prober.py`).

---

## 6. Analysis & figures (one source of truth)

| Figure | From | Metric |
|--------|------|--------|
| F1 decodability vs step (4 lines) | E2 | resolved bits / balanced acc, matched bank |
| F2 freeze & rescue | E1 | eval success + ep_num_fails (the falls→0 diagnostic) |
| F3 PBIM adjudication | E1 | success: none vs raw-e3b vs pbim-e3b |
| F4 mechanism | E3 + E4 | e3b−none vs gae_lambda; e3b−none sparse vs aligned (MM) |
| F5 entanglement / capacity | E1 | bonus benefit vs cell param-count |
| F6 eff-RM-size (P5) | E2 | eff_rm_size per arm vs step (penalty-none → 1) |
| F7 (appendix) sanctuary toggle | E6 | freeze present/absent ± sanctuary |
| F8 (appendix) retention contrast | E5 | e3b≈none on S13; noveld reversal |

All probe outputs are JSON lines from `memrl/probes/decode_memory.py` (decodability
+ eff_rm_size records); aggregate with a small notebook.

---

## 7. Pre-registration block (file before unblinding the 5-seed comparisons)

**Superseded and locked by `docs/preregistration.md` (filed 2026-06-16).** That file
is the authoritative source for the headline claim, primary metric, statistical tests,
equivalence margins, the locked idle-fraction / lag-Δ-retention / PBIM-ρ rules, and the
inclusion/survivorship rules. The summary below is retained for orientation.

- Decodability decision rule (E1/E2): §3.
- α is **not** a claimed taxonomy; if reported, measured under a fixed reference
  occupancy with a locked threshold (see review §6) — otherwise descriptive only.
- Metrics: headline on eval success; shaped return reported separately; arms
  return-matched only up to the measured ε (P1).
- Exclusions: a run counts only if it reached ≥95% of budget; crashed-name
  duplicates de-duplicated by run id; every exclusion listed in the appendix.
- The λ-sweep (E3) and the k=1 collapse (E9) are declared as tests that can
  falsify our own P2 — report whichever way they fall.
