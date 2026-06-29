# Paper skeleton — "Memory Needs Exploration" (memory × intrinsic-exploration interaction)

> **NEW DOC OF RECORD (2026-06-25). Supersedes the reward-machine framing
> (`paper_full.md`, `paper_sealing_penalty_skeleton.md`, `paper_skeleton.md`).** Reward
> machines / Myhill–Nerode / Mealy–Moore are CUT. The freeze + PBIM survive as *empirical*
> results with no RM vocabulary. Spine = the (verified-unstudied) {memory cell} ×
> {intrinsic exploration} interaction. Literature grounding + the decision: session 45ff0f88.

Evidence tags: **[n5]** MysteryPath n=4–5 (clean) · **[n2]** S13 directional · **[reg]**
register env (Tiny/Autoencode) · **[free]** existing-log analysis · **[NEW]** needs a run.

---

## 1. The one idea (AMPLIFICATION-LED, 2026-06-25)

In partially observable RL, intrinsic exploration and memory architecture are not
independent design choices, and — counter to intuition — **exploration does not equalize
architectural memory differences, it AMPLIFIES them.** You would expect a bonus to be a
great equalizer (give the weak cell the data it needs to catch up); instead the same
bonus lifts a strong cell far more than a weak one, *widening* the architecture gap. This
super-additive cell×bonus interaction is the object the field missed by studying the two
axes apart (memory-cell papers fix the algorithm and vary the cell; exploration papers fix
the architecture and vary the bonus). We map the interaction systematically and find it is
**super-additive**, **conjunctive** (a memoryless agent with the best bonus moves
maximally yet cannot be rescued), and capable of **catastrophic breakdown** (a faithful
penalty collapses it to zero; only a non-potential bonus reopens it).

## 2. Title / one-sentence thesis

**Title:** *Exploration Amplifies, Not Equalizes: How Intrinsic Bonuses Interact with
Recurrent Memory in Partially Observable RL* (alt: *Memory Needs Exploration*).

**Thesis:** Intrinsic exploration interacts *super-additively* with memory architecture in
POMDPs — the same bonus widens, rather than closes, the gap between strong and weak
recurrent cells (RetNet/GDN +.46–.53 vs GRU/LSTM +.02–.07 off a shared baseline) — while
remaining strictly *conjunctive* (it cannot substitute for memory: a memoryless agent with
the best bonus still fails), and, in the extreme, the difference between a memory agent
that learns and one that collapses to inaction.

> ★ **This framing is GATED on one experiment (§8 P0-HP).** The amplification is a
> ranking-style claim; its fatal alternative reading is "exploration amplifies *tuning*
> differences" (GDN 0.40→0.82 from HP alone, a swing larger than every cross-cell gap). It
> headlines iff the amplification SURVIVES per-cell-best-HP. If it washes out, the
> HP-immune freeze (§3 C4) is the only clean novelty and must lead instead.

## 3. Contributions (claim exactly these, in priority order)

- **C1 — The first systematic memory×exploration interaction study.** A {6-cell zoo:
  GRU, LSTM, RetNet, GatedDeltaNet, Mamba2, Memoryless} × {none, E3B, NovelD, RND, PBIM} ×
  {MysteryPath, S13, TinyReproduce, Autoencode} matrix under online recurrent PPO. Fills
  the gap POPGym/RLBenchNet (fix the algorithm) and Henaff/E3B/NovelD (fix the
  architecture) leave open. **[n5 + n2 + reg]**
- **C2 — Conjunctive necessity / exploration is a THIRD axis.** Neither half suffices
  alone: `Memoryless + best bonus` floors (S13: flat **0.45** across none/E3B/NovelD;
  MysteryPath: E3B gives the *highest* falls 22.3 yet **0.00** success), and `cell + no
  bonus` fails or freezes. ⚠️ Framed as a **third axis orthogonal to Ni 2023's
  memory/credit decoupling — necessary-but-not-sufficient, cannot rescue a memoryless
  agent.** Do NOT claim "explains Ni / the missing half": our own no-bonus column
  *reproduces* Ni (GRU-none .09 ≈ Memoryless .06), and exploration **amplifies** rather
  than equalizes the gap (opposite of "missing half"). **[n5 + n2]**
- **C3 — Geometry-matching (DEMOTED to descriptive — "extends RLeXplore", not the
  keystone).** *Which* bonus helps appears to track task memory-geometry (E3B carries
  MysteryPath; NovelD carries S13). ⚠️ **Currently under-powered** (S13 reversal n=2 with
  seeds disagreeing s0 .98/s1 .33; on MysteryPath NovelD ≤ E3B by only −0.00..−0.11 = no
  dissociation; the RND "mismatched-global" arm is unrun on the high-signal cells and where
  it exists on S13 it *helps* — GRU-RND .856, pointing the wrong way). Demote to "extends
  and is consistent with RLeXplore's episodic-vs-global observation" **unless P0-7/P0-8
  power it** (§8). **[n2 — weakest claim]**
- **C6 — Super-additivity: exploration AMPLIFIES architectural memory differences
  (PROMOTED).** The interaction that earns the title: sparse `e3b−none` is RetNet **+.53**,
  GDN **+.46** vs GRU **+.07**, LSTM **+.02** off a shared ~.16–.20 baseline. Exploration
  does not equalize cells — it *widens* the cell gap (super-additive cell×bonus
  interaction). A named result with an interaction-term test, not a footnote. **[n5, free]**

### ★ HEADLINE ORDER (amplification-led, user decision 2026-06-25): C6 → C2 → C1 → C4 → C3.

The **interaction is the frame**; the headline *content* is C6 (super-additive
amplification — the counterintuitive interaction) + C2 (conjunctive necessity). C4 (freeze)
is the **dramatic "collapse corner" of the interaction**, a prominent supporting section —
NOT the thesis. C3 trails (demote-or-power per §8).

- **C4 — The freeze + PBIM (the dramatic collapse-corner result, SUPPORTING not headline).**
  A faithful, *return-matched* −0.008 penalty collapses all **6/6** cells to inaction
  (success 0.00 **and** falls 0.0, n=5) while the return-matched sparse twin stays alive
  (.16–.20, falls 9–39); only a **non-potential** bonus revives them (E3B .24–.62, NovelD
  .17–.71), and a **potential delivery of the identical signal (PBIM) cannot** (ρ=0). The
  most extreme cell of the interaction table: where the interaction goes to zero. DP:
  freezing strictly sub-optimal at trained γ=0.99 (V\*=0.42; flips by γ≈0.95 — disclose).
  **No reward-machine vocabulary.** **[n5]**
- **C5 — (boundary) The α≈0 register null.** When retention is *forced* and there is no
  coverage to gain (TinyReproduce/Autoencode), the bonus is inert (E3B ≈ none every cell).
  The interaction needs a *discoverable* memory demand. **[reg]**

> **Honest risk of the amplification-led order:** C6 is the user's preferred headline and
> the genuine interaction, but it is HP-confounded until §8 P0-HP shows it survives
> per-cell-best-HP. The freeze (C4) is HP-immune and 5/6 reviewers call it the only
> non-derivative object — so if amplification washes out under best-HP, the order flips back
> to C4-led. **The per-cell-best-HP run settles which framing leads; run it first.**

## 4. Figures

- **F1 (HEADLINE) — the interaction matrix.** Heatmap: success across {cell} × {bonus},
  one panel per env. The eye sees: bonus helps cells differently; Memoryless row flat;
  penalty arm collapses without bonus. **[n5 + n2]**
- **F2 — conjunctive necessity.** The 2×2 of {has-memory?, has-exploration?}: Memoryless+
  bonus (moves, fails), cell+none (stalls/freezes), cell+bonus (solves); + the Memoryless-
  flat-0.45 bars on S13. **[n5 + n2]**
- **F3 — geometry-matching.** NovelD vs E3B per env: E3B wins MysteryPath, NovelD wins S13;
  RND (global) interferes. Annotate with the RLeXplore connection. **[n2]**
- **F4 — the freeze + rescue, return-matched.** Per-cell bars: sparse-none (alive) vs
  penalty-none (0/0, 6/6) vs penalty-E3B/NovelD (rescued) vs penalty-PBIM (still 0/0). The
  return-matched twin is the load-bearing control. **[n5]**
- **F5 — the behavioral signature.** idle-fraction by arm×bonus (non-potential → 0;
  none/PBIM → high), already logged in `action_frac_0`. **[n5, free]**
- **Appendix:** α≈0 register null; per-cell-best-HP cell-ranking audit (rankings are
  HP-conditioned — GDN 0.40→0.82 from HP alone); the DP optimality bound at trained γ;
  cautionary decode-probe note (random-init ≈ trained → copy-capacity).

## 5. Related work / positioning (verified 2026-06-25)

- **vs memory-cell benchmarks (POPGym, Morad 2023; RLBenchNet, Smirnov–Gu 2025; IAM):**
  they fix the algorithm (plain PPO/PQN, no bonus) and vary the cell. We add the
  exploration axis — the interaction is the contribution.
- **vs exploration-bonus papers (E3B Henaff 2022; NovelD Zhang 2021; Global-vs-Episodic
  Henaff 2023; RND, ICM):** they fix the architecture (single LSTM/encoder). We vary it.
- **vs Ni 2023 (NeurIPS oral, memory ⊥ credit assignment):** they show LSTM often fails
  to beat memoryless for a *fixed* reward. We add exploration as a **third axis orthogonal
  to their memory/credit decoupling** (necessary-but-not-sufficient). ⚠️ Do NOT say "the
  missing variable / explains Ni" — our own no-bonus column reproduces their result and
  exploration *amplifies* the gap (opposite of "equalizing missing half").
- **vs RLeXplore (Yuan 2024) — THE closest:** they observed (2 architectures, one finding)
  that intrinsic rewards can perform *worse* with LSTM policies ("episodic vs global
  cross-purposes") but did not systematize it. ⚠️ "Episodic-vs-global" already *is* a
  geometry statement → C3 is "**extends and is consistent with** RLeXplore," NOT "resolves."
  The real deltas vs RLeXplore are **C1 breadth** (a cell zoo, not 2 arch), **C6
  amplification** (gap-widening, which they do not report), and **C4 freeze/PBIM** (not in
  their paper at all). Lean there.
- **vs "Impact of Intrinsic Rewards on Exploration" (2501.11533, 2025) & RLeXplore:** BOTH
  fix the architecture (feedforward/single-LSTM, MiniGrid) and vary the bonus → more
  confirmation of the gap, not competitors.
- **vs MIKASA (2502.10550, 2025) — use as FRAMING, not a competitor:** it gives a memory-task
  **taxonomy (object / spatial / sequential / capacity)** and vector diagnostic envs, but
  studies memory architectures with **no exploration axis**. Organize our env suite by its
  taxonomy (see §5a) and cite it as the memory-benchmark landscape.
- **vs Lazy Agents (Liu 2023):** passive collapse, but *sparse-reward multi-agent* RL —
  not penalty-induced freeze in single-agent memory POMDPs. Different setting.
- **vs "Memory Retention Is Not Enough" (2026):** memory encoded-but-not-used, *no
  exploration axis* — complementary; we give an exploration-side handle on use.
- **vs RATE / Memory Gym / POPGym-Arcade / "Memory-Improvable Domains" (2508.00046):** the
  memory-benchmark landscape we draw envs from; none cross with exploration.

## 5a. Environment suite (reviewer-grounded, 2026-06-25)

Every env is gated by the **memory-demand test** (does Memoryless *fail*? — verified from
our own wandb, not paper descriptions) and organized by the MIKASA taxonomy.

| MIKASA category | env | Memoryless vs recurrent (our runs) | α | role | status |
|---|---|---|---|---|---|
| **spatial** | MysteryPath-Grid | .10 vs .60 ✅ needs mem | α>0 | headline: freeze + amplification | ✅ n=5 |
| **object / cue** | MiniGrid-MemoryS13 | .45 vs .99 ✅ needs mem | α>0 | 2nd embodied, NovelD catalyst | ⚠️ n=2 → P0-8 |
| **sequential / capacity** | TinyReproduce, Autoencode | .00 vs .88 ✅ needs mem | α≈0 | C5 boundary null, exact RM | ✅ reg |

**The 2nd α>0 env — corrected by data + lit review:**
- ❌ **RedBlueDoors DEAD** (our own data): Memoryless **0.82** ≈ recurrent 0.99 → NOT a
  memory task (door state is observable on approach). My earlier recommendation was wrong;
  the runs settle it.
- ⚠️ **POPGym Battleship risky** (POPGym's *own* analysis: "memory minimally effective,
  all 13 models converge to ~same reward"). Use ONLY after a local check shows recurrent ≫
  Memoryless on our config; else drop.
- ⭐ **MiniHack-Memento — THE reviewer-expected add.** Why a reviewer demands it: it is
  **E3B's own headline benchmark** (and NovelD's/RIDE's), so "you build the paper on E3B but
  don't run its benchmark?" is the sharpest possible objection. Why it's safe: **memory
  REQUIRED — confirmed** (cue = sleeping-monster shown only at episode start → leaves view →
  "memoryless feedforward baselines cannot succeed"), same cue→retain→choose structure as
  S13. `Corridor-R2/R3/R5` scales the *exploration* difficulty while holding the cue → a
  built-in amplification knob. CPU-only, no display (unlike MiniWorld — no GL nightmare);
  cost is the NLE C++ build (§5b). **[NEW — top recommendation]**

**T-Maze (Ni 2023) — DOWNGRADED to optional/appendix, PPO-questionable:**
Ni solved it with **off-policy DDQN + ε-greedy** (not PPO); the extreme regime (mem-len
1500) needs specific architectures. On-policy recurrent PPO is feasible only at *moderate*
length, and TBPTT caps the horizon. Worse: short T-Maze has **no exploration** (walk
straight → α≈0, bonus inert); exploration only appears at long length, where PPO's memory
breaks. → NOT a core interaction env. Optional Ni-positioning vignette ONLY if a quick
recurrent-PPO test passes at moderate length (we may have a port from the lmu_ppo thesis).

**Justifiably EXCLUDED (cite + reason — a stronger position than a half-baked run):**
- **Memory Maze** (3D memory standard, Danijar 2022): cite as the 3D extension. Reason is
  now *principled*, not just infra — continuous-3D state breaks count-based exploration (the
  MiniWorld lesson), confounding the interaction with a perception/exploration degeneracy.
- **Craftax** (open-ended, JAX): exclude on pipeline; **cite its finding** "PPO/PPO-RNN >
  ICM/E3B/RND" as independent support for our "bonus help is task-dependent" point.
- **ViZDoom (MyWayHome/Two-Colors)**: engine infra; Two-Colors is α≈0 (living-reward),
  already covered by our register envs.

**2nd freeze env (P1 under amplification-led framing):** MortarMayhem (toggle built — but
run a Memoryless arm first; never had one, memory-demand UNCONFIRMED) or the tiny
sealed-corridor (DP in `optimal_prober.py`); SearingSpotlights = no-sanctuary control.

## 5b. MiniHack-Memento integration scope (handoff)

| piece | spec |
|---|---|
| **dependency** | `pip install minihack` → pulls `nle` (NetHack Learning Env). System deps for the C++ build: `cmake`, `build-essential`, `libbz2-dev`, `flex`, `bison`. **CPU-only, no display/GL** (the upside vs MiniWorld). Add to pyproject `[minihack]` extra + Dockerfile apt. |
| **env ids** | `MiniHack-Memento-F2-v0` (long corridor, 2-fork — primary), `MiniHack-Memento-F4-v0` (4-fork, harder memory), `MiniHack-Memento-Short-F2-v0` (PPO-easy sanity), `MiniHack-Corridor-R2/R3/R5-v0` (the exploration-difficulty knob). |
| **obs** | request `observation_keys=("glyphs_crop","blstats")`. `glyphs_crop` = 9×9 int matrix of glyph IDs (categorical, ~5991 glyphs) centered on agent. NEW frontend needed: **glyph embedding (nn.Embedding) → small CNN → vector**, concat blstats; current FlatEncoder/PixelEncoder handle Box-float not categorical-int-with-embedding. (~the MiniWorld-wrapper level of work, but obs-encoding is the new piece, no GL.) |
| **actions** | navigation tasks use a reduced `actions=` set (8 compass moves), Discrete(8). |
| **reward / success** | sparse: +1 reaching the cue-matched target (terminate); wrong fork = trap = terminate, 0. Emit `is_success`. |
| **wrapper** | `memrl/envs/minihack_wrappers.py` (glyph-crop extract + reduced actions + is_success) + dispatch `MiniHack-*` in `envs/__init__.py`. |
| **registry arm** | `MiniHack-Memento-F2` × SIX cells × {none, e3b, noveld, rnd} × n=3–5. |
| **first check** | local: confirm **Memoryless floors / recurrent solves** (the conjunction gate) before any cluster run — same discipline that just saved us on RedBlueDoors. |

## 6. Evidence ledger

| bucket | items |
|---|---|
| **[n5] clean** | MysteryPath grid: freeze 6/6, rescue, PBIM=none ρ=0, Memoryless+E3B moves-but-fails, idle signal |
| **[n2] directional** | S13: bonus helps cells, Memoryless flat 0.45, NovelD≥E3B reversal |
| **[reg]** | Tiny α≈0 null (E3B≈none) |
| **[free]** | within-cell Δ(bonus−none) ranked by baseline (amplifies-not-equalizes); idle-fraction; DP at trained γ |
| **[NEW] — see §8** | the gaps the stress-test surfaces |

## 7. Honest tensions (state to ourselves)

1. **Exploration AMPLIFIES the cell gap on sparse, it does not equalize** (RetNet/GDN +.5,
   GRU/LSTM +.05 from a shared ~.16–.20 baseline). Frame as **conjunction**, never
   "substitutability"/"weak→strong" (also tuning-confounded — GDN 0.40→0.82 from HP).
2. **"Of course exploration helps"** is the field null. The non-trivial content is
   conjunctive necessity (C2) + geometry-matching (C3) + the freeze (C4), NOT "helps."
3. **Geometry-matching is n=2 on S13** — the most under-powered claim; needs the runs in §8.
4. **RLeXplore proximity** — must be crisp about the delta (systematic + mechanism), or a
   reviewer reads C1/C3 as a restatement.

## 8. Additional experiments (re-ranked for the AMPLIFICATION-LED framing, 2026-06-25)

**P0-HP — THE GATE (run FIRST). Decides whether amplification can headline.**
| # | Experiment | Cost | Have data | Closes |
|---|---|---|---|---|
| **P0-HP** | **Per-cell-best-HP audit of the bonus-lift** across the 6 cells on MysteryPath (+S13). Re-analyze the existing 18-combo×3-cell + GDN local sweeps for cells with best-HP data; small targeted lr/λ sweep for the rest. **Verdict rule (pre-commit): if `e3b−none` is still super-additive in cell strength at per-cell-best HP → amplification HEADLINES (C6 leads). If the gaps collapse → it was tuning; C4 (freeze) leads instead.** | cheap-local (partial free) | partial | **the whole amplification-led framing** |

**P0 — must run before submission.** Four are FREE / have-data and close 2 of the 3 kill-shots.
| # | Experiment | Cost | Have data | Closes |
|---|---|---|---|---|
| P0-1 | Reframe C2 → "third axis"; re-analyze MysteryPath n=5: Δ(cell−Memoryless) widens under bonus (RetNet .09→.39, GDN .06→.30, Memoryless .06→.02). Delete "explains Ni". | free | ✅ | C2 field-null + Ni-overclaim |
| P0-2 | Promote C6 amplification: e3b−none vs cell strength + seed CIs + **cell×bonus interaction-term test** vs additive null. | free | ✅ | "of course exploration helps" |
| P0-3 | Commit the **non-deprecated n=5 post-bugfix** MysteryPath export (penalty+sparse+pbim) to repo as summary CSV. C4 is currently wandb-only / un-auditable. | free | ✅ | headline un-auditable |
| P0-4 | Own optimum-preservation as **bounded + γ-disclosed**: ε=p·E[falls\|π*], V\*=0.42 at γ=0.99, disclose flip by γ≈0.95; stop saying "exactly optimum-preserving". | free | ✅ (DP done) | freeze "γ-fragile" |
| **P0-5** | **2nd freeze env** (sanctuary tiny-corridor [DP exists] or MortarMayhem): penalty-{none,e3b,pbim} × ≥2 cells × n=5. none→0/0, e3b reopens, pbim stays 0/0. | cluster-cheap | ❌ | **freeze single-env (kill-shot #1)** |
| **P0-6** | **Within-env sanctuary TOGGLE** (same env/penalty, with vs without a reachable zero-cost absorbing action, n=5) — isolates "sanctuary" from "penalty magnitude". | cluster-cheap | ❌ | sanctuary-necessity is n=1+n=1 |
| **P0-7** | **Full RND ("mismatched-global") arm on the high-signal cells** (RetNet/GDN/Mamba2) on MysteryPath n=5 — RND currently only on GRU/Memoryless. Completes {matched=E3B, mismatched=RND, same cell, same task}. | cluster-cheap | ❌ | **C3-vs-RLeXplore (kill-shot #3)** |
| **P0-8** | **S13 → n=5** × 6 cells × {none,e3b,noveld,RND}, per-cell-best HP, **pre-registered** paired test + locked decision rule before unblinding. Resolves s0/s1 disagreement + whether NovelD≥E3B survives + whether GDN-none really solves S13 (.94). | cluster-cheap→exp | ❌ | C3 dissociation, S13 n=2 |

**Re-ranking under the amplification-led framing:**
- **P0-HP runs first** (gates whether C6 headlines) — partly free, mostly cheap-local.
- **P0-5/P0-6 (2nd freeze env + sanctuary toggle) DROP to P1** — the freeze is now a
  *supporting* collapse-corner section, so single-env exposure is less fatal (still worth
  one replication for the C4 vignette).
- **P0-7/P0-8 stay P0** — they gate C3's fate AND provide the geometry generality the
  amplification story benefits from. Run them before unblinding; if S13-RND keeps helping,
  demote C3 to "extends RLeXplore."
- **NEW P1 — 2nd α>0 env for amplification/conjunction generality:** **MiniGrid-RedBlueDoors**
  (memory-demanding + sparse + standard intrinsic testbed) × {6 cells} × {none,e3b,noveld,rnd}
  × n=3. Verify recurrent ≫ Memoryless first. (Battleship only if a local check shows memory
  helps on our config — POPGym says it usually doesn't.) **[NEW]**

**P1 — strengthening (if budget):** P1-1 same-cell sign-flip fig (RetNet × {E3B,NovelD,RND} × {MysteryPath,S13}, free once P0-7/8 land); P1-2 per-cell-best-HP audit; P1-3 reconcile headline numbers to converged last5 (GRU .14/RetNet .39/GDN .30) not max; P1-4 **PBIM on an alive/helping arm** (sparse RetNet/GDN — show ρ=0 isn't a frozen-agent artifact); P1-5 freeze **dose-response** p∈{−.008,−.025,−.05,−.1} vs DP threshold; P1-6 Memoryless+E3B with frame-stack k∈{4,8} (capacity-vs-exploration); P1-7 HP-sensitivity envelope as a first-class table; P1-8 related-work delta tables (freeze vs Lazy-Agents vs BAMDP-Shaping).

**P2 — defer (rebuttal-phase only):** pre-registered measured novelty-geometry α; predictive-geometry env; full per-cell-best-HP grid + per-bonus λ + matched-param budget; Ni's Passive/Active T-Maze cross. **Do NOT spend cluster here pre-submission** — these only matter if we keep over-claiming C1-as-ranking / C3-as-mechanism.

**3 kill-shots if P0 skipped:** (1) headline un-auditable + single-env [P0-3,5,6]; (2) C2 is field-null + own data refutes "missing half" [P0-1,2, FREE]; (3) C3 is RLeXplore restated on n=2 noise with the RND leg pointing the wrong way [P0-7,8].

## 9. Section outline

1. **Intro — two halves studied apart.** The gap + the freeze in 30 seconds.
2. **Setup** — cell zoo, bonus suite, env suite, the return-matched penalty/sparse arms.
3. **The interaction (C1)** — F1, the matrix.
4. **Conjunctive necessity (C2)** — F2 + the Ni-2023 resolution.
5. **Geometry-matching (C3)** — F3 + the RLeXplore resolution.
6. **The freeze (C4)** — F4–F5, the sharp phenomenon + PBIM control + DP.
7. **Boundary & limits (C5)** — α≈0 register null; no-3D (continuous-state exploration
   degeneracy — the MiniWorld methods note); per-cell-HP honesty.
8. **Related work / conclusion.**
