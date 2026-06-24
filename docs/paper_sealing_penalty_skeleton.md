# Paper skeleton — "The Sealing Penalty" (phenomenon-first reframe, 2026-06-24)

> **Supersedes the headline of `paper_full.md` ("Exploration Selects the Reward Machine a
> Memory Agent Learns") and `paper_skeleton.md` ("Decodability Is Not Use").** Same empirical
> spine, demoted theory. Decision + adversarial stress-test that produced this: session
> 45ff0f88 (wf_756fe884). The reward-machine apparatus is kept ONLY as a light interpretation
> (Observations, not Theorems); the headline is the **phenomenon**.

Evidence tags: **[n5]** MysteryPath n=4–5 grid (clean) · **[n2]** S13 directional · **[local]**
local/low-n · **[NEW]** needs a run · **[free]** no-compute analysis on existing logs.

---

## 1. Thesis (one sentence)

> A **faithful, return-matched, optimum-preserving** off-path penalty **seals** a sparse-reward
> memory POMDP — every recurrent cell collapses to an absorbing do-nothing policy (the *freeze*:
> success 0.00 **and** falls 0.0, 6/6 cells) — and only a **non-potential** episodic bonus
> reopens it (a policy-invariant *potential* with the identical signal provably cannot), with
> memory and exploration **conjunctively** (not substitutably) necessary to escape.

This is the intuitive memory×exploration paper, carried by the cleanest, most novel object in the
data (the freeze + the PBIM control), with the over-abstraction (RM-selection / Mealy–Moore
"theorem") demoted and the unsupported claims (substitutability, "behavioral-not-densification")
cut.

## 2. Title options
- **The Sealing Penalty: How a Faithful Penalty Collapses Memory Agents, and Which Exploration Reopens Them**
- Sealed Into Inaction: Optimum-Preserving Penalties, the Freeze, and Non-Potential Exploration in Memory POMDPs

## 3. Contributions (claim exactly these)

- **C1 — The freeze (the phenomenon).** A faithful −0.008 off-path penalty, *return-matched* to a
  sparse twin (same optimum), drives **6/6 memory cells** (GRU, LSTM, RetNet, GDN, Mamba2,
  +Memoryless) to an absorbing do-nothing policy: success **0.00 AND falls 0.0**, while the
  return-matched `sparse-none` twin stays *alive* (success .16–.20, falls 9–39). The penalty does
  not change the optimum yet destroys learning. **[n5]**
- **C2 — Only non-potential exploration reopens it (the control that makes it a mechanism).**
  Episodic non-potential bonuses rescue every memory cell (E3B .24–.62, NovelD .17–.71); the
  **identical E3B signal delivered as a policy-invariant potential (PBIM)** leaves the agent
  frozen (ρ = (pbim−none)/(e3b−none) = 0) — and on the *sparse* arm it even mildly freezes
  capable cells (GRU/LSTM → ~0). The observable signature is **idle-fraction**: non-potential
  bonuses drive idle → .01–.04; `none`/PBIM stay .21–.43. PBIM≈none is the BAMDP-Shaping
  (Ng-1999) prediction — we use it as the **control** it licenses, not as a theorem. **[n5]**
- **C3 — Conjunctive necessity (the memory×exploration interaction, stated correctly).** Memory
  and exploration are **complements, not substitutes**: `Memoryless+E3B` has the *highest* falls
  (22.3) yet 0.00 success (coverage, no memory); `cell+none` freezes (memory, no coverage); only
  the conjunction escapes. On S13, Memoryless is flat 0.45 across every bonus. **[n5 + n2]**
- **C4 — Boundary conditions that sharpen the mechanism.** The freeze requires an **avoidable**
  penalty (a reachable do-nothing sanctuary): within-env sanctuary toggle / SearingSpotlights
  (wandering spotlights, no sanctuary) does **not** freeze **[NEW — gating run]**. And when there
  is **no coverage to gain** (forced-retention register task, TinyReproduce), the bonus is inert
  (e3b ≈ none everywhere) — the α≈0 null. **[n on Tiny; SS/toggle NEW]**
- **C5 — (Methods, cautionary) Behavioral, not decoded.** We measure realized behavior, not
  decodability: a teacher-forced decode probe credits an *untrained* net nearly as much as a
  trained one (GRU random 7.03 ≈ trained 7.04 bits) — copy-capacity, not learned memory. We
  retire it to a cautionary result + random-init floor. **[local]**

**CUT from contributions:** "exploration selects the reward machine" headline; the Mealy–Moore
"theorem"/"keystone" language (→ a one-line Ng-1999/BAMDP corollary used as the PBIM control);
substitutability / "weak cell → strong cell" / curriculum (data refutes the sign — see §6); the
"behavioral not densification" dichotomy (false under our own densification theory; the PBIM null
is a double-null that cannot isolate it — see §6); eff-RM-size as an "instrument" (→ a demoted,
caveated coverage-style figure).

## 4. Figures (priority order)

- **F1 (HEADLINE) — the freeze + the rescue, return-matched.** Per-cell bars: `sparse-none`
  (alive) vs `penalty-none` (0/0, 6/6) vs `penalty-e3b`/`penalty-noveld` (rescued) vs
  `penalty-pbim` (still 0/0). The return-matched twin is the load-bearing control. **[n5]**
- **F2 — the idle-fraction mechanism.** idle by arm×bonus: non-potential → 0; potential ≥ none.
  The behavioral bridge from "what" to "why." Already logged (`action_frac_0`). **[n5, free]**
- **F3 — conjunctive necessity.** The 2×2 of {coverage?, memory?}: Memoryless+e3b (coverage, no
  memory → 0), cell+none penalty (memory, no coverage → freeze), cell+e3b (both → rescue);
  S13 Memoryless flat 0.45. **[n5 + n2]**
- **F4 — the sanctuary boundary.** within-env sanctuary toggle (freeze ↔ no-freeze) and/or
  SS-anti (no freeze). The causal test that the freeze needs an *avoidable* penalty. **[NEW]**
- **F5 — generality beyond one gridworld.** the freeze/rescue/PBIM contrast replicated on a 2nd
  env (S13 penalty arm) and 3D pixels (MiniWorld-Sign). **[NEW]**
- **Appendix:** DP optimality bound at the TRAINED γ (see §5); α≈0 Tiny null; retired decode
  probe + random-init floor; per-cell-best-HP cell-ranking audit (cell ordering is HP-conditioned:
  GDN 0.40→0.82 from HP alone).

## 5. Honesty fixes baked in (do these before submission)

- **DP / K4 at the trained γ, not γ=1.0.** The "V\*=0.76, ~4× margin" is undiscounted. At the
  actual training **γ=0.99** (`train.py:565`): **V\*=0.42, p̄=−0.0255 (~3.2×)** — freeze still
  *strictly* sub-optimal, so K4 PASSES, but report this number and **disclose** that by γ≈0.95
  freezing becomes optimal. (Verified: `optimal_prober.py --gamma 0.99`.) Cite the DP as a
  corridor-POMDP *consistency check*, never as proof about MysteryPath.
- **eff-RM-size demoted.** It is `knowledge.tobytes()` distinct-grid-configs ≈ coverage, n=1
  cell/snapshot. Present as a descriptive figure with the caveat in-text ("tracks visited-state
  count; success+falls carry the freeze"). Either run the coverage-conditioned control or drop it.
- **No "theorem."** "A policy-invariant potential cannot change the realized reward stream's
  argmax (Ng et al. 1999; BAMDP-Shaping, Lidayan et al. 2025); we use PBIM as the control this
  corollary licenses." One sentence.
- **Per-cell-best-HP** for any cell-ranking claim; disclose the GDN tuning result.

## 6. The honest tensions (state these to ourselves, not the reviewers)

1. **Substitutability is dead — the sign is backwards.** On sparse MPG all cells start ≈.16–.20;
   the e3b delta is set by *capacity to use coverage*, not weakness: RetNet +.53, GDN +.46 vs GRU
   +.07, LSTM +.02. The bonus **amplifies** the architecture gap. ⇒ headline conjunction, not
   substitution. **[free table — §C3]**
2. **"Behavioral not densification" is unsafe.** Non-potential bonus densifies the critic by our
   own theory (idle→0 is a *consequence* of densification); the PBIM null is a double-null (no
   V_int gradient once frozen; harmful TD-lag on sparse) that cannot isolate the channel. Stay
   agnostic on the mechanism, or run PBIM on an *alive/helping* arm (sparse RetNet/GDN) to
   adjudicate. **[NEW if we want the claim]**
3. **The freeze lives on one env.** Single-env exposure is the #1 acceptance risk. C4/F4/F5 exist
   to close it.

## 7. Section outline

1. **Intro — a penalty that changes nothing, and breaks everything.** The freeze, in 30 seconds.
2. **Setup.** Cells, bonuses, the return-matched penalty/sparse/aligned arms, the behavioral
   readouts (success, falls, idle).
3. **The freeze (C1).** F1 + the return-matched control + the trained-γ DP bound.
4. **Reopening it: non-potential vs potential (C2).** Rescue + the PBIM control + idle mechanism.
   The Ng-1999/BAMDP corollary as the control's license (one paragraph, no theorem).
5. **Memory × exploration is conjunctive (C3).** F3 + the Memoryless/α≈0 controls.
6. **Boundary conditions & generality (C4, C5/F4–F5).** sanctuary toggle; 2nd env; 3D pixels;
   the cautionary decode-probe note.
7. **Related work.** BAMDP-Shaping (PBIM=control), Ni 2023, RM-RL (we don't assume an RM), the
   exploration lit (we add the penalty-seal + conjunction the lit doesn't predict), RATE (genre).
8. **Limitations.** single-env→multi-env honesty; corridor-DP is an abstraction; eff-RM-size≈
   coverage; cell ranking is HP-conditioned.
