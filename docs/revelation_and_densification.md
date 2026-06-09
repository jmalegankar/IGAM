# Revelation & densification — the theory upgrade for the Memory × Exploration paper

*Synthesizes the SCDP (Sealed-Credit Decision Process) analysis [external theory chat, 2026-06-08]
into this repo's experimental program. Supersedes the framing in
`dense_memory_exploration.md` where they conflict; that doc's experiment designs stand.
All citations below were verified by the source chat against primary records; **re-verify
before camera-ready** (especially Toro Icarte ICML 2018 page numbers).*

---

## 1. The core conceptual upgrade: TWO axes, not one

The paper so far used one axis: reward **density** (sparse ⇄ dense). The SCDP analysis shows
that what actually governs whether dense rewards / bonuses can help is a *different* axis:

**Revelation** — does the goal posterior `Pr(G | F_t)` (the Doob process of the goal event,
conditioned on the *observable history* `F_t = σ(o_{0:t}, a_{0:t-1})`) move in a controllable,
action-discriminative way before the terminal step?

- **Revealing task:** probing moves the posterior — progress information *exists* in the
  observation channel, whether or not the reward pays it out.
- **Sealed task** (SCDP axiom (S)): the posterior is flat pre-terminal — the optimal advantage
  has *no* `F_t`-measurable content. No history-measurable dense reward can be both faithful
  and informative (degeneracy lemma: the ideal potential `Φ* = V*(b_t)` is a function of `t`
  alone, so its shaping term cancels out of every action comparison). Any *informative* dense
  proxy is then hackable in the Skalse et al. (NeurIPS 2022) sense. Sealing can even be
  *computational* (Golowich–Moitra–Rohatgi FOCS 2024, via LPN-hardness).

Reward **density** is then a property of *how much of the available revelation is paid out as
reward*:

| | revelation present | revelation absent (sealed) |
|---|---|---|
| **paid** (dense reward) | Regime 1 — dense memory tasks | *impossible* (nothing faithful to pay) |
| **unpaid** (sparse reward) | **Regime 2 — sparse-but-revealing** | Regime 3 — forced-sparse (SCDP) |

**Key fact about our benchmarks (proved in the source chat):** MysteryPath-Grid and
MiniGrid-Memory are **revealing, not sealed**. MysteryPath reveals path membership through the
red-cross fall-off feedback + relocation-to-origin (and natively *ships* a faithful dense
reward, the +0.1 first-visit progress term). MiniGrid-Memory's cue is *observed* — the memory
demand is retention, not sealed inference. So **all our experiments live in the left column**,
moving between Regime 1 and Regime 2. That is not a weakness — it is the regime where theory
says the bonus question is genuinely *quantitative* (in sealed tasks, theory already answers
it: bonuses provably can't help). The paper should *say this explicitly*: we chart the regime
where the answer is contingent, and inherit the sealed regime's answer from theory.

## 2. The sharpened thesis: bonuses are revelation *monetizers*

All our intrinsic bonuses (RND, NovelD, E3B — including E3B's IDM-learned φ) are computed from
observations/actions, i.e. they are **`F_t`-measurable dense signals**. The thesis becomes:

> **A novelty bonus converts observation-channel variation into reward. It can therefore help
> only where *unpaid revelation* exists — Regime 2. In Regime 1 the revelation is already paid
> (the bonus adds only its bias); in Regime 3 there is nothing to monetize (novelty is
> goal-orthogonal by the sealing axiom).**

### The benefit decomposition (proposition to state in the paper)

Write the shaped reward `r = r_ext + λ·b`. Decompose the bonus against the task's faithful
progress potential `Φ*` (which *exists* in revealing tasks):

```
b_t  =  α·ΔΦ*_t  +  ε_t          α = alignment of the bonus with faithful progress
                                  ε = residual (novelty unrelated to progress)
benefit(bonus) ≈ densification-gain(sparsity, α) − bias-cost(λ, ε)
```

- **Densification gain** is critic-side: with sparse `r_ext`, the `α·ΔΦ*` component supplies
  the pre-terminal value-learning signal the task withholds (observable as faster
  `explained_variance` recovery — our B6 metric). It *vanishes* when `r_ext` is already dense
  (the faithful signal is paid → redundant).
- **Bias cost** is policy-side: the `ε` component is a non-potential-based reward — it shifts
  the optimal policy of the shaped MDP (Ng–Harada–Russell 1999 necessity; PBIM), and any
  informative proxy is hackable (Skalse 2022). This cost is paid in *every* regime.

**Predictions** (each tested by an arm we have or have staged):

| Regime / arm | densification | bias | net e3b−none |
|---|---|---|---|
| Regime 2: sparse MysteryPath | large | small vs. it | **> 0** ✓ (20M headline) |
| Regime 1-anti: dense-**fall** (−0.008 off-path) | ≈0 (paid) | **antagonistic** | **< 0** (B2, staged) |
| Regime 1-aligned: dense-**progress** (+0.1) | ≈0 (paid) | redundant-ish | **≈ 0** (old runs — complete!) |
| Regime 3: sealed | — | only bias | ≤ 0 (theory; optional micro-demo) |

The **non-monotonicity** is the headline: moving dense → sparse → sealed, the bonus's benefit
goes **≤0 → >0 → 0**. "Exploration bonuses don't solve hard exploration; they monetize cheap
information." The naive belief (sparser ⇒ bonuses help more) is wrong at *both* ends.

### Why dense-fall is *antagonistic*, mechanistically (and how we measure it)

E3B pays for novel embeddings. In MysteryPath, off-path probes are (i) novel → subsidized by
the bonus, and (ii) penalized −0.008 in the fall arm (note: below the per-step bonus scale ~0.02, so the bonus out-bids the fine at the margin — num_fails is the diagnostic; −0.05 is the robustness lever if needed). Moreover, after a fall the *correct*
behavior is retracing the memorized path prefix — minimally novel → the bonus actively
*disincentivizes* the right recovery. So in the fall arm the bonus pushes against the task
signal at the margin. **Measurable signature: `rollout/ep_num_fails_mean` (now logged) should
be higher under e3b than none in the dense-fall arm.** In the sparse arm the same probing is
*unpriced information gathering* — there the subsidy is roughly aligned with the optimal
(probe-to-learn) policy. This is the concrete, falsifiable face of "bias cost."

### Bonus ⊥ policy-memory (verified) — and the two-memories point

`e3b_idm` is **architecturally independent of the policy's memory**: `IDMPhi.encode` uses the
observation only (`phi_sources.py:55`); `cell_state`/`side` pass through the interface unused;
the IDM trains on `(o, a, o')`. Coupling to the policy is behavioral only (bonus → reward →
policy → trajectory distribution). Consequences:
1. **Fair cell comparison** — every architecture faces statistically the same bonus; the cell
   ranking under e3b is not contaminated by the bonus reading cell internals.
2. **E3B carries its own memory** — the per-episode ellipsoid is an episodic memory *of the
   reward channel* (a soft first-visit detector over φ-space). Two memories: the policy's (for
   acting) and the bonus's (for credit) — the "memory migrated into the reward function"
   phenomenon. The bonus is `F_t`-measurable (SCDP-admissible / C1-compliant) precisely because
   the ellipsoid is a function of observable history. (`CellInnovationPhi`/`eps_mem` is the
   in-house C1-*violating* contrast class — a future ablation.)
3. **Curriculum confound, adjudicated by B2.** Systematically collecting an episodic-novelty
   bonus requires the *policy* to know where it has been — chasing e3b is a memory-exercising
   auxiliary objective. Rival explanations for "e3b helps": **densifier** (benefit = reward
   densification → vanishes/flips in dense arms) vs **curriculum** (benefit = memory training →
   persists in dense arms). The dense-fall arm discriminates: `e3b ≤ none` ⇒ densifier;
   `e3b > none` ⇒ curriculum. Either way B2 names the mechanism.

### Why e3b ≫ noveld (retroactive, now explained)

The faithful densifier on MysteryPath is *first-visit path-tile progress* (the env's own +0.1
term). E3B-episodic ≈ pays first-visit-ish novelty per episode → high α with the faithful
signal, *by construction non-saturating* (the ellipsoid resets every episode). NovelD/RND's
estimator trains to convergence on the visited distribution (`noveld_loss → 0`) → its α decays
to zero over training. **Bonus benefit tracks alignment-over-time with the faithful densifier**
— a measurable claim (see alignment probe, §4).

## 3. Exactness results we get for free (state as lemmas)

1. **Fall penalty preserves the optimal policy EXACTLY.** For any policy π:
   `J_fall(π) = J_sparse(π) − p·E[Σ discounted falls]  (p=0.008) ≤ J_sparse(π)`, with equality iff π
   never falls. The sparse-optimal π* never falls ⇒ it stays optimal, and
   `J_fall(π*) = J_sparse(π*) = 1.0` (return-matched). The fall arm changes *learnability
   only*, not the solution. (This is why we chose it over progress for B2.)
2. **Progress reward is ≈ potential-based.** `+0.1·(first-visit tile)` = `Φ' − Φ` with
   `Φ = 0.1·(#tiles visited)` — exactly PBRS at γ=1, approximately at γ=0.995 (per-step drift
   ≤ ~0.0005·N), plus the standard episode-termination caveat (Grzes). So the progress arm is
   *near*-policy-invariant but return-inflating (max ≈ 1.8 ≠ 1.0) — exactly the magnitude
   confound the fall arm removes. Keeping BOTH arms turns the confound into a *robustness
   axis*: if the flip holds under both densifications, it isn't an artifact of either.
3. **Both dense arms are faithful; the bonus is not.** The dense arms ≈ PBRS w.r.t. observable
   counters (falls, first-visits) — policy-preserving. The bonus is non-potential-based and
   policy-shifting. **B2+ therefore already contains the faithful-vs-unfaithful densifier
   contrast** that we previously scheduled as a separate B3 run. The remaining surgical B3
   (PBRS-ified *bonus*) stays on the roadmap but drops in priority.

## 4. What this does to the experiment program

**Status windfall:** the old "+0.1 progress" project (`memrl-mpg-densetoggle`) is **complete**
— 3 cells × {none, e3b} × 5 seeds, all finished at 10M. It is no longer a deprecated design;
it is **arm (c), Regime 1-aligned**, free of charge.

| # | Experiment | Regime / role | Status |
|---|---|---|---|
| 1 | **B2 sparse + dense-fall** (staged) | Regime 2 + Regime 1-anti; the flip | **launch as-is** (now with `num_fails`/`success` logging) |
| 2 | **Arm (c) re-analysis** — old +0.1 runs | Regime 1-aligned; bonus ≈ redundant | **free**, pull from wandb |
| 3 | **Alignment probe** (local, ~1–2M CPU) | measure `corr(b_t, faithful-progress_t)` for e3b/noveld/rnd on sparse MysteryPath → predicts their benefit ordering | cheap, high value (mechanism figure) |
| 4 | **B6** expl_var-recovery vs density | critic-side densification mechanism | free (logged) |
| 5 | **B3-surgical**: PBRS-ified bonus | causal isolation of bias term | after B2 lands |
| 6 | **Sealed micro-demo** (purpose-built tiny env) | Regime 3 boundary | optional / appendix |
| 7 | **B4** POPGym dense-memory | Regime 1 generality (independent env suite) | cheap, vector envs |

**The headline figure** becomes a 3-bar (or 4-with-sealed) panel per cell: `e3b − none` on
{dense-anti, dense-aligned, sparse} — predicted {−, ≈0, +}. Four ordered, theory-derived
predictions from one decomposition; even partial confirmation (e.g. dense-anti ≈ 0 instead
of < 0) is interpretable inside the framework (alignment term small), not a dead paper.

### The sealed-arm trap (do NOT build it naively)

Tempting follow-up: "seal MysteryPath" (no red cross, no relocation signature) to realize
Regime 3 in-env. **This breaks axiom (M):** MysteryPath's memory demand *is* remembering
observed fall-feedback; remove the feedback and there is nothing to remember — with a fixed
maze it collapses to a memoryless position→action task, with per-episode random mazes it
becomes unlearnable noise. (Same lesson the source chat proved for MiniGrid-Memory: retention
tasks get (M) *from* observability; sealing kills them.) A correct Regime-3 instance needs
(M) and (S) from *different* latents — e.g. **cue ⊕ key**: an observed episodic cue `c` (must
be remembered → M) combined with a never-observed key `k` (sealed → S), terminal reward
`1[a_τ = f(c,k)]`, with `|k|` large enough that the 1-bit/episode terminal channel can't leak
it within budget. Expected result: *nobody* solves it, **and the bonus is demonstrably
busy-but-useless** (novelty consumed, return flat) — paired with a privileged-unsealed control
(give the shaped reward access to `k` → solvable), which proves the bottleneck is sealing,
not capacity. Small vector-obs env, GRU only, 3 conditions × 5 seeds — appendix-scale.

## 5. What changes in the paper's claims (and what must NOT be claimed)

- **Do not claim** "memory tasks need sparse rewards" — false in general (Regime A/RDP), and
  our own envs are counterexamples.
- **Do not pitch** MysteryPath as exploration-hard in the sealed/combination-lock sense. Its
  difficulty is *unpaid revelation* + memory, which is exactly why bonuses work there.
- **Do claim** the two-axis decomposition (density ⊥ revelation ⊥ memory), the monetizer
  thesis with the benefit decomposition, the within-task three-regime realization, and the
  practitioner rule:
  1. Is progress revealed in your observations? **No** → bonuses can't help; densify by
     *unsealing* (instrumentation, verifier decomposition — cf. RLVR vs hackable PRMs),
     not by novelty.
  2. Revealed and already paid (dense)? → skip bonuses; expect harm if your dense signal
     anti-correlates with novelty (penalty-shaped tasks).
  3. Revealed but unpaid (sparse)? → bonuses help; prefer *episodic, non-saturating* bonuses
     whose novelty aligns with the task's faithful progress variable (E3B-style over
     RND-style).
- **Theory status honesty:** the propositions are *assembled* from cited results (Ng 1999;
  Skalse 2022; Liu et al. 2022; Devlin–Kudenko 2012; Bacchus et al. 1996) — present as
  propositions-with-cited-lemmas, claim the *synthesis and the experimental realization* as
  the contribution, not the component theorems.

## 6. Risks

- **Dense-anti arm shows ≈0 instead of <0:** framework survives (α small / penalty rarely
  binding); headline softens from "harmful" to "useless when paid." Check `ep_num_fails_mean`
  to see whether the antagonism mechanism even engaged.
- **Freeze-at-start in dense-fall `none` runs:** watch return ≈ 0 + low `num_fails`
  simultaneously (the freeze signature — now distinguishable from "probes carefully").
- **Old +0.1 runs predate the B2 episode-boundary fix** (~0.8 % uniform mis-scoring,
  e3b arms only) — negligible for the ≈0 prediction; note in appendix.
- **Citation re-verification** before camera-ready (all inherited from the source chat).
