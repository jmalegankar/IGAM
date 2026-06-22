# Pre-registration — "Exploration Bonuses and Recurrent Memory" (AAAI)

> **RE-FILED 2026-06-21 — see `paper_skeleton.md` § "Appendix A. Pre-registration
> (re-filed)".** This file's headline (the freeze fixed point) and its mechanism leg
> (S1 "behavioral, not representational," carried by the decode / lag-Δ retention probe —
> §3 S1b/S1c, §7 item 3) are **superseded**. The decode probe was shown to measure
> copy-capacity, not learned memory (random-init untrained nets decode ≈ as much as
> trained, because it is teacher-forced) and is **retired** to a cautionary methods result.
> The headline moved from freeze-only to the **realized reward-machine** thesis. Per this
> file's own §0 honesty rider, that move is documented here, not silent. Read Appendix A
> for the governing registration; everything below is retained for provenance.

**Filed 2026-06-16, BEFORE unblinding the n=5 post-fix grid.** Companion to
`experiment_plan_v2.md` (§3 decision rules, §7 stub — this document supersedes and
locks §7), `theory_v2_memory_training.md`, `project_aaai_review_findings`.

This document fixes the **headline claim, primary metric, statistical tests,
equivalence margins, metric definitions, and inclusion rules** before any post-fix
`memrl-memtrain-*` run is read at n=5. All prior wandb projects (`memrl-mysterypath-grid`
etc.) are DEPRECATED (n=3, lr 2.5e-4/λ0.01, pre-bugfix code) and are reported, if at
all, only as pilot/directional data clearly labelled as such — never as a registered
result.

---

## 0. Why this file exists (the honesty rider)

The mechanism narrative moved three times during development
(densification → credit-to-memory-writes → behavioral/non-optimum-preserving), each
pivot forced by a contradicting result we found ourselves. That history is a
**liability if the final signs are read as post-hoc, and an asset if we register the
final claim and its decision rules now and then report confirmations *and* failures
against this file.** We therefore commit to:

1. Reporting every registered test **in the direction it falls**, including the ones
   that can falsify our own theory (PBIM adjudication, λ-sweep, lag-Δ retention,
   k=1 collapse). A null or a flip is a reported result, not a discarded run.
2. Listing every excluded run and the rule that excluded it.
3. Not introducing a new post-hoc grouping, sign, or threshold after unblinding
   without flagging it explicitly as exploratory.

---

## 1. The HEADLINE claim (locked)

The paper leads with **one** object, chosen because it is the only finding outside
the four nearest priors (BAMDP-Shaping / Lidayan-Dennis-Russell ICLR 2025; Ni et al.
NeurIPS 2023; Henaff 2023; Taiga et al. 2020):

> **The freeze fixed point.** A *faithful, approximately optimum-preserving* fall
> penalty (−0.008, in a regime where the per-step fall probability of the optimal
> policy p ≪ p̄ ≈ 0.1–0.25, so persistent freezing is *provably sub-optimal*) drives a
> recurrent learner into an **absorbing, do-nothing fixed point** (eval success → 0,
> falls → 0) in all 6 cells. A non-potential exploration bonus **un-seals** it
> (rescues success to ≈ its sparse-bonus level). The bonus's role here is
> **anti-freeze un-sealing, not value-of-information exploration** — and crucially,
> the *penalty self-seals while the bonus un-seals*: an asymmetry outside the
> BAMDP-shaping frame (which only adds reward to aid exploration and does not predict
> a faithful penalty sealing a recurrent learner into an absorbing state).

**Supporting (not headline) claims**, each with a locked rule in §3:
- (S1) The rescue mechanism is **behavioral** — the bonus drives idle-fraction → 0
  (the load-bearing behavioral signal), not a more decodable memory representation.
- (S2) The benefit is **non-optimum-preserving** — it largely vanishes when the same
  bonus is delivered as a policy-invariant potential (PBIM). *Caveat (registered):*
  this is BAMDP-shaping's own prediction (the benefit = the non-potential bias/VoI
  term), so S2 is presented as **confirming the BAMDP decomposition on recurrent
  gridworlds**, not as a novel surprise. The novelty is the freeze object, not S2.
- (S3) The effect is **memory-dependent** — Memoryless + bonus moves maximally yet
  succeeds 0 (entanglement); under `none`, no cell beats the Memoryless floor.

**What would falsify the headline:** (a) penalty-none does NOT drive falls → 0 (i.e.
no absorbing fixed point — freezing was just slow learning); or (b) the −0.008 regime
is shown to be one where freezing is in fact optimal/near-optimal play (p ≳ p̄, so the
penalty is not optimum-preserving and there is no pathology). Both are checked in §3.

**Demoted to context (NOT headline):** the α-taxonomy (descriptive only — §5),
"bonuses train memory" (refuted by S1), and the 20M "3×" number (reported honestly
with seed counts, not as the contribution).

---

## 2. Primary metric, sample, and statistics (locked)

- **Primary metric:** `eval/success_rate` (max 1.0 on **all** arms, including
  penalty/aligned). Shaped/episodic return is reported only in a separately labelled
  panel and is **never** used to rank across density arms (the frozen agent's shaped
  return 0.0 beats the rescued agent's negative return — return ranks doing-nothing
  above working, so it is disqualified as a cross-arm headline metric).
- **Sample:** n = 5 seeds per cell × density × bonus. A run counts only if it reached
  **≥ 95 % of its step budget** (§6 exclusions).
- **Effect tests (difference, e.g. e3b−none):** per-cell two-sided **Mann-Whitney U**
  (primary; rank-based, no normality assumption) with **Welch's t** as a secondary
  check; report effect size + **bootstrap 95 % CI** (10 000 resamples over seeds).
  An effect is "confirmed" only if the CI excludes 0 in the predicted direction.
- **Null/equivalence tests (e.g. "PBIM ≈ none", "e3b ≈ none on retention"):** a null
  is NEVER claimed from p > 0.05. We use **TOST equivalence** with a pre-registered
  margin (§3, per test) — the null is "confirmed" only if the 90 % CI of the
  difference lies **entirely within** ±margin. If the CI is wider than the margin, the
  result is reported as **"inconclusive (underpowered)"**, not as a null.
- **Pooling:** primary analysis is **per-cell**. Any pooled-across-cells number is
  secondary, reported with the per-cell scatter, never as the sole evidence.

---

## 3. Locked decision rules (per registered finding)

Metric is `eval/success_rate` at matched budget unless stated. "CI" = bootstrap 95 %
over seeds. Each rule states the **confirm** and **falsify/null** branches.

### F2 — Freeze fixed point + rescue (the headline)
- **Freeze:** confirmed iff `penalty(−0.008)-none`: success ≈ 0 (CI upper < 0.05)
  **AND** `ep_num_fails` → 0 (the absorbing-state diagnostic; CI upper < 1 fall/ep),
  while `sparse-none` on the same cell has falls > 0 AND success > 0. Required in
  **≥ 5/6 cells** to call it general. *Falsify:* if falls stay > 0 under penalty-none,
  there is no absorbing fixed point — report as "slow learning," not freeze.
- **Rescue:** confirmed iff `penalty-e3b` success ≥ 0.5 × `sparse-e3b` (CI lower > 0)
  on the cells that froze.
- **−0.008 vs −0.1 (registered framing):** the **pathology claim uses ONLY −0.008**
  (the optimum-preserving regime). −0.1 is reported as a *separate* "rational-freeze"
  regime (where freezing may be near-optimal), never folded into the pathology
  evidence. We additionally report the measured `ε = p·E[falls|π*]` (§E6 tiny-maze DP
  + measured `ep_num_fails` of the converged sparse agent) to substantiate that
  −0.008 is return-matched to sparse within ε.

### S1 / behavioral mechanism — idle-fraction (load-bearing)
- **Definition (locked):** `idle_frac` = fraction of environment steps at which the
  agent emits the no-op / "stay" action (action index 0 in MysteryPath-Grid),
  averaged over **≥ 100 evaluation episodes** rolled from the final snapshot with the
  **deterministic (argmax) policy**. Logged as a first-class metric `eval/idle_frac`
  (to be wired into eval before unblinding), distinct from the training-time
  `debug/action_frac_0` histogram.
- **Rule:** S1 confirmed iff `idle_frac(e3b) < 0.05` **AND** `idle_frac(none) > 0.15`
  **AND** `idle_frac(pbim)` is within 0.05 of `idle_frac(none)` (i.e. PBIM idles like
  none). Reported with CIs across n=5. *This is the registered evidence that the
  rescue is behavioral; it does not depend on the decodability probe.*

### S2 — PBIM adjudication (non-optimum-preserving), with equivalence test
- **Recovered-benefit fraction (locked):** `ρ = (pbim−none)/(e3b−none)` on success at
  matched budget, per cell, bootstrap CI on ρ.
  - `ρ ≤ 0.25` (CI upper < 0.25) → **non-optimum-preserving / behavioral (S2 confirmed)**.
  - `ρ ≥ 0.75` (CI lower > 0.75) → **densification dominant (S2 FALSE)** — report it.
  - otherwise → **partial**, reported as such.
- **Equivalence leg:** "PBIM ≈ none" is claimed only if TOST on `(pbim−none)` is
  inside the margin `δ_eq = 0.25 × (e3b−none)` (i.e. PBIM recovers < a quarter of the
  raw benefit), 90 % CI. Underpowered → "inconclusive."
- **Dose-response (registered, adjudicates densification vs behavioral):** interpolation
  arm `r = r_ext + λ[(1−β)·raw_e3b + β·potential_e3b]`, β ∈ {0, 0.25, 0.5, 0.75, 1}
  on GRU + RetNet, sparse MysteryPath. *Prediction under S2:* success(β) **monotone
  decreasing** in β (benefit erodes as the bonus is potential-ized). *Under
  densification:* success(β) **flat** in β. Report the slope with CI either way.
- **Validity guard (already verified):** PBIM telescoping `disc_sum` std/|mean| < 0.3
  (Φ = V_int, valid potential); re-confirm on the n=5 runs.

### S1b — Decodability (F1) [downgraded until the retention probe lands]
- **Rule (as in plan §3):** decode minimal-RM state from the frozen recurrent state,
  **matched-coverage bank**, report **linear AND MLP** probes with bootstrap CIs.
  Confirmed-helpful iff `sparse-e3b > sparse-none` in resolved bits (CI-separated)
  AND `penalty-none ≈ chance` (eff_rm_size ≈ 1) AND `penalty-e3b > chance`.
- **Registered caveat:** on MysteryPath the path stays observable, so K saturates and
  a flat result is **uninterpretable (power, not science)**. We therefore do NOT state
  the strong negative ("not representational") from MysteryPath alone. The
  load-bearing instrument is the lag-Δ retention probe below.

### S1c — lag-Δ retention probe on Autoencode (settles "not representational")
- **Setup:** decode the remaining-to-reproduce suit (the **exact** minimal-RM state,
  `info["rm_state"]`) from the recurrent state at the *use* step, as a function of
  **lag Δ** since the item was last observed (Δ up to 2N−1; info is NOT observable at
  use time → forced retention).
- **Positive control (MUST pass or the probe is declared blind, not the science):**
  the probe must (i) recover a just-shown item (Δ small) above the shuffled-label
  chance floor, AND (ii) rank strong cells > Memoryless. If the positive control
  fails, we report "probe underpowered," not a null.
- **Main rule:** S1 (behavioral, not representational) **confirmed** iff, conditional
  on the positive control passing, decode-accuracy(e3b) ≈ decode-accuracy(none) at
  matched long lag (TOST, margin = 0.5 × the Δ=small → Δ=long accuracy drop).
  **FALSIFIED** iff `e3b > none` (CI-separated) at long lag — the bonus *does* improve
  the retained representation; thesis S1 is wrong and we report it.

### S3 — Entanglement / memory-dependence
- Confirmed iff `Memoryless+e3b ≤ Memoryless+none` (CI), AND under `none` no cell's
  success CI exceeds the Memoryless+none CI, AND the benefit-vs-param-count slope is
  reported (capacity gradient). Memoryless+e3b idle_frac → 0 with success 0 and
  falls > 0 is the registered "moves maximally, still fails" datum.

### P2 — GAE-λ horizon (E3; can falsify our own theory, and earns the Ni-2023 delta)
- **Rule:** both the `sparse-none` deficit AND the `e3b−none` gap **shrink as
  gae_lambda → 1** (GRU-only, λ ∈ {0.8, 0.9, 0.95, 0.99, 1.0}, n=3+). *If they do not
  shrink, P2 (GAE-horizon credit reach) is wrong* and the credit story is revised in
  the paper. This is the quantity Ni 2023 does not measure (reward-density × bonus ×
  credit-reach); the delta to Ni is *earned only if this confirms*.

### Effective RM size (P5)
- `eff_rm_size(penalty-none) → 1` (collapse to a trivial 1-state machine); `e3b`
  re-inflates it. Validated end-to-end on TinyReproduce / Autoencode where the true
  minimal RM is enumerable (`info["rm_state"]` ground truth), not a proxy.

---

## 4. Generality / 2nd-positive-env (registered scope guard)

"Bonus helps" must be shown on **a second, structurally different positive env** or
the empirical claim is explicitly scoped to a MysteryPath case study.
- **Battleship (E13):** confirm iff `e3b > none` on sparse (CI), `e3b ≈ none` on dense
  (TOST), Memoryless ≪ memory cells. **Report `expose_action_coords` on/off
  sensitivity**; if the effect requires the coord-augmented obs, the positive claim is
  scoped to coord-augmented observations (registered, not hidden).
- **Fallback (registered):** if no clean 2nd positive env lands by the deadline, the
  title/abstract are scoped to "a case study in MysteryPath," and generality is stated
  as a hypothesis — *not* asserted.

---

## 5. α-taxonomy — DESCRIPTIVE only (registered demotion)

With 4 signs over 4 envs there are **0 residual degrees of freedom**, so the taxonomy
is **not** a predictive theory and we do not use "predicts" language. It is reported
as a **descriptive organizing scheme** for the observed e3b−none signs. The α<0 (SS
hazard) leg is re-run with the **fixed** multi-head E3B (verify `e3b_idm_acc > chance`
is logged) — the prior α<0 was a frozen-random-embedding bug and the question is
re-opened, not assumed.

*Optional predictive upgrade (only if it lands):* measure `α_env` from a fixed
reference occupancy and predict a **held-out env's** e3b−none sign before unblinding
it. Only then may the taxonomy be called predictive. If it does not land, the
descriptive demotion above stands and no predictive claim is made.

---

## 6. Inclusion / exclusion + survivorship (locked)

- A run counts iff it reached **≥ 95 % of its step budget**. Crashed runs are
  reported but excluded from the headline statistic (their last-step values may appear
  only in a separately labelled "partial" panel).
- Crashed-name duplicates de-duplicated by wandb run id; every exclusion listed in an
  appendix table with seed, cell, arm, and crash step.
- **Survivorship check (registered):** test that the crash rate is **independent of
  arm** (e.g. χ² of crashed vs completed across density arms). Freezing arms run
  cheaper and could crash differently; if crash rate correlates with arm, the
  matched-step comparison is re-done at the min common step and the dependence is
  reported.

---

## 7. Registered "tests that can falsify our own theory"

We commit to reporting these in whichever direction they fall:
1. **PBIM dose-response** (§3 S2) — flat-in-β would refute the non-OP reading.
2. **λ-sweep** (§3 P2) — no shrink as λ→1 would refute the GAE-horizon credit story.
3. **lag-Δ retention** (§3 S1c) — e3b > none at long lag would refute "behavioral, not
   representational."
4. **−0.008 optimality** (§3 F2) — if the optimal fall rate p ≳ p̄, the penalty is not
   optimum-preserving and the "pathology" framing is withdrawn.
5. **k=1 collapse** (E9) and **SS α<0 re-run** (§5) — reported as they fall.
