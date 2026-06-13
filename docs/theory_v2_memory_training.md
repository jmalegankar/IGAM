# Theory v2 — Exploration bonuses as memory trainers (corrected)

*Supersedes `theory_memory_density.md` and §3–4a of `revelation_and_densification.md`
where they conflict. Drafted 2026-06-12 after the AAAI review
(`project_aaai_review_findings`). The earlier "exactness" lemma and Proposition
T1 ("zero direct credit") are **withdrawn as false**; what follows is what we can
actually prove or measure. Claim the **synthesis, the memory-training thesis, and
the experimental realization** — not the component theorems.*

---

## 0. Thesis (one sentence)

On sparse partially-observed memory tasks, an episodic novelty bonus's dominant
effect is **not exploration but the training of the recurrent memory**: it works
through a purely *behavioral* channel (densifying reward and reshaping the
trajectory distribution so the policy gradient can reach the memory writes),
which is why it (i) cannot substitute for memory, (ii) becomes redundant once
reward is dense, and (iii) rescues a memory that a faithful penalty has collapsed
to the empty statistic.

We *verified in code* (2026-06-12) that the bonus has **no architectural path**
into the policy's memory: `IDMPhi` owns a separate encoder (`phi_sources.py:130`,
trained by its own optimizer `e3b_module.py:154`). The only coupling is
behavioral. That fact is the thesis's backbone, not a caveat.

---

## 1. Setup

POMDP `(S, A, O, T, Z, r, γ)`, hidden state `s`, observation `o`. Recurrent
policy `π_θ(a_t | m_t)`, memory `m_t = f_θ(m_{t−1}, o_t)`. Trained by recurrent
PPO with GAE(γ, λ) and TBPTT chunk length `k`. Episode length `T`.

**D1 — write–use gap Δ (memory axis).** Info observable at write time `t_w` must
condition behavior at use time `t_u`; `Δ = t_u − t_w`. Memory-essential because
`o_{t_u}` is aliased (Singh–Jaakkola–Jordan 1994; Littman 1994).

**D2 — use–pay gap δ (density axis).** Distance from a use to the reward event
that credits it, signed: sparse terminal `δ = T − t_u`; aligned-dense `δ ≈ 0`
positive at uses; anti-dense `δ ≈ 0` negative at errors.

Our toggles manipulate δ and event-sign at fixed Δ. (They do **not** hold the
optimal policy exactly fixed — see P1.)

---

## 2. Corrected propositions

### P1 — Approximate optimum preservation (replaces the false "exact" lemma)

The old claim "the sparse-optimal π\* never falls ⇒ J_fall(π\*) = J_sparse(π\*) =
1.0" is **false**: MysteryPath regenerates an invisible maze every episode
(`mystery_path_grid.py:153–168`), so falling **is** the information-gathering
action and any realizable (history-conditioned) policy with above-chance success
has `E[falls] > 0`.

> **P1.** For a per-step error penalty of magnitude `p` on an event `e`,
> `J_p(π) = J_0(π) − p·E_π[#e]`, so for any π,
> `|J_p(π) − J_0(π)| ≤ p·E_π[#e]`. The optimum is preserved **approximately**,
> with bound `ε = p·E_{π\*}[#e]`. For MysteryPath falls at `p = 0.008` and an
> optimal prober's `E[falls] ≈ 4–10/ep`, `ε ≈ 0.03–0.16`. The penalty strictly
> shifts the optimal *in-episode probing rate* (optimal probe rate falls as `p`
> rises; standing still becomes optimal once `p·E[remaining falls]` exceeds the
> marginal success value — i.e. **freezing is rational for `p ≳ p̄ = S\*/F\* ≈
> 0.1–0.25`**).

Consequences:
- "Return-matched exactly" → "return-matched up to `ε`, measured."
- The **−0.1 freeze (n=15) is near-optimal play**, not a pathology — *only the
  −0.008 freeze (p ≪ p̄) is a genuine learning-dynamics collapse*. Lead with
  −0.008; present −0.1 as the "freeze is also rational here" regime.
- MortarMayhem's aligned arm *is* exactly return-matched (`4×0.25 = 1.0`,
  verified) — carry exact statements there, not on MysteryPath.

*Validation:* DP on a small fixed maze (belief-MDP) to compute `E[falls | π\*]`;
empirical fall-count of a converged sparse agent. (`memrl/probes/` + tiny env.)

### P2 — Credit to memory writes is a GAE-horizon phenomenon (replaces T1)

The old T1 ("sparse ⇒ zero direct credit, ever; `⌈Δ/k⌉` bootstrap hops") is
**false for our algorithm**: GAE runs over the full rollout *before* chunking
(`ppo.py:385`), so the advantage at `t_w` already contains the terminal reward
with weight `(γλ)^{T−1−t_w}` in one update, independent of `k`.

> **P2.** Decompose the gradient credit to the write computation at `t_w`:
> 1. **Advantage channel (k-independent):** the within-chunk PPO term
>    `Â_{t_w}∇log π` carries the terminal reward attenuated by `(γλ)^{Δ}` and
>    scaled by event rarity `∝ 1/p(success)`. With γλ = 0.945 and `Δ ≈ 10–20`,
>    this is 5–50% — small and high-variance, **not zero**.
> 2. **Pathwise-BPTT channel (k-dependent):** the gradient
>    `∂(loss at t > t_w+k)/∂f_θ(write)` is **exactly zero** — truncation severs
>    the parameter gradient through the carried (detached) state. This is the
>    only thing TBPTT(k) actually cuts.
>
> The sparse-vs-dense gap is therefore **quantitative** (the `(γλ)^Δ ×
> 1/p(success)` factor on channel 1), governed by **λ and p(success)** — *not* by
> chunk-hopping. The real, tunable horizon is `1/(1−γλ) ≈ 18` steps.

Consequences:
- **The mechanism test is a λ-sweep, not (only) a k-sweep.** Predict: the
  sparse-none deficit and the e3b benefit both shrink as λ → 1 (channel 1
  reaches further).
- The k-sweep still probes channel 2, but its prediction is **non-monotone**:
  the e3b−none gap should **peak at intermediate k and collapse at k = 1** (where
  channel 2 is dead for *both* arms, so neither can learn the write via BPTT).
  Run k-aligned to episodes, with a staleness control, and report
  `n_updates`/`grad_norm` per k. Exempt GTrXL (parameter-free cache ⇒ no channel-2
  bottleneck).

### P3 — The bonus trains memory through the behavioral channel (the thesis)

Because the IDM encoder is architecturally separate (verified), the bonus reaches
the policy's memory only via `bonus → Â → ∇log π → trajectory distribution`. That
single channel carries two effects:

> **P3a (densification).** The bonus manufactures dense reward events
> (`δ_eff ≈ 0`) that let the critic learn a pre-terminal value, supplying
> channel-1 advantage where sparse reward gives almost none. Potential-based by
> nature (does not move π\*). Isolated by **PBIM-e3b** (`exploration/pbim.py`).
>
> **P3b (distribution-shift / curriculum).** Chasing *episodic* novelty requires
> the policy to track where it has been within the episode — collecting the bonus
> is itself a memory-exercising objective. This is the non-potential residual
> (the "ε" that moves π and that Skalse 2022 calls hackable).

Adjudication (the load-bearing ablation): **PBIM-e3b ≈ raw-e3b ⇒ P3a dominates
(densification); PBIM-e3b < raw-e3b ⇒ P3b is load-bearing (curriculum).** Either
is a result. The *outcome* of P3 (memory becomes trained) is measured directly by
the decodability probe (`memrl/probes/decode_memory.py`); the *channel* is pinned
by PBIM and by the entangled aux-loss arm (causal confirmation + practical recipe).

Corollaries (all already supported or cheaply testable):
- **Entanglement:** the bonus needs a memory to train ⇒ Memoryless+bonus fails
  (verified: 20M Memoryless-e3b = 0.000); under no-bonus, no cell beats the
  Memoryless floor.
- **Redundancy under density:** when reward is already dense, the densification
  term vanishes and only the bias remains ⇒ e3b ≈ none on aligned-dense.
- **Capacity gradient:** weaker cells (GRU/LSTM) have more un-trained write
  capacity to gain ⇒ larger bonus benefit than strong cells (RetNet/GDN/Mamba2),
  measurable as benefit-vs-capacity slope.

### P4 — Freeze is a degenerate fixed point of reward-shaped representation learning

> **P4.** Under an anti-aligned penalty whose penalized event is the task's
> revelation channel, with a reachable **absorbing zero-penalty set** ("passive
> sanctuary"), the joint (policy, representation) gradient has a trivial fixed
> point: the policy converges to the sanctuary and the memory to the **empty
> sufficient statistic** (nothing needs to be remembered to do nothing). The
> bonus, by paying for information at the sanctuary, removes the absorbing set
> from the agent's incentive landscape and restores a non-trivial fixed point
> (rescue).

This is the memory-representation instantiation of BAMDP shaping's "bonuses fix
information-starved exploration." The **sanctuary** is the precise condition;
MysteryPath has one (stand still on the start tile, verified zero-cost),
SearingSpotlights does not (spotlights wander). The clean test is a **within-env
sanctuary toggle** (add/remove a zero-cost stay action), which separates
"sanctuary" from "penalty information-content" — confounded in the cross-env
MysteryPath-vs-SS comparison.

*Measured directly:* the decodability probe should show penalty-none memory FLAT
at chance (empty statistic) and penalty-e3b recovering — the freeze and rescue
seen **in the representation**, unifying them with the sparse 3× as one phenomenon.

### P5 — The reward machine is the minimal memory the reward demands

*Theoretical anchor (added 2026-06-12). Reframes the "sufficient statistic" in
T2/P4 as a canonical automaton, and gives the decodability probe a named target.*

A **reward machine** (Toro Icarte et al. 2018) is the finite-state transducer of a
history-dependent reward; its minimal form is the **Myhill–Nerode automaton of the
reward function** — states = equivalence classes of histories indistinguishable in
*all future reward*. So the minimal reward machine is exactly **the minimal memory
the reward function requires**: an optimal recurrent memory `m_t` converges, in the
limit, to a sufficient statistic **no coarser than (and possibly finer than)** the
minimal-RM state of the reward it is trained on (see the stability note below — a
shaping-robust representation may keep extra reward-relevant distinctions). The
decodability probe's latent (the confirmed path/off-path knowledge
grid) *is* the MysteryPath minimal-RM state; the probe measures how close `m_t` is
to it.

Crucially, the operative reward is the **realized** reward (what the agent receives
following its own behavior), not the **specified** reward. This sharpens P4:

> **P5 (freeze = collapse to the trivial reward machine).** Under penalty + a
> passive sanctuary, the *realized* reward function along the converged policy has
> a **degenerate minimal reward machine** (≈ a single state: "do nothing →
> constant reward"). The memory correctly converges to that one-state machine —
> the freeze is the agent learning the minimal RM of the *realized* reward, not a
> failure to learn the *specified* task. The bonus alters the realized reward so
> its minimal RM is non-trivial again (it must track novelty/position), forcing
> the memory to re-grow — **the bonus re-inflates the reward machine.**

**The duality (states it cleanly).** Two ways a representation departs from the
minimal RM, around the same object:
- **Over-refinement** — the representation is *finer* than the minimal RM (carries
  reward-irrelevant distinctions). Potential-based shaping over those spurious
  states can inject reward differences between behaviorally-identical histories.
  A non-minimal RM **admits** such a destabilizing potential — but, correcting an
  earlier overclaim, **minimality is *sufficient but not necessary* for shaping-
  stability**: the smallest shaping-stable perfect RM can be *strictly larger* than
  the minimal one. (Minimality optimizes a **Mealy** machine — reward on
  transitions; a potential Φ is a **Moore** object — one value per state, and a
  consistent PBRS must telescope `γΦ(u′)−Φ(u)=Δ(canonical reward)` on every edge.
  When the Mealy-minimal machine merges two states reached by edges of *different*
  reward, no Φ satisfies both edges, so stability *requires splitting* — adding
  states removes Φ-conflicts. Hence `minimal-RM ≤ smallest-stable-RM ≤ over-refined`,
  strict in the Mealy-vs-Moore gap; cf. Toro Icarte et al. JAIR 2022 §"Mealy can
  need fewer states." **Background, not a contribution** — a formal treatment, with
  a witness reward function *in that gap*, belongs in a separate paper.)
- **Under-training** — the representation is *coarser* than the minimal RM
  (un-trained writes under sparse credit). The freeze is collapse all the way to
  the trivial 1-state machine. This is *our* regime: too *little* trained memory,
  which the bonus fixes. PBRS telescoping (Ng et al. 1999) immunizes the
  over-refinement direction at the *return* level, so our results live cleanly on
  this under-trained pole.

**Methodological caveat for PBIM (where the over-refinement remark earns its
keep).** PBRS over an *over-refined* statistic — one carrying distinctions finer
than the realized reward's equivalence classes — manufactures spurious per-step
shaping differences between behaviorally-identical histories. The right safeguard
is **not** "use the minimal statistic" (minimality is neither required nor
certifiable) but **"fit the potential to a future-reward-*sufficient* statistic,"**
whose every induced distinction is reward-relevant. PBIM fits Φ to the intrinsic
*value* `V_int` — sufficient by construction — **not** raw φ-features, so the
densification arm cannot smuggle in shaping-instability. This is **testable**:
the `raw-φ-PBIM` vs `V_int-PBIM` control (optional, E1/E2) should show raw-φ-PBIM
shifting the success-defined optimum vs `none` (beyond the P1 return-matched bound)
while `V_int-PBIM` does not — demonstrating the safeguard rather than asserting it.

*Measured for free:* the decodability probe also emits an **effective reward-machine
size** = `2^{decodable bits of the minimal-RM state}` (and a clustering cross-check).
Predicted: penalty-none collapses toward **1 state** (freeze), sparse-e3b and
penalty-e3b grow it. This is the freeze/rescue/3× expressed as one scalar.

---

## 3. Positioning (engage, don't reinvent)

- **"Revealing POMDP" is a named formalism** (Jin et al. 2020 single-step; Liu et
  al. 2022 weakly-revealing + OMLE; Chen–Wang 2023 lower bounds): observations
  leak information about latent state, enabling sample-efficient learning. Our
  "revelation"/"sealed" axis is the goal-posterior specialization. **Adopt the
  term and cite it**, or rename — do not present it as new. Opportunity: their
  rank/identifiability conditions could formalize "revelation" rigorously.
- **BAMDP Shaping** (Lidayan–Dennis–Russell, ICLR 2025) is the abstract parent:
  all pseudo-rewards = shaping in a Bayes-Adaptive MDP; value = value-of-info +
  physical-state value; potential bonuses resist hacking; treats noisy-TV. **Our
  delta:** the *memory* axis they lack (credit to recurrent writes), controlled
  density toggles on memory benchmarks, and the freeze fixed point as an emergent
  BAMDP phenomenon. Frame the paper as the empirical, memory-specific
  instantiation + a new fixed point.
- **Ni et al. 2023** ("When Do Transformers Shine in RL? Decoupling Memory from
  Credit Assignment") owns the memory⊥credit-assignment split. **Our delta:** P2
  is the *reward-density-dependent* refinement — density decides whether credit
  reaches the write — and we use the *bonus* as the intervention.
- **Henaff et al. ICML 2023** (global vs episodic bonuses, contextual MDPs):
  closest "find the axis that governs when bonuses help." Relate explicitly;
  ours is the memory-training mechanism beneath their episodic-helps result.
- **Reward machines** (Toro Icarte et al. ICML 2018; + Myhill–Nerode minimality):
  the minimal RM as "the minimal memory the reward demands" anchors P5 and the
  probe's target. The shaping-stability⟺minimality remark (the over-refinement
  dual) is background, not a claimed result. A standalone formal treatment of
  shaping-stability (algorithm + RM-learning connection + bound) is a *separate*
  paper, not part of this one.

---

## 4. Experiment → claim map

| Claim | Experiment | Status |
|---|---|---|
| P1 bound on optimum shift | DP on tiny maze + converged-agent fall count | to build (tiny env) |
| P2 GAE-horizon mechanism | **λ-sweep** {0.8…1.0}; corrected k-sweep (2nd) | to run (cheap) |
| P3 outcome: memory trained | **decodability probe** none/e3b/penalty over steps | harness built |
| P5 freeze = trivial RM; rescue re-inflates | **effective-RM-size** readout (same probe) | harness built |
| P3 channel: densify vs curriculum | **PBIM-e3b** vs raw-e3b | impl built |
| P3 causal confirmation + recipe | entangled IDM-aux-loss arm | to build (secondary) |
| P3 entanglement | Memoryless + capacity gradient (5 cells) | have Memoryless; add cells |
| P3 redundancy under density | MortarMayhem aligned (return-matched) | seed-0; need 5 seeds |
| P4 freeze fixed point | fallpenalty (have, n=5) + decodability | have phenomenon |
| P4 sanctuary condition | **within-env sanctuary toggle** | to build |
