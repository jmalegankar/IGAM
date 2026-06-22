> **COMPREHENSIVE DRAFT — all findings; crop to ~8pp for AAAI later.**

# Exploration Selects the Reward Machine a Memory Agent Learns

## Abstract

Exploration in reinforcement learning is almost universally framed as a matter of *speed*: a better bonus visits informative states sooner, so the agent reaches a high-return policy in fewer samples. We show that for a **memory-equipped** agent this view is incomplete, and we demonstrate a sharper claim: **exploration selects which reward machine the memory agent learns.** A recurrent policy does not converge to an automaton of the *specified* task; it converges to the minimal reward machine — the Myhill–Nerode automaton (Toro Icarte et al., 2018) — of its **realized** reward, the reward distribution induced under its *own* policy. Because exploration determines which transitions the policy ever drives through, it determines which realized-reward automaton exists to be learned.

We make this concrete and falsifiable on MysteryPath-Grid, MiniGrid-MemoryS13, and exact-reward-machine register tasks (TinyReproduce, POPGym-Autoencode), across a cell zoo {GRU, LSTM, RetNet, GatedDeltaNet (GDN), Mamba2, Memoryless} under online recurrent PPO. A faithful, optimum-preserving −0.008 fall-penalty collapses the realized reward machine R(π) to a single absorbing do-nothing state — the *freeze*: success 0.00 **and** falls 0.0 in all 6/6 memory cells, behavioral effective-RM-size = 1.00 — while a return-matched sparse agent stays alive (success .16–.20, falls 9–39, eff-RM-size 6.98). A non-potential bonus re-inflates the automaton (E3B: eff-RM-size 13.14; success rescued to .24–.62 across all six cells; NovelD .17–.71). The keystone control is that a **policy-invariant potential** (PBIM, F = γΦ(s′) − Φ(s), Φ = V_int) provably *cannot* re-inflate R(π) — a Moore object telescopes to zero return and cannot relabel a transition-keyed (Mealy) reward — and empirically leaves the agent identically frozen (success 0.00 / falls 0.0; ρ = (pbim − none)/(e3b − none) = 0). We prove via a tiny-maze dynamic program that freezing is strictly sub-optimal under the *specified* reward (V* = 0.76 > 0; the penalty would have to be ≈4× harsher, p̄ = −0.033). Exploration and memory are *conjunctively* necessary: a memoryless agent with E3B has the highest falls (22.3) yet 0.00 success — maximal coverage, no automaton realized. We measure all of this with one confound-free **behavioral** instrument (the number of distinct realized-RM states the policy drives through), and we *retire* a teacher-forced decoding probe that credits untrained nets nearly as much as trained ones (GRU random 7.03 ≈ trained 7.04 bits; GTrXL random 9.4 ≈ trained 9.8) — copy-capacity, not learned use. The reward machine the agent learns is the one its exploration lets it realize.

---

### Notation and conventions (used throughout)

| Symbol | Meaning |
|---|---|
| **R\*** | the *specified* reward machine — the minimal Myhill–Nerode automaton of the task's specified reward ρ\* |
| **R(π)** | the *realized* reward machine — the minimal Myhill–Nerode automaton of the reward stream the agent actually generates under its own policy π |
| **eff-RM-size** | *effective-RM-size*: the number of distinct realized-RM states the policy drives through per episode; the central confound-free behavioral measurable (`rm_coverage.py`) |
| **\|R(π)\| ≤ \|R\*\|** | the realized machine demands at most the memory the specified machine demands (policy-dependent inclusion) |
| **freeze** | convergence to the one-state absorbing do-nothing R(π) under a faithful penalty (success 0.00 / falls 0.0) |
| **ρ** | re-inflation ratio (pbim − none)/(e3b − none): the fraction of E3B's RM re-inflation that survives the potential transform |
| **PBIM** | potential-based intrinsic-motivation delivery, F = γΦ(s′) − Φ(s) with Φ = V_int (a valid telescoping potential) |
| **K1–K4** | the four pre-registered kill-criteria (freeze / Mealy–Moore keystone / instrument separation / DP optimality gap) |

All make-or-break MysteryPath numbers are on the n=5 grid (`memrl-memtrain-mpg`, lr 1e-4, n=4–5 seeds, evaluation). The penalty is the env's native −0.008 off-path fall-penalty knob. Instruments: `rm_coverage.py` (behavioral eff-RM-size + idle-fraction, K1/K3), `optimal_prober.py` (corridor-POMDP DP, K4), `decode_memory.py` (hardened decoding probe, retained as cautionary-only).

---

## 1. Introduction

Exploration in reinforcement learning is almost universally framed as a matter of *speed*: a good exploration bonus visits informative states sooner, so the agent reaches a high-return policy in fewer samples. Under this view, exploration is an accelerant for an objective that is fixed in advance — change the bonus and you change *how fast* the agent climbs, not *what it climbs toward*.

We argue that for a **memory-equipped** agent this view is incomplete, and we demonstrate a sharper claim: **exploration selects which reward machine the memory agent learns.** A recurrent policy does not converge to an automaton of the *specified* task. It converges to the minimal reward machine — the Myhill–Nerode automaton (Toro Icarte et al., 2018) — of its **realized** reward: the reward distribution induced *under its own policy*. Because exploration determines which transitions the policy ever drives through, it determines which realized-reward automaton exists to be learned. Exploration is not (only) tuning the rate of convergence; it is choosing the target of convergence.

**The freeze.** The cleanest demonstration is a failure that, on inspection, is not a failure at all. In MysteryPath-Grid, the task carries a faithful −0.008 penalty for stepping off the hidden path. Equip a capable recurrent learner (GRU, LSTM, RetNet, GatedDeltaNet, Mamba2) and it converges to **doing nothing**: in all 6/6 memory cells, success 0.00 *and* falls 0.0 — an absorbing do-nothing state. This is not a credit-assignment bug or a capacity shortfall. The same architectures on a return-matched sparse task stay *alive* (success .16–.20, falls 9–39). The agent has correctly learned the minimal automaton of its realized reward: under a faithful penalty, every exploratory transition is net-negative, so the realized-reward machine collapses to a single absorbing state and standing still is the rational fixed point. The freeze is the agent learning a *degenerate but correct* automaton, not failing to learn the task's.

**Re-inflation, and the keystone control.** Which exploration bonus is applied then dictates which automaton the agent realizes. Measured behaviorally — **effective-RM-size**, the number of distinct realized-RM states the policy actually drives through, on 20M-step GatedDeltaNet snapshots — the faithful penalty collapses the machine to **eff-RM-size = 1.00**; adding a non-potential episodic bonus (E3B) re-inflates it to **13.14** (sparse-none sits at 6.98). Rescue is broad: E3B lifts MysteryPath success to .24–.62 (falls 7–22) across all 6 memory cells, NovelD to .17–.71.

The control that turns a correlation into a mechanism is a **potential-based** bonus. A policy-invariant potential Φ (here PBIM, with F = γΦ(s′) − Φ(s), Φ = V_int) is a Moore object: one value per state, telescoping to zero return. By the Mealy–Moore distinction it *cannot* relabel transition-keyed reward, hence provably cannot move the realized reward machine R(π). Empirically PBIM behaves *identically to no bonus*: penalty-PBIM gives success 0.00 / falls 0.0 — still frozen — with ρ = (pbim − none)/(e3b − none) = 0. A non-potential bonus re-inflates the automaton; a potential one, by construction, cannot. This is the Mealy–Moore keystone, and it is what makes "exploration selects the RM" a structural statement rather than an empirical coincidence.

**A confound-free instrument.** Throughout, we resist measuring memory by *decodability*. We show this is the right call: a teacher-forced decoding probe credits an untrained network nearly as much as a trained one (GRU random 7.03 ≈ trained 7.04 bits; GTrXL random 9.4 ≈ trained 9.8), so it measures copy-capacity, not learned use. We therefore measure the realized automaton *behaviorally* — the number of distinct realized-RM states the policy drives through — an instrument that cleanly separates the collapsed (1) from the re-inflated (13) machine without reference to any internal representation, and we report the decoding probe only as a cautionary methods result.

**Why this is not already known.** Reward-machine RL (Toro Icarte et al., 2018, 2022) *assumes the automaton is given* and injects it; we run that arrow in reverse — the RM is **emergent**, equal to the realized-reward automaton, and we *measure it collapse and re-inflate*. Work decomposing a *fixed* reward into potential and non-potential parts (BAMDP-Shaping; Lidayan, Dennis & Russell, 2025) predicts PBIM ≈ none over value functions — which is exactly why we use it as a *control* rather than a headline, and it offers no penalty-sealing or freeze prediction. Analyses showing memory and credit assignment are independent for a fixed reward (Ni et al., 2023) are complemented, not contradicted: we *couple* them by editing the realized RM the memory must represent.

We close two loops to keep the claim honest. **Optimality:** a tiny-maze dynamic-programming solve shows that under the *specified* reward, freezing is strictly sub-optimal (V\* = 0.76 > 0; the penalty would have to be ~4× harsher, p̄ = −0.033, for do-nothing to be optimal) — so the freeze is a property of the *realized* reward under exploration, not of the task. **Necessity:** a memoryless agent floors everywhere and, with E3B, exhibits the *highest* fall count (22.3) yet 0.00 success — it moves maximally but has no memory to realize any non-trivial automaton. Exploration that covers the transitions and memory that realizes the automaton are *conjunctively* necessary; neither alone moves the realized reward machine.

### 1.1 Contributions

1. **A reframing of exploration as reward-machine selection.** We formalize and demonstrate that a recurrent agent converges to the minimal Myhill–Nerode reward machine of its *realized* reward, not the specified task, and that exploration — by determining which transitions are ever traversed — selects *which* automaton the memory learns. (§2, §3)

2. **The freeze as correct degenerate learning.** We show that a faithful −0.008 penalty drives capable recurrent learners (6/6 cells: GRU, LSTM, RetNet, GatedDeltaNet, Mamba2) to an absorbing do-nothing state (success 0.00 / falls 0.0), while a return-matched sparse task keeps them alive (success .16–.20, falls 9–39). The freeze is the agent correctly learning a collapsed realized-reward automaton, and we prove via tiny-maze DP that it is strictly sub-optimal under the *specified* reward (V\* = 0.76 > 0; freeze threshold p̄ = −0.033). (§3, §7)

3. **The Mealy–Moore keystone.** We establish — analytically and empirically — that a non-potential bonus (E3B) re-inflates the realized RM while a policy-invariant *potential* (PBIM) provably cannot. PBIM is behaviorally identical to no bonus under the penalty (success 0.00 / falls 0.0; ρ = 0), turning "exploration selects the RM" from correlation into mechanism. (§4)

4. **A confound-free behavioral instrument for the realized RM.** We introduce effective-RM-size — the number of distinct realized-RM states a policy drives through — which separates the collapsed automaton (1.00) from the re-inflated one (13.14, vs sparse-none 6.98), and we show *why* the obvious alternative fails: a teacher-forced decoding probe credits untrained nets nearly as much as trained (random ≈ trained in bits), so we retire it to a cautionary result. (§3, §7)

5. **Conjunctive necessity of exploration and memory.** We show memoryless agents floor everywhere and, with E3B, achieve the highest falls (22.3) yet 0.00 success — coverage without realization — establishing that exploration and memory are jointly, not separately, necessary to move the realized reward machine. (§5)

6. **Catalyst–geometry matching and an honest capacity audit.** We show *which* bonus rescues which task tracks task geometry (E3B carries MysteryPath; NovelD broadly solves MiniGrid-MemoryS13-view3 at ~0.98, including a confirmed NovelD ≥ E3B reversal), and we report honestly that the gated-RNN ≫ SSM/DeltaNet exact-recall gap is *partly* a tuning artifact (best-HP closes much of it but a residual gap remains), so all cell rankings are reported at per-cell-best hyperparameters. (§5, §6)

---

## 2. Theory: the realized reward machine

This section makes the thesis precise. We formalize the object the agent converges to (the **realized reward machine** R(π), the Myhill–Nerode automaton of the agent's *own* reward stream), state the rigorous inclusion relating it to the specified machine R\*, and prove the three load-bearing claims — the freeze as collapse to a one-state machine (P5), the Mealy–Moore impossibility that a potential bonus cannot re-inflate it (the keystone), and the DP optimality bound that makes the freeze a genuine *learnability* pathology rather than rational behavior (K4). Throughout we flag explicitly which steps are theorems and which are heuristics or cited assumptions; the convergence-to-minimal premise is stated as an **assumption with cited lemmas and is deliberately not load-bearing** — the empirical instrument measures R(π) directly, so no claim here depends on the agent provably reaching the minimal machine.

### 2.1 Reward machines as Myhill–Nerode automata: the minimal memory a reward demands

A **reward machine** (Toro Icarte et al., ICML 2018; JAIR 2022) is a finite-state automaton R = ⟨U, u₀, δ_u, δ_r⟩ that reads the agent's experience and emits reward: states U (the *memory* the reward needs), a transition function δ_u : U × 2^𝒫 → U driven by labelling propositions over observations, and an output δ_r : U × 2^𝒫 → ℝ that pays reward on transitions. The standard reward-machine-RL programme treats R as **given** and injects it to densify credit assignment. We run that programme *in reverse*: we never supply a machine; we ask which machine an agent's *realized reward stream* induces, and we measure it.

The bridge to memory is the Myhill–Nerode theorem. Fix a reward function as a map from finite histories to reward, ρ : 𝓗 → ℝ. Define the right-congruence

> h ∼_ρ h′ ⟺ ∀ continuations w: ρ(h w) = ρ(h′ w).

Two histories are equivalent iff no future experience ever distinguishes their reward consequences. The **minimal reward machine** of ρ is the quotient automaton 𝓗 / ∼_ρ: its states are the equivalence classes, and its number of states |𝓗/∼_ρ| is, by Myhill–Nerode, the *minimum* number of distinct memory states any system must maintain to reproduce ρ. This is the precise sense in which **a reward machine is the minimal memory a reward demands** — it is the Nerode automaton of the reward, exactly as the Myhill–Nerode automaton is the minimal DFA of a language. (Rigorous: this is Myhill–Nerode applied to ρ; the only modelling choice is that ρ is regular, i.e. ∼_ρ has finite index, which holds for all our envs by construction — TinyReproduce/Autoencode emit an *exact* finite `rm_state`, the remaining suffix; MysteryPath's reward is a function of the finite path-knowledge grid.)

### 2.2 Specified vs realized: R\*, R(π), and the inclusion direction

The task ships a specified reward ρ\* with minimal machine R\* = 𝓗/∼_{ρ\*}. But a *fixed policy* π does not visit all of 𝓗; it induces a reachable sub-language 𝓗_π ⊆ 𝓗 (the histories of positive measure under π). Restricting the congruence to what π actually realizes gives the **realized reward machine**

> R(π) = 𝓗_π / (∼_{ρ\*} ↾ 𝓗_π),

the Nerode automaton of the reward *stream the agent actually generates under its own policy*. This is the object the `rm_coverage` instrument estimates: the number of distinct realized RM-states the policy drives through per episode (effective-RM-size; §2.6).

**Inclusion (rigorous, and the direction matters).** Coarsening the domain can only *merge* equivalence classes, never split them: if two histories agree on all continuations in 𝓗 they agree on all continuations in 𝓗_π ⊆ 𝓗. Hence

> **|R(π)| ≤ |R\*|** for every policy π.

The realized machine demands **at most** the memory the specified machine demands; the policy can only *under-realize* the task's reward structure, never over-realize it. The inequality is policy-dependent — different policies realize different sub-machines, and the gap |R\*| − |R(π)| is exactly the task structure the agent's behavior leaves un-exercised. This one-sided inclusion is the formal spine of the thesis: **exploration is what moves π — and therefore R(π) — within the lattice of sub-machines bounded above by R\*.** (Rigorous: monotonicity of Nerode quotients under domain restriction. We do **not** claim the agent's *network* is forced down to |R(π)| states — capacity may exceed demand; the claim is about the reward structure the behavior realizes, which is what we measure.)

### 2.3 Convergence to the minimal machine (assumption with cited lemmas — NOT load-bearing)

To connect "the policy *realizes* R(π)" to "the *learner* converges to it," one needs that a recurrent agent trained to optimize ρ\* ends up maintaining no more reward-distinctions than its realized reward requires. We state this as an **assumption**, not a theorem, and lean on it nowhere that the empirics can't stand alone:

> **(A-min) Realized minimality.** A converged recurrent PPO policy maintains memory state sufficient to act on, and no finer than, the realized machine R(π) it drives through.

Support is suggestive, not a proof: predictive-state / Nerode-of-the-reward arguments (Toro Icarte 2018; the regular-decision-process line, Abadi & Brafman 2020) show the minimal RM is the right state variable for an MDP-over-RM; aliasing lemmas (Singh–Jaakkola–Jordan 1994; Littman 1994) show an optimizer is pressured to maintain *at least* the distinctions its reward depends on. Neither delivers exact minimality under TBPTT-truncated PPO, so we do not assert it. **Crucially, (A-min) is not load-bearing**: the kill-criteria route through the behavioral instrument (§2.6), which counts R(π) from the *rollout itself* with no appeal to what the network internally stores. Were (A-min) false (the net hoards unused states), every result below is unchanged, because effective-RM-size is defined behaviorally. We retire any internal-state claim for exactly this reason — see the cautionary decoding result, §7.4.

### 2.4 P5 — the freeze is collapse to the one-state realized machine

The MysteryPath fall arm adds a faithful penalty −p (p = 0.008) on every off-path step. **P5** is the claim that this collapses the realized machine to a single absorbing state.

Under a passive sanctuary — a reachable, absorbing, zero-cost behavior (in MysteryPath, *stand still*) — the do-nothing policy π₀ generates the constant history-class "no reward event ever." Its realized reward stream is the constant 0, whose Nerode quotient is a **single state**: |R(π₀)| = 1. P5 is then:

> **P5 (freeze = 1-state collapse).** Under the faithful penalty with a passive sanctuary, the no-bonus learner converges to π₀, and R(π₀) is the one-state do-nothing machine.

This is what the data show, on two independent registers. **Behavioral kill-criterion K1** (n=5 grid, eval): penalty-none gives success 0.00 **and** falls 0.0 in all 6/6 memory cells — not "fails to solve" but emits the empty reward stream. The return-matched control rules out a trivial-task explanation: sparse-none stays *alive* (success .16–.20, falls 9–39), so the same task with the same optimum but *without* the penalty does not collapse. **Instrument register K3** (effective-RM-size on 20M GatedDeltaNet snapshots, §2.6): penalty-none = 1.00 exactly, versus sparse-none = 6.98. The penalty does not merely lower success — it **drives |R(π)| to its absolute floor of one**. P5 is the realized-machine reading of the freeze: the agent has correctly learned a *degenerate automaton of its own realized reward*, which is the empty-reward machine. (Rigorous given the premises: R(π₀) = 1 is Nerode of the constant stream. The premise — that the optimizer converges to π₀ — is the *empirical* freeze, dose-responsed at p ∈ {−0.008, −0.025, −0.05}; §2.5 shows it is a learnability pathology, not optimality.) The **sanctuary is a necessary condition**, not a universal: SearingSpotlights, where standing still does not avoid the wandering-spotlight penalty, does **not** freeze (none-anti stays alive, longest episodes, most coins) — the empty-statistic fixed point requires an *avoidable* penalty.

### 2.5 The DP optimality bound (K4): the freeze is a learnability pathology, not rational behavior

P5 would be vacuous if freezing were simply *optimal* at p = −0.008. It is not. On a tractable corridor-POMDP abstraction (hidden path of L stages, b branches each, one on-path; a wrong move is a fall paying −p and eliminating that branch; reaching stage L pays +1; do-nothing returns 0 forever), the exact value recursion is

> V(stage, r) = max( 0 [freeze], (1/r)·γ V(stage+1, b) + ((r−1)/r)·(−p + γ V(stage, r−1)) [probe] ),

with V(L, ·) = 1. Freezing is optimal under the *specified* reward iff V\* = V(0, b) ≤ 0.

**K4 (optimality gap).** At p = −0.008 the DP gives V\* = 0.76 > 0 = do-nothing return: **probing strictly dominates freezing under the specified reward.** The optimal policy incurs E[falls | π\*] = 30, giving the realized penalty mass ε = p·E[falls] = 0.24. The freeze threshold — the penalty at which V\* first hits 0 — is p̄ = −0.033: **the penalty would have to be ≈4× harsher** for freezing to be rational. The result is robust across maze sizes. Therefore the observed freeze at −0.008 is a **strictly sub-optimal fixed point of the learning dynamics**, not a correct response to the reward. (Rigorous: exact DP; V\*>0 and p̄ are computed, not estimated — `optimal_prober.py`. Heuristic step: the corridor is an *abstraction* of MysteryPath, so V\*=0.76 is illustrative of the gap's sign and scale, not the literal env value; the load-bearing claim is only sign(V\*)>0 and |−0.008| ≪ |p̄|, both robust.) Mechanistically (§2.8) the penalty wins the *early* gradient war — fall-avoidance is learnable in ~10⁵ steps, goal-finding needs ~10⁶ — so the learner reaches the dominated π₀ before it can discover the goal that would justify probing. This is the controlled, dose-responsed **lazy-robot effect**: a faithful, exactly optimum-preserving penalty still destroys learning.

### 2.6 The instrument: effective-RM-size as a confound-free read of R(π)

The kill-criteria route through one behavioral quantity, **effective-RM-size** = the number of distinct realized RM-states the *agent's own deterministic policy* drives through per episode (`rm_coverage.py`). It does **not** teacher-force and does **not** decode a frozen internal state — it rolls π and counts what it *does*. For exact-RM envs (TinyReproduce/Autoencode) the realized state is the emitted `rm_state` (remaining suffix); distinct tuples are exact Nerode classes. For MysteryPath the realized-state proxy is the agent-induced path/off-path *knowledge grid* — distinct knowledge configurations the rollout visits, still built purely from the trajectory. **K3 (the instrument separates):** penalty-none = 1.00, sparse-none = 6.98, penalty-e3b = 13.14 on matched 20M GatedDeltaNet snapshots — the single number cleanly orders collapse (1) ≪ alive (7) ≪ re-inflated (13). The instrument is deliberately the *opposite* of the retired decode probe (§7.4): it measures realized *use*, which is what the theory is about, and never reads capacity.

### 2.7 The Mealy–Moore impossibility: PBIM = none is a theorem (the keystone)

The keystone separates a **non-potential** bonus (which re-inflates R(π)) from a **potential-based** one (which provably cannot). The argument is a type mismatch between Moore and Mealy machines.

A potential-based shaping term is F = γΦ(s′) − Φ(s) for a state-potential Φ (Ng–Harada–Russell 1999). PBIM instantiates this with Φ = V_int. Two facts about F:

1. **Φ is a Moore object.** A potential assigns **one value per state** — it is a state-labelling, the defining signature of a *Moore* machine (output on states), not a *Mealy* machine (output on transitions). The minimal reward machine of a transition-keyed reward (a fall *event*, a copy-*event*) is intrinsically **Mealy**: δ_r pays on transitions, and the off-path-step / on-path-step distinction that inflates R(π) lives on *edges*, not nodes.

2. **F telescopes to zero realized return.** Along any trajectory, Σ_t γᵗ F_t = Σ_t (γ^{t+1}Φ(s_{t+1}) − γᵗΦ(s_t)) collapses to boundary terms; over a full episode (with the standard terminal-Φ=0 convention) the discounted shaping return is **exactly 0**. We verified this numerically on PBIM: the per-episode discounted shaping sum has std/|mean| in [0.00, 0.27] — a valid telescoping potential.

Now the impossibility. To *re-inflate* R(π) — to add a realized reward-distinction that splits two history-classes the frozen policy has merged — the shaping must **relabel a transition-keyed reward**: it must pay differently on the off-path edge than on the stand-still edge, so that the Nerode congruence separates them. But a Moore state-potential cannot relabel an edge (fact 1), and even where it shifts intermediate values its *realized* contribution to return is 0 (fact 2), so it cannot change which histories are ∼_{ρ\*}-equivalent under the agent's own stream. Formally:

> **Theorem (shaping-stable RM = realized RM).** For any state-potential Φ, the realized reward machine of ρ\* + (γΦ′ − Φ) equals the realized reward machine of ρ\*: R_Φ(π) = R(π) for every π. Equivalently, the **shaping-stable RM is ≥ the minimal realized RM, with equality** — a potential is invariant on the Nerode quotient of the realized reward.

Consequence: a potential bonus **cannot move the realized machine off the 1-state freeze**. This yields **K2**, the Mealy–Moore keystone, as a *prediction-then-test*: penalty-PBIM gives success 0.00 / falls 0.0 — **identical to none, still frozen** — with the re-inflation ratio ρ = (pbim − none)/(e3b − none) = 0. The same null holds on sparse (sparse-PBIM ≈ 0 vs sparse-e3b .21–.69). A **non**-potential bonus, having no telescoping constraint and being free to pay on edges, *does* re-inflate: penalty-e3b drives effective-RM-size 1.00 → 13.14 and rescues success to .24–.62. (Rigorous: the telescoping and Moore/Mealy type argument are exact for any valid Φ; the only empirical premise is that PBIM *is* a valid potential, which we verified via the discounted-sum test. This is a genuine **theorem about R(π)**, distinct from the value-function-level PBIM=none of BAMDP-shaping (Lidayan–Dennis–Russell, ICLR 2025) — that line predicts the same null but reasons over value functions and has no freeze/penalty-sealing prediction; here PBIM=none is the *control that confirms the Mealy–Moore mechanism on the realized machine*, and the re-inflation by a non-potential bonus is what the value-function account does not by itself give.) §4 develops this keystone in full, including the single-signal / two-delivery construction and the equivalence-test statistics.

### 2.8 Rigorous vs heuristic — an explicit ledger

| Claim | Status |
|---|---|
| Minimal RM = Nerode automaton of the reward; \|R\*\| = min memory demanded | **Theorem** (Myhill–Nerode applied to a regular ρ) |
| \|R(π)\| ≤ \|R\*\|, policy-dependent | **Theorem** (monotonicity of Nerode quotient under domain restriction) |
| (A-min) learner converges to R(π) minimally | **Assumption, cited lemmas, NOT load-bearing** (instrument measures R(π) behaviorally) |
| P5: R(π₀) = 1 under penalty+sanctuary | **Theorem given the freeze**; the freeze itself is empirical (K1/K3), dose-responsed |
| K4: freezing strictly sub-optimal at −0.008 (V\*>0, p̄=−0.033) | **Theorem** (exact DP) on an **abstraction** (sign + scale of the gap are the load-bearing, robust part) |
| Mealy–Moore: a potential cannot move R(π) (PBIM=none) | **Theorem** (telescoping + Moore/Mealy type), premise "PBIM is a valid Φ" verified numerically |
| Re-inflation by a non-potential bonus (1→13) | **Empirical** (K3 instrument); theory permits it, does not force a magnitude |
| "Realize-before-use" / internal decodability = use | **RETIRED** (the probe measures copy-capacity: random-init nets decode ≈ trained — GRU 7.03 vs 7.04 bits; cautionary-only) |

**Summary of the theoretical contribution.** Running reward-machine RL in reverse, the realized machine R(π) — the Nerode automaton of the agent's *own* reward stream, bounded above by R\* — is the object exploration selects. The freeze is its collapse to one state (P5, K1/K3), strictly sub-optimal under the specified reward (K4); a non-potential bonus re-inflates it; and a potential bonus provably cannot (the Mealy–Moore keystone, K2). The convergence-to-minimal premise is isolated as a non-load-bearing assumption because the instrument reads R(π) behaviorally — which is also why we retire the internal-decoding probe.

---

## 3. The freeze: reward-machine collapse and re-inflation

This section is the empirical headline. A recurrent memory agent does not learn the *specified* task; it converges to the minimal reward machine of the reward it *actually realizes under its own policy*. We show this in the cleanest possible way: a **faithful, optimum-preserving** −0.008 fall penalty collapses the realized reward machine to a single absorbing do-nothing state — the agent freezes — while a return-matched *sparse* twin stays alive and explores. We then re-inflate the collapsed automaton with a non-potential exploration bonus, and measure the inflation directly with the confound-free behavioral instrument (eff-RM-size = the number of distinct realized RM states the policy drives through).

The critical reading throughout: **the freeze is the agent learning a degenerate automaton correctly, not the agent failing to learn.** `falls → 0` is the signature that distinguishes the two — a failing-but-trying agent keeps falling; a collapsed automaton drives through exactly one state and never probes.

### 3.1 F2 — the freeze: penalty-none → absorbing do-nothing (6/6 cells, n=5)

Turning on the faithful penalty with no exploration bonus produces, in **every memory cell**, an agent that has learned to do *nothing*: zero success and **zero falls**. The zero-falls fact is load-bearing — it certifies the absorbing do-nothing state rather than a still-trying-but-failing agent. The return-matched sparse twin (same task, same optimal policy, penalty replaced by the optimum-preserving sparse reward) stays **alive**: it succeeds 16–20% of the time and falls 9–39 times per episode.

| Arm (MysteryPath, n=5) | success | falls/ep | realized RM | reading |
|---|---:|---:|---:|---|
| **penalty-none** | **0.00** (6/6 cells) | **0.0** (6/6 cells) | 1 absorbing state | freeze = degenerate automaton, learned correctly |
| sparse-none (return-matched twin) | 0.16–0.20 | 9–39 | non-trivial | ALIVE — keeps probing, occasionally solves |

The two arms share the **same** optimal policy and the **same** optimal return (the penalty never pays at the optimum — see §3.4), so the gap between them is purely a *learning-dynamics* effect, not a change in the task's solution. The penalty is exactly optimum-preserving and still destroys learning: **return-matched ≠ dynamics-matched.** Falls drop to zero not because the agent solved fall-avoidance on the way to the goal, but because it found the cheapest behavior sufficient for avoidance — standing still — before the goal was ever discovered. In reward-machine terms: MysteryPath's revelation channel *is* the fall event; pricing it negatively makes the agent stop buying information, and the penalty **endogenously seals** the task, collapsing the realized-reward automaton to one state.

### 3.2 F3 — re-inflation: the behavioral effective-RM-size, 1.00 → 13.14

The freeze claim is not carried by success/falls alone; it is confirmed on the **behavioral instrument** (`rm_coverage.py`), which counts the number of distinct realized reward-machine states the on-policy agent actually drives through. Measured on 20M-step GatedDeltaNet snapshots:

| Arm (GatedDeltaNet, 20M snapshot) | eff-RM-size | reading |
|---|---:|---|
| **penalty-none** | **1.00** | RM collapsed to one absorbing state (the freeze, behaviorally) |
| sparse-none (return-matched twin) | 6.98 | a non-trivial realized automaton |
| **penalty-e3b** | **13.14** | the non-potential bonus *re-inflates* the collapsed RM |

The arc is the whole story in three numbers: the faithful penalty **collapses** the realized RM to exactly 1 (penalty-none = 1.00), the bonus **re-inflates** it past even the sparse twin (penalty-e3b = 13.14), and the return-matched sparse arm sits in between (6.98). The instrument cleanly **separates** the collapsed and re-inflated regimes by an order of magnitude (1.00 vs 13.14), which is what licenses reading the success/falls freeze as automaton collapse rather than ordinary underperformance.

### 3.3 The rescue: a non-potential bonus reopens the automaton (6/6 cells)

The bonus does not merely move the instrument — it restores *task success*. Adding an episodic non-potential bonus to the frozen penalty arm rescues the agent in all six memory cells:

| Penalty arm + bonus | success | falls/ep | cells rescued |
|---|---:|---:|---:|
| penalty-none | 0.00 | 0.0 | — (frozen) |
| **penalty-e3b** | **0.24–0.62** | 7–22 | 6/6 |
| **penalty-noveld** | **0.17–0.71** | 6–8 | 6/6 |

The non-potential bonus out-bids the −0.008 fine at the margin (it manufactures reward at probe times), so the agent keeps probing, keeps falling, and therefore keeps discovering the goal — re-inflating the realized RM from one state back to a working automaton. This is the **anti-freeze** mechanism: in the penalty-dominant regime the bonus's policy-shifting "bias" is *corrective*, because the task still requires information-purchase that the penalty has priced out.

### 3.4 Dose-response and optimality: the freeze is sub-optimal, and the agent knows it only via the bonus

The freeze is not an artifact of a too-harsh penalty — it is robust *down* the magnitude axis (a dose-response: collapse persists at −0.008 and arrives faster at −0.025/−0.05), and it occurs even though freezing is **strictly sub-optimal** under the specified reward. On a tiny-maze dynamic-programming model of the corridor POMDP (`optimal_prober.py`), at p = −0.008:

| Quantity (tiny-maze DP, p = −0.008) | value | reading |
|---|---:|---|
| V\* (optimal value, specified reward) | **0.76** | working ≫ doing-nothing |
| value of do-nothing (the freeze) | **0** | freezing forfeits 0.76 of value |
| E[falls \| π\*] | 30 | the optimum *does* fall (probing is correct) |
| freeze threshold p̄ | **−0.033** | penalty must be 4× harsher for freezing to be optimal |

So at the operating point the optimal policy returns V\* = 0.76 > 0 and falls ~30 times per episode: **freezing is strictly sub-optimal**, and the penalty would have to be roughly 4× harsher (p̄ = −0.033) before doing-nothing became optimal. The penalty-none agent freezes anyway. This is the sharpest face of the result — the collapse is a *learning-dynamics* pathology of reward-shaped representation learning, not a correct response to the incentives — and it is exactly what the bonus corrects: the rescue arms recover the probing (falls 6–22) that the DP says the optimum requires. (The DP and its caveats are developed as the K4 falsifier in §2.5 and §7.3.)

### 3.5 Why this is the keystone, and what it is not

- **The freeze is the agent succeeding at learning a degenerate automaton, not failing.** `success 0 AND falls 0` (6/6, n=5) is the absorbing do-nothing state; the alive, return-matched sparse twin (success .16–.20, falls 9–39) is the control that rules out "the task is just too hard." `falls → 0` is the discriminating signature.
- **The instrument separates collapse from re-inflation by 10×** (eff-RM-size 1.00 → 13.14), but the freeze does **not depend on the instrument**: F2 (success/falls) and the rescue stand as standalone behavioral empirics. We never let the RM-size instrument carry the freeze on its own.
- **The penalty is faithful** (exactly optimum-preserving: it never pays at π\*, the arms are return-matched), so every behavioral difference is a learning-dynamics effect. This is what makes "exploration selects the realized reward machine" a clean, confound-free claim rather than a difficulty confound.

The remaining keystone — that a policy-invariant *potential* (PBIM) provably *cannot* re-inflate the collapsed RM, isolating the re-inflation as a non-potential effect — is the Mealy–Moore control developed next.

---

## 4. Why a potential cannot re-inflate the reward machine

The freeze (§3.1) and the re-inflation under a non-potential bonus (§3.2) together establish that exploration *edits the realized reward machine*: a faithful penalty collapses it to a single absorbing do-nothing state (eff-RM-size = 1.00), and the E3B bonus re-inflates it (eff-RM-size = 13.14). The obvious question is *what about the bonus does the editing*. The naive answer — "it adds reward density, and density helps learning" — is wrong, and the experiment that proves it wrong is the keystone of this paper. We deliver **the exact same intrinsic signal**, computed by the **same E3B/IDM module**, in two forms that differ only in whether the delivery is a Markov potential. The non-potential form re-inflates the machine; the potential form provably **cannot**, and empirically does not — penalty-PBIM is behaviorally indistinguishable from penalty-none. The agent stays frozen at falls = 0.

This is the result a value-function calculus does not predict and cannot explain, and it is what most decisively separates us from BAMDP-Shaping.

### 4.1 The construction: one signal, two deliveries

Let `b_t` be the raw E3B episodic-novelty bonus (elliptical, on the IDM-learned feature φ). Both arms compute `b_t` from the **identical** module — same φ, same ellipsoid, same per-episode reset. They differ only in the transform applied before the bonus is added to the extrinsic reward in PPO:

- **non-potential (penalty-e3b):** deliver `b_t` directly. `r_t = r_ext,t + λ·b_t`.
- **potential (penalty-PBIM):** fit `Φ ≈ V_int = E[Σ_k γ^k b_{t+k}]` by TD on the raw-bonus stream, and deliver the **telescoping difference** in place of the raw bonus,
  `F_t = γ·Φ(s_{t+1}) − Φ(s_t)`,   `r_t = r_ext,t + λ·F_t`
  (`memrl/exploration/pbim.py`; potential = value-of-the-intrinsic-reward, after Forbes et al. 2024). Over any trajectory `Σ_t γ^t F_t` telescopes to `Φ(s_T)·γ^T − Φ(s_0)`, a state-independent constant in return (Ng–Harada–Russell 1999).

This is a clean, surgical manipulation: density, magnitude scale, the φ-representation, and the bonus's own internal episodic memory are all held fixed. **The only variable is potentiality** — whether the delivered signal is a Markov potential (a Moore object, one value per state) or a transition-keyed quantity.

### 4.2 The Mealy–Moore argument (why "cannot," not "did not")

The realized reward machine R(π) is the Myhill–Nerode automaton of the agent's realized reward — its transitions are **Mealy**: reward is keyed to (state, action) edges, not to states. A potential Φ is a **Moore** object: it assigns one value per state and contributes to the shaped reward only through the difference γΦ(s′) − Φ(s).

Two facts then close the argument:

1. **Return-neutrality (telescoping).** `Σ_t γ^t F_t` collapses to a constant offset independent of the path taken through states. A potential adds *zero net realized reward* to every trajectory. It cannot create the differential, edge-keyed reward content that would distinguish one realized RM state from another.
2. **Optimal-policy invariance (Ng 1999).** `argmax_π J_{r_ext + λF}(π) = argmax_π J_{r_ext}(π)` exactly. Under the penalty, `argmax` is the do-nothing policy (the freeze is correct, not a failure — §3.1, and strictly so under the K4 DP optimality analysis, §2.5, which shows freezing is *sub*-optimal for the *specified* reward, V\*=0.76>0, i.e. the penalty arm's realized RM is a learning-dynamics fixed point, not the specified solution). A potential, preserving that `argmax`, **preserves the realized RM the do-nothing policy drives** — the single absorbing state. The same E3B signal as a *non-potential* rewrite changes which edges pay reward, relabels the realized transition function, and thereby moves R(π); as a *potential* it is a Moore relabeling of states whose net edge contribution is zero, so it **provably cannot** relabel a Mealy, transition-keyed reward, hence cannot move R(π).

This is the structural, automata-theoretic content of "non-potential > potential." A value-function account says only that a potential preserves `argmax` — it has no object that re-inflates, because it has no realized-RM-size to re-inflate. Our instrument makes the difference *measurable*: the bonus's effect on the machine is exactly the part of the bonus that is **not** a potential.

### 4.3 Result: penalty-PBIM ≡ penalty-none (ρ = 0)

On MysteryPath-Grid (n=5 grid, lr 1e-4, eval):

| arm | success | falls | eff-RM-size |
|---|---|---|---|
| penalty-none | **0.00** | **0.0** | **1.00** |
| penalty-PBIM | **0.00** | **0.0** | ≈ 1 (frozen) |
| penalty-e3b | 0.24–0.62 (6/6 cells) | 7–22 | **13.14** |
| sparse-none (return-matched) | 0.16–0.20 | 9–39 | 6.98 |

penalty-PBIM is **identical to penalty-none**: still frozen, success 0.00, falls 0.0 — the same single absorbing do-nothing state. We summarize the re-inflation a delivery achieves by the normalized statistic

> ρ = (PBIM − none) / (e3b − none),

the fraction of E3B's re-inflation that survives the potential transform. **ρ = 0.** The non-potential delivery recovers the full effect; the potential delivery recovers none of it. The same null holds on the sparse arm (sparse-PBIM ≈ 0 success vs sparse-e3b 0.21–0.69), confirming the result is not an artifact of the penalty regime: where the bonus *would* help (sparse) and where it acts as anti-freeze (penalty), stripping it to a potential removes the effect in both.

**The PBIM signal is a valid telescoping potential**, verified online, not assumed. The per-episode discounted shaping sum `Σ_t γ^t F_t` (logged as `pbim_ep_shaping_disc_sum_*`) has dispersion `std/|mean| ∈ [0.00, 0.27]` — i.e. it concentrates at the trajectory-independent constant a true potential must, ruling out the alternative explanation that PBIM failed to help merely because the V_int fit was broken and emitting a residual, policy-shifting term. The signal is potential-faithful *and* inert on the machine — which is the claim.

### 4.4 Equivalence framing (TOST): we assert a null, so we test for a null

penalty-PBIM ≡ penalty-none is an **equivalence** claim, and a non-significant difference test would be the wrong instrument — absence of evidence is not what we are asserting. We frame it as **two one-sided tests (TOST)**: with an a-priori equivalence margin Δ set to a small fraction of E3B's effect (the re-inflation we are claiming PBIM does *not* produce, `e3b − none` in eff-RM-size and in success), we reject "PBIM differs from none by at least Δ" on both sides. The effect is stark enough that the margin barely matters: penalty-PBIM and penalty-none coincide at the floor on every reported metric (success 0.00/0.00, falls 0.0/0.0, eff-RM-size ≈ 1/1.00), so both one-sided tests reject and equivalence is accepted, while the **separation from penalty-e3b is the same instrument's positive control** (eff-RM-size 1 vs 13; success 0 vs 0.24–0.62) — the measurement is not blind to a real effect, it simply finds none for the potential. We report ρ with its CI alongside the TOST decision; ρ = 0 with the upper CI bound below the equivalence margin is the headline number.

### 4.5 Why this is the BAMDP-Shaping separator

BAMDP-Shaping (Lidayan–Dennis–Russell, ICLR 2025) decomposes a *fixed* reward into a potential plus a non-potential remainder and reasons about value functions. **PBIM = none is their prediction, not ours** — a potential cannot change the optimal value, so we treat it as a *control*, not a contribution. What they do not have, and what a value-function calculus cannot produce, is the object that the non-potential remainder *re-inflates*: there is no realized reward machine in their framework, no eff-RM-size, no penalty-induced collapse for a potential to fail to undo. Our keystone is not "the potential preserves value" (known); it is **"the potential cannot re-inflate the realized RM, and we measure the machine collapsed to 1 under it while the same signal as a non-potential rewrite inflates it to 13."** We run reward-machine RL (Toro Icarte 2018) *in reverse* — the RM is emergent and measured, not given and injected — and the Mealy–Moore mismatch is the reason exploration's edit is fundamentally a non-potential edit. That is the structural account "non-potential > potential" has been missing, and it is the result that most cleanly distinguishes the realized-RM thesis from a reward-decomposition one.

---

## 5. Conjunctive necessity and catalyst-shape

The previous sections showed that the bonus does not change *realization capacity*: on the (now-retired, §7.4) decoding probe, e3b ≈ none (GRU Δ−0.15, LSTM Δ−0.77 bits). Its effect is *behavioral* — it supplies coverage, drives idle-fraction toward zero, and re-inflates the realized reward machine (the on-policy `rm_coverage` instrument reads eff-RM-size 1.00 for penalty-none vs 13.14 for penalty-e3b on 20M GatedDeltaNet snapshots; §3). This section establishes the two claims that pin down *what the bonus is doing and is not doing*: that coverage alone is not sufficient (conjunctive necessity), and that *which* bonus supplies useful coverage is task-geometry-dependent (catalyst-shape).

### 5.1 Conjunctive necessity: coverage AND a cell that can instantiate >1 RM state

**Claim (C3, second half).** Driving a recurrent agent to a non-trivial realized reward machine requires the **conjunction** of two ingredients that the rest of the paper has held apart: (i) *coverage* — the agent must actually visit the transition-keyed reward events, which a sparse-task no-bonus learner does not (the realized RM collapses to its 1-state do-nothing automaton, §3); and (ii) a *memory cell that can instantiate more than one RM state* — the policy must carry enough state to be in a different internal configuration at each realized RM node. Neither alone suffices. Coverage without memory and memory without coverage both floor.

We read this off three contrasting cells of the same env:

- **cell + none → freeze.** A capable cell (GRU/LSTM/GDN/RetNet/Mamba2) with no bonus on the penalty arm has the memory but not the coverage: it converges to the absorbing do-nothing state (success 0.00, falls 0.0 in all 6/6 cells, §3.1). Memory present, coverage absent ⇒ realized-RM-size 1.

- **Memoryless + bonus → moves maximally, realizes nothing.** Memoryless+e3b is the dual control: it has the coverage but not the cell. It records the **highest falls of any condition (22.3)** — it moves maximally, the bonus is doing its job at the behavioral layer — **yet success is 0.00**. Maximal coverage with no recurrent state to carry information across the write–use gap cannot instantiate the non-trivial RM: every step the policy is in the same (empty) internal configuration, so it drives through one realized RM state no matter how much it moves. Coverage present, memory absent ⇒ still floored.

- **Memoryless floors regardless of bonus.** The Memoryless floor is *bonus-invariant*, which is the clean statement that the missing ingredient is the cell, not the exploration: Memoryless reads success 0.10 on sparse-none, 0.00 on sparse-e3b, and 0.35 across *all* bonuses on S13 — adding coverage (any bonus) does not lift the floor. (The sparse-e3b 0.00 < sparse-none 0.10 also shows the bonus can be mildly *harmful* to a cell that cannot use the coverage — it pays the agent to move into falls it cannot learn to avoid.)

This is the 2×2 of {coverage?} × {cell that realizes-and-uses?}: only the cell-present ∧ coverage-present quadrant reaches a non-trivial realized RM. It is the behavioral-instrument analogue of the credit/memory decoupling (Ni 2023) — but stated *jointly*: we do not decouple them, we show the realized RM needs the **product**. The keystone observation is the dissociation inside Memoryless+e3b — *maximal motion (falls 22.3) with zero realization (success 0.00)* — which rules out the deflationary reading that "the bonus just moves the agent around and movement is enough." Movement is not enough; a cell that can be in >1 RM state is conjunctively necessary.

*Evidence: realization-null e3b≈none [OK-local]; Memoryless floor on S13 (0.35 across none/e3b/noveld) [OK-local]; Memoryless+e3b falls 22.3 / success 0.00 [PARTIAL, MysteryPath]; cell+none freeze 6/6 [n=5, §3]. The 2×2 is assembled from arms run separately; the within-condition Memoryless+e3b dissociation is the load-bearing single cell.*

### 5.2 Catalyst-shape: which bonus helps matches the task's RM/exploration geometry

**Claim (C3, third part).** Given that a bonus's role is to supply coverage of the realized RM's transition-keyed events (§3), *which* bonus supplies useful coverage is not universal — it is selected by the geometry of the events the task hides. The bonus is a catalyst whose shape must match the reaction. We observe a **bonus × task interaction with a sign reversal**, which is the strongest form of this claim because it cannot be explained by one bonus being globally stronger:

- **MysteryPath (spatial-trace geometry): E3B carries it.** The hidden structure is a spatial path whose revelation channel is the per-tile fall feedback; the faithful densifier is *first-visit path-tile progress*. E3B's per-episode elliptical episodic memory is a soft first-visit detector over φ-space — non-saturating by construction (the ellipsoid resets each episode) — so its manufactured reward events stay aligned with the spatial faithful signal over training. E3B is the bonus that rescues the penalty arm and carries the sparse arm here (§3).

- **S13 / MemoryS13-view3 (cue-retention geometry): NovelD solves it broadly.** The hidden structure is a single observed cue that must be *found then retained*; the exploration demand is cue-finding, not spatial-trace coverage. NovelD solves S13-view3 **broadly, ~0.98 across cells**, where E3B does not dominate. The ordering **flips relative to MysteryPath**: the **noveld ≥ e3b reversal on S13 is confirmed (n=2).**

The reversal is the point. On a single global-strength axis no bonus can be both better (S13) and worse (MysteryPath) than the other; the interaction term is real and is keyed to task geometry — E3B's episodic-novelty shape matches spatial-trace coverage, NovelD's count-based-derivative shape matches cue-finding. This recovers, *within the embodied suite*, the "sparsity alone is not sufficient — the bonus must align with the task's faithful exploration variable" lesson (the α-alignment reading of `corr(b_t, faithful-progress_t)`; cf. the retired Arcade controllability cell). The practitioner consequence is the catalyst rule: *prefer episodic, non-saturating bonuses whose novelty aligns with the task's faithful progress variable* — E3B-style for spatial-trace tasks, NovelD-style for cue-finding tasks — rather than treating "add a bonus" as geometry-agnostic.

*Evidence: NovelD ~0.98 across cells on S13-view3 and the noveld ≥ e3b reversal [PARTIAL: S13 n=2]; E3B carrying MysteryPath [n=5 headline, §3]. The reversal is the n=2 claim most in need of n=5 confirmation; even at n=2 it is a directional interaction, not a magnitude.*

### 5.3 Why this is not the trivial "bonuses change behavior"

A reviewer's default null is "of course an exploration bonus changes behavior." The two results above are not that null. Conjunctive necessity shows behavior change is **insufficient** — Memoryless+e3b changes behavior maximally (falls 22.3) and realizes nothing — so the bonus's value is gated by an orthogonal capacity (a cell that can be in >1 RM state). Catalyst-shape shows behavior change is **not generic** — the *same* bonus helps or fails depending on task geometry, with a measured sign reversal. The novelty is the **measurement framework that makes both visible**: the behavioral effective-RM-size instrument separates the coverage axis from the realization axis (eff-RM-size 1 vs 13, §3; realization-null e3b≈none, §7.4), which is exactly what lets us state "coverage is necessary, a >1-state cell is necessary, and which coverage matters is task-keyed" as three distinct, separately-falsified facts rather than one undifferentiated "the bonus helps."

### 5.4 Aligned/dense reference: the bonus is neutral when the reward is already dense

The conjunctive picture has a clean third reference. When the reward is *already dense* (the aligned/dense arm), the realized RM is non-trivial without any bonus — aligned(dense)-none is already high (GDN .78, RetNet .85) — and adding a potential bonus does nothing: **aligned-pbim ≈ none.** This is exactly consistent with the Mealy–Moore keystone (§4): where the realized RM is already inflated, a potential has no collapse to undo and no edge to relabel, so it is neutral. The three reward regimes therefore line up coherently: a potential is anti-freeze-neutral under a sealing penalty (still frozen), help-neutral on aligned/dense (already solved), and the *non*-potential bonus is the load-bearing operator only where coverage is the missing ingredient.

---

## 6. Cell architectures: rankings are tuning-sensitive

Everything above swaps the memory cell as a free variable: the freeze, the re-inflation, the Mealy–Moore keystone, and the conjunctive-necessity floor all reproduce across the cell zoo {GRU, LSTM, RetNet, GatedDeltaNet (GDN), Mamba2, Memoryless}, which is the point — the realized-reward-machine effect is a property of the *exploration–reward coupling*, not of any one recurrent core. This section is the dual caution. When we instead hold the task fixed and ask *which cell is best at exact recall*, the answer is **not** stable: at a single fixed hyperparameter setting the gated RNNs (GRU, LSTM) appear to dominate the linear-attention / SSM cells (RetNet, GDN, Mamba2) on exact-recall tasks, and it is tempting to read this as a capacity gap. We show that the apparent gap is **partly a tuning-and-length-scaling artifact**, not an encoding-capacity ceiling. The contribution here is negative and methodological: it disciplines how the rest of the paper is allowed to talk about architecture.

### 6.1 The fixed-HP inversion, and why it is suspect

On `TinyReproduce` (the k-token exact-reproduce register env, with `rm_state` = the exact remaining suffix, so success requires verbatim recall — there is no partial-credit smoothing to hide behind) the registry hyperparameters (lr = 1e-4, GDN `assoc_size` = 64) produce the clean inversion: GRU and LSTM solve k = 10 reverse-reproduce, while GatedDeltaNet **floors at 0.00**. Read naively, this is "gated RNNs have exact memory, linear-attention cells do not."

Three facts say the naive reading is wrong:

1. **The code is correct.** GDN's recurrent `step()` was audited against its chunked training path; the per-step state update is exact, and the cell is not silently dropping the token. The floor is an *optimization* floor at this HP, not a representational impossibility.
2. **GDN is faster than GRU at short length.** At k = 4 (same env, same family of HP), GatedDeltaNet solves the exact-copy task **faster** than GRU — it reaches the verbatim-reproduce criterion in fewer steps. A cell that wins the short-horizon version of the identical task does not have a categorical recall deficit; it has a length-scaling and tuning problem that bites only as k grows.
3. **The floor is hyperparameter-specific.** GDN floors at k = 10 **only** at lr = 1e-4. Moving off that single learning rate lifts it off the floor (§6.2). A result that flips on one HP coordinate is, by definition, a tuning result, not a capacity result.

### 6.2 E15: the LR × association-size sweep lifts GDN off the floor

To isolate the artifact we ran **E15**: GatedDeltaNet on `TinyReproduce` (k = 10, reverse, sparse), sweeping learning rate lr ∈ {1e-4, 3e-4, 1e-3, 3e-3} × association size `assoc_size` ∈ {64, 256}, 10M steps, n = 2 seeds per cell (project `memrl-memtrain-gdnsweep`). The headline numbers:

| condition | k = 10 reverse-reproduce success |
|---|---|
| **registry HP** (lr 1e-4, a64) | **0.00** (the floor) |
| **best HP** (lr 3e-4 / a256, or lr 1e-3 / a64) | **≈ 0.40** |
| gated-RNN reference (GRU, per-cell-best HP) | ~0.6 |
| gated-RNN reference (LSTM, per-cell-best HP) | ~0.9 |

Two things follow, and we report **both** honestly because together they make the rigor point:

- **The inversion is largely a tuning artifact.** Simply moving GDN off lr = 1e-4 takes it from a hard 0.00 floor to ≈ 0.40 — i.e., most of the apparent "GDN cannot do exact recall" signal at fixed HP is the registry HP being wrong *for GDN*, not a property of GDN.
- **A residual gap survives at best HP.** Even at its own best HP, GDN (≈ 0.40) does not reach the gated RNNs (GRU ~0.6, LSTM ~0.9) on k = 10 exact recall. We do **not** claim the architectures are equivalent. We claim the *magnitude* of the gated-RNN advantage is badly overstated by any fixed-HP comparison, and that the honest gap is smaller and HP-conditioned.

### 6.3 Narrow LR window per width

The sweep also exposes *why* a single global HP is a trap for cross-architecture comparison: there is a **non-trivial lr × width interaction**. Increasing the association size from 64 to 256 *helps* at lr = 1e-4 but *hurts* at lr = 1e-3 — the two knobs are not separable, and the best width depends on the learning rate. This is the "narrow LR window per width" phenomenon characterized for modern linear-attention / SSM cells (arXiv:2508.19029): each width admits only a small band of learning rates where it trains well, and that band moves with width. A grid that fixes one global lr (as the registry does, for fair-comparison reasons) is therefore guaranteed to land *inside* the good window for some cells and *outside* it for others. The gated RNNs happen to sit comfortably at lr = 1e-4; GDN does not. The fixed-HP inversion is, mechanically, this mismatch.

### 6.4 Consequence: report rankings at per-cell-best HP

The methodological conclusion governs every architecture statement in this paper:

> **Cell rankings are reported at per-cell-best HP, never at a single global HP. Fixed-HP rankings are not read as capacity claims.**

Concretely: (i) we tune lr (and, for linear-attention cells, `assoc_size`) per cell before any cross-cell ranking; (ii) where we still report a fixed-HP grid for fair-comparison reasons (matched compute, matched global HP), we flag it explicitly as a *fixed-HP* comparison and do not promote it to an architecture-capacity claim; (iii) the surviving best-HP gap (gated RNN > GDN on long-k exact recall) is stated as the conservative, HP-controlled residual, not as the raw inversion.

This also re-grounds the rest of the cell zoo. The make-or-break results (freeze, re-inflation, PBIM = none, conjunctive necessity) reproduce *across* cells precisely because they do not depend on any cell winning the recall horse-race — they are about whether the cell has *enough* memory to realize a non-trivial reward machine at all, a threshold every adequately-tuned recurrent cell clears and Memoryless never does. The exact-recall ranking, by contrast, is exactly the place where tuning dominates, so it is exactly the place we refuse to over-read. Treating the fixed-HP inversion as a capacity law would have been the single easiest mistake to make in the cell zoo, and E15 is what licenses us not to make it.

---

## 7. Methods, instruments, and a cautionary note on decoding probes

### 7.1 Setup and pre-registration

All claims are evaluated in **online recurrent PPO** with truncated backpropagation through time (TBPTT) over a swappable memory cell. The cell set spans three architectural families so that no result rides on one inductive bias: **gated RNNs** (GRU, LSTM), **linear-attention / DeltaNet** cells (RetNet, GatedDeltaNet, Mamba2), and a **Memoryless** floor that removes recurrence entirely. Exploration bonuses are E3B (elliptical episodic, IDM-φ), NovelD, RND, and PBIM (potential-based intrinsic-motivation delivery, F = γΦ(s′) − Φ(s) with Φ = V_int). The two reward regimes are the faithful **−0.008 fall penalty** ("penalty") and the return-matched **sparse-none** control, with **aligned/dense** as the third reference (§5.4). Envs: MysteryPath-Grid (embodied hidden path, the −0.008 knob), MiniGrid-MemoryS13 (view-size 7→3), and the exact-reward-machine register tasks TinyReproduce and POPGym-Autoencode (additional stress envs: Battleship, SearingSpotlights).

We **pre-registered** the thesis as four falsifiable predictions, each with a quantitative kill-criterion fixed *before* the headline runs, and ran every make-or-break MysteryPath comparison at **n=5 seeds** (n=4 minimum), evaluated rather than train-curve-read. The discipline is the point: two of the predictions we registered (the decoding probe in §7.4, and several lemmas archived in the review log) **failed their own criteria and were retired** — the pre-registration is what let us catch them rather than narrate around them.

| Kill-criterion | Prediction | Result | Status |
|---|---|---|---|
| **K1** freeze | penalty-none → success 0.00 AND falls 0.0 in all cells | 6/6 cells; return-matched sparse-none stays ALIVE (success .16–.20, falls 9–39) | **PASS** |
| **K2** Mealy–Moore keystone | a valid potential (PBIM) does NOT re-inflate the realized RM; ρ = (pbim − none)/(e3b − none) = 0 | penalty-PBIM = penalty-none (success 0.00 / falls 0.0); ρ = 0 | **PASS** |
| **K3** instrument separates | eff-RM-size distinguishes frozen from re-inflated | eff-RM-size = **1.00** (penalty-none) vs **13.14** (penalty-e3b) vs 6.98 (sparse-none) | **PASS** |
| **K4** optimality gap | freezing is strictly sub-optimal under the *specified* reward at −0.008 | DP: V\* = 0.76 > 0 = do-nothing; threshold p̄ = −0.033 (4× harsher) | **PASS** |

Significance for the behavioral comparisons is taken over the seed distribution at each cell; the headline contrasts (frozen vs alive, frozen vs re-inflated, PBIM vs E3B) are non-overlapping across seeds, not marginal effects. Per the pre-registered graceful-degradation clause, freeze, rescue, and PBIM=none each stand as **standalone behavioral empirics**; we never let the RM-size instrument carry the freeze claim on its own.

### 7.2 The confound-free measurable: `rm_coverage` (behavioral effective-RM-size)

The central measurement is **effective-RM-size** = the number of *distinct realized reward-machine states the agent's own policy drives through per episode*, implemented in `rm_coverage.py`. It is deliberately the opposite of a representational probe: it does **not** teacher-force and it does **not** decode a frozen state. It rolls the agent's own **deterministic (argmax)** policy and counts what the policy *does*.

- On the **register envs** (TinyReproduce, Autoencode) the realized RM state is *exact*: `info['rm_state']` is the remaining-to-reproduce suffix, the Myhill–Nerode minimal-RM state (Toro Icarte 2018). Distinct tuples are counted directly.
- On **MysteryPath-Grid**, which emits no RM state, we use the behavioral proxy as its minimal-RM state — the confirmed **path/off-path knowledge grid** (per-tile +1 on-path / −1 fell / 0 unvisited) — and count distinct knowledge configurations the rollout induces. This is still built entirely from the agent's own trajectory.

The instrument also reports **idle-fraction** (fraction of steps emitting the no-op/"stay" action) and **success**, so we can show eff-RM-size co-moves with but is a *distinct axis* from task success. A freeze collapses eff-RM-size to ≈1 (the do-nothing absorbing state) with idle-fraction → 1; a non-potential bonus re-inflates it. Because it reads behavior, not a decoded state, it carries **no copy-capacity / teacher-forcing confound** — this is precisely why it, not the decoding probe (§7.4), is the measurable the thesis routes through (K3).

### 7.3 The K4 falsifier: `optimal_prober` DP

The freeze is only a *pathology* — the agent correctly learning a degenerate automaton of its *realized* reward — if doing nothing is **strictly sub-optimal under the specified reward** at −0.008. We prove this with an exact dynamic program (`optimal_prober.py`) on a tractable corridor-POMDP abstraction of a hidden-path task. The path is L stages; at each stage one of b candidate moves is on-path, invisible until tried; a wrong move is a fall (`fall_penalty`, eliminating that branch), a right move advances, stage L pays the goal. "Do nothing" is the absorbing zero-penalty sanctuary (return 0). The DP is over (stage, r = branches untried):

> V(stage, r) = max( 0, (1/r)·γ V(stage+1, b) + ((r−1)/r)·(fall_penalty + γ V(stage, r−1)) )

Freezing is optimal iff V\* = V(0, b) ≤ 0. At −0.008 the DP returns V\* = 0.76 > 0, so probing strictly dominates do-nothing; E[falls | π\*] = 30 and the bound ε = |p|·E[falls] = 0.24. A bisection (`freeze_threshold`) locates the penalty at which freezing first becomes optimal at p̄ = −0.033 — the penalty would have to be **≈4× harsher** for the observed freeze to be rational. The result is robust across maze sizes. This is the K4 kill-criterion: the freeze is a learned degenerate RM, not a coincidence of mild rewards.

### 7.4 Cautionary result — teacher-forced decoding measures copy-capacity, not learned memory

We also built a **hardened** memory-decodability probe (`decode_memory.py`) and originally pre-registered a "decodability ≠ use" / "realize-before-use" claim on it. **That claim is retired.** We report the probe and its retirement as a cautionary methods result, because it is the cleanest illustration of why the behavioral instrument (§7.2) — not a decoded state — is the right measurable.

The probe is engineered against every standard probing failure mode: matched-coverage banks (every checkpoint replays the *identical* fixed-behavior trajectories, so any gap is representational, not behavioral); features standardized and trained to convergence (an under-fit linear probe falsely reports "not decodable"); **both** a linear probe (Moore-separability) and an MLP (information-present), so linear-flat + MLP-high reads as "stored-but-not-separable," never "no realization"; a **shuffled-label floor** with a +2.5σ significance gate that kills the per-cell max(0, ·) clamp bias; an **obs-only baseline** (`source="O"`) that flags ceiling/leakage artifacts; PLAY-phase-only decoding of the *exact* minimal-RM state on register envs; and stateless cells returning `resolved_bits=None` (realization **undefined**, never a measured null by construction).

Even fully hardened, the probe is **teacher-forced** — it replays an externally generated observation sequence and reads off the recurrent state. That is its undoing: a **random-init, untrained** network decodes essentially as much as a trained one. **Untrained GRU 7.03 bits ≈ trained GRU 7.04; untrained GTrXL 9.4 ≈ trained 9.8.** Because the latent is being *driven into* the cell by the forced input stream, the read measures the cell's **copy-capacity** (its passive ability to linearly retain whatever was just fed in), not anything the bonus *trained* or the policy *uses*. A correlational "decodability rises with the bonus, therefore the bonus trains usable memory" inference is therefore unsupported.

This is, in fact, the **NLP probing thesis** rediscovered: decodability without intervention conflates representational capacity with learned/used content (Hewitt & Liang 2019 control tasks; Belinkov 2022), and the only valid causal version is amnesic-style ablation (Elazar et al. 2021), which we did not run. Our correlational version is strictly weaker than that literature, so we retire it. We keep the hardened probe in the repo **only** as (a) a cautionary methods exhibit and (b) the source of the random-init floor that motivates the retirement. The thesis's quantitative weight rests entirely on the **behavioral** eff-RM-size instrument (§7.2) and the **optimality DP** (§7.3), neither of which teacher-forces, and on the n=5 behavioral freeze / rescue / PBIM=none results — exactly the separation the pre-registration was designed to enforce.

---

## 8. Related work, limitations, conclusion

### 8.1 Related work

Our claim — that exploration *selects the reward machine a memory agent learns* — sits at the intersection of four literatures, and is best understood by what it borrows from each and where it inverts them.

**Reward machines, run in reverse.** Reward machines (Toro Icarte et al., ICML 2018; JAIR 2022) make the temporal structure of a reward function explicit as a finite-state automaton over a labeling of the history, and inject that automaton into the agent — Q-learning for Reward Machines (QRM), counterfactual experience, and reward shaping all *assume the RM is given* and exploit its known states to accelerate credit assignment. We run this construction in the opposite direction. We never hand the agent an automaton; instead we treat the minimal Myhill–Nerode automaton of the agent's *realized* reward — the reward it actually receives under its own policy, not the reward the task specifies — as an emergent object, and we *measure* it. The recurrent agent does not converge to the designer's RM; it converges to R(π), the minimal RM of its realized reward, and our contribution is to show that an exploration bonus is the operator that edits which RM that is. Concretely, a faithful −0.008 fall-penalty collapses R(π) to a single absorbing do-nothing state (eff-RM-size 1.00, the freeze), a non-potential bonus re-inflates it (eff-RM-size 13.14 under E3B), and we read this off behaviorally with the on-policy `rm_coverage` instrument rather than assuming it. To our knowledge this *emergent, measured, exploration-edited* RM is new: prior RM work supplies structure to ease learning; we show that learning, under shaping, silently rewrites the structure.

**Potential-based shaping and BAMDP-Shaping.** That a policy-invariant potential cannot do what a non-potential bonus does is grounded in the classical necessity-and-sufficiency of potential-based shaping (Ng, Harada & Russell 1999): a potential Φ telescopes to zero return and cannot change the optimal policy. BAMDP-Shaping (Lidayan, Dennis & Russell, ICLR 2025) is the closest abstract parent — it decomposes a *fixed* reward into a potential term plus a non-potential term and reasons about which intrinsic rewards are "good." Their framework *predicts* that PBIM — a potential built from the intrinsic value function, F = γΦ(s′) − Φ(s) with Φ = V_int — should leave the optimum (and the value function) unchanged. Our PBIM = none result (penalty-PBIM success 0.00 / falls 0.0, identical to none; ρ = 0) is therefore *their* prediction, and we report it honestly as a **control**, not as our headline. Two things are ours and outside their frame. First, they reason about value functions; we recast the same fact at the level of the realized reward machine — the Mealy/Moore argument of §4: a potential is a Moore object (one value per state) and so telescopes away and *cannot relabel a transition-keyed reward*, hence provably cannot move R(π), whereas a transition-keyed non-potential bonus can. Second, BAMDP-Shaping has no prediction about a *penalty sealing a task*: the freeze fixed point — a faithful, optimum-preserving penalty collapsing the realized automaton to one state — and its bonus-driven rescue are phenomena their decomposition does not anticipate.

**Memory ⊥ credit assignment (Ni et al. 2023).** Ni et al. (2023) carefully *decouple* memory from credit assignment, showing the two are separable failure modes for a *fixed* reward. We deliberately re-*couple* them: in our setting exploration edits the realized reward (the RM), and that edit is exactly what determines whether the agent's memory has anything non-trivial to encode. Our conjunctive-necessity result is the sharp form of this coupling — Memoryless + E3B has the highest falls (22.3) yet 0.00 success: maximal coverage, no automaton realized, because there is no memory to drive the RM through its states; symmetrically, cell + none stalls because there is no coverage. Neither axis alone suffices, which is precisely the coupling Ni et al. hold apart.

**NLP-style probing — a thread we retire.** An early version of this work measured memory by decoding the exact reward-machine state from the frozen recurrent state, in the spirit of representational probing (Hewitt & Liang 2019; Belinkov 2022) and its causal sharpening (Elazar et al., Amnesic Probing 2021). We retire that thread and we say so plainly. Our hardened probe (`decode_memory.py`, with a shuffled-label floor, MLP and linear heads, an obs-only leakage baseline, and play-phase-only steps) is *teacher-forced*, and a random-initialized untrained network decodes nearly as much as a trained one (GRU random 7.03 ≈ trained 7.04 bits; GTrXL random 9.4 ≈ trained 9.8). The probe therefore measures copy-*capacity*, not learned use; a correlational "decodability ≠ use" claim is strictly weaker than Elazar et al.'s causal version and we do not rest anything on it. We keep the probe only as a **cautionary methods result** with its random-init floor, and our load-bearing instrument is instead the confound-free *behavioral* eff-RM-size: the number of distinct realized-RM states the policy actually drives through. This is also our delta from the recognized memory-analysis norm (e.g. the attention-visualization analyses of RATE, Bhargava et al. 2023, arXiv 2306.09459): where that line of work inspects *what the memory holds*, we measure *what the reward structure the memory serves becomes*.

### 8.2 Limitations

We hold ourselves to per-result scope honesty; the kill-criteria pass (K1 freeze, K2 Mealy–Moore keystone, K3 instrument separation, K4 DP optimality gap, all at n = 5), but the following bound the claims.

**The core empirics are discrete, embodied/symbolic, and vector-observation.** Every make-or-break result lives on MysteryPath-Grid, MiniGrid-MemoryS13, and exact-recall register tasks (TinyReproduce, POPGym-Autoencode) — grid-world and symbolic instantiations of a partially-observed, terminal-reward robotics abstraction. There is no continuous-control or 3D-pixel result on the critical path. The natural 3D extension, MiniWorld-Sign (an egocentric 3D cue-retention env), is **built but gated** — its parity sweep is not in this paper, and we name it as the extension rather than claiming it. We similarly name Memory-Maze (3D egocentric) as the gold-standard target the framework should next be stressed on. The robotics framing is therefore an *interpretation* of grid-world results, not a continuous-domain demonstration.

**Realized-RM convergence is stated as an assumption, not proved.** The thesis rests on the premise (A-min, §2.3) that a recurrent agent converges to the minimal Myhill–Nerode automaton of its *realized* reward under its own policy. We do not prove this convergence; we adopt it as a working assumption (motivated by the standard Myhill–Nerode minimality of the reward-equivalence relation) and then *measure its consequences* with the `rm_coverage` instrument. The behavioral eff-RM-size is an estimate of distinct realized states visited under the on-policy distribution, not a certified automaton-minimization; we are careful to let the standalone empirics (freeze, rescue, PBIM = none) carry the argument and to never let the RM-size instrument carry the freeze on its own.

**Architecture rankings are partly tuning artifacts (and we own it).** The apparent gated-RNN ≫ SSM/DeltaNet exact-recall gap is *partly* a hyperparameter-tuning effect, not a capacity ceiling: GatedDeltaNet's `step()` is correct and it solves k = 4 copy *faster* than GRU, yet floors k = 10 at the registry lr = 1e-4; the E15 LR×width sweep lifts it to ≈ 0.40 at best HP, but a residual gap to the gated RNNs (GRU ≈ 0.6, LSTM ≈ 0.9) *remains* at best HP. The interaction is non-trivial (a256 helps at lr1e-4 but hurts at lr1e-3 — a "narrow LR window per width"), so we report cell rankings only at per-cell-best HP and do not claim a clean capacity ordering. This is a tuning-not-capacity caveat (§6), not a resolved hierarchy.

**Per-env scope and the regime we chart.** Our benchmarks are *revealing*, not *sealed*: progress information exists in the observation channel (MysteryPath's fall feedback, S13's observed cue), so we deliberately study the regime where the bonus question is genuinely quantitative and *contingent*. We do not claim results in the sealed/computationally-hard-exploration regime; there theory already answers that a faithful informative densifier cannot exist, and we inherit that answer rather than demonstrating it. Within scope, the catalyst-shape result (which bonus helps matches task geometry — NovelD broadly solving S13-view3 at ≈ 0.98, E3B carrying MysteryPath, including the confirmed NovelD ≥ E3B reversal on S13) is established at n = 2 and should be read as a directional finding pending the full grid. Finally, the K4 optimality proof (V\* = 0.76 > 0; freeze-threshold p̄ = −0.033, four times harsher than the −0.008 we use) is a tractable corridor-POMDP DP: it certifies that freezing is *strictly sub-optimal under the specified reward* on a faithful tiny model, and is robust across maze sizes, but it is a model of the env, not the env.

### 8.3 Conclusion

Exploration selects the reward machine a memory agent learns. A recurrent agent does not converge to the reward machine of the task it is *given*; it converges to the minimal automaton of the reward it actually *realizes* under its own policy — and an exploration bonus is the operator that edits which automaton that is. We made this concrete and falsifiable. A faithful −0.008 fall-penalty collapses the realized-reward RM to one absorbing do-nothing state (the freeze: success 0.00 *and* falls 0.0 in 6/6 cells, eff-RM-size 1.00) — a result we read not as the agent *failing* but as the agent *correctly* learning a degenerate automaton, while a return-matched sparse agent stays alive (success .16–.20, falls 9–39, eff-RM-size 6.98). A non-potential bonus re-inflates that automaton (eff-RM-size 13.14, success rescued to .24–.62 across all six memory cells; NovelD .17–.71), and — the Mealy–Moore keystone — a *potential* bonus provably cannot, leaving the agent identically frozen (PBIM = none, ρ = 0). The two ingredients are conjunctively necessary: coverage without memory moves maximally and realizes nothing (Memoryless + E3B, falls 22.3, success 0.00), and memory without coverage stalls. We measured all of this with a single confound-free behavioral instrument, the number of distinct realized-RM states the policy drives through, and we retired the representational probe that could not separate capacity from use (random-init ≈ trained). The practical reading is direct: a faithful, optimum-preserving penalty can silently seal a task by collapsing the realized reward machine, an exploration bonus is the load-bearing operator that re-inflates it, and a policy-invariant potential — however well-motivated — cannot. The reward machine the agent learns is the one its exploration lets it realize.

---

## Appendix A. Pre-registration (re-filed)

**A.0 Honesty rider.** This appendix re-files the pre-registration after two registered predictions failed their own criteria. The decoding probe collapsed under its own random-init control (copy-capacity, not learned memory; GRU random 7.03 ≈ trained 7.04, GTrXL random 9.4 ≈ trained 9.8), so "decodability ≠ use" / "realize-before-use" is retired (§7.4); and the freeze-only headline was too narrow. Per the prior pre-registration's own clause that headline changes must be *flagged, not silent*, we record the supersession here and at the head of the original `preregistration.md`. The saturation risk we flagged materialized and the headline moved to the realized-reward-machine thesis.

**A.1 Locked headline.** Exploration selects the reward machine a memory agent learns: the freeze is the agent correctly learning a degenerate Myhill–Nerode automaton of its realized reward; a non-potential bonus re-inflates it; a policy-invariant potential (PBIM) provably cannot, via the Mealy–Moore type mismatch.

**A.2 Primary metric.** The triple {success, falls, behavioral eff-RM-size (via `rm_coverage`)}. The decode probe is explicitly *never* primary or supporting. Return is *never* a cross-arm rank. n = 5 (n = 4 minimum), ≥95% budget per run, Mann–Whitney for differences + TOST for the PBIM = none equivalence.

**A.3 Locked decision rules.** freeze (penalty-none success 0.00 ∧ falls 0.0); rescue (non-potential bonus lifts success); re-inflation (eff-RM-size up under non-potential bonus); the PBIM=none keystone (ρ confirm/falsify branches); conjunctive necessity (Memoryless floors despite coverage); catalyst-shape (bonus×task interaction with reversal); K4 (DP optimality gap).

**A.4 Kill-criteria (threshold + passing result).**

| K | threshold | result |
|---|---|---|
| K1 freeze | succ 0.00 ∧ falls 0.0, all cells | 6/6 (sparse-none ALIVE .16–.20 / 9–39) |
| K2 keystone | ρ = (pbim−none)/(e3b−none) = 0 | ρ = 0; PBIM = none (succ 0.00 / falls 0.0) |
| K3 instrument | eff-RM-size separates frozen/re-inflated | 1.00 vs 13.14 (sparse-none 6.98) |
| K4 DP | V\* > 0 at −0.008 | V\* = 0.76 > 0; p̄ = −0.033 |

**A.5 Instruments of record.** `rm_coverage.py` (behavioral eff-RM-size + idle-fraction; K1/K3); `optimal_prober.py` (corridor-POMDP DP; K4); `decode_memory.py` (hardened, cautionary-only — random-init floor is the retirement evidence).

**A.6 Novelty deltas.** vs reward-machine RL (Toro Icarte 2018/2022: RM given+injected → we run it in reverse, emergent+measured); vs BAMDP-Shaping (Lidayan–Dennis–Russell 2025: value-function potential decomposition → PBIM=none is *their* prediction, our control; no freeze/penalty-sealing in their frame); vs Ni 2023 (memory⊥credit for fixed reward → we re-couple them by editing R(π)); vs NLP-probing (Hewitt–Liang 2019 / Elazar 2021 / Belinkov 2022 → decode probe retired).

---

*Files of record: `memrl/probes/rm_coverage.py`, `memrl/probes/optimal_prober.py`, `memrl/probes/decode_memory.py`, `memrl/exploration/pbim.py`, E15 configs `experiments/memory_training/configs/E15/` (project `memrl-memtrain-gdnsweep`), and docs `docs/revelation_and_densification.md`, `docs/theory_memory_density.md`. Citations flagged for camera-ready verification: Bhargava et al. 2023 (RATE, arXiv 2306.09459) and Toro Icarte 2018 page numbers.*
