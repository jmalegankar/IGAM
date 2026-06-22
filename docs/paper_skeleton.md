# Paper skeleton — "Decodability Is Not Use" (working title)

> **⚠️ SUPERSEDED 2026-06-21 by `docs/paper_full.md` ("Exploration Selects the Reward
> Machine a Memory Agent Learns").** The "Decodability Is Not Use" thesis below was
> RETIRED: the decode probe was shown to measure copy-capacity (random-init untrained ≈
> trained), and "decodability ≠ use" is the NLP probing thesis (Hewitt-Liang / Elazar).
> The live paper is the realized-reward-machine framing in `paper_full.md` (freeze = RM
> collapse, Mealy–Moore keystone, behavioral eff-RM-size). This file is kept for
> provenance and can be repurposed as the AAAI ~8pp crop of `paper_full.md`.

*2026-06-19. SUPERSEDES `paper_plan_memory_training.md` (which predates the catalyst /
conjunctive-necessity / realize-before-use pivots). Companion to `preregistration.md`,
`theory_v2_memory_training.md`, `reference_rate_memory_standard`, `project_e1_first_results`.*

Evidence tags used below: **[OK-local]** confirmed locally (low n) · **[PARTIAL]** some
data, not n=5 · **[PEND-cluster]** needs the n=5 grid · **[NEEDS-X]** cheap local/wiring
step named · **[BUILT]** infra ready, unrun.

---

## 1. Thesis (one sentence)

In recurrent RL agents, task-relevant memory is **realized** (encoded and linearly
decodable) *early and even in agents that fail the task*; the bottleneck — and the
variable that discriminates architectures and explains what exploration bonuses do —
is **use**, not encoding. So **decodability ≠ task performance**, and measuring memory
by what is decodable (or by attention maps) is misleading.

## 2. Title / framing options
- "Decodability Is Not Use: Measuring Memory in Recurrent RL Agents"
- "Memory Is Realized Early, Used Late"
- "The Behavioral Catalyst: Exploration Bonuses Change Use, Not Representation"

## 3. Contributions (claim exactly these)

- **C1 — A calibrated realization probe.** Decode the *exact* minimal reward-machine
  state from the frozen recurrent state, with a positive control, shuffled-label
  significance floor, obs-only leakage baseline, linear (Moore-separability) **and**
  MLP probes, on forced-retention (play-phase-only) steps. A rigorous alternative to
  attention-visualization (the RATE/field norm) for "what does the memory hold."
  **[OK-local: hardened + validated on a real LSTM checkpoint — positive control passes,
  obs-baseline 0, linear≈MLP.]**
- **C2 — Realization ≠ use (the headline).** Memory is encoded early and decodable
  *even at 0% success* (GTrXL 9.8/18 bits at 0% success; GatedDeltaNet 9.4 bits at
  15%). Across cells realization *tracks but does not equal* success (LSTM 11.4@.95,
  GRU 7@.6, GDN 9.4@.15). The discriminating axis is **use**, not encoding;
  decodability bounds capacity, not use. **[OK-local; NEEDS early-checkpoint GRU/LSTM
  probe to show realize-before-use is universal; PEND-cluster for n=5.]**
- **C3 — The bonus is a behavioral catalyst + conjunctive necessity.** On the probe,
  **e3b ≈ none** (the bonus does *not* change realization: GRU Δ−0.15, LSTM Δ−0.77);
  its effect is behavioral (drives coverage / idle→0). Solving a sparse memory POMDP
  requires the **conjunction** of bonus-supplied coverage and a memory cell that
  realizes *and* uses: Memoryless+bonus moves maximally yet fails; cell+none stalls.
  Which bonus helps **matches the task's exploration geometry** (NovelD on S13's
  cue-finding; E3B on MysteryPath's spatial trace — the noveld≥e3b reversal).
  **[realization-null OK-local; idle-fraction NEEDS wiring; entanglement PARTIAL (S13
  n=2, MysteryPath deprecated-n3); catalyst-shape PARTIAL (S13 n=2).]**
- **C4 — (Rigor) Apparent architecture rankings are tuning/length effects.** The
  gated-RNN ≫ SSM/DeltaNet "exact-recall" gap at fixed HP is a tuning/length-scaling
  artifact, not encoding capacity: GatedDeltaNet's `step()` is correct, it solves k=4
  copy *faster than GRU*, but floors k=10 at the registry's lr=1e-4. Cell rankings must
  be reported at **per-cell-best HP**. **[k=4 OK-local; k=10 LR×state PEND-cluster.]**

PBIM=none and the freeze fixed point are **demoted to controls/vignettes**, not
contributions: PBIM=none is BAMDP-Shaping's *own* prediction (the non-potential term),
so it is reported as a control that the benefit is non-potential — not as novelty.

## 4. Figures (priority order, with evidence)

- **F1 (method) — the probe + positive control.** LSTM (solving) lag-Δ retention curve
  (pos0 .97 → decay → primacy bump), obs-baseline at chance, linear≈MLP. Establishes
  the instrument is false-negative-safe and leakage-free. **[OK-local n=1 seed.]**
- **F2 (HEADLINE) — realization ≠ use.** Scatter: realized bits vs eval-success across
  cells × checkpoints. Anchors: GTrXL 9.8 @ 0%, GDN 9.4 @ .15, LSTM 11.4 @ .95.
  Shows the dissociation + that the bonus moves the *use* axis, not the *realization*
  axis. **[OK-local; NEEDS early GRU/LSTM ckpt; PEND-cluster n=5.]**
- **F3 — behavioral catalyst / realization-null.** e3b vs none on the probe (≈equal on
  solving cells) paired with the *behavioral* panel (idle-fraction → 0 under the
  bonus). **[realization OK-local; idle NEEDS `eval/idle_frac` wired.]**
- **F4 — conjunctive necessity / entanglement.** Memoryless floors & is bonus-invariant
  (S13: 0.35 across none/e3b/noveld) + Memoryless+bonus moves-but-fails (MysteryPath)
  + cell+none stalls. The 2×2 of {coverage?, realization+use?}. **[PARTIAL.]**
- **F5 — catalyst-shape matches the task.** noveld≥e3b reversal: NovelD solves S13-view3
  broadly (~0.98 across cells); E3B carries MysteryPath. **[PARTIAL: S13 n=2.]**
- **F6 — cell rankings are tuning.** GDN solves k=4 (faster than GRU) but floors k=10 at
  lr=1e-4; ranking at per-cell-best HP. **[k=4 OK-local; k=10 PEND-cluster.]**
- **Appendix:** PBIM=none control; freeze fixed point (vignette, n=5 at −0.008);
  env-suite table + justified exclusions vs RATE; S13 view-size knob (3 vs 7).

## 5. Section outline

1. **Intro — the twin illusions.** (i) bonuses help by densifying value;
   (ii) good memory = decodable memory. We refute both: the benefit is behavioral
   (C3), and decodability ≠ use (C2).
2. **The realization probe (method).** C1 — the calibrated instrument vs attention-viz.
3. **Realization ≠ use.** C2, F1–F2 — the headline.
4. **The behavioral catalyst + conjunctive necessity.** C3, F3–F5.
5. **Cell rankings are tuning, not capacity.** C4, F6.
6. **Related work.** RATE / BAMDP-Shaping / Ni 2023 / Henaff–Taiga (see §7).
7. **Limitations & conclusion.** Core is discrete+vector; MiniWorld-Sign is the 3D
   extension; per-env scope honesty.

## 6. Gap-check

**vs the pre-registration (`preregistration.md`):**
- positive control + shuffled floor + obs-baseline + MLP eff-RM — **DONE** (probe hardened).
- n=5 + Mann-Whitney + TOST equivalence — **[PEND-cluster].**
- idle-fraction as a first-class eval metric — **[NEEDS wiring].**
- λ-sweep (P2 / Ni delta) — **[PEND-cluster].**
- Autoencode 2nd register env (probe-GT, α≈0) — **[BUILT, unrun].**
- Battleship (2nd α>0, generality) — **[PEND-cluster].**
- α-taxonomy demoted to descriptive — framing **DONE** (writing).

**vs the top-conference rigor bar** (thoroughness, a real benchmark suite, multi-seed
significance — accepted memory papers like RATE merely *exemplify the level*; we are not
positioning against RATE, it is a different genre):
- breadth: discrete embodied (**S13**, **MysteryPath**) + symbolic register (**TinyReproduce**,
  **Autoencode [BUILT]**) + a search env (**Battleship**) + **3D pixel (MiniWorld-Sign [BUILT]
  — candidate keystone, see §10)**.
- multi-seed + significance (n=5, TOST/Mann-Whitney): **[PEND-cluster]** — this is the
  rigor gap that most determines acceptance.
- excluded-env rationale stated up front (ViZDoom-2Color α≈0; Memory-Maze coverage-
  neutered; T-Maze PPO-degenerate) — a stricter coverage×realization criterion.

## 7. Novelty positioning (the deltas)

- **vs the field's memory-measurement norm:** memory in RL agents is typically
  "analyzed" by attention maps or by task performance — use-agnostic, decodability-style
  proxies. We show **decodability ≠ use** with a calibrated probe → reframes how the
  field measures memory. (This is a *methodological* novelty, not a positioning against
  any one paper.)
- **vs BAMDP-Shaping (ICLR 2025):** PBIM=none is *their* prediction (benefit = the
  non-potential term) → our **control**, not headline. Realize-vs-use is outside their
  frame.
- **vs Ni 2023:** they decouple memory ⊥ credit-assignment; we add the realization-vs-use
  dissociation and the bonus-as-toggle on coverage. Different axis.
- **vs Henaff 2023 / Taiga 2020:** "a bonus changes behavior" is the expected default;
  the **measurement framework + the dissociation** is the novelty.

## 8. Status ledger (the honest map)

| bucket | items |
|---|---|
| **OK-local (low n)** | probe validated; realization-null GRU/LSTM; realize-before-use (GTrXL 9.8@0%); Memoryless floor (S13); NovelD-catalyst (S13 n=2); GDN solves k=4 / floors k=10@lr1e-4; GDN code correct |
| **PEND-cluster** | n=5 E1 grid (master gate); PBIM n=5 + equivalence; λ-sweep; Battleship; GDN k=10 LR×state; S13 finish + dedupe |
| **NEEDS (cheap)** | early-ckpt GRU/LSTM probe (confirm realize-before-use universal); wire `eval/idle_frac`; run Autoencode |
| **BUILT-shelved** | MiniWorld-Sign (3D generality) |
| **WRITING** | related-work deltas; reframe `theory_v2` to decodability≠use |

## 9. The single biggest risk

Realization saturates *early and cheaply* (GTrXL 10 bits at 0%), so the realization-null
(e3b≈none) and even the cell-realization comparison may be **low-information** — the
action is all in **use**. Mitigation: make the paper *about* that (C2 is the headline,
not a footnote); carry the behavioral claims on idle-fraction/coverage (C3), and the
architecture claims on per-cell-best-HP success (C4), not on decodability alone.

*Update (2026-06-21): this risk MATERIALIZED.* The decode probe was shown to measure
**copy-capacity, not learned memory** — random-init untrained nets decode about as much
as trained ones (GRU random 7.03 ≈ trained 7.04 bits; GTrXL random 9.4 ≈ trained 9.8),
because the probe is teacher-forced. "Decodability ≠ use" / "realize-before-use" is
therefore **retired** (it is the NLP-probing thesis — Hewitt–Liang 2019 control tasks,
Elazar Amnesic-Probing 2021, Belinkov 2022 — and our correlational version is weaker).
The headline has moved to the **realized reward-machine** thesis, carried by the
confound-free behavioral instrument (`rm_coverage`), not the probe. The decode probe is
kept only as a hardened **cautionary methods result** (+ the random-init floor). The
re-filed registration that locks this new headline is Appendix A below; it re-files
`preregistration.md` (2026-06-16) and the C1/C2 "Decodability Is Not Use" framing of
§§1–9 above, which predate the probe collapse.

---

## Appendix A. Pre-registration (re-filed)

**Re-filed 2026-06-21. This document re-files the earlier pre-registration**
(`docs/preregistration.md`, filed 2026-06-16) **and supersedes the "Decodability Is Not
Use" headline of `paper_skeleton.md` §§1–9 / `paper_plan_memory_training.md`.** The prior
prereg's mechanism leg ("S1 behavioral, not representational," carried by the decode /
lag-Δ retention probe — its §3 S1b/S1c and §7 item 3) is **withdrawn**: the decode probe
was shown to measure copy-capacity, not learned memory (random-init untrained nets decode
≈ as much as trained — GRU random 7.03 ≈ trained 7.04 bits; GTrXL random 9.4 ≈ trained
9.8 — because it is teacher-forced). The headline moves from a **narrower freeze-only /
decodability claim** to the **realized reward-machine** thesis. **This move is documented,
not silent** (see the honesty rider, §A.1). Everything below is locked and is the
governing registration of record.

### A.0 Why this re-filing exists (honesty rider)

The earlier prereg deliberately committed (its §0) to flagging any post-unblinding change
of headline/sign/threshold as exactly that — a change, not a silent edit. We invoke that
clause here. What changed and why:

1. **The decode/realization probe collapsed under its own control.** Its hardened
   random-init floor (the floor *we* added to `decode_memory.py`) showed untrained nets
   decode ≈ as much as trained ones, because the probe is teacher-forced. So
   "decodability ≠ use" / "realize-before-use" measures **copy-capacity**, not learned
   memory, and is **retired**. It is also the NLP-probing thesis already (Hewitt–Liang
   2019 control tasks; Elazar *Amnesic-Probing* 2021, who did it *causally*; Belinkov
   2022 survey), so our correlational version was weaker. We keep the hardened probe
   (shuffled-label floor, MLP, obs-baseline, play-phase) **only as a cautionary methods
   result + the random-init floor** — it is never a primary or supporting metric.
2. **The freeze-only headline was too narrow.** The freeze fixed point survives clean
   (n=5, K1 below), but its *mechanism* is sharper than "anti-freeze un-sealing": the
   penalty collapses the agent's **realized reward machine** (the Myhill–Nerode automaton
   of its *realized* reward under its own policy; Toro Icarte 2018) to a single absorbing
   do-nothing state, a non-potential bonus **re-inflates** it, and a policy-invariant
   **potential (PBIM) provably cannot** re-inflate it (Mealy–Moore: a potential Φ is a
   Moore object — one value per state — telescopes to 0 return and cannot relabel
   transition-keyed reward, so it cannot move the realized RM `R(π)`). This is the new
   headline (§A.2).

Standing commitments (carried over from the prior prereg §0, still binding): report every
registered test **in the direction it falls** (a null/flip is a result, not a discarded
run); list every excluded run + the excluding rule; never introduce a post-unblinding
grouping/sign/threshold without flagging it exploratory. Graceful degradation is
pre-committed: **freeze + rescue + PBIM=none stand as standalone empirics; the RM-size
instrument never carries the freeze by itself.**

### A.1 The HEADLINE claim (locked — re-filed)

> **Exploration selects the reward machine a memory agent learns.** A recurrent agent
> converges to the **minimal reward machine of its REALIZED reward** (the Myhill–Nerode
> automaton under its *own* policy; Toro Icarte 2018), **not** the specified task. A
> faithful −0.008 fall penalty collapses that realized-reward RM to **one absorbing
> do-nothing state** — the freeze is the agent **correctly learning a degenerate
> automaton**, not failing to learn. A **non-potential** bonus (E3B/NovelD) **re-inflates**
> the realized RM; a policy-invariant **potential** (PBIM) **provably cannot** (Mealy–Moore:
> Φ is a Moore object, telescopes to 0 return, cannot relabel transition-keyed reward →
> cannot move `R(π)`). Measured by **one confound-free behavioral instrument:
> effective-RM-size** = the number of distinct realized-RM states the policy drives
> through.

This is the only object outside the four nearest priors; the novelty deltas are in §A.6.

### A.2 Primary metric (locked)

The primary readout is the triple **{eval success_rate, ep_num_fails (falls),
behavioral effective-RM-size}** — the third measured on-policy by `rm_coverage.py`
(rolls the agent's OWN deterministic policy and counts distinct realized-RM states +
`idle_frac`; it does **not** teacher-force and does **not** decode a frozen state).

- **The decode probe is NEVER a primary or supporting metric.** It is reported only as a
  cautionary methods result (§A.0 item 1).
- Shaped/episodic **return is never a cross-arm ranking metric** (the frozen agent's
  shaped return 0.0 out-ranks a working agent's negative return — return ranks
  doing-nothing above working).
- **Sample:** n = 5 seeds per cell × density × bonus (n=4 accepted where stated); a run
  counts iff it reached **≥ 95 %** of its step budget. Inclusion/exclusion + arm-vs-crash
  survivorship rules carry over unchanged from `preregistration.md` §6.
- **Statistics (unchanged from prior prereg §2):** difference tests = per-cell two-sided
  **Mann–Whitney U** (primary) + Welch's t (secondary), effect size + bootstrap 95 % CI
  (10 000 resamples); equivalence/null tests = **TOST** within a pre-registered margin,
  with the 90 % CI inside ±margin, else "inconclusive (underpowered)" — a null is never
  claimed from p > 0.05. Analysis is **per-cell**; any pooled number is secondary.

### A.3 Locked decision rules (per registered finding)

- **Freeze (F2):** confirmed iff `penalty(−0.008)-none` has success ≈ 0 (CI upper < 0.05)
  **AND** falls → 0 (CI upper < 1 fall/ep) — the absorbing do-nothing diagnostic — while
  **return-matched** `sparse-none` on the same cell stays ALIVE (success > 0 AND
  falls > 0). Required in **≥ 5/6 cells**. *Falsify:* falls stay > 0 under penalty-none ⇒
  "slow learning," not freeze.
- **Rescue:** confirmed iff `penalty-{e3b,noveld}` success ≥ 0.5 × `sparse-{e3b,noveld}`
  (CI lower > 0) on the cells that froze.
- **Re-inflation (F3, the instrument leg):** `eff_rm_size(penalty-none) → 1` (collapse to
  the trivial 1-state machine; CI upper < 2), and `penalty-e3b ≫ 1` (CI-separated above
  both penalty-none and 1). Validated end-to-end on register envs (TinyReproduce /
  Autoencode) where the exact minimal RM is `info['rm_state']` ground truth.
- **Mealy–Moore keystone / PBIM=none (F4):** the recovered-benefit fraction
  `ρ = (pbim−none)/(e3b−none)` (on success **and** on eff-RM-size), per cell, bootstrap
  CI. **Confirmed iff ρ ≈ 0** (CI upper < 0.25 — PBIM recovers < ¼ of the bonus's
  re-inflation), with the TOST "PBIM ≈ none" leg inside margin `δ_eq = 0.25·(e3b−none)`.
  *Falsify:* ρ ≥ 0.75 (CI lower > 0.75) ⇒ a potential DOES re-inflate the realized RM ⇒
  the Mealy–Moore keystone is wrong and we report it. **Validity guard (carry-over):** PBIM
  telescoping `disc_sum` std/|mean| < 0.3 (Φ = V_int is a valid potential) — re-confirm on
  the n=5 runs.
- **Conjunctive necessity (C3/F5):** confirmed iff **Memoryless floors everywhere**
  (`Memoryless+e3b ≤ Memoryless+none`, CI) AND no cell's success under `none` exceeds the
  Memoryless+none CI, with the registered datum that **Memoryless+e3b moves maximally
  (highest falls) yet succeeds 0** = coverage without memory to realize the RM.
- **Catalyst-shape (F6):** which bonus rescues **matches task geometry** — NovelD carries
  S13-view3, E3B carries MysteryPath; the **noveld ≥ e3b reversal on S13** is the
  registered prediction. Reported as it falls.
- **K4 optimality (DP, see §A.4):** `optimal_prober.py` returns `V* > 0` at −0.008 (CI/grid
  over maze sizes) ⇒ freezing is strictly sub-optimal under the *specified* reward ⇒ the
  pathology framing holds. The pathology claim uses **only −0.008**; −0.1 is a *separate*
  rational-freeze regime, never folded in.

### A.4 The four kill-criteria (thresholds AND passing results stated)

All four are **PASS at n=5**. Each is pre-registered as a binary gate; the result is
stated so the file is auditable post-unblinding.

| # | Kill-criterion (threshold) | Result (PASS) |
|---|---|---|
| **K1** | **Freeze.** `penalty(−0.008)-none`: success ≈ 0 AND falls ≈ 0 in ≥ 5/6 cells, while return-matched `sparse-none` stays alive. | **PASS — 6/6 cells:** penalty-none success 0.00 AND falls 0.0 (absorbing do-nothing). Return-matched `sparse-none` ALIVE: success .16–.20, falls 9–39. |
| **K2** | **Mealy–Moore keystone.** A policy-invariant potential does NOT re-inflate the realized RM: `ρ = (pbim−none)/(e3b−none) ≈ 0`. | **PASS — ρ = 0:** `penalty-PBIM` success 0.00 / falls 0.0 = IDENTICAL to none = still frozen; PBIM ≈ none on sparse too (sparse-pbim ≈ 0 vs sparse-e3b .21–.69). PBIM verified a valid telescoping potential (disc_sum std/\|mean\| 0.00–0.27). |
| **K3** | **Instrument separates.** Behavioral eff-RM-size distinguishes collapse from re-inflation (penalty-none → ~1, penalty-e3b ≫ 1). | **PASS** (20M GatedDeltaNet snapshots via `rm_coverage`): penalty-none eff-RM-size **1.00**, penalty-e3b **13.14**, sparse-none **6.98**. The penalty collapses the RM to 1; the non-potential bonus re-inflates to 13. |
| **K4** | **DP optimality gap.** Exact corridor-POMDP DP: `V* > 0` at −0.008 ⇒ freezing strictly sub-optimal under the specified reward; locate freeze threshold p̄ ≪ −0.008. | **PASS — V\* = 0.76 > 0** (do-nothing = 0); E[falls\|π\*] = 30, P1 ε = \|p\|·E[falls] = 0.24; freeze threshold **p̄ = −0.033** (the penalty must be ~4× harsher for freezing to be optimal). Robust across maze sizes. |

### A.5 Instruments of record

- **`memrl/probes/rm_coverage.py`** — the confound-free behavioral instrument: on-policy
  eff-RM-size + `idle_frac` (rolls the agent's own deterministic policy; no teacher
  forcing, no frozen-state decode). **Carries K1/K3.** Register envs use exact
  `info['rm_state']`; MysteryPath uses the behavioral path/off-path knowledge-grid proxy
  named by `theory_v2`.
- **`memrl/probes/optimal_prober.py`** — corridor-POMDP exact DP. **Carries K4.**
- **`memrl/probes/decode_memory.py`** — hardened decode probe (shuffled-label floor, MLP,
  obs-baseline, play-phase, random-init floor). **CAUTIONARY METHODS RESULT ONLY** — never
  a primary/supporting metric (§A.0 item 1).

### A.6 Novelty deltas (locked positioning)

- **vs BAMDP-Shaping (Lidayan–Dennis–Russell, ICLR 2025):** they decompose a *fixed*
  reward into potential + non-potential; **PBIM = none is THEIR prediction**, so it is our
  **control**, not headline. They have no penalty-sealing / freeze prediction and reason
  about value functions, not reward machines.
- **vs Ni 2023 (NeurIPS):** they show memory ⊥ credit-assignment for a *fixed* reward; we
  **couple** them by editing the *realized* RM.
- **vs reward-machine RL (Toro Icarte 2018; JAIR 2022):** they **assume the RM is GIVEN**
  and inject it; we run it **in reverse** — the RM is *emergent* (the realized-reward
  automaton) and we **measure** it collapse / re-inflate.
- **vs NLP-probing (Hewitt–Liang 2019; Elazar 2021; Belinkov 2022):** we **retire** our
  decode probe (it is their thesis, and weaker in our correlational form) and keep it only
  as a cautionary methods result.
