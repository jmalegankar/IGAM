# Paper skeleton — "Decodability Is Not Use" (working title)

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
