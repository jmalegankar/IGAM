# Paper plan — "Exploration Bonuses Train Memory, They Don't Explore"

*AAAI target. Drafted 2026-06-12. Companion to `theory_v2_memory_training.md` and
`project_aaai_review_findings`. This is the spine, the figures, the experiment
grid, and what to cut. Be ruthless about scope.*

---

## 1. Title / framing options

- **"Exploration Bonuses Train Memory, They Don't Explore"** (preferred — sharp,
  falsifiable, memorable)
- "Credit, Not Curiosity: Why Exploration Bonuses Help Recurrent Agents on Sparse
  Memory Tasks"
- "The Bonus Trains the Memory: A Mechanistic Account of Intrinsic Reward in POMDPs"

## 2. One-sentence thesis

*On sparse partially-observed memory tasks an episodic novelty bonus's dominant
effect is training the recurrent memory — via a behavioral credit-assignment
channel, not exploration — which is why it cannot substitute for memory, becomes
redundant under dense reward, and rescues a memory that a faithful penalty has
collapsed to the empty statistic.*

## 3. Contributions (claim exactly these)

1. **Mechanism:** the bonus has no architectural path to the policy's memory
   (verified); its benefit is *credit-to-memory-writes*, decomposed into
   densification (P3a, isolated by PBIM) and distribution-shift/curriculum (P3b).
2. **Direct measurement:** a *memory-decodability probe* showing the bonus trains
   the representation — one curve unifies the sparse speed-up, the freeze, and the
   rescue.
3. **The freeze fixed point:** a faithful, *approximately* optimum-preserving
   penalty self-seals a recurrent learner into the empty-statistic do-nothing
   fixed point; the same bonus acts as anti-freeze. Sanctuary condition tested
   within-env.
4. **Corrected credit theory (P1–P2):** optimum-preservation is approximate with
   a measured bound; sparse-credit failure is a GAE-horizon (λ, p(success)), not a
   TBPTT-truncation, phenomenon — with the λ-sweep as the clean test.

> Do **not** claim: the α-taxonomy as a theory; "memory tasks need sparse
> rewards"; exact optimum preservation; "e3b ≫ noveld"; the SCDP sealing chain as
> a contribution (cite it as background only).

## 4. Section outline

1. **Intro** — the puzzle: bonuses "help exploration" on memory tasks, but these
   tasks aren't exploration-hard in the combinatorial sense. Claim: they're
   *credit-hard*, and the bonus is doing credit assignment to memory. 4 contribs.
2. **Setup & background** — POMDP, recurrent PPO, TBPTT/GAE; revealing POMDPs
   (Jin/Liu); BAMDP shaping (Lidayan); Ni et al. memory⊥credit.
3. **Theory** — D1/D2; P1 (approx optimum, with bound); P2 (GAE-horizon credit);
   P3 (behavioral memory-training, densify vs curriculum); P4 (freeze fixed point,
   sanctuary). Short, proposition-with-cited-lemma style.
4. **The bonus trains memory (core)** — decodability probe across arms/steps. The
   centerpiece figure. PBIM ablation adjudicates the channel. Entanglement +
   capacity gradient.
5. **The freeze fixed point** — fallpenalty results (success + falls→0 diagnostic),
   rescue, within-env sanctuary toggle, decodability of the frozen vs rescued
   memory.
6. **Mechanism checks** — λ-sweep (P2); redundancy under dense reward
   (MortarMayhem, return-matched); the retention contrast (S13: bonus does little).
7. **Related work** — differentiate BAMDP shaping, Ni, Henaff, Taiga, EIPO.
8. **Limitations** — grid-worlds only (name Memory Maze as extension); approximate
   optimum; n=5; behavioral-channel only (entangled arm as forward-looking recipe).

## 5. Figures (in priority order)

- **F1 (centerpiece): decodability vs training step**, 4 lines — sparse-none,
  sparse-e3b, penalty-none (flat at chance), penalty-e3b (recovers). One panel per
  cell or pooled. *This is the paper.*
- **F2: freeze/rescue** — success_rate and ep_num_fails (the falls→0 signature)
  for sparse/penalty × none/e3b, 5 cells, n=5. Bars + the falls diagnostic.
- **F3: PBIM adjudication** — raw-e3b vs PBIM-e3b vs none, sparse MysteryPath +
  MortarMayhem. Shows densification-vs-curriculum split.
- **F4: mechanism** — λ-sweep (e3b−none vs λ), and the redundancy panel
  (e3b−none on sparse vs aligned-dense, MortarMayhem return-matched).
- **F5: entanglement / capacity gradient** — bonus benefit vs cell capacity
  (Memoryless at the floor; weak→strong slope).
- **F6 (appendix): within-env sanctuary toggle** — freeze with sanctuary, no
  freeze without.

## 6. Experiment grid (be disciplined)

**Cells (6):** GRU, LSTM (weak); GatedDeltaNet, RetNet, Mamba2 (strong);
Memoryless (floor). GTrXL → appendix only. Report per-cell param counts.

**Envs (3 core + 1 stretch + 1 diagnostic):**
- MysteryPath-Grid — headline (freeze/rescue, decodability, 3×, entanglement).
- MiniGrid-MemoryS13 — retention contrast (predict bonus≈neutral; the e3b<noveld
  reversal is a *feature*: front-loaded revelation).
- MortarMayhem-Grid — the only exactly return-matched density toggle (redundancy).
- SearingSpotlights — STRETCH, **only after the MultiDiscrete E3B fix**; gives the
  no-sanctuary freeze boundary. Cut if it slips.
- Tiny fixed maze — DP optimal-prober (P1 bound) + sanctuary toggle (P4).

**Arms:** {none, e3b_idm, pbim_e3b_idm} everywhere; +{penalty} on MysteryPath/SS;
+{aligned-dense} on MortarMayhem; +{entangled-aux} on MysteryPath only (secondary).

**Seeds:** n=5 everywhere reported with significance (Mann-Whitney / Welch);
state seed counts and crash-exclusions explicitly.

**New runs needed (priority order):**
1. Snapshots on existing + new runs (`--snapshot-steps 500000,2000000,5000000,10000000`)
   → enables F1. *Cheap: re-run or fine-tune from latest with snapshots on.*
2. PBIM-e3b: 6 cells × {MysteryPath sparse, MortarMayhem aligned} × 5 seeds.
3. λ-sweep: GRU × λ∈{0.8,0.9,0.95,0.99,1.0} × {none,e3b} × MysteryPath × 3 seeds.
4. MortarMayhem 5-seed completion (have seed-0).
5. Tiny-maze DP + sanctuary toggle (cheap, CPU).
6. SS re-run after the e3b reshape fix (stretch).
7. Entangled-aux arm (secondary, confirmation + recipe).

## 7. The decodability probe protocol (pre-register this)

- **Bank:** fixed behavior policy (uniform-random primary; a reference checkpoint
  as sensitivity), identical across arms → matched coverage (the key control).
- **Latent:** running knowledge grid (confirmed on/off-path) — primary; full path
  map — secondary (extrapolation).
- **Probe:** linear (headline) + small MLP (capacity check); class-balanced;
  80/20 split; metric = mean per-visited-cell balanced accuracy.
- **Decision rule (filed before unblinding):** if F1 shows
  sparse-e3b > sparse-none and penalty-none ≈ chance while penalty-e3b > chance,
  C1 is confirmed. If linear≈chance but MLP separates, report as "nonlinear memory"
  (weaker form). Have F2+F3 as the fallback paper if F1 is null.

## 8. Risks & rebuttal prep

- **F1 confounded by coverage** → matched-bank control; report both behavior banks.
- **PBIM shows densification fully explains it** → reframe to "credit-assignment,
  not exploration" (still beats BAMDP-Shaping on the memory axis + freeze); keep F2.
- **"This is just an auxiliary task"** → the headline bonus never touches memory
  weights (verified); the entangled arm is explicitly the *contrast*, not the claim.
- **"Only grid-worlds"** → 3 memory types (spatial/retention/sequence) + name
  Memory Maze as the extension; scope honestly.
- **Reviewer reproduces GTrXL counterexample to P2** → P2 exempts cache-attention
  cells explicitly; k-sweep is GRU-only.
- **Data hygiene** → fix the 20M no-none-baseline (run none at 20M or demote
  headline to 10M 3.03×); de-dup crashed-name runs; re-select shared HPs with a
  none-arm sweep or report the e3b-selection bias.

## 9. Pre-submission fix checklist (from the review)

- [ ] Fix E3B MultiDiscrete reshape bug (`e3b_module.py`); re-run SS or cut it.
- [ ] Fix done-step bonus misattribution (zero bonus at done / attribute to new ep).
- [ ] Replace exact lemma with P1 (+ measured bound); rewrite T3.
- [ ] Rewrite T1 → P2; run λ-sweep.
- [ ] α-taxonomy → one descriptive paragraph (or pre-registered α_ref protocol).
- [ ] 20M none baseline (or demote headline); report seeds/crashes per cell.
- [ ] Per-cell param counts; GTrXL → appendix.
- [ ] Cut "e3b ≫ noveld"; report the S13 reversal honestly.
- [ ] Adopt/ cite "revealing POMDP"; cite BAMDP Shaping + Ni as primary related.
