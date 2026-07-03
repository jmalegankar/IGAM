# Cluster run plan — remaining experiments (for Bridge)

Everything below is wired into `experiments/memory_training/registry.py` and launches through the
standard flow. **Prereq (once):** commit `gated-lmu`, then rebuild + push the image (it clones
`GIT_REF=gated-lmu` at build time), so the new configs + the reward wrappers
(`MemoryRewardWrapper` in `memrl/envs/minigrid_wrappers.py`; `OraclePotentialWrapper` in
`memrl/envs/memory_gym_wrappers.py`) are baked in.

Standard flow per experiment:
```bash
python -m experiments.memory_training.registry --emit <ID>      # writes configs/ + scripts/
DRY_RUN=1 EID=<E> ENV=<env> ONLY='s0_' k8s/launch-memtrain-jobs.sh   # preview seed-0 wave
EID=<E> ENV=<env> ONLY='s0_' k8s/launch-memtrain-jobs.sh             # launch seed-0 wave
# inspect wandb → then launch the rest (drop ONLY, or stage by seed/density)
```

---

## NEW cluster runs

### EXP-7 + EXP-1 — S13, reward- & metric-matched to MysteryPath  (`memrl-s13-matched`, 360 runs)
The headline gap-closer: the equalization sign is currently n=2 on the *discounted* native reward
(a confound vs MysteryPath's flat +1). This runs S13 under a **flat +1 / binary success\_rate**
matched to MysteryPath, at n=5, both views, plus the S13 **freeze** arm.

- Grid: **common 6-cell zoo** {GRU, LSTM, RetNet, GatedDeltaNet, Mamba2, Memoryless} × {none, e3b\_idm,
  noveld, pbim\_e3b\_idm} × {`sparseV3`, `sparseV7`, `freezeV3`} × 5 seeds (Fable §1: main-text = 6-cell;
  12-cell native zoo stays appendix-only in `memrl-memtrain-s13`).
- Emit + stage:
  ```bash
  python -m experiments.memory_training.registry --emit E1:S13
  # sparse arms first (the amplify-vs-equalize test), seed-0 wave:
  DRY_RUN=1 EID=E1 ENV=S13 ONLY='s0_.*sparseV' k8s/launch-memtrain-jobs.sh
  EID=E1 ENV=S13 ONLY='s0_.*sparseV' k8s/launch-memtrain-jobs.sh
  # then the rest of sparse (seeds 1-4), then the freeze arm:
  EID=E1 ENV=S13 ONLY='sparseV'      k8s/launch-memtrain-jobs.sh
  EID=E1 ENV=S13 ONLY='freezeV3'     k8s/launch-memtrain-jobs.sh
  ```
- **Metric:** `eval/success_rate` (the wrapper emits `is_success` on every arm). Native discounted
  runs stay archived in `memrl-memtrain-s13` for reference.
- **Decision rules:**
  - *EXP-7 (sparse):* cell×bonus interaction term **negative** (equalize) on S13 vs **positive**
    (amplify) on MPG, at n=5 with floor/ceiling censoring. NovelD ≥ E3B as catalyst.
  - *EXP-1 (freeze):* `freezeV3-none` freezes in ≥5/6 core cells (success CI-upper < 0.05 AND
    idle `action_frac_0` CI-lower > 0.15) while `sparseV3-none` stays alive; e3b/noveld rescue;
    `pbim` stays frozen, ρ ≈ 0. *If `sparseV3-none` also collapses, the per-move penalty is too
    large — halve `move_penalty` and re-run.*

### EXP-5 — GatedDeltaNet best-HP α≈0 closure  (`memrl-tiny-exp5`, 20 runs)
Closes the floor-confound on the α≈0 control: E15 showed GDN un-floors to ~0.35–0.40 at lr 1e-3 /
assoc 64, but only with `intrinsic=none`. This adds the E3B contrast at that best HP.
```bash
python -m experiments.memory_training.registry --emit E5
DRY_RUN=1 EID=E5 ONLY='s0_' k8s/launch-memtrain-jobs.sh
EID=E5 k8s/launch-memtrain-jobs.sh          # 20 runs, cheap
```
- **Decision rule:** E3B−none within the α≈0 margin (TOST, |Δ| ≤ 0.05) **while GDN reaches
  ~0.35–0.40** (headroom) and the bonus is active (`bonus_mean` > 0). ⇒ α≈0 is not a floor artifact.

### H-POT — oracle-potential rescue arm  (`memrl-mpg-oraclepot`, 9 runs)
The PBIM-section survival kit (attack A5). MysteryPath **penalty** arm + a task-informed state
potential Φ=−β·d(agent,goal) delivered as PBRS on r_ext (`OraclePotentialWrapper`, `intrinsic=none`).
Tests whether an oracle potential reopens the freeze where PBIM's bonus-derived potential cannot.
Telescoping validity already verified locally (`experiments/analysis/oracle_potential_check.py`:
residual 9e-16, mean|F|≈0.0074 at β=0.02).
```bash
python -m experiments.memory_training.registry --emit HPOT
DRY_RUN=1 EID=HPOT ONLY='s0_' k8s/launch-memtrain-jobs.sh
EID=HPOT ONLY='s0_' k8s/launch-memtrain-jobs.sh     # seed-0 wave (3 runs), then s1_/s2_
```
- **Before the full run:** tune β so mean|F| (logged as `oracle_absF`) matches the sparse-e3b arm's
  delivered bonus (λ·`bonus_mean`) within ~2×; bump β in the HPOT Density if it is >2× low. γ=0.995
  is already pinned to the HP γ (telescoping consistency).
- **Decision rule (both are locked wins):** Φ_oracle **rescues** (success CI-lower > 0 on ≥2/3 cells)
  ⇒ "task-informed potentials can rescue; converting the bonus to a potential destroys its rescue
  power — the value lives in the non-potential, transition-keyed residual." Φ_oracle **does not
  rescue** (all cells CI-overlap 0) ⇒ "neither the potential delivery of the bonus nor an oracle
  potential reopens the freeze." Compare against the `penalty`-none freeze in `memrl-memtrain-mpg`.

---

## ANALYSIS-ONLY — no cluster runs (data already exists)

### EXP-6 — cell×bonus interaction-term test  (FREE)
Run once EXP-7's S13 seeds land. ART-ANOVA (`success ~ cell + bonus + cell:bonus`) on the existing
MPG grid (`memrl-memtrain-mpg`) + the new `memrl-s13-matched` grid; report the `cell:bonus` omnibus,
partial η², and the two planned conjunctive contrasts. Establishes the interaction is real (not two
main effects) and that its sign reverses MPG↔S13. Pure stats pass.

### EXP-9 — PBIM on the alive arm (ρ_alive; the Moore/Mealy keystone)  (Bridge's RM contribution)
**The runs already exist** — `memrl-memtrain-mpg`, `sparse` density, {none, e3b_idm, pbim_e3b_idm},
RetNet + GatedDeltaNet, n=4–5. Pull ρ_alive = (pbim−none)/(e3b−none) per cell **conditional on the
V_int-informative guard**: check `pbim_V_int_mean` / potential-loss and the value-variance are
materially above the frozen-arm level (i.e. the agent IS exploring, so PBIM had room to help). If
the guard passes and ρ_alive ≈ 0 (CI-upper < 0.25), the Moore/Mealy Observation survives the test
built to break it. No new training.

---

## OPTIONAL / needs work

- **MPG backfill to n=5:** several (cell, arm) in `memrl-memtrain-mpg` are n=4. Re-emit E1:MysteryPath
  and launch only the missing seed wave (`ONLY='s4_...'`) for the under-seeded cells.
- **Counterfactual replay** (the one genuinely-new idea from the reviewer pitch): train the bonus arm
  on trajectories collected *without* the bonus, to separate "bonus changes the data" from "bonus
  changes the optimization." **Not wired — needs a code change** (a replay buffer / off-policy data
  path). Infra-first; do not launch until built.

---

## Owner split
- Bridge (cluster + RM): launch EXP-7/EXP-1 + EXP-5 + H-POT; own EXP-9 (ρ_alive) and EXP-6 (stats) —
  these are the reward-machine / Moore–Mealy contributions.
- Local (Spark/laptops): probe analysis, figure generation, the counterfactual-replay infra if we
  decide to build it.

## Local analysis instruments (BUILT + tested this session — run over checkpoints/data, no new training)
- `memrl/probes/instruments.py` — H-INST behavioral batch (I-1 repeat-fall / I-2 frontier-probe /
  I-3 distinct-tiles / I-6 dual-eval + falls/idle/success). Run over MPG n=5 snapshots once pulled:
  `python -m memrl.probes.instruments --run-dir <run> --snapshots 500000,...,20000000 --out inst.csv`.
- `experiments/analysis/registered_stats.py` — the registered contrasts R1/R2/R4/TOST/ρ
  (`--selftest` passes). Feed a tidy grid CSV (env,cell,arm,bonus,seed,success[,auc]) → decision-rule
  verdicts. This is how EXP-7/EXP-1/EXP-5/H-POT reads become claims.
- `memrl/probes/axis_classifier.py` — I-5 random-policy PRM-contingency table (the A2 axis
  definition). Tiny `rm_state` exact-validation passes; MysteryPath goal = agent-contingent (P=0.013).
  `python -m memrl.probes.axis_classifier --env both`. S13 view-row lands with I-4.
- TODO local (additive, no cluster): I-4 (S13 cue-exposure geometry → the S13 axis row + the
  H-KNOB-B dose-response), I-7 (Tiny bonus discriminativeness), H-POT §3.1 Φ non-degeneracy
  diagnostic (needs the pulled PBIM penalty checkpoints).
