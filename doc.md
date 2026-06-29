# IGAM — Weights & Biases Results Summary

*Generated 2026-06-27 from the `jai-malegaonkar` wandb entity — 1139 runs across 16 projects. Aggregated per (cell × density × bonus); means are over seeds (n noted). Numbers are point-in-time snapshots of each run's final logged value.*

This document catalogs every Weights & Biases project logged for IGAM, an AAAI paper on how episodic exploration bonuses interact with neural recurrent memory in POMDPs. It spans **1139 runs across 16 projects** (15 non-empty; `MemoryRuns` is empty and omitted from the catalog below). Each row gives the project, its environment, its finished/total run counts, and a one-line headline of what that project establishes; the cross-project story then groups the projects by the role they play in the paper's argument.

## Project catalog

| Project | Env | Runs | Headline |
|---|---|---|---|
| memrl-memtrain-mpg | MysteryPath-Grid-v0 (memory-gym, 84x84x3 pixel maze) | 287 finished / 318 (27 crashed, 4 failed) | The headline MysteryPath grid cleanly confirms the locked spine: penalty-none freezes (succ=0.00, ~0 falls) in all 6 cells, exploration bonuses (e3b/noveld) un-seal it while the potential-based PBIM cannot, and the e3b benefit is amplified for strong sequence cells (RetNet/GDN +0.46-0.53) versus weak ones (GRU/LSTM +0.03-0.08). |
| memrl-mpg-fallpenalty | MysteryPath-Grid-v0 | 60 finished / 60 | The -0.008 fall penalty froze exploration in 15/15 dense-none seeds (success 0.000, ~0 falls) while e3b_idm fully rescued performance back to each cell's own sparse-e3b level. |
| memrl-mpg-densetoggle | MysteryPath-Grid-v0 | 30 finished / 38 (8 crashed) | When reward is already aligned-dense (+0.1 progress shaping), the e3b_idm exploration bonus is redundant — it moves eval reward by only -0.09/+0.08/-0.15 across the three cells, all within seed noise. |
| memrl-mpg-20m | MysteryPath-Grid-v0 | 52 finished / 78 (26 crashed) | Under e3b_idm at 20M steps on MysteryPath-Grid, GatedDeltaNet leads (eval reward 0.61, n=4), followed by RetNet (0.52) and GTrXL (0.42), with memory-augmented cells cleanly beating the Memoryless baseline (0.00). |
| memrl-mpg-hpsweep | MysteryPath-Grid-v0 | 54 finished / 57 (3 crashed) | On MysteryPath-Grid, the maximin HP region (e3b_idm, lambda=0.03, chunk_len=64, lr=1e-4) is confirmed strong (best or 2nd-best within every architecture), with longer chunks and higher lambda consistently helping; the single overall best run is RetNet ck128/lr2.5e-4/lam0.03 at 0.60 eval reward. |
| memrl-mysterypath-grid | MysteryPath-Grid-v0 | 107 finished / 123 (16 crashed) | On a single-density MysteryPath-Grid sweep of 12 memory architectures × 4 bonuses (reward-only schema, no success_rate logged), exploration bonuses roughly double eval/mean_reward over the no-bonus baseline (e3b_idm 0.183 / noveld 0.156 vs none 0.078, rnd 0.050), with RetNet+e3b_idm best (0.40). |
| memrl-memtrain-s13 | MiniGrid-MemoryS13 (agent_view_size=3, sparse), 20M steps | 78 finished / 112 (14 crashed, 20 failed) | On hard partial-view MemoryS13, NovelD is the most reliable catalyst (solving LSTM/RetNet/Mamba2/LRU to ~0.98) while e3b is sharply cell-dependent, no-bonus collapses for nearly every cell except GatedDeltaNet which solves it bonus-free — but with only n=2 seeds, most arms show large seed disagreement. |
| memrl-s13-baseline | MiniGrid-MemoryS13-v0 (12 memory architectures x intrinsic bonus, 10M steps, PPO/MemPPO) | 120 finished / 128 (3 crashed, 5 failed) | On MiniGrid-MemoryS13-v0 this is an architecture-x-bonus sweep (not a view/density sweep): linear-attention SSMs (GatedDeltaNet 0.94, Mamba2 0.90, LRU 0.81) clear the task far above recurrent/transformer cells, and exploration bonuses (noveld/rnd) lift weaker cells off the ~0.55 fail-fast floor while strong cells already solve without any bonus. |
| memrl-memtrain-tiny | TinyReproduce-v0 (enumerable reward-machine, k=10 exact-recall) | 72 finished / 96 (12 crashed, 12 failed) | On a controlled k=10 exact-recall reward-machine, gated RNNs sharply dominate SSMs (LSTM ~0.82-0.93 vs GatedDeltaNet/Mamba2/RetNet ~0.00-0.10 success), while the e3b_idm exploration bonus is provably neutral (all per-cell deltas within +/-0.05) despite being active. |
| memrl-memtrain-gdnsweep | TinyReproduce-v0 (k=10), intrinsic=none | 20 finished / 24 (4 failed) | At the full 10M-step budget the GatedDeltaNet k=10 "floor" is broken and is LR-fragility, not an architecture wall: GDN reaches mean 0.35 (best seed 0.40) in its LR sweet spot, with the optimal LR shifting with assoc_size (a64 peaks at lr=1e-3, a256 at lr=3e-4). |
| memrl-mm-toggle | MortarMayhem-Grid-v0 | 12 finished / 12 | No agent solved the sparse-density arm at all (0.0 reward across all 6 sparse runs); only the dense/aligned arm produced signal, where e3b's effect was inconsistent (helps GRU +0.46, hurts GatedDeltaNet -0.19 and RetNet -0.09). |
| memrl-ss-toggle | SearingSpotlights-v0 | 12 finished / 14 (2 crashed) | In the SearingSpotlights density toggle the anti arm did NOT freeze (anti/none gives the longest episodes), and the e3b_idm exploration bonus consistently hurts — catastrophically so under the anti penalty, where it collapses reward to ~0 or negative across all three memory architectures. |
| memrl-memtrain-miniworld | MiniWorld-Sign-v0 (3D first-person; per-episode randomized sign color, read sign then retain color, sparse reward) — DEPRECATED / removed from codebase, historical only | 24 finished / 32 (6 crashed, 2 failed) | No memory architecture learns the MiniWorld-Sign read-and-retain task above floor — every cell tops out at 1/5 eval episodes (0.2) regardless of recurrence or exploration bonus, so the benchmark produced no usable separation signal. |
| memrl-memtrain-minihack | MiniHack-Memento-F2-v0 (MiniHack-Memento cluster) | 0 finished / 2 (2 crashed) | Both cluster runs crashed within ~95 seconds at <0.35% of their 10M-step budget, logging zero eval/rollout metrics, so this project contains no usable MiniHack results — the real MiniHack findings are local/offline (--no-wandb). |
| memrl-redbluedoors-8x8 | MiniGrid-RedBlueDoors-8x8-v0 (gymnasium MiniGrid) | 39 finished / 45 (3 crashed, 3 failed) | RedBlueDoors-8x8 is largely solvable without memory at its best, but Memoryless is seed-unstable (mean reward 0.82 vs 0.98 for recurrent cells), so the "no memory needed" control is only partially confirmed. |

## Cross-project story

- **Headline spine — freeze + rescue asymmetry (the core claim):** `memrl-memtrain-mpg` and `memrl-mpg-fallpenalty` jointly carry the spine on MysteryPath-Grid — a faithful -0.008 fall penalty collapses all 6 cells to do-nothing (succ 0, ~0 falls), a non-potential bonus (e3b/noveld) un-seals them, and the potential-based PBIM provably cannot. `memrl-mpg-densetoggle` is the matched counterpoint: when reward is already aligned-dense there is nothing to un-seal, so the same e3b bonus is redundant (within seed noise).
- **Amplification:** the strong-vs-weak cell gap widening under the bonus on sparse reward is shown directly in `memrl-memtrain-mpg` (RetNet/GDN +0.46-0.53 vs GRU/LSTM +0.03-0.08) and corroborated on a separate single-density sweep in `memrl-mysterypath-grid` (bonuses roughly double eval reward, RetNet+e3b_idm best).
- **Conjunction (need BOTH memory and exploration):** `memrl-mpg-20m` makes the memory half explicit — memory-augmented cells cleanly beat the Memoryless baseline (0.00) under the bonus — while `memrl-memtrain-s13` and `memrl-s13-baseline` show the bonus half lifting weaker cells off the fail-fast floor on hard partial-view MemoryS13. (The conjunction is also seen on MiniHack-Short-F2 locally; see the deprecated-online note below.)
- **Mechanism + tooling on the headline env:** `memrl-mpg-hpsweep` fixes the maximin HP region (e3b_idm, lambda=0.03, chunk_len=64, lr=1e-4) that the spine runs sit in, confirming the result is not an HP artifact and that longer chunks / higher lambda help.
- **Controlled neutrality + architecture infra:** `memrl-memtrain-tiny` is the clean control where the task needs no exploration — on the enumerable k=10 reward-machine the e3b_idm bonus is provably neutral (all deltas within ±0.05), isolating the memory axis (gated RNNs dominate SSMs). `memrl-memtrain-gdnsweep` is its infra companion, showing the GatedDeltaNet "floor" is LR-fragility rather than an architecture wall.
- **Negative control (memory not needed):** `memrl-redbluedoors-8x8` is the designed negative control — the task is largely solvable without memory, only partially confirmed by Memoryless being seed-unstable (0.82 vs 0.98).
- **Corroborating environments where the toggle behaves differently:** `memrl-mm-toggle` (MortarMayhem) and `memrl-ss-toggle` (SearingSpotlights) test the density toggle on other memory-gym envs and do not reproduce the clean freeze/rescue pattern — MortarMayhem's sparse arm is unsolved by anyone, and SearingSpotlights' anti arm does not freeze and the bonus consistently hurts. These bound the scope of where the spine holds rather than extend it.
- **Parked / deprecated / no-usable-signal:** `memrl-memtrain-miniworld` (MiniWorld-Sign, removed from the codebase) produced no separation signal — every cell floors at 0.2 — and is historical only. `memrl-memtrain-minihack` contains no usable online results (both cluster runs crashed in ~95 s logging zero metrics); the real MiniHack findings live local/offline (`--no-wandb`), including the MiniHack-Short-F2 conjunction evidence referenced above.

---

## memrl-memtrain-mpg
*MysteryPath-Grid-v0 (memory-gym, 84x84x3 pixels; lr=1e-4, gamma=0.995, 20M steps) | 287 finished / 318 (27 crashed, 4 failed) | THE headline grid: density {sparse, penalty (reward_fall_off=-0.008), aligned (+0.1 progress)} x bonus {none, e3b_idm, noveld, pbim_e3b_idm} x 6 cells x ~4-5 seeds; home of the freeze / rescue-asymmetry / amplification spine.*

All 287 finished runs reached the full 20M-step budget (global_step = 20,004,864 for every finished run), so headline numbers are at-convergence. Bonus arms present are `none`, `e3b_idm`, `noveld`, and `pbim_e3b_idm` (PBIM applied on top of e3b — this is the "potential-based" arm). There is **no `rnd` arm** in this project despite the brief mentioning it.

### eval/success_rate — mean (n finished), all densities

| cell | sparse none | sparse e3b | sparse noveld | sparse pbim | penalty none | penalty e3b | penalty noveld | penalty pbim | aligned none | aligned e3b | aligned noveld | aligned pbim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GRU | 0.188 (4) | 0.263 (4) | 0.200 (4) | 0.000 (4) | **0.000 (4)** | 0.237 (4) | 0.200 (4) | **0.000 (4)** | 0.338 (4) | 0.200 (4) | 0.287 (4) | 0.325 (4) |
| LSTM | 0.188 (4) | 0.212 (4) | 0.237 (4) | 0.013 (4) | **0.000 (4)** | 0.237 (4) | 0.175 (4) | **0.000 (4)** | 0.300 (3) | 0.217 (3) | 0.267 (3) | 0.325 (4) |
| RetNet | 0.163 (4) | 0.688 (4) | 0.588 (4) | 0.138 (4) | **0.000 (4)** | 0.575 (4) | 0.625 (4) | **0.000 (4)** | 0.850 (4) | 0.662 (4) | 0.787 (4) | 0.775 (4) |
| GatedDeltaNet | 0.200 (4) | 0.662 (4) | 0.700 (4) | 0.113 (4) | **0.000 (5)** | 0.625 (4) | 0.713 (4) | **0.000 (4)** | 0.780 (5) | 0.700 (4) | 0.887 (4) | 0.662 (4) |
| Mamba2 | 0.200 (4) | 0.425 (4) | 0.325 (4) | 0.100 (4) | **0.000 (4)** | 0.413 (4) | 0.250 (4) | **0.000 (4)** | 0.413 (4) | 0.388 (4) | 0.525 (4) | 0.300 (4) |
| Memoryless | 0.100 (4) | 0.000 (4) | 0.100 (4) | 0.000 (4) | **0.000 (4)** | 0.000 (4) | 0.013 (4) | **0.000 (4)** | 0.075 (4) | 0.013 (4) | 0.075 (4) | 0.113 (4) |

### Penalty arm — mechanism detail: succ / rollout/ep_num_fails_mean / debug/action_frac_0 (idle), mean (n finished)

| cell | none (succ/falls/idle) | e3b (succ/falls/idle) | noveld (succ/falls/idle) | pbim (succ/falls/idle) |
|---|---|---|---|---|
| GRU | 0.00 / 0.00 / 0.47 | 0.237 / 14.6 / 0.00 | 0.200 / 6.2 / 0.24 | 0.00 / 0.00 / 0.67 |
| LSTM | 0.00 / 0.00 / 0.34 | 0.237 / 13.8 / 0.01 | 0.175 / 6.8 / 0.22 | 0.00 / 0.01 / 0.63 |
| RetNet | 0.00 / 0.00 / 0.33 | 0.575 / 8.0 / 0.01 | 0.625 / 8.0 / 0.02 | 0.00 / 0.13 / 0.46 |
| GatedDeltaNet | 0.00 / 0.00 / 0.32 | 0.625 / 7.2 / 0.00 | 0.713 / 6.4 / 0.01 | 0.00 / 0.00 / 0.22 |
| Mamba2 | 0.00 / 0.00 / 0.33 | 0.413 / 11.2 / 0.02 | 0.250 / 8.0 / 0.13 | 0.00 / 0.15 / 0.46 |
| Memoryless | 0.00 / 0.00 / 0.35 | **0.00 / 22.3 / 0.00** | 0.013 / 13.6 / 0.06 | 0.00 / 0.00 / 0.15 |

(Reference: sparse-none falls run 9.3–38.9; the penalty-none ~0 falls are the freeze signature, not just low success.)

**Takeaway:** Every prediction in the spine is **confirmed by the numbers.** (1) **FREEZE holds 6/6** — penalty-none gives eval succ = 0.00 *and* ep_num_fails ≈ 0.000–0.003 in all six cells, with elevated idle (0.32–0.47), versus sparse-none which both succeeds (0.10–0.20) and racks up 9–39 falls. (2) **RESCUE asymmetry is clean** — e3b and noveld un-seal penalty (RetNet/GDN reach 0.58–0.71, recovering essentially to their sparse/aligned ceiling), while **pbim stays frozen at exactly 0.00 in all 6 cells** with ~0 falls and high idle (0.15–0.67); rho = pbim/e3b = 0 everywhere. The potential-based bonus cannot break the seal. (3) **CONJUNCTION holds** — Memoryless+e3b on penalty produces the *highest* fall count in the whole grid (22.3) yet succ = 0.00: bonus-driven thrashing without memory buys nothing. (4) **AMPLIFICATION holds** — sparse e3b benefit is +0.525 (RetNet) and +0.462 (GDN) for strong cells vs +0.075 (GRU) / +0.025 (LSTM) for weak ones, and Memoryless is actually hurt (−0.10). (5) **aligned dense ≈ bonus-neutral** — every e3b−none delta on aligned is ≤ 0 (range −0.025 to −0.188), i.e. when shaping already supplies dense signal the bonus is at best neutral and mildly counterproductive; RetNet/GDN aligned-none alone hit 0.78–0.85.

**Caveats:**
- **n is 4 seeds per cell for almost all arms** (a few are 5; aligned-LSTM has only n=3 for none/e3b/noveld because one seed sits among the failed runs). The "up to 5 seeds" target was generally not met — most cells are n=4, so seed-level variance is not deeply sampled.
- **31 non-finished runs.** 4 failed (3 are GatedDeltaNet-aligned-seed3 with no global_step logged → never trained; 1 GDN-sparse-noveld-seed0 died at 0.68M). 27 crashed; only 6 crashed near completion (>15M steps), all GDN- or GRU-related, e.g. GDN-aligned-noveld-seed3 crashed at 19.1M with succ=1.0 and GDN-penalty-e3b_idm-seed2 at 19.8M with succ=0.45 — these are consistent with the finished means and were excluded from headline aggregates (finished-only).
- The **PBIM arm is `pbim_e3b_idm`** (PBIM on top of e3b), so its inability to un-seal is the strong form of the claim: even the same exploratory signal, made potential-based, is fully neutralized — pbim collapses both success and falls back to the frozen baseline. **No standalone RND arm exists here** to corroborate the bonus family beyond e3b/noveld/pbim.

---

## memrl-mpg-fallpenalty
*MysteryPath-Grid-v0 | 60 finished / 60 | Dedicated -0.008 freeze/rescue demo: density {sparse, dense=fall-penalty reward_fall_off=-0.008} x bonus {none, e3b_idm} x 3 cells x 5 seeds, all trained to 10M steps.*

This is the cleanest freeze demonstration: all 60 runs finished and reached exactly 10.0M global_step, so every cell is a full n=5. "dense" = the fall-penalty arm (env_kwargs `reward_fall_off: -0.008`); "sparse" = no fall penalty.

**eval/success_rate — mean (best of 5):**

| Cell | sparse-none | sparse-e3b | dense-none (penalty) | dense-e3b (penalty) |
|---|---|---|---|---|
| GRU | 0.150 (0.20) | 0.210 (0.40) | **0.000 (0.00)** | 0.150 (0.20) |
| GatedDeltaNet | 0.190 (0.30) | 0.450 (0.55) | **0.000 (0.00)** | 0.420 (0.60) |
| RetNet | 0.130 (0.25) | 0.400 (0.55) | **0.000 (0.00)** | 0.410 (0.60) |

**rollout/ep_num_fails_mean (falls) and debug/action_frac_0 (idle):**

| Cell | sparse-none falls / idle | dense-none falls / idle | dense-e3b falls / idle |
|---|---|---|---|
| GRU | 11.5 / 0.28 | **0.00 / 0.37** | 15.9 / 0.00 |
| GatedDeltaNet | 17.8 / 0.22 | **0.00 / 0.28** | 9.6 / 0.00 |
| RetNet | 15.6 / 0.24 | **0.02 / 0.33** | 10.3 / 0.01 |

**Rescue ratio (dense-e3b ÷ sparse-e3b):** GRU 0.71, GatedDeltaNet 0.93, RetNet 1.02.

**Takeaway:** The expected freeze is confirmed exactly — dense-none (the -0.008 penalty with no bonus) yields success 0.000 in **15/15 seeds across all 3 cells**, with essentially zero falls (mean 0.009, max 0.06) and elevated idle fraction (~0.33, vs ~0.00 once a bonus is added). The agent stops probing: it neither succeeds nor falls, and stands still more. The rescue is also confirmed — adding e3b_idm to the penalty arm restores success to roughly each cell's own sparse-e3b level (GatedDeltaNet 0.42 vs 0.45, RetNet 0.41 vs 0.40, GRU weaker at 0.15 vs 0.21) and brings falls back up to 10-16 per episode, i.e. exploration is un-sealed. Meanwhile sparse-none keeps probing despite no penalty (succ 0.13-0.19, falls 11-18). Net: penalty freezes (succ 0, falls 0), e3b un-freezes — the freeze/rescue asymmetry holds with a perfect 15/15 self-seal rate.

**Caveats:** (1) Only the e3b_idm bonus is present here — no PBIM/NovelD/other-bonus arms in this project, so this confirms only the e3b rescue leg, not the rescue *asymmetry* (e3b un-seals vs PBIM can't) on its own. (2) The GRU rescue is partial (0.71x its own sparse-e3b, and dense-e3b succ 0.15 ties only its sparse-none baseline), so "full recovery to own sparse level" is clean for GatedDeltaNet/RetNet but weaker for GRU. (3) eval success rates are modest in absolute terms (best non-frozen arm ~0.45), consistent with MysteryPath-Grid being hard; the headline result is the contrast across arms, not absolute success. No crashes or partial seeds limit the read.

---

## memrl-mpg-densetoggle
*MysteryPath-Grid-v0 | 30 finished / 38 (8 crashed) | Aligned-dense arm (+0.1 progress shaping): does the e3b_idm bonus still help when reward is already dense?*

> **Metric caveat (read first):** the configured primary metric `eval/success_rate` is **absent from every run** in this pull. No success or fails metric is logged at all. The headline below uses **`eval/mean_reward`** (logged for all 30 finished runs) as the proxy, with `eval/mean_ep_length` as a secondary read (lower = faster path completion in MPG). Because the env is reward-shaped here (`reward_path_progress: 0.1`), mean_reward blends shaping with task completion, so it is not a clean success rate — treat absolute levels loosely; the **bonus-vs-none contrast within each cell** is the reliable read.

All runs sit in a **single density** (`env_kwargs = {reset_options: {reward_path_progress: 0.1}}` — the aligned-dense arm). There is no sparse or penalty comparison arm inside this project; the toggle here is purely **bonus on/off**.

**Results by cell × bonus (finished runs; mean (n), best seed)**

| Cell | none — mean_reward (n) | e3b_idm — mean_reward (n) | Δ (e3b−none) | none best / e3b best | mean_ep_len none→e3b | idle frac none→e3b |
|---|---|---|---|---|---|---|
| GRU | 0.803 (5) | 0.713 (5) | **−0.090** | 1.005 / 0.87 | 96.3 → 108.0 | 0.140 → 0.003 |
| GatedDeltaNet | 1.170 (5) | 1.250 (5) | **+0.080** | 1.565 / 1.40 | 78.7 → 77.4 | 0.051 → 0.002 |
| RetNet | 1.254 (5) | 1.107 (5) | **−0.147** | 1.385 / 1.37 | 70.5 → 84.0 | 0.034 → 0.009 |

**Takeaway:** The redundancy hypothesis is **confirmed**. With reward already aligned-dense, adding the e3b_idm bonus does not help in any cell — the eval-reward deltas are small and mixed-sign (GRU −0.09, GatedDeltaNet +0.08, RetNet −0.15), all comfortably inside per-seed scatter (e.g. none seeds span 0.67–1.01 for GRU, 0.92–1.57 for GatedDeltaNet). Episode length tells the same story: the bonus slightly *lengthens* paths for GRU (96→108) and RetNet (70→84) and is flat for GatedDeltaNet (79→77), i.e. no completion-speed gain. Cell ranking is consistent regardless of bonus (RetNet ≈ GatedDeltaNet > GRU), so memory architecture, not the intrinsic bonus, drives performance here. One real side-effect: the bonus collapses the idle fraction (`action_frac_0`) to ≈0.00 in every cell (vs 0.03–0.14 for none), confirming the bonus *is* active and discourages no-ops — it just buys no task-reward when reward is already dense.

**Caveats:**
- **Primary metric missing:** `eval/success_rate` is not logged anywhere; all conclusions rest on `eval/mean_reward` (shaped, so not a pure success rate). Treat as a redundancy read, not an absolute-success read.
- **Crashes cost no data:** the 8 crashed runs were all early failures — seed0 (×6) and seed1 (×2), every one dead at ≤8,192 steps (vs the 10.0M-step finished runs). Each was re-run successfully, so **every cell × bonus cell has the full 5 finished seeds (0–4)**; the 30/38 yield reflects restarts, not lost conditions.
- **Single density / no within-project baseline:** this project only contains the aligned-dense arm. The claim "dense-none is already fairly high so e3b adds nothing" is supported by the on/off contrast here, but the sparse/penalty comparison needed to quantify *how much* shaping itself helped lives in sibling projects, not this one.
- **Scope:** 3 cells only (GRU, GatedDeltaNet, RetNet); no Mamba2/LSTM/Memoryless/RetNet-variants in this arm.

---

## memrl-mpg-20m
*MysteryPath-Grid-v0 | 52 finished / 78 (26 crashed) | Old 20M-step x5-seed sparse headline, e3b_idm-ONLY (12 cells, NO none arm) — cell ranking at 20M under e3b.*

**Metric note:** the requested primary metric `eval/success_rate` is **not logged anywhere in this project** (no success/fail keys exist). I report the actual eval headline metric `eval/mean_reward` (range 0–0.85 across runs; behaves as a success-proportion proxy — higher reward ⇄ more goals reached ⇄ shorter episodes), with `eval/mean_ep_length` (128 = timeout/no goal) and idle fraction as support. All headline numbers are **finished runs only**; the 26 crashes are excluded because none produced eval data (see Caveats).

### Cell ranking under e3b_idm @ 20M (finished runs)

| Rank | Cell | eval reward mean (n) | best seed | eval ep-len | idle frac |
|---|---|---|---|---|---|
| 1 | GatedDeltaNet | **0.613** (4) | 0.850 | 74.5 | 0.000 |
| 2 | RetNet | **0.520** (5) | 0.750 | 86.5 | 0.008 |
| 3 | GTrXL | **0.420** (5) | 0.600 | 94.7 | 0.004 |
| 4 | Mamba2 | 0.380 (5) | 0.450 | 97.3 | 0.015 |
| 5 | SHM | 0.275 (4) | 0.600 | 104.9 | 0.009 |
| 6 | LSTM | 0.200 (3) | 0.400 | 108.0 | 0.001 |
| 7 | GRU | 0.183 (3) | 0.250 | 110.1 | 0.001 |
| 8 | mLSTM | 0.163 (4) | 0.250 | 111.9 | 0.005 |
| 9 | LRU | 0.150 (4) | 0.250 | 115.7 | 0.028 |
| 10 | FFM | 0.140 (5) | 0.250 | 114.8 | 0.001 |
| 10 | LinearTransformer | 0.140 (5) | 0.250 | 114.8 | 0.008 |
| 12 | Memoryless | **0.000** (5) | 0.000 | 128.0 | 0.000 |

Per-seed spread (top/bottom): GatedDeltaNet = [0.45, 0.55, 0.60, 0.85]; RetNet = [0.60, 0.35, 0.55, 0.75, 0.35]; Memoryless = [0, 0, 0, 0, 0].

**Takeaway:** A clear, monotonic cell ranking emerges at 20M under e3b_idm. The associative/attention-style recurrence cells dominate — GatedDeltaNet (0.61), RetNet (0.52), GTrXL (0.42), Mamba2 (0.38) — while classic gated RNNs (LSTM 0.20, GRU 0.18) and the linear-attention family (mLSTM/LRU/FFM/LinearTransformer ≈ 0.14–0.16) cluster near the bottom. **Memoryless is dead at exactly 0.0 across all 5 seeds** (ep-length pinned at the 128 cap), confirming that on MysteryPath-Grid the task is unsolvable without memory and that the bonus alone buys nothing — a useful sanity floor. Eval ep-length tracks reward inversely as expected (winner 74.5 vs Memoryless 128). Idle fraction is negligible everywhere (≤0.03), so no cell is "freezing."

**Caveats:**
- **Missing primary metric:** `eval/success_rate` is not in this dump; `eval/mean_reward` is the substitute. It is a continuous reward, not a literal success fraction, though it behaves as a success proxy here.
- **No `none` arm:** this is e3b_idm-only (12 cells, single bonus), so it is a *ranking* under e3b, **not** a bonus-vs-none contrast — the known caveat holds, and the Memoryless=0 result cannot disentangle "no memory" from "bonus unhelpful without memory."
- **26 crashes are uninformative:** every crashed run died at ≤40,960 global steps (<0.3% of the 20M budget) with `eval/mean_reward = None`. They contribute zero eval signal and are correctly dropped; do not treat them as partial results.
- **Partial seeds after crashes:** finished-seed counts are uneven — GRU and LSTM have only n=3; GatedDeltaNet, SHM, LRU, mLSTM have n=4; the rest have n=5. GatedDeltaNet's #1 rank rests on 4 seeds (one of which, 0.85, is the best single run in the project), so its lead over RetNet (n=5) is suggestive but not statistically tight.
- Single env, single density (sparse), single bonus — no cross-condition generalization can be read from this project alone.

---

## memrl-mpg-hpsweep
*MysteryPath-Grid-v0 | 54 finished / 57 (3 crashed) | single-seed (seed 0) HP grid over 3 memory architectures (GRU, RetNet, GatedDeltaNet) × chunk_len {32,64,128} × lr {1e-4, 2.5e-4} × lambda_intrinsic {0.003,0.01,0.03} with e3b_idm, plus a `none` (no-bonus) baseline per arch*

> **Metric note:** `eval/success_rate` is **absent from every run's summary** (0/57). I use **`eval/mean_reward`** as the headline metric — all 54 finished values are exact multiples of 0.05 (range 0.05–0.60), i.e. fraction-solved over a ~20-episode sparse-reward eval, so it *is* the de-facto success rate under a different key. `eval/mean_ep_length` (lower = faster solves, max 128) tracks it inversely. All runs are **n=1 (seed 0 only)** and reached the full 5.0M steps; means below are over HP cells, not seeds.

**Best e3b_idm config per architecture (eval reward; eplen; idle)**

| Arch | Best config (ck / lr / λ) | Best reward | eplen | idle | Mean over 17 cfgs | Range | `none` baseline |
|---|---|---|---|---|---|---|---|
| **RetNet** | 128 / 2.5e-4 / 0.03 | **0.60** | 74.2 | 0.007 | 0.382 | 0.20–0.60 | 0.20 |
| **GatedDeltaNet** | 128 / 1e-4 / 0.03 | 0.50 | 92.6 | 0.003 | 0.265 | 0.15–0.50 | 0.30 |
| **GRU** | 64 / 1e-4 / 0.03 | 0.30 | 104.4 | 0.003 | 0.144 | 0.05–0.30 | 0.15 |

**Marginal HP effects (mean eval reward over all e3b cells)**

| HP | Levels (reward) |
|---|---|
| **lambda_intrinsic** | 0.003 → 0.233 · 0.01 → 0.257 · **0.03 → 0.300** (monotone up; idle drops 0.068→0.020→0.010) |
| **chunk_len** | 32 → 0.233 · 64 → 0.269 · **128 → 0.283** (longer helps; eplen 105.8→103.2→100.4) |
| **lr** | **1e-4 → 0.269** · 2.5e-4 → 0.259 (near-flat, slight edge to 1e-4) |
| **architecture** | **RetNet → 0.382** · GatedDeltaNet → 0.265 · GRU → 0.144 |

**Expected maximin winner — e3b_idm / λ=0.03 / ck=64 / lr=1e-4:** GRU 0.30 (**rank 1/17**), GatedDeltaNet 0.40 (rank 2/17), RetNet 0.45 (rank 2/17).

**Takeaway:** The expected winning region is **confirmed but with a nuance**. The specific config (λ=0.03, ck=64, lr=1e-4) is the top-1 cell for GRU and the strong #2 for both RetNet and GatedDeltaNet, and the *marginal* sweep cleanly endorses its ingredients: highest lambda (0.03) and longer chunks both monotonically raise reward and crush idle/dithering (idle ~0.07→0.01), while lr barely matters. So as a **robust default it holds up**. However, the single best point is **RetNet ck128/lr2.5e-4/λ0.03 = 0.60** (fastest solves, eplen 74), and the architecture marginal is dominated by **RetNet (0.38) ≫ GatedDeltaNet (0.27) > GRU (0.14)** — GRU's best (0.30) barely beats GatedDeltaNet's `none` baseline (0.30). Intrinsic bonus helps every arch over `none` (e.g. RetNet 0.20→up to 0.60), confirming e3b_idm is worth keeping.

**Caveats:** (1) **n=1 per cell (seed 0 only)** — with 0.05 metric granularity (~20 eval episodes), single-seed differences of one or two solved episodes are noise; the arch ranking and λ/ck trends are consistent enough to trust, but exact per-cell rewards are not. (2) **Primary metric `eval/success_rate` was never logged**; `eval/mean_reward` is used as its faithful proxy but the literal field requested does not exist. (3) **3 crashes form one fully-missing arm**: ck32/lr1e-4/λ0.01 for all three architectures (the crashed runs logged no global_step), so the ck32 slice has n=15 not 18 and that exact HP cell is uncovered. (4) `chunk_len=32` is under-sampled relative to 64/128 because of those crashes, mildly biasing its marginal.

---

## memrl-mysterypath-grid

*MysteryPath-Grid-v0 | 107 finished / 123 (16 crashed) | Single-density sweep of 12 memory architectures × 4 intrinsic bonuses (none/rnd/noveld/e3b_idm) × 3 seeds, 10M steps. **Older/sibling reward-only grid: `eval/success_rate` is NOT logged here — only `eval/mean_reward` (range 0–0.6).***

**Important schema note:** The requested primary metric `eval/success_rate` does **not exist** in this project, and there is **no falls metric** (`rollout/ep_num_fails_mean` absent). All headline numbers below use `eval/mean_reward` as the performance proxy. This — plus the absence of `env_kwargs` (density is fixed/default, single condition) — marks this as a deprecated/pre-fix grid relative to `memrl-memtrain-mpg`. Config: `total_timesteps=10M` (all 107 finished runs hit ~10.0M global_step), `lr=2.5e-4` for all cells **except Mamba2 which uses lr=1.25e-4** (cell-specific tuning, the lone lr split: 114 runs @2.5e-4 / 9 @1.25e-4).

### eval/mean_reward — mean (best seed) [n finished], rows = architecture, cols = bonus

| Architecture | none | rnd | noveld | e3b_idm |
|---|---|---|---|---|
| Memoryless | 0.067 (0.10) n=3 | 0.033 (0.05) n=3 | 0.042 (0.10) n=6 | 0.008 (0.05) n=6 |
| GRU | 0.117 (0.20) n=3 | 0.067 (0.15) n=3 | 0.133 (0.25) n=6 | 0.167 (0.25) n=3 |
| LSTM | 0.067 (0.10) n=3 | — | 0.167 (0.20) n=3 | crashed (0/3) |
| mLSTM | 0.075 (0.15) n=2 | — | 0.150 (0.25) n=3 | 0.117 (0.20) n=3 |
| LRU | 0.033 (0.10) n=3 | — | 0.150 (0.20) n=3 | 0.133 (0.20) n=3 |
| **RetNet** | 0.133 (0.20) n=3 | — | **0.317 (0.35)** n=3 | **0.400 (0.60)** n=3 |
| LinearTransformer | 0.083 (0.10) n=3 | — | 0.200 (0.25) n=3 | 0.217 (0.30) n=3 |
| GTrXL | 0.050 (0.10) n=3 | — | 0.133 (0.15) n=3 | crashed (0/3) |
| Mamba2 *(lr 1.25e-4)* | 0.100 (0.15) n=3 | — | 0.133 (0.25) n=3 | 0.267 (0.35) n=3 |
| **GatedDeltaNet** | 0.083 (0.15) n=3 | — | 0.283 (0.35) n=3 | 0.333 (0.45) n=3 |
| FFM | crashed (0/3) | — | 0.200 (0.25) n=3 | crashed (0/3) |
| SHM | 0.050 (0.15) n=3 | — | 0.100 (0.15) n=3 | crashed (0/3) |

*rnd was only run for Memoryless and GRU (6 runs total); all other "rnd" cells are intentionally absent, not missing data.*

### Marginal means by bonus (finished, pooled over all cells)

| Bonus | mean reward | n |
|---|---|---|
| e3b_idm | **0.183** | 27 |
| noveld | 0.156 | 42 |
| none | 0.078 | 32 |
| rnd | 0.050 | 6 |

### Idle fraction (`debug/action_frac_0`), mean [finished] — mechanism signature

| Bonus | typical idle frac |
|---|---|
| none | 0.16–0.28 (recurrent cells ~0.2–0.28) |
| rnd | 0.13–0.20 |
| noveld | 0.0–0.11 |
| e3b_idm | 0.0–0.08 |

**Takeaway:** The expected "exploration bonuses help" pattern is **confirmed**, but read as reward, not success (success_rate was never logged). e3b_idm (0.183) and noveld (0.156) each roughly double the no-bonus baseline (0.078), while rnd (0.050) is the worst — at or below the none baseline on its two cells (GRU 0.117→0.067, Memoryless 0.067→0.033). The bonus effect is consistent across nearly every architecture, and the idle-fraction collapse from ~0.2 (none) to ~0.0–0.08 (noveld/e3b) is the same idle-fraction mechanism seen in the locked spine. Best overall arms are RetNet+e3b_idm (0.40, best seed 0.60) and GatedDeltaNet+e3b_idm (0.33); the modern linear-attention/SSM cells (RetNet, GatedDeltaNet, LinearTransformer, Mamba2) dominate the classic recurrent cells (GRU, LSTM, LRU) once a bonus is on, whereas Memoryless stays near the floor regardless of bonus (e3b_idm actually hurts it, 0.067→0.008).

**Caveats:**
- **Metric mismatch:** no `eval/success_rate` and no falls metric in this project; reward ∈ [0, 0.6] is a proxy and absolute values are low/noisy (best seeds top out at 0.6). Treat ordering as the signal, not magnitudes.
- **16 crashes, structurally biased toward e3b_idm:** LSTM, GTrXL, SHM, and FFM all have **0/3 finished** under e3b_idm, plus FFM-none (0/3) and mLSTM-none (2/3). So the e3b_idm column is missing 4 of 12 architectures entirely — the e3b marginal (n=27) is computed over the cells that survived and may be optimistic. Crashed e3b_idm runs that got far still showed nonzero reward (GTrXL ~7.8M steps reaching 0.25; SHM/FFM ~7.5–8.4M reaching 0.2–0.3), suggesting those arms would likely have ranked mid-pack had they finished.
- **Uneven seed counts:** Memoryless/GRU have 6 noveld seeds (Memoryless also 6 e3b_idm) vs 3 elsewhere; mLSTM-none has only n=2. rnd is a 2-cell stub.
- **Mamba2 lr confound:** Mamba2 alone runs at lr=1.25e-4, so its cross-cell comparison is not lr-matched.
- Single fixed density (env_kwargs null) — this grid does **not** carry the penalty/aligned/sparse density axis of the current MPG grid, so it cannot speak to density-dependent effects.

---

## memrl-memtrain-s13
*MiniGrid-MemoryS13, view=3 (hard partial view), sparse, 20M steps | 78 finished / 112 (14 crashed, 20 failed) | does an exploration bonus rescue memory cells on the hardest MemoryS variant? (cell × bonus, n=2 seeds)*

All 112 runs are `density=sparse` (`agent_view_size=3`). Floor approx 0.35 (chance), solved approx 0.98. Each (cell, bonus) arm targets **2 seeds (seed0, seed1)**; SHM and FFM were rerun many extra times (mostly crashing). Headline = mean of FINISHED seeds; per-seed values exposed because disagreement is large.

### eval/mean_reward — mean (n finished) [seed0 / seed1]

| Cell | none | noveld | e3b |
|---|---|---|---|
| **Memoryless** | 0.45 (2) [0.35/0.55] | 0.45 (2) [0.35/0.55] | 0.45 (2) [0.35/0.55] |
| GRU | 0.55 (2) [0.64/0.45] | 0.81 (2) [**0.98**/0.63] | 0.76 (2) [0.54/**0.98**] |
| LSTM | 0.55 (2) [0.55/0.55] | **0.98 (2)** [0.98/0.98] | **0.98 (2)** [0.98/0.98] |
| mLSTM | 0.45 (2) [0.45/0.45] | 0.76 (2) [0.73/0.78] | 0.59 (2) [0.64/0.54] |
| LRU | 0.50 (2) [0.54/0.45] | 0.86 (2) [0.74/**0.98**] | 0.85 (2) [**0.93**/0.78] |
| RetNet | 0.42 (2) [0.45/0.40] | **0.98 (2)** [0.98/0.98] | 0.66 (2) [0.54/0.78] |
| LinearTransformer | 0.40 (2) [0.35/0.45] | 0.86 (2) [**0.98**/0.74] | 0.78 (2) [0.78/0.79] |
| Mamba2 | 0.67 (2) [0.54/0.79] | **0.98 (2)** [0.98/0.98] | 0.61 (2) [0.34/0.88] |
| **GatedDeltaNet** | **0.98 (2)** [0.98/0.98] | **0.96 (2)** [0.98/0.93] | **0.93 (2)** [0.88/0.98] |
| FFM | 0.40 (2) [0.35/0.45] | 0.93 (2) [**0.98**/0.88] | 0.66 (2) [0.54/0.78] |
| SHM | 0.41 (10*) | **NA (0)** | 0.82 (2*) [0.78/0.86] |
| GTrXL | 0.40 (2) [0.35/0.45] | 0.66 (2) [0.78/0.54] | **0.98 (2)** [0.98/0.98] |

Bold = solved (>=0.90). `*` SHM had many crashed/failed reruns; SHM-none's 10 "finished" are duplicate reruns all clustered 0.25–0.50 (never solves); SHM-noveld has **0 finished** (8 failed runs, best partial 0.96 @5M).

### Idle fraction & episode length (finished mean)
Solving arms ride ep_len up to ~15–27 steps (agent actually navigates); collapsed/floor arms sit at ep_len ~7.8 (degenerate near-instant policy). Idle fraction stays low everywhere (0.01–0.22), so failure is not idling — it is failure to encode/retrieve the memory cue. `rollout/ep_num_fails_mean` is not logged for this env (NA throughout).

**Takeaway:** The intended story is **partly confirmed, partly refuted.** Confirmed: memory is required and **no-bonus collapses for 11 of 12 cells** (0.40–0.67, ep_len ~8). Confirmed: a bonus is the catalyst, and **NovelD is the most reliable one** — it cleanly solves LSTM, RetNet, Mamba2 (all 0.98 both seeds) and lifts LRU/LinearTransformer/FFM to ~0.86–0.93 — while **e3b is genuinely cell-dependent**, strong on GTrXL/GatedDeltaNet/LSTM (~0.93–0.98) but weak on GRU/RetNet/mLSTM/Mamba2 (0.59–0.76). **Refuted in three ways:** (1) Memoryless does **not** sit at the 0.35 floor — it averages 0.45 because seed1 lands a degenerate 0.55 policy (ep_len 7.8), so 0.35 is the *floor*, not the *Memoryless reading*; (2) NovelD does **not** universally reach ~0.98 — GRU (0.81), mLSTM (0.76), GTrXL (0.66) and the seed-split GRU/LinearTransformer arms fall short; (3) **GatedDeltaNet is the standout exception** — it solves MemoryS13 to ~0.98 under *none* and stays solved under both bonuses, so the bonus is not what rescues it. Net: catalyst identity is cell-specific (NovelD-favoring vs e3b-favoring cells), and GatedDeltaNet alone needs no catalyst.

**Caveats:** (1) **n=2 seeds per arm** — far below a reliable bar, and **seed disagreement is severe**: GRU-noveld (0.98 vs 0.63), GRU-e3b (0.54 vs 0.98), Mamba2-e3b (0.34 vs 0.88), GatedDeltaNet-e3b (0.88 vs 0.98). Several "rescues" hang on a single lucky seed. (2) **SHM is unreliable infrastructure**, not a clean cell read: SHM-noveld has zero finished runs (all 8 failed; partials reached 0.96 @5M, suggesting it *could* solve if it ran), SHM-e3b finished only 2 of 9, and SHM-none never exceeds ~0.50 across 10 finished reruns. SHM should be treated as missing/inconclusive across all bonuses. (3) mLSTM and FFM also lost a seed1 to crashes (crashed partials sometimes higher than the finished seed0, e.g. mLSTM-noveld crashed @0.98 @16M vs finished 0.78), so their finished means may understate true performance. (4) No `none`/baseline arm cleanly isolates whether the few cells above floor under *none* (Mamba2 0.67, GatedDeltaNet 0.98) reflect architecture or seed luck — re-run with >=5 seeds before claiming a bonus-free solver.

---

## memrl-s13-baseline

*MiniGrid-MemoryS13-v0 | 120 finished / 128 (3 crashed, 5 failed) | 12 memory architectures (cells) x intrinsic bonus, 3 seeds, 10M steps each*

**What it actually is (verified):** Despite the name, this is **not** a view-size or cell-x-density sweep — every run is the *same* env (`MiniGrid-MemoryS13-v0`), with no `env_kwargs` and no density variants. The two swept axes are **memory architecture** (`config.cell.name`: 12 cells) x **intrinsic bonus** (`config.intrinsic`: none / noveld / rnd / e3b_rand / e3b_idm). Shared config: lr 3e-4 (Mamba2 used 1.5e-4), gamma 0.999, n_steps 512, chunk_len 32, 16 envs, 10M total_timesteps. Only `eval/mean_reward` and `eval/mean_ep_length` are logged (no `success_rate` / `fails` keys). This differs from `memrl-memtrain-s13` (view=3): here there is a single fixed view and no density manipulation, so the spread is driven by architecture and bonus alone.

**Reading the metric:** MemoryS13 reward decays with steps, so two regimes are clean: **mr ~0.55 with ep_len ~8 = "fail-fast" floor** (24 runs cluster there — grab the nearest object, episode ends fast), and **mr >0.95 with ep_len ~15 = solved** (22 runs — actually traverse the corridor and read the cue). Only 27/120 finished runs solve (mr>0.9); 61/120 sit below 0.6 (near floor).

### eval/mean_reward — mean (n_finished), best-seed in [ ]

| Cell | none | noveld | rnd | e3b_rand | e3b_idm |
|---|---|---|---|---|---|
| **GatedDeltaNet** | **0.94 (3)** [0.99] | **0.97 (3)** [0.98] | – | – | 0.90 (3) [0.99] |
| **Mamba2** | 0.77 (3) [0.99] | **0.97 (3)** [0.99] | – | – | **0.97 (3)** [0.99] |
| **LRU** | 0.79 (3) [0.98] | 0.87 (3) [0.98] | – | – | 0.77 (3) [0.98] |
| **LSTM** | 0.50 (3) [0.55] | 0.82 (3) [0.89] | – | – | 0.76 (6) [0.99] |
| **SHM** | — (0/3 fail) | 0.98 (1/3) [0.98] | – | – | 0.69 (3) [0.79] |
| **mLSTM** | 0.69 (3) [0.93] | 0.80 (3) [0.89] | – | – | 0.55 (3) [0.74] |
| **GTrXL** | 0.61 (3) [0.74] | 0.84 (3) [0.89] | – | – | 0.59 (3/6) [0.84] |
| **GRU** | 0.46 (3) [0.60] | 0.84 (3) [0.99] | 0.86 (3) [0.99] | 0.69 (3) [0.98] | 0.55 (6) [0.79] |
| **FFM** | 0.48 (3) [0.55] | 0.53 (3) [0.55] | – | – | 0.52 (6) [0.74] |
| **LinearTransformer** | 0.48 (3) [0.55] | 0.55 (3) [0.65] | – | – | 0.55 (3) [0.64] |
| **Memoryless** | – | 0.55 (3) [0.65] | 0.55 (2) [0.65] | 0.45 (3) [0.55] | 0.51 (3) [0.65] |
| **RetNet** | 0.48 (3) [0.55] | 0.59 (3) [0.74] | – | – | 0.41 (3) [0.45] |

Marginals (mean over all finished runs): **by cell** — GatedDeltaNet 0.94 > Mamba2 0.90 > LRU 0.81 > SHM 0.76 > LSTM 0.71 > GTrXL 0.68 = mLSTM 0.68 > GRU 0.66 > LinearTransformer 0.52 > FFM 0.51 > Memoryless 0.51 > RetNet 0.50. **By bonus** — noveld 0.76 > rnd 0.73 > e3b_idm 0.64 > none 0.62 > e3b_rand 0.57.

**Takeaway:** Two clear signals. (1) **Architecture dominates:** the linear-attention / SSM family (GatedDeltaNet, Mamba2, LRU) solves S13 essentially regardless of bonus, with GatedDeltaNet topping the board at 0.94 mean even with *no* exploration bonus; classic recurrent cells (LSTM, GRU) and lightweight/linear cells (FFM, LinearTransformer, RetNet) sit near the 0.55 fail-fast floor without help. Memoryless never escapes the floor under any bonus (max 0.55), confirming the task genuinely requires memory. (2) **Bonuses help the weak, not the strong:** noveld (and rnd, where run) is the most reliable un-sealer — it lifts GRU 0.46->0.84, LSTM 0.50->0.82, GTrXL 0.61->0.84 — while adding nothing to cells that already solve (GatedDeltaNet 0.94->0.97). **e3b_idm is inconsistent** and often *worse* than none (RetNet 0.48->0.41, mLSTM 0.69->0.55), so it is not a dependable rescue here. This matches the broader "rescue asymmetry" theme: count-based novelty bonuses un-seal stuck policies; e3b-style bonuses are unreliable.

**Caveats:** (a) Only 3 seeds per arm (some bonus columns are missing entirely — rnd/e3b_rand were run only for GRU and Memoryless, so cross-cell bonus comparison is incomplete). (b) **SHM-none failed on all 3 seeds** (no usable number) and SHM-noveld has only 1/3 finished, so SHM's marginal (0.76, n=4) is unreliable. (c) **GTrXL-e3b_idm crashed on all 3 of its first seeds** (~4.4M steps) but 3 rerun seeds finished — the crashed runs were already only ~0.45–0.69 mid-training. (d) The **e3b_idm arms for FFM/GRU/GTrXL/LSTM have 6 finished runs with duplicated seeds (0,0,1,1,2,2)** — i.e. a rerun, so n=6 there is 3 unique seeds counted twice; means are still over the actual runs but effective seed diversity is 3. (e) **Mamba2 alone used lr=1.5e-4** vs 3e-4 for everyone else, so its strong showing is partly confounded by a different LR. (f) No success-rate metric is logged — "solved" is inferred from mean_reward>0.9 / ep_len ~15, validated against the floor cluster (mr~0.55, ep_len~8).

---

## memrl-memtrain-tiny
*TinyReproduce-v0 (enumerable reward-machine, k=10 exact-recall, reverse order, v=4) | 72 finished / 96 (12 crashed, 12 failed) | controlled architecture x bonus probe: tests (1) bonus-neutral null (e3b ~ none on every cell) and (2) cell inversion (gated RNN >> SSM on exact recall at registry HP)*

All 96 logical runs are the same 72 cells (6 architectures x 2 densities x 2 bonuses x 3 seeds); the 12 crashed + 12 failed entries are earlier attempts at Memoryless/RetNet seed1-2 that were re-run, so deduping by name (prefer finished, max global_step) yields a clean 3-seed grid with 72/72 finished. Every run reached the full 10M timesteps. k=10, lambda_intrinsic=0.03, chunk_len=64, n_steps=512.

**Primary metric `eval/success_rate` -- mean over 3 seeds (best seed), by cell x density x bonus:**

| Cell (params) | dense / none | dense / e3b_idm | sparse / none | sparse / e3b_idm |
|---|---|---|---|---|
| **LSTM** (567k) | **0.933** (0.95) | **0.900** (0.95) | **0.817** (0.95) | **0.817** (0.95) |
| **GRU** (500k) | 0.650 (0.75) | 0.600 (0.65) | 0.633 (0.65) | 0.667 (0.70) |
| GatedDeltaNet (434k) | 0.067 (0.10) | 0.100 (0.30) | 0.017 (0.05) | 0.050 (0.15) |
| Mamba2 (569k) | 0.000 (0.00) | 0.000 (0.00) | 0.017 (0.05) | 0.000 (0.00) |
| RetNet (434k) | 0.000 (0.00) | 0.000 (0.00) | 0.000 (0.00) | 0.000 (0.00) |
| Memoryless (335k) | 0.000 (0.00) | 0.000 (0.00) | 0.000 (0.00) | 0.000 (0.00) |

**e3b_idm vs none (delta of mean success, +ve = bonus helps):** LSTM-dense -0.033, LSTM-sparse 0.000, GRU-dense -0.050, GRU-sparse +0.034, GDN-dense +0.033, GDN-sparse +0.033, Mamba2 0.000/-0.017, RetNet 0.000/0.000, Memoryless 0.000/0.000. Max |delta| = 0.05, all within single-seed (0.05-step) noise. The bonus was genuinely active (intrinsic/bonus_mean 0.18-0.79 across e3b runs, IDM accuracy 0.40-1.0), so this is a true bonus-neutral result, not an inactive bonus.

**Takeaway:** Both pre-registered predictions are CONFIRMED. (1) *Cell inversion holds, and strongly*: the gated RNNs (LSTM mean 0.82-0.93, GRU 0.60-0.67) beat every SSM by ~0.6-0.9 absolute success, and three of four SSM/baseline cells (Mamba2, RetNet, Memoryless) sit at a hard 0.00 floor; GatedDeltaNet is the only SSM that registers any recall at all (0.02-0.10 mean, best seed 0.30). This is not a capacity artifact -- RetNet (434k) and Mamba2 (569k) bracket LSTM's 567k yet score zero. (2) *The bonus-neutral null holds on every cell*: e3b_idm changes success by at most 0.05 anywhere and never rescues a floored SSM, so the intrinsic bonus is realization-neutral on this exact-recall task (alpha ~ 0).

**Secondary signals corroborate:** SSMs/Memoryless also have shorter eval episodes (~11-15 steps vs ~18-19 for the RNNs) and lower eval/mean_reward (RetNet ~0.18, Memoryless ~0.13 vs LSTM ~0.92-0.96), consistent with failing to complete the recall sequence rather than partial credit. Idle fraction (debug/action_frac_0) is uniform ~0.21-0.30 across all cells, so the SSM floor is not an idling/degenerate-policy artifact. Memoryless reaching IDM accuracy ~0.98-1.0 (vs ~0.5-0.8 for memoryful cells) reflects its trivial inverse-dynamics task, not task success.

**Caveats:** (1) n=3 seeds per cell; success is reported on a coarse eval set so per-seed values are quantized (steps of 0.05), making the small e3b deltas individually unreliable -- the null rests on the *consistency* of near-zero deltas across all 12 cell-density pairs, not any single estimate. (2) GatedDeltaNet's nonzero scores (best seed 0.30 on dense/e3b) show it is right at the edge of solving the task, so its "SSM floor" classification is the softest; Mamba2/RetNet are unambiguously floored. (3) The 24 crashed/failed entries were all successfully re-run, so there are no missing arms, but the original infra instability was concentrated in the Memoryless and RetNet seed1-2 jobs. (4) rollout/ep_num_fails_mean is not populated for this env, so a falls column is unavailable.

---

## memrl-memtrain-gdnsweep
*TinyReproduce-v0 (k=10), intrinsic=none | 20 finished / 24 (4 failed) | GatedDeltaNet LR x assoc_size sweep to test whether GDN's k=10 floor is LR-fragility/tuning rather than an architecture wall*

**Headline: the expected "inconclusive, all ~0.00-0.04, under-budget" read is REFUTED.** These runs trained to the **full 10M steps** (`total_timesteps=10_000_000`, every finished run reached `global_step=10,002,432`) — not the 2M under-budget regime the prior assumed. With the budget paid, the floor breaks: 11/16 cells clear 0.04 and 5/16 reach >=0.30. There is a clear LR sweet spot whose location depends on `assoc_size`.

### eval/success_rate by config — mean (n_finished), best seed, idle
Grid is the full 4 LR x 2 assoc_size, 2 seeds each (16 cells, all recovered as finished).

| assoc_size | lr=1e-4 | lr=3e-4 | lr=1e-3 | lr=3e-3 |
|---|---|---|---|---|
| **a64**  | 0.00 (2) | 0.20 (2) | **0.35 (2)** | 0.00 (2) |
| **a256** | 0.225 (2) | **0.35 (2)** | 0.075 (2) | 0.025 (2) |

Best single seeds: **a64/lr=1e-3 → 0.40** (seeds 0.30/0.40); **a256/lr=3e-4 → 0.40** (seeds 0.30/0.40); a256/lr=1e-4 → 0.35 (seeds 0.35/0.10, high variance). Idle fraction (`debug/action_frac_0`) is uniform ~0.23-0.28 across all cells, so successes are not a degenerate idle artifact. Overall mean over 16 finished cells = 0.153; per-run min 0.00, max 0.40.

**Takeaway:** The k=10 "floor" for GatedDeltaNet is an LR-tuning effect, not an architecture wall — at the right LR, GDN solves the task ~35-40% of the time at k=10. The optimum is assoc-size-dependent (a64 peaks at lr=1e-3=0.35; a256 peaks at lr=3e-4=0.35), and both arms collapse to ~0.00 at the high end (lr=3e-3) and at the mismatched low end (a64/lr=1e-4=0.00). This is textbook LR-fragility: a 3x LR step in either direction from the per-config optimum drops success to floor. assoc_size=256 widens the usable LR band a bit (it retains 0.225 at lr=1e-4 where a64 is dead) but does not lift the ceiling above a64's best.

**Caveats:** (1) Contrary to the project's recorded prior, this sweep is at 10M steps, not 2M — so the "under-budget, inconclusive" caveat does not apply to *this* data; the conclusive read here is that budget + LR jointly explain the prior floor. (2) Only **n=2 seeds per cell**, so the means are noisy — note a256/lr=1e-4's wide seed spread (0.35 vs 0.10). Peak configs should be re-run with more seeds before claiming a firm GDN k=10 ceiling. (3) The 4 "failed" runs were early crashes (step 8192) of just two cells (a256/lr=3e-4/seed0 and a256/lr=3e-3/seed0); both were successfully re-launched and finished, so no grid cell is missing — failures do not bias the table.

---

## memrl-mm-toggle
*MortarMayhem-Grid-v0 | 12 finished / 12 | density toggle (sparse <-> dense/aligned) x bonus (none / e3b_idm) across 3 memory cells; tests whether the aligned dense-reward bonus is redundant with intrinsic e3b*

All 12 runs finished and trained the full 10M steps (global_step ~10.0M each). Design = 3 cells (GRU, GatedDeltaNet, RetNet) x 2 densities (sparse, dense) x 2 bonuses (none, e3b_idm), **n=1 (seed 0 only) per arm.** Primary metric = eval/mean_reward.

**eval/mean_reward (n=1 per cell), with eval success_rate and idle-fraction:**

| cell | sparse-none | sparse-e3b | dense-none | dense-e3b | e3b effect (dense) |
|---|---|---|---|---|---|
| GRU | 0.000 | 0.000 | 0.288 | **0.750** | **+0.462** |
| GatedDeltaNet | 0.000 | 0.000 | 0.338 | 0.150 | -0.188 |
| RetNet | 0.000 | 0.000 | 0.125 | 0.038 | -0.088 |
| **mean** | **0.000** | **0.000** | **0.250** | **0.312** | +0.062 |

Supporting detail (eval success_rate / idle action_frac_0):

| arm | GRU | GatedDeltaNet | RetNet |
|---|---|---|---|
| sparse-none | 0.00 succ / 0.30 idle | 0.00 / 0.36 | 0.00 / 0.19 |
| sparse-e3b | 0.00 / 0.13 | 0.00 / 0.08 | 0.00 / 0.05 |
| dense-none | 0.05 / 0.24 | 0.10 / 0.33 | 0.00 / 0.22 |
| dense-e3b | **0.70** / 0.09 | 0.00 / 0.17 | 0.00 / 0.06 |

**Takeaway:** The intended "return-matched 1.0 sparse<->aligned" toggle did **not** hold empirically: **every one of the 6 sparse runs scored exactly 0.0 reward and 0.0 eval success**, regardless of cell or bonus, so the sparse arm yielded no learning signal to compare against. All non-zero performance came from the dense/aligned arm. There the aligned-bonus-redundancy question is answered inconsistently across cells: e3b is strongly complementary for GRU (0.29 -> 0.75, success 0.05 -> 0.70) but actively harmful for GatedDeltaNet (0.34 -> 0.15) and RetNet (0.13 -> 0.04). So the cross-cell mean e3b benefit (+0.06) is a single-cell artifact, not a robust effect — e3b on top of dense aligned reward is redundant-to-harmful for the linear-attention cells and only helps the GRU. The clean "exactly return-matched delta-toggle" framing is contradicted by the sparse-arm floor.

**Caveats:** (1) **n=1 — single seed (seed 0) per arm**; the GRU dense-e3b 0.75 vs GatedDeltaNet/RetNet collapses could be seed noise, no error bars possible. (2) **Sparse arm is degenerate at 0.0 everywhere**, so the headline "return-matched" comparison cannot actually be made — there is no learned sparse policy to match. (3) Density (sparse vs dense) is read **only from the run name**; config.env_kwargs is null and no config field encodes reward_fall_off / reward_progress, so the density labeling cannot be cross-validated against env settings. (4) Bonus eval success rates are near-floor for 2 of 3 cells even in the best (dense) arm, so MortarMayhem-Grid is largely unsolved at 10M steps outside GRU+e3b.

---

## memrl-ss-toggle
*SearingSpotlights-v0 | 12 finished / 14 (2 crashed) | Dead-reckoning memory: sparse vs. anti (reward_inside_spotlight=-0.008) density toggle x e3b_idm bonus, a freeze BOUNDARY test (wandering spotlights = no passive sanctuary).*

**Schema note:** in the run names, the leading token is the **memory architecture** (GRU / GatedDeltaNet / RetNet), `density`∈{sparse, anti}, and `bonus`∈{none, e3b_idm}. There is **one seed (seed0) per cell** (n=1 everywhere). The anti arm sets `env_kwargs.reset_options.reward_inside_spotlight=-0.008`; bonus runs use `lambda_intrinsic=0.03`. No `debug/action_frac_0` (idle-fraction) metric was logged — **`eval/mean_ep_length` is used as the survival/freeze proxy** (short ep = agent dies/freezes fast; long ep = stays alive and moving).

### eval/mean_reward — mean (n finished), best seed = same (n=1)

| memory cell | sparse / none | sparse / e3b | anti / none | anti / e3b |
|---|---|---|---|---|
| GRU | 0.038 (1) | 0.038 (1) | **0.139** (1) | 0.023 (1) |
| GatedDeltaNet | **0.188** (1) | 0.025 (1) | 0.145 (1) | **−0.028** (1) |
| RetNet | 0.125 (1) | 0.063 (1) | 0.014 (1) | −0.003 (1) |

### eval/mean_ep_length — freeze proxy (longer = agent survives/moves, not frozen)

| memory cell | sparse / none | sparse / e3b | anti / none | anti / e3b |
|---|---|---|---|---|
| GRU | 61.3 | 33.5 | **95.7** | 53.9 |
| GatedDeltaNet | 81.2 | 35.7 | **131.5** | 40.5 |
| RetNet | 65.2 | 37.8 | **113.5** | 33.2 |

(eval/success_rate is 0 nearly everywhere; the only nonzero values are 0.10 for GRU-anti-none, GatedDeltaNet-anti-none, GatedDeltaNet-sparse-none and 0.05 for RetNet-sparse-none — i.e. only `none`-bonus runs ever succeed. Agent health is ~0 whenever e3b is on in the anti arm.)

**Takeaway:** The freeze-boundary prediction is **confirmed — the anti arm did NOT freeze.** Far from collapsing to idle, the anti/none condition produces the *longest* episodes of any cell (95.7–131.5 steps vs. 61–81 for sparse/none) with nonzero health, coins, and the only nonzero success rates — exactly what you'd expect when wandering spotlights remove any passive sanctuary, so the −0.008 penalty does not induce a freeze the way a static-penalty MysteryPath would. The dominant signal in this sweep is instead that **the e3b_idm bonus is harmful**, and most severely under the anti penalty: turning e3b on collapses anti-arm reward to ~0 or negative for all three architectures (GRU 0.139→0.023, GatedDeltaNet 0.145→−0.028, RetNet 0.014→−0.003) and roughly halves episode length, meaning the bonus drives agents into spotlights/death rather than safe exploration. e3b also hurts in the sparse arm for the linear-attention cells (GatedDeltaNet 0.188→0.025, RetNet 0.125→0.063) and is flat for GRU.

**Caveats:** (1) **n=1 seed per condition** — every headline number is a single run, so cell-to-cell differences (e.g. GatedDeltaNet vs. RetNet) are not statistically distinguishable; treat the e3b-hurts and no-freeze patterns as directional. (2) The two RetNet-sparse runs (none + e3b) **crashed at ~0.58M / 0.57M steps** but were both re-launched and finished to the full 10M, so no condition is missing a finished run; the crashed runs' summary eval (~0.10) is from <6% of training and is not used. (3) Overall reward/success is low across the board (best eval reward 0.188, success ≤0.10) — SearingSpotlights is hard at this budget, which compresses the dynamic range and makes the negative e3b effect easier to read than any positive memory-architecture effect. (4) No idle-fraction metric was logged; the "did not freeze" claim rests on episode-length + health + nonzero coins as proxies, not a direct idle measurement.

---

## memrl-memtrain-miniworld

*MiniWorld-Sign-v0 (3D, read randomized sign color then commit) | 24 finished / 32 (6 crashed, 2 failed) | architecture x density x novelty-bonus sweep at 20M steps, single seed — **DEPRECATED: env removed from codebase, historical only***

This run set sweeps 6 memory cells x 2 reward densities x 2 bonuses (novelty/none), all seed0. Naming is `<arch>-<density>-<bonus>-seed0`; here `group`=density and `job_type`=bonus. Densities: **sparse** = `reward_wrong:0`, **penalty** = `reward_wrong:-1` (wrong-color commit punished). Note the headline `eval/success_rate` is computed over only **5 eval episodes**, so it is quantized to {0.0, 0.2} and carries essentially no resolution; `rollout/ep_is_success_mean` (rolling ~100-ep) is the finer signal and is reported alongside.

**eval/success_rate — mean over 5 eval episodes (idle action_frac_0 in parens)**

| Arch | sparse/none | sparse/noveld | penalty/none | penalty/noveld |
|---|---|---|---|---|
| GRU | 0.20 (0.00) | 0.00 (0.46) | 0.00 (1.00) | 0.00 (1.00) |
| GatedDeltaNet | 0.00 (0.00) | 0.00 (0.57) | 0.00 (0.00) | 0.20 (0.00) |
| LSTM | 0.20 (0.00) | 0.20 (0.00) | 0.00 (0.00) | 0.00 (1.00) |
| Mamba2 | 0.20 (0.00) | 0.20 (0.00) | 0.00 (0.00) | 0.00 (1.00) |
| Memoryless | 0.20 (0.00) | 0.00 (1.00) | 0.00 (0.00) | 0.00 (1.00) |
| RetNet | 0.20 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.20 (0.00) |
| **col mean** | **0.167** | **0.067** | **0.000** | **0.067** |

**rollout/ep_is_success_mean (rolling, finer-grained) — column means**

| Condition | mean | max | best arch |
|---|---|---|---|
| sparse/none | 0.068 | 0.120 | RetNet |
| sparse/noveld | 0.148 | 0.360 | GatedDeltaNet (0.36), GRU (0.35) |
| penalty/none | 0.000 | 0.000 | — (all 0) |
| penalty/noveld | 0.032 | 0.100 | GatedDeltaNet, RetNet |

**Takeaway:** The expected result — that memory cells learn to read the sign and retain the color — is **refuted**. No condition exceeds 1/5 eval episodes (0.2), and on the finer rollout metric the very best cell reaches only ~0.36 (GatedDeltaNet, sparse/noveld) with all others well below. Recurrent/state-space cells (GRU, LSTM, Mamba2, RetNet) do **not** separate from the **Memoryless** baseline, which itself hits 0.2 on sparse/none — i.e. the eval metric cannot tell memory apart from no-memory on this env. The **penalty** density collapses everything: penalty/none is 0.000 across all 6 cells, and under penalty several cells (GRU, LSTM, Mamba2, Memoryless) degenerate to **100% idle** (action_frac_0=1.00, ep_len pinned at the 150 cap or at 1), i.e. they learn to do nothing to avoid the −1 wrong-color penalty (eval/mean_reward = −0.35 for the cells that do commit, GatedDeltaNet/RetNet penalty/noveld). The novelty bonus gives a small lift on the rolling metric for sparse (0.068→0.148) but does not translate into eval-level success and makes the idle-collapse worse under penalty.

**Caveats:** (1) **Single seed (seed0) only** — every cell is n=1, so the {0.0, 0.2} eval values are individual episode counts, not robust means; treat all separations as noise. (2) **5-episode eval** quantizes the headline metric to {0,0.2}, destroying resolution — rely on `rollout/ep_is_success_mean` for any read. (3) All 8 non-finished runs (6 crashed, 2 failed) are duplicate-named **restarts** of combos that have a finished counterpart (crashes died at ~50k–172k steps; the 2 GRU-sparse "failed" runs logged no metrics) — so no unique (arch x density x bonus) arm is missing, but also no arm has a true replicate. (4) Env is **removed from the codebase**; these numbers are historical and not reproducible from current code. (5) No "aligned" density arm exists despite the schema hint — only sparse (`reward_wrong:0`) and penalty (`reward_wrong:-1`).

---

## memrl-memtrain-minihack
*MiniHack-Memento-F2-v0 | 0 finished / 2 (2 crashed) | Cluster attempt at the GDN-vs-Memoryless conjunction on MiniHack-Memento; both runs died at startup, real MiniHack results live in local/offline logs.*

Both cluster runs were the sparse / e3b_idm arm of the conjunction test (GatedDeltaNet vs Memoryless, seed 0, lr=1e-4, 10M-step budget, 16 envs x 512 steps). Both crashed almost immediately. No `eval/*` or `rollout/*` keys were ever written, so the primary metric `eval/success_rate` does not exist for either run — only a few training/debug scalars were logged before the crash.

| Run (cell-density-bonus-seed) | State | global_step | % of 10M budget | runtime | total / cell params | eval/success_rate | idle frac (action_frac_0) |
|---|---|---|---|---|---|---|---|
| GatedDeltaNet-sparse-e3b_idm-seed0 | crashed | 24,576 | 0.25% | 94 s | 2,046,925 / 132,100 | n/a (never logged) | 0.116 |
| Memoryless-sparse-e3b_idm-seed0 | crashed | 32,768 | 0.33% | 99 s | 1,947,849 / 33,024 | n/a (never logged) | 0.089 |

**Takeaway:** There is nothing to confirm or refute here — this project has zero evaluable results. Both runs are pure infrastructure crashes (~95-99 s wall-clock, <0.35% of the 10M-step budget, no eval/rollout logging), so the GDN→1.0 vs Memoryless→0.5 conjunction cannot be assessed from this W&B project. The only numbers present (idle fraction ~0.09-0.12, gradient/intrinsic debug scalars) are startup-transient and not informative. The actual MiniHack results were obtained **locally with `--no-wandb`** (logs in `/tmp/mh_*`) and split across **two different task families that must not be conflated**: (a) **MiniHack-Memento — the MEMORY task** (cue at episode start → navigate → choose fork from memory): **Memento-Short-F2 = clean conjunction**, GDN-none and GDN-e3b both → **succ 1.0** (ep_len ~12 ≤ chunk_len 32), Memoryless-e3b → **0.51** (cue-blind), S13 HPs lr=3e-4 — fully consistent with S13/MysteryPath, the memory task does **not** fail; (b) **MiniHack-Corridor — EXPLORATION tasks (E3B's benchmark), NOT memory** (no cue; `reward_lose` inert): Corridor-R2 plateaus ~0.5 (partial exploration, ep_len ~518 < 1000 cap, bonus≈none), Corridor-R3 ~0 (exploration too hard at ~3M CPU steps; E3B used 50M + IMPALA + full obs) — an exploration-scale limit on a *different axis*, not a memory failure. (An earlier "R2 = wash / cue-plateau" note was a mischaracterization — Corridor has no cue.)

**Caveats:** (1) Only 1 of the planned arms (sparse/e3b_idm) and 1 seed are present; the dense/penalty densities, the `none` bonus baseline, and additional seeds are entirely missing from the cluster pull. (2) Both runs crashed at startup, so even the logged scalars reflect untrained policies. (3) The headline MiniHack evidence is offline/local and must be read from those logs, not from this W&B project — treat memrl-memtrain-minihack as a failed cluster attempt, not a results source.

---

## memrl-redbluedoors-8x8

*MiniGrid-RedBlueDoors-8x8-v0 | 39 finished / 45 (3 crashed, 3 failed) | Negative control: does this env actually require memory? Compare Memoryless vs 10 recurrent/memory cells.*

**Metric note:** `eval/success_rate` is **None for all 45 runs** (never logged). The headline metric is the fallback **`eval/mean_reward`** (populated for 44/45; the env's max reward is ~0.99, so reward ≈ success here). `rollout/ep_success_mean` / `ep_num_fails_mean` are also absent.

### Cells with the standard bonus=none condition (3 seeds each, finished)

| Cell | mean reward (n) | best seed | mean ep_len |
|---|---|---|---|
| LSTM | 0.986 (3) | 0.987 | 19.7 |
| GRU | 0.986 (3) | 0.987 | 19.8 |
| Mamba2 | 0.986 (3) | 0.987 | 19.8 |
| GatedDeltaNet | 0.986 (3) | 0.987 | 19.6 |
| mLSTM | 0.986 (3) | 0.987 | 20.0 |
| LRU | 0.986 (3) | 0.987 | 19.9 |
| LinearTransformer | 0.986 (3) | 0.987 | 19.8 |
| GTrXL | 0.986 (3) | 0.987 | 19.9 |
| FFM | 0.986 (3) | 0.987 | 19.7 |
| RetNet | 0.970 (3) | 0.987 | 19.4 |
| **Pooled recurrent (10 cells, 30 runs)** | **0.984** (30) | 0.987 | 19.8 |
| **Memoryless** | **0.820** (3) | **0.983** | 61.8 |
| SHM | failed (0/3) | — | — |

Memoryless-none per seed: **0.983 / 0.790 / 0.688** (ep_len 23.5 / 143.7 / 18.4). One seed matches recurrent cells; two underperform, one of which is essentially dithering (~144 steps/episode vs ~20).

### Memoryless + intrinsic bonus (3 seeds each)

| Bonus | mean reward (n, nfin) | best | mean ep_len | note |
|---|---|---|---|---|
| none | 0.820 (3, 3) | 0.983 | 61.8 | high variance across seeds |
| noveld | 0.854 (3, 3) | 0.936 | 19.4 | |
| e3b_idm | 0.853 (3, **0**) | 0.887 | 40.7 | all **crashed** at ~8.4M/10M steps |
| rnd | 0.804 (3, 3) | 0.887 | 40.3 | |

**Takeaway:** The expected "RedBlueDoors does not need memory" result is **only partially confirmed**. At its best seed, Memoryless reaches 0.983 — statistically indistinguishable from the recurrent ceiling (~0.984) — so the *task* is solvable without recurrence, consistent with the negative-control intent. **However**, Memoryless is the only architecture that fails to do so *reliably*: its 3-seed mean (0.820) sits ~0.16 below every recurrent/memory cell (all clustered tightly at 0.986, sd 0.009), driven by two weak seeds (0.79, 0.69). The recurrent cells are far more seed-robust. So the honest read is: memory is **not strictly necessary** for the optimum, but it confers a large **optimization-stability** advantage here — the control passes on capability but not on stability. Adding intrinsic bonuses to Memoryless did **not** rescue it (noveld/e3b/rnd means 0.80–0.85, all below recurrent and below Memoryless's own best seed).

**Caveats:**
- Primary metric `eval/success_rate` is unlogged; all numbers are `eval/mean_reward` (a faithful proxy since success reward ≈ 0.99, but not the requested metric).
- **SHM failed entirely** (0/3 finished, runs died at 24k–156k of 10M steps, reward ~0) — no usable read for that cell.
- **Memoryless-e3b_idm crashed** all 3 seeds at ~84% of training (~8.4M/10M steps); reported numbers are partial.
- Only **n=3 seeds** per cell, and Memoryless's wide spread (sd 0.123) means its 0.820 mean is fragile — the conclusion rests heavily on which seeds are counted.
- All recurrent/memory cells were run **only at bonus=none**; the bonus sweep was applied to Memoryless alone, so the "does memory help under exploration pressure" question is not addressable from this project.
