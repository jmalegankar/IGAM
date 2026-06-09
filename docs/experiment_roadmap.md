# Experiment roadmap — Memory × Exploration paper

Living tracker. Pairs with `docs/dense_memory_exploration.md` (the thesis + design) and
the memory note `project_memory_exploration_paper.md`.

**Paper claim spine:**
- **K1** Memory cells differ in capability on memory tasks (the ranking).
- **K2** Memory ∧ exploration are *both* required on a sparse memory task (entanglement;
  bonus can't substitute for memory — the Memoryless control).
- **K3 (headline)** The exploration bonus is a **reward *densifier***: it helps *only* when
  reward is sparse; on a **dense** memory task it is neutral-to-harmful. Effect **flips** with
  reward density, memory held fixed.
- **K4 (mechanism)** The benefit = densification (critic `explained_variance` ↑, decaying with
  density); the cost = **policy bias** (isolated by the potential-based / PBIM control).

Status: ✅ done · 🟡 partial/in-flight · 🔭 generated, not launched · ⬜ to design.

---

## A. Done / in-flight

| # | Experiment | Env | Design | Status | Result | Significance | wandb |
|---|---|---|---|---|---|---|---|
| A1 | S13 baseline | MiniGrid-MemoryS13 | 11 cells × {none, rnd, noveld, e3b_rand, e3b_idm}, all-cells × {none, noveld, e3b_idm}, 3 seeds, 10M | ✅ (~116 runs) | GatedDeltaNet 3/3; Mamba2/LRU/mLSTM/SHM ~1/3; rest ~0.5. GRU+NovelD 2/3, GRU+RND 2/3, GRU+E3B 1/3. **Memoryless+bonus all fail.** | First **entanglement** signal (K2) on a MiniGrid memory task; but S13 conflates memory+exploration → motivates a cleaner env. | memrl-s13-baseline |
| A2 | RedBlueDoors-8x8 (3×3 view) | MiniGrid-RedBlueDoors | cells × none + Memoryless × {none,e3b,rnd,noveld} | 🟡 (demoted) | GRU 0.93 vs Memoryless 0.50 — memory *advantaged*, not *required*. | **Negative methodological result:** RedBlueDoors is "solvable without memory" → not a clean memory test. Cut from headline; keep as a cautionary control. | memrl-3x3-redbluedoors-8x8 |
| A3 | ObstructedMaze-1Q | MiniGrid-ObstructedMaze | GRU none vs e3b (local validation) | 🟡 (validation) | e3b ~0.03 vs none 0 at 1.6M — bonus bites but env too hard/slow. | Confirms direction; too costly for the headline. Optional secondary entangled env. | (local) |
| A4 | **MysteryPath main** | MysteryPath-Grid (sparse) | 11 cells × none + (11+Memoryless) × {e3b_idm, noveld} + GRU/Memoryless × rnd, 3 seeds, 10M | ✅ (16 crashed) | **Memory cells 0.069→0.211 (e3b, 3×) / 0.155 (noveld); Memoryless 0.057→0.017 (bonus can't rescue).** RetNet>GatedDeltaNet>Mamba2 lead. e3b≫noveld (RND saturates, noveld_loss→0). Critic `expl_var` ↑ with bonus. | **The clean K1+K2 result** + first **mechanism** hint (K4: densification). The paper's current core. | memrl-mysterypath-grid |
| A5 | **HP sweep** | MysteryPath-Grid | GRU/RetNet/GatedDeltaNet × λ{.003,.01,.03} × ck{32,64,128} × lr{1e-4,2.5e-4} (e3b) + none-ck128 probe, 1 seed, 5M | ✅ (3 combos crashed→reruns) | **Winner (maximin): e3b, λ=0.03, ck=64, lr=1e-4.** λ decisive; ck128 RetNet-skewed+OOM. | Locks global HPs for the headline (fair-comparison: global, not per-cell). | memrl-mpg-hpsweep |
| A6 | Code audit | — | 7 adversarial reviewers vs source papers, re-verified | 🟡 (workflow w60njg91b) | pending | Reviewer-proofing (cells/exploration/PPO correctness; FFM divergence root-cause). | — |

---

## B. To run (prioritized)

### B1 🔭 **20M × 5-seed headline sweep** — *launch next*
- Env MysteryPath-Grid (sparse). 12 cells (11 + Memoryless) × e3b_idm × winning HPs. 2 cells/GPU, 30 jobs.
- **Proves K1 (clean cell ranking at scale) + K2 (Memoryless control), 5 seeds for credibility, at 20M where curves still climb.**
- Status: generated + committed (`experiments/mysterypath_20m/`, `memrl-mpg-20m`). Needs image rebuild + `k8s/launch-mpg20m-jobs.sh`.
- Cost: 60 runs × 20M pixel ≈ the bulk of compute. **Gate: stage seed 0 (6 jobs) first.**

> **Theory upgrade (2026-06-09):** see `revelation_and_densification.md` — the SCDP synthesis
> reframes the program around *revelation vs density*, makes the old +0.1 runs a free third arm
> (Regime 1-aligned), and sharpens B2's predictions ({−, ≈0, +} across dense-anti / dense-aligned
> / sparse). Where that doc and this one conflict, it wins.

### B2 ⬜ **Dense-MysteryPath toggle** — *the headline experiment (K3)*
- Same maze, `env_kwargs: {reward_fall_off: -0.008}` (off-path penalty, verified) → **dense, identical memory demand, same optimal return (1.0).**
- Design: **{sparse, dense} × {none, e3b_idm}** × cells × 5 seeds, winning HPs.
- **Proves K3: e3b helps sparse, goes neutral/harmful dense — the effect flips with reward density, memory fixed.** This is what converts the paper from "trivial" to a contribution.
- Cost: ~small relative to B1 (can reuse cells; even GRU+RetNet+GatedDeltaNet × {sparse,dense}×{none,e3b}×5 = 60 runs is enough to start). **Cheapest highest-leverage run.**

### B3 ⬜ **PBIM / potential-based-bonus control** — *the knockout (K4)*
- On **dense** MysteryPath: **{none, raw-e3b, potential-based-e3b (PBIM/GRM)}** × cells × seeds.
- **Proves K4: if raw-e3b hurts dense but potential-based-e3b doesn't, the harm is policy *bias*, not exploration.** Mechanism-level, novel for the memory question.
- Build cost: implement a potential-based wrapper for the bonus (Φ-shaping; ~a module). Medium.

### B4 🔭 **POPGym-Arcade density toggle** — *K3 generality + the controllability prediction* (supersedes the old B4)
- **Built:** `experiments/arcade_densetoggle/` + `k8s/launch-arcdt-jobs.sh` (app=memrl-arcdt, 40 jobs at 2 runs/GPU, project `memrl-arcade-toggle`). Pixel 84×84 (resized from 128) → same PixelEncoder + packing profile as MysteryPath.
- `DeferredReward` wrapper: natively-dense task → sparse twin (terminal lump; same return, same π*). The REVERSE of MysteryPath's toggle.
- **BattleShipEasy** (controllable revelation, ~10 reward events/ep natively) → predict the reverse flip: e3b > none only when deferred-sparse. **CountRecallEasy** (uncontrollable; dealt stream; obs-richness MATCHED to BattleShip — this is why Arcade replaced vector POPGym, whose tiny discrete obs confounded the contrast) → predict e3b ≈ none at both densities — *sparsity alone is not sufficient; controllable revelation is*.
- Bonus: the suite's `partial_obs` flag gives **B5** (observability counterfactual) on the same tasks later.
- Prereq: image rebuilt with the `popgym-arcade` extra (Dockerfile updated). Cost: jax env-stepping is CPU-side (~250+ env-steps/s per run); 10M ≈ overnight per job.

### B5 ⬜ **POPGym-Arcade observability counterfactual** — *certifies the memory axis*
- Env POPGym-Arcade, **{full_obs, partial_obs}** × {none, e3b} × cells. (wrapper fixed; jax-cpu.)
- **Proves the manipulated variable is *memory* (toggle observability, task fixed); supports "CNN encoder everywhere."**
- Cost: pixel, GPU; needs Docker bake of jax+popgym-arcade.

### B6 ⬜ **Mechanism analysis: densification vs density** — *the "why" (K4)*
- Not a new run: analyze `explained_variance`, episode length, intrinsic health across B2's sparse/dense runs.
- **Shows the densification benefit *shrinks as reward density rises* → causal account, not benchmarking.**

### B7 ⬜ **FFM stability fix → re-include**
- FFM crashes deterministically (`expl_var=-33`, value blowup, NOT RAM). Fix locally (value/grad clip or cell numerics; audit will root-cause), then FFM rejoins the zoo cleanly.
- **Removes a hole; completes the 11-cell zoo for K1.**

### B8 ⬜ (optional breadth) Memory Maze · bsuite-MemoryLength · MortarMayhem/SearingSpotlights
- Heavyweight/secondary envs for generality if reviewers want more. Memory Maze = gold-standard memory isolation; bsuite = tunable sparse-memory ruler.

---

## C. Claim → experiment → figure

| Claim | Experiments | Figure |
|---|---|---|
| K1 cell ranking | A1, A4, **B1**, B4 | ranking bars across envs; cross-env consistency |
| K2 memory ∧ exploration both needed (sparse) | A4, **B1** | Memoryless+bonus vs cell+bonus vs cell+none |
| **K3 bonus flips with reward density** | **B2**, B4 | sparse vs dense × none vs bonus (the money figure) |
| **K4 mechanism (densification + bias)** | **B3**, B6, A4 | expl_var vs density; raw vs potential-based bonus |
| memory ⊥ sparsity (existence) | B4 (POPGym control) | dense-reward + memory-required exemplar |
| memory-axis validity | B5 (full/partial) | observability ablation |
| reviewer-proofing | A6 audit + clean harness | (appendix) |

---

## D. Critical path
**B1 (launch) → B2 (dense toggle) → B6 (mechanism) → B3 (PBIM)** is the minimal path to the full
K1–K4 story. B4 (POPGym dense) is the cheapest generality add and de-risks K3. B7 (FFM) and B5
(Arcade) round out the zoo and the memory-axis control. T-Maze: **dropped** (PPO-hostile).
