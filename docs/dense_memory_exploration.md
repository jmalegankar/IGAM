# Is it Memory, or is it Sparsity?
### Decoupling memory from reward-density: exploration bonuses are reward *densifiers*, not memory aids

Research note / paper plan. Status: hypothesis + design, pre-experiment.

---

## 0. The problem (the reviewer's critique, sharpened)

Showing "an exploration bonus helps a memory cell on a **sparse-reward** memory task" is
**trivial** — that is what exploration bonuses are for. The canonical memory benchmarks
(T-Maze, MysteryPath, MiniGrid-Memory/S13, bsuite-MemoryLength) are all **sparse by
construction** (a single terminal reward), so any "memory × exploration synergy" measured
on them is **confounded with reward sparsity**. The hard, AAAI-worthy question:

> **What does an exploration module do in a memory task whose reward is *dense*?**

Our own MysteryPath result already points at the answer: the E3B bonus helped, and the
*mechanism* looked like **reward densification** — the critic's `explained_variance` jumped
when the bonus was added, i.e. the bonus shortened the credit-assignment horizon. If the
benefit is densification, it should **disappear** when the reward is already dense.

---

## 1. Four grounded claims (with sources)

**C1 — Memory-requirement ⊥ reward-density (they are independent axes).**
"Memory required" = optimal action depends on past observations (a POMDP). "Reward dense"
= informative signal most steps. These do not imply each other. Existence proof:
**POPGym Velocity-/Position-Only CartPole & Pendulum** — reward is **dense** (+1 per
balanced step) yet the agent **must use memory** to integrate the unobserved
velocity/position ([POPGym, Morad et al. 2023](https://arxiv.org/abs/2303.01859)).
*Confidence: high.*

**C2 — Exploration bonuses HURT when reward is not sparse.**
Taïga et al., *On Bonus-Based Exploration Methods in the ALE*
([arXiv:2109.11052](https://arxiv.org/abs/2109.11052)): the bonus methods that win on
sparse Montezuma's Revenge "**often underperform ε-greedy on easy-exploration Atari
games**" and "**do not provide meaningful gains over the simpler ε-greedy scheme.**" So
when exploration isn't the bottleneck, bonuses are a *net cost*. *Confidence: high.*

**C3 — The cost is policy bias (the benefit is densification).**
Non-potential-based intrinsic rewards **provably alter the optimal policy**: "adding IM
rewards … can inadvertently alter the optimal policy, leading to suboptimal behavior"
([Potential-Based Intrinsic Motivation (PBIM), 2024](https://arxiv.org/abs/2410.12197);
[PBRS for Intrinsic Motivation, 2024](https://arxiv.org/abs/2402.07411)). The classic
failure is reward-hacking / "**noisy-TV**" — the agent gets addicted to a novelty source
and abandons the task. Potential-based shaping (Ng et al. 1999) is the only form that
densifies *without* changing the optimal policy; raw bonuses are not potential-based.
*Confidence: high.*

**C4 — The two forces separate cleanly by reward density.**
- *Benefit = densification* → scales with how sparse the reward is → **→0 when dense.**
- *Cost = policy bias* → present regardless → **dominates when dense.**
⇒ Net effect of the bonus **flips sign with reward density**, holding memory fixed.

---

## 2. The central hypothesis (falsifiable)

> **H.** An intrinsic exploration bonus aids memory-cell learning *only* via reward
> **densification**, which is valuable solely under sparse reward. Holding the memory
> demand fixed and increasing reward density, the bonus's net effect **monotonically
> degrades from helpful → neutral → harmful**, because the densification benefit decays to
> zero while the policy-altering bias persists.

**Predictions (each a figure):**
1. Sparse memory task: bonus ↑ performance (replicates the "trivial" result — needed baseline).
2. Dense memory task, same memory demand: bonus → neutral or ↓ performance.
3. The bonus's `explained_variance` gain shrinks as reward density rises (mechanism = densification).
4. A **potential-based** version of the same bonus (PBIM/GRM) is *not* harmful on the dense
   task → isolates the harm as **bias**, not exploration.

---

## 3. The crux: how to keep reward-sparsity from entangling with the memory requirement

This is the methodological core. The trap is that the standard memory benchmarks vary
*both* axes at once. The fix is to manipulate **each axis on an independent toggle, holding
the other (and the task) fixed**, plus controls that certify each axis is doing what we think.

**Toggle A — vary REWARD DENSITY, hold memory + task fixed.**
Take one memory task and change *only* its reward schedule:
- **MysteryPath-Grid**: Sparse = goal-only (default). Dense = a **−0.1 penalty each step the
  agent steps OFF the invisible path** (`reward_fall_off=-0.1`); the progress term stays 0.
  Identical maze, identical invisible-path memory demand — only the per-step FEEDBACK density
  changes. Crucially, an optimal agent never falls off, so the **optimal return stays exactly 1.0,
  identical to sparse** — this gives density *without* the reward-magnitude confound a +0.1
  progress reward would introduce (which would lift the max return to ~1.8). **This is the
  within-task control and the headline figure.**
- Generic recipe: add a potential-based dense shaping of the *extrinsic* signal; the optimal
  policy is provably unchanged (Ng 1999), so any performance change is about *density*, not task.

**Toggle B — vary MEMORY REQUIREMENT, hold reward + task fixed.**
The cleanest way to certify "memory is (not) the bottleneck" is the **observability
counterfactual**: run the *same* task fully-observable (memory unnecessary) vs
partially-observable (memory necessary).
- **POPGym-Arcade** ships every env with a `partial_obs` flag → built-in full/partial pair
  ([POPGym-Arcade, 2025](https://arxiv.org/abs/2503.01450)).
- **POPGym control**: full-state CartPole vs velocity-only CartPole — same dense reward,
  memory toggled by masking the observation.
- For MysteryPath we can add a *visible-path* variant (memory off) vs invisible (memory on).

**Controls that certify the axes:**
1. **Memoryless-cell control** (we already have a `Memoryless` cell): if it solves the task,
   memory was *not* required → that task is mislabeled. This is how we caught RedBlueDoors
   being "solvable without memory."
2. **Maximin / per-cell reporting**: the effect must hold for the *worst* cell, not be driven
   by one architecture (we already adopted this for HP selection).
3. **Potential-based bonus control (PBIM/GRM)**: separates the bonus's densification benefit
   from its bias cost.
4. **Reward-scale / λ sweep**: rules out the confound that the dense result is just a
   mis-tuned intrinsic weight rather than a real density effect (a reviewer *will* ask).

**The resulting clean design** is a 3-factor cross — **{memory: on/off} × {reward:
sparse/dense} × {bonus: none/raw/potential-based}** — with the memoryless + maximin controls.
Memory and density are each toggled independently on the *same* task, so neither can be
blamed for the other.

---

## 4. The environment suite (the 2×2 + the toggles)

|                         | **sparse reward**                                              | **dense reward**                                                              |
|-------------------------|----------------------------------------------------------------|-------------------------------------------------------------------------------|
| **memory NOT required** | easy control                                                   | trivial control (full-obs POPGym control; full-obs MysteryPath)               |
| **memory REQUIRED**     | T-Maze, **MysteryPath (sparse)**, **MiniGrid-Memory/S13**, bsuite-MemoryLength, Memory Maze | **← the under-studied cell:** **POPGym Velocity/Pos-Only CartPole/Pendulum**, **MysteryPath (dense via `reward_fall_off`)**, POPGym-Arcade dense tasks |

**"S13, MysteryMaze, what else" — the concrete list:**
- **MiniGrid-MemoryS13** — sparse, memory + exploration *entangled* (random spawn). Our
  "everything entangled" point; keep as the messy real-world anchor.
- **MysteryPath-Grid** — the **toggle env**: sparse (goal-only) ⇄ dense (`reward_fall_off=-0.1`
  off-path penalty, optimal return still 1.0), same memory demand. This carries the headline
  within-task experiment.
- **POPGym Velocity-/Position-Only CartPole & Pendulum** — **dense + memory required**, vector
  obs, cheap. The clean dense-memory test bed ([POPGym](https://arxiv.org/abs/2303.01859)).
- **POPGym-Arcade** — pixel; **full/partial-obs toggle** = the memory axis held against fixed
  reward ([2503.01450](https://arxiv.org/abs/2503.01450)). "CNN encoder everywhere."
- **Memory Maze** — 3D, *purpose-built to isolate long-term memory from exploration*
  ([2210.13383](https://arxiv.org/abs/2210.13383)); the gold-standard memory-isolation env if
  we want a heavyweight point.
- **bsuite-MemoryLength** — sparse, *tunable* memory length N ([bsuite](https://arxiv.org/abs/1908.03568));
  the controllable sparse-memory ruler.
- **MIKASA** — a memory benchmark explicitly *disentangling memory from unrelated challenges*
  ([2502.10550](https://arxiv.org/abs/2502.10550)); cite as methodological precedent.

Minimum viable suite for the paper: **MysteryPath (sparse⇄dense toggle)** as the headline,
**POPGym Velocity-Only control** as the independent dense-memory confirmation, **POPGym-Arcade
full/partial** to certify the memory axis, **S13** as the entangled real-world anchor.

---

## 5. Paper outline (AAAI)

1. **Introduction.** The conflation: memory benchmarks are sparse, so "memory needs
   exploration" is untested vs "sparse needs exploration." Contributions: (i) memory ⊥
   reward-density, with a disentangled suite; (ii) the bonus effect *flips* with reward
   density holding memory fixed; (iii) mechanism = densification (benefit) + policy bias
   (cost), isolated via a potential-based control; (iv) a practical rule: don't add
   exploration bonuses to dense-reward memory tasks.
2. **Related work.** Memory-RL benchmarks (POPGym, Memory Gym, Memory Maze, MIKASA, bsuite);
   intrinsic motivation / bonus-based exploration (RND, ICM, NovelD, E3B; Taïga's dense-game
   warning); reward shaping & policy invariance (Ng 1999, PBIM, GRM, noisy-TV); the decoupling
   tradition (Ni et al. 2023 — memory vs credit assignment). **Gap:** no one decouples memory
   from *reward density* for exploration.
3. **Two orthogonal axes.** Formalize memory-requirement and reward-density; argue
   independence; give the POPGym velocity-only existence proof.
4. **Hypothesis & mechanism.** H + the benefit-decays/cost-persists argument; tie to PBIM.
5. **A disentangled benchmark.** The toggles (reward-density, observability), the controls
   (memoryless cell, maximin, PBIM bonus, λ sweep), the env suite (§4).
6. **Experiments.**
   - 6.1 Sparse memory: bonus helps (replicate the trivial baseline).
   - 6.2 Dense memory (same task): bonus neutral/hurts — **the flip**.
   - 6.3 Mechanism: `explained_variance` gain vs reward density; densification shrinks.
   - 6.4 Isolating the cost: raw vs potential-based (PBIM) bonus on the dense task.
   - 6.5 Behavioral analysis: penalty-state visitation / noisy-TV under dense+penalty reward.
   - 6.6 Across the cell zoo + across envs (generality).
7. **Discussion.** "Memory tasks don't need exploration — *sparse* tasks do." Practical
   guidance; when (if ever) a bonus helps a dense memory task; limitations.
8. **Conclusion.**

---

## 6. Immediate next experiment (turns hypothesis → headline)

**Dense-MysteryPath toggle**, reusing the locked 20M HPs (e3b_idm, λ=0.03, ck=64, lr=1e-4):
`{reward: sparse, dense} × {none, e3b_idm}` × cells × seeds, where *dense* = MysteryPath with a
**−0.1 off-path penalty** (`reward_fall_off=-0.1`, env_kwargs, verified). Same maze, same memory
demand, **same optimal return (1.0)** — density without a magnitude confound. The prediction:
e3b helps sparse, stops helping / hurts dense. One cheap run; flips the whole paper from
"trivial" to "mechanistic."

---

## Sources
- Taïga et al., *On Bonus-Based Exploration Methods in the ALE* — https://arxiv.org/abs/2109.11052
- PBIM, *Potential-Based Intrinsic Motivation* — https://arxiv.org/abs/2410.12197
- *Potential-Based Reward Shaping for Intrinsic Motivation* — https://arxiv.org/abs/2402.07411
- POPGym — https://arxiv.org/abs/2303.01859 · POPGym-Arcade — https://arxiv.org/abs/2503.01450
- Memory Maze — https://arxiv.org/abs/2210.13383
- MIKASA / Memory-Benchmark-Robots — https://arxiv.org/abs/2502.10550
- bsuite — https://arxiv.org/abs/1908.03568
- Ni et al., *When Do Transformers Shine in RL?* — https://arxiv.org/abs/2307.03864
