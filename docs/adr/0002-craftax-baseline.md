# ADR 0002: Phase B Craftax-Symbolic MFRL Baseline

**Status:** Approved
**Date:** May 2026
**Decision-maker:** Jai

---

## Context

Phase B requires a rigorous, unassailable success criterion evaluated against the Craftax-symbolic 1M environment. The Craftax benchmark has seen rapid SOTA progression in 2024–2025, with reported results spanning a wide range (2.3% to 69.66% reward at 1M steps depending on architecture).

To ensure an apples-to-apples architectural evaluation, the Phase B comparator must be the strongest published model-free, recurrent PPO policy (PPO-RNN). This explicitly requires excluding:

- Transformer-based world models
- Dyna-style imagination loops
- Auxiliary world-model or pixel-reconstruction losses
- Convolutional networks operating on pixel observations
- Off-policy algorithms (Q-learning, actor-critic with replay)

The architecture must be a pure model-free policy combining PPO with a recurrent network operating on Craftax's 8268-dimensional symbolic observation vector.

---

## Question

What is the strongest published number on Craftax-symbolic 1M for a model-free PPO + recurrent policy (no world model, no transformer, no auxiliary world-model loss, no pixel CNN, no off-policy algorithm)?

---

## Findings

An exhaustive analysis of the Craftax literature yields the verified baseline metrics from Matthews et al. (2024) and the official open-source Craftax leaderboard. The vast majority of recent high-performing agents are disqualified under the strict Phase B constraints. Accurate baseline isolation requires systematically dissecting and discarding these confounders.

### Disqualified results

#### Dedieu et al. (2025) MBRL — 67.42% / 69.66%
**Disqualified: model-based + transformer world model + pixel observations.**

Achieves unprecedented 67.42% (later 69.66%) at 1M Craftax-classic steps via a "Dyna with warmup" loop training on real + imagined trajectories. Uses a transformer world model with nearest-neighbor tokenizer on image patches and "block teacher forcing" for joint future-token reasoning. Operates on pixels.

#### Simulus (Cohen et al., 2025) — 6.6%
**Disqualified: model-based + tokenized world model.**

Token-based, planning-free world model. Avoids imaginary rollouts but is fundamentally a world model with multi-modal tokenization, prioritized world-model replay, and intrinsic motivation targeting epistemic uncertainty.

#### Efficient MBRL — 5.4%
**Disqualified: model-based + pixel observations.**

#### Dedieu et al. (2025) MFRL — 55.49% (Craftax-classic) / 4.63% (full Craftax)
**Disqualified: pixel observations + CNN encoder.**

A genuinely model-free PPO baseline, but operates on 63×63 pixel inputs via CNN+RNN. The authors explicitly note their MFRL number uses image input, contrasting it with prior baselines that used symbolic input. Invalid for the symbolic-only Phase B comparison.

#### Moon et al. (2024) — 46.91%
**Disqualified: stateless CNN policy on pixels.**

Stateless modified Impala ResNet operating on pixels without frame stacking. Lacks the required recurrent component and operates in the wrong observation modality.

#### PQN (Gallici et al., 2025)
**Disqualified: off-policy Q-learning, not PPO.**

Simplified Deep Online Q-Learning. Highly competitive with PPO-RNN and trains up to 50× faster than DQN, using LayerNorm and vectorized envs to stabilize training. But it is an off-policy Q-learning framework, not PPO.

### The qualifying baseline

After stripping away model-based architectures (DreamerV3, Simulus, IRIS, Dedieu MBRL), CNN-pixel models (Dedieu MFRL, Moon), and off-policy algorithms (PQN), the foundational baselines are those established in the original Craftax paper (Matthews et al., 2024) and maintained on the official Craftax-Symbolic-v1 leaderboard.

The original Craftax paper provides exhaustive, rigorously tuned baseline evaluations across both 1B and 1M step budgets, with PPO-RNN evaluated across 10 independent seeds. At the 1M step boundary on symbolic observations, the published results are:

| Algorithm | Architecture | Observation | Reward (% of max) |
|---|---|---|---|
| Simulus | Model-Based (Tokenized) | Pixel | 6.6% |
| Efficient MBRL | Model-Based | Pixel | 5.4% |
| **PPO-RNN** | **Model-Free (Recurrent)** | **Symbolic Vector** | **2.3%** |
| PPO | Model-Free (Stateless) | Symbolic Vector | 2.2% |
| RND | Model-Free + Intrinsic | Symbolic Vector | 2.2% |
| ICM | Model-Free + Intrinsic | Symbolic Vector | 2.2% |

**The strongest published number for a pure model-free PPO algorithm equipped with a recurrent policy on Craftax-symbolic 1M is 2.3% of maximum reward (226).**

The PPO-RNN gain over stateless PPO is only 0.1pp — within statistical noise. Standard intrinsic-motivation baselines (RND, ICM) match or marginally underperform stateless PPO. Without auxiliary representation learning or world-model machinery, the recurrent policy is unable to distill the high-dimensional symbolic vector into actionable long-term memory within 1M environment interactions.

### Interpretation: the bottleneck is exploration, not memory

The 0.1pp PPO-RNN-vs-PPO gap is the most informative number in this analysis. It indicates that no recurrent policy is encoding useful long-horizon state at this sample budget — the agent has not reached the deeper achievements (mining, smelting, dungeons) that create memory-relevant dependencies. Both stateless and recurrent agents are stuck at the basic procedural achievements (collect saplings, place crafting tables).

**Memory becomes the dominant bottleneck only after exploration cracks past roughly 10% reward.** Below that threshold, no memory architecture can demonstrate value because the agent isn't reaching states where memory matters.

This interpretation directly shapes Phase B's tiered success criteria.

---

## Decision

**The targeted Phase B baseline is 2.3% of maximum reward on Craftax-symbolic 1M.**

Phase B success is defined in tiers relative to this baseline (per `plan.md` Weeks 15–17):

- **Tier 1 (publication-worthy minimum):** ≥5% reward at 1M steps. Greater than 2× the PPO-RNN baseline; clears all standard intrinsic-motivation methods.
- **Tier 2 (strong result):** ≥10% reward at 1M steps. Crosses the threshold where memory starts mattering; first model-free symbolic-PPO method to reach the memory-relevant achievement tier.
- **Tier 3 (headline result):** ≥20% reward at 1M steps. Substantially exceeds model-based methods (Simulus 6.6%, Efficient MBRL 5.4%) without world-model machinery.
- **Tier 4 (stretch):** Beyond 20% — approaching DreamerV3-class pixel results without pixels.

---

## Rationale

The 2.3% reward metric is the official, verified published benchmark for the PPO-RNN algorithm on the Craftax-1M symbolic leaderboard. Setting this as the Phase B comparator ensures a mathematically pure architectural evaluation. Overcoming the 2.3% threshold definitively validates the sample-efficiency and representational superiority of the proposed Phase B architecture relative to standard recurrent baselines in open-ended, sparsely-rewarded RL environments.

The tiered framing reflects the exploration-vs-memory bottleneck observation: rewards may compound non-linearly past the ~10% threshold, and we do not yet know where IGAM will land. Tiered criteria allow Phase B to define success at multiple levels of ambition without prematurely committing to a single hard target.

---

## Consequences

- **The bar for "this matters scientifically" is genuinely low** in absolute terms (anything >5% beats the published ceiling). Mitigation: lean on the *ablation table* rather than the headline number. A 5% headline + clean ablations isolating cell, auxiliary, and episodic contributions is more publishable than a 15% headline without ablations.

- **Phase B's exploration-bottleneck framing depends on the 2.3% baseline being current.** If a recent paper has published a stronger MFRL+symbolic+1M result we missed, the framing shifts. **Mitigation: re-verify the leaderboard at the start of Week 9 (Phase B start)** before committing to the Tier 1/2/3 thresholds. Document any new baselines in an ADR amendment.

- **The Phase B ablation table must isolate contributions.** Required ablation rows:
  - Full IGAM (cell + IDM + JEPA + episodic + lifelong)
  - − JEPA (IDM-only auxiliary)
  - − episodic (lifelong only)
  - − lifelong (episodic only)
  - − FSQ (continuous + kNN episodic on raw φ)
  - − IDM + JEPA (random encoder, like lmu_ppo random-φ baseline)
  - LSTM with full IGAM auxiliary+episodic (isolates cell contribution)
  - IGAM cell with no auxiliary or episodic (isolates auxiliary+episodic contribution)

  This level of ablation rigor is required because the headline number alone is not the contribution; the *components* are the contribution.

- **The "memory-relevant tier" claim (≥10%) requires evidence.** If IGAM hits Tier 2, the paper claims that crossing 10% reward is the regime where memory begins mattering. This claim must be supported by an explicit ablation: at Tier 2 reward levels, IGAM-with-cell vs. IGAM-with-LSTM should diverge meaningfully. If they don't diverge, the "memory matters now" framing is wrong and the paper's narrative shifts.

---

## Alternatives considered

### Alternative 1: Set a single hard target (e.g., 10%)

Reasoning: tiered criteria are a hedge; commit to a single ambitious target.

Rejected because: the exploration-vs-memory bottleneck means rewards likely compound non-linearly. We do not yet know whether the threshold for "exploration is solved" is at 5%, 10%, or 20%, and committing to a single target before that threshold is empirically located risks either underclaiming success or setting an unreachable bar. Tiered criteria adapt to whichever regime IGAM lands in.

### Alternative 2: Compare to model-based methods (Simulus 6.6%, Efficient MBRL 5.4%)

Reasoning: the most informative comparison is to the strongest published methods, regardless of architecture class.

Rejected because: model-based methods have access to auxiliary signals (world-model gradients, imagination rollouts) that fundamentally change the optimization problem. Comparing IGAM directly to these inflates IGAM's apparent contribution if it wins (we'd be claiming superiority over methods that should win on those signals) and unfairly penalizes IGAM if it loses (we'd be holding a model-free method to a model-based bar). Including these methods in the comparison table for context is fine; using them as the primary baseline is not.

### Alternative 3: Use pixel-based MFRL baselines (Dedieu MFRL 55.49%, Moon 46.91%)

Reasoning: the strongest model-free numbers come from pixel-based architectures.

Rejected because: pixel and symbolic observation modalities are not comparable. Pixel agents benefit from spatial inductive biases (CNN feature sharing, locality) that don't apply to symbolic vectors, and conversely symbolic agents have access to precise discrete entity information that pixel agents must learn to extract. Comparing across modalities measures encoder priors, not algorithmic contribution.

### Alternative 4: Run our own PPO-RNN baseline to verify 2.3%

Reasoning: don't trust the published number; replicate it ourselves.

Partially adopted: the Phase B plan includes "Vanilla PPO + LSTM with our IDM+JEPA auxiliary" as an internal ablation, which provides some replication signal. However, full PPO-RNN replication on Craftax-symbolic at the original paper's compute budget would consume ~2 weeks for marginal information gain (the published number is from 10 seeds with rigorous tuning). Not worth the time. If our internal comparator deviates substantially from 2.3%, that is itself diagnostic information.

---

## Revisit triggers

This ADR should be reopened if:

1. **A new paper publishes a stronger MFRL+symbolic+1M result.** Mandatory check at start of Week 9 (Phase B). Update tiers accordingly.

2. **The Craftax leaderboard introduces a new evaluation modality** that is more relevant to the Phase B contribution than the current 1M step budget.

3. **Our internal Phase B replication of PPO-RNN deviates substantially from 2.3%.** If our baseline run scores e.g., 4% under different hyperparameters, the published number may not be the relevant comparator and the tiers need recalibration.

4. **Phase B results land between tiers in ways that suggest the tier boundaries are mis-calibrated.** For example, if multiple ablations cluster around 8-9% reward, the Tier 1/Tier 2 boundary at 10% is ill-placed and should be reconsidered.