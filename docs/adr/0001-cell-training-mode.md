# ADR 0001: Cell Training Mode Selection

**Status:** Approved
**Date:** May 2026
**Decision-maker:** Jai

---

## Context

Phase A requires selecting a training mode for advanced linear-attention architectures (Gated DeltaNet, mLSTM, RWKV-7) operating under on-policy Proximal Policy Optimization (PPO) with Truncated Backpropagation Through Time (TBPTT) chunks.

The choice is between two paths:

1. **Hardware-accelerated parallel prefix scan** — the published training mode for these cells in language-modeling contexts. Reformulates state transitions as associative operators and computes hidden states across a sequence in O(log N) parallel steps using fused CUDA kernels. Maximum throughput.

2. **Single-step recurrent training** — process each timestep sequentially, similar to traditional LSTM/GRU training. O(N) wall-clock cost, but allows step-by-step gradient clipping and bounded gradient flow.

The decision rests on whether parallel-scan training is stable under the high-variance gradients characteristic of on-policy RL.

---

## Question

Are there published reports of training instability when running Gated DeltaNet, mLSTM, or RWKV-7 under on-policy PPO with TBPTT chunks?

---

## Findings

**Yes — widespread, structural instability is documented across all three architectures under PPO with parallel-scan training.**

### Failure mode 1: Numerical degradation in matrix-memory cells (mLSTM, Gated DeltaNet)

Matrix-memory architectures accumulate state via covariance/outer-product updates that lack the bounded saturation of LSTM cells. The condition number of the memory matrix can degrade rapidly across timesteps. When PPO's high-variance advantage signals propagate backward through these dense matrices, the gradients interact multiplicatively and explode.

The literature explicitly cites "numerical precision issues during reinforcement learning (RL) training" as a primary blocker for matrix-memory adoption in dynamic environments. The dedicated normalizer state introduced in mLSTM is insufficient to counteract the extreme gradient variance from the PPO loss.

### Failure mode 2: Logarithmic error compounding in parallel scans

Parallel prefix scans compute gradients logarithmically across sequence chunks. This is mathematically elegant for stationary supervised learning but compounds numerical errors logarithmically when chunks contain:

- Abrupt episodic resets
- Extreme reward spikes
- Highly exploratory, near-random actions

A single massive outlier in the advantage estimation can corrupt the entire backward pass. If the sequence requires negative eigenvalues in state-transition matrices for state tracking, on-the-fly recomputation during the backward scan can lead to catastrophic numerical divergence, filling memory matrices with non-numeric values and permanently destroying the policy.

### Failure mode 3: Non-stationary gating disruption (Gated DeltaNet)

Gated DeltaNet's data-dependent gating mechanisms shift immediately and violently when PPO executes aggressive updates based on localized reward spikes. Because the parallel scan computes downstream outputs based on these volatile gates, a single poorly-conditioned chunk can produce "massive activations" that corrupt recurrent hidden states across chunk boundaries.

Researchers have introduced *Preconditioned DeltaNet* to address the curvature of the least-squares loss, explicitly citing the need to address "training instabilities we observed when training" linear recurrences. Successful applications in continuous control robotics have required heavily regularized forgetting gates and strictly controlled chunkwise updates.

### Failure mode 4: Algorithm abandonment (RWKV-7)

For RWKV-7 specifically, the literature reveals a mathematically formalized trade-off between the model's expressive capacity (its ability to recognize regular languages via non-diagonal Householder transformations) and operational stability under high-entropy inputs.

The severity of this instability is best evidenced by the lengths to which researchers go to *avoid* using standard PPO when training RWKV-7:

- **EGGROLL** (Evolution Strategies framework) is explicitly justified by its ability to optimize RWKV-7 architectures on objectives "for which gradients are unavailable or noisy," avoiding backpropagation through time entirely.
- **CISPO** (a custom RL algorithm) clips importance sampling weights rather than token updates, directly acknowledging the inherent instability of standard RL paradigms when applied to sub-quadratic memory structures.

These workarounds are not folkloric; they are published responses to documented instabilities.

### Synthesis

The convergence of empirical evidence and architectural analysis establishes definitively that running advanced linear-attention cells (Gated DeltaNet, mLSTM, RWKV-7) under on-policy PPO utilizing hardware-accelerated parallel scans over chunked trajectories is prone to catastrophic instability.

The primary failure modes are structural and geometric:

1. Matrix-based memory states lack bounded saturation, susceptible to gradient explosion under high-variance PPO advantages.
2. Parallel prefix scans exacerbate this by compounding gradients logarithmically across sequences containing extreme non-stationarity, episodic boundaries, and exploratory noise.
3. Rapid shifting of data-dependent gates during aggressive policy updates produces massive residual stream activations that irreversibly corrupt hidden state memory passed between trajectory chunks.

---

## Decision

**Implement single-step recurrent training as the Phase A default.**

The parallel-scan training path is stubbed but not implemented in Phase A. The IGAMCell class will expose a `forward_step(x_t, h_t, W_t, n_t)` method for single-step recurrent execution; the parallel-scan equivalent will raise `NotImplementedError`.

---

## Rationale

The empirical and theoretical evidence dictates that hardware-aware parallel scans across linear-attention matrix states are fundamentally incompatible with the raw, high-variance gradients of standard PPO without extensive custom preconditioning or gradient-free workarounds.

Accepting the ~3× wall-clock penalty of serial recurrent execution serves as a necessary derisking maneuver for Phase A. Serial execution:

- **Ensures numerical stability** by preventing logarithmic error compounding.
- **Provides bounded gradient flow** through per-step gradient clipping.
- **Guarantees reliable temporal credit assignment** by isolating gradient explosions before they propagate across the entire sequence chunk.
- **Aligns with the existing lmu_ppo TBPTT infrastructure** — single-step recurrent training is the path of least resistance from the existing thesis codebase.

The wall-clock penalty is real but the implementation cost is *lower* than parallel-scan; this ADR pulls toward the architecture that is easier to build, not harder.

---

## Consequences

- **Increased compute budget for Phase A.** Benchmarks at 5M-25M steps × 3 seeds × multiple ablations may run 3× longer than parallel-scan estimates. Plan accordingly.
- **Abandons the memory optimization benefits of fused-kernel parallel scans.** Phase A will not benefit from hardware-aware throughput optimizations.
- **Requires focused engineering optimization of the recurrent step logic.** Mitigations include: amortizing per-step Python overhead via `torch.compile` or jitted JAX equivalents, batching across parallel envs to maximize GPU utilization within a single timestep, and minimizing host-device synchronization in the rollout loop.
- **Introduces a Phase B revisit obligation.** Once the cell, encoder, and policy have stabilized in their Phase A configuration, parallel-scan training will be retried on a small benchmark (Week 18). Some of the documented instabilities are cold-start pathologies that may not recur in a warm-started configuration. If parallel-scan proves stable in Phase B, throughput is recovered for Phase C scaling. If it remains unstable, we have concrete evidence for the paper that the instability is not a cold-start artifact.

---

## Alternatives considered

### Alternative 1: Use parallel-scan with aggressive gradient clipping

Reasoning: gradient clipping at chunk boundaries might tame the worst gradient explosions and recover throughput.

Rejected because: the literature documents that the failure mode is not just gradient magnitude — it's matrix conditioning. Clipped gradients still produce ill-conditioned state matrices when the underlying advantage signal is highly variable. Patching the symptom does not address the structural incompatibility.

### Alternative 2: Use parallel-scan with PPO modifications (CISPO, importance-weight clipping)

Reasoning: published workarounds exist; adopt them rather than reinventing.

Rejected because: CISPO and similar algorithms are themselves an active research area with their own failure modes. Adopting them would couple the IGAM architecture to a custom RL algorithm, complicating ablations and making the paper's contribution harder to isolate. Single-step recurrent + standard PPO is the cleaner experimental setup.

### Alternative 3: Switch to Evolution Strategies (EGGROLL-style)

Reasoning: ES bypasses backprop entirely, sidestepping all parallel-scan instability.

Rejected because: ES is dramatically less sample-efficient than PPO on the target benchmarks, has no established Craftax/POPGym baselines for comparison, and would force the paper into an entirely different experimental tradition. Out of scope for the master's thesis.

### Alternative 4: Use a non-matrix-memory cell that doesn't have these issues (LSTM, GRU, S4D)

Reasoning: avoid the problem class entirely.

Rejected because: the IGAM contribution depends on the matrix-memory associative read structure. Switching to LSTM/GRU forfeits the central architectural commitment. S4D is a viable fallback if matrix-memory cells fail entirely in Phase A, but the deep research argues matrix-memory dominates S4D on memory-intensive POPGym tasks under standard supervised training, so we should at least try matrix-memory under single-step recurrent before falling back further.

---

## Revisit triggers

This ADR should be reopened if:

1. **Single-step recurrent training itself proves unstable** in Phase A. This would indicate a deeper architectural problem requiring a cell-class change, not just a training-mode change.
2. **The ~3× wall-clock penalty proves prohibitive** for the Phase A timeline. Mitigation effort would shift toward `torch.compile` / JAX optimization before considering parallel-scan.
3. **A new paper publishes a working PPO + parallel-scan + matrix-memory result.** If a published method demonstrates stability, port their stabilization technique and revisit.
4. **The Phase B revisit experiment (Week 18) shows parallel-scan is stable** in warm-started configurations. Update this ADR and proceed to use parallel-scan for Phase C scaling.