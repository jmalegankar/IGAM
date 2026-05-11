# ADR 0004: Source of Data-Dependent Gates in the IGAM Cell

**Status:** Approved
**Date:** May 2026
**Decision-maker:** Jai

---

## Context

The README's Architectural Commitments specified the IGAM cell as Gated DeltaNet with:

> α_t = σ(W_α h_t), β_t = σ(W_β h_t), data-dependent gates.

and a state shape of `(h, W, n)`, suggesting a small recurrent hidden state `h_t` evolving alongside the matrix memory. This conflicts with the canonical Gated DeltaNet (Yang et al. 2024) which computes gates directly from the input `x_t` and has state `(W, n)`.

The ambiguity was not consciously chosen — it was inertia from the lmu_ppo LMU cell's structure (which has both `h` and `m`). Resolving it before implementation begins, because every downstream design choice depends on it: rollout buffer state shape, episode-mask machinery, gradient flow patterns, and the rhetorical "the cell is a pure key-value associative memory" claim that motivates IGAM.

---

## Question

Should the data-dependent gates `α_t` and `β_t` be computed from a separate recurrent hidden state `h_t`, or directly from the encoded observation `φ(o_t)`?

---

## Findings

### Design A — Gates from `φ(o_t)` (canonical Gated DeltaNet)

State: `(W_t, n_t)`. No separate `h_t`. Gates and queries projected directly from the encoder output:

```
α_t = σ(W_α φ(o_t))
β_t = σ(W_β φ(o_t))
q_t = W_Q φ(o_t)
k_t = W_K φ(o_t)
v_t = W_V φ(o_t)
```

This matches Yang et al. 2024 Gated DeltaNet, Schlag 2021 DeltaNet, RWKV-7, and every cell in the `flash-linear-attention` library.

**Buys:**
- Canonical formulation. Published stability analysis (ADR 0001), ablations, and reference implementations all assume this.
- Simpler state. One fewer object to initialize, mask at episode boundaries, and serialize in the rollout buffer.
- Cleaner mathematics. The cell is a pure key-value associative memory; the innovation `δ_t = v_t − W_{t-1} k_t / ||k_t||²` has an unambiguous interpretation.
- Preserves the central rhetorical claim: "the same predictive-coding object is the cell, the lifelong bonus, and the FSQ key." With a separate `h_t` the cell isn't purely an associative memory anymore.

**Costs:**
- Gates have no direct access to past context — they can't condition on what's already stored in `W_t`. Has to be communicated indirectly through the next `y_t = q_t W_t` read.

### Design B — Gates from a separate `h_t`

State: `(h_t, W_t, n_t)`. `h_t` evolves via some auxiliary recurrence (LSTM-cell, GRU-cell, or RWKV-7 token-shift). Gates project from `h_t`.

**Buys:**
- Gates can implement context-dependent write strategies. If `h_t` encodes "I just saw a key," gates can write aggressively.
- More expressive class of dynamics overall.

**Costs:**
- Adds complexity. Two recurrent states, two initializations, two episode-mask points, two terms in the buffer.
- `h_t` needs its own update rule — yet another design decision (LSTM cell? GRU cell? token-shift?), each with its own failure modes.
- Second place where instability can arise. ADR 0001's analysis is on Design A; picking Design B puts us off the literature's well-trodden stability path.
- The lmu_ppo experience supports the simpler design: in `lmu_t.py`'s LMUCell the `h_t` recurrence introduced real complexity (E_h projection, spectral norm constraints, separate gradient flow), and several rejected design choices in the thesis Appendix D come from exactly this coupled-state regime.

---

## Decision

**Adopt Design A. Cell state is `(W_t, n_t)`. Gates and projections compute directly from `φ(o_t)`.**

Concrete spec:

```
α_t = σ(W_α φ(o_t))                            # per-head scalar decay
β_t = σ(W_β φ(o_t))                            # per-head scalar write strength
q_t = W_Q φ(o_t),   k_t = W_K φ(o_t),   v_t = W_V φ(o_t)
q̃_t = L2norm(q_t),  k̃_t = L2norm(k_t)

δ_t = v_t − W_{t-1} k̃_t                        # innovation (side output)
W_t = α_t · W_{t-1} · (I − β_t k̃_t k̃_t^T) + β_t · v_t k̃_t^T
n_t = α_t · n_{t-1} + β_t · k̃_t
y_t = q̃_t^T W_t / max(|q̃_t^T n_t|, ε)
```

The actor-critic head receives `concat(φ(o_t), y_t)` — structurally the same shape as lmu_ppo's `cat([h, m.mean(dim=1)])`, with `φ(o_t)` in the role `h` played and `y_t` in the role `m_pooled` played.

---

## Rationale

**Preserves the rhetorical claims.** The cell is still a pure key-value associative memory. The innovation `δ_t` is unambiguously the predictive error of the cell's own memory model. The "one mathematical object" framing survives.

**Preserves the lmu_ppo dynamic-query insight without inheriting its complexity.** `q_t` still depends on the current observation through `W_Q φ(o_t)` — the query is state-dependent. We don't need a separate `h_t` to get this; it falls out of the linear-attention formulation naturally.

**Stays on the canonical path.** ADR 0001's stability analysis assumes Design A. Sticking with it means the published failure modes and mitigations apply directly. Design B would put us in an unexplored regime where stability properties have to be discovered empirically.

**The lmu_ppo experience supports it.** Several thesis Appendix D failures come from coupled-state pathologies that Design A simply doesn't have.

---

## Consequences

- **Cell state is `(W_t, n_t)` everywhere.** Rollout buffer, episode-mask helper, checkpoint serialization all simplify.
- **No `h_t` initialization decision.** One fewer hyperparameter to tune.
- **Actor-critic head input is `concat(φ(o_t), y_t)`** — not `concat(h_t, y_t)`. Total dim: `d_φ + hidden_size`.
- **Phase A 2×2 ablation gradient is cleaner** — every matrix-memory cell (`LinearTransformer`, `DeltaNet`, `mLSTM`, `Gated DeltaNet`/IGAM) has the same `(matrix_state, normalizer)` shape, so ablations don't confound "state structure" with "cell type."
- **Gradient flow is simpler to debug.** One recurrent path through `W_t`; no parallel path through `h_t`.

---

## Alternatives considered

### Design B — gates from separate `h_t`

Rejected (see Findings). The expressivity gain isn't worth the added complexity, the loss of the "pure associative memory" rhetorical framing, and the divergence from the canonical literature's stability analysis.

### Hybrid — gates from `concat(φ(o_t), y_{t-1})`

Compromise: include the previous *read output* `y_{t-1}` in the gate input, giving gates access to "what the memory just said about my current context" without introducing a separate recurrent state. `y_{t-1}` is computed anyway, so the only cost is a slightly wider gate projection.

Deferred — kept as a **Phase A revisit candidate** if Design A proves insufficient. Principle: start with the simplest version that has a chance of working, ablate to verify it's sufficient, upgrade only if ablation shows otherwise.

---

## Revisit triggers

This ADR should be reopened if:

1. **Phase A ablation shows the gates can't distinguish "first visit" from "tenth visit" to similar states.** Symptom: the cell writes aggressively in both, leading to W_t saturation. Upgrade path: switch to the hybrid `concat(φ(o_t), y_{t-1})` form.
2. **A published paper demonstrates that gates-from-`h_t` materially help on POPGym / Craftax / MiniGrid-Memory benchmarks.** Update ADR and adopt their formulation.
3. **An IGAM-specific instability appears that traces to gate input.** Investigate before assuming this ADR is wrong.
