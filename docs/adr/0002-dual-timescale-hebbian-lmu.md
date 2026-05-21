# ADR 0002: Dual-Timescale Hebbian LMU (DTH-LMU)

**Status:** Proposed  
**Date:** May 2026  
**Decision-maker:** Jai

---

## Context

The Phase A ablation (ADR 0001) characterises what `SelectiveLMU` *does well*: tasks
where the recent sliding window is sufficient (RepeatPrevious). The GEX replication
experiments now expose three diagnostic failures that share a common cause:

| Task | Failure | Why LegT fails |
|---|---|---|
| AutoencodeMedium | Content recall by episode index | M = Σ cₙPₙ(τ) — inversion under-determined when inputs overlap in polynomial space |
| CountRecallMedium | Cumulative statistics over full episode | Fixed window θ discards observations outside it; degree-0 LegT coefficient is a *windowed* mean, not a running sum |
| BattleshipEasy | Spatial key→value lookup by (r,c) | Polynomial basis has no spatial inductive bias; grid cell visited 80 steps ago is indistinguishable from one visited 5 steps ago |

All three failures are **associative memory** failures: the task requires storing
`(key → value)` pairs and retrieving by key. LegT is optimal for *temporal
dynamics* (what is the shape of the signal in the recent window?) but structurally
wrong for *arbitrary association* (what was at key k?).

The fix cannot be purely a wider window or more polynomial modes. It requires a
complementary mechanism that is associative by construction.

---

## Decision

Introduce the **Dual-Timescale Hebbian LMU (DTH-LMU)**: a cell with three
co-operating memory systems that together cover the full spectrum from moment-to-moment
dynamics to full-episode episodic recall.

```
                     ┌──────────────────────────────────────────────────┐
                     │                  DTH-LMU step                    │
  x_t ──────────────►│                                                  │
                     │  u_x = x · normalize(e_x)   (shared encoder)     │
                     │                                                  │
                     │  [Fast cell]  m_f ← A_f m_f + B_f u_x          │
                     │               (LegT, θ_f small, gated write)    │
                     │               y_f  = C_f(h) · m_f               │
                     │                   ↓ innovation                   │
                     │  [Slow cell]  m_s ← A_s m_s + B_s u_x          │
                     │               (LegT, θ_s large, no gate)        │
                     │               y_s  = C_s(h) · m_s               │
                     │                                                  │
                     │  [Matrix M]   k = φ(W_K x),  v = W_V x         │
                     │               M ← M + v⊗k / d_k                │
                     │               r = M · φ(W_Q h)                  │
                     │                                                  │
                     │  h_new = tanh(W_x x + W_h h + W_mf y_f         │
                     │               + W_ms y_s + W_r r)               │
                     └──────────────────────────────────────────────────┘
```

---

## Architecture

### State

| Key | Shape | Description |
|---|---|---|
| `h` | `(B, H)` | Hidden state — unchanged |
| `m_f` | `(B, K_f, D_f, C)` | Fast LMU Legendre coefficients |
| `m_s` | `(B, K_s, D_s, C)` | Slow LMU Legendre coefficients |
| `M` | `(B, d_k, d_v)` | Hebbian matrix memory |

All state components are zeroed on episode boundaries via `apply_episode_mask`.
`M` in particular **must** be zeroed per batch element at episode start — the
outer-product accumulation is within-episode only.

### Forward pass (one time step)

```python
# 1. Shared input encoding
u_x = x * F.normalize(e_x, dim=0)           # (B, C) — shared between fast and slow

# 2. Fast LMU — selective working memory
u_actual_f, _, innov = _compute_write(u_x, E_h_f(h_normed), pool(e_m_f, m_f))
Am_f  = einsum("kij,bkjc->bkic", A_f, m_f) * salience_gate_f(h_normed)
m_f   = Am_f + B_f * u_actual_f              # (B, K_f, D_f, C)
C_f   = normalize(W_query_f(h_normed))       # (B, D_f)
y_f   = mix_f(h_normed) · einsum("bd,bkdc->bkc", C_f, m_f)  # (B, C)

# 3. Slow LMU — pure integrator, no gate
Am_s  = einsum("kij,bkjc->bkic", A_s, m_s)
m_s   = Am_s + B_s * u_x                    # (B, K_s, D_s, C) — writes u_x directly
C_s   = normalize(W_query_s(h_normed))       # (B, D_s)
y_s   = mix_s(h_normed) · einsum("bd,bkdc->bkc", C_s, m_s)  # (B, C)

# 4. Hebbian matrix memory
k     = F.normalize(W_K(x), dim=-1)          # (B, d_k)
v     = W_V(x)                               # (B, d_v)
M     = M + v.unsqueeze(-1) * k.unsqueeze(-2) / d_k  # (B, d_k, d_v)
q     = F.normalize(W_Q(h_normed), dim=-1)   # (B, d_k)
r     = einsum("bkv,bk->bv", M, q)           # (B, d_v)

# 5. Hidden update
h_new = tanh(W_x(x) + W_h(h_prev) + W_mf(y_f) + W_ms(y_s) + W_r(r))
```

---

## Component-by-Component Decisions

### Gate

**Fast cell — keep softsign_sum (or tanh_product), orthogonal W_pre.**

The existing gate is well-motivated for the fast cell. Working memory should be
*selective*: only surprising inputs update the short-term polynomial approximation.
The innovation signal (gate + innov computation) is still the right abstraction here.
Orthogonal W_pre preserves innovation norm and is retained.

**Slow cell — no gate, no E_h, no W_pre. Pure integrator.**

The slow cell's job is to accumulate statistics over the full episode without
filtering. A write gate would prevent rare-but-important events (a specific token
appearing once per episode) from being represented in the slow basis — exactly
what CountRecall requires. Design: `gate_type="none"` with `E_h_s` not
instantiated. The slow cell writes `u_x` directly: `m_s ← A_s m_s + B_s u_x`.
No E_h means no u_h term, no u_m term, no W_pre needed.

**Matrix M — always-write initially (ungated outer product).**

The 1/d_k normalization bounds `||M||_F ≤ T·||v||_max/√d_k`. For T=500,
d_k=64, that is ≈ 7.8·||v||_max — manageable. The hidden state's query W_Q can
learn to suppress low-SNR retrievals via its output projection.

Innovation-gated write (`M ← M + σ(w_g·||innov||) · v⊗k`) is the natural next
step: high prediction error → strong episodic encoding, exactly the hippocampal
novelty-gating mechanism. This is deferred to the first ablation rather than
baked in from the start, so the baseline DTH-LMU behaviour is easy to reason about.

**There is no new gate type.** The innovation-gated write, when added, reuses the
existing `innov` side output from the fast cell — no new gate_type enum value.
It is a scaling factor on the outer product, not a new gate on the polynomial write.

### Multichannel

Both fast and slow cells use the **shared** encoder `e_x ∈ R^C`:
`u_x = x * normalize(e_x)`. One encoder, two ODE integrators. Both memory
states are `(B, K, D, C)` in the multichannel regime.

Rationale for sharing: the "which input channels matter?" question has one answer
per input — splitting it into fast-relevant and slow-relevant channels would
add complexity without clear benefit at this stage. Separate e_x for each cell
is a deferred ablation.

Matrix M uses independent projections `W_K: R^I→R^{d_k}`, `W_V: R^I→R^{d_v}`.
These are full-rank projections from the raw input and are not shared with e_x —
the key/value space for associative recall is separate from the LMU channel
encoding.

### W_query (dynamic readout)

Two **separate** readout projections, both from `h_normed`:

| Projection | Shape | Reads from |
|---|---|---|
| `W_query_f: Linear(H, D_f)` | `(B, D_f)` | Fast LMU Legendre modes |
| `W_query_s: Linear(H, D_s)` | `(B, D_s)` | Slow LMU Legendre modes |
| `W_Q: Linear(H, d_k)` | `(B, d_k)` | Matrix memory keys |

The fast query asks "which recent polynomial mode is relevant right now?"; the
slow query asks "which long-range statistical mode is relevant?". These are
categorically different questions and should not share weights.

Init for both W_query variants: orthogonal with gain=0.01 (matching the existing
W_query init rationale — memory starts with negligible influence and the cell
learns where to point). W_Q: Xavier normal (the matrix retrieval is a different
modality; small-gain init would suppress early retrieval too aggressively).

### Orthogonal W_pre

**Fast cell: retained.** The innovation vector entering the polynomial write has
norm that matters for gradient propagation. Orthogonal parametrisation (Cayley)
preserves this.

**Slow cell: not instantiated.** The bare integrator writes u_x with no
transformation. There is no innovation vector to preserve.

**Matrix M: not applicable.** L2-normalisation of k (`φ = F.normalize`) is the
analogous norm-preservation mechanism for the key. The value v = W_V(x) is
Xavier-init and does not require orthogonal constraints.

### Slow cell timescale: θ_s = θ²/D (derived, not free)

The slow window is not a free hyperparameter. It is fully determined by the fast
cell's own resolution ratio:

```
ρ = θ/D          (steps per Legendre mode — the fast cell's temporal resolution)
θ_s = θ · ρ = θ²/D
```

**Geometric interpretation:** the fast and slow windows are consecutive rungs of
a log-scale hierarchy with ratio ρ. Each cell covers an equal span in log-time:
`log(θ_s/1) / log(θ/1) = log(θ²/D) / log(θ) = 2 − log_θ(D)`. For D ≪ θ this
is ≈ 2 — both cells cover roughly the same number of log-time octaves.

**Validity condition:** requires D < θ (i.e. ρ > 1). At D = θ the slow window
equals the fast window; at D > θ the formula gives θ_s < θ (nonsense). Enforced
with an assertion at construction time.

**Episode-coverage constraint:** the slow window covers a full episode of length
T when D ≤ θ²/T. For θ = 100, T = 500: D ≤ 20. For T = 155 (POPGym Hard): D ≤ 64.
This should be checked when choosing D for a given task.

| D | ρ = θ/D | θ_s = θ²/D | POPGym Hard (155) | 500-step episode |
|---|---|---|---|---|
| 8  | 12.5 | 1250 | ✓ | ✓ |
| 16 | 6.25 | 625  | ✓ | ✓ |
| 32 | 3.1  | 312  | ✓ | ✗ |
| 64 | 1.56 | 156  | ✓ (barely) | ✗ |

A `_legs_zoh_matrices()` function (true scale-invariant LegS, no fixed window)
remains a future upgrade. This is a **revisit trigger** (see below).

### Hidden update

```
h_new = tanh(
    W_x · x         +   # direct input path
    W_h · h_prev    +   # standard recurrence
    W_mf · y_f      +   # fast LMU readout
    W_ms · y_s      +   # slow LMU readout (new)
    W_r · r             # matrix retrieval (new)
)
```

`W_mf`, `W_ms`, `W_r` are separate `Linear(C, H)`, `Linear(C, H)`, `Linear(d_v, H)`
projections. They are not combined into a single projection — keeping them separate
lets ablations zero-out individual memory pathways cleanly.

`r` is layer-normed before W_r to prevent the unbounded M accumulation from
dominating h: `r_normed = LayerNorm(d_v)(r)`. This is always on (not a flag).

---

## Why the Math Works

### Matrix retrieval approximation

Given writes `M = (1/d_k) Σ_s v_s ⊗ k_s`, querying with normalised key `k_τ`:

```
M · k_τ = (1/d_k) Σ_s v_s (k_s · k_τ)
         = v_τ · (1/d_k) ||k_τ||²  +  (1/d_k) Σ_{s≠τ} v_s (k_s · k_τ)
         ──────────────────────────    ────────────────────────────────────
              signal                             interference
```

In d_k dimensions, random unit vectors have `E[k_s · k_τ] = 0` and
`Var[k_s · k_τ] = 1/d_k`. For T written entries the interference norm is
`O(T / d_k)`. With d_k = 64 and T = 500: interference ≈ 7.8·||v||_max (bounded
by the same quantity as ||M||_F). The model learns to suppress this residual
via the W_r projection and layernorm on r.

For tasks with well-separated keys (distinct spatial positions for Battleship,
distinct token types for CountRecall, fixed sinusoidal time embeddings for
Autoencode), interference is much lower in practice since trained keys cluster away
from each other.

### Slow cell for counting

The degree-0 Legendre coefficient of the slow LMU (large θ_s) approximates the
mean of the input over the window. With θ_s >> T_episode, the "window" covers the
full episode, so the degree-0 coefficient approximates the *episode-level running
mean*. Higher-degree coefficients track higher-order moments. The slow cell
therefore gives the hidden state access to `E[u_x]`, `Var[u_x]`, etc. over the
full episode — exactly what CountRecall needs.

---

## Neuroscience Grounding

This architecture is a direct implementation of **Complementary Learning Systems
(CLS) theory** (McClelland, McNaughton, O'Reilly 1995):

| Component | Brain analog | Function |
|---|---|---|
| Fast LMU (small θ_f, gated) | Prefrontal working memory | Selective moment-to-moment dynamics |
| Slow LMU (large θ_s, ungated) | Neocortex | Slow statistical learning, distributional summary |
| Hebbian matrix M | Hippocampus | One-shot episodic binding via LTP |
| Write `v ⊗ k` | Long-term potentiation (LTP) | Hebbian synaptic strengthening on co-activation |
| Read `M · q` | Pattern completion | Partial cue → full memory reconstruction |
| (future) Innovation transfer gate | Hippocampal novelty gating | Theta-oscillation-gated encoding on prediction error |

The CLS prediction: fast binding (hippocampus / M) handles single-episode recall;
slow statistics (neocortex / slow LMU) handles cross-episode regularities. Within
the RL episodic setting, the "neocortex" analog is the within-episode slow
accumulator and the "hippocampus" analog is the per-episode M reset.

---

## Consequences

### New modules

- `Hebbian Memory` class: holds `M` state, `W_K`, `W_V`, `W_Q`, `W_r`
  projections and the LN-on-r step. Can be a standalone `nn.Module` for
  testability.
- `_slow_lgt_matrices(memory_size, theta)`: trivially reuses `_legt_zoh_matrices`.
  No new math needed.
- `DTHLMU(GatedLMU)` subclass: wires fast cell + slow cell + HebbianMemory.

### Parameter count (example: H=64, D=32, C=64, d_k=64, d_v=64)

| Component | Params | Notes |
|---|---|---|
| Fast cell (SelectiveLMU) | ~130k | Unchanged |
| Slow cell (bare integrator) | ~35k | No E_h, no W_pre, no salience; W_query_s + W_ms |
| Hebbian memory | ~12k | W_K + W_V + W_Q + W_r + LN |
| **DTH-LMU total** | ~177k | ~1.4× SelectiveLMU |

### Backward compatibility

- `SelectiveLMU` and `GatedLMU` are unchanged.
- State key `"m"` in the existing cells is *not* renamed — the DTH-LMU uses
  `"m_f"` and `"m_s"` as new keys. No migration of existing checkpoints needed.
- `DTHLMU` can be added as a new class in `memrl/cell/gated_lmu.py` alongside
  the existing hierarchy.

---

## What Is Explicitly Deferred (Ablation Plan)

These decisions are **not made in this ADR**. They are the ablation roadmap for
the DTH-LMU paper.

| # | Ablation | What it tells us |
|---|---|---|
| A1 | Fast + slow LMU only (no M) | Is the two-timescale recurrence sufficient? |
| A2 | Fast LMU + M only (no slow) | Does M solve the failures without the slow integrator? |
| A3 | Slow LMU + M only (no fast) | Does the slow cell + M replace the gated fast cell entirely? |
| A4 | LegS vs. large-θ LegT for slow branch | Is true scale-invariance worth the time-varying recurrence complexity? |
| A5 | Innovation-gated write vs. always-write on M | Does novelty-gating improve M's SNR in practice? |
| A6 | Shared vs. separate e_x for fast/slow | Does the slow cell benefit from its own channel weighting? |
| A7 | d_k, d_v sweep | What associative capacity is actually needed per task? |
| A8 | LN-on-r vs. no-LN-on-r | Does retrieval normalization matter for h stability? |

---

## Alternatives Considered

### Alternative — push n_scales higher, larger scale_factor

Extending the existing multi-scale θ hierarchy (e.g. K=5, scale_factor=4) to
cover longer timescales. Rejected because it only addresses CountRecall (wider
window → more history) but does not address AutoencodeMedium (still polynomial
superposition, not content-addressable) or Battleship (still no spatial structure).
A wider window is a *quantitative* improvement on the same failure mode.

### Alternative — DNC-style full read/write head

Differentiable Neural Computer memory: a full (N×W) external memory with content-
and location-based addressing. Addresses all three failures. Rejected: O(N) reads
and writes per step, N×W state per batch, complex addressing mechanism with known
training instability. Not justified when the targeted failure modes have narrower
fixes.

### Alternative — Attention over stored key-value cache (Transformer-style)

Store the last T input embeddings and do multi-head attention. Addresses
AutoencodeMedium and Battleship. Rejected for RL: O(T²) compute per episode,
T grows with episode length, and the cache requires O(T·d) memory per parallel
environment. The Hebbian matrix M achieves the same retrieval property in O(d²)
space and O(d²) compute per step regardless of T.

### Alternative — Full LegS implementation now

Implement the time-varying LegS recurrence immediately. Deferred because (1) the
time-varying A_d_t requires either per-step matrix construction or precomputed
buffers indexed by t, breaking the fixed-buffer assumption; (2) large-θ LegT
is empirically indistinguishable from LegS within a single episode when θ >>
episode length; (3) the ablation (A4 above) will tell us whether LegS is worth
the engineering cost.

---

## Revisit Triggers

1. **A1 shows fast + slow LMU alone solves all three tasks.** If the Hebbian
   matrix M adds nothing, the DTH-LMU collapses to a two-timescale LMU — simpler
   story, still valid paper.
2. **A2 shows M alone (without the slow cell) solves all three tasks.** The slow
   LMU integrator is dead weight; drop it. Two-component architecture.
3. **The slow cell shows training instability on long episodes.** Large-θ LegT
   is known to have large condition numbers at high θ. Promote LegS if the ZOH
   matrices for θ_s > 2000 are poorly conditioned.
4. **M retrieval noise dominates h on tasks > 1000 steps.** The O(T/√d_k)
   interference bound grows linearly. If this kills performance at long horizons,
   promote the innovation-gated write (ablation A5) to the default.
5. **Phase A baseline results show SelectiveLMU already passes AutoencodeMedium.**
   If the Phase A compute reveals the failures diagnosed here don't actually occur
   at 15M params, re-scope the DTH-LMU motivation to a harder benchmark (e.g.
   Craftax partial obs, POPGym-Hard, or a custom long-horizon task).
