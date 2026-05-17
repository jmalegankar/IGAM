# ADR 0005: SelectiveLMU Ablation Plan for Paper 1

**Status:** Proposed
**Date:** May 2026
**Decision-maker:** Jai

---

## Context

The Phase A research direction has pivoted. The original plan (per the README)
was "IGAM = Gated DeltaNet on POPGym + Craftax with auxiliary objectives." That
remains Paper 2. Paper 1 is now a separate, narrower contribution:

> **A selective multichannel Legendre Memory Unit that beats canonical recurrent
> cells on long-horizon partially-observable RL at matched param count.**

The headline cell is `SelectiveLMU` (see `igam/cell/gated_lmu.py`), which
augments the thesis's `GatedLMU` with three design choices drawn from recent
selective-SSM literature:

1. **Multi-scale theta** (HiPPO Zoo §3.4, Goffinet et al. 2026 — `arxiv:2602.21340`)
2. **Salience gate** (HiPPO Zoo §3.2)
3. **Bounded leak gates** (sigmoid-shaped, inspired by Stable Hadamard Memory,
   Le et al. 2025 — `arxiv:2410.10132`)

Plus three "freebies" — LayerNorm on `h_prev`, a `u_x → y_internal` skip, and
an orthogonal parametrization on `W_pre`.

Preliminary results on POPGym RepeatPrevious are encouraging:

| Task | Baseline GatedLMU-tuned | SelectiveLMU (lean, mid-run) |
|---|---|---|
| Medium (103-step) best | +0.264 @ 2.73M | **+0.500 @ 1.02M** |
| Hard (155-step) best | +0.083 @ 786k (then collapse to -0.287) | +0.063 @ 516k, no collapse at 2M |

These numbers are at matched param count (~25k cell params, ~127k total
including encoders/heads for both).

The encouragement is real but the *attribution* is unclear: is the gain from
multi-scale theta, the salience gate, the LayerNorm, the orthogonal `W_pre`, or
some interaction? Without ablations we cannot honestly claim any single design
choice as the cause. A reviewer asking "is it really the gates, or just the
extra capacity from K=3 banks?" would not have an answer.

---

## Question

**Which architectural changes in `SelectiveLMU` actually drive the performance
gains over the thesis `GatedLMU`?**

Sub-questions, listed by what each ablation arrow isolates:

1. Does **multichannel `u`** alone (vs. canonical scalar-input LMU) help?
2. Does **multi-scale theta** alone (no gating) help?
3. Does the **softsign-sum write gate** alone (single theta) help?
4. Does **multi-scale + gate** beat each in isolation?
5. Does the **salience gate** add anything *given* multi-scale + write gate are already in?

We also need to know whether the **no-gate, multi-scale, multichannel** cell
already does most of the work — if so, the paper's central claim shifts from
"gating wins" to "multi-scale Legendre basis wins."

---

## Findings

The existing `GatedLMU` API already supports the entire matrix via flags. No
new cell code needed; this is a configuration exercise. Each row toggles
exactly one knob from the row above it.

### Ablation matrix

| # | Config | Cell + kwargs | Gate | Multich. | Multi-scale | Salience | Freebies | Isolates |
|---|---|---|---|---|---|---|---|---|
| **0** | `LMU` (Voelker 2019) | `LMU` | — | ✗ (scalar) | ✗ | ✗ | ✗ | Anchor — canonical baseline |
| **1** | `MCLMU` | `GatedLMU(gate_type='none', n_scales=1)` | ✗ | ✓ | ✗ | ✗ | ✗ | Does multichannel `u` alone help? (vs row 0) |
| **2** | `MSLMU` | `GatedLMU(gate_type='none', n_scales=3, +freebies)` | ✗ | ✓ | ✓ | ✗ | ✓ | **Does multi-scale alone help, with NO gating?** (vs row 1) |
| **3** | `GatedLMU` (thesis) | `GatedLMU(gate_type='softsign_sum', n_scales=1)` | ✓ | ✓ | ✗ | ✗ | ✗ | Does the write gate help in isolation? (vs row 1) |
| **4** | `GatedMSLMU` | `GatedLMU(gate_type='softsign_sum', n_scales=3, +freebies)` | ✓ | ✓ | ✓ | ✗ | ✓ | Gate + multi-scale, no salience (vs rows 2, 3) |
| **5** | `SelectiveLMU` (headline) | `SelectiveLMU(n_scales=3)` | ✓ | ✓ | ✓ | ✓ | ✓ | Does salience help given everything else? (vs row 4) |

Where `+freebies` = `layer_norm=True, readout_skip_scale=0.1, orthogonal_W_pre=True`.

### Arrows and what they tell us

| Arrow | Δ | Tells us |
|---|---|---|
| 0 → 1 | multichannel | Does scalar→vector input help on POPGym? |
| 1 → 2 | multi-scale + freebies | **Critical no-gate test.** Architecture alone vs. with gating. |
| 1 → 3 | write gate | Does softsign-sum gating help without other changes? |
| 2 → 4 | write gate (given MS) | Does the gate add value on top of multi-scale? |
| 3 → 4 | multi-scale (given gate) | Does multi-scale add value on top of the gate? |
| 4 → 5 | salience gate | Does the per-scale data-dependent decay add anything? |

If 1→2 (`MCLMU → MSLMU`) shows a large jump, the paper's main claim
becomes "**multi-scale Legendre basis** is the key contribution, gating is
ornamental." If 1→3 (`MCLMU → GatedLMU`) is the bigger jump, gating is the
core. If both contribute (3→4 and 2→4 are both positive), the story is
"multi-scale + gating compose superlinearly." Each outcome is a publishable
story; we don't know which one is true yet.

---

## Decision

**Run the 6-cell ablation on POPGym RepeatPrevious {Easy, Medium, Hard} at 1
seed first; promote the top 3 cells to 3 seeds for headline error bars.**

Concrete plan:

```
Phase 1 — 1 seed × 6 cells × 3 difficulties = 18 runs
  Order:
    1. Wait for current SelectiveLMU Medium+Hard to complete (~30 min remaining).
    2. Launch MSLMU (row 2) Medium+Hard FIRST.  ← critical no-gate test.
    3. Launch GatedMSLMU (row 4) Medium+Hard.
    4. Launch MCLMU (row 1) Medium+Hard.
    5. Backfill row 0, 3 (we already have GatedLMU thesis data, so really
       just LMU Medium+Hard).
    6. Backfill all 6 cells on Easy (cheaper — most cells solve it).

Phase 2 — 3 seeds × top 3 cells × 3 difficulties = 27 runs
  Top 3 to be selected from Phase 1 results.
  Adds error bars to the headline table.

Phase 3 — extension (post-paper-draft)
  Add a 4th POPGym task (Autoencode or similar) for breadth.
  Add the thesis memory sweep as the "controlled difficulty" figure.
```

**Compute budget on M3 (2 runs in parallel, ~55 min/run):**

- Phase 1: 18 runs ÷ 2 parallel × ~55 min ≈ 8 hours
- Phase 2: 27 runs ÷ 2 parallel × ~55 min ≈ 12 hours
- Total: ~20 hours wall-clock

Doable over 1 night + 1 day. If we get GPU access later, reduce ~5×.

---

## Rationale

**Without ablations the paper has no defensible attribution claim.** A reviewer
will ask exactly the question above ("is it the gates or the architecture?"),
and the only honest answer is "we don't know" — which is fatal.

**The no-gate row (MSLMU) is the single most important data point.** It
distinguishes between two very different stories:

- *Story A (gates are the contribution):* MSLMU is mediocre, gated cells win.
  Paper sells "selective gating on polynomial memory."
- *Story B (architecture is the contribution):* MSLMU is competitive,
  gating adds <0.1 to best-eval. Paper sells "multi-scale Legendre basis
  is what's missing from canonical LMUs."

Both are publishable. We just need to know which one is true. The cost is
two more runs (MSLMU on Medium and Hard) — a few hours on M3.

**Reusing the existing `GatedLMU` API minimizes implementation risk.** No new
code; same cell, just different `gate_type` and flag values. Already covered
by the 234-test suite.

**Param-matched comparisons throughout.** Every cell in the ablation is at
~25k cell params (or ~127k total including encoders). Eliminates the "more
capacity wins" critique.

**Phase 1 → Phase 2 staging is honest.** We don't claim seeds we haven't run.
Phase 1's 1-seed results are exploratory; Phase 2 promotes only the winners
to publication-grade evidence. If Phase 1 shows the headline cell loses, we
go back to the drawing board *before* burning 27 runs of compute.

---

## Consequences

- **Configs to add** under `benchmarks/phase_a/ablation/`:
  - `mclmu_medium.yaml`, `mclmu_hard.yaml`
  - `mslmu_medium.yaml`, `mslmu_hard.yaml`
  - `gated_mslmu_medium.yaml`, `gated_mslmu_hard.yaml`
  - `lmu_medium.yaml`, `lmu_hard.yaml`
  - (SelectiveLMU configs already exist under `tuning/`; will be linked from the table.)
- **Easy variants** added once Phase 1 Medium+Hard shows the winning cells.
- **Paper 1 table** will be auto-generated from `runs/<config>/<cell>/seed_*/eval/evaluations.npz`
  via a `scripts/collect_ablation.py` (to be written).
- **Seed-1 results** form the exploratory ablation; **seed-{0,1,2}** results form the headline.
- **Cost reminder:** ~20 hours wall-clock on M3 to complete Phase 1 + Phase 2.

---

## Alternatives considered

### Alternative — skip ablation, publish on SelectiveLMU vs. baselines only

Tempting because it's faster. Rejected: a paper claiming "selective LMU wins"
without attribution data fails review. The headline table needs *internal*
arrows (showing why we made each design choice), not just *external* arrows
(showing we beat LSTM/Mamba).

### Alternative — full 2⁵ ablation over the 5 flags

Total combinations: 32. Many are redundant (e.g. salience-on-without-gates makes
little semantic sense). Rejected as wasteful. The 6-row matrix above covers
every interesting arrow with no redundant runs.

### Alternative — run ablation only on Medium (the strongest baseline result)

Rejected. The Hard collapse is a meaningful failure mode for `GatedLMU`-tuned
that one of our extensions might fix. If `MSLMU` doesn't collapse on Hard, that
alone justifies multi-scale theta independently of any reward number. Hard is
the most informative difficulty.

### Alternative — promote Hadamard calibration back into the ablation

Considered. Decided against for Phase 1: the heavy Hadamard variant
underperformed baseline on Hard (run at `/tmp/igam_selective/`, killed at ~1M),
and a leaner factorized variant hasn't been built yet. Hadamard is **Phase 3**
work — bring it back once the simpler ablation lands and we know whether it's
worth the engineering investment.

---

## Revisit triggers

This ADR should be reopened if:

1. **Phase 1 reveals the headline cell loses to a simpler row.** If MSLMU
   beats SelectiveLMU on both Medium and Hard, the salience gate is dead
   weight and the paper's lead cell changes.
2. **A row gives wildly variable results across seeds in Phase 2.** Suggests
   the gain is variance, not signal — investigate seed sensitivity before
   committing to that cell.
3. **The thesis memory sweep contradicts the POPGym ordering.** If the
   ranking on the synthetic sweep differs from POPGym, neither benchmark
   alone supports the paper's claim; pick the one with better external
   validity (probably POPGym) and explain the discrepancy.
4. **Phase 2 finishes and the headline gap is small (<0.05).** May not be a
   compelling enough result to publish; reconsider scope or move to a harder
   benchmark before drafting.
