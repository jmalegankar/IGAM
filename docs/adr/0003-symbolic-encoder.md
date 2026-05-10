# ADR 0003: Craftax-Symbolic Encoder Design

**Status:** Approved
**Date:** May 2026
**Decision-maker:** Jai

---

## Context

Phase B requires an encoder φ: O → R^d_φ that operates on Craftax-symbolic's 8268-dimensional observation vector. This encoder is the foundation for four downstream consumers:

1. The IDM auxiliary (predict a_t from φ(o_t), φ(o_{t+1}))
2. The JEPA auxiliary (predict φ_EMA(o_{t+1}) from φ(o_t), a_t)
3. The IGAM cell input (φ produces the keys, values, and inputs that drive W_t updates)
4. The FSQ episodic-bonus bottleneck (φ codes the count-min sketch keys)

The Craftax-symbolic observation vector is not a homogeneous feature distribution. It is a flattened concatenation of structurally distinct semantic groups, each with its own type, scale, sparsity profile, and inductive-bias requirement. The default deep-RL choice — a flat MLP on the full 8268-dim vector — is the path of least resistance from the existing lmu_ppo MLP encoder pattern, but it is unlikely to be optimal for an encoder whose downstream consumers depend critically on feature carving.

This ADR records the encoder design choice for Phase B and the rationale for not using the flat MLP.

---

## Question

What encoder architecture should be used for Craftax-symbolic observations in Phase B?

---

## Findings

### Structure of the Craftax-symbolic observation

The Craftax-symbolic observation decomposes into semantically distinct groups with different encoding requirements. **The exact dimension allocation must be confirmed against the Craftax-Classic-v1 environment spec at implementation time** (Week 9, before this ADR's design is committed to code), but the broad structure is well-established:

- **Local map / terrain view.** A 2D grid (≈9×11 tiles in Craftax-Classic) of categorical block types, mob presence, and tile attributes. Spatially structured. One-hot or categorical encoding per tile. The largest single component by dimension.
- **Inventory.** Discrete counts of each item type (wood, stone, iron, food, water, weapons, etc.). Low-dimensional. Counts range 0–99+. Critical for memory-relevant dependencies — this is exactly where "did I mine iron at step 50" lives.
- **Player intrinsic state.** Scalar continuous features: health, hunger, thirst, energy, XP, level. Very low-dimensional, bounded ranges, mostly slow-changing.
- **Achievement flags.** Binary indicators for unlocked achievements. Sparse, bursty (most flags 0, occasional flips). The achievement-flip events are the canonical "memory-relevant" transitions.
- **Direction / facing.** 4-way categorical or one-hot.

These groups differ along every relevant dimension: scale, sparsity, distribution shape, temporal volatility, and the required inductive bias. They also differ in *gradient relevance*: the terrain map dominates raw dim count, but the semantic content driving long-horizon dependencies lives almost entirely in the inventory and achievements.

### Failure modes of a flat MLP on 8268 dims

**Gradient dominance by raw dim count.** A flat MLP applies the same weight matrix across all 8268 dims. The terrain group, occupying the bulk of the input, dominates gradient norms. Inventory and achievement signals — the semantically critical features — produce small gradient contributions that risk being washed out during training, particularly under high-variance PPO advantages.

**Scale mismatch.** Without per-group normalization, raw inventory counts (0–99), intrinsic state scalars (0.0–1.0), and binary achievements (0/1) co-exist in the same input vector. Standard input normalization (running mean/std) over the full 8268-dim vector cannot account for the fact that different dims belong to fundamentally different distributions.

**Spatial structure ignored.** The 9×11 terrain grid has translation-equivariant local structure that a flat MLP cannot exploit. Two adjacent stone tiles look like two arbitrary input positions to a flat MLP.

**FSQ codebook miscarving — the silent killer.** This is the failure mode that ties this ADR to the Phase B success criteria most directly. If the encoder fuses terrain and inventory before producing the φ that feeds FSQ, the high-variance terrain dimensions will dominate the FSQ projection. The codebook will partition primarily by terrain configuration, and "crafted sword vs. no sword with otherwise identical observation" — the canonical Phase B diagnostic from `plan.md` Week 17 — will produce the *same* FSQ code in both cases. The episodic novelty signal then never fires on inventory transitions, the count-min sketch never differentiates achievement-bearing states, and Phase B reward stagnates near the PPO-RNN baseline.

This failure mode is silent until Phase B reward fails to clear Tier 1 — at which point the encoder is the suspect and the diagnostic loop is expensive (multi-day FSQ-code dump analysis, encoder retraining, etc.).

### Naive alternative: variance-standardization layer

A learned per-dim variance-standardization layer (running mean/std normalization, applied per-dim) ahead of the flat MLP addresses the scale-mismatch failure mode but does nothing for gradient dominance, ignores spatial structure, and provides no per-group inductive bias. It is a partial fix that does not address the semantically critical problem (FSQ codebook carving). External evaluations have suggested this as an intervention; it is rejected here because it solves the smallest of the four failure modes and leaves the rest intact.

### Semantic-group sub-encoder approach

The principled alternative is to encode each semantic group with a sub-encoder appropriate to its structure, project each to a common output dimension, concatenate, and pass through a small fusion MLP:

```
φ(o) = MLP_fusion(concat(
    φ_terrain(terrain_grid),
    φ_inventory(inventory_counts),
    φ_intrinsic(intrinsic_state),
    φ_achievements(achievement_flags),
    φ_direction(facing)
))
```

Each sub-encoder is sized to its input — `φ_intrinsic` is a tiny MLP, `φ_terrain` is the largest. Output dims are equalized across sub-encoders so that no group's representation dominates the fusion MLP by raw dim count. The fusion MLP is small (one or two hidden layers) and produces the d_φ output.

This design addresses each failure mode directly:

- **Gradient dominance:** equal output dims per group equalize each group's contribution at the fusion stage.
- **Scale mismatch:** each sub-encoder normalizes its own input distribution and applies the appropriate per-group activation.
- **Spatial structure:** `φ_terrain` can use a small CNN or per-tile embedding + spatial pooling rather than treating tiles as arbitrary input positions.
- **FSQ codebook carving:** the fusion output mixes group-level features at equal weight rather than dim-weighted raw inputs, substantially raising the probability that FSQ codes partition by inventory/achievement transitions rather than by terrain configuration.

### Spatial inductive bias on the terrain group

Two viable choices for `φ_terrain`:

1. **Per-tile embedding + flatten + MLP.** Embed each block-type category to a small vector (≈8–16 dim), flatten the resulting (9×11×D) tensor, MLP to the per-group output dim. Cheap; no spatial pooling but still benefits from learned tile semantics.
2. **Per-tile embedding + small CNN.** Same embedding, then a 2-layer CNN (3×3 convs), then global pool. More expressive; captures local spatial patterns (e.g., "iron next to stone").

Option 1 is the MVP. Option 2 is the upgrade if Phase B per-sub-encoder diagnostics implicate `φ_terrain` as the bottleneck. We start with Option 1 and treat Option 2 as a documented escalation path.

### Phase A vs. Phase B applicability

This ADR applies to Phase B only. Phase A observations (POPGym tasks, MiniGrid-Memory) are low-dimensional and either homogeneous (POPGym vectors) or already spatially structured at small scale (MiniGrid grids). The semantic-group design is over-engineering for Phase A and is not introduced there. Phase A continues to use per-task MLPs and the existing MiniGrid encoder pattern.

---

## Decision

**The Phase B encoder for Craftax-symbolic uses semantic-group sub-encoders with equal-dim outputs and a fusion MLP.**

Concrete specification for the Phase B implementation (subject to dimension confirmation in Week 9):

- **`φ_terrain`:** per-tile embedding (block-type → 16 dim) → flatten → MLP → 64-dim output.
- **`φ_inventory`:** per-feature standardization → 2-layer MLP (hidden 64) → 64-dim output.
- **`φ_intrinsic`:** 2-layer MLP (hidden 32) → 64-dim output.
- **`φ_achievements`:** linear projection or shallow MLP → 64-dim output.
- **`φ_direction`:** one-hot pass-through or linear projection → 64-dim output (small input, padded to match).
- **Fusion:** concat (≈320 dim) → 2-layer MLP → d_φ = 256.
- **Per-sub-encoder gradient norms logged** as a first-class Phase B diagnostic (see `plan.md` Logging discipline).

The naive flat MLP and the flat-MLP-with-variance-standardization are both rejected as defaults. The flat MLP is preserved as a Phase B ablation row to quantify the gain from semantic grouping.

---

## Rationale

The semantic-group design is the smallest intervention that addresses all four flat-MLP failure modes (gradient dominance, scale mismatch, spatial structure, FSQ codebook carving) simultaneously. It is also the design most likely to make Phase B reward gaps *attributable*: by logging per-sub-encoder gradient norms, we can diagnose which semantic group the encoder is failing on, rather than performing whole-encoder ablations on a flat MLP that cannot localize the failure.

The cost is moderate engineering complexity (5 sub-encoders + fusion MLP) and a slightly larger parameter count. Both are acceptable given the diagnostic value. The design also makes per-group ablations possible (e.g., "what if `φ_terrain` is a frozen random projection?") which is a useful hammer for the Phase B ablation table.

---

## Consequences

- **Implementation cost (Week 9):** ~1–2 days additional vs. flat MLP. The sub-encoders are individually trivial; the fusion plumbing and per-group config are the bulk of the work.
- **Diagnostic richness gained.** Per-sub-encoder gradient norms become a first-class Phase B diagnostic. If Phase B reward < 5%, the first signal we examine is whether terrain dominates gradient norms while inventory and achievements are silent — the canonical "encoder failed to carve inventory" signature. This diagnostic is *unavailable* under a flat MLP.
- **FSQ codebook configuration unchanged.** d_fsq = 6, L = 5 (codebook ≈ 15,625). The fusion output feeds FSQ as before. The change is upstream of FSQ.
- **Sub-encoder hyperparameter sensitivity.** Five sub-encoders means five sets of layer-size choices. Default to small (one or two hidden layers, hidden dim 64) and resist tuning per-group hyperparameters until Phase B diagnostics demand it. Per-group sweeps are explicitly out of scope for the initial Phase B run.
- **Per-group ablations enabled.** The Phase B ablation table can include rows like "− `φ_terrain` (random projection)", "− `φ_achievements` (zero)", and "all-flat" (the rejected default), localizing which semantic group is doing the work. This is more informative than monolithic encoder-on/off ablations.
- **Phase B `plan.md` deep-diagnostic list updated.** Add: "If reward < 5%: dump per-sub-encoder gradient norms and FSQ code-vs-inventory cross-tab. If terrain gradients dominate or FSQ codes are uncorrelated with inventory state, escalate to fallback (Alternative 4) or upgrade `φ_terrain` to Option 2."

---

## Alternatives considered

### Alternative 1: Flat MLP on the full 8268-dim vector

Reasoning: simplest baseline; matches lmu_ppo's POPGym/MiniGrid encoder pattern.

Rejected because: silent failure modes (gradient dominance, FSQ codebook miscarving) are exactly the kind that produce stagnant Phase B results that are expensive to diagnose. The simplicity gain is not worth the diagnostic blindness. Preserved as a Phase B ablation row to quantify the gain from semantic grouping.

### Alternative 2: Flat MLP + per-dim learned variance standardization

Reasoning: addresses scale mismatch without the structural complexity of sub-encoders. This is the intervention recommended by external architectural evaluations.

Rejected because: scale mismatch is the smallest of the four failure modes. Per-dim variance standardization does nothing for gradient dominance, spatial structure, or FSQ codebook carving. It is a partial fix to the wrong problem. The recommendation reflects a misdiagnosis of where the flat-MLP failure originates.

### Alternative 3: Single large CNN treating the 8268-dim vector as a 1D signal

Reasoning: exploits any local correlation structure; uniform architecture across groups.

Rejected because: the 8268-dim layout is a flattened concatenation, not a signal with consistent local structure. A 1D CNN's locality bias is meaningless across the inventory/achievement/intrinsic boundaries. The "uniform architecture" benefit is illusory.

### Alternative 4: Per-group FSQ with code concatenation (skip the fusion MLP for the FSQ path)

Reasoning: guarantees that inventory and achievement features cannot be washed out of the count-min sketch keys, by computing a separate FSQ code per group and concatenating.

Partially adopted as a documented fallback. The default Phase B implementation has one fused FSQ on the fusion output, matching the original IGAM design and keeping the count-min sketch's effective codebook manageable. *If* Phase B diagnostics show that the FSQ codebook is partitioning primarily by terrain (the canonical failure signal), per-group FSQ is the documented escalation path. The cost is a much larger effective codebook and weaker per-cell counts; we adopt this only if needed.

### Alternative 5: Transformer encoder on tokenized observations

Reasoning: maximally expressive; the field's default for structured inputs.

Rejected because: massive overkill for the Phase B compute budget; introduces additional optimization difficulty (pre-LN/post-LN choices, attention temperature, dropout schedules); and the inductive bias of self-attention is not well-matched to a small fixed-structure observation. Worth revisiting in Phase C if the encoder becomes the bottleneck at scale, but explicitly out of scope for Phase B.

---

## Revisit triggers

This ADR should be reopened if:

1. **Phase B reward stagnates below Tier 1 (5%) AND per-sub-encoder gradient norms are balanced.** This rules out the encoder-carving failure mode this ADR was designed to prevent. The encoder design is not the bottleneck and we look elsewhere (FSQ projection, IDM signal, lifelong-bonus normalization).

2. **FSQ codebook utilization analysis shows partitioning primarily by terrain rather than inventory/achievements.** Switch to per-group FSQ (Alternative 4 fallback) and re-run.

3. **`φ_terrain` shows up as the dominant gradient-norm consumer despite equal output dims.** Upgrade `φ_terrain` to Option 2 (per-tile embedding + small CNN) and re-test.

4. **The Craftax-symbolic observation specification changes** in a future Craftax release. Re-derive the semantic groups from the new spec and update sub-encoder definitions.

5. **The Week 9 dimension audit reveals the assumed group structure is wrong.** If, on inspection, the 8268-dim vector decomposes differently from the structure assumed in Findings, redesign the sub-encoders before implementation. The semantic-group *principle* survives; the specific group definitions may need revision.

6. **A published Craftax-symbolic encoder design demonstrates a substantially different decomposition working better.** Adopt it.