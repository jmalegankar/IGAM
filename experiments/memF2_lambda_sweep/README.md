# Memento-F2 — e3b λ-sweep (bonus-vs-traversal)

**Question.** On `MiniHack-Memento-F2-v0` the e3b bonus *hurts* (the agent diffuses past
the corridor instead of marching to the fork, ep_len 71 → ~1000+, success → 0). Is that:
- **(scale)** the bonus simply overpowering the −0.01/step "hurry" gradient — recoverable
  by lowering λ — i.e. *not* a task-type effect; or
- **(task-level)** a genuine bonus×task interaction that survives any λ.

**Design.** Dense regime, GDN, `chunk_len=128`, vary **only** `lambda_intrinsic` over
{0, 0.001, 0.003, 0.01, 0.03} (λ=0 = the `none` anchor). Dense = `penalty_step:-0.01`
(the traversal incentive must be present for λ to compete against it). Each arm also sets
`reward_lose:0.0` (zero Memento's native −1 wrong-fork → clean +0.5 cue-blind baseline)
and `normalize_phi:true` (the φ-explosion fix — raw glyph IDs otherwise blow `b_max`→~2e8
and pin the bonus at its clip).

**Run.** `bash experiments/memF2_lambda_sweep/run_sweep.sh`

**Read it (primary signal = `rollout/ep_len_mean`, responds <0.5M):**
- ep_len → ~71 / `action_frac_1` → ~0.9 as λ↓ → harm was bonus-overpowering-traversal
  (scale). Crossover near **λ≈0.01** (march when `λ·E[bonus]/0.01 < 1`; check
  `intrinsic/bonus_mean`).
- At ck128, success caps at ~0.5 **even at λ=0** — that is the chunk-truncation horizon
  wall, *expected*; the ">0.5 / is-the-corridor-learnable" question is the separate
  `ck=n_steps=2048` discriminator, not this sweep. So the clean ck128 result is
  "low-λ e3b ≈ none (≈0.5, marches); high-λ e3b → 0 (wanders)."
- If e3b stays **below the none anchor even at low λ with ep_len normal** → genuine
  task-level interaction.

**Scope.** Even the best case here only shows e3b *stops hurting* on F2 — it cannot *help*,
because F2's binding constraint is the ~57-step cue→fork BPTT gap (exploration is not the
bottleneck). Note: Memento is **not** an E3B benchmark (verified — E3B's MiniHack suite is
MultiRoom / Corridor-R5 / KeyRoom / Labyrinth / LavaCrossing + skills; no Memento), so a
coverage bonus hurting a single-path memory corridor is expected, not anomalous.
