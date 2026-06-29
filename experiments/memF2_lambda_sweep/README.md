# Memento-F2 — e3b λ-sweep (bonus-vs-traversal, in the learnable regime)

**Question.** On `MiniHack-Memento-F2-v0` the e3b bonus *hurts* (the agent diffuses past
the corridor instead of marching to the fork, ep_len 72 → 1000+, success → 0). Is that:
- **(scale)** the bonus simply overpowering the −0.01/step "hurry" gradient — recoverable
  by lowering λ, i.e. *not* a task-type effect; or
- **(task-level)** a genuine bonus×task interaction that survives any λ — even when the
  agent still marches and the corridor is otherwise learnable.

**Why ck = n_steps = 2048 (not 128).** `ep_len(λ)` is chunk-independent, so 128 would
show the march/diffuse crossover — but at ck128 **success is pinned at the horizon wall
(~0.5) for every λ** (the 57-step cue→fork BPTT is truncated). So ck128 can only show
e3b *stops hurting*, never whether low-λ e3b lets the corridor be *learned*. The
discriminator (ck=ns "none", success climbs >0.5) proves ck=ns is the learnable regime,
so the sweep is matched to it. Then the decisive read: at a λ low enough to keep the
agent marching, does e3b **climb like the none-discriminator** (→ neutral, harm was pure
scale/traversal) or **plateau below it** (→ genuine interference with cue-learning)?

**Design.** Dense (`penalty_step:-0.01` so λ competes with the hurry gradient), GDN,
`chunk_len = n_steps = 2048`, vary `lambda_intrinsic` ∈ {0.003, 0.01, 0.03} bracketing the
~0.01 crossover. λ=0 anchor = the running ck=ns `none` discriminator. Each arm sets
`reward_lose:0.0` (zero Memento's native −1 wrong-fork → clean +0.5 cue-blind baseline)
and `normalize_phi:true` (φ-explosion fix — raw glyph IDs otherwise pin the bonus at its
clip).

**Run.** `bash experiments/memF2_lambda_sweep/run_sweep.sh`  (heavy GPU — 3 ck=2048 BPTT
jobs on top of the discriminator; run just `lam0.003` first if contended).

**Read it:**
- `rollout/ep_len_mean` (fast, <0.5M): → ~72 / `action_frac_1`→~0.9 as λ↓ confirms the
  march/diffuse crossover (march when `λ·E[bonus]/0.01 < 1`, crossover ≈0.01; check
  `intrinsic/bonus_mean`).
- `eval/success_rate` (the new signal vs ck128): low-λ e3b **climbs toward the
  none-discriminator** → e3b neutral once it stops overpowering traversal (scale-only).
  Low-λ e3b marches but **stays flat ~0.5 while none climbs** → genuine bonus×memory
  interference. High-λ (0.03) is expected to diffuse → 0 regardless.

**Scope / attribution.** Even the best case shows e3b *doesn't hurt* on F2 — it can't
*help*, because F2's bottleneck is the cue→fork BPTT gap, not exploration. Memento is
**not** an E3B benchmark (verified — E3B's MiniHack suite is MultiRoom / Corridor-R5 /
KeyRoom / Labyrinth / LavaCrossing + skills; no Memento), so a coverage bonus hurting a
single-path memory corridor is expected, not anomalous.
