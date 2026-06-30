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

## Findings so far + NovelD (2026-06-29)

First (parallel) run, before a clean sequential pass: **e3b collapses on F2 at every λ down
to 0.003.** It never marches — ep_len drops 3371→~1037 by 0.5M and stays there for 2.5M,
succ 0, while the ck=ns `none` discriminator marches (ep_len 73) and climbs to 0.60. And it
does so at λ=0.003 where `λ·bonus≈0.0009` is **11× below the −0.01 hurry** — so it's not a
steady-state reward balance, it's an **early-training policy collapse into a diffusion
basin**. → leans the genuine-task-level branch, not recoverable-by-λ. (Caveat: that run was
parallel and OOM-killed the discriminator at 1.31M; re-run clean + sequential, and λ=0.001
is the decisive untested point.)

**NovelD arms added.** Mechanistically NovelD may behave differently: it's a first-visit-
gated RND *difference*, rewarding novelty *increase* and killing credit for backtracking, so
on a linear corridor its gradient points *forward* (deeper = more novel = toward the fork),
unlike e3b's any-within-episode-novelty (which pays off-path wandering). **Caveat:** NovelD's
first-visit gate hashes the *observation*, and our obs is the *egocentric* crop — a uniform
corridor looks like the same state every step, which could misfire the gate and push NovelD's
bonus → ~0 (benign, but for the wrong reason). Watch `intrinsic/bonus_mean`: ~0 ⇒ the crop
makes the corridor look identical; healthy + marching ⇒ genuinely forward-aligned.

**Scope / attribution.** Even the best case shows the bonus *doesn't hurt* on F2 — it can't
*help*, because F2's bottleneck is the cue→fork BPTT gap, not exploration. Memento is
**not** an E3B benchmark (verified — E3B's MiniHack suite is MultiRoom / Corridor-R5 /
KeyRoom / Labyrinth / LavaCrossing + skills; no Memento), so a coverage bonus hurting a
single-path memory corridor is expected, not anomalous.
