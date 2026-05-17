# GatedLMU research plan (`gated-lmu` branch)

This file scopes the next round of experiments on `gated-lmu`. It sits next to the architecture in [memrl/cell/gated_lmu.py](memrl/cell/gated_lmu.py) and the baselines in [memrl/cell/lmu.py](memrl/cell/lmu.py).

The plan is the salvageable core of an earlier (claude.ai) literature review — minus citation errors, fabricated results, and "redesign as SHM"-style recommendations — plus a focused minimal-compute experimental sequence (F1–F5) to settle the architectural questions before committing to a full sweep.

---

## E. What's actually worth building toward

Stripping the prior review's noise, the real thesis is:

1. **LMU has not been benchmarked as a PPO policy core on POPGym.** That's a real gap; we're positioned to fill it.
2. **The dynamic Legendre-coefficient readout (`C_t = norm(W_query · h_prev)`) is the most novel piece.** Interpretability via `C_t` over time is something FFM / SHM / Mamba can't naturally do — Legendre coefficients have a canonical meaning (polynomial order ↔ effective timescale within the window), so attending over them is interpretable in a way Mamba's input-dependent `C_t` is not.
3. **A clean K-ablation and θ-sweep would be a real RL contribution** if presented honestly.
4. **Headline framing should be "first systematic LMU study in RL,"** not "LMU wins SOTA." The aggregate-win story is hard to defend against SHM; the gap-filling story is honest and defensible.

Everything below serves these four points.

---

## F. Concrete tests (F1–F5)

Cheap experiments, ordered by bang-for-buck. Each is a single config or one-line code edit, run on `popgym-RepeatPreviousMedium-v0` (the task we already have 5 baselines for) at the existing 2M-step budget, seed 0 unless noted.

### F1 — Architecture or hyperparameters?

**Question.** The headline gap was GatedLMU-tuned final eval **0.53 / best 0.65** vs plain LMU final eval **−0.25**. But GatedLMU-tuned also runs at `lr=2e-4` (2× higher) and `memory_size=128` (2× larger) and `theta=200.0`. Is the gap *architecture* or just *better hyperparameters*?

**Test.** Run plain **LMU** with the GatedLMU-tuned config — same env, same step budget, same PPO geometry, **and**:
- `lr = 2e-4`
- `memory_size = 128`
- `theta = 200.0`

i.e. match every hyperparameter we can; the only delta is the cell itself.

**Implementation.** Add a yaml at `benchmarks/phase_a/ablation/lmu_medium_tuned.yaml` mirroring `gated_lmu_medium_tuned.yaml` but with `cell.name: LMU` and `cell.kwargs: {memory_size: 128, theta: 200.0}`. One file, no code change.

**Cost.** ~3–4 hours (1 run).

**Decision value.**
- LMU-tuned eval > 0.4 → most of the gap is hyperparameter tuning. The "GatedLMU is architecturally better" claim doesn't hold and we should refocus on what the *gating* contributes (F2/F4) rather than on the comparison to vanilla LMU.
- LMU-tuned eval ≤ 0 (stuck near random) → the architecture is doing real work; the gap is genuine. Keep going.

### F2 — Lesion study: which extension is doing the work?

**Question.** GatedLMU's "delta" over plain LMU is a stack: multichannel `u`, gated write, dynamic readout, salience gate (off by default), K=3 multi-scale (off by default), LayerNorm (off), readout skip (off), orthogonal `W_pre` (off). Which actually move the needle?

**Test.** Lesion sweep, all on the same task with matched hyperparams (use the GatedLMU-tuned lr/memory/theta from F1 to keep things comparable):

1. **GatedLMU (baseline)** — multichannel + gated write + dynamic readout, K=1, no other extensions. The "bare" GatedLMU per the [docstring](memrl/cell/gated_lmu.py:99).
2. **GatedLMU + salience gate** — `salience_gate=True`.
3. **GatedLMU + K=3 multi-scale** — `n_scales=3, scale_factor=2.0`.
4. **GatedLMU + "freebies"** — `layer_norm=True, readout_skip_scale=0.1, orthogonal_W_pre=True`.
5. **SelectiveLMU** — all the above on (what we already have).

**Implementation.** 5 yaml configs; toggle constructor flags via `cell.kwargs`. No code edits needed — the flags already exist.

**Cost.** 5 runs × ~3h = ~15h.

**Decision value.** Isolates which design choices matter. The earlier review hand-waved about this; we can just measure it. If e.g. (1) ≈ (5), the kitchen-sink extensions are decorative. If (3) dominates, multi-scale is the story. If only (5) wins, the extensions only work together.

### F3 — Does the h_{t−1} gate actually cost wallclock at our chunk size?

**Question.** The prior review claimed h_{t−1}-conditioned gating costs 5–50× wallclock vs x_t-only because it "breaks parallel scan." For PPO + TBPTT at `chunk_len=32` that intuition shouldn't apply (the chunk unrolls sequentially regardless of gate type), but let's measure it.

**Test.** Add a `gate_source: Literal["h_prev", "x_t"]` flag in `GatedLMU.__init__`. When `"x_t"`, the salience-gate input, the Hadamard-calib input, **and** the `pred = u_h + u_m` predictor get replaced by their x-only counterparts (e.g. `pred` → `e_h(x)` or just `0`; salience uses `W_g(x)` instead of `W_g(h_normed)`). Note this is **only** for the throughput measurement — it deletes the predictive-coding signal, so don't use these runs for sample-efficiency claims. Run side-by-side on the GatedLMU baseline; compare `time/fps` from TB.

**Implementation.** ~30 min edit to `_compute_write` and the gate-input lines around [gated_lmu.py:377](memrl/cell/gated_lmu.py:377) plus [gated_lmu.py:382](memrl/cell/gated_lmu.py:382). 2 runs.

**Cost.** ~30 min code + 2 runs × ~3h = ~6h.

**Decision value.**
- Wallclock within 1.5× → the prior review's "5–50× penalty" claim is wrong for our regime. We can stop apologizing for h_{t−1}-conditioned gating in the paper.
- Wallclock >3× → the prior review's point has weight. We'd want to discuss it explicitly.

### F4 — Does the predictive-coding innovation actually help?

**Question.** The central design claim of GatedLMU is that writing `softsign(u_x) + softsign(pred − u_x)` (a predictive-coding-style innovation) beats writing `u_x` raw. If `gate_type="none"` (`u_actual = u_x + pred`, no innovation, no W_pre) gives the same final reward, the gating story is decorative.

**Test.** GatedLMU baseline with `gate_type="softsign_sum"` vs `gate_type="none"`. Same hyperparams (F1-matched lr/memory/theta). Same task, same step budget. Both are first-class supported in [`_compute_write`](memrl/cell/gated_lmu.py:291).

**Implementation.** 2 yaml configs differing in one line. No code edit.

**Cost.** 2 runs × ~3h = ~6h.

**Decision value.** Directly tests the central design claim.
- `softsign_sum` clearly > `none` → predictive-coding write is doing real work; defensible.
- `softsign_sum` ≈ `none` → the gating mechanism is decorative; refocus the paper on the readout/multichannel pieces, not on innovation.

### F5 — K=1 vs K=3 multi-scale

**Question.** SelectiveLMU defaults to `n_scales=3` (banks at `{θ/2, θ, 2θ}`). Does multi-scale help on RepeatPreviousMedium, or is K=1 enough?

**Test.** SelectiveLMU at `n_scales=1` vs `n_scales=3`. Single config flag.

**Cost.** 2 runs × ~3h = ~6h.

**Decision value.**
- K=3 clearly > K=1 → multi-scale carries weight on this task; defend the choice.
- K=1 ≈ K=3 → K=3 is over-engineered for this task; either ablate on a longer-horizon task where multi-scale should matter more, or simplify to K=1 in the headline cell.

---

## Total budget for F1–F5

~12 runs, ~36 wall-hours. (Compare to the 425–475 runs / ~2000 GPU-hours the prior review proposed.) These five answer the most important architectural questions; everything bigger should wait until we know which version of GatedLMU is actually worth scaling up.

## Execution order

| Order | Test | Rationale |
|---|---|---|
| 1 | **F1** | Single most important. If LMU-tuned matches GatedLMU-tuned, the framing of the paper changes completely. |
| 2 | **F4** | Tests the central design claim. Cheap and decisive. |
| 3 | **F2** | Lesion study tells us which extensions to keep in the headline architecture. |
| 4 | **F5** | Settles K=1 vs K=3 on the task we already have results for. |
| 5 | **F3** | Wallclock measurement; only matters if we keep h_{t−1} gating after F2/F4. |

## What's NOT in this plan (deliberately deferred)

- **Knockout / saliency tooling.** The prior review pitched this as existing infrastructure — it doesn't. ~200 lines to build (eval hook that zeros `y_internal` at chosen timesteps; eval hook that logs `C_t` to TB). Worth doing **after** F1–F5 tell us whether the architecture has signal. No point building interpretability tools for a cell that doesn't beat its tuned baseline.
- **θ-sweep.** Real question, real contribution, but defer until F1–F5 settle whether GatedLMU is worth investing in.
- **Full POPGym-Hard suite + SHM/FFM/Mamba baselines.** This is the headline experiment, but it's premature until F1–F5.
- **MiniGrid-Memory / Memory-Gym transfer.** Same — premature.
- **Architecture B / x_t-only redesign** from the prior review. Deletes the predictive-coding signal; not a fix to GatedLMU.

## Verification of each test

Each F-test logs the same scalars we extracted earlier (eval/mean_reward, rollout/ep_rew_mean, train/*, debug/*, time/fps). The same TB-parsing script that produced the 5-baseline comparison can ingest these.
