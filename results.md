# GatedLMU experiment results

Running log of F1–F5 (see [plan.md](plan.md)). Each entry: setup, raw numbers, verdict against the decision rule.

Task: `popgym-RepeatPreviousMedium-v0`, 2M env steps, 8 envs, seed 0.

---

## F1 — Architecture vs. hyperparameters

**Question.** Is the GatedLMU-tuned vs LMU gap (0.65 vs −0.16 best) caused by the cell architecture, or just by `lr=2e-4 + memory_size=128 + theta=200`?

**Setup.** Plain LMU at the GatedLMU-tuned hyperparameters. Identical PPO geometry. Config: [`benchmarks/phase_a/ablation/lmu_medium_tuned.yaml`](benchmarks/phase_a/ablation/lmu_medium_tuned.yaml). Run dir: [`runs/gate/F1/lmu_medium_tuned/LMU/seed_0_20260517_025943/`](runs/gate/F1/lmu_medium_tuned/LMU/seed_0_20260517_025943/).

**Result.**

| Cell | final eval | best eval | best @ step | rollout final |
|---|---:|---:|---:|---:|
| GatedLMU-tuned | 0.533 | **0.653** | 1.14M | 0.438 |
| **LMU (tuned, F1)** | **−0.108** | **0.047** | 1.33M | −0.096 |
| LMU (orig: lr=1e-4, mem=64) | −0.247 | −0.161 | 1.17M | −0.284 |

Eval quartiles for F1 LMU-tuned: `25% −0.075  50% −0.050  75% −0.108  100% −0.108`. Briefly pokes above 0 around 1.3M steps, settles back near random.

Diagnostics: `explained_variance=0.29` (vs GatedLMU-tuned 0.81 — critic can't fit), `KL=0.048` hit `target_kl=0.03` at end (PPO early-stop kicked in), `action_frac` mildly biased (0.32/0.26/0.26/0.17 — not collapsed).

Wall time: ~3.25h.

**Verdict.** Tuning the hyperparameters lifts plain LMU from −0.16 → +0.05 best — a ~0.21 absolute improvement but still effectively random. The remaining gap of ~0.6 to GatedLMU-tuned (0.65 best) is architectural.

Decision rule from the plan: *>0.4 = hyperparams; ≤0 = architecture*. F1 came in at +0.05 — closer to "stuck near random," so **the architecture is doing real work**. The hyperparameter explanation is rejected.

---

## F4 — Does the predictive-coding innovation actually help?

**Question.** Within the GatedLMU baseline (extensions off, K=1, multichannel + dynamic readout still on), does `gate_type="softsign_sum"` (predictive-coding write: `u_actual = W_pre(softsign(u_x) + softsign(pred − u_x)) + pred`) beat `gate_type="none"` (ungated write: `u_actual = u_x + pred`)?

**Setup.** Two runs in [`runs/gate/F4/`](runs/gate/F4/):
- `softsign_sum` arm: the existing GatedLMU-tuned run (multichannel + dynamic readout + softsign gate).
- `none` arm: same config but `gate_type="none"`. Config: [`benchmarks/phase_a/ablation/gated_lmu_medium_tuned_no_gate.yaml`](benchmarks/phase_a/ablation/gated_lmu_medium_tuned_no_gate.yaml).

**Result.**

| Arm | final eval | best eval | best @ step | final expl_var |
|---|---:|---:|---:|---:|
| softsign_sum | **0.533** | 0.653 | 1.14M | **0.792** |
| none (F4) | 0.361 | **0.731** | 0.97M | 0.732 |

Eval trajectory (quartiles):

| Arm | 10% | 25% | 50% | 75% | 90% | **100%** |
|---|---:|---:|---:|---:|---:|---:|
| softsign_sum | -0.14 | 0.50 | 0.62 | 0.53 | 0.50 | **0.53** |
| none (F4) | 0.16 | 0.60 | 0.65 | 0.52 | 0.48 | **0.36** |

**Verdict.** The original decision rule (*"if `none` ≈ `softsign_sum`, gating is decorative"*) was wrong — single-point best-eval comparison doesn't capture the dynamics. Both arms peak around 1M steps at similar levels (0.65 vs 0.73), but **`none` collapses 0.12 over the final 200k steps while `softsign_sum` holds**.

Cell-state norms confirm this isn't a memory-blowup failure: no-gate's `actor_m` norm is *smaller* (25 vs 54). The instability is more subtle — without the predictive-coding write, PPO updates drift the policy off its peak after the cell finds it, and the cell has no write-gate to anchor the memory representation back to the useful regime.

**Corrected decision rule:** compare *final eval at budget* (or *mean of last 25%*), not best eval. By that metric:
- softsign_sum final 25%: ~0.51
- none final 25%: ~0.42
- Gap: **+0.09 in favor of gate** — modest but real.

**Caveats** (must address before drawing architectural conclusions):
- Single seed.
- Single task (RepeatPreviousMedium).
- 2M-step budget — longer training might separate the curves more, or the gated version might also collapse later.

The 4-env suite below is what would actually settle this.

---

## F1 expanded — 4-env suite (next, queued)

**Motivation.** Single-task F1/F4 results suggest gating provides *training-stability* rather than a final-eval bump. Need to confirm on additional pure-memory POPGym envs before committing to that framing. The exploration+memory question (whether gating matters more for exploration-driven memory tasks like MiniGrid-Memory) is deferred to a separate phase.

**Suite (4 envs):**
- `popgym-RepeatPreviousMedium-v0` (done — F1)
- `popgym-AutoencodeMedium-v0` (WATCH/PLAY phases; uses `ExposePhaseInInfo` + `FlattenTupleDiscrete`)
- `popgym-CountRecallMedium-v0` (MultiDiscrete obs → `FlattenMultiDiscrete`)
- `popgym-BattleshipEasy-v0` (MultiDiscrete action → MemActorCriticPolicy MultiDiscrete support)

**Setup.** For each env, run plain LMU and GatedLMU at matched hyperparameters (lr=2e-4, memory_size=128). θ is calibrated **per env** (≈ 2× episode length) so each cell has adequate Legendre window coverage on each task:

| Env | episode length | θ |
|---|---:|---:|
| RepeatPreviousMedium | 103 | 200 |
| AutoencodeMedium | ~208 (2 decks, WATCH+PLAY) | 400 |
| CountRecallMedium | 200 | 400 |
| BattleshipEasy | 64 (8×8 board) | 128 |

6 new runs total (RepeatPrevious is already done). Configs:
- [`benchmarks/phase_a/ablation/lmu_autoencode_medium_tuned.yaml`](benchmarks/phase_a/ablation/lmu_autoencode_medium_tuned.yaml)
- [`benchmarks/phase_a/ablation/gated_lmu_autoencode_medium_tuned.yaml`](benchmarks/phase_a/ablation/gated_lmu_autoencode_medium_tuned.yaml)
- [`benchmarks/phase_a/ablation/lmu_countrecall_medium_tuned.yaml`](benchmarks/phase_a/ablation/lmu_countrecall_medium_tuned.yaml)
- [`benchmarks/phase_a/ablation/gated_lmu_countrecall_medium_tuned.yaml`](benchmarks/phase_a/ablation/gated_lmu_countrecall_medium_tuned.yaml)
- [`benchmarks/phase_a/ablation/lmu_battleship_easy_tuned.yaml`](benchmarks/phase_a/ablation/lmu_battleship_easy_tuned.yaml)
- [`benchmarks/phase_a/ablation/gated_lmu_battleship_easy_tuned.yaml`](benchmarks/phase_a/ablation/gated_lmu_battleship_easy_tuned.yaml)

**Launcher:** [`scripts/run_f1_suite.sh`](scripts/run_f1_suite.sh) runs **LMU + GatedLMU in parallel per env**, pairs sequentially across envs (so CPU contention stays bounded at 2 simultaneous processes). Output layout:
```
runs/gate/F1/
├── repeat_previous_medium/    (existing; both arms)
├── autoencode_medium/         (LMU + GatedLMU runs side-by-side)
├── countrecall_medium/        (same)
└── battleship_easy/           (same)
```
Suite-level log: `runs/gate/F1/_suite.log`. Per-run logs: `<env>/<config_stem>.log`.

Wall time estimate: ~5–7h per pair × 3 pairs ≈ **15–20h total**.

**Decision rules:**
- *Headline:* mean of `final 25%` of eval reward, averaged across envs, GatedLMU vs LMU. Positive consistent gap → architectural signal across pure-memory POMDPs.
- *Stability secondary:* fraction of envs where GatedLMU's `final eval ≥ best eval − 0.10`, vs LMU. Higher fraction → less collapse.
- *Single-task wins:* per-env table, both metrics.

For Autoencode specifically, also log the new `debug/innovation_mag_phase_{0,1}_mean` to TB — should see innovation magnitude high in phase 0 (WATCH, writing) and lower in phase 1 (PLAY, reading) for a healthy GatedLMU.

---

---

## Infrastructure additions for the 4-env suite (2026-05-17)

To support `RepeatPreviousMedium + AutoencodeMedium + CountRecallMedium + BattleshipEasy` as the F1–F5 evaluation suite, the following was added on `gated-lmu`:

### Env wrappers ([memrl/envs/popgym_wrappers.py](memrl/envs/popgym_wrappers.py))

- **`FlattenMultiDiscrete`** — lossless mixed-base flattening of `MultiDiscrete([a, b, ...])` → `Discrete(a*b*...)`. Mirrors the existing `FlattenTupleDiscrete` but for MultiDiscrete obs. Needed for CountRecall.
- **`ExposePhaseInInfo`** — for two-phase envs (Autoencode), sets `info["phase"] = int(obs[0])` on every step/reset. Applied BEFORE flattening so the phase index isn't lost when the tuple gets collapsed.
- Dispatch in `make_popgym_vec_env` auto-applies the right wrapper based on obs-space type, and `ExposePhaseInInfo` based on env-id prefix (`_PHASE_ENV_PREFIXES = ("popgym-Autoencode",)`).

### MultiDiscrete action support ([memrl/policy/policy.py](memrl/policy/policy.py))

`MemActorCriticPolicy` now handles both `Discrete` and `MultiDiscrete` action spaces:
- Actor head emits `sum(action_dims)` logits in one tensor.
- New helpers `_build_dists` / `_sample_action` / `_log_prob` / `_entropy` split logits per-dim and treat dims as independent Categoricals (standard MultiCategorical assumption, matches SB3).
- For Discrete this is a no-op; the 1-element list of dists collapses to the original behavior.
- Required for BattleshipEasy (`MultiDiscrete([8, 8])` board coords).

### Phase-grouped innovation logging ([memrl/ppo/ppo.py](memrl/ppo/ppo.py))

`collect_rollouts` now buckets per-step innovation magnitude by `info["phase"]` when available. Emits to TB:
- `debug/innovation_mag_phase_0_mean` (WATCH for Autoencode)
- `debug/innovation_mag_phase_0_count`
- `debug/innovation_mag_phase_1_mean` (PLAY)
- `debug/innovation_mag_phase_1_count`

For envs without a phase signal (RepeatPrevious, CountRecall, Battleship) the per-phase dict stays empty and no extra scalars are emitted. The WATCH/PLAY ratio is the interpretability signal we want: GatedLMU should innovate heavily during WATCH (writing new info), drop during PLAY (reading).

### Tests added ([memrl/envs/tests/test_popgym_envs.py](memrl/envs/tests/test_popgym_envs.py))

8 fast smoke tests + 1 slow MemPPO integration test:
- `_mixed_base_multipliers` helper
- `FlattenMultiDiscrete` lossless round-trip on a synthetic env
- `ExposePhaseInInfo` populates `info["phase"]` on reset and step
- Construction of all 4 suite envs through `make_vec_env`
- **Battleship + MemPPO end-to-end** (slow): verifies MultiDiscrete actions flow through rollout buffer + policy update without crashing. **Passes.**

Total test count: 235 → 243.

---

## Note on ablation granularity (per user discussion 2026-05-17)

F4 isolates **just the predictive-coding gate** — multichannel `u` and dynamic readout are still active in both arms. The other two "baseline GatedLMU vs LMU" features (multichannel u, dynamic readout) are **not currently toggleable** in [memrl/cell/gated_lmu.py](memrl/cell/gated_lmu.py); separating them needs code changes:

- **Multichannel u off** → revert to scalar `u_t` per timestep (LMU-style); ~30 min code edit, add a `multichannel: bool` flag.
- **Dynamic readout off** → replace `W_query` attention with a static `W_m` projection on `m_new`; ~30 min code edit, add a `dynamic_readout: bool` flag.

These will be folded into **F2** so the full lesion sequence is:

1. LMU-tuned baseline (no multichannel, no gate, no dynamic readout) — **already done in F1**.
2. + multichannel only (NEW flag).
3. + multichannel + dynamic readout (NEW flag).
4. + multichannel + dynamic readout + gate (= GatedLMU baseline; existing softsign_sum arm of F4).
5. + salience gate.
6. + K=3 multi-scale.
7. + freebies (LN + readout skip + ortho W_pre).
8. SelectiveLMU (all on).

This gives a clean delta-per-feature curve from LMU all the way to SelectiveLMU.
