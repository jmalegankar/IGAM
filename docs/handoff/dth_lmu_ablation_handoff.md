# DTH-LMU Hebbian-Rule Ablation — GPU Handoff

**Date written:** 2026-05-21
**Hardware target:** consumer GPU (tested designs for ~8GB VRAM, e.g. RTX 3070)
**Status:** code merged on local mac, ready to run on the GPU box

You are picking up a research run mid-stream. The user (Jai) prepared the
code on a Mac CPU box; you are running it on the GPU. Don't re-architect
anything; your job is to execute the ablation, monitor it, and report back.

---

## 1. What you're testing in one sentence

**Does upgrading the DTH-LMU's Hebbian matrix from additive writes
(`M += k⊗v/A`) to an innovation-gated delta rule
(`M = α(h)·M + σ(a·ε_mem − b)·(v − Mk) k^T`) improve POPGym performance on
the three diagnostic tasks where the additive cell failed (AutoencodeMedium,
CountRecallMedium, BattleshipEasy)?**

The hypothesis (synthesized from three deep-research docs the user asked
to evaluate): targeted overwrite via the delta rule fixes the
catastrophic-interference floor that hits additive Hebbian writes at long
horizons, and gating writes on memory-conditioned prediction error
(`ε_mem = mean((v − M·k)²)`) preserves capacity for genuinely novel events.

If this works, it validates the **Doc 1 (Gated DeltaNet) + Doc 2
(memory-conditioned curiosity)** synthesis. The full vision (LegS-tensor M,
world predictor, Mamba-3 discretization) is deferred to a v2 paper.

---

## 2. Recent code changes you should be aware of

Two changes landed this session before handoff. Don't re-do them. Look at
them if anything below confuses you.

### 2a. LegS stability fix in `memrl/cell/dth_lmu.py`

The original LegS branch used forward Euler `m + dt·(−A·m + B·u)` with
`dt = 1/t`. This is **spectrally stable but non-normal in 2-norm** — the
matrix `(I − A/t)` has 2-norm ≈ 20 for all `t < 500`, causing transient
explosion to `1e21` in float32 within ~20 steps, saturating `tanh(h)` and
spiking the critic to inf.

**Fix:** per-step implicit Euler via `solve_triangular`. A is lower-triangular
so each step is one O(D²) solve. Numerically verified stable from t=1
through 200 steps with bounded `m_legs.max() < 1`. The running-mean property
of degree-0 coefficient is preserved with ~1% bias at t=100 (negligible).

If you see LegS-related instability come back, check `step()` around the
`# ── LegS ──` block — it should be using `torch.linalg.solve_triangular`,
not the previous forward-Euler form.

### 2b. Hebbian mode refactor

`_HebbianMemory` now has four selectable modes via the `mode` kwarg:

| mode | update rule | role |
|---|---|---|
| `additive` | `M += (k⊗v)/A` | baseline (the failing one) |
| `delta` | `M += (v − Mk) k^T` (β=α=1) | DeltaNet — tests "does targeted overwrite alone help?" |
| `gated_delta` | `α(h)·M + β(h)·(v − Mk) k^T` | Gated DeltaNet — tests "do learned scalar gates add value?" |
| `gated_delta_eps` | `α(h)·M + σ(a·ε_mem − b)·(v − Mk) k^T` | **recommended** — tests "does memory-conditioned curiosity help?" |

`ε_mem = mean((v − M·k)²)` is computed in all modes and exposed as a side
output `"eps_mem"` (alongside the existing `"innovation"` from the fast LMU)
so plots can compare prediction-error trajectories across modes.

Default in `DTHLMU.__init__` is `hebbian_mode="gated_delta_eps"`. The four
diagnostic configs in `benchmarks/phase_a/ablation/dth_lmu_*_15M.yaml`
explicitly set it; the runner overrides this per variant.

**Validated on Mac CPU:** all four modes run cleanly on a 20-step toy
trajectory; ε_mem decreases monotonically across modes
(additive 0.144 → delta 0.079 → gated_delta 0.048 → gated_delta_eps 0.030),
which is the mechanism working as designed in microcosm.

---

## 3. What to run

### 3a. First — sanity check that training launches

Run one minimal job to confirm the GPU box has all deps and the cell
imports cleanly:

```bash
python -c "
import torch
from memrl.cell import DTHLMU
print('CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
for mode in ('additive', 'delta', 'gated_delta', 'gated_delta_eps'):
    cell = DTHLMU(input_size=8, hidden_size=64, hebbian_mode=mode).cuda()
    s = cell.init_state(4, device='cuda')
    for _ in range(20):
        h, s, side = cell.step(torch.randn(4, 8, device='cuda'), s)
    print(f'{mode:18s} h finite={h.isfinite().all().item()} eps_mem={side[\"eps_mem\"].mean():.4f}')
"
```

Expected: all four modes finite, with `eps_mem` decreasing across modes
(additive highest, gated_delta_eps lowest).

### 3b. Then — the ablation grid

Use `experiments/dth_lmu_ablations/run_ablations.py`. It generates per-variant
configs in `experiments/dth_lmu_ablations/_generated/` and launches
`train.py` for each `(task × mode × seed)` combination.

**Recommended starting command** (matches Jai's expected scope — adjust
`--parallel` based on what fits in your GPU's VRAM):

```bash
python -m experiments.dth_lmu_ablations.run_ablations \
    --tasks autoencode_medium battleship_easy countrecall_medium \
    --modes additive delta gated_delta_eps \
    --seeds 0 1 \
    --timesteps 5_000_000 \
    --n-epochs 4 \
    --parallel 3
```

That's 18 runs × 5M steps. On a 3070 with `--parallel 3`, estimate ~8–12 hours
wall-clock.

### 3c. Parallelism tuning on first launch

Before committing to the full grid, run **one variant with --parallel 1** for
a few minutes to measure GPU utilization and VRAM:

```bash
python -m experiments.dth_lmu_ablations.run_ablations \
    --tasks countrecall_medium --modes gated_delta_eps --seeds 0 \
    --timesteps 500_000 --n-epochs 4 --parallel 1
```

While it runs, in another terminal: `nvidia-smi -l 2`. Record:
- VRAM used per run (call it `V` MB)
- GPU utilization %

Then pick `--parallel` for the full grid:
- If GPU util < 60% AND `V × N < 0.85 × total_VRAM`: bump N
- If GPU util > 90% at N=2: stay at 2
- For a 3070 (8 GB), 3 is the typical sweet spot

If you see CUDA OOM, drop `--parallel` by 1.

### 3d. Optional speedup that's free on Ampere

Add this to the top of `train.py` (just after imports) for ~1.5–2×
matmul speedup on the 3070 (it's Ampere):

```python
import torch
torch.set_float32_matmul_precision('high')   # enable TF32
```

This is a one-line change with no accuracy implications worth caring about
for RL. Apply it if you can; revert if anything looks broken.

---

## 4. What to look for in the results

### 4a. Where logs land

- **Per-run stdout** (when `--parallel > 1`): captured to
  `experiments/dth_lmu_ablations/_generated/_logs/<label>.log`
- **TensorBoard scalars + checkpoints**: `runs/dth_lmu_ablations/<config_stem>/DTHLMU/seed_<n>_<timestamp>/`
- **Generated configs** (what was actually run): `experiments/dth_lmu_ablations/_generated/dth_lmu_<task>_<mode>_<N>M.yaml`

### 4b. The comparisons that matter

Plot mean episode return vs steps for each `(task, mode)` group, averaged
over seeds. The pattern we **expect** to see:

| Task | What we expect |
|---|---|
| **AutoencodeMedium** | `gated_delta_eps` ≫ `additive`; `delta` somewhere in between. This task is the cleanest test of "does targeted overwrite + ε gating help content recall?" |
| **BattleshipEasy** | Similar pattern. Spatial key-value lookup is what delta-rule memory is designed for. |
| **CountRecallMedium** | All modes should perform comparably. The LegS implicit-Euler fix landed before this run, so the running-mean computation that this task needs is already working; the Hebbian update mode shouldn't change much here. If `gated_delta_eps` *hurts* here, something's wrong. |
| **RepeatPrevious** (only if run) | All modes comparable. This is a fast-LMU task; Hebbian rarely matters. |

**Report back to user with:** 1 plot per task (lines for each mode, shaded
seed variance), plus a table of final-100k-step mean return per (task, mode).

### 4c. ε_mem diagnostic

The cell now exposes `"eps_mem"` as a side output. If the TensorBoard
logger captures it (check `memrl/ppo/` or wherever side outputs are
recorded), the trajectories should show:
- `additive`: ε_mem monotonically grows (interference accumulating)
- `delta`: ε_mem stays bounded but doesn't necessarily decrease
- `gated_delta_eps`: ε_mem decreases over time as the cell learns to write only on novel keys

If side outputs aren't already logged, don't bother adding the logging
for this run — it's a nice-to-have, not the headline metric.

---

## 5. Things that might go wrong

### "ModuleNotFoundError: popgym"

POPGym isn't installed by default. The Mac box doesn't have it either —
that's why the import check during code development used a different
training entry point. On the GPU box you'll need `pip install popgym` in
the venv before runs work. If POPGym is missing, the configs will fail at
env construction, not at cell construction.

### Critic still goes to inf at start of training

Means the LegS implicit-Euler fix isn't active. Verify the LegS section of
`memrl/cell/dth_lmu.py:step()` uses `torch.linalg.solve_triangular(A_impl,
...)` and **not** the old forward-Euler form
`m_legs + dt * (-A @ m_legs + B * u_x)`. If the file got reverted, restore
from git.

### OOM with `--parallel N`

Drop N by 1. Each DTH-LMU run is small (cell params ~100k) but PPO's
rollout buffer + multiple env workers add up.

### Runs hang or appear to deadlock

The `ThreadPoolExecutor` in the runner only blocks Python threads;
subprocesses run independently. If you `Ctrl-C` the runner, lingering
`python -m train` subprocesses won't die automatically. Check with `pgrep
-f "python -m train"` and `kill` them manually.

### β saturates at 0 or 1 in `gated_delta_eps`

If you see `eps_mem` consistently very large or very small, the gate
saturates and the mode degenerates. The `(a_M, b_M)` parameters should
learn to recenter, but if they're stuck, the fix is to standardize
`ε_mem` by a running stddev before passing to σ. This is a known
limitation noted in the design discussion — don't fix it on the GPU run;
flag it and let Jai decide.

---

## 6. What to report back

Once the grid finishes (or partway, if it'll take >24h), prepare a short
summary:

1. **One sentence per task** comparing the three modes' final returns.
2. **A plot** (or text table) of mean ± seed-stddev return at 5M steps.
3. **Whether the hypothesis held**: did `gated_delta_eps` beat `additive`
   on AutoencodeMedium and BattleshipEasy by ≥10% relative? That's the
   threshold Jai set for "synthesis validated, proceed to Stage 2".
4. **Any anomalies**: instabilities, NaN losses, modes that diverged,
   wall-clock surprises.

Don't write a long writeup. Half a page of bullets is fine — Jai prefers
honest assessment over polished narrative.

---

## 7. Where the reasoning behind all this lives

If you want to understand *why* the architecture is the way it is, the
relevant files in this repo:

- **`docs/adr/0002-dual-timescale-hebbian-lmu.md`** — original design rationale
  (CLS theory, fast/slow LMU split, Hebbian outer-product motivation)
- **`memrl/cell/dth_lmu.py`** — the cell itself; docstrings explain each
  component (fast LegT bank, slow LegT bank, LegS integrator, Hebbian M)
- **`memrl/cell/lmu.py`** — the canonical LMU baseline and the HiPPO
  matrix builders `_legt_zoh_matrices` and `_legs_matrices`

The deep-research docs Jai had me evaluate aren't checked in (they were
ephemeral chat context). The synthesis I landed on, in two sentences:

> Replace additive Hebbian writes with the Gated DeltaNet rule (Doc 1's
> central insight: this is the proven mechanism for targeted memory edits
> in linear attention). Drive the write strength with memory-conditioned
> prediction error ε_mem (Doc 2's insight: the cell becomes curious about
> gaps in its own hippocampus). Defer everything else (LegS-tensor M,
> world-model predictor, Mamba-3 discretization, complex-valued state) to
> a v2 paper.

---

## 8. Files you'll touch (or that touch you)

| File | Why it matters |
|---|---|
| `memrl/cell/dth_lmu.py` | The cell. Don't edit unless something's broken. |
| `memrl/cell/lmu.py` | HiPPO matrix builders. Don't touch. |
| `experiments/dth_lmu_ablations/run_ablations.py` | The runner. Read once before launching. |
| `experiments/dth_lmu_ablations/_generated/*.yaml` | Generated per-variant configs (created by the runner). Inspect if a run misbehaves. |
| `experiments/dth_lmu_ablations/_generated/_logs/*.log` | Per-parallel-run stdout. Tail these to debug. |
| `benchmarks/phase_a/ablation/dth_lmu_*_15M.yaml` | Base task configs the runner derives variants from. |
| `train.py` | The PPO training entry point. Add TF32 line here if you want the free speedup. |
| `runs/dth_lmu_ablations/` | Where TensorBoard logs + checkpoints land. |

Good luck. If anything is genuinely ambiguous, ask Jai before improvising —
this is a research run with real opportunity cost, not a quick eval.
