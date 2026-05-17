# IGAM Phase 1 Ablation — Run Instructions

Hi! Thanks for running these. This is a handoff doc with copy-paste commands.
Full background lives in `docs/adr/0005-selective-lmu-ablation.md` if you want
the why; below is just the *how*.

You're running **8 RL training jobs** that compare ablations of a recurrent
memory cell on two POPGym tasks. Each job is **5M env steps**:

- **~55 min/job** on Apple M3 (10-core CPU)
- **~10 min/job** on a 3070-class GPU
- **~5 min/job** on an A100

All jobs are independent — run them in whatever parallelism your machine
supports. No need to do them in order; the priority labels just tell me which
results I most want first if your time is limited.

---

## Setup

```bash
cd <repo_root>                              # e.g. /Users/jmalegaonkar/Documents/IGAM
source .venv/bin/activate || true           # if you use venv; otherwise ensure deps installed
mkdir -p /tmp/igam_ablation
```

Verify it works:
```bash
.venv/bin/python -c "from memrl.cell import SelectiveLMU; print('ok')"
```
Should print `ok`.

---

## The 8 commands

Each command is one job. They write a log to `/tmp/igam_ablation/<name>.log` and
full eval data under `runs/<config_stem>/<cell_name>/seed_0_<timestamp>/`.

### Priority 1 — the critical no-gate test

```bash
# MSLMU Medium
.venv/bin/python train.py \
  --config benchmarks/phase_a/ablation/mslmu_medium.yaml \
  --seed 0 \
  > /tmp/igam_ablation/mslmu_medium.log 2>&1

# MSLMU Hard
.venv/bin/python train.py \
  --config benchmarks/phase_a/ablation/mslmu_hard.yaml \
  --seed 0 \
  > /tmp/igam_ablation/mslmu_hard.log 2>&1
```

### Priority 2

```bash
# GatedMSLMU Medium
.venv/bin/python train.py \
  --config benchmarks/phase_a/ablation/gated_mslmu_medium.yaml \
  --seed 0 \
  > /tmp/igam_ablation/gated_mslmu_medium.log 2>&1

# GatedMSLMU Hard
.venv/bin/python train.py \
  --config benchmarks/phase_a/ablation/gated_mslmu_hard.yaml \
  --seed 0 \
  > /tmp/igam_ablation/gated_mslmu_hard.log 2>&1
```

### Priority 3

```bash
# MCLMU Medium
.venv/bin/python train.py \
  --config benchmarks/phase_a/ablation/mclmu_medium.yaml \
  --seed 0 \
  > /tmp/igam_ablation/mclmu_medium.log 2>&1

# MCLMU Hard
.venv/bin/python train.py \
  --config benchmarks/phase_a/ablation/mclmu_hard.yaml \
  --seed 0 \
  > /tmp/igam_ablation/mclmu_hard.log 2>&1
```

### Priority 4

```bash
# LMU Medium
.venv/bin/python train.py \
  --config benchmarks/phase_a/ablation/lmu_medium.yaml \
  --seed 0 \
  > /tmp/igam_ablation/lmu_medium.log 2>&1

# LMU Hard
.venv/bin/python train.py \
  --config benchmarks/phase_a/ablation/lmu_hard.yaml \
  --seed 0 \
  > /tmp/igam_ablation/lmu_hard.log 2>&1
```

---

## Running in parallel

### On a GPU host
Just launch as many as fit in GPU memory. Each job uses ~1-2 GB. A 3070 (8GB)
should hold 4 jobs comfortably. Append `&` to background each command, e.g.:

```bash
.venv/bin/python train.py --config benchmarks/phase_a/ablation/mslmu_medium.yaml --seed 0 > /tmp/igam_ablation/mslmu_medium.log 2>&1 &
.venv/bin/python train.py --config benchmarks/phase_a/ablation/mslmu_hard.yaml   --seed 0 > /tmp/igam_ablation/mslmu_hard.log   2>&1 &
.venv/bin/python train.py --config benchmarks/phase_a/ablation/gated_mslmu_medium.yaml --seed 0 > /tmp/igam_ablation/gated_mslmu_medium.log 2>&1 &
.venv/bin/python train.py --config benchmarks/phase_a/ablation/gated_mslmu_hard.yaml   --seed 0 > /tmp/igam_ablation/gated_mslmu_hard.log   2>&1 &
wait
echo "first 4 done"
```

Then the next 4. Or fewer in parallel if VRAM is tight.

### On a Mac M3 / multi-core CPU
Prefix each command with thread limits so two jobs don't fight over cores.
10-core machine → 2 parallel × 4 threads each = 8 cores used, 2 free:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  .venv/bin/python train.py --config benchmarks/phase_a/ablation/mslmu_medium.yaml --seed 0 > /tmp/igam_ablation/mslmu_medium.log 2>&1 &

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  .venv/bin/python train.py --config benchmarks/phase_a/ablation/mslmu_hard.yaml --seed 0 > /tmp/igam_ablation/mslmu_hard.log 2>&1 &

wait
```

For 4 in parallel, set threads to 2 each instead.

---

## Health checks while running

Tail any log to confirm a job hasn't crashed:

```bash
tail -20 /tmp/igam_ablation/mslmu_medium.log
```

**Signs of a healthy run:**
- `ep_rew_mean` is a finite number (typically starts around -0.5, may climb toward 0 or positive)
- `state_actor_m_norm_max` stays under ~500
- `grad_norm` under ~10 (or shows the clipping value 0.5)
- `approx_kl` under ~0.05
- New `iterations` ticking up roughly every 30-60 seconds

**Signs of trouble — kill the job and let me know:**
- `state_actor_m_norm_max > 1000` (memory exploding)
- `approx_kl > 0.1` for several iterations (PPO diverging)
- `Traceback` anywhere in the log
- `fps` drops below ~50 (something is seriously stuck)

To kill a specific run:
```bash
ps aux | grep train.py | grep mslmu_medium      # find the PID
kill <PID>
```

To kill ALL training runs (nuclear option):
```bash
pkill -f "train.py"
```

---

## What to send back when done

The minimum I need is a single summary line per run. Run this after all 8
complete:

```bash
.venv/bin/python -c "
import numpy as np, glob
for d in sorted(glob.glob('runs/*/seed_0_*/eval/evaluations.npz')):
    z = np.load(d); r = z['results'].mean(axis=1); ts = z['timesteps']
    run = '/'.join(d.split('/')[1:3])
    print(f'{run:55} best={r.max():+.3f}@{int(ts[r.argmax()]):>8,}  last={r[-1]:+.3f}@{int(ts[-1]):>8,}  evals={len(ts)}')
"
```

Paste me the output. That's all I need to fill in the results table.

If you want to also send the raw data (it's small, total <1 MB), tar it up:

```bash
tar czf ablation_results.tar.gz \
  runs/lmu_*/LMU/seed_0_*/eval/ \
  runs/mclmu_*/GatedLMU/seed_0_*/eval/ \
  runs/mslmu_*/GatedLMU/seed_0_*/eval/ \
  runs/gated_mslmu_*/GatedLMU/seed_0_*/eval/ \
  /tmp/igam_ablation/*.log
```

---

## Troubleshooting

**Q: `train.py` errors with `ModuleNotFoundError: No module named 'igam'`**
A: You're not in the repo root. `cd` to the directory containing `train.py` and try again.

**Q: `train.py` errors with `KeyError: 'SelectiveLMU'` or similar**
A: The cell registry isn't picking up the new cells. Check that `igam/cell/__init__.py` exports `SelectiveLMU`. If not, the repo is stale — let me know.

**Q: The job runs but `state_actor_m_norm_max` goes to 1e4 after iteration 2**
A: This is a known failure mode from a previous bug — should be fixed. If you see it on the current configs, the fix didn't ship. Kill the run and ping me.

**Q: I want to override the seed**
A: Just change `--seed 0` to `--seed 1` (or any int). Each seed writes to a separate timestamped directory under `runs/`. (Phase 2 will run multiple seeds — for Phase 1, seed 0 only.)

**Q: A run finished but produced `nan` for `ep_rew_mean`**
A: That's an instability. Send the log and I'll diagnose.

**Q: How do I know it's done?**
A: The log ends with `Done. Final model saved to ...`. Also `total_timesteps` reaches `5000000`-ish (it goes a bit past due to rollout boundary alignment).

---

Thanks again — looking forward to the results.
