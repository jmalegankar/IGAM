# Moving the DTH-LMU ablation to the M4 Mac

**Date:** 2026-05-23
**Why:** The 3070 Ti box bottlenecks on per-step CPU↔GPU sync (lots of tiny
kernel launches, low arithmetic intensity). Apple Silicon's unified memory
eliminates that class of stall, so for this specific workload — small
(200 K param) recurrent cell with chunked TBPTT — the M4 often matches or
beats a 3070 Ti in wall-clock despite the lower peak compute.

The code is already Mac-clean: every CUDA call sits behind a
`torch.cuda.is_available()` guard, and I added `MEMRL_DEVICE`/`--device`
support so SB3's "auto" (which skips MPS) doesn't strand you on CPU.

---

## 1. What goes with you

| What | Where on Windows | Where on Mac |
|---|---|---|
| Source code | `C:\Users\jaima\OneDrive\Documents\research\IGAM\` | `~/research/IGAM` (or wherever) |
| Existing checkpoints | `runs/dth_lmu_ablations/**/latest.pt` | same relative path |
| Generated configs | `experiments/dth_lmu_ablations/_generated/*.yaml` | same |
| Per-run logs | `experiments/dth_lmu_ablations/_generated/_logs/` | optional — for forensic only |

Two transfer options:
1. **Git** — push from Windows, clone on Mac. Note that `latest.pt` files
   are large (~3 MB each, ×18 = ~54 MB total). Either commit them, use
   git-lfs, or `rsync`/iCloud them separately.
2. **iCloud Drive** — your `OneDrive\Documents\research\IGAM` may already
   sync via the OneDrive client. If your Mac has OneDrive installed,
   the project just shows up there. Easier but slower; verify the
   `latest.pt` files actually synced before resuming.

---

## 2. Setting up the venv on the Mac

```bash
# In the project root:
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If `popgym` fails to install on Apple Silicon, it's usually `pygame` or
`opencv-python` needing system libs. The handoff doc mentions the Mac
already had the code running, so this should "just work" — but if you
hit a wall, the failure mode is usually clear from pip's error.

Verify the cell:

```bash
python scripts/_smoke_dth_modes.py
```

Expected output (on M4, MPS picked up):

```
perf: {'tf32': True, 'cudnn_benchmark': True, ..., 'cuda_available': False, ...}
CUDA: False | device: CPU       ← script's print; not the actual training device
additive           h finite=True eps_mem=~0.20
delta              h finite=True eps_mem=~0.07
gated_delta        h finite=True eps_mem=~0.03
gated_delta_eps    h finite=True eps_mem=~0.07
```

(The "CUDA: False / device: CPU" line is just the smoke script's own
print. Training will use MPS via `resolve_device()`.)

---

## 3. Resuming the ablation

Same command as on Windows, plus the device override:

```bash
export MEMRL_DEVICE=mps
# (or pass --device mps to train.py explicitly)

python -m experiments.dth_lmu_ablations.run_ablations \
    --tasks autoencode_medium battleship_easy countrecall_medium \
    --modes additive delta gated_delta_eps \
    --seeds 0 1 \
    --timesteps 5_000_000 \
    --n-epochs 4 \
    --parallel 1
```

Note: **`--parallel 1` on Mac** is probably right at first — M4 base has
10 CPU cores but unified memory means each subprocess competes for the
same GPU. Single-process maxes out the GPU without contention. If you
see GPU util well under 80%, try `--parallel 2`.

The runner will auto-detect existing `latest.pt` files and resume each
run from where the Windows box left it.

### If MPS errors on some op
PyTorch's MPS backend has occasional op-coverage gaps. Symptom: the
worker errors out on first iteration with `NotImplementedError: The
operator ... is not currently supported on the MPS backend.`

Quick fix: set `PYTORCH_ENABLE_MPS_FALLBACK=1` before launching. That
silently falls back to CPU for unsupported ops. Slightly slower than
pure MPS but should keep things running.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Pause/resume still works identically
- Drop a `STOP` file at the repo root → all workers exit cleanly,
  flushing `latest.pt`.
- Ctrl+C → same.
- Re-run the same command → auto-resumes.

---

## 4. Watching it run

Open another terminal in the project dir:

```bash
tail -f experiments/dth_lmu_ablations/_generated/_logs/*.log | \
    grep -E "fps|iterations|eval/|ERROR|Traceback"
```

What "fast enough" looks like: on an M4 base, expect `fps` somewhere in
the 80-150 range (vs ~58 on the Windows 3070 Ti box). On M4 Pro/Max,
150-250. If it's under 30 something's wrong (probably falling back to
CPU for a hot op).

---

## 5. What's portable, what's not

- **Policy weights, optimizer state, step count, RNG**: cross-platform.
  `torch.save` is bit-identical across CUDA/MPS/CPU.
- **Episode state**: not preserved across resumes anyway (clean
  episode-boundary semantics — see the original session summary).
- **Per-CUDA RNG state**: saved on Windows, silently skipped on Mac
  (`torch.cuda.is_available()` is False there). No problem.
- **Determinism vs. continuous run**: the resumed-on-Mac trajectory
  will diverge slightly from a continuous-on-Windows trajectory due to
  different RNG paths after resume + MPS vs CUDA numerics. For a
  research ablation comparing modes pairwise within the same hardware,
  this doesn't matter.

---

## 6. If the M4 is slower than expected

In order of suspicion:
1. **PyTorch MPS fallback fired**: check `PYTORCH_MPS_HIGH_WATERMARK_RATIO`
   and the worker log for any "falling back to CPU" warnings. Forcing
   `PYTORCH_ENABLE_MPS_FALLBACK=1` masks crashes but slows things down.
2. **Device wasn't picked up**: confirm the worker log says
   `perf: device=mps (...)` not `device=cpu`.
3. **Wrong `--parallel`**: with a single GPU shared by all subprocesses,
   --parallel 2-3 saturates faster than --parallel 1, but only if env
   stepping has slack. For us env stepping is fast, so single process
   is likely the right call.

If after all that it's still slow: install `triton-windows` on the
3070 Ti box (`pip install triton-windows` + `MEMRL_COMPILE=1`) and run
there with compile. That was the other realistic 2× lever I had no way
to verify without a working Triton install.
