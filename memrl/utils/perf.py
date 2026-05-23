"""Performance knobs for the 3070 Ti + i9-11900K box.

These are applied at process start in train.py. They are conservative
defaults — verified safe for PPO on the DTH-LMU cell — but can be
disabled or tuned via env vars without code edits:

    MEMRL_TF32=0              disable TF32 (matmul stays in fp32)
    MEMRL_CUDNN_BENCHMARK=0   disable cudnn algo autotuning
    MEMRL_TORCH_THREADS=4     cap intra-op CPU threads
                              (default: 4 — picked so 3 parallel runs share 16
                              logical cores without oversubscribing)
    MEMRL_COMPILE=1           torch.compile the policy (off by default —
                              opt-in because the first iteration takes a
                              30-60s warmup, and a tiny fraction of cells
                              hit Dynamo errors on Windows. When it works
                              it gives ~1.3-2x on the PPO update step.)
    MEMRL_DEVICE=auto         device to use. Default "auto" picks CUDA →
                              MPS (Apple Silicon) → CPU in that order. SB3's
                              built-in "auto" ignores MPS, so override is
                              needed to get GPU acceleration on Macs.

The runner sets MEMRL_TORCH_THREADS per worker based on --parallel.
"""

from __future__ import annotations

import os

import torch


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip() not in ("0", "false", "False", "")


def apply_perf_defaults() -> dict:
    """Apply TF32 / cudnn benchmark / thread caps. Returns the chosen settings
    so the caller can log them."""
    tf32             = _env_bool("MEMRL_TF32", True)
    cudnn_benchmark  = _env_bool("MEMRL_CUDNN_BENCHMARK", True)
    torch_threads    = int(os.environ.get("MEMRL_TORCH_THREADS", "4"))

    if tf32:
        # Enables TF32 matmul on Ampere (3070 Ti is sm_86). ~1.5–2x speedup on
        # large fp32 GEMMs with no meaningful accuracy hit for RL.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cudnn_benchmark:
        # PPO rollouts use fixed batch shapes within a run; let cudnn pick the
        # fastest kernel for each shape. Adds a one-time warmup cost.
        torch.backends.cudnn.benchmark = True

    # Cap CPU threads. The env-stepping happens in the same process as the
    # learner (DummyVecEnv), so torch eating all 16 logical cores hurts both
    # ourselves and any sibling parallel runs.
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
        # interop threads default to a small number already; leave them alone.

    return {
        "tf32": tf32,
        "cudnn_benchmark": cudnn_benchmark,
        "torch_threads": torch_threads,
        "compile": _env_bool("MEMRL_COMPILE", False),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def resolve_device(override: str | None = None) -> str:
    """Pick a torch device string honoring MEMRL_DEVICE / --device override.

    SB3's built-in ``device="auto"`` chooses CUDA → CPU and **ignores MPS**.
    On an Apple Silicon Mac that means training falls back to CPU even
    though the MPS GPU is sitting idle. This helper picks CUDA → MPS → CPU
    instead. Pass the result to ``MemPPO(device=...)``.

    Resolution order:
        1. ``override`` argument (CLI --device)
        2. ``MEMRL_DEVICE`` env var
        3. "auto" → first available of cuda / mps / cpu
    """
    val = override or os.environ.get("MEMRL_DEVICE", "auto")
    val = val.strip().lower()
    if val in ("cuda", "mps", "cpu"):
        return val
    # auto path
    if torch.cuda.is_available():
        return "cuda"
    # mps backend was added in PyTorch 1.12; check for both build + runtime availability.
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return "mps"
    return "cpu"


def maybe_compile_policy(policy):
    """Wrap policy submodules with ``torch.compile`` when MEMRL_COMPILE=1.

    Compiles the encoder, cell, and head modules — not the full policy — so
    Dynamo doesn't have to introspect SB3's distribution/sampling code,
    which it sometimes chokes on. Each compiled module is replaced on the
    policy in-place; original Python attributes are preserved so
    ``policy.cell_actor.init_state()`` etc. still work.

    Returns True if compile was attempted, False otherwise. Falls back
    silently (no compile) if Triton isn't installed — Triton wheels for
    Windows are spotty and a missing-Triton crash would otherwise only
    appear deep inside the first forward pass, after `latest.pt` was
    already loaded but before anything useful happened.
    """
    if not _env_bool("MEMRL_COMPILE", False):
        return False

    # Pre-check: torch.compile's default 'inductor' backend uses Triton.
    # If Triton isn't available, compile would defer the crash to first
    # forward — disable up front instead so resume still progresses.
    try:
        import triton  # noqa: F401
    except ImportError:
        print("  perf: MEMRL_COMPILE=1 requested but `triton` is not "
              "installed (Triton wheels for Windows are unreliable). "
              "Proceeding uncompiled — unset MEMRL_COMPILE or install "
              "triton-windows to silence this notice.")
        return False

    try:
        # Use the default ('inductor') backend; pre-grouped per-module compile
        # avoids the issues SB3's full-policy compile sometimes hits.
        for name in ("encoder_actor", "encoder_critic",
                     "cell_actor", "cell_critic",
                     "actor", "critic"):
            mod = getattr(policy, name, None)
            if mod is None:
                continue
            compiled = torch.compile(mod, mode="reduce-overhead", dynamic=False)
            setattr(policy, name, compiled)
        print("  perf: torch.compile applied to policy submodules "
              "(first iter will be slow — Inductor warmup)")
        return True
    except Exception as e:    # pragma: no cover — env-specific
        print(f"  perf: torch.compile failed ({e}); proceeding uncompiled.")
        return False
