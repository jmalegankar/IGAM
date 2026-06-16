# syntax=docker/dockerfile:1
#
# memrl / IGAM training image for a Kubernetes GPU cluster.
#
# Clones the repo, installs all libs, and leaves the container sitting at the
# repo root ready to run the S13 baseline launch scripts
# (experiments/s13_baseline/scripts/run_*.sh).
#
# Build:
#   DOCKER_BUILDKIT=1 docker build -t <registry>/memrl-s13:latest .
#   # pin a commit/branch:        --build-arg GIT_REF=<sha-or-branch>
#   # private repo (BuildKit):    --secret id=ghtoken,src=token.txt   (see note below)
#
# Run (single GPU pod, whole sweep):
#   docker run --gpus all -e WANDB_API_KEY=xxx -v /data/runs:/workspace/IGAM/runs \
#       <registry>/memrl-s13:latest
#
# Run one cell (override the default command):
#   docker run --gpus all -e WANDB_API_KEY=xxx <img> \
#       bash experiments/s13_baseline/scripts/run_GRU.sh
#
# CPU smoke test (no GPU, no wandb):
#   docker run -e DEVICE=cpu -e EXTRA="--no-wandb --total-timesteps 8192" \
#       -e SEEDS=0 -e PARALLEL=0 <img> bash experiments/s13_baseline/scripts/run_GRU.sh

# ── Base: CUDA-enabled PyTorch (torch 2.6.0 + CUDA 12.4 preinstalled) ────────
# Using the official build means the multi-GB GPU torch wheel isn't downloaded
# again, and `--device cuda` works without extra CUDA setup. 2.6.0 satisfies the
# repo's `torch>=2.2`, so `pip install .` leaves this torch untouched.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# ── Which repo / ref to bake in ─────────────────────────────────────────────
ARG REPO_URL=https://github.com/jmalegankar/IGAM.git
ARG GIT_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── System deps ─────────────────────────────────────────────────────────────
# git: to clone. libgl1/libglib2.0-0: pulled in by some gymnasium/minigrid
# render paths; cheap insurance against import-time failures (training itself
# does not render).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── Clone the repo ──────────────────────────────────────────────────────────
# Public repo: this works as-is. Private repo with BuildKit secret instead:
#   RUN --mount=type=secret,id=ghtoken \
#       git clone --depth 1 --branch ${GIT_REF} \
#       https://x-access-token:$(cat /run/secrets/ghtoken)@github.com/jmalegankar/IGAM.git IGAM
RUN git clone --depth 1 --branch ${GIT_REF} ${REPO_URL} IGAM

WORKDIR /workspace/IGAM

# ── Python deps ─────────────────────────────────────────────────────────────
# `pip install ".[wandb]"` pulls all declared deps — including minigrid (core;
# imported eagerly by memrl/envs/__init__.py), popgym (base; Autoencode/
# Battleship), and wandb (needed by configs' `wandb: true`; disable per-run with
# EXTRA="--no-wandb").
# NOTE: the `popgym-arcade` extra is deliberately NOT installed — it pulls `jax`,
# which on this CUDA base image resolves to CUDA jaxlib + bundled nvidia-* wheels
# (a multi-GB layer). The arcade experiments were cut and no current env uses it
# (popgym-arcade-* wrappers import lazily, only for those env ids), so omitting it
# keeps the image small. Re-add ".[wandb,popgym-arcade]" only to revive an arcade run.
RUN python -m pip install --upgrade pip \
 && python -m pip install ".[wandb]"

# ── memory-gym (MysteryPath / MortarMayhem pixel memory benchmarks) ──────────
# Out-of-band because memory-gym pins pygame==2.4.0, which has no wheel for
# modern Pythons and fails to build. Install a modern pygame first, then
# memory-gym with --no-deps so the pin can't downgrade it. Imported lazily (only
# when an id like MysteryPath-Grid-v0 is requested), so inert for other runs.
RUN python -m pip install "pygame>=2.6" \
 && python -m pip install memory-gym --no-deps

# ── Runtime knobs the run_*.sh scripts read (all overridable at `docker run`) ─
#   PYTHON   : no .venv in the image, so use the container interpreter.
#   DEVICE   : GPU by default; set DEVICE=cpu for a smoke test.
#   RUNS_DIR : output root — mount a volume/PVC here so results survive the pod.
#   SEEDS    : seeds per cell. PARALLEL=1 runs them concurrently on one GPU.
ENV PYTHON=python \
    DEVICE=cuda \
    RUNS_DIR=runs/s13_baseline \
    SEEDS="0 1 2" \
    PARALLEL=1

# Sanity: fail the build if the package / required libs don't import.
RUN python -c "import memrl, train, minigrid, wandb, stable_baselines3, memory_gym; print('imports OK')"

# Default command: run the full sweep (cells serial, seeds parallel within each).
# Override with a per-cell script for one-pod-per-cell scheduling, e.g.:
#   args: ["bash", "experiments/s13_baseline/scripts/run_GRU.sh"]
# CMD ["bash", "experiments/s13_baseline/scripts/run_all.sh"]
