"""S13 infra probe: smoke + fps for every candidate cell at real config HPs.

Builds MemPPO on MiniGrid-MemoryS13-v0 with the full_system.yaml HPs and runs
~2 rollouts per cell, reporting wall-clock fps (and crashes). Used to estimate
the cost of the cells × intrinsics S13 sweep BEFORE committing to 5M runs.
"""
from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore")

from memrl.envs import make_vec_env
from memrl.exploration import make_intrinsic
from memrl.ppo import MemPPO
from train import DEFAULT_CELL_KWARGS, make_cell_factory

# full_system.yaml HPs (S13 reference).
HP = dict(
    n_envs=16, encoder_dim=128, encoder_hidden=256, lr=3e-4,
    n_steps=512, n_epochs=4, gamma=0.999, gae_lambda=0.98, clip_range=0.2,
    ent_coef=0.008, vf_coef=1.0, max_grad_norm=0.5, target_kl=0.05,
    chunk_len=16, n_chunks_per_batch=32,
)
ENV = "MiniGrid-MemoryS13-v0"
CELLS = ["GRU", "LSTM", "LRU", "Mamba2", "FFM", "GatedDeltaNet", "SHM", "GTrXL", "DTHLMU"]
PROBE_STEPS = HP["n_steps"] * HP["n_envs"] * 2   # ~2 rollouts


def build(cell, intrinsic="none"):
    env = make_vec_env(ENV, n_envs=HP["n_envs"], seed=0)
    factory = make_cell_factory(cell, DEFAULT_CELL_KWARGS.get(cell, {}), HP["encoder_dim"])
    im = None
    if intrinsic != "none":
        from memrl.policy.encoder import FlatEncoder  # obs_dim helper
        import numpy as np
        obs_dim = int(np.prod(env.observation_space.shape))
        im = make_intrinsic(intrinsic, n_envs=HP["n_envs"], obs_dim=obs_dim,
                            n_actions=env.action_space.n, device="cpu")
    model = MemPPO(
        env=env, cell_factory=factory, lr=HP["lr"], n_steps=HP["n_steps"],
        n_epochs=HP["n_epochs"], gamma=HP["gamma"], gae_lambda=HP["gae_lambda"],
        clip_range=HP["clip_range"], ent_coef=HP["ent_coef"], vf_coef=HP["vf_coef"],
        max_grad_norm=HP["max_grad_norm"], target_kl=HP["target_kl"],
        lambda_intrinsic=0.1 if im else 0.0, intrinsic_module=im,
        intrinsic_source="eps_mem", encoder_dim=HP["encoder_dim"],
        encoder_hidden=HP["encoder_hidden"], chunk_len=HP["chunk_len"],
        n_chunks_per_batch=HP["n_chunks_per_batch"], verbose=0, seed=0, device="cpu",
    )
    return env, model


def n_params(model):
    return sum(p.numel() for p in model.policy.parameters())


print(f"S13 probe — env={ENV}, HPs from full_system.yaml, ~{PROBE_STEPS} steps/cell\n")
print(f"{'cell':16s} {'status':8s} {'params':>10s} {'fps':>7s} {'5M_run':>9s}")
print("-" * 56)
for cell in CELLS:
    try:
        env, model = build(cell)
        t0 = time.perf_counter()
        model.learn(total_timesteps=PROBE_STEPS)
        dt = time.perf_counter() - t0
        fps = PROBE_STEPS / dt
        eta_5m = 5_000_000 / fps / 3600  # hours
        print(f"{cell:16s} {'OK':8s} {n_params(model):>10,} {fps:>7.0f} {eta_5m:>7.1f}h")
        env.close()
    except Exception as e:
        print(f"{cell:16s} {'FAIL':8s}  {type(e).__name__}: {str(e)[:60]}")

# Intrinsic plumbing check on the cheapest cell.
print("\nIntrinsic plumbing (GRU):")
for intr in ["e3b_rand", "rnd", "noveld", "icm", "e3b_obs"]:
    try:
        env, model = build("GRU", intrinsic=intr)
        model.learn(total_timesteps=HP["n_steps"] * HP["n_envs"])
        print(f"  {intr:10s} OK")
        env.close()
    except Exception as e:
        print(f"  {intr:10s} FAIL  {type(e).__name__}: {str(e)[:70]}")
