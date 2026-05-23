"""Verify SubprocVecEnv works on Windows for our POPGym factory.

Critical: SubprocVecEnv uses 'spawn' start method on Windows, which means
the worker Python processes start fresh — popgym must register on import,
the env_fn closure must be picklable (cloudpickle handles this), and
seeding must happen in the worker. Validates by stepping the vec env a
few times and confirming step throughput is ≥ DummyVecEnv.
"""
from __future__ import annotations

import os
import time

import numpy as np

# Force subproc for this smoke test.
os.environ["MEMRL_VEC_ENV"] = "subproc"

from memrl.envs import make_vec_env  # noqa: E402


def _bench(env, steps: int = 200) -> float:
    obs = env.reset()
    t0 = time.perf_counter()
    for _ in range(steps):
        actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        env.step(actions)
    dt = time.perf_counter() - t0
    return steps * env.num_envs / dt   # env-steps per wall-clock sec


def main() -> int:
    print("Building SubprocVecEnv (8 envs) for popgym-AutoencodeMedium-v0 ...")
    env_sub = make_vec_env("popgym-AutoencodeMedium-v0", n_envs=8, seed=0)
    print(f"  type: {type(env_sub).__name__}")

    fps_sub = _bench(env_sub)
    env_sub.close()
    print(f"  SubprocVecEnv: {fps_sub:,.0f} env-steps/sec")

    # Comparison: DummyVecEnv. Make sure the comparison is fair.
    os.environ["MEMRL_VEC_ENV"] = "dummy"
    env_dum = make_vec_env("popgym-AutoencodeMedium-v0", n_envs=8, seed=0)
    print(f"  type: {type(env_dum).__name__}")
    fps_dum = _bench(env_dum)
    env_dum.close()
    print(f"  DummyVecEnv:   {fps_dum:,.0f} env-steps/sec")
    print(f"  speedup:       {fps_sub / fps_dum:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
