"""Microbenchmark: per-step .item() syncs vs batched end-of-rollout drain.

Mimics the previous collect_rollouts diagnostic path (.item() per state
norm × 12 keys × N steps) vs the new batched approach (stack on GPU,
single .cpu() at end). Doesn't need MemPPO — just a DTHLMU cell.
"""
from __future__ import annotations

import time

import torch

from memrl.cell import DTHLMU
from memrl.utils import apply_perf_defaults


def main() -> None:
    apply_perf_defaults()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cell = DTHLMU(input_size=8, hidden_size=64, hebbian_mode="gated_delta_eps").to(device)
    s = cell.init_state(8, device=device)
    x = torch.randn(8, 8, device=device)

    N = 1024  # one rollout's worth of timesteps

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            h, s, _ = cell.step(x, s)

    # OLD: per-step .item() syncs (12 per step)
    s_old = cell.init_state(8, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    old_norms = []
    with torch.no_grad():
        for _ in range(N):
            h, s_old, _ = cell.step(x, s_old)
            old_norms.append({k: v.norm().item() for k, v in s_old.items()})  # 12 syncs
    torch.cuda.synchronize()
    dt_old = time.perf_counter() - t0

    # NEW: stack on GPU, drain once at end
    s_new = cell.init_state(8, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rows = []
    keys = None
    with torch.no_grad():
        for _ in range(N):
            h, s_new, _ = cell.step(x, s_new)
            if keys is None:
                keys = tuple(s_new.keys())
            rows.append(torch.stack([s_new[k].norm() for k in keys]))
    norms_arr = torch.stack(rows).cpu().numpy()
    torch.cuda.synchronize()
    dt_new = time.perf_counter() - t0

    print(f"N={N} steps, batch=8, cell=DTHLMU (gated_delta_eps)")
    print(f"  OLD (per-step .item()): {dt_old*1000:7.1f} ms  "
          f"=> {N/dt_old:6.0f} steps/sec")
    print(f"  NEW (batched drain):    {dt_new*1000:7.1f} ms  "
          f"=> {N/dt_new:6.0f} steps/sec")
    print(f"  speedup: {dt_old/dt_new:.2f}x  "
          f"({(dt_old-dt_new)*1000:.1f} ms saved per rollout)")


if __name__ == "__main__":
    main()
