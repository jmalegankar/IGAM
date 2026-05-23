"""Open latest.pt from the smoke run and verify all expected keys are present
and shapes look right. Doesn't load into a model — purely a structural check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def main(path: str) -> int:
    p = Path(path)
    if not p.exists():
        print(f"missing: {p}")
        return 1
    state = torch.load(p, map_location="cpu", weights_only=False)

    expected = {
        "policy_state", "optimizer_state", "num_timesteps", "_n_updates",
        "_last_obs", "_last_episode_starts", "_cell_state",
        "torch_rng", "torch_cuda_rng", "numpy_rng", "py_rng",
    }
    missing = expected - set(state)
    extra   = set(state) - expected
    print(f"file: {p}  size: {p.stat().st_size/1024:.1f} KiB")
    print(f"missing keys: {missing or 'none'}")
    print(f"extra keys:   {extra or 'none'}")
    print(f"num_timesteps: {state['num_timesteps']:,}")
    print(f"_n_updates:    {state['_n_updates']}")
    print(f"_last_obs shape: "
          f"{getattr(state['_last_obs'], 'shape', type(state['_last_obs']).__name__)}")
    print(f"_last_episode_starts: {state['_last_episode_starts']}")
    print(f"_cell_state keys ({len(state['_cell_state'])}):")
    for k, v in state["_cell_state"].items():
        print(f"  {k:24s} shape={tuple(v.shape)} dtype={v.dtype}")
    print(f"policy_state #params: "
          f"{sum(v.numel() for v in state['policy_state'].values()):,}")
    print(f"optimizer_state keys: {list(state['optimizer_state'].keys())}")
    print(f"torch_cuda_rng: "
          f"{'<saved>' if state['torch_cuda_rng'] is not None else 'None'}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
