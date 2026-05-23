"""One-off sanity smoke test: all four Hebbian modes step cleanly on GPU.

Replicates section 3a of docs/handoff/dth_lmu_ablation_handoff.md but goes
through memrl.utils.apply_perf_defaults so we also verify the perf knobs
are tolerated by the cell.
"""
from __future__ import annotations

import torch

from memrl.cell import DTHLMU
from memrl.utils import apply_perf_defaults


def main() -> None:
    perf = apply_perf_defaults()
    print(f"perf: {perf}")
    print("CUDA:", torch.cuda.is_available(),
          "| device:",
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for mode in ("additive", "delta", "gated_delta", "gated_delta_eps"):
        cell = DTHLMU(input_size=8, hidden_size=64, hebbian_mode=mode).to(device)
        s = cell.init_state(4, device=device)
        for _ in range(20):
            h, s, side = cell.step(torch.randn(4, 8, device=device), s)
        eps = side["eps_mem"].mean().item()
        print(f"{mode:18s} h finite={h.isfinite().all().item()} eps_mem={eps:.4f}")


if __name__ == "__main__":
    main()
