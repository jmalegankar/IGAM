"""Shared pytest fixtures for cell tests.

Provides:
- `device` — CPU by default; override via env var `IGAM_TEST_DEVICE=cuda`.
- `seed_all` — autouse fixture that seeds torch/random/numpy deterministically.
- `cell_name_and_factory` — parametrized fixture yielding (name, factory)
  pairs for every cell registered in `CELL_FACTORIES`. New cells slot in here.

The factory takes `(input_size, hidden_size)` and returns a constructed cell
with test-friendly defaults for any extra hyperparameters.
"""

from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np
import pytest
import torch

from igam.cell import (
    GRU,
    LMU,
    LSTM,
    SHM,
    DeltaNet,
    GatedDeltaNet,
    LinearTransformer,
    Mamba2,
    RecurrentCell,
    RetNet,
    S4D,
    mLSTM,
)


# Test-friendly defaults for cells that require extra hyperparameters.
# Keep n_heads / d_state / memory_size small so the test suite stays fast.
CELL_FACTORIES: dict[str, Callable[[int, int], RecurrentCell]] = {
    "GRU":               lambda i, h: GRU(i, h),
    "LSTM":              lambda i, h: LSTM(i, h),
    "LMU":               lambda i, h: LMU(i, h, memory_size=8, theta=64.0),
    "LinearTransformer": lambda i, h: LinearTransformer(i, h, n_heads=2),
    "S4D":               lambda i, h: S4D(i, h, d_state=16),
    "Mamba2":            lambda i, h: Mamba2(i, h, n_heads=2, d_state=16),
    "DeltaNet":          lambda i, h: DeltaNet(i, h, n_heads=2),
    "RetNet":            lambda i, h: RetNet(i, h, n_heads=2),
    "mLSTM":             lambda i, h: mLSTM(i, h, n_heads=2),
    "GatedDeltaNet":     lambda i, h: GatedDeltaNet(i, h, n_heads=2),
    "SHM":               lambda i, h: SHM(i, h, L=32),  # smaller L for fast tests
}


@pytest.fixture
def device() -> torch.device:
    """Test device. CPU by default; override via IGAM_TEST_DEVICE env var."""
    env_device = os.environ.get("IGAM_TEST_DEVICE")
    if env_device:
        return torch.device(env_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def seed_all() -> None:
    """Auto-applied: seed torch / Python random / numpy / CUDA deterministically.

    Same seed every test ⇒ deterministic, reproducible failures.
    """
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(params=list(CELL_FACTORIES.keys()))
def cell_name_and_factory(request) -> tuple[str, Callable[[int, int], RecurrentCell]]:
    """Parametrize a test over every cell.

    Returns (name, factory) where factory(input_size, hidden_size) → cell.
    Tests using this fixture run once per registered cell.
    """
    name = request.param
    return name, CELL_FACTORIES[name]
