"""DEPRECATED shim → use `registry.py` (the E1–E14 single source of truth).

This file used to generate a 3-experiment subset (core grid + λ-sweep +
Autoencode). That has been superseded by the declarative registry, which defines
all of E1–E14 with wandb routing and dependency status.

    python -m experiments.memory_training.registry --list      # show all
    python -m experiments.memory_training.registry --emit E1    # emit one
    python -m experiments.memory_training.registry --emit-all   # emit all unblocked
"""
from __future__ import annotations

from .registry import main

if __name__ == "__main__":
    main()
