"""Base interface for recurrent memory cells.

Design contract (read this first):

* `init_state(batch_size, device, dtype)` — allocate a fresh state.
  Also used by the rollout buffer to discover state shapes by calling with
  batch_size=1 and reading `.shape[1:]` off each component. `device` defaults
  to None; subclasses MUST call `self._resolve_device(device)` first thing
  to infer the cell's parameter device when None is passed.

* `step(x, state, episode_start)` — single timestep. The cell is responsible
  for resetting state where `episode_start` is True BEFORE consuming x.
  Cells with zero initial state should call `apply_episode_mask` (cheap,
  no allocation). Cells with learnable / non-zero initial state should call
  `self.reset_state` instead (correctly uses `init_state` semantics).

* `reset_state(state, mask)` — out-of-step reset to initial values, used for
  diagnostics or manual env control. Has a default implementation that
  delegates to `init_state`; override for performance or custom semantics.

* `forward_sequence(x_seq, state_init, episode_starts)` — full-sequence
  training pass. Default delegates to `run_sequence` (step-by-step loop).
  Cells with parallel-scan training override this for speed. The numerical
  equivalence between an override and `run_sequence` is the test contract.

* `detach_state(state)` — free helper to detach state at TBPTT chunk
  boundaries so backprop doesn't flow across chunks.

The dict-state convention lets the rollout buffer iterate components
generically: adding a new state component to a cell doesn't break consumers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn


# Per-step cell state: dict from component name → tensor of shape (B, *).
#   LSTM:           {"h": (B, H), "c": (B, H)}
#   LMU:            {"h": (B, H), "m": (B, N)}
#   DeltaNet:       {"h": (B, H), "W": (B, n_heads, d_k, d_v), "n": (B, n_heads, d_k)}
State = dict[str, Tensor]

# Per-step diagnostic / auxiliary outputs from `step` (e.g. {"innovation": δ_t}),
# each tensor (B, *).
# Contract: a given cell instance MUST emit the same set of keys at every
# timestep. `run_sequence` validates this and raises on violation.
SideOutputs = dict[str, Tensor]

# Sequence-stacked side outputs from `forward_sequence` / `run_sequence`,
# each tensor (T, B, *). Same Python type as `SideOutputs`; the alias
# disambiguates per-step vs sequence-stacked in return signatures.
StackedSideOutputs = dict[str, Tensor]


class RecurrentCell(nn.Module, ABC):
    """Abstract base for recurrent memory cells.

    Subclasses must implement `init_state` and `step`. They may optionally
    override `reset_state` (for non-zero initial state) and `forward_sequence`
    (for parallel-scan training).

    Output dimension contract:
        `step` returns y of shape (B, output_size). If the cell's natural
        internal feature width differs from output_size (e.g., LMU with
        hidden_size != output_size, or matrix-memory cells with d_v*n_heads !=
        output_size), the cell is responsible for adding its own output
        projection — typically an `nn.Linear(internal_dim, output_size)` in
        `__init__` and applied at the end of `step`. Consumer code (policy,
        buffer, tests) can rely on (B, output_size) without knowing the
        cell's internals.
    """

    input_size: int
    output_size: int

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    # --- required ----------------------------------------------------------

    @abstractmethod
    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        """Allocate a zero- (or learnable-) initialized state.

        Also used by the rollout buffer for shape discovery (batch_size=1,
        read `.shape[1:]` off each component).

        Args:
            batch_size: number of parallel sequences
            device:     target device; if None, inferred from the cell's
                        parameters via `self._resolve_device`. Subclasses
                        MUST call `device = self._resolve_device(device)`
                        before using it.
            dtype:      target dtype
        """
        ...

    @abstractmethod
    def step(
        self,
        x: Tensor,
        state: State,
        episode_start: Optional[Tensor] = None,
    ) -> tuple[Tensor, State, SideOutputs]:
        """Single-step recurrent forward.

        Args:
            x:              (B, input_size)
            state:          dict matching `init_state`, each value (B, *)
            episode_start:  (B,) bool. True at episode boundaries; the cell
                            resets state for those samples BEFORE consuming x.
                            None ⇒ no reset.

        Returns:
            y:          (B, output_size)
            new_state:  dict with same shapes as `state`
            side:       diagnostics; may be empty {}. The set of keys MUST
                        be the same at every step for a given cell instance.
        """
        ...

    # --- overridable with defaults ----------------------------------------

    def _resolve_device(self, device: Optional[torch.device]) -> torch.device:
        """Return `device` if given, else infer from the cell's parameters.

        Falls back to CPU for cells with no parameters (degenerate case).
        """
        if device is not None:
            return device
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def reset_state(self, state: State, mask: Tensor) -> State:
        """Reset state to `init_state` values where mask is True.

        Default uses `self.init_state` to construct fresh values and selects
        between fresh and current via `torch.where`. Cells with zero initial
        state can ignore this method (the cheaper `apply_episode_mask` free
        function gives the same result). Cells with learnable initial state
        should rely on this default, since it correctly respects `init_state`.

        Override for performance if you have a cheaper reset (e.g., to avoid
        allocating a full-batch fresh state when only a few envs need reset).

        Args:
            state:  current state dict
            mask:   (B,) bool. True at positions to reset.
        """
        if not mask.any():
            return state

        any_t = next(iter(state.values()))
        batch_size = any_t.shape[0]
        fresh = self.init_state(batch_size, any_t.device, dtype=any_t.dtype)

        out: State = {}
        for k, t in state.items():
            m = mask
            while m.dim() < t.dim():
                m = m.unsqueeze(-1)
            out[k] = torch.where(m, fresh[k], t)
        return out

    def forward_sequence(
        self,
        x_seq: Tensor,
        state_init: State,
        episode_starts: Optional[Tensor] = None,
    ) -> tuple[Tensor, State, StackedSideOutputs]:
        """Run the cell over a length-T sequence.

        Default delegates to `run_sequence`, which iterates `step` in Python.
        Cells with a parallel-scan training kernel should override this for
        speed.

        The override contract: `forward_sequence` MUST be numerically
        equivalent to `run_sequence(self, ...)` on the same input, up to
        floating-point reduction-order differences. The determinism test
        suite compares them directly.

        Args:
            x_seq:           (T, B, input_size)
            state_init:      starting state, matching `init_state`'s shapes
            episode_starts:  optional (T, B) bool

        Returns:
            y_seq:        (T, B, output_size)
            final_state:  end-of-sequence state
            side_seq:     dict[str, (T, B, ...)] of stacked side outputs;
                          empty {} if the cell emits no side outputs
        """
        return run_sequence(self, x_seq, state_init, episode_starts)


# --- helpers ---------------------------------------------------------------


def apply_episode_mask(state: State, episode_start: Tensor) -> State:
    """Zero every tensor in `state` where `episode_start` is True.

    Cheap fast path for cells with zero initial state. No allocation when
    no resets are needed (the all-False case, which is ~99% of RL steps).

    Cells with non-zero initial state should call `self.reset_state` instead;
    this helper unconditionally zeros, which is incorrect for learnable init.

    Args:
        state:          current state dict
        episode_start:  (B,) bool. True at positions to zero.
    """
    if not episode_start.any():
        return state

    out: State = {}
    for k, t in state.items():
        m = episode_start
        while m.dim() < t.dim():
            m = m.unsqueeze(-1)
        out[k] = torch.where(m, torch.zeros_like(t), t)
    return out


def detach_state(state: State) -> State:
    """Detach every tensor in `state` from the autograd graph.

    Call at TBPTT chunk boundaries to prevent backprop from flowing across
    chunks (which would either OOM or, worse, silently produce wrong
    gradients for the bootstrap-value pathway).

    Functional: returns a new dict; does not mutate the input.
    """
    return {k: t.detach() for k, t in state.items()}


def run_sequence(
    cell: RecurrentCell,
    x_seq: Tensor,
    state_init: State,
    episode_starts: Optional[Tensor] = None,
) -> tuple[Tensor, State, StackedSideOutputs]:
    """Iterate `cell.step` over T timesteps. Reference implementation.

    Used as ground truth for determinism tests against parallel-scan
    overrides of `forward_sequence`. Validates that the cell's side-output
    keys are stable across the sequence; raises if a cell emits inconsistent
    keys (which would otherwise be silently dropped by the stacking logic).

    Args:
        x_seq:           (T, B, input_size)
        state_init:      starting state
        episode_starts:  optional (T, B) bool

    Returns:
        y_seq:        (T, B, output_size)
        final_state:  end-of-sequence state
        side_seq:     dict[str, (T, B, ...)] of stacked side outputs;
                      empty {} if the cell emits no side outputs
    """
    T = x_seq.shape[0]
    state = state_init
    ys: list[Tensor] = []
    sides_per_t: list[SideOutputs] = []

    for t in range(T):
        es_t = episode_starts[t] if episode_starts is not None else None
        y_t, state, side_t = cell.step(x_seq[t], state, es_t)
        ys.append(y_t)
        sides_per_t.append(side_t)

    y_seq = torch.stack(ys, dim=0)

    side_seq: StackedSideOutputs = {}
    if sides_per_t and sides_per_t[0]:
        expected_keys = set(sides_per_t[0].keys())
        for t, s in enumerate(sides_per_t):
            if set(s.keys()) != expected_keys:
                raise RuntimeError(
                    "Side-output keys must be stable across timesteps for a "
                    f"given cell instance. Step 0 emitted {expected_keys}, "
                    f"step {t} emitted {set(s.keys())}. Cells that conditionally "
                    "emit diagnostics should emit a sentinel (e.g. zeros or NaN) "
                    "rather than omitting the key."
                )
        for k in expected_keys:
            side_seq[k] = torch.stack([s[k] for s in sides_per_t], dim=0)

    return y_seq, state, side_seq
