"""Multi-layer Gated DeltaNet — stacked single-layer cells.

The single-layer GatedDeltaNet failed to break out of POPGym's AR floors at
2M steps. Published DeltaNet language models use 4-12 layers; published RL
recurrent cells use 1; we test the middle ground (2-4) on POPGym.

Architecture (N layers):

    x_t ──► [Layer 0: GatedDeltaNet] ──► h⁽⁰⁾_t
              │
              ▼
            [Layer 1: GatedDeltaNet] ──► h⁽¹⁾_t
              │
              ▼
              ...
              ▼
            [Layer N-1: GatedDeltaNet] ──► h_t (cell output)

Each layer maintains its own M_i (associative memory) and h_i (hidden state).
The cell output is the deepest layer's h. Layer-i sees layer-(i-1)'s h as
input, so deeper layers refine the memory-conditioned representation of x_t.

Side outputs: each layer's eps_mem is exposed as `eps_mem_l{i}`, plus
`eps_mem` = mean across layers (back-compat with single-layer name so
intrinsic-reward modules keying off "eps_mem" still work).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from memrl.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask
from memrl.cell.gated_deltanet import GatedDeltaNet


class MultiLayerGatedDeltaNet(RecurrentCell):
    """N-layer stack of GatedDeltaNet cells.

    Args:
        input_size:  C  observation / encoder dim
        hidden_size: H  output dim (and inter-layer dim)
        n_layers:    L  number of stacked layers (default 2)
        assoc_size:  A  memory dim per layer
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int = 2,
        assoc_size: int = 64,
    ) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)
        if n_layers < 1:
            raise ValueError("n_layers must be ≥ 1")
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.assoc_size = assoc_size

        # Layer 0 takes input_size; layers 1..N-1 take hidden_size (the
        # previous layer's output).
        layers = []
        for i in range(n_layers):
            in_dim = input_size if i == 0 else hidden_size
            layers.append(GatedDeltaNet(
                input_size=in_dim,
                hidden_size=hidden_size,
                assoc_size=assoc_size,
            ))
        self.layers = nn.ModuleList(layers)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        """Flat dict {h_l0, M_l0, h_l1, M_l1, ...} so RecurrentCell's
        apply_episode_mask + rollout buffer work without per-cell special-casing."""
        dev = self._resolve_device(device)
        state: State = {}
        for i in range(self.n_layers):
            state[f"h_l{i}"] = torch.zeros(batch_size, self.hidden_size,
                                           device=dev, dtype=dtype)
            state[f"M_l{i}"] = torch.zeros(batch_size, self.assoc_size,
                                           self.assoc_size, device=dev, dtype=dtype)
        return state

    def step(
        self,
        x: Tensor,
        state: State,
        episode_start: Optional[Tensor] = None,
    ) -> tuple[Tensor, State, SideOutputs]:
        if episode_start is not None:
            state = apply_episode_mask(state, episode_start)

        new_state: State = {}
        side_combined: SideOutputs = {}
        eps_per_layer = []

        cur_input = x
        for i, layer in enumerate(self.layers):
            # Build a sub-state for this layer using the cell's expected keys.
            sub_state = {"h": state[f"h_l{i}"], "M": state[f"M_l{i}"]}
            h_i, sub_new_state, sub_side = layer.step(cur_input, sub_state)
            new_state[f"h_l{i}"] = sub_new_state["h"]
            new_state[f"M_l{i}"] = sub_new_state["M"]
            if "eps_mem" in sub_side:
                side_combined[f"eps_mem_l{i}"] = sub_side["eps_mem"]
                eps_per_layer.append(sub_side["eps_mem"])
            cur_input = h_i

        # Back-compat: provide a top-level `eps_mem` (mean across layers)
        # so intrinsic-reward modules keying off "eps_mem" still work.
        if eps_per_layer:
            side_combined["eps_mem"] = torch.stack(eps_per_layer, dim=0).mean(dim=0)

        # The cell's output is the deepest layer's hidden state.
        return cur_input, new_state, side_combined
