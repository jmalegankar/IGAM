"""Reference baselines for MQAR diagnosis.

Two cells are exposed:

  PureDeltaNetCell
      Minimal published-style DeltaNet — query is derived from the CURRENT
      input (q = W_q(x_t)), not from the previous hidden state. This is the
      design choice that makes DeltaNet able to answer "what is v(q_t)?" at
      the same timestep it sees q_t. Trains fast on MQAR (~5000 steps to 95%+
      on the easy config).

  HebbianOnlyDTHLMU(mode=...)
      Our DTH-LMU with the fast LMU, slow LMU, and LegS *output projections*
      zeroed and frozen (W_mf, W_ms, W_ml = 0). Isolates the Hebbian retrieval
      path. The aux branches still run internally (wasted compute) but their
      contribution to h_t is exactly zero, so the model relies purely on the
      Hebbian r_h term.

Use `run_baselines.py` to compare them against the full cell.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from memrl.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask
from memrl.cell import DTHLMU


# ---------------------------------------------------------------------------
# Pure DeltaNet — the published reference architecture, single-head
# ---------------------------------------------------------------------------

class PureDeltaNetCell(RecurrentCell):
    """Minimal Gated DeltaNet cell.

    Per-step update:
        k = ℓ2(W_k x_t)                                  # (B, A)
        v = W_v x_t                                       # (B, A)
        q = ℓ2(W_q x_t)                                   # (B, A)  ← from x_t, not h!
        α = σ(W_α x_t)         β = σ(W_β x_t)            # gates from x_t
        p = M_{t-1} k                                     # current prediction at k
        M_t = α·M_{t-1} + β·(v − p) k^T                   # gated delta
        r = LN(M_t · q)                                   # read at q
        h_t = W_o r                                       # output

    The query is computed from x_t at the same timestep — that's why a single
    layer of this can do MQAR. Our DTH-LMU reads with W_Q(h_{t-1}), which can't.

    No accumulating h beyond M and a thin output. Output_size = hidden_size.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        assoc_size: int = 64,
    ) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)
        self.hidden_size = hidden_size
        self.assoc_size  = assoc_size

        self.W_k = nn.Linear(input_size, assoc_size, bias=False)
        self.W_v = nn.Linear(input_size, assoc_size, bias=False)
        self.W_q = nn.Linear(input_size, assoc_size, bias=False)
        self.W_alpha = nn.Linear(input_size, 1, bias=True)
        self.W_beta  = nn.Linear(input_size, 1, bias=True)
        self.ln_r = nn.LayerNorm(assoc_size)
        self.W_o  = nn.Linear(assoc_size, hidden_size, bias=False)

        for w in (self.W_k, self.W_v, self.W_q, self.W_o):
            nn.init.xavier_normal_(w.weight)
        # α init → σ(4) ≈ 0.98 (preserve memory by default)
        nn.init.zeros_(self.W_alpha.weight)
        nn.init.constant_(self.W_alpha.bias, 4.0)
        # β init → σ(0) = 0.5 (moderate write)
        nn.init.xavier_normal_(self.W_beta.weight, gain=0.1)
        nn.init.zeros_(self.W_beta.bias)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        dev = self._resolve_device(device)
        return {
            "M": torch.zeros(batch_size, self.assoc_size, self.assoc_size,
                             device=dev, dtype=dtype),
        }

    def step(
        self,
        x: Tensor,
        state: State,
        episode_start: Optional[Tensor] = None,
    ) -> tuple[Tensor, State, SideOutputs]:
        if episode_start is not None:
            state = apply_episode_mask(state, episode_start)
        M = state["M"]

        k = F.normalize(self.W_k(x), dim=-1)              # (B, A)
        v = self.W_v(x)                                    # (B, A)
        q = F.normalize(self.W_q(x), dim=-1)              # (B, A)

        # Gated delta update
        p = torch.einsum("bkv,bk->bv", M, k)              # (B, A) prediction at k
        delta_v = v - p
        update  = k.unsqueeze(-1) * delta_v.unsqueeze(-2) # (B, A, A)
        alpha   = torch.sigmoid(self.W_alpha(x)).squeeze(-1)
        beta    = torch.sigmoid(self.W_beta(x)).squeeze(-1)
        M_new   = alpha.view(-1, 1, 1) * M + beta.view(-1, 1, 1) * update

        # Read at q
        r = torch.einsum("bkv,bk->bv", M_new, q)          # (B, A)
        h = self.W_o(self.ln_r(r))                        # (B, H)

        return h, {"M": M_new}, {}


# ---------------------------------------------------------------------------
# Hebbian-only DTH-LMU: full cell but aux output projections frozen at zero
# ---------------------------------------------------------------------------

def hebbian_only_dthlmu(
    input_size: int,
    hidden_size: int,
    hebbian_mode: str = "gated_delta_eps",
    **kwargs,
) -> DTHLMU:
    """Build a DTH-LMU and freeze W_mf, W_ms, W_ml at zero.

    Result: the fast LMU, slow LMU, and LegS branches still RUN internally
    (state still updates) but their contribution to the hidden update is zero.
    The cell relies entirely on:
        h_t = tanh(W_x x + W_h h_prev + r_h)        # Hebbian retrieval only
    """
    cell = DTHLMU(
        input_size=input_size,
        hidden_size=hidden_size,
        hebbian_mode=hebbian_mode,
        **kwargs,
    )
    with torch.no_grad():
        cell.W_mf.weight.zero_()
        cell.W_ms.weight.zero_()
        cell.W_ml.weight.zero_()
    for p in cell.W_mf.parameters():
        p.requires_grad = False
    for p in cell.W_ms.parameters():
        p.requires_grad = False
    for p in cell.W_ml.parameters():
        p.requires_grad = False
    return cell
