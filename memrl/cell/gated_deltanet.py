"""Gated DeltaNet — single-layer recurrent memory cell for RL.

Mechanism (per-step):

    k_t = ℓ₂(W_k x_t)         (B, A)   key, L2-normalised
    v_t = W_v x_t              (B, A)   value
    q_t = ℓ₂(W_q x_t)         (B, A)   query, from current input
    α_t = σ(W_α x_t)           (B,)     decay gate
    β_t = σ(W_β x_t)           (B,)     write strength

    M_t = α_t · (I − β_t k_t k_tᵀ) M_{t-1}  +  β_t v_t k_tᵀ
    r_t = M_t q_t              (B, A)   associative read

    h_t = tanh(W_x x_t + W_h h_{t-1} + W_r LN(r_t))

Why each piece:
  * **q from x_t** (not h_{t-1}): a token's own query against memory built from
    past (k_s, v_s) is the design that lets a single layer answer "what is
    v(q)?" at the same timestep it sees q. This is the key difference vs the
    DTH-LMU's `W_Q(h_{t-1})` read; without it, single-layer recurrent has the
    timing problem we measured (cell can't respond to a query at the timestep
    of the query itself).
  * **L2-normalised k, q**: makes `‖I − β k kᵀ‖₂ = max(1, |1 − β|) ≤ 1` for
    β ∈ (0, 1), giving the Widrow–Hoff stability the delta rule needs.
  * **Scalar α decay**: matches Yang et al. (Gated DeltaNet, ICLR 2025).
    Decays the WHOLE matrix (not channel-wise). Cheap, stable, sufficient.
  * **Hidden state h alongside M**: matches the RL convention of every memory
    cell in this codebase (LSTM, GRU, LMU, DTH-LMU). The cell's "output" is
    h; M is the auxiliary associative store. Policy/value heads read h.

References:
  Schlag, Irie, Schmidhuber 2021 — DeltaNet (linearised attention = delta rule)
  Yang, Wang, Zhang, Shen, Kim, NeurIPS 2024 — chunkwise parallel DeltaNet
  Yang, Kautz, Hatamizadeh, ICLR 2025 — Gated DeltaNet (this design)

State:
    h  (B, H)        hidden state for the policy
    M  (B, A, A)     associative memory (delta-rule-updated)

Side outputs:
    eps_mem  (B,)    ‖v − M_{t-1} k‖² — memory-conditioned prediction error,
                      exposed for diagnostics (Agency Principle ablation).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from memrl.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask


class GatedDeltaNet(RecurrentCell):
    """Single-layer Gated DeltaNet for RL.

    Args:
        input_size:   C  observation / encoder dim
        hidden_size:  H  hidden state dim (cell output)
        assoc_size:   A  associative memory dim (M is A×A)
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

        # K / V / Q projections from the current input.
        # The "q from x" choice (vs DTH-LMU's "q from h_prev") is the
        # design change that lets a single-layer recurrent cell answer
        # queries at the timestep of the query token.
        self.W_k = nn.Linear(input_size, assoc_size, bias=False)
        self.W_v = nn.Linear(input_size, assoc_size, bias=False)
        self.W_q = nn.Linear(input_size, assoc_size, bias=False)

        # Per-token scalar gates: α (decay), β (write strength).
        self.W_alpha = nn.Linear(input_size, 1, bias=True)
        self.W_beta  = nn.Linear(input_size, 1, bias=True)

        # LN + projection on the read, plus the standard hidden update.
        self.ln_r = nn.LayerNorm(assoc_size)
        self.W_r  = nn.Linear(assoc_size, hidden_size, bias=False)
        self.W_x  = nn.Linear(input_size,  hidden_size, bias=True)
        self.W_h  = nn.Linear(hidden_size, hidden_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for w in (self.W_k, self.W_v, self.W_q, self.W_r, self.W_h):
            nn.init.xavier_normal_(w.weight)
        nn.init.xavier_normal_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)
        # α init → σ(4) ≈ 0.98: preserve memory by default. Cell learns
        # when to forget; the prior is "keep".
        nn.init.zeros_(self.W_alpha.weight)
        nn.init.constant_(self.W_alpha.bias, 4.0)
        # β init → σ(0) = 0.5: moderate writes. Some signal from x_t but
        # not full overwrite; the cell learns when each write matters.
        nn.init.xavier_normal_(self.W_beta.weight, gain=0.1)
        nn.init.zeros_(self.W_beta.bias)

    # ── RecurrentCell interface ────────────────────────────────────────────

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        dev = self._resolve_device(device)
        return {
            "h": torch.zeros(batch_size, self.hidden_size, device=dev, dtype=dtype),
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

        h_prev = state["h"]                                       # (B, H)
        M      = state["M"]                                       # (B, A, A)

        # ── Projections ─────────────────────────────────────────────────────
        k = F.normalize(self.W_k(x), dim=-1)                       # (B, A)
        v = self.W_v(x)                                             # (B, A)
        q = F.normalize(self.W_q(x), dim=-1)                       # (B, A)

        # ── Prediction error (diagnostic side-output) ───────────────────────
        # ε_mem = ‖v − M_{t-1}·k‖² — what the matrix predicts for THIS key,
        # before the write. Used downstream by ablations testing whether
        # memory-conditioned curiosity helps (Agency Principle).
        Mk      = torch.einsum("bkv,bk->bv", M, k)                 # (B, A)
        eps_mem = (v - Mk).pow(2).mean(-1)                         # (B,)

        # ── Gated delta rule update ─────────────────────────────────────────
        # M_t = α · (I − β k kᵀ) M_{t-1}  +  β v kᵀ
        #
        # Convention: M[b, k_idx, v_idx]. So:
        #   write_term[b, k, v] = β · k[b, k] · v[b, v]
        #   erase_term[b, k, v] = β · k[b, k] · (kᵀ M)[b, v]   (because
        #     (k kᵀ M)[b, k, v] = k[b, k] · Σ_k' k[b, k'] M[b, k', v])
        alpha = torch.sigmoid(self.W_alpha(x)).squeeze(-1)         # (B,)
        beta  = torch.sigmoid(self.W_beta(x)).squeeze(-1)          # (B,)
        # kM[b, v] = Σ_k k[b, k] · M[b, k, v]
        kM = torch.einsum("bkv,bk->bv", M, k)                      # (B, A)
        erase_term = beta.view(-1, 1, 1) * (k.unsqueeze(-1) * kM.unsqueeze(-2))
        write_term = beta.view(-1, 1, 1) * (k.unsqueeze(-1) * v.unsqueeze(-2))
        M_new = alpha.view(-1, 1, 1) * (M - erase_term) + write_term  # (B, A, A)

        # ── Read at current query ──────────────────────────────────────────
        # r = M_t · q  with M[b, k_idx, v_idx], q on k_idx, output in v space
        r = torch.einsum("bkv,bk->bv", M_new, q)                   # (B, A)
        r = self.W_r(self.ln_r(r))                                 # (B, H)

        # ── Hidden update ──────────────────────────────────────────────────
        h_new = torch.tanh(self.W_x(x) + self.W_h(h_prev) + r)     # (B, H)

        return h_new, {"h": h_new, "M": M_new}, {"eps_mem": eps_mem}
