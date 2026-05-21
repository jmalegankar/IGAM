"""Dual-Timescale Hebbian LMU (DTH-LMU) — ADR 0002.

Four co-operating memory systems, each answering a distinct question:

  Fast LMU  — gated HiPPO-LegT, window θ, K multi-scale banks.
              Selective working memory: high-innovation inputs update
              strongly; predicted inputs barely register.

  Slow LMU  — bare HiPPO-LegT integrator, window θ_s = θ²/D.
              θ_s is derived, not free: the fast cell's resolution ratio
              ρ = θ/D gives θ_s = θ·ρ — consecutive rungs in a log-scale
              hierarchy. Uniform-weight statistics over the last θ²/D steps.
              Requires D < θ.

  LegS      — HiPPO-LegS integrator, no window, no gate.
              m(t) = m(t-1) + (1/t)·(−A·m(t-1) + B·u(t-1))
              A, B are fixed continuous matrices (no ZOH); step size 1/t
              is the only time-varying quantity. Degree-0 coefficient =
              exact running mean over the full episode.

  Hebbian M — outer-product associative matrix.
              Write: M ← M + φ(k)⊗v / A.  Read: M · φ(q).
              Content-addressable, O(A²) state, T-independent.

State:
    h      (B, H)        hidden state
    m_f    (B, K, D, C)  fast LegT coefficients  [K = n_scales]
    m_s    (B, 1, D, C)  slow LegT coefficients
    m_legs (B, D, C)     LegS coefficients
    t      (B, 1)        episode step counter (0-init; cell uses t+1)
    M      (B, A, A)     Hebbian matrix           [A = assoc_size]

Side outputs:
    innovation  (B, C)   fast-cell prediction error

Neuroscience grounding (Complementary Learning Systems, McClelland 1995):
    fast  ↔ prefrontal working memory
    slow  ↔ neocortex (medium-range statistics)
    LegS  ↔ entorhinal cortex (scale-invariant cumulative coding)
    M     ↔ hippocampus (one-shot episodic binding via LTP)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from memrl.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask
from memrl.cell.lmu import _legs_matrices, _legt_zoh_matrices


# ---------------------------------------------------------------------------

class _HebbianMemory(nn.Module):
    """Outer-product key-value associative memory.

    Write:  k = φ(W_K x),  v = W_V x
            M ← M + k⊗v / A
    Read:   q = φ(W_Q h)
            r = LN(M · q)  →  W_r  →  R^H

    φ = L2-normalise. Retrieval M·k_τ ≈ v_τ when keys are near-orthogonal;
    interference from T entries is O(T/√A), kept bounded by LayerNorm.
    """

    def __init__(self, input_size: int, hidden_size: int, assoc_size: int) -> None:
        super().__init__()
        self.assoc_size = assoc_size
        self.W_K = nn.Linear(input_size,  assoc_size, bias=False)
        self.W_V = nn.Linear(input_size,  assoc_size, bias=False)
        self.W_Q = nn.Linear(hidden_size, assoc_size, bias=False)
        self.ln_r = nn.LayerNorm(assoc_size)
        self.W_r  = nn.Linear(assoc_size, hidden_size, bias=False)
        for m in (self.W_K, self.W_V, self.W_Q, self.W_r):
            nn.init.xavier_normal_(m.weight)

    def write(self, x: Tensor, M: Tensor) -> Tensor:
        """M ← M + φ(W_K x) ⊗ (W_V x) / A.  Returns (B, A, A)."""
        k = F.normalize(self.W_K(x), dim=-1)                      # (B, A)
        v = self.W_V(x)                                            # (B, A)
        return M + k.unsqueeze(-1) * v.unsqueeze(-2) / self.assoc_size

    def read(self, h_normed: Tensor, M: Tensor) -> Tensor:
        """LN(M · φ(W_Q h)) → W_r.  Returns (B, H)."""
        q = F.normalize(self.W_Q(h_normed), dim=-1)                # (B, A)
        r = torch.einsum("bav,ba->bv", M, q)                       # (B, A)
        return self.W_r(self.ln_r(r))                              # (B, H)


# ---------------------------------------------------------------------------

class DTHLMU(RecurrentCell):
    """Dual-Timescale Hebbian LMU (ADR 0002).

    Args:
        input_size:         C  observation dimension
        hidden_size:        H  hidden / output dimension
        memory_size:        D  Legendre order for all three LMU cells
        theta:              θ  fast-cell window; slow window = θ²/D
        n_scales:           K  multi-scale banks in the fast cell
        scale_factor:       fast-cell θ spacing: [θ·sf^k for k in offsets]
        assoc_size:         A  Hebbian key/value dimension
        residual_scale:     ε  fast-cell anti-collapse weight
        readout_skip_scale: α  fast-cell u_x skip weight on readout
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int  = 32,
        theta: float      = 100.0,
        n_scales: int     = 3,
        scale_factor: float = 2.0,
        assoc_size: int   = 64,
        residual_scale: float      = 0.05,
        readout_skip_scale: float  = 0.1,
    ) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)

        if memory_size >= theta:
            raise ValueError(
                f"memory_size ({memory_size}) must be < theta ({theta}) "
                f"so the slow window θ²/D = {theta**2/memory_size:.1f} > θ."
            )

        self.hidden_size       = hidden_size
        self.memory_size       = memory_size
        self.n_scales          = n_scales
        self.residual_scale    = residual_scale
        self.readout_skip_scale = readout_skip_scale
        C = input_size

        # ── HiPPO matrices ──────────────────────────────────────────────────

        # Fast LegT: K banks at θ·sf^k for k in [-(K//2), ..., K - K//2 - 1]
        offsets  = range(-(n_scales // 2), n_scales - n_scales // 2)
        A_f, B_f = zip(*[_legt_zoh_matrices(memory_size, theta * scale_factor**k)
                         for k in offsets])
        self.register_buffer("A_f", torch.stack(A_f))   # (K, D, D)
        self.register_buffer("B_f", torch.stack(B_f))   # (K, D, 1)

        # Slow LegT: single bank at θ_s = θ²/D
        A_s, B_s = _legt_zoh_matrices(memory_size, theta**2 / memory_size)
        self.register_buffer("A_s", A_s.unsqueeze(0))   # (1, D, D)
        self.register_buffer("B_s", B_s.unsqueeze(0))   # (1, D, 1)

        # LegS: fixed continuous matrices, no ZOH; step size 1/t at runtime
        A_l, B_l = _legs_matrices(memory_size)
        self.register_buffer("A_legs", A_l)              # (D, D)
        self.register_buffer("B_legs", B_l)              # (D, 1)

        # ── Fast-cell parameters ────────────────────────────────────────────
        self.e_x    = nn.Parameter(torch.empty(C))
        self.E_h    = nn.Linear(hidden_size, C, bias=False)
        self.e_m_f  = nn.Parameter(torch.zeros(n_scales, memory_size))
        self.W_pre  = nn.Linear(C, C, bias=False)          # orthogonal post-init
        self.W_g    = nn.Linear(hidden_size, n_scales, bias=True)
        self.W_mix  = nn.Linear(hidden_size, n_scales, bias=True)
        self.W_qf   = nn.Linear(hidden_size, memory_size, bias=False)

        # ── Slow / LegS readout ─────────────────────────────────────────────
        self.W_qs   = nn.Linear(hidden_size, memory_size, bias=False)
        self.W_ql   = nn.Linear(hidden_size, memory_size, bias=False)

        # ── Hebbian memory ──────────────────────────────────────────────────
        self.hebbian = _HebbianMemory(input_size, hidden_size, assoc_size)

        # ── Hidden update ───────────────────────────────────────────────────
        self.ln_h = nn.LayerNorm(hidden_size)
        self.W_x  = nn.Linear(input_size,  hidden_size, bias=True)
        self.W_h  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_mf = nn.Linear(C, hidden_size, bias=False)
        self.W_ms = nn.Linear(C, hidden_size, bias=False)
        self.W_ml = nn.Linear(C, hidden_size, bias=False)

        self._reset_parameters()
        nn.utils.parametrizations.orthogonal(self.W_pre)

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.e_x, -1.0, 1.0)
        nn.init.xavier_normal_(self.E_h.weight)
        nn.init.eye_(self.W_pre.weight)   # identity; orthogonal param preserves it

        # Small-gain init: readout starts near-zero, cell learns where to point
        for W in (self.W_qf, self.W_qs, self.W_ql):
            nn.init.orthogonal_(W.weight, gain=0.01)

        # Uniform softmax at init; near-identity salience (sigmoid(8) ≈ 0.9997)
        nn.init.zeros_(self.W_mix.weight);  nn.init.zeros_(self.W_mix.bias)
        nn.init.zeros_(self.W_g.weight);    nn.init.constant_(self.W_g.bias, 8.0)

        for W in (self.W_x, self.W_h, self.W_mf, self.W_ms, self.W_ml):
            nn.init.xavier_normal_(W.weight)
        nn.init.zeros_(self.W_x.bias)

    # ── RecurrentCell interface ─────────────────────────────────────────────

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        dev = self._resolve_device(device)
        D, C, A = self.memory_size, self.input_size, self.hebbian.assoc_size
        z = dict(device=dev, dtype=dtype)
        return {
            "h":      torch.zeros(batch_size, self.hidden_size,    **z),
            "m_f":    torch.zeros(batch_size, self.n_scales, D, C, **z),
            "m_s":    torch.zeros(batch_size, 1,             D, C, **z),
            "m_legs": torch.zeros(batch_size,                D, C, **z),
            "t":      torch.zeros(batch_size, 1,                   **z),
            "M":      torch.zeros(batch_size, A,             A,    **z),
        }

    def step(
        self,
        x: Tensor,
        state: State,
        episode_start: Optional[Tensor] = None,
    ) -> tuple[Tensor, State, SideOutputs]:
        if episode_start is not None:
            state = apply_episode_mask(state, episode_start)

        h_prev  = state["h"]        # (B, H)
        m_f     = state["m_f"]      # (B, K, D, C)
        m_s     = state["m_s"]      # (B, 1, D, C)
        m_legs  = state["m_legs"]   # (B, D, C)
        t       = state["t"]        # (B, 1)
        M       = state["M"]        # (B, A, A)

        h_n  = self.ln_h(h_prev)                               # (B, H)  pre-normed
        u_x  = x * F.normalize(self.e_x, dim=0)               # (B, C)  shared encoder

        # ── Fast LMU ────────────────────────────────────────────────────────

        mix    = F.softmax(self.W_mix(h_n), dim=-1)            # (B, K)
        u_m    = torch.einsum(                                  # (B, C)
                     "bk,bkc->bc", mix,
                     torch.einsum("kd,bkdc->bkc", self.e_m_f, m_f))
        pred      = self.E_h(h_n) + u_m                        # (B, C)
        innov     = self._softsign(u_x) + self._softsign(pred - u_x)   # (B, C)
        u_act     = self.W_pre(innov) + pred + self.residual_scale * u_x.detach()

        Am_f   = torch.einsum("kij,bkjc->bkic", self.A_f, m_f)
        Am_f   = Am_f * torch.sigmoid(self.W_g(h_n)).unsqueeze(-1).unsqueeze(-1)
        m_f_   = Am_f + self.B_f.unsqueeze(0) * u_act.unsqueeze(1).unsqueeze(2)

        C_f    = F.normalize(self.W_qf(h_n), dim=-1)           # (B, D)
        y_f    = torch.einsum("bk,bkc->bc", mix,
                     torch.einsum("bd,bkdc->bkc", C_f, m_f_))  # (B, C)
        y_f    = y_f + self.readout_skip_scale * u_x

        # ── Slow LMU ────────────────────────────────────────────────────────

        m_s_   = (torch.einsum("kij,bkjc->bkic", self.A_s, m_s)
                  + self.B_s.unsqueeze(0) * u_x.unsqueeze(1).unsqueeze(2))

        C_s    = F.normalize(self.W_qs(h_n), dim=-1)           # (B, D)
        y_s    = torch.einsum("bd,bkdc->bkc", C_s, m_s_).squeeze(1)   # (B, C)

        # ── LegS ────────────────────────────────────────────────────────────

        t_      = t + 1.0                                       # (B, 1)  1-indexed
        dt      = (1.0 / t_).unsqueeze(-1)                      # (B, 1, 1)
        m_l_    = (m_legs
                   + dt * (-torch.einsum("ij,bjc->bic", self.A_legs, m_legs)
                           + self.B_legs.unsqueeze(0) * u_x.unsqueeze(1)))

        C_l    = F.normalize(self.W_ql(h_n), dim=-1)           # (B, D)
        y_l    = torch.einsum("bd,bdc->bc", C_l, m_l_)         # (B, C)

        # ── Hebbian M ───────────────────────────────────────────────────────

        M_     = self.hebbian.write(x, M)
        r_h    = self.hebbian.read(h_n, M_)                    # (B, H)

        # ── Hidden update ───────────────────────────────────────────────────

        h_ = torch.tanh(
            self.W_x(x) + self.W_h(h_prev)
            + self.W_mf(y_f) + self.W_ms(y_s) + self.W_ml(y_l)
            + r_h
        )                                                       # (B, H)

        return h_, {
            "h": h_, "m_f": m_f_, "m_s": m_s_,
            "m_legs": m_l_, "t": t_, "M": M_,
        }, {"innovation": innov}

    @staticmethod
    def _softsign(x: Tensor) -> Tensor:
        return x / (1.0 + x.abs())
