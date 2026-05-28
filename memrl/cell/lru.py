"""Linear Recurrent Unit (LRU) — Orvieto et al. 2023 (ICML).

"Resurrecting Recurrent Neural Networks for Long Sequences"
https://arxiv.org/abs/2303.06349

A deep-linear diagonal complex recurrence, engineered to match S4/S5 on
long-range tasks while being conceptually a plain RNN (no HiPPO, no
discretization of a continuous SSM — the recurrence is defined directly in
discrete time). Three ingredients make it work:

  1. **Diagonal complex state.**  h_t = Λ h_{t-1} + B x_t, Λ = diag(λ) ∈ ℂ^N.
     Each state dim is an independent scalar complex recurrence.

  2. **Stable exponential parameterization.**  λ_j = exp(−exp(ν_j) + i·exp(θ_j)).
     The modulus is |λ_j| = exp(−exp(ν_j)) ∈ (0, 1) for any real ν_j, so the
     recurrence is stable by construction — no projection / clipping needed.
     ν is the log of the (negative) log-modulus; θ_log is the log-phase.

  3. **Ring initialization + normalization.**  Eigenvalues are initialized in an
     annulus r_min ≤ |λ| ≤ r_max on the complex plane (phases in [0, max_phase]).
     For long memory you push r_max → 1. The input is scaled by
     γ = sqrt(1 − |λ|²) so the hidden state has unit variance at init regardless
     of where on the ring λ sits (without this, near-unit-circle λ blow the
     state up).

Math (per step):
    λ      = exp(−exp(ν) + i·exp(θ_log))         # (N,) complex, |λ| ∈ (0,1)
    γ      = exp(γ_log) = sqrt(1 − |λ|²)          # (N,) real, input normalizer
    h_t    = λ ⊙ h_{t-1} + γ ⊙ (B x_t)           # (B, N) complex
    y_t    = Re(C h_t) + D ⊙ x_t                  # (B, H) real
    y_out  = LN(W_o y_t)                           # (B, H)

where B ∈ ℂ^{N×H}, C ∈ ℂ^{H×N}, D ∈ ℝ^H.

Parameterization / storage (mirrors S4D's real-rep-of-complex convention):
    ν       ℝ^N            (nu_log; modulus param)
    θ_log   ℝ^N            (theta_log; phase param)
    γ_log   ℝ^N            (input normalizer, log-space)
    B       ℝ^{N×H×2}      (view_as_real of complex)
    C       ℝ^{H×N×2}      (view_as_real of complex)
    D       ℝ^H

Initialization (Orvieto 2023 Appendix):
    u1, u2 ~ U(0,1)^N
    ν      = log(−½ · log(u1·(r_max² − r_min²) + r_min²))
    θ_log  = log(max_phase · u2)
    γ_log  = ½ · log(1 − exp(−exp(ν))²)
    B, C   ~ complex Glorot: re,im iid N(0, 1/(2·fan_in))
    D      ~ N(0, 1)

Defaults r_min=0, r_max=1, max_phase=2π are the paper's. For RL memory tasks
you typically want r_max close to 1 (long memory) — expose as a constructor arg.

Deliberate omissions (match the codebase's cell-vs-block split, see s4d.py):
  - No parallel scan in forward_sequence; the base loop runs `step` T times.
    The recurrence is associative, so an associative-scan override is the
    natural future speedup (Orvieto §3.3).
  - No block-level GLU/MLP/skip; the LRU *block* in the paper wraps the
    recurrence with those. We keep recurrence + Linear + LN, as S4D does.

State:        {"h": (B, d_state, 2)}     # complex state, real-rep
Side outputs: {}
Output:       (B, hidden_size)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from memrl.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask


class LRU(RecurrentCell):
    """Linear Recurrent Unit in single-step recurrent form."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        d_state: Optional[int] = None,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 2.0 * math.pi,
    ) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)
        if not (0.0 <= r_min <= r_max <= 1.0):
            raise ValueError(f"need 0 ≤ r_min ≤ r_max ≤ 1, got ({r_min}, {r_max})")
        self.hidden_size = hidden_size
        self.d_state = d_state if d_state is not None else hidden_size
        H, N = hidden_size, self.d_state

        # Input projection (encoder dim → H), so B/C/D operate in H-space
        # regardless of input_size. Mirrors S4D's input_linear.
        self.input_linear = nn.Linear(input_size, H, bias=False)

        # ── Diagonal Λ parameters (modulus ν, phase θ_log) ───────────────────
        # |λ| = exp(−exp(ν)) ∈ [r_min, r_max]; phase = exp(θ_log) ∈ [0, max_phase].
        u1 = torch.rand(N)
        u2 = torch.rand(N)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        theta_log = torch.log(max_phase * u2)
        self.nu_log    = nn.Parameter(nu_log)
        self.theta_log = nn.Parameter(theta_log)

        # Input normalizer γ = sqrt(1 − |λ|²), in log-space.
        lam_mod = torch.exp(-torch.exp(nu_log))                # (N,)
        gamma_log = 0.5 * torch.log(1.0 - lam_mod**2 + 1e-8)
        self.gamma_log = nn.Parameter(gamma_log)

        # ── Complex projections B (N×H), C (H×N); real skip D (H) ────────────
        # Complex Glorot: re, im iid N(0, 1/(2·fan_in)).
        B = torch.randn(N, H, dtype=torch.cfloat) / math.sqrt(2.0 * H)
        C = torch.randn(H, N, dtype=torch.cfloat) / math.sqrt(2.0 * N)
        self.B = nn.Parameter(torch.view_as_real(B))           # (N, H, 2)
        self.C = nn.Parameter(torch.view_as_real(C))           # (H, N, 2)
        self.D = nn.Parameter(torch.randn(H))

        # ── Output mixing (cell-level; block GLU/MLP omitted) ────────────────
        self.output_linear = nn.Linear(H, H, bias=False)
        self.output_ln = nn.LayerNorm(H)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.xavier_uniform_(self.input_linear.weight)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        device = self._resolve_device(device)
        # Complex hidden stored as real (B, N, 2).
        return {
            "h": torch.zeros(batch_size, self.d_state, 2, device=device, dtype=dtype),
        }

    def step(
        self,
        x: Tensor,
        state: State,
        episode_start: Optional[Tensor] = None,
    ) -> tuple[Tensor, State, SideOutputs]:
        if episode_start is not None:
            state = apply_episode_mask(state, episode_start)

        # Materialize complex diagonal dynamics.
        lam = torch.exp(-torch.exp(self.nu_log)
                        + 1j * torch.exp(self.theta_log))      # (N,) cfloat
        gamma = torch.exp(self.gamma_log)                      # (N,) real
        B_c = torch.view_as_complex(self.B)                    # (N, H) cfloat
        C_c = torch.view_as_complex(self.C)                    # (H, N) cfloat

        h_prev = torch.view_as_complex(state["h"])             # (B, N) cfloat

        # Project encoder input to H channels, then mix into N-dim complex state.
        u = self.input_linear(x)                               # (B, H) real
        Bx = u.to(B_c.dtype) @ B_c.t()                         # (B, N) cfloat
        h_new = lam.unsqueeze(0) * h_prev + gamma.unsqueeze(0) * Bx   # (B, N)

        # Readout: y = Re(C h) + D⊙u.  h (B,N) @ C^T (N,H) = (B,H).
        y = (h_new @ C_c.t()).real + self.D * u                # (B, H) real
        y_out = self.output_ln(self.output_linear(y))          # (B, H)

        return y_out, {"h": torch.view_as_real(h_new)}, {}
