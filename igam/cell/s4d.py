"""S4D baseline — Gu, Gupta, Goel, Re 2022 (NeurIPS).

"On the Parameterization and Initialization of Diagonal State Space Models"
https://arxiv.org/abs/2206.11893

S4D is the diagonal variant of S4 (Gu, Goel, Re 2021). It achieves comparable
performance to full S4 with dramatically simpler parameterization: the state
matrix A is constrained to be diagonal complex, so the recurrence factorizes
into N independent scalar SSMs per channel. This is the practical baseline
that almost everyone uses today — the original full S4 (DPLR form) is rarely
implemented directly anymore.

Why S4D for the IGAM ablation table:
  - Represents the "structured state-space" family in the Phase A baseline
    comparison (vs Linear Transformer / DeltaNet / Gated DeltaNet).
  - Uses HiPPO-LegS-derived initialization (S4D-Lin), which is the principled
    answer to the "scaled Legendre" question raised when we picked LegT for
    the LMU baseline. LMU = LegT (sliding window); S4D-Lin ≈ LegS (entire
    history, scale-adapting).
  - Diagonal complex state with conjugate-pair structure is the simplest
    representation of a stable LTI system with oscillatory modes.

Math (per step, per channel h, per state-dim n):
    Continuous SSM:    ẋ_h_n(t) = A_h_n x_h_n(t) + u_h(t)       # B fixed to 1
                       y_h(t)   = 2 · Re(Σ_n C_h_n x_h_n(t)) + D_h u_h(t)
    ZOH discretization at step size dt_h:
                       A_d  =  exp(dt · A)                       # (H, N/2) complex
                       B_d  =  (A_d − 1) / A                      # (H, N/2) complex
                       x_t  =  A_d ⊙ x_{t−1}  +  B_d · u_t       # per-channel diagonal
                       y_t  =  2 · Re(Σ_n C · x_t)  +  D · u_t   # per-channel readout

Parameterization (Gu 2022 §3):
    A  =  −exp(log_A_real)  +  i · A_imag        # ensures Re(A) < 0 ⇒ stable
    log_dt  ∈  ℝ^H                                # log-space for dt > 0
    A is stored via (log_A_real, A_imag), both ℝ^{H × N/2}
    C is stored as ℝ^{H × N/2 × 2} (view_as_real of complex tensor)
    D is real ℝ^H
    Only N/2 complex parameters (= N real) per channel; the other "half"
    of state is implicit via the conjugate-pair structure (hence the factor 2
    in the readout, which sums the real part of the conjugate-symmetric sum).

Initialization:
    log_A_real  =  log(0.5)             # Re(A) = −0.5 at init (Gu 2022)
    A_imag  via `init`:
        's4d-lin' (default, recommended): A_imag = π · [0, 1, ..., N/2−1]
            Approximation of S4-FouT; imaginary parts are evenly-spaced
            Fourier frequencies. Best empirical performance per Gu 2022 §5.
        's4d-inv': A_imag = (π/2)·(N/(2k+1) − 1) for k = 0..N/2−1
            HiPPO-LegS-N inspired; emphasizes lower frequencies. Slightly
            worse on most tasks but more interpretable.
    C: complex normal, real and imag parts iid N(0, 1).
    D: real normal N(0, 1) — per-channel scalar skip connection.
    log_dt: uniform in [log(dt_min), log(dt_max)]; defaults dt_min=0.001,
            dt_max=0.1 (S4 paper). The range matters: it sets the timescales
            the cell can resolve.

Optimizer note (Gu 2021, 2022):
    SSM parameters (log_dt, log_A_real, A_imag) benefit from:
      - Lower learning rate (typically 1e-3 vs 1e-2 for other params)
      - Zero weight decay (WD pushes them toward 0, destabilizing dynamics)
    The standard S4 codebase handles this via a per-parameter `_optim`
    attribute consumed by a custom optimizer factory. We do NOT replicate
    that here — manage at the optimizer-config level in train code if
    needed. For Phase A POPGym tasks with small models, defaults usually
    work; the per-param LR matters more at scale.

Precision note:
    Internal compute uses complex64 (cfloat). State is stored as float32
    via view_as_real and viewed as complex64 when needed. float64 (cdouble)
    is not supported end-to-end — init_state(dtype=float64) returns a
    float64 real representation, but the step kernel will fail or produce
    silently-cast complex64 results. Stick to float32 in practice.

Deliberate omissions:
  - No FFT convolutional view of `forward_sequence`. The base class loop
    runs `step` T times, per ADR 0001's single-step-recurrent commitment.
    The convolutional kernel `K[t] = 2·Re(Σ_n C·exp(dt·A)^t · (exp(dt·A)−1)/A)`
    is well-defined; a future `forward_sequence` override could use it for
    parallel training.
  - No GLU output mixing (the official S4D block has Conv1d + GLU + GELU).
    These belong to the full S4D *block* (analogous to a transformer block),
    not the SSM cell itself. We keep just Linear + LN to mirror the cell-
    interface design used by LSTM/GRU/LMU/LinearTransformer.
  - No dropout, no bidirectionality.

State:        {"x": (B, hidden_size, d_state // 2, 2)}     # complex state, real-rep
Side outputs: {}
Output:       (B, hidden_size)
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from igam.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask


S4DInit = Literal["s4d-lin", "s4d-inv"]


class S4D(RecurrentCell):
    """Diagonal State Space Model (S4D) in single-step recurrent form."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        d_state: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        init: S4DInit = "s4d-lin",
    ) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)
        if d_state % 2 != 0:
            raise ValueError(f"d_state ({d_state}) must be even (conjugate-pair structure)")
        if not 0.0 < dt_min < dt_max:
            raise ValueError(f"need 0 < dt_min < dt_max, got dt_min={dt_min}, dt_max={dt_max}")

        self.hidden_size = hidden_size
        self.d_state = d_state
        self.n_half = d_state // 2
        H, N_half = hidden_size, self.n_half

        # Input projection: encoder dim → H per-channel scalar inputs.
        # bias=False keeps the SSM input zero when x=0.
        self.input_linear = nn.Linear(input_size, H, bias=False)

        # Per-channel log step size, init uniform in [log(dt_min), log(dt_max)].
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # A parameterized as A = -exp(log_A_real) + 1j * A_imag, both (H, N/2).
        # Re(A) is negative by construction ⇒ continuous system is stable.
        self.log_A_real = nn.Parameter(math.log(0.5) * torch.ones(H, N_half))

        # Imaginary part of A — the initialization that defines S4D-Lin vs S4D-Inv.
        k = torch.arange(N_half, dtype=torch.float32)
        if init == "s4d-lin":
            # S4D-Lin: A_imag = π · k. Even Fourier-frequency spacing — best
            # empirical performance per Gu 2022 §5.
            a_imag_init = math.pi * k
        elif init == "s4d-inv":
            # S4D-Inv: A_imag = (π/2) · (N/(2k+1) − 1). HiPPO-LegS-inspired,
            # lower frequencies emphasized.
            a_imag_init = (math.pi / 2.0) * (d_state / (2.0 * k + 1.0) - 1.0)
        else:
            raise ValueError(f"init must be 's4d-lin' or 's4d-inv'; got {init!r}")
        # All channels share the same A_imag init; they diverge through training.
        A_imag = a_imag_init.unsqueeze(0).expand(H, -1).contiguous()
        self.A_imag = nn.Parameter(A_imag)

        # C: complex Gaussian, stored as real (H, N/2, 2) via view_as_real.
        # Use view_as_complex at forward time. The std of cfloat-randn is √2
        # (one each from real and imag), matching the reference impl.
        C_complex = torch.randn(H, N_half, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C_complex))

        # D: per-channel real skip connection.
        self.D = nn.Parameter(torch.randn(H))

        # Output mixing — simple linear + LN. Keeps the cell consistent with
        # the rest of the IGAM cell family. No GLU / GELU here (those belong
        # to the full S4D block; we're the SSM unit).
        self.output_linear = nn.Linear(H, H, bias=False)
        self.output_ln = nn.LayerNorm(H)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Input and output projections: Xavier uniform — calibrated for the
        # downstream LN.
        nn.init.xavier_uniform_(self.input_linear.weight)
        nn.init.xavier_uniform_(self.output_linear.weight)
        # SSM parameters (log_dt, log_A_real, A_imag, C, D) keep their
        # construction-time inits — those ARE the principled S4D defaults
        # per Gu 2022 and re-initializing here would clobber them.

    # --- interface ---------------------------------------------------------

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        device = self._resolve_device(device)
        # Complex state stored as real (B, H, N/2, 2). view_as_complex
        # interprets the last dim as [real, imag].
        return {
            "x": torch.zeros(
                batch_size, self.hidden_size, self.n_half, 2,
                device=device, dtype=dtype,
            ),
        }

    def step(
        self,
        x: Tensor,
        state: State,
        episode_start: Optional[Tensor] = None,
    ) -> tuple[Tensor, State, SideOutputs]:
        if episode_start is not None:
            state = apply_episode_mask(state, episode_start)

        # Project to H per-channel scalar inputs.
        u = self.input_linear(x)                                        # (B, H) real

        # Materialize complex parameters.
        dt = torch.exp(self.log_dt)                                     # (H,) real
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag              # (H, N/2) cfloat
        dtA = A * dt.unsqueeze(-1)                                      # (H, N/2)
        A_d = torch.exp(dtA)                                            # (H, N/2)
        # B_d = (A_d - 1) / A — implicit B = 1 vector per the S4D convention.
        B_d = (A_d - 1.0) / A                                           # (H, N/2)
        C = torch.view_as_complex(self.C)                               # (H, N/2)

        # State as complex via view_as_real ↔ view_as_complex round trip.
        x_state = torch.view_as_complex(state["x"])                     # (B, H, N/2) cfloat

        # Recurrent step: per-channel diagonal recurrence.
        # u is real (B, H); promote to (B, H, 1) cfloat so it broadcasts
        # against (H, N/2) complex parameters into (B, H, N/2) complex.
        u_complex = u.to(A_d.dtype).unsqueeze(-1)                       # (B, H, 1)
        x_new = A_d.unsqueeze(0) * x_state + B_d.unsqueeze(0) * u_complex  # (B, H, N/2)

        # Readout: y = 2 · Re(Σ_n C · x) + D · u. The factor 2 captures the
        # conjugate-symmetric contribution from the implicit other N/2 modes.
        y_ssm = 2.0 * (C.unsqueeze(0) * x_new).real.sum(dim=-1)         # (B, H)
        y = y_ssm + self.D * u                                          # (B, H) — skip

        # Output mixing.
        y_out = self.output_ln(self.output_linear(y))                   # (B, hidden_size)

        # Pack complex state back into real (B, H, N/2, 2) for storage.
        new_state: State = {"x": torch.view_as_real(x_new)}

        return y_out, new_state, {}
