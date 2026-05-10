"""Linear Transformer cell — Katharopoulos et al. 2020.

"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
https://arxiv.org/abs/2006.16236

The simplest matrix-memory linear-attention cell. Serves as the lower-bound
ablation for the Gated DeltaNet family in the Phase A baseline table:
strip away gating (α_t, β_t), strip away the delta rule, strip away the
dynamic query — what's left is outer-product accumulation with a kernel
feature map. If IGAM can't beat this, every component IGAM adds is
making the model strictly worse than the simplest matrix-memory cell.

Math (per step, per head h):
    q_t, k_t, v_t  =  W_q x_t, W_k x_t, W_v x_t                  # all (B, d_head)
    φ(x)  =  elu(x) + 1                                          # paper feature map
    S^h_t  =  S^h_{t-1}  +  φ(k_t)^T v_t                          # matrix state, (d_head, d_head)
    z^h_t  =  z^h_{t-1}  +  φ(k_t)                                # normalizer, (d_head,)
    num_t  =  φ(q_t)  @  S^h_t                                    # (d_head,)
    den_t  =  φ(q_t)  ·  z^h_t  +  ε                              # scalar
    y^h_t  =  num_t  /  den_t                                     # (d_head,)
    y_t   =  W_O  concat_h(y^h_t)                                 # (B, hidden_size)

Modifications over Katharopoulos 2020:
  - Pre-LN on input + Output-LN after W_O (modern transformer stability;
    LLaMA / RetNet practice)
  - Fused QKV projection (one matmul, chunked)
  - bias=False on linears; learnable shifts in LN's affine
  - Xavier init on all projections

Feature map choice (`elu(x) + 1`):
  Original paper default. Ensures non-negativity (kernel similarity is
  always ≥ 0), giving a well-defined unnormalized attention distribution.
  Alternatives discussed in the literature:
    - Identity: works surprisingly well per Schoelkopf 2024; no guarantee
      of non-negative similarity, normalizer can become negative
    - FAVOR+ / Performer-style random features (Choromanski 2021): closer
      to true softmax approximation, but adds randomness
    - 1 + elu vs softmax(K)/softmax(Q): Schlag 2021 "Linear Transformers
      Are Secretly Fast Weight Programmers" advocates explicit
      normalization
  We keep elu+1 to match the canonical baseline. Switch via subclassing
  `_feature_map` if needed.

Normalizer (`use_normalizer`):
  Default True (matches paper). Set False to drop the z_t state and the
  num/den division. The "drop normalizer" variant is documented in
  Schoelkopf 2024 as empirically stable and is the typical choice in
  modern linear-attention implementations (RetNet, GLA, Mamba lineage).
  We keep it on by default to remain a faithful Katharopoulos baseline;
  flip it off for the "simpler still" ablation row.

Causal masking and update order:
  The state is updated with (k_t, v_t) BEFORE the read with q_t, so the
  output y_t sees the diagonal (position t's own key/value), matching
  standard causal self-attention. Episode resets via `apply_episode_mask`
  happen first, so y_t after an episode boundary depends only on x_t.

Deliberate omissions:
  - No parallel-scan / chunkwise training kernel — per ADR 0001 Phase A
    uses single-step recurrent only. The chunkwise form (Hua 2022, Yang
    2024) is a future `forward_sequence` override.
  - No positional encoding — the recurrent state is inherently position-
    aware via accumulation order. RoPE / ALiBi are LM-domain tricks; for
    RL the time-step itself is the position.
  - No dropout / attention dropout — same rationale as GRU/LSTM cells.

State:        {"S": (B, n_heads, d_head, d_head),
               "z": (B, n_heads, d_head)}              # present iff use_normalizer
Side outputs: {}                                       # no innovation analog
Output:       (B, hidden_size)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from igam.cell.base import RecurrentCell, SideOutputs, State, apply_episode_mask


class LinearTransformer(RecurrentCell):
    """Katharopoulos 2020 Linear Transformer in single-step recurrent form."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_heads: int = 4,
        use_normalizer: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(input_size=input_size, output_size=hidden_size)
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
            )
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = hidden_size // n_heads
        self.use_normalizer = use_normalizer
        self.eps = eps

        # Pre-LN over the input embedding — standard pre-norm transformer.
        self.input_ln = nn.LayerNorm(input_size)

        # Fused QKV projection: one matmul, chunked into (q | k | v) along the
        # last axis. bias=False — shifts come from LN affines.
        self.qkv_linear = nn.Linear(input_size, 3 * hidden_size, bias=False)

        # Output projection — concat-of-heads (hidden_size) back to hidden_size.
        # In a full transformer block this is followed by residual + FFN; we
        # don't have those here, so out_linear acts as the cell's read head.
        self.out_linear = nn.Linear(hidden_size, hidden_size, bias=False)

        # Post-LN on the cell output. RetNet / LLaMA-style stability: keeps
        # downstream activations well-conditioned even when S, z grow large
        # over long episodes.
        self.output_ln = nn.LayerNorm(hidden_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Xavier uniform for all projections, matching the Katharopoulos
        # reference implementation. No explicit 1/sqrt(d_head) scaling — the
        # original paper doesn't use it (no softmax to temper), and Xavier
        # already accounts for fan-in. LN affines stay at γ=1, β=0.
        nn.init.xavier_uniform_(self.qkv_linear.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)

    # --- feature map -------------------------------------------------------

    @staticmethod
    def _feature_map(x: Tensor) -> Tensor:
        """φ(x) = elu(x) + 1.

        elu(x) ∈ [-1, ∞) ⇒ φ(x) ∈ [0, ∞). Non-negativity is what makes
        the linear-attention dot product a valid (unnormalized) kernel
        similarity. Override by subclassing for alternative kernels.
        """
        return F.elu(x) + 1.0

    # --- interface ---------------------------------------------------------

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> State:
        device = self._resolve_device(device)
        H, D = self.n_heads, self.d_head
        state: State = {
            "S": torch.zeros(batch_size, H, D, D, device=device, dtype=dtype),
        }
        if self.use_normalizer:
            state["z"] = torch.zeros(batch_size, H, D, device=device, dtype=dtype)
        return state

    def step(
        self,
        x: Tensor,
        state: State,
        episode_start: Optional[Tensor] = None,
    ) -> tuple[Tensor, State, SideOutputs]:
        if episode_start is not None:
            state = apply_episode_mask(state, episode_start)

        B = x.shape[0]
        H, D = self.n_heads, self.d_head

        # Pre-LN, then fused QKV projection, then reshape into heads.
        x_normed = self.input_ln(x)                                     # (B, input_size)
        qkv = self.qkv_linear(x_normed)                                 # (B, 3*hidden_size)
        q, k, v = qkv.chunk(3, dim=-1)                                  # each (B, hidden_size)
        q = q.reshape(B, H, D)                                          # (B, H, D)
        k = k.reshape(B, H, D)
        v = v.reshape(B, H, D)

        # Kernel feature map on q and k. v stays unmapped (linear-attention
        # kernel only acts on the similarity side).
        phi_q = self._feature_map(q)                                    # (B, H, D)
        phi_k = self._feature_map(k)                                    # (B, H, D)

        # State update: S^h_t = S^h_{t-1} + φ(k_t)^T v_t — outer product
        # per head. einsum 'bhi,bhj->bhij' produces (B, H, D, D).
        S_prev = state["S"]
        S_new = S_prev + torch.einsum("bhi,bhj->bhij", phi_k, v)        # (B, H, D, D)

        # Read: numerator = φ(q_t) @ S^h_t — matrix-vector product per head.
        # einsum 'bhi,bhij->bhj' produces (B, H, D).
        numerator = torch.einsum("bhi,bhij->bhj", phi_q, S_new)         # (B, H, D)

        new_state: State = {"S": S_new}

        if self.use_normalizer:
            # Normalizer update: z^h_t = z^h_{t-1} + φ(k_t)
            z_prev = state["z"]
            z_new = z_prev + phi_k                                      # (B, H, D)
            new_state["z"] = z_new

            # Denominator: φ(q_t) · z^h_t — scalar per (B, H). Clamped to
            # eps to handle the t=0 case (z=0 at episode start before any
            # update would give 0/0; with the update-before-read order we
            # already have z_new = φ(k_t) > 0, but eps is cheap insurance).
            denom = torch.einsum("bhi,bhi->bh", phi_q, z_new)           # (B, H)
            denom = denom.clamp(min=self.eps).unsqueeze(-1)             # (B, H, 1)
            y_per_head = numerator / denom                              # (B, H, D)
        else:
            # Drop-normalizer variant (Schoelkopf 2024 et al.). The output
            # magnitude grows with episode length; output_ln downstream
            # compensates.
            y_per_head = numerator

        # Concat heads back into a single vector, project, LN.
        y_flat = y_per_head.reshape(B, self.hidden_size)                # (B, hidden_size)
        y_out = self.output_ln(self.out_linear(y_flat))                 # (B, hidden_size)

        return y_out, new_state, {}
