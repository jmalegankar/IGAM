"""PBIM — Potential-Based Intrinsic Motivation wrapper.

Wraps ANY episodic intrinsic module (canonically E3BIDM) and delivers its
guidance in *potential-based* form, so the shaped reward provably preserves the
optimal policy of the TRUE task (Ng–Harada–Russell 1999) while still densifying
the critic's learning signal.

Why this module exists (the paper's load-bearing ablation, "C2")
----------------------------------------------------------------
The headline claim is that an episodic bonus trains the recurrent memory through
a *behavioral* channel: bonus → advantage → trajectory distribution → the data
the memory write is trained on. That channel carries TWO things at once:

  (1) DENSIFICATION — denser reward events let the critic learn value pre-terminal
      (a potential-based effect: it does not change the optimal policy);
  (2) POLICY BIAS / DISTRIBUTION-SHIFT — the non-potential residual ε that
      actually moves the policy (Skalse 2022 hackability; the "ε cost").

Raw E3B mixes the two. PBIM isolates (1): it keeps the densification but strips
the policy-altering bias by construction. The adjudication:

  * PBIM-e3b  ≈  raw-e3b   ⇒  the benefit is DENSIFICATION (critic-side credit).
  * PBIM-e3b  <  raw-e3b   ⇒  the behavioral/distribution channel is load-bearing
                             (chasing novelty *exercises* memory in a way that
                              pure densification does not).

Either outcome is a result, and the pair pins down the mechanism behind the
decodability curve (see docs/decodability_probe_spec.md).

Construction of the potential
-----------------------------
Following the PBIM transform (Forbes et al. 2024, "Potential-Based Reward Shaping
for Intrinsic Motivation"), the potential is the *value function of the intrinsic
reward*:  Φ(s) ≈ V_int(s) = E[ Σ_k γ^k b_{t+k} ].  We learn V_int by TD on the
raw bonus stream, then deliver

      F_t = γ · Φ(s_{t+1}) − Φ(s_t)

as the shaping reward IN PLACE OF the raw bonus. Over any trajectory this
telescopes to a constant offset (−Φ(s_0) plus the terminal Φ), so the optimal
policy of r_ext is unchanged, but the per-step F front-loads the intrinsic
guidance into the value targets exactly as a dense reward would.

The potential head sits on the base module's φ feature (detached), so it inherits
E3B's controllable-feature representation for free and adds negligible compute.

NOTE on the ≥0 contract: unlike other intrinsic modules, PBIM returns a SIGNED
shaping term (potential differences are negative as often as positive). This is
correct — PPO adds `lambda_intrinsic * bonus` to the reward (ppo.py) and a signed
PBRS term is exactly what policy-invariance requires. Do not clip it to ≥0.

Sign, normalization, and the episodic boundary (following Forbes Eq. 34)
------------------------------------------------------------------------
Three ingredients make this the faithful, normalized PBIM rather than a raw
learned-value potential:

  * SIGN — Forbes' potential is Φ_F = −V_int, so the interior shaping is
    F_t = γΦ_F(s_{t+1}) − Φ_F(s_t) = +(b − b̄): the SAME directional guidance as
    the raw bonus (what the densification arm must test). The naive Φ = +V_int
    delivers −(b − b̄) (the "consumption" sign), which pushes learning the wrong
    way; we realize Φ_F by delivering V_int(s_t) − γV_int(s_{t+1}).
  * NORMALIZATION (Eq. 34) — we center the raw bonus by a running mean b̄ before
    it enters the potential. This keeps every per-step F small so no single
    (terminal) step dominates the +1 task reward — the un-normalized, uncentered
    version drove a stall/spike pathology (agents ran out the clock to avoid the
    large −Φ(s_{T−1}) terminal kick) — and makes the fit target ~0-mean so V_int
    is bounded even at γ→1 (this is what actually prevents the S13 V_int→3×10⁶
    runaway; the terminal anchor below is then belt-and-suspenders).
  * TERMINAL Φ=0, enforced in BOTH places. DELIVERY: on a done step the true
    transition is (s_{T−1} → terminal) with Φ_F(terminal)=0, so the boundary term
    is γ·0 − Φ_F(s_{T−1}) = +V_int(s_{T−1}); the per-episode discounted shaping sum
    then telescopes EXACTLY to +V_int(s_0) ≈ 0 (endpoint-only ⇒ invariant). PBIM
    sets `zero_bonus_on_done = False` so MemPPO's E3B-motivated done-step zeroing
    doesn't clobber it. TD FIT: boundary rows anchor V(φ(s_{T−1})) toward 0 rather
    than being dropped (a dropped-boundary pure bootstrap lets the constant mode
    drift; centering already removes the drift's fuel, the anchor pins the mode).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .base import IntrinsicRewardModule
from .e3b_module import E3BIDM


class PBIM(IntrinsicRewardModule):
    """Potential-based wrapper around an episodic intrinsic module.

    Emits the terminal boundary term −Φ(s_{T−1}) on done steps (see module
    docstring) — MemPPO must NOT zero it (`zero_bonus_on_done = False`).

    Args:
        base:            an instantiated IntrinsicRewardModule with a `.phi`
                         feature source (E3BIDM is the intended base).
        gamma:           discount used by training (MUST match PPO's gamma so the
                         telescoping is consistent with the value targets).
        potential_hidden: width of the V_int head.
        lr:              Adam LR for the potential head.
        fit_epochs:      passes over the buffered transitions per update().
        scale:           optional output scale on F (keeps F on the same numeric
                         scale as the raw bonus; the paper holds lambda_intrinsic
                         fixed across arms and tunes nothing here).
    """

    zero_bonus_on_done = False

    def __init__(
        self,
        base: IntrinsicRewardModule,
        *,
        gamma: float = 0.995,
        potential_hidden: int = 128,
        lr: float = 1e-3,
        fit_epochs: int = 4,
        scale: float = 1.0,
        ema_momentum: float = 0.99,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(n_envs=base.n_envs, device=device)
        if not hasattr(base, "phi"):
            raise ValueError(
                "PBIM needs a base module exposing `.phi` (a PhiSource). "
                "Use E3BIDM / E3B as the base."
            )
        self.base = base
        self.gamma = float(gamma)
        self.fit_epochs = int(fit_epochs)
        self.scale = float(scale)

        feat_dim = int(base.phi.dim)
        self.potential = nn.Sequential(
            nn.Linear(feat_dim, potential_hidden), nn.ReLU(),
            nn.Linear(potential_hidden, potential_hidden), nn.ReLU(),
            nn.Linear(potential_hidden, 1),
        ).to(self.device)
        # Small init so Φ starts near 0 (shaped reward starts near 0).
        for m in self.potential.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

        self.opt = torch.optim.Adam(self.potential.parameters(), lr=lr)

        # Running mean b̄ of the RAW bonus (Forbes Eq. 34 normalization). We
        # center the bonus (b − b̄) before it enters the potential, both in the
        # fit target and — via the Bellman identity — in the delivered F. This
        # (a) matches Forbes' preferred normalized transform, (b) keeps the
        # per-step shaping small so no single terminal step dominates the +1 task
        # reward (the stall/spike pathology of the un-normalized version), and
        # (c) makes the fit target ~0-mean so V_int stays bounded even at γ→1 on
        # long horizons — centering removes the divergence at its source, the
        # terminal anchor is then just belt-and-suspenders.
        # ema_momentum controls how fast b̄ tracks the bonus. E3B's raw bonus
        # DECAYS over training (novelty falls as φ/ellipsoid learn); a too-slow b̄
        # lags that decay, leaving a persistently-negative centered bonus whose
        # return-to-go accumulates to |V_int|~O(10) on long horizons (S13,
        # γ=0.999·T=845) and nudges a residual explore-and-dawdle. A faster b̄
        # (lower momentum) tracks the decay and keeps the centered bonus ~0.
        self._bonus_ema: float = 0.0
        self._ema_momentum: float = float(ema_momentum)

        # Transition buffer of φ-FEATURES (small) for fitting V_int.
        # Stores (phi_s, phi_s2, centered_b, cross_boundary) per env per step.
        self._feat_s: list[np.ndarray] = []
        self._feat_s2: list[np.ndarray] = []
        self._raw_b: list[np.ndarray] = []
        self._boundary: list[np.ndarray] = []
        self._last_diag: dict[str, float] = {}

    # ── potential helper ────────────────────────────────────────────────────
    @torch.no_grad()
    def _phi_feat(self, obs: Any, side: dict, cell_state: dict) -> Tensor:
        """φ feature for obs (detached, on device). IDMPhi.encode is obs-only."""
        return self.base.phi.encode(obs, side, cell_state).to(self.device)

    @torch.no_grad()
    def _potential(self, feat: Tensor) -> Tensor:
        return self.potential(feat).squeeze(-1)

    # ── required ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def compute(self, obs, last_obs, action, episode_start, side, cell_state) -> np.ndarray:
        # 1) advance the base module (updates the episodic ellipsoid, records its
        #    own raw-bonus diagnostics) and grab the raw bonus.
        raw_b = self.base.compute(obs, last_obs, action, episode_start, side, cell_state)
        raw_b = np.nan_to_num(np.asarray(raw_b, dtype=np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)

        # 1b) center by the running mean b̄ (Forbes Eq. 34). The potential is the
        #     value of the CENTERED bonus, so the delivered guidance is +（b − b̄)
        #     (same direction as raw e3b, de-biased) rather than the raw bonus.
        #     Sanitize before it enters the fit target: a non-finite b̄ would make
        #     the potential head all-NaN with no recovery (Adam never un-NaNs),
        #     and the delivered F is nan_to_num'd downstream so it would silently
        #     mask a dead potential. Cheap insurance (inert with the E3B base).
        b_centered = np.nan_to_num(raw_b - self._bonus_ema,
                                   nan=0.0, posinf=0.0, neginf=0.0)
        self._bonus_ema = (self._ema_momentum * self._bonus_ema
                           + (1.0 - self._ema_momentum) * float(raw_b.mean()))

        # 2) features for s_t (last_obs) and s_{t+1} (obs).
        feat_s = self._phi_feat(last_obs, side, cell_state)
        feat_s2 = self._phi_feat(obs, side, cell_state)

        # 3) shaped reward. With Φ = V_int(centered bonus) the potential-difference
        #    γΦ(s_{t+1}) − Φ(s_t) equals −(b − b̄) by the Bellman identity — the
        #    "consumption" sign. Forbes' potential is Φ_F = −V_int, which delivers
        #    +(b − b̄): the SAME directional guidance as the raw bonus (what the
        #    densification arm must test), just de-biased. We realize Φ_F by
        #    NEGATING the difference here. On done rows the true transition is
        #    (s_{T−1} → terminal) with Φ_F(terminal) ≡ 0, so the boundary term is
        #    γ·0 − Φ_F(s_{T−1}) = +V_int(s_{T−1}); the per-episode discounted
        #    shaping sum then telescopes exactly to +V_int(s_0) ≈ 0 (centered),
        #    endpoint-only ⇒ policy-invariant.
        V_s = self._potential(feat_s)
        V_s2 = self._potential(feat_s2)
        F = V_s - self.gamma * V_s2                    # = +(b − b̄) interior (Bellman)
        es = torch.as_tensor(np.asarray(episode_start, dtype=bool), device=self.device)
        F = torch.where(es, V_s, F)                    # terminal: +V_int(s_{T−1})
        F_np = (self.scale * F).detach().cpu().numpy().astype(np.float32)
        F_np = np.nan_to_num(F_np, nan=0.0, posinf=0.0, neginf=0.0)

        # 4) stash features + CENTERED target for the V_int fit in update().
        self._feat_s.append(feat_s.detach().cpu().numpy())
        self._feat_s2.append(feat_s2.detach().cpu().numpy())
        self._raw_b.append(b_centered)
        self._boundary.append(np.asarray(episode_start, dtype=bool))

        # 4b) telescoping / return-neutrality check: the DISCOUNTED per-episode
        #     shaping sum Σ_t γ^t F_t. With the terminal boundary term delivered
        #     (step 3), this telescopes EXACTLY to +V_int(s_0) — endpoint-only, so
        #     it must CONCENTRATE (start-state variation only) and stay BOUNDED and
        #     NEAR ZERO (V_int is the value of the ~0-mean CENTERED bonus). A large
        #     or return-correlated value ⇒ the "potential" is not telescoping (the
        #     shaping-instability the P5 caveat warns about).
        #     NOTE the done row IS the closing episode's terminal step: its F
        #     (+V_int(s_{T−1})) is added at the OLD discount γ^{T−1} BEFORE the
        #     episode's sum is closed out and the counters reset.
        es_arr = np.asarray(episode_start, dtype=bool)
        if getattr(self, "_disc_sum", None) is None or self._disc_sum.shape != F_np.shape:
            self._disc_sum = np.zeros_like(F_np)
            self._gamma_pow = np.ones_like(F_np)
            self._ep_F_buf: list[float] = []
        self._disc_sum += self._gamma_pow * F_np
        for i in range(F_np.shape[0]):
            if es_arr[i]:                                  # this row ended the episode
                self._ep_F_buf.append(float(self._disc_sum[i]))
                self._disc_sum[i] = 0.0
                self._gamma_pow[i] = 1.0
            else:
                self._gamma_pow[i] *= self.gamma

        self._record_bonus(F_np)  # diagnostics track the SHAPED term
        return F_np

    # ── optional hooks ──────────────────────────────────────────────────────
    def update(self, rollout: Any) -> dict[str, float]:
        # (a) train the base's learned φ / IDM exactly as usual.
        base_diag = self.base.update(rollout)

        # (b) fit V_int by TD on the buffered CENTERED-bonus stream.
        if not self._feat_s:
            return {f"base_{k}": v for k, v in base_diag.items()}

        s = torch.as_tensor(np.concatenate(self._feat_s, 0), device=self.device)
        s2 = torch.as_tensor(np.concatenate(self._feat_s2, 0), device=self.device)
        b = torch.as_tensor(np.concatenate(self._raw_b, 0), device=self.device)
        bnd = torch.as_tensor(np.concatenate(self._boundary, 0), device=self.device)

        last_loss = 0.0
        if s.shape[0] > 0:
            N = s.shape[0]
            bs = 4096
            for _ in range(self.fit_epochs):
                perm = torch.randperm(N, device=self.device)
                for i in range(0, N, bs):
                    idx = perm[i:i + bs]
                    with torch.no_grad():
                        # Terminal anchor (Φ(terminal)=0): boundary rows fit
                        # V(φ(s_{T−1})) toward 0. Their stored b/s2 belong to the
                        # NEXT episode's reset obs (junk here) — the where() masks
                        # both out. Without any anchored row the fit is a pure
                        # bootstrap and Φ's constant mode drifts unboundedly at
                        # γ→1 (the S13 V_int→3e6 runaway).
                        boot = b[idx] + self.gamma * self.potential(s2[idx]).squeeze(-1)
                        target = torch.where(bnd[idx], torch.zeros_like(boot), boot)
                    pred = self.potential(s[idx]).squeeze(-1)
                    loss = torch.mean((pred - target) ** 2)
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    self.opt.step()
                    last_loss = float(loss.detach())

        with torch.no_grad():
            v_mean = float(self.potential(s).mean()) if s.shape[0] else 0.0

        # clear buffers
        self._feat_s.clear(); self._feat_s2.clear()
        self._raw_b.clear(); self._boundary.clear()

        out = {f"base_{k}": v for k, v in base_diag.items()}
        out.update({
            "pbim_potential_loss": last_loss,
            "pbim_V_int_mean": v_mean,
        })
        self._last_diag = out
        return out

    def reset_envs(self, env_ids) -> None:
        self.base.reset_envs(env_ids)

    def diagnostics(self) -> dict[str, float]:
        base = super().diagnostics()  # shaped-term mean/max/std
        base.update(self._last_diag)
        # Telescoping check: mean / |mean| / std of the discounted per-episode
        # shaping sum over episodes that ENDED this rollout. Near-zero & stable
        # ⇒ return-neutral (true potential); large/growing ⇒ instability.
        buf = getattr(self, "_ep_F_buf", None)
        if buf:
            arr = np.asarray(buf, dtype=np.float32)
            base["pbim_ep_shaping_disc_sum_mean"] = float(arr.mean())
            base["pbim_ep_shaping_disc_sum_absmean"] = float(np.abs(arr).mean())
            base["pbim_ep_shaping_disc_sum_std"] = float(arr.std())
            base["pbim_ep_shaping_n"] = float(len(arr))
            self._ep_F_buf = []
        return base


def make_pbim_e3b_idm(
    *,
    n_envs: int,
    obs_dim: int,
    n_actions: int,
    gamma: float = 0.995,
    device: torch.device | str = "cpu",
    **kwargs,
) -> PBIM:
    """Convenience builder: PBIM wrapping the canonical E3BIDM base.

    `kwargs` are forwarded to E3BIDM (lambda_reg, hidden_dim, lr, idm_epochs, …)
    EXCEPT PBIM-specific keys (gamma, potential_hidden, lr_pbim, fit_epochs, scale,
    ema_momentum) which are popped here. Keep E3BIDM's config IDENTICAL to the
    raw-e3b arm so the only difference between arms is potential-vs-raw delivery.
    """
    pbim_keys = {
        "potential_hidden": kwargs.pop("potential_hidden", 128),
        "lr": kwargs.pop("lr_pbim", 1e-3),
        "fit_epochs": kwargs.pop("fit_epochs", 4),
        "scale": kwargs.pop("scale", 1.0),
        "ema_momentum": kwargs.pop("ema_momentum", 0.99),
    }
    base = E3BIDM(n_envs=n_envs, obs_dim=obs_dim, n_actions=n_actions,
                  device=device, **kwargs)
    return PBIM(base, gamma=gamma, device=device, **pbim_keys)
