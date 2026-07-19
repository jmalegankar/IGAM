"""RIDE — Rewarding Impact-Driven Exploration (Raileanu & Rocktäschel, ICLR 2020).

Bonus = L2 distance between the learned embeddings of *consecutive* states,
discounted by the episodic visitation count of the state landed in:

    R_IDE(s_t, a_t) = ‖φ(s_{t+1}) − φ(s_t)‖₂ / sqrt(N_ep(s_{t+1}))          (Eq. §4)

where N_ep(s_{t+1}) is the number of times s_{t+1} has been visited THIS
episode (initialized to 1 on first visit → undiscounted; sqrt-decayed on
revisits). This is the paper's fix for the go-back-and-forth failure mode:
without the count term the agent can farm bonus by oscillating between two
states with a large embedding gap.

The embedding φ (θ_emb) is learned by BOTH a forward and an inverse dynamics
model, exactly as in ICM (Pathak 2017):

    φ̂_{t+1} = f_fwd(φ_t, a_t)      L_fwd = ‖φ(s_{t+1}) − φ̂_{t+1}‖²
    â_t     = f_inv(φ_t, φ_{t+1})  L_inv = CE(â_t, a_t)
    L_emb   = ω_fwd·L_fwd + ω_inv·L_inv

Key differences from the codebase's `ICM`:
  * The BONUS is the consecutive-embedding distance (impact), NOT the forward
    prediction error. The forward model exists only to *shape φ*, never to
    score the reward — so RIDE's bonus does not vanish as the forward model
    gets accurate (a central selling point of the paper, §4/§6.1).
  * The forward target IS detached (canonical ICM, Pathak 2017:
    `||f_fw(φ_t,a_t) − φ(s_{t+1}).detach()||²`). This is the single fix that
    prevents representation collapse; the RIDE loss coefficients are the
    published ones (forward 10, inverse 0.1). Training θ_emb through BOTH sides
    of the forward loss (target NOT detached) makes φ≈const the global optimum
    (f_fw learns the constant, L_fw→0), collapsing the impact bonus ‖φ(s')−φ(s)‖
    to 0 — verified empirically (fwd_loss→0, inverse acc stuck at chance 0.25,
    bonus 0.7→0.01). With the detach, the published 10/0.1 coefficients train
    fine on the synthetic chain and the real S13/MPG envs (inv_acc→1, bonus
    stable). The official RIDE (facebookresearch/impact-driven-exploration:
    forward_loss_coef=10.0, inverse_loss_coef=0.1, NO detach) avoids collapse via
    a CONVOLUTIONAL embedding; we use a flat-MLP embedding to match the φ of
    E3B/NovelD (bonuses differ by mechanism, not architecture), so the un-detached
    loss collapses. Hence the ONLY deviation from published RIDE is this
    ICM-standard forward-target detach; coefficients and the impact-reward
    definition ‖φ(s')−φ(s)‖/√N_ep are unchanged.

Like the paper, θ_emb / θ_fwd / θ_inv are trained ONLY by L_fwd + L_inv, never
by the RL loss (this module is entirely separate from the policy network).

RIDE is episodic in the count term (reset each episode) but its embedding φ is
lifelong (trained across episodes). Multi-head inverse model + boundary-drop in
update() mirror `E3BIDM` so MultiDiscrete action spaces train instead of
crashing.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .base import IntrinsicRewardModule
from .e3b import RunningStd
from .rnd import _MLP, _obs_to_tensor


class _EpisodicCount:
    """Per-env hash map: state key → visits this episode.

    Returns N_ep INCLUDING the current visit, so first visit → 1 (matches the
    paper's "initialized to 1"), second → 2, … The reward divides by sqrt(N_ep),
    so a first visit is undiscounted.
    """

    def __init__(self, n_envs: int) -> None:
        self.counts: list[dict] = [dict() for _ in range(n_envs)]

    def visit_then_count(self, keys: list) -> np.ndarray:
        out = np.empty(len(self.counts), dtype=np.float32)
        for i, k in enumerate(keys):
            c = self.counts[i].get(k, 0) + 1
            self.counts[i][k] = c
            out[i] = c
        return out

    def reset_envs(self, env_ids) -> None:
        for i in env_ids:
            self.counts[i] = dict()


class RIDE(IntrinsicRewardModule):
    """Impact-driven exploration bonus.

    Args:
        n_envs:        parallel envs
        obs_dim:       flattened obs dim
        n_actions:     action vocab (Discrete). For MultiDiscrete pass action_dims.
        feature_dim:   φ dim
        hidden_dim:    internal MLP width
        lr:            Adam LR for φ + fwd + inv
        forward_coef:  ω_fwd weight on L_fwd (paper/official code: 10.0)
        inverse_coef:  ω_inv weight on L_inv (paper/official code: 0.1)
        idm_epochs:    passes over the rollout's transitions per update()
        idm_batch:     minibatch size for dynamics training
        normalize:     divide the (count-discounted) bonus by its running std
        max_bonus:     clip returned bonus
        action_dims:   per-sub-action vocab sizes (Discrete → [n]; MultiDiscrete
                       → list(nvec)). Enables a multi-head inverse model.
    """

    def __init__(
        self,
        n_envs: int,
        obs_dim: int,
        n_actions: int,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        forward_coef: float = 10.0,  # canonical RIDE (Raileanu 2020); collapse is
        inverse_coef: float = 0.1,   # prevented by DETACHING the forward target, not
                                     # by reweighting — see the fwd-loss note below.
        idm_epochs: int = 1,
        idm_batch: int = 512,
        normalize: bool = True,
        max_bonus: float = 10.0,
        action_dims: Sequence[int] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(n_envs=n_envs, device=device)
        self.obs_dim = obs_dim
        self.n_actions = int(n_actions)
        # action_dims: [n] for Discrete, list(nvec) for MultiDiscrete. The inverse
        # model is multi-head (one softmax per sub-action) and the forward model
        # takes the concatenated per-sub-action one-hot as its action input.
        self.action_dims = [int(d) for d in action_dims] if action_dims else [int(n_actions)]
        self.act_dim = int(sum(self.action_dims))
        self.feature_dim = feature_dim
        self.forward_coef = float(forward_coef)
        self.inverse_coef = float(inverse_coef)
        self.idm_epochs = int(idm_epochs)
        self.idm_batch = int(idm_batch)
        self.normalize = normalize
        self.max_bonus = max_bonus

        # φ encoder — trained via forward + inverse dynamics (θ_emb).
        self.encoder = _MLP(obs_dim, hidden_dim, feature_dim).to(self.device)
        # Forward model: (φ_t, onehot(a_t)) → φ̂_{t+1}.
        self.fwd = _MLP(feature_dim + self.act_dim, hidden_dim, feature_dim).to(self.device)
        # Inverse model: (φ_t, φ_{t+1}) → concatenated per-sub-action logits.
        self.inv = _MLP(feature_dim * 2, hidden_dim, self.act_dim).to(self.device)

        params = (list(self.encoder.parameters())
                  + list(self.fwd.parameters())
                  + list(self.inv.parameters()))
        self.opt = torch.optim.Adam(params, lr=lr)
        self.rms = RunningStd() if normalize else None

        self.counts = _EpisodicCount(n_envs)
        # N_ep diagnostics — to MEASURE (not assume) whether the episodic count
        # actually increments on this env. If ride_n_ep_max stays ~1 the obs-hash
        # never repeats (count inert); if it grows, the count works but only
        # ATTENUATES (÷√N) the reset reward rather than eliminating it.
        self._nep_sum = 0.0; self._nep_cnt = 0; self._nep_max = 0.0; self._nep_gt1 = 0
        # Opt into env infos so we can read a discrete pose key (info["novelty_key"])
        # for the episodic count on continuous-state envs; else we hash raw obs.
        self.wants_infos = True

    # ── action one-hot (multi-head) ───────────────────────────────────────────
    def _onehot_actions(self, a_t: Tensor) -> Tensor:
        """a_t: (N, K) long → (N, sum(action_dims)) float."""
        parts = [F.one_hot(a_t[:, j], num_classes=self.action_dims[j]).float()
                 for j in range(len(self.action_dims))]
        return torch.cat(parts, dim=-1)

    # ── bonus ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def compute(self, obs, last_obs, action, episode_start, side, cell_state,
                infos=None) -> np.ndarray:
        x_prev = _obs_to_tensor(last_obs, self.device)   # s_t
        x_cur  = _obs_to_tensor(obs,      self.device)   # s_{t+1}
        phi_prev = self.encoder(x_prev)
        phi_cur  = self.encoder(x_cur)
        # Impact = ‖φ(s_{t+1}) − φ(s_t)‖₂  (L2 norm, NOT squared — per Eq. §4).
        impact = (phi_cur - phi_prev).norm(dim=-1)
        bonus = impact.cpu().numpy().astype(np.float32)

        # Episodic count discount N_ep(s_{t+1}). Reset envs that just started a
        # new episode (episode_start=dones) BEFORE counting, so the new episode's
        # first obs counts as 1.
        reset_ids = np.where(episode_start)[0].tolist()
        if reset_ids:
            self.counts.reset_envs(reset_ids)
        n = len(self.counts.counts)
        # Prefer a DISCRETE env-provided pose key so the count saturates on
        # continuous-state envs; else hash the raw obs (exact for discrete-grid
        # envs, whose frames are finite). Mirrors NovelD's key selection.
        if (infos is not None and len(infos) == n
                and all(isinstance(inf, dict) and "novelty_key" in inf for inf in infos)):
            keys = [infos[i]["novelty_key"] for i in range(n)]
        else:
            flat = np.asarray(obs).reshape(n, -1)
            keys = [flat[i].tobytes() for i in range(n)]
        n_ep = self.counts.visit_then_count(keys)          # (n_envs,) ≥ 1
        self._nep_sum += float(n_ep.sum()); self._nep_cnt += int(n_ep.size)
        self._nep_max = max(self._nep_max, float(n_ep.max()))
        self._nep_gt1 += int((n_ep > 1).sum())
        bonus = bonus / np.sqrt(n_ep)

        # Running-std normalization (codebase convention; keeps λ_intrinsic
        # scale-comparable to RND/NovelD/E3B/ICM). Update stats BEFORE dividing.
        if self.normalize:
            self.rms.update(bonus)
            bonus = bonus / max(self.rms.std, 1e-8)
        bonus = np.nan_to_num(bonus, nan=0.0, posinf=self.max_bonus, neginf=0.0)
        bonus = np.clip(bonus, 0.0, self.max_bonus)
        self._record_bonus(bonus)
        return bonus

    def reset_envs(self, env_ids) -> None:
        self.counts.reset_envs(env_ids)

    # ── train φ + forward + inverse ───────────────────────────────────────────
    def update(self, rollout: Any) -> dict[str, float]:
        """Train the embedding + forward + inverse models on within-episode
        (sₜ, aₜ, sₜ₊₁) transitions from the just-collected rollout.

        Consecutive pairs that cross an episode boundary (episode_starts[t+1]==1)
        are dropped so the dynamics models never see a (terminal → reset) pair.
        """
        if not (hasattr(rollout, "observations") and hasattr(rollout, "actions")):
            return {}
        obs = np.asarray(rollout.observations)
        act = np.asarray(rollout.actions)
        if obs.ndim < 2 or obs.shape[0] < 2:
            return {}
        T, B = obs.shape[0], obs.shape[1]
        K = len(self.action_dims)
        obs_f = obs.reshape(T, B, -1)
        s   = obs_f[:-1].reshape(-1, self.obs_dim)
        s2  = obs_f[1:].reshape(-1, self.obs_dim)
        a   = act.reshape(T, B, K)[:-1].reshape(-1, K)

        eps = getattr(rollout, "episode_starts", None)
        if eps is not None:
            keep = np.asarray(eps).reshape(T, B)[1:].reshape(-1) == 0
            s, s2, a = s[keep], s2[keep], a[keep]

        N = s.shape[0]
        if N < 2:
            return {}

        s_t  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        a_t  = torch.as_tensor(a,  dtype=torch.long,    device=self.device)  # (N, K)
        splits = list(self.action_dims)

        last_fwd, last_inv, last_acc = 0.0, 0.0, 0.0
        for _ in range(max(1, self.idm_epochs)):
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, self.idm_batch):
                idx = perm[i:i + self.idm_batch]
                phi_s  = self.encoder(s_t[idx])
                phi_s2 = self.encoder(s2_t[idx])
                a_oh   = self._onehot_actions(a_t[idx])
                # Forward loss — target DETACHED (canonical ICM). Training θ_emb
                # through the target makes φ≈const the global optimum and collapses
                # the impact bonus; detaching grounds φ in the inverse loss.
                phi_hat = self.fwd(torch.cat([phi_s, a_oh], dim=-1))
                L_fwd = (phi_hat - phi_s2.detach()).pow(2).sum(dim=-1).mean()
                # Inverse loss — multi-head CE (one softmax per sub-action).
                logits = self.inv(torch.cat([phi_s, phi_s2], dim=-1))
                chunks = torch.split(logits, splits, dim=-1)
                L_inv = sum(F.cross_entropy(chunks[j], a_t[idx, j]) for j in range(K)) / K

                L = self.forward_coef * L_fwd + self.inverse_coef * L_inv
                self.opt.zero_grad()
                L.backward()
                self.opt.step()
                last_fwd, last_inv = float(L_fwd.item()), float(L_inv.item())
                accs = [(chunks[j].argmax(-1) == a_t[idx, j]).float().mean() for j in range(K)]
                last_acc = float(torch.stack(accs).mean())

        out = {"ride_fwd_loss": last_fwd,
               "ride_inv_loss": last_inv,
               "ride_inv_acc":  last_acc,
               "ride_n":        float(N)}
        if self.rms is not None:
            out["ride_running_std"] = float(self.rms.std)
        # N_ep stats over the rollout: does the episodic count actually increment?
        if self._nep_cnt:
            out["ride_n_ep_mean"] = self._nep_sum / self._nep_cnt
            out["ride_n_ep_max"] = self._nep_max
            out["ride_frac_revisit"] = self._nep_gt1 / self._nep_cnt
        self._nep_sum = 0.0; self._nep_cnt = 0; self._nep_max = 0.0; self._nep_gt1 = 0
        return out
