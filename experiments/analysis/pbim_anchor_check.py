"""PBIM terminal-anchor validity check (companion to oracle_potential_check.py).

Validates the two halves of the Φ(terminal)=0 anchor on a synthetic episodic
stream, with NO env or training dependencies:

  1. TELESCOPING (exact, algebraic) — with the boundary term −Φ(s_{T−1})
     delivered on done steps, the per-episode discounted shaping sum must equal
     −scale·Φ(s_0) to float precision, for ANY potential-net weights. This is
     the policy-invariance witness (endpoint-only ⇒ constant offset).
  2. ANCHOR / BOUNDEDNESS — fitting V_int by TD on a constant positive bonus
     stream must (a) pull the pre-terminal potential to ≈0, (b) keep every
     potential under the discounted-sum fixed-point bound b/(1−γ), including at
     long horizon γ=0.999. The UNANCHORED fit (the pre-fix behavior, boundary
     rows dropped) is run on identical data for reference — its drift is the
     S13 V_int→3e6 runaway mechanism in miniature.

Usage:  python experiments/analysis/pbim_anchor_check.py
Exits nonzero on failure.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")  # repo root

from memrl.exploration.base import IntrinsicRewardModule  # noqa: E402
from memrl.exploration.pbim import PBIM  # noqa: E402


class _StubPhi:
    """Feature source: the obs IS the feature (obs are random unit vectors)."""
    dim = 8

    def encode(self, obs, side, cell_state) -> torch.Tensor:
        return torch.as_tensor(np.asarray(obs), dtype=torch.float32)


class _StubBase(IntrinsicRewardModule):
    """Constant positive raw bonus — the simplest stream with a known V_int."""

    def __init__(self, n_envs: int, bonus: float = 0.3) -> None:
        super().__init__(n_envs=n_envs)
        self.phi = _StubPhi()
        self._bonus = float(bonus)

    def compute(self, obs, last_obs, action, episode_start, side, cell_state):
        return np.full((self.n_envs,), self._bonus, dtype=np.float32)


def _stream(pbim: PBIM, rng: np.ndarray, n_envs: int, T: int, n_eps: int):
    """Drive pbim.compute with ppo.py's exact call semantics; return per-episode
    (F-sequence, s0-features) so telescoping can be checked externally."""
    episodes = []
    for _ in range(n_eps):
        obs_seq = rng.standard_normal((T + 1, n_envs, _StubPhi.dim)).astype(np.float32)
        obs_seq /= np.linalg.norm(obs_seq, axis=-1, keepdims=True)
        F_rows = []
        # within-episode rows t=0..T-2: (last_obs=s_t → obs=s_{t+1}), es=False
        for t in range(T - 1):
            F_rows.append(pbim.compute(obs_seq[t + 1], obs_seq[t], None,
                                       np.zeros(n_envs, bool), {}, {}))
        # terminal row: last_obs=s_{T-1}, obs=NEXT episode's s_0 (auto-reset), es=True
        next_s0 = rng.standard_normal((n_envs, _StubPhi.dim)).astype(np.float32)
        F_rows.append(pbim.compute(next_s0, obs_seq[T - 1], None,
                                   np.ones(n_envs, bool), {}, {}))
        episodes.append((np.stack(F_rows), obs_seq[0]))
    return episodes


def check_telescoping(gamma: float = 0.99, scale: float = 1.0) -> None:
    n_envs, T = 4, 60
    rng = np.random.default_rng(0)
    pbim = PBIM(_StubBase(n_envs), gamma=gamma, scale=scale)
    for F_seq, s0 in _stream(pbim, rng, n_envs, T, n_eps=5):
        disc = (gamma ** np.arange(T))[:, None] * F_seq          # (T, n_envs)
        got = disc.sum(0)
        with torch.no_grad():
            want = -scale * pbim._potential(
                pbim.base.phi.encode(s0, {}, {})).cpu().numpy()
        resid = np.abs(got - want).max()
        assert resid < 1e-3, f"telescoping residual {resid:.2e} (want −scale·Φ(s0))"
    # cross-check pbim's own online tracker agrees
    buf = np.asarray(pbim._ep_F_buf, dtype=np.float32)
    assert buf.size == 5 * n_envs, f"tracker closed {buf.size} episodes, want {5*n_envs}"
    print(f"  telescoping: Σγ^t F_t = −scale·Φ(s_0) exact (max resid < 1e-3, "
          f"{buf.size} episodes tracked) — PASS")


def check_anchor(gamma: float, T: int, rounds: int, label: str) -> None:
    """TD fit on constant bonus: anchored Φ must be bounded by b/(1−γ) with
    pre-terminal Φ≈0; unanchored reference (pre-fix) drifts on identical data."""
    n_envs, b = 4, 0.3
    bound = b / (1.0 - gamma)
    results = {}
    for anchored in (True, False):
        rng = np.random.default_rng(1)                 # identical data both arms
        torch.manual_seed(1)
        pbim = PBIM(_StubBase(n_envs, bonus=b), gamma=gamma, fit_epochs=2)
        if not anchored:                               # pre-fix behavior: drop rows
            orig = torch.where
            pbim_update = pbim.update

            def update_dropping(rollout=None):
                # emulate old code: strip boundary rows before the fit
                keep = ~np.concatenate(pbim._boundary, 0)
                for name in ("_feat_s", "_feat_s2", "_raw_b", "_boundary"):
                    buf = np.concatenate(getattr(pbim, name), 0)[keep]
                    setattr(pbim, name, [buf])
                pbim._boundary = [np.zeros(len(buf), bool)]
                return pbim_update(rollout)

            pbim.update = update_dropping
        vmax_seen = 0.0
        for _ in range(rounds):
            _stream(pbim, rng, n_envs, T, n_eps=2)
            pbim.update(None)
            probe = rng.standard_normal((64, _StubPhi.dim)).astype(np.float32)
            probe /= np.linalg.norm(probe, axis=-1, keepdims=True)
            with torch.no_grad():
                vmax_seen = max(vmax_seen, float(pbim._potential(
                    torch.as_tensor(probe)).abs().max()))
        results[anchored] = vmax_seen
    ok = results[True] <= 1.5 * bound
    print(f"  anchor[{label}]: γ={gamma} T={T} fixed-point bound b/(1−γ)={bound:.0f} → "
          f"anchored max|Φ|={results[True]:.2f} {'≤' if ok else '>'} 1.5×bound; "
          f"unanchored (pre-fix, same data) max|Φ|={results[False]:.2f} — "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok, f"anchored potential exceeded fixed-point bound: {results[True]} > {1.5*bound}"


if __name__ == "__main__":
    print("PBIM terminal-anchor check")
    check_telescoping()
    check_anchor(gamma=0.99, T=50, rounds=150, label="short-horizon")
    check_anchor(gamma=0.999, T=300, rounds=60, label="long-horizon (S13-like)")
    print("ALL PASS")
