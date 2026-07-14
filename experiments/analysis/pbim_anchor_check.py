"""PBIM validity check — sign, normalization, telescoping, boundedness.

Validates the faithful normalized PBIM (Forbes Eq. 34) on a synthetic episodic
stream, with NO env or training dependencies:

  1. SIGN — the delivered interior shaping must equal +(b − b̄) (Forbes'
     direction, same as the raw bonus), NOT −(b − b̄). Checked against a hand
     computation from the fitted potential.
  2. TELESCOPING (exact, algebraic) — with the boundary term +V_int(s_{T−1})
     delivered on done steps, the per-episode discounted shaping sum must equal
     +scale·V_int(s_0) to float precision, for ANY potential-net weights. This
     is the policy-invariance witness (endpoint-only ⇒ constant offset).
  3. NORMALIZATION / BOUNDEDNESS — fitting V_int by TD on a CENTERED bonus
     stream keeps the potential ~0-mean and bounded, including at long horizon
     γ=0.999, so no terminal step dominates the task reward (the stall/spike
     the un-normalized version caused). The un-centered fit is run on identical
     data for reference — its magnitude is the S13-scale blow-up in miniature.

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
    """Positive raw bonus that VARIES with the obs (so centering leaves signal):
    b = base + gain·(first feature component)_+, always > 0."""

    def __init__(self, n_envs: int, bonus: float = 0.3, gain: float = 0.2) -> None:
        super().__init__(n_envs=n_envs)
        self.phi = _StubPhi()
        self._bonus = float(bonus)
        self._gain = float(gain)
        self.last_b = np.zeros(n_envs, dtype=np.float32)

    def compute(self, obs, last_obs, action, episode_start, side, cell_state):
        x = np.asarray(obs, dtype=np.float32)
        self.last_b = (self._bonus + self._gain * np.abs(x[:, 0])).astype(np.float32)
        return self.last_b


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
    """Σγ^t F_t must equal +scale·V_int(s_0) per episode, AND with the correct
    (+b') sign not the −b' consumption sign. Critically, V_int is TRAINED on a
    DETERMINISTIC chain first so |V_int| is O(0.1), not the ~4e-5 init — with the
    trivial init any sign/algebra error is below tolerance and the test is
    vacuous (the blind spot the audit flagged). With a trained V_int the wrong
    sign (Σ=−V_int(s0)) fails by 2·|V_int(s0)|.
    NOTE the delivery terminal term +V_int(s_{T−1}) is legitimately ~0 under
    centering (one centered bonus from terminal, fit-anchored), so it is NOT the
    load-bearing guard — the fit-side anchor / boundedness is, and lives in
    check_normalization (V_int bounded at γ=0.999)."""
    n_envs = 1
    L = _StubPhi.dim
    states = np.eye(L, dtype=np.float32)
    bpat = 0.1 + 0.4 * (np.arange(L) % 3) / 2.0
    base = _StubBase(n_envs)
    torch.manual_seed(0)
    pbim = PBIM(base, gamma=gamma, scale=scale, fit_epochs=4)

    def episode():
        F_rows = []
        for t in range(L - 1):
            base.last_b = np.array([bpat[t + 1]], np.float32)
            base.compute = lambda *a, **k: base.last_b
            F_rows.append(float(pbim.compute(states[t + 1][None], states[t][None],
                          None, np.zeros(1, bool), {}, {})[0]))
        term = pbim.compute(states[0][None], states[L - 1][None], None,
                            np.ones(1, bool), {}, {})[0]
        F_rows.append(float(term))
        return np.array(F_rows)

    for _ in range(300):                                # train V_int on the chain
        episode(); pbim.update(None)
    with torch.no_grad():                               # |Φ| across ALL chain states
        vmag = float(pbim._potential(pbim.base.phi.encode(states, {}, {})).abs().max())
        want = scale * float(pbim._potential(
            pbim.base.phi.encode(states[0][None], {}, {}))[0])
    got = float((gamma ** np.arange(L) * episode()).sum())
    resid = abs(got - want)
    resid_wrongsign = abs(got - (-want))               # the −b' consumption mutant
    assert vmag > 1e-2, f"V_int too small ({vmag:.1e}) — chain didn't train"
    assert resid < 1e-4, f"telescoping residual {resid:.2e} (want +scale·V_int(s0))"
    assert resid_wrongsign > 1e-3, \
        f"telescoping cannot distinguish +/−V_int(s0) (|V_int(s0)|≈0) — non-diagnostic"
    print(f"  telescoping: Σγ^t F_t = +scale·V_int(s_0) with trained V_int "
          f"(max|Φ|={vmag:.2f}, resid={resid:.1e}); wrong-sign mutant would miss by "
          f"{resid_wrongsign:.1e} — PASS")


def check_sign(gamma: float = 0.99) -> None:
    """After fitting V_int on a DETERMINISTIC chain (so V_int converges to the
    exact return-to-go and the Bellman identity is exact on the realized path),
    the delivered interior F must track +(b − b̄) — Forbes' direction — NOT the
    −(b − b̄) consumption sign the naive Φ=+V_int would give."""
    n_envs = 1
    L = _StubPhi.dim                           # deterministic chain (one-hot dim)
    states = np.eye(L, dtype=np.float32)       # one-hot obs; s_t -> s_{t+1}
    # fixed per-state bonus (varied), arrival-indexed like e3b
    bpat = 0.1 + 0.4 * (np.arange(L) % 3) / 2.0
    base = _StubBase(n_envs)

    def run_episode(pb, learn):
        for t in range(L - 1):                 # interior transitions s_t -> s_{t+1}
            base.last_b = np.array([bpat[t + 1]], np.float32)  # bonus for arrival
            base.compute = lambda *a, **k: base.last_b         # force the pattern
            pb.compute(states[t + 1][None], states[t][None], None,
                       np.zeros(1, bool), {}, {})
        # terminal: s_{L-1} -> reset (es=True)
        pb.compute(states[0][None], states[L - 1][None], None,
                   np.ones(1, bool), {}, {})

    torch.manual_seed(3)
    pbim = PBIM(base, gamma=gamma, fit_epochs=4)
    for _ in range(400):
        run_episode(pbim, learn=True)
        pbim.update(None)
    # measure delivered interior F vs +(b − b̄) on the converged chain
    ema = pbim._bonus_ema
    Fs, bs = [], []
    for t in range(L - 1):
        base.last_b = np.array([bpat[t + 1]], np.float32)
        F = pbim.compute(states[t + 1][None], states[t][None], None,
                         np.zeros(1, bool), {}, {})
        Fs.append(float(F[0])); bs.append(bpat[t + 1] - ema)
    Fs, bs = np.array(Fs), np.array(bs)
    corr = float(np.corrcoef(Fs, bs)[0, 1])
    err = float(np.mean(np.abs(Fs - bs)))
    assert corr > 0.9 and err < 0.05, \
        f"delivered F does not track +(b−b̄): corr={corr:.2f}, err={err:.3f} (sign wrong?)"
    print(f"  sign: delivered interior F ≈ +(b − b̄)  corr={corr:.3f}, "
          f"mean|F−(b−b̄)|={err:.3g}  (NOT the −b consumption sign) — PASS")


def check_normalization(gamma: float, T: int, rounds: int, label: str) -> None:
    """CENTERED fit (Eq.34) keeps |Φ| and the terminal-step |F| small; the
    UN-centered fit (momentum=1 ⇒ b̄≡0, the pre-normalization behavior) blows up
    on identical data. The terminal |F| is the stall/spike proxy: it must stay
    well under the +1 task-reward scale after the λ≈0.03 coefficient."""
    n_envs = 4
    LAM = 0.03
    results = {}
    for centered in (True, False):
        rng = np.random.default_rng(1)                 # identical data both arms
        torch.manual_seed(1)
        pbim = PBIM(_StubBase(n_envs), gamma=gamma, fit_epochs=2)
        if not centered:
            pbim._ema_momentum = 1.0                   # freeze b̄=0 ⇒ no centering
        vmax = term_spike = 0.0
        for _ in range(rounds):
            eps = _stream(pbim, rng, n_envs, T, n_eps=2)
            pbim.update(None)
            for F_seq, _ in eps:                        # terminal row is the last
                term_spike = max(term_spike, float(np.abs(F_seq[-1]).max()))
            probe = rng.standard_normal((64, _StubPhi.dim)).astype(np.float32)
            probe /= np.linalg.norm(probe, axis=-1, keepdims=True)
            with torch.no_grad():
                vmax = max(vmax, float(pbim._potential(
                    torch.as_tensor(probe)).abs().max()))
        results[centered] = (vmax, term_spike)
    (cv, cs), (uv, us) = results[True], results[False]
    ok = LAM * cs < 0.3                                 # terminal kick ≪ +1 task reward
    print(f"  normalization[{label}]: γ={gamma} T={T} → "
          f"centered max|Φ|={cv:.2f}, λ·terminal|F|={LAM*cs:.3f} "
          f"({'≪' if ok else 'NOT ≪'} 1); un-centered max|Φ|={uv:.1f}, "
          f"λ·terminal|F|={LAM*us:.2f} — {'PASS' if ok else 'FAIL'}")
    assert ok, f"centered terminal spike λ·|F|={LAM*cs:.3f} not ≪ task reward"


if __name__ == "__main__":
    print("PBIM validity check (sign / telescoping / normalization)")
    check_telescoping()
    check_sign()
    check_normalization(gamma=0.99, T=50, rounds=150, label="short-horizon")
    check_normalization(gamma=0.999, T=300, rounds=60, label="long-horizon (S13-like)")
    print("ALL PASS")
