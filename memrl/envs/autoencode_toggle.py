"""Autoencode density toggle — a known-reward-machine retention task.

POPGym Autoencode shows a sequence of cards (WATCH), then the agent reproduces
them (PLAY). Stock reward is dense: +1/N correct, −1/N wrong, runs to the end
(max return 1.0). We re-shape it into a clean sparse↔dense δ-toggle at FIXED
memory demand (the minimal reward machine — store the sequence, reproduce it,
merge play states by remaining-suffix — is identical for both arms; only WHERE
reward is paid changes):

  DENSE   = +1/N per correct step, 0 and TERMINATE on first wrong
            (the MortarMayhem rule: per-use positive credit).
  SPARSE  = DENSE wrapped in the repo's DeferredReward → the same per-episode
            return paid as ONE terminal lump sum. Exactly return-matched to
            DENSE at γ=1 (≈ at γ<1); the ONLY difference is reward timing (δ).

Both arms TERMINATE on first wrong, so episode-length dynamics are identical and
the optimum (all correct) is 1.0 on both — the cleanest possible δ manipulation
at fixed Δ. This is the automata-theoretic illustration of the paper's D1/D2
separation and the ground-truth env for the decodability / effective-RM-size
probe (the remaining-to-reproduce stack IS the minimal-RM state; read it from
`env.unwrapped.deck["system"]` or from `info["rm_state"]`).

Order: stock Autoencode reproduces in REVERSE (LIFO/stack) — card 1 is held the
full 2N−1 steps, the maximal memory demand, which is GOOD for a memory paper.
`order="inorder"` (FIFO) reproduces in shown order (uniform Δ); it is supported
via sequence tracking and matches the colleague's theoretical example.

This wrapper sits on the RAW popgym env (obs = (mode, card_suit)), BEFORE the
phase/flatten wrappers. A reverse-mode consistency assertion checks the tracked
comparison against the stock reward sign, so the tracking is self-verifying.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class AutoencodeDensityToggle(gym.Wrapper):
    """Apply the dense MortarMayhem rule (+1/N correct, 0+terminate on wrong) to
    POPGym Autoencode, with selectable reproduction order. Wrap in DeferredReward
    for the sparse twin.

    Args:
        env:    a raw popgym Autoencode env (obs = Tuple(mode, card_suit)).
        order:  "reverse" (stock LIFO) or "inorder" (FIFO).
        strict: if True (default), assert reverse-mode correctness matches the
                stock reward sign (self-check); set False to silence in prod.
    """

    def __init__(self, env: gym.Env, order: str = "reverse", strict: bool = True):
        super().__init__(env)
        if order not in ("reverse", "inorder"):
            raise ValueError(f"order must be 'reverse' or 'inorder', got {order!r}")
        self.order = order
        self.strict = strict
        self._N = int(self.env.unwrapped.deck.num_cards)
        self._scale = 1.0 / self._N
        self._shown: list[int] = []
        self._play_idx = 0

    @staticmethod
    def _card_suit(obs) -> int:
        # raw Autoencode obs = (mode_value, card_suit)
        return int(np.asarray(obs[1]).reshape(-1)[0])

    def _target(self) -> int:
        if self.order == "reverse":
            return self._shown[self._N - 1 - self._play_idx]   # LIFO: last shown first
        return self._shown[self._play_idx]                      # FIFO: first shown first

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._shown = [self._card_suit(obs)]                    # card 1 dealt at reset
        self._play_idx = 0
        info = dict(info)
        info["phase"] = "watch"
        info["rm_state"] = tuple(self._shown)                   # sequence stored so far
        return obs, info

    def step(self, action):
        obs, r_stock, term, trunc, info = self.env.step(action)
        info = dict(info)

        if len(self._shown) < self._N:
            # WATCH: this step dealt a card; record it, no reward, no termination.
            self._shown.append(self._card_suit(obs))
            info["phase"] = "watch"
            info["rm_state"] = tuple(self._shown)
            return obs, 0.0, term, trunc, info

        # PLAY: compare action to the target under the chosen order.
        target = self._target()
        correct = int(action) == int(target)
        if self.strict and self.order == "reverse":
            # stock pays +scale iff its (reverse) comparison is correct → self-check
            assert (r_stock > 0) == correct, (
                f"reverse-mode tracking desync: stock_r={r_stock} correct={correct} "
                f"play_idx={self._play_idx} target={target} action={action}"
            )

        if correct:
            reward = self._scale
            self._play_idx += 1
            done = self._play_idx >= self._N        # reproduced all → success
            terminated = bool(term or done)
        else:
            reward = 0.0                            # MortarMayhem rule: no credit
            terminated = True                       # ...and terminate on first wrong

        # remaining-to-reproduce = the exact minimal-RM state at this point
        if self.order == "reverse":
            remaining = tuple(self._shown[: self._N - self._play_idx][::-1])
        else:
            remaining = tuple(self._shown[self._play_idx:])
        info["phase"] = "play"
        info["rm_state"] = remaining
        info["n_correct"] = self._play_idx
        info["is_correct"] = correct
        # success = reproduced ALL N correctly (the optimum, return 1.0). Always
        # present in play-phase info so Monitor(info_keywords=...) never KeyErrors.
        succeeded = bool(self._play_idx >= self._N)
        info["success"] = float(succeeded)
        info["is_success"] = succeeded
        return obs, float(reward), terminated, bool(trunc), info
