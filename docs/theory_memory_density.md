# Theory note — Memory × Reward Density (formalization)

*Standalone version of `revelation_and_densification.md` §4a (doc of record if they drift:
this file). Drafted 2026-06-11. Status: propositions-with-cited-lemmas; the Δ/δ
formalization, the synthesis, and the experimental realization are the claimed contribution.*

## Setup

POMDP with hidden state; recurrent policy `π_θ(a_t | m_t)` over memory
`m_t = f_θ(m_{t−1}, o_t)`; recurrent PPO, TBPTT chunk length `k`, GAE. Episode length `T`.

## The two gaps (definitions)

**D1 — write–use gap Δ (the MEMORY axis).** Task-relevant information is *observable* at a
write opportunity `t_w` (cue shown; fall felt; lit glimpse of coin/exit) and must *condition
behavior* at a use time `t_u` (junction turn; next path step; navigating dark).
`Δ = t_u − t_w`. Memory-essentiality = the info must cross Δ inside `m` because `o_{t_u}` is
aliased (Singh–Jaakkola–Jordan 1994; Littman 1994).

**D2 — use–pay gap δ (the DENSITY axis).** Distance from the use to the reward event that
credits it, with a sign:
- sparse terminal: `δ = T − t_u`, one positive event at the end;
- aligned-dense: `δ ≈ 0`, positive events *at uses* (MM `+1/cc` per command; MysteryPath
  `+0.1` per tile);
- anti-dense: `δ ≈ 0`, **negative** events *at errors* (fall / spotlight penalties) — paying
  avoidance rather than use.

**Design as theorem-shaped object:** our toggles manipulate δ (and event sign) while Δ,
dynamics, observations, and π\* are held fixed — return-matched exactly (optimum-preservation
lemmas: the optimum never pays the penalties; MM aligned max = cc·(1/cc) = sparse max = 1.0;
SS both arms max 1.25).

## Propositions

**T1 (truncated credit to memory writes — exact core).** Under TBPTT(k), gradients do not
cross chunk boundaries. Therefore the *direct* credit (policy-gradient or value-gradient) to
the write computation at `t_w` originates only from advantages/TD-errors in `[t_w, t_w + k)`.
- Dense with Δ < k: the use-event pays inside the write's chunk → **one-hop credit, every
  episode the event fires**.
- Sparse with `T − t_w > k`: **zero direct credit, ever**. The write learns only through
  bootstrapped value propagation: ~`⌈(T − t_w)/k⌉` sequential TD stages across updates, each
  adding bootstrap bias/variance, on a source signal with sparse-return variance
  (∝ 1/p(success)).
*The zero-direct-credit statement is exact (definition of truncation); the multi-hop cost is
a standard TD-propagation argument and is stated as such.*

**T1.1 (the bonus, memory-specifically).** An episodic novelty bonus manufactures reward
events at probe/visit times: it sets an effective `δ_eff ≈ 0` with alignment α to true
progress. For α > 0 it **restores one-hop credit to writes that sparse extrinsic reward
cannot reach** — the memory-level mechanism of densification gain.

> **Falsifiable prediction (the theory's own test): the bonus's benefit grows as k shrinks.**
> Validation experiment **B7-theory** (`experiments/tbptt_density/`):
> GRU × **k ∈ {1, 2, 4, 8, 16, 32, 64, 128}** × {none, e3b_idm} × sparse MysteryPath ×
> 3 seeds = 48 runs. `n_chunks_per_batch = 2048/k` holds the PPO batch (in steps) constant
> across k, so ONLY truncation varies. k=1 is the clean extreme: state still carries forward
> at inference, but no gradient ever crosses a step — all temporal credit must come through
> the value bootstrap. Predicted: e3b−none widens monotonically as k→1; at k=128 (≥ episode
> length, full BPTT) the truncation channel is closed and the residual gap isolates the
> non-truncation part of densification (variance reduction).

**T2 (memory stores what gets paid — the "leverage" direction).** The gradient shaping memory
*content* is proportional to advantages at use times; training drives `m_t` toward a
sufficient statistic **for the reward events actually received**, not for the task as
specified:
- aligned-dense → events at progress → stored statistic must contain the task info (reward
  content = task content);
- sparse → rare success events carry goal content at variance ∝ 1/p(success) — learnable,
  slowly; the bonus's manufactured events substitute;
- anti-dense → events at errors → the cell is trained toward the **cheapest statistic
  sufficient for avoidance**. If a passive sanctuary exists (MysteryPath: don't move), that
  statistic is **empty — the freeze is the degenerate fixed point of reward-shaped
  representation learning**. SS (no sanctuary: spotlights wander, standing still doesn't
  avoid the penalty) tests whether the empty-statistic fixed point requires *avoidability* —
  the registered boundary-condition question (none-anti sign deliberately open).

**T3 (optimum preservation ⇒ everything observed is dynamics).** The toggles preserve π\*
exactly (penalties never paid by the optimum; MM return-matched by construction), so the
freeze, the rescue, and the flip are **learning-dynamics phenomena, not task changes**:
return-matched ≠ dynamics-matched, at both the policy and the representation level.

## How the empirical program instantiates this

| Object in theory | Experiment |
|---|---|
| δ-manipulation at fixed Δ | the three arms on MysteryPath; MM sparse↔aligned; SS sparse↔anti |
| δ_eff-manipulation (bonus) | {none, e3b} factor everywhere |
| T1 truncation channel | **B7-theory k-sweep** (above) |
| T1 bootstrap-chain length | B6: `explained_variance` recovery traces vs density |
| T2 empty-statistic fixed point | the freeze (measured n=15 cluster + local dose-response); SS boundary test |
| T3 | optimum-preservation lemmas in §3 of the revelation doc |

## Task constants (from env source, for normalization & figures)

| Env (our configs) | best score (both arms) | episode length |
|---|---|---|
| MysteryPath-Grid | 1.0 | ≤128 (timeout); optimal ≈ path length ~10–20 |
| MortarMayhem-Grid (cc=4) | **1.0** (sparse 1.0 terminal; aligned 4×0.25) | `max_episode_steps=47`; successful eps run the full schedule ≈47 (show 4×(3+1)=16 + exec 4×~8); first-command failure ends ≈22 → **ep_len is itself a progress signal** |
| SearingSpotlights | **1.25** (coin 0.25 + exit 1.0) | ≤256 (timeout); death after 5 lit steps (health=5, damage=1); optimal ≈30–60 (84px arena, 3px/step, spawn-dependent) |

## Citations to attach (T1–T3 are propositions over these lemmas)
TBPTT truncation (Williams & Peng 1990); TD propagation; aliasing/memoryless suboptimality
(Singh et al. 1994; Littman 1994); PBRS necessity + Φ=V* (Ng et al. 1999); hackability
(Skalse et al. 2022); BAMDP shaping umbrella (Yang et al. 2024) — we instantiate it for the
density×revelation×memory axis; Mon-MDPs (Parisi et al. 2024) as the reward-channel
observability cousin.
