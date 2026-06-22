# The reward-machine method — how the code works, how we applied it, and how we got to
# "exploration selects the reward machine a memory agent learns"

*A technical handoff for Bridge. Companion to `docs/paper_full.md` (the paper) and
`docs/preregistration.md` Appendix A (the re-filed pre-registration).*

---

## 0. The idea in one paragraph

A **reward machine** (RM) is the task's reward viewed as the *minimal* finite automaton
needed to predict all future reward — formally the Myhill–Nerode quotient of the reward
function (Toro Icarte et al. 2018). Two histories are the *same RM state* iff they yield
identical future reward under every future action; so the minimal RM is exactly **the
minimal memory the reward demands**. The twist we exploit: a recurrent agent does not
converge to the RM of the *specified* task — it converges to the RM of the reward it
**realizes under its own policy** (call it `R(π)`, vs the specified `R*`). Because the
policy's exploration decides which transitions ever get visited, exploration decides
which realized automaton exists to be learned. Everything below is the machinery that
turns that sentence into measured, falsifiable numbers.

---

## 1. Where the RM state comes from (the envs)

For the **register tasks** we instrument the env to emit the *exact* minimal-RM state in
the `info` dict, so we never have to estimate it.

- **`memrl/envs/tiny_reproduce.py`** (k-token reproduce; watch k tokens, then replay):
  - `reset()` → `info["rm_state"] = tuple(self._seq[:self._shown])` (line ~85).
  - `step()` watch phase → `info["rm_state"] = tuple(self._seq[:self._shown])` (~98);
    play phase → `info["rm_state"] = self._remaining()` (~118), the remaining-to-reproduce
    suffix. **This *is* the minimal RM state**: two histories are reward-equivalent iff
    they have the same remaining suffix.
- **`memrl/envs/autoencode_toggle.py`** — the same idea on POPGym-Autoencode (sister
  register env), `info["rm_state"]` = remaining suffix.
- **MysteryPath-Grid** has *no* emitted RM state (it's an embodied hidden-path task), so
  we use the object the theory says **is** its RM state: the agent's confirmed
  path/off-path **knowledge grid** (which tiles it has stepped on and learned are
  path/off-path). See §2.2.

Key point for Bridge: on the register tasks the RM state is ground-truth and exact; on
MysteryPath it's a behavioral *proxy* built from the agent's own trajectory. Both are
read from what the agent **does**, never from its hidden activations.

---

## 2. The instrument: behavioral effective-RM-size (`memrl/probes/rm_coverage.py`)

This is the central measurable. It answers: **how many distinct realized-RM states does
the policy actually drive through?** Freeze → 1 (the do-nothing absorbing state); a
working/rescued policy → many.

### 2.1 How `rollout_coverage()` works
1. Load a checkpoint via `build_policy_from_snapshot()` (reused from `decode_memory.py`).
2. Roll the agent's **own deterministic (argmax) policy** on-policy — *not* teacher-forced,
   *not* a random bank. This is the crucial design choice (see §4): we measure what the
   policy does, not what's decodable.
3. Per episode, accumulate a `set` of realized RM states:
   - register envs: `tuple(info["rm_state"])` after each step (exact);
   - MysteryPath: the bytes of the running confirmed path/off-path **knowledge grid**
     (`knowledge.tobytes()`), updated from the agent's (x,y) and the hidden path.
4. `eff_rm_size` for the episode = `len(that set)`; we also record `idle_frac` (fraction
   of the no-op/stay action) and `success`. Aggregate means over ~50–100 episodes.

### 2.2 Why the MysteryPath proxy is legitimate
`theory_v2_memory_training.md` (P5) shows the confirmed-knowledge grid *is* the
MysteryPath minimal-RM state — two histories are reward-equivalent iff they've confirmed
the same path/off-path tiles. A frozen agent that never moves confirms exactly one
configuration (the start) → eff-RM-size = 1. An exploring agent confirms many → it grows.

### 2.3 Run it
```bash
python -m memrl.probes.rm_coverage --run-dir <dir with snapshot_step*.pt> \
    --n-episodes 100 --idle-action 0
# → one JSON line per snapshot: {step, eff_rm_size, idle_frac, success, ...}
```
On the real MysteryPath 20M GatedDeltaNet snapshots this gave: **penalty-none = 1.00,
penalty-e3b = 13.14, sparse-none = 6.98** — the collapse and the re-inflation as one
number. On Tiny it validated against the exact RM (separates solving LSTM 18.0 / GRU 17.7
from under-tuned GDN 13.9 by realized coverage).

---

## 3. The optimality proof: `memrl/probes/optimal_prober.py` (kill-criterion K4)

The freeze is only a *pathology* if doing nothing is **strictly sub-optimal** under the
specified reward at −0.008 (otherwise it's just rational play and the whole reframe is
vacuous — that's kill-criterion K4). We prove it with an exact DP on a tractable
corridor-POMDP abstraction of a hidden-path task.

- `solve(L, b, fall_penalty)`: a length-`L` corridor, `b` candidate moves per stage, one
  on-path; a wrong move = a fall (`fall_penalty`, eliminate that branch), a right move
  advances; reaching stage L pays the goal. Exact value iteration over states
  `(stage, #untried branches)`; "do nothing" = stay forever = return 0.
- Returns `V*` (optimal return), `E[falls|π*]`, the P1 bound `ε = |penalty|·E[falls]`, and
  whether freezing is optimal.
- `freeze_threshold(L, b)`: bisects for the penalty `p̄` at which freezing first becomes
  optimal.

Result (L=20, b=4, MysteryPath-like): **at −0.008, V\* = 0.76 > 0 = do-nothing → freezing
is strictly sub-optimal**; E[falls|π*] = 30, ε = 0.24; `p̄ = −0.033` (the penalty would
have to be ~4× harsher for freezing to be optimal). Robust across maze sizes.
```bash
python -m memrl.probes.optimal_prober --L 20 --b 4 --fall-penalty -0.008
```

---

## 4. The probe we RETIRED — and why it matters (`memrl/probes/decode_memory.py`)

We first tried to measure memory the way the field usually does: **decode** the RM state
from the network's frozen hidden vector (a linear/MLP probe, with a shuffled-label floor,
obs-only leakage baseline, play-phase-only forced retention — it's a *hardened* probe).
It died on a control we ran ourselves:

> **A random-initialized, untrained network decodes nearly as much as a fully trained one**
> (GRU random 7.03 ≈ trained 7.04 bits; GTrXL random 9.4 ≈ trained 9.8).

Because the probe is **teacher-forced** (the env is driven with the correct tokens), the
hidden state is just a delay-line for the inputs we fed it, so decodability measures
*copy-capacity*, not learned memory. This is also why "decodability ≠ use" / "realize
before use" is a dead end for us: it's the verbatim NLP-probing result (Hewitt–Liang
2019; Elazar *Amnesic Probing* 2021, who did it causally; Belinkov 2022), and our
correlational version is weaker. **So we keep `decode_memory.py` only as a cautionary
methods result + the random-init floor, and we measure the RM behaviorally instead
(`rm_coverage.py`).** This is the single most important methodological lesson to carry
forward: for recurrent RL memory, measure what the *policy realizes*, not what's
*decodable* from a frozen vector.

---

## 5. How we applied it (the analysis pipeline)

1. **Train the n=5 grid** (`experiments/memory_training/registry.py`, MysteryPath arm:
   {sparse, penalty(−0.008), aligned} × {none, e3b, pbim, noveld} × 6 cells × 5 seeds),
   with snapshots on (`--snapshot-steps … --snapshot-to-wandb`). Runs to `memrl-memtrain-mpg`.
2. **Read success + falls** off wandb per (cell, density, bonus): `eval/success_rate` and
   `rollout/ep_num_fails_mean`. The freeze = success 0 **and** falls 0 (absorbing
   do-nothing, not failing).
3. **Pull a few snapshots** (penalty-none, penalty-e3b, sparse-none) as wandb artifacts
   and run `rm_coverage.py` on them → the exact eff-RM-size (1.00 → 13.14).
4. **Run the DP** (`optimal_prober.py`) for the K4 optimality gap.
5. **Adjudicate the keystone**: compare penalty-{none, e3b, pbim} — does the *potential*
   (PBIM) re-inflate? `ρ = (pbim − none)/(e3b − none)`.

Everything routes through one grid + two offline instruments. No new training is needed
for the headline beyond the grid.

---

## 6. The reasoning chain to the conclusion

This is the logic that gets us from the measurements to "exploration selects the reward
machine a memory agent learns":

1. **Freeze = RM collapse.** Under the faithful −0.008 penalty, every cell converges to
   success 0 **and** falls 0 (6/6, n=5), while the return-matched sparse twin stays alive.
   The behavioral instrument reads **eff-RM-size = 1**. The DP proves doing nothing is
   *strictly sub-optimal* under the specified reward (V\*=0.76>0). ⇒ the agent has learned
   the correct minimal RM of its **realized** reward, which the penalty collapsed to one
   state. It is not failing the task; it learned a degenerate-but-correct automaton.

2. **A non-potential bonus re-inflates the RM.** Adding E3B/NovelD rewrites the realized
   reward so its minimal RM is non-trivial again; eff-RM-size goes **1 → 13**, success is
   rescued (.24–.62, 6/6). Exploration *re-grew* the automaton.

3. **A potential bonus provably cannot (the keystone).** Deliver the *same* E3B signal as
   a policy-invariant potential (PBIM, `F = γΦ(s′) − Φ(s)`, Φ = V_int). A potential is a
   **Moore** object (one value per state; telescopes to zero return), so it cannot relabel
   a **transition-keyed** (Mealy) reward → it cannot move `R(π)`. Empirically PBIM is
   *identical to no bonus*: penalty-PBIM = success 0 / falls 0, still frozen, **ρ = 0**.
   This is the *structural* reason "non-potential > potential" that a value-function
   calculus does not give, and it's what makes the claim a theorem, not a coincidence.

4. **Both are conjunctively necessary.** Coverage without a cell that can instantiate >1
   RM state buys nothing: Memoryless+E3B has the *highest* falls (22.3) yet 0 success —
   maximal coverage, no automaton realized.

Put together: the penalty, the bonus, or the *kind* of bonus each select **which reward
machine the agent ends up having learned**, measured directly as the realized-RM size it
covers. Exploration is not just tuning the rate of convergence — it chooses the target.

---

## 7. Files, at a glance

| file | role |
|---|---|
| `memrl/envs/tiny_reproduce.py`, `autoencode_toggle.py` | emit exact `info["rm_state"]` |
| `memrl/probes/rm_coverage.py` | **the instrument** — on-policy behavioral eff-RM-size + idle_frac |
| `memrl/probes/optimal_prober.py` | the corridor-POMDP DP → K4 optimality gap |
| `memrl/probes/decode_memory.py` | the **retired** decoding probe (cautionary only; random-init floor) |
| `experiments/memory_training/registry.py` | the n=5 grid + the E15 GDN sweep |
| `docs/paper_full.md` | the full paper draft built on all of the above |
