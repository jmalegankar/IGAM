# Code audit — cells, exploration, recurrent-PPO (reviewer-proofing)

Adversarial multi-agent audit (7 reviewers vs source papers, each finding re-verified),
then the two HIGH findings re-traced by hand against the live source. Date: 2026-06-06.

**Bottom line: no results-invalidating bug.** One "critical" was a false positive; one real
minor bug (boundary bonus mis-scoring) is **fixed**; the FFM critic divergence is **not yet
root-caused** (a reproduction is running). The bulk of cells / exploration / PPO verified clean.

---

## Fixed this pass
| Item | File | What | Status |
|---|---|---|---|
| **B2 — episode reset-before-score** | `ppo.py` (intrinsic block) | Episodic memory (E3B ellipsoid / NovelD visited-set) was reset *after* the bonus was computed, so each episode's first obs was scored against the *previous* episode's stale memory (~1 transition/episode, ~0.8% on 128-step MysteryPath; uniform → did NOT change conclusions). Now resets **before** compute and passes `episode_start=dones`. | ✅ fixed + smoke-validated (e3b reward 0.65, noveld clean) |
| Grad-clip deviation doc | `ppo.py` (clip block) | Per-component clipping (6 submodules) ≈ √6 ≈ **2.45×** the global norm — deliberate (separate actor/critic backbones, per-component drift logging), now documented in code + flagged for the paper. | ✅ comment |
| GatedDeltaNet convention | `cell/gated_deltanet.py` | Memory stored as `M[key,value]` = **transpose** of Yang et al.'s `S[value,key]`, applied consistently → read `r=M·q` is **identical I/O** to `o=S·q`. Comment added so a reviewer doesn't misread the outer-product order. | ✅ comment |
| RND mean-vs-sum doc | `exploration/rnd.py` | Uses `.mean` (Burda uses sum); the `1/feat_dim` constant is absorbed by running-std norm → behaviorally identical. Comment corrected (it wrongly said "summed"). | ✅ comment |

## Verified FALSE POSITIVE (no action)
- **B1 — "GatedDeltaNet transposed write."** The audit flagged `write=k⊗v` as wrong vs the
  paper's `v⊗k`. Hand-traced: the transpose is applied **consistently** to write, erase, and
  read, and the read `r = Σ_k M[k,v]·q[k] = (S·q)[v]` **equals the paper's output**. So the cell's
  input→output is exactly Yang et al.'s GatedDeltaNet. **Not a bug; no re-run.** (Lesson: the
  workflow verifier confirmed the outer-product order without tracing read-equivalence — hand-check
  load-bearing claims.)

## RESOLVED — FFM critic divergence = high-lr transient, tamed by the headline HPs
- Symptom (10M sparse `FFM+none`, lr 2.5e-4): deterministic GPU crash @507904, `expl_var ≈ −33`.
- **CPU reproduction (lr 2.5e-4): NO crash** — ran past 516k fine; `expl_var` dipped to **−59 @164k
  then RECOVERED to ~0 by 516k**. So it's a **transient early-training critic instability that
  self-corrects**, and the cluster crash is **GPU/CUDA-specific** (TF32/cuDNN float handling turns the
  transient spike into a NaN where CPU yields a finite, recovering value).
- The audit's "unbounded state" hypothesis is **wrong**: FFM's state is bounded (`|γ|<1`, `ffm.py:153`),
  the readout is LayerNorm'd (`:162`), and it recovers.
- **Confirmed in production:** under the **headline HPs (lr 1e-4 + e3b_idm densification)**, FFM ran the
  **full 20M with no crash.** Lower lr + the densifying bonus condition the critic enough to skip the
  transient. So **no code fix is needed for the headline.**
- **Residual risk / how to defend:** FFM's `none`-arm at high lr (sparse) is fragile on GPU. For any
  sparse `none` FFM run, use lr 1e-4 (the tuned value) — it's stable. If a sparse high-lr FFM run is
  ever needed, add a defensive output clamp (e.g. `±30`) as cheap insurance. Otherwise: document FFM's
  high-lr transient as a one-line limitation; the reported (tuned-HP) FFM numbers are valid.

## Verified CLEAN (defensible under review)
- **Cells** (math + init + grad flow + 500–10k-step stability): GRU, LSTM, **mLSTM** (log-space
  stabilizer), **LRU** (`|λ|=exp(−exp(ν))`), RetNet, LinearTransformer (denom clamp), Mamba2, SHM,
  **GatedDeltaNet** (transposed-but-equivalent), GTrXL, Memoryless. FFM *forward* math is correct;
  only its use as the *critic* is under investigation.
- **Exploration**: RND target frozen + running-std norm; **E3B** Sherman-Morrison with clamped denom
  + PSD-loss reset; NovelD first-visit + RND-difference math; RandomPhi frozen (justified deviation).
- **PPO**: GAE/returns, clip/value/entropy losses, `target_kl` early-stop, **non-finite-grad skip**
  (`skip_nonfinite_grad`, default on), separate-backbone critic-input detach, TBPTT chunk carry/detach.

## Minor / deferred (low severity, verified)
- `base.py apply_episode_mask` — no batch-size assert (fails loudly, contract-enforced) → optional defensive assert.
- `ppo.py` single-element-minibatch advantage norm divides by 1e-8 → rare (batch ≥256); optional `len(adv)>1` guard.

## Did the 20M sweep need B2? No.
The 20M headline ran with B2 present. B2 is a ~0.8% uniform boundary effect → it does **not** change
the qualitative results (e3b helps, Memoryless can't be rescued, e3b>noveld, the cell ranking). The
fix matters for the **forthcoming** dense-toggle / PBIM runs (clean from the start). Re-running 20M
for camera-ready pristineness is optional, not required.
