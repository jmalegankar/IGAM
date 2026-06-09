# Handoff to fable-5 — Memory × Exploration paper, B2 experiment build

*Session handoff. Written to stand alone — you can pick up cold.*

## 0. The paper this serves (context)
We're writing an AAAI paper (IGAM project, branch `gated-lmu`). **Central thesis:**
memory-requirement and reward-density are *orthogonal* axes, and **exploration bonuses
(RND / NovelD / E3B) are reward *densifiers*, not exploration aids** — they help only when
reward is *sparse*, and are neutral-to-harmful when reward is *dense*. Mechanism: (a)
densification (the bonus gives the critic a learnable signal when extrinsic reward is sparse)
+ (b) policy bias (a non-potential-based intrinsic reward shifts the optimal policy — PBIM).

Prior sessions established: the clean *entangled* env (MysteryPath-Grid: invisible-path memory
task), an 11-cell + Memoryless zoo, a 20M × 5-seed headline sweep, and tuned HPs
(`e3b_idm`, λ=0.03, chunk_len=64, lr=1e-4).

## 1. What this session was for
Build and harden **B2 — the dense/sparse toggle**, *the gating experiment* that decides whether
the paper is publishable. "Exploration helps sparse memory tasks" is trivial; the novel claim is
that the **same bonus flips sign** when you make the **same memory task** dense. B2 tests exactly
that — toggle reward density on one memory task, hold memory fixed, measure whether e3b's benefit
inverts.

## 2. What we did, and why

### (a) Cleared the FFM blocker
FFM's critic had crashed deterministically at step 507904 on the cluster (`explained_variance≈−33`).
Reproduced on local CPU: **no crash** (ran past 516k), `expl_var` dipped to −59 early then
**recovered to ~0**. So it's a *transient* early-training instability that's GPU/CUDA-specific in
its terminal (NaN) form — **not a cell bug**. Confirmed in production: under the headline HPs
(lr 1e-4 + e3b densification) FFM ran the full 20M clean. **No code fix needed for the headline.**
(See `docs/code_audit.md`.)

### (b) AAAI readiness call
Current results = the *trivial half* + a strong mechanistic seed → borderline/reject as-is.
Path to **accept**:
1. **Reframe** to the counterintuitive headline — "bonuses are densifiers, harmful on dense memory."
2. **Add a theory proposition** — net bonus value > 0 iff sparsity exceeds a λ/horizon threshold;
   the potential-based form of the bonus is policy-invariant (Ng 1999).
3. **Make B3 a *causal* control** — swap e3b for its potential-based form → the harm vanishes, the
   sparse benefit stays → isolates *bias* from *exploration*.
4. **Unify the cell zoo with the thesis** — stronger memory cells benefit less from densification.
5. **Generality** across ≥2 envs + a practitioner decision rule.

**B2 gets us in the room; B3 + theory get the accept.**

### (c) Built B2
`experiments/mysterypath_densetoggle/` — **{GRU, RetNet, GatedDeltaNet} × {sparse, dense} ×
{none, e3b_idm} × 5 seeds = 60 runs**, locked to the winning HPs, 10M budget. Plus k8s launcher
`k8s/launch-mpgdt-jobs.sh` + template `k8s/mpgdt-job-template.yaml` (app=`memrl-mpgdt`, **no PVC** —
wandb is the system of record).

**Only 3 cells (not the full 12) on purpose:** B2 is a cheap *gate*. Three architecturally-diverse
cells (gated RNN / linear-attention / matrix-state SSM) suffice to answer "does it flip?"; the full
zoo (4× compute, 120 jobs) is the camera-ready follow-up *after* the gate passes.

### (d) ⭐ Redefined "dense" — the key design decision
Originally dense = `+0.1` per first-visit path tile (progress reward). We **switched to a `−0.008` (=1/128, horizon-normalized)
penalty for stepping off the path** (`reward_fall_off=-0.008`) and **removed the progress reward**.

**Why:** the progress reward pushed the max return to ~1.8 (variable 1.5–2.2 — path length varies
5–12 tiles), so dense ≠ sparse in *return scale* and a reviewer could call the flip a magnitude
artifact. The penalty leaves the **optimal return at exactly 1.0, identical to sparse** (an optimal
agent never falls off) — so dense and sparse share the same achievable return *and* optimal policy;
**only the per-step feedback density differs.** Clean, return-matched disentanglement.

**Trade-off flagged:** the penalty gives *corrective* (negative) feedback, not *goal-directed*
(positive) shaping — it may densify less toward the goal, and carries a small "freeze at start" risk
(mitigated by keeping the penalty ≪ the +1.0 goal + PPO entropy). **Watch the `none`-dense return
for a collapse-to-0 freeze signature.** Counter-argument: penalty + reset-to-start gives immediate
per-step path-correctness feedback, so the agent can learn the path via RL rather than blind search.
Verified the mechanic: penalty fires on fall-off, zero progress reward.

### (e) Operational hardening
- **wandb project = `memrl-mpg-fallpenalty`** — new project, separate from the deprecated `+0.1`
  runs (which live in `memrl-mpg-densetoggle`).
- **k8s Job names tagged `-dense-fall`** (e.g. `memrl-mpgdt-s0-gru-dense-fall`). *Why it matters:*
  `kubectl apply` is declarative — applying a *same-named* Job against the old completed `+0.1` jobs
  would **no-op** (Jobs are immutable), so the new penalty runs would silently never start. The
  rename makes them distinct Job objects that run fresh and coexist with the old ones. (Done by
  tagging the dense *script filenames* `_dense_fall`; the internal density key stays `dense`, so
  configs and wandb run-names are unchanged.)
- **Eval uses extrinsic reward only (verified):** the bonus is added solely in `ppo.py:314`
  (rollout collection); eval is a stock SB3 `EvalCallback` on a separate held-out env,
  deterministic, recording only `env.step()` reward. The e3b-vs-none comparison is on clean task
  reward. (For dense, eval reward includes the penalty — compare *within* a density; consider also
  logging a goal-success rate for an unambiguous "did it finish" signal.)
- **Resource asks** (MysteryPath uses `DummyVecEnv` → ~1 core + ~1.2 GB RAM/run, GPU-bound):
  GPU `limits:1`, CPU `requests:3/limits:6`, memory `requests:6Gi/limits:32Gi`, `OMP_NUM_THREADS:2`.
  Packing is **2 runs/GPU** (none+e3b) — one small model can't saturate a GPU; 3/GPU OOM'd.

## 3. Current state
**Everything is staged; nothing committed or pushed** (the user does the final commit). Staged:
the B2 experiment (generator + 12 configs + 30 scripts), k8s launcher + template, the audit code
fixes (`ppo.py`, `gated_deltanet.py`, `rnd.py`), and the docs (`code_audit.md`,
`dense_memory_exploration.md`, `experiment_roadmap.md`, this file).

## 4. How to run B2
```bash
# Prereq — the image clones the branch at build time, so the new penalty configs +
# project only reach the cluster after a push + rebuild:
git push origin gated-lmu
DOCKER_BUILDKIT=1 docker build --build-arg GIT_REF=gated-lmu -t r0hanpat1l/memory:latest .
docker push r0hanpat1l/memory:latest

# Launch (each job runs none + e3b in parallel on one GPU):
ONLY='_dense'  k8s/launch-mpgdt-jobs.sh   # 15 dense-fall jobs  → project memrl-mpg-fallpenalty
ONLY='_sparse' k8s/launch-mpgdt-jobs.sh   # 15 sparse jobs

# Watch / clean up:
kubectl get jobs -l app=memrl-mpgdt
kubectl delete jobs -l app=memrl-mpgdt
```
**Predicted result (the flip):** `e3b − none` is *positive* on sparse, *≤ 0* on dense. If that holds
with the same optimal return (1.0) in both arms, the paper's core claim is established.

## 5. Next after B2
B3 (potential-based causal control) → cell×bonus re-analysis (≈free, reuses 20M + B2 data) →
theory proposition → B6 (expl_var vs density) → B4 (POPGym dense-memory for generality).
