# S13 memory-only baseline — published cells × PPO (intrinsic=none), 3 seeds

The **x-axis** of the memory × exploration study: how far does each *published*
memory cell get on `MiniGrid-MemoryS13-v0` from **memory alone**, with NO
exploration bonus? The exploration columns (rnd / e3b_rand / noveld / icm /
e3b_obs) are a separate follow-up sweep — this batch is the baseline only.

**30 runs** = 10 cells × 3 seeds (0, 1, 2). One config YAML per cell in
`configs/` and one launch script per cell in `scripts/` (each runs its 3 seeds
in parallel); the seed is supplied at launch via `--seed`.

Cells (all published architectures — none of our own; DTHLMU/GatedLMU/SelectiveLMU
are deliberately excluded here):
**GRU, LSTM, mLSTM, LMU, LRU, Mamba2, FFM, GatedDeltaNet, SHM, GTrXL.**

## The task regime (important)

S13 here runs with MiniGrid's **random agent spawn** — the `MemoryStartWrapper`
is intentionally NOT applied (current `train.py` does not wire it). So the agent
must *both* (a) navigate back to the hint room to observe the cue **and**
(b) remember the cue across the corridor to pick the matching object at the
junction. (a) is an exploration bottleneck; (b) is the memory test. A pure-memory
baseline has to solve the exploration part on its own — the condition under
which an exploration bonus can later help.

HPs follow the thesis S13 reference, reviewed (not blindly inherited) for this
baseline. The choices that matter on S13:

- **TBPTT `chunk_len=32` × 32 chunks/batch** — raised from the thesis's 16.
  S13 is 13×13 with `max_steps=845`; the cue→decision recall span (corridor
  traversal) is ~13–25 steps, so a 16-step gradient window often straddles the
  cue-encoding and the decision and starves the credit path that teaches the
  cell to *store* the cue. 32 covers the span with margin; full-episode BPTT is
  infeasible. Hidden state carries across chunks — only the gradient truncates.
- **γ=0.999** — effective horizon ~1000 ≥ max_steps 845, so the terminal reward
  actually propagates across a full episode (γ=0.99 would not).
- **vf_coef=1.0** — with the project's separate critic backbone there's no
  policy/value gradient interference, and the sparse terminal value target
  benefits from the stronger weight.
- **ent_coef=0.008** — kept modest on purpose: entropy is the baseline's *only*
  exploration driver, so it's left low to keep the "memory-only" reading clean
  (exploration bonuses are the separate sweep's variable, not smuggled in here).
- **total_timesteps=10M** (up from the thesis's 5M) — memory-only on an
  exploration-bottlenecked task converges slowly, so 10M avoids under-training
  confounding the cell comparison (a slow cell shouldn't read as a weak one).
- **eval every 5 rollouts × 20 episodes** — ~244 evals over 10M (good curve
  resolution at ~5% overhead, vs ~1200 evals if left at every-rollout); 20
  episodes tame S13's random-spawn eval variance.
- 16 envs, encoder 128/256, GAE λ=0.98, clip 0.2, n_epochs 4, target_kl 0.05,
  lr 3e-4, max_grad_norm 0.5. Per-cell kwargs from `train.py:DEFAULT_CELL_KWARGS`
  (Mamba2 `d_state=128` for param parity, etc.).

## Environment

Use the repo's **`.venv`** — that's where `wandb` is installed and logged in.

```bash
source .venv/bin/activate          # or prefix commands with .venv/bin/python
```

On the cloud GPU box, use any env that has this repo's deps **plus wandb**
(`pip install wandb` and `wandb login` if not already).

## Run it

One launch script per cell in `scripts/`; each runs that cell's 3 seeds **at
once** (in parallel), logging to `_logs/<cell>_seed<n>.log`.

```bash
# from repo root. Defaults: PYTHON=.venv/bin/python, DEVICE=cuda, SEEDS="0 1 2".
bash experiments/s13_baseline/scripts/run_GTrXL.sh   # one cell, 3 seeds parallel
bash experiments/s13_baseline/scripts/run_all.sh     # all 10 cells (serial), seeds parallel
```

Env knobs honored by every script: `PYTHON`, `DEVICE` (e.g. `cpu`), `RUNS_DIR`,
`SEEDS="0 1"`, `PARALLEL=0` (seeds sequential — use if GPU memory is tight),
`EXTRA="--no-wandb"` (extra `train.py` flags). Example, one cell on CPU with
seeds serial:

```bash
DEVICE=cpu PARALLEL=0 bash experiments/s13_baseline/scripts/run_SHM.sh
```

Each per-cell script fires 3 concurrent `train.py` processes — on one GPU that's
3× model memory (fine for these ~1M-param cells; flip `PARALLEL=0` otherwise).
Resumable via `--resume-from <run_dir>` (continues the same wandb run).

Outputs: `runs/s13_baseline/<config_stem>/<cell>/seed_<n>_<stamp>/` with
`eval/evaluations.npz`, TensorBoard logs, and checkpoints.

## Weights & Biases

Enabled by default in every config (`wandb: true`, project
**`memrl-s13-baseline`**). `train.py` calls `wandb.init(sync_tensorboard=True)`,
which mirrors **all** TensorBoard scalars MemPPO records — train losses,
`approx_kl`, `clip_fraction`, `rollout/ep_rew_mean`, `ep_len_mean`, `time/fps`,
the `intrinsic/*` stats, and the `debug/*` diagnostics (per-action fractions,
actor/critic state-norms) — plus the full config and wandb's auto system metrics.
Runs are grouped by env, tagged by cell/intrinsic, named `<cell>-none-seed<n>`.

- disable one run: `--no-wandb`
- override target: `--wandb-project NAME` / `--wandb-entity TEAM`
- dry/local (no cloud upload): `WANDB_MODE=offline`
- if wandb isn't installed in the active interpreter, logging is skipped with a
  warning (the run still trains) — so always launch from `.venv`.

## Per-run cost (reference)

Per (cell, seed), measured on this Mac (CPU, 16 envs). **A 3070 will differ** —
bigger speedups for the heavier cells (GTrXL, SHM, Mamba2), less for the small
ones (env stepping + the recurrent scan are partly latency-bound). NB: fps were
measured at the old `chunk_len=16`; the 10M ETA = 2× the 5M figure and does not
add the modest `chunk_len=32` overhead — so treat these as lower bounds.

| cell          | params* | fps  | 10M ETA (Mac-CPU) |
|---------------|--------:|-----:|------------------:|
| LSTM          | 1.07M   | 1143 | 2.4h |
| mLSTM         | 0.94M   | 1069 | 2.6h |
| GRU           | 0.99M   | 1001 | 2.8h |
| LMU           | 0.88M   | ~1000 (est) | ~2.8h |
| LRU           | 0.99M   |  966 | 2.9h |
| FFM           | 0.94M   |  976 | 2.8h |
| GatedDeltaNet | 0.93M   |  597 | 4.7h |
| Mamba2        | 0.93M   |  462 | 6.0h |
| SHM           | 0.99M   |  402 | 6.9h |
| GTrXL         | 1.53M   |  335 | 8.3h |

\*full policy (encoder + cell + heads). Each per-cell script runs its 3 seeds
concurrently, so per-cell wall-clock ≈ the table value (a bit more under CPU
contention); `run_all.sh` (cells serial) ≈ Σ ≈ **~42h Mac-CPU**, far less on GPU.

## Analyze when done

```bash
python experiments/s13_baseline/aggregate.py --runs-dir runs/s13_baseline
```
Prints the cell × (seed-averaged) final eval reward and writes
`_results/runs_long.csv` + `_results/grid_final.csv`. (wandb also gives live
curves grouped per env.)

## Adding cells

Append to `CELLS` in `generate_configs.py` and re-run it. Other published cells
in `train.py:CELL_REGISTRY`: S4D, DeltaNet, RetNet, LinearTransformer.
(Our own cells — DTHLMU, GatedLMU, SelectiveLMU, MultiLayerGatedDeltaNet — are
intentionally omitted from this published-baseline batch.)
