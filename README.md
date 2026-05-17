# memrl

A library of recurrent / SSM memory cells and partially-observable RL environment wrappers, designed around a common cell interface and a TBPTT-aware PPO trainer.

This is the **`main` branch — the stable library**. Active research lives on separate branches:

- [`igam`](../../tree/igam) — the Innovation-Gated Associative Memory thesis: headline cell (Gated DeltaNet), full 2×2 ablation matrix, benchmark configs, and design notes.
- [`gated-lmu`](../../tree/gated-lmu) — Gated LMU and Selective LMU variants from the lmu_ppo thesis work.

### Contribution workflow

`main` is the upstream for shared infrastructure — new baseline cells, new env wrappers, fixes to the trainer or interface. Research branches consume `main` rather than copy from it:

1. **New baseline or wrapper** → branch from `main`, PR back to `main`.
2. **Research-branch work** → commit directly on the research branch (`igam`, `gated-lmu`, ...).
3. **Pulling main into a research branch** → `git checkout <research-branch> && git merge main`. Do this periodically so the research branch inherits library improvements; small deltas mean easier conflict resolution.

Experimental cells and project-specific notes stay on their research branch and don't backport to `main`.

## Install

```bash
pip install -e .

# Optional env suites
pip install -e '.[memory-gym]'
pip install -e '.[popgym-arcade]'

# Dev (pytest + xdist)
pip install -e '.[dev]'
```

Python ≥ 3.10. Core dependencies: `torch ≥ 2.2`, `stable-baselines3 ≥ 2.0`, `gymnasium`, `popgym`.

## Supported cells

Every cell implements the [`RecurrentCell`](memrl/cell/base.py) interface — `init_state`, `step`, `forward_sequence`, `reset_state` — and is drop-in interchangeable in the trainer.

| Cell | File | Reference |
| --- | --- | --- |
| GRU | [gru.py](memrl/cell/gru.py) | Cho et al. 2014 |
| LSTM | [lstm.py](memrl/cell/lstm.py) | Hochreiter & Schmidhuber 1997 |
| LMU | [lmu.py](memrl/cell/lmu.py) | Voelker et al. 2019 |
| Linear Transformer | [linear_transformer.py](memrl/cell/linear_transformer.py) | Katharopoulos et al. 2020 |
| RetNet | [retnet.py](memrl/cell/retnet.py) | Sun et al. 2023 |
| DeltaNet | [deltanet.py](memrl/cell/deltanet.py) | Schlag 2021; Yang et al. 2024 |
| S4D | [s4d.py](memrl/cell/s4d.py) | Gu et al. 2022 |
| Mamba2 | [mamba2.py](memrl/cell/mamba2.py) | Dao & Gu 2024 |
| mLSTM | [mlstm.py](memrl/cell/mlstm.py) | Beck et al. 2024 (xLSTM) |
| SHM | [shm.py](memrl/cell/shm.py) | Le et al. 2024 (Stable Hadamard Memory) |

Test-friendly hyperparameter defaults for every cell live in [`memrl/cell/tests/conftest.py`](memrl/cell/tests/conftest.py); CLI defaults for `train.py` live in [`DEFAULT_CELL_KWARGS`](train.py).

## Supported environment suites

[`make_vec_env(env_name, n_envs, seed)`](memrl/envs/__init__.py) dispatches on the env id and returns an SB3 `VecEnv` ready for the trainer.

| Suite | Env-id pattern | Wrapper |
| --- | --- | --- |
| [POPGym](https://github.com/proroklab/popgym) | `popgym-*-v0` | [popgym_wrappers.py](memrl/envs/popgym_wrappers.py) |
| [MiniGrid](https://github.com/Farama-Foundation/Minigrid) | `MiniGrid-*` | [minigrid_wrappers.py](memrl/envs/minigrid_wrappers.py) |
| [memory-gym](https://github.com/MarcoMeter/endless-memory-gym) | `MortarMayhem*`, `MysteryPath*`, `SearingSpotlights*`, `Endless-*` | [memory_gym_wrappers.py](memrl/envs/memory_gym_wrappers.py) |
| [popgym-arcade](https://github.com/bolt-research/popgym-arcade) | `popgym-arcade-*` | [popgym_arcade_wrappers.py](memrl/envs/popgym_arcade_wrappers.py) |

Each wrapper handles the suite-specific quirks (one-hot encoding for MiniGrid's categorical image, tuple-discrete flattening for POPGym, image normalization for memory-gym, JAX-PRNG plumbing for popgym-arcade's gymnax envs) so downstream code sees a uniform `VecEnv`.

## Quickstart

```python
from memrl.cell import LSTM
from memrl.envs import make_vec_env
from memrl.ppo import MemPPO

env = make_vec_env("popgym-RepeatPreviousEasy-v0", n_envs=8, seed=0)
model = MemPPO(
    env,
    cell_factory=lambda input_size: LSTM(input_size, hidden_size=128),
    encoder_dim=64,
)
model.learn(total_timesteps=1_000_000)
```

Or via the YAML-config CLI:

```bash
python train.py --config benchmarks/phase_a/ablation/lstm_medium.yaml
python train.py --config benchmarks/phase_a/ablation/lstm_medium.yaml --cell GRU --seed 1
```

Configs live under [`benchmarks/`](benchmarks); see the YAML schema docstring in [`train.py`](train.py).

## Architecture notes

- **State as dict.** Every cell's recurrent state is a `dict[str, Tensor]` keyed by component (`h`, `c`, `W`, ...). The rollout buffer iterates components generically — no per-cell branches downstream.
- **TBPTT.** Set `chunk_len` and `n_chunks_per_batch` in the YAML; `forward_sequence` handles batched chunks with carried-over state. Episode boundaries are masked via `apply_episode_mask`.
- **Side outputs.** `step` can return a `SideOutputs` dict alongside the new state — used by downstream training signals (e.g., per-step innovation for intrinsic rewards on the research branches).
- **Encoder split.** Optional separate-backbone actor/critic (`separate_backbones=True` in `MemPPO`) — useful when a shared encoder bottlenecks one head.

## Tests

```bash
pytest                 # fast contract + smoke tier (~7s)
pytest -m slow         # per-cell synthetic-recall threshold checks (~85s)
```

The fast tier exercises shape discovery, dtype propagation, episode-start reset, `forward_sequence`-vs-manual-loop equivalence, and gradient flow across every registered cell. The slow tier runs a small Multi-Query Associative Recall (MQAR) task and asserts each cell clears a per-cell accuracy threshold.

## License

See [LICENSE](LICENSE) if present.
