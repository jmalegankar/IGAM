"""Train an IGAM cell on a POPGym task (or any registered gym env).

Usage:
    python train.py --config benchmarks/phase_a/popgym_repeat_previous_easy.yaml
    python train.py --config <path> --seed 1 --total-timesteps 500_000

Each run writes to `runs/<benchmark>/<cell>/seed_<n>_<timestamp>/`:
    - config.yaml      — exact config used
    - git_hash.txt     — commit hash at run start
    - env.txt          — pip freeze output
    - tensorboard/     — TB logs (loss, grad norms, state norms, innovation)
    - final_model.zip  — SB3 checkpoint at run end

YAML schema:
    env_name:          string — gym env id, e.g. "popgym-RepeatPreviousEasy-v0"
    n_envs:            int    — number of parallel envs
    total_timesteps:   int    — total env steps to train for
    seed:              int    — base seed (env i seeded with seed + i)

    cell:
        name:          string — one of CELL_REGISTRY keys (GRU, IGAM, …)
        kwargs:        dict   — passed to the cell's __init__

    encoder_dim:       int
    encoder_hidden:    int (optional)

    lr:                float
    n_steps:           int   — env steps per rollout per env
    n_epochs:          int   — PPO update epochs per rollout
    gamma:             float
    gae_lambda:        float
    clip_range:        float
    ent_coef:          float
    vf_coef:           float
    max_grad_norm:     float

    chunk_len:           int — TBPTT chunk length (typ. 16)
    n_chunks_per_batch:  int — chunks per PPO minibatch
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3.common.callbacks import EvalCallback

from igam.cell import (
    GRU,
    LMU,
    LSTM,
    DeltaNet,
    GatedDeltaNet,
    LinearTransformer,
    Mamba2,
    RecurrentCell,
    RetNet,
    S4D,
    mLSTM,
)
from igam.envs import make_popgym_vec_env
from igam.ppo import IGAMPPO


# Cell registry — keep in sync with `igam.cell.__init__.__all__`.
CELL_REGISTRY: dict[str, type[RecurrentCell]] = {
    "GRU": GRU,
    "LSTM": LSTM,
    "LMU": LMU,
    "LinearTransformer": LinearTransformer,
    "S4D": S4D,
    "Mamba2": Mamba2,
    "DeltaNet": DeltaNet,
    "RetNet": RetNet,
    "mLSTM": mLSTM,
    "GatedDeltaNet": GatedDeltaNet,
    # IGAM is an alias for GatedDeltaNet (the cell IS the IGAM cell).
    "IGAM": GatedDeltaNet,
}


def make_cell_factory(cell_name: str, cell_kwargs: dict[str, Any], hidden_size: int):
    """Closure that produces (input_size) -> RecurrentCell.

    `input_size` will equal the encoder's output dim (`encoder_dim`).
    `hidden_size` defaults to `input_size` if not in cell_kwargs.
    """
    if cell_name not in CELL_REGISTRY:
        raise ValueError(
            f"Unknown cell {cell_name!r}. Available: {sorted(CELL_REGISTRY)}"
        )
    cls = CELL_REGISTRY[cell_name]
    kwargs = dict(cell_kwargs)
    kwargs.setdefault("hidden_size", hidden_size)

    def factory(input_size: int) -> RecurrentCell:
        return cls(input_size=input_size, **kwargs)

    return factory


def make_run_dir(config_path: str, cfg: dict, runs_dir: str) -> Path:
    """Create runs/<benchmark>/<cell>/seed_<n>_<timestamp>/ and snapshot config."""
    benchmark = Path(config_path).stem
    cell_name = cfg["cell"]["name"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_dir) / benchmark / cell_name / f"seed_{cfg['seed']}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot effective config.
    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Git hash at run start (best effort).
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"
    (run_dir / "git_hash.txt").write_text(git_hash + "\n")

    # Pip freeze (best effort).
    try:
        env_freeze = subprocess.check_output(
            ["pip", "freeze"], stderr=subprocess.DEVNULL,
        ).decode()
    except Exception:
        env_freeze = "unknown\n"
    (run_dir / "env.txt").write_text(env_freeze)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help="override config total_timesteps",
    )
    parser.add_argument("--runs-dir", default="runs", help="root dir for run outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = args.total_timesteps

    run_dir = make_run_dir(args.config, cfg, args.runs_dir)
    print(f"Run dir: {run_dir}")

    env = make_popgym_vec_env(
        env_name=cfg["env_name"],
        n_envs=cfg["n_envs"],
        seed=cfg["seed"],
    )

    factory = make_cell_factory(
        cell_name=cfg["cell"]["name"],
        cell_kwargs=cfg["cell"].get("kwargs", {}),
        hidden_size=cfg["encoder_dim"],   # cell hidden_size == encoder_dim by default
    )

    model = IGAMPPO(
        env=env,
        cell_factory=factory,
        lr=cfg["lr"],
        n_steps=cfg["n_steps"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        clip_range=cfg.get("clip_range", 0.2),
        ent_coef=cfg.get("ent_coef", 0.01),
        vf_coef=cfg.get("vf_coef", 0.5),
        max_grad_norm=cfg.get("max_grad_norm", 0.5),
        target_kl=cfg.get("target_kl"),
        encoder_dim=cfg["encoder_dim"],
        encoder_hidden=cfg.get("encoder_hidden", 128),
        chunk_len=cfg["chunk_len"],
        n_chunks_per_batch=cfg["n_chunks_per_batch"],
        tensorboard_log=str(run_dir),
        verbose=1,
        seed=cfg["seed"],
    )

    # EvalCallback: periodic deterministic-policy eval on a held-out env.
    # eval_freq is in vec-env steps (one rollout = n_steps vec steps), so this
    # runs eval once per ~`cfg["eval_every_rollouts"]` rollouts. Default: every
    # rollout. Eval env is seeded differently from training so we measure
    # generalization, not memorization.
    eval_env = make_popgym_vec_env(
        env_name=cfg["env_name"],
        n_envs=1,
        seed=cfg["seed"] + 10_000,
    )
    eval_freq = cfg["n_steps"] * cfg.get("eval_every_rollouts", 1)
    eval_cb = EvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=cfg.get("n_eval_episodes", 10),
        log_path=str(run_dir / "eval"),
        best_model_save_path=str(run_dir / "best_model"),
        deterministic=True,
        verbose=1,
    )

    model.learn(total_timesteps=cfg["total_timesteps"], callback=eval_cb)
    model.save(str(run_dir / "final_model"))
    print(f"Done. Final model saved to {run_dir / 'final_model.zip'}")


if __name__ == "__main__":
    main()
