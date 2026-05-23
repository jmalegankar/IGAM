"""Train entry point for GEX-replication experiments.

Progressive experiment sequence (5 steps):
  1. Pure PPO (no cell)          — should fail: no memory
  2. PPO + LMU                   — should fail: memory but agent can't explore hint room
  3. PPO + LMU + wrapper         — works slowly: wrapper removes exploration bottleneck
  4. PPO + GatedLMU + wrapper    — faster: gate improves pure memory
  5. PPO + GatedLMU + E3B        — full system: gate + exploration, no wrapper needed

Architectural ablation (runs on the hardest condition: no wrapper + E3B):
  full_system / no_gate / no_ortho_wpre / no_dynamic_wquery / no_residual / vanilla_lmu

Usage:
    python -m experiments.gex_replication.train --config experiments/gex_replication/configs/03_ppo_lmu_wrapper.yaml
    python -m experiments.gex_replication.train --config <path> --seed 1

YAML schema (extends base MemPPO schema with):
    use_wrapper:   bool  — apply MemoryStartWrapper (default false)
    beta_ep:       float — E3B bonus scale (default 0.0 = disabled)
    phi_source:    str   — E3B feature source (default 'random_encoder')
    lambda_reg:    float — E3B regularization λ (default 1.0)
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3.common.callbacks import EvalCallback

from memrl.cell import (
    DTHLMU,
    GRU,
    LMU,
    LSTM,
    GatedLMU,
    RecurrentCell,
    SelectiveLMU,
)
from memrl.envs import make_vec_env
from .ppo import MemPPO


CELL_REGISTRY: dict[str, type[RecurrentCell]] = {
    "GRU":        GRU,
    "LSTM":       LSTM,
    "LMU":        LMU,
    "GatedLMU":   GatedLMU,
    "SelectiveLMU": SelectiveLMU,
    "DTHLMU":     DTHLMU,
}

DEFAULT_CELL_KWARGS: dict[str, dict[str, Any]] = {
    "GRU":          {},
    "LSTM":         {},
    "LMU":          {"memory_size": 32, "theta": 64.0},
    "GatedLMU":     {"memory_size": 32, "theta": 100.0, "gate_type": "softsign_sum"},
    "SelectiveLMU": {"memory_size": 32, "theta": 100.0, "gate_type": "softsign_sum",
                     "n_scales": 3, "scale_factor": 2.0, "readout_skip_scale": 0.1},
    "DTHLMU":       {"memory_size": 32, "theta": 100.0, "n_scales": 3,
                     "scale_factor": 2.0, "assoc_size": 64,
                     "hebbian_mode": "gated_delta_eps"},
}


def make_cell_factory(cell_name: str, cell_kwargs: dict[str, Any], hidden_size: int):
    if cell_name not in CELL_REGISTRY:
        raise ValueError(f"Unknown cell {cell_name!r}. Available: {sorted(CELL_REGISTRY)}")
    cls = CELL_REGISTRY[cell_name]
    kwargs = dict(cell_kwargs)
    kwargs.setdefault("hidden_size", hidden_size)

    def factory(input_size: int) -> RecurrentCell:
        return cls(input_size=input_size, **kwargs)

    return factory


def make_run_dir(config_path: str, cfg: dict, runs_dir: str) -> Path:
    benchmark = Path(config_path).stem
    cell_name = cfg["cell"]["name"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_dir) / benchmark / cell_name / f"seed_{cfg['seed']}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"
    (run_dir / "git_hash.txt").write_text(git_hash + "\n")

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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--cell", default=None, help="override cell.name")
    parser.add_argument("--beta-ep", type=float, default=None, help="override beta_ep")
    parser.add_argument("--use-wrapper", action="store_true", default=None,
                        help="override use_wrapper to True")
    parser.add_argument("--runs-dir", default="runs/gex_replication")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = args.total_timesteps
    if args.cell is not None:
        if args.cell not in CELL_REGISTRY:
            raise ValueError(f"Unknown --cell {args.cell!r}.")
        cfg["cell"]["name"] = args.cell
        cfg["cell"]["kwargs"] = DEFAULT_CELL_KWARGS[args.cell].copy()
    if args.beta_ep is not None:
        cfg["beta_ep"] = args.beta_ep
    if args.use_wrapper:
        cfg["use_wrapper"] = True

    run_dir = make_run_dir(args.config, cfg, args.runs_dir)
    print(f"Run dir: {run_dir}")

    use_wrapper = cfg.get("use_wrapper", False)
    env = make_vec_env(
        env_name=cfg["env_name"],
        n_envs=cfg["n_envs"],
        seed=cfg["seed"],
        use_wrapper=use_wrapper,
    )

    factory = make_cell_factory(
        cell_name=cfg["cell"]["name"],
        cell_kwargs=cfg["cell"].get("kwargs", {}),
        hidden_size=cfg["encoder_dim"],
    )

    lr_cfg = cfg["lr"]
    if cfg.get("linear_lr_decay", False):
        initial_lr = float(lr_cfg)
        lr_arg = lambda progress_remaining: initial_lr * progress_remaining  # noqa: E731
    else:
        lr_arg = lr_cfg

    model = MemPPO(
        env=env,
        cell_factory=factory,
        lr=lr_arg,
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
        shared_backbones=cfg.get("shared_backbones", False),
        chunk_len=cfg["chunk_len"],
        n_chunks_per_batch=cfg["n_chunks_per_batch"],
        beta_ep=cfg.get("beta_ep", 0.0),
        phi_source=cfg.get("phi_source", "random_encoder"),
        lambda_reg=cfg.get("lambda_reg", 1.0),
        tensorboard_log=str(run_dir),
        verbose=1,
        seed=cfg["seed"],
    )

    eval_env = make_vec_env(
        env_name=cfg["env_name"],
        n_envs=1,
        seed=cfg["seed"] + 10_000,
        use_wrapper=use_wrapper,
    )
    eval_freq = cfg["n_steps"] * cfg.get("eval_every_rollouts", 1)
    eval_cb = EvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=cfg.get("n_eval_episodes", 20),
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
