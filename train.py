"""Train a recurrent memory cell on a POPGym task (or any registered gym env).

Usage:
    python train.py --config benchmarks/phase_a/popgym_repeat_previous_easy.yaml
    python train.py --config <path> --seed 1 --total-timesteps 500_000

    # Resume a run that was paused (Ctrl+C or STOP file dropped):
    python train.py --config <path> --resume-from runs/.../seed_0_20260523_180000

Each run writes to `runs/<benchmark>/<cell>/seed_<n>_<timestamp>/`:
    - config.yaml      — exact config used
    - git_hash.txt     — commit hash at run start
    - env.txt          — pip freeze output
    - tensorboard/     — TB logs (loss, grad norms, state norms, innovation)
    - latest.pt        — resumable checkpoint (policy + optimizer + cell state
                         + RNG + step counter). Rewritten every
                         --save-freq-steps; also on SIGINT or STOP file.
    - DONE             — marker file written when training completes normally.
    - final_model.zip  — SB3 checkpoint at run end.

Pause & resume:
    - Ctrl+C once → callback saves latest.pt, exits cleanly with code 0.
      Second Ctrl+C force-quits.
    - `touch <run_dir>/STOP`  (or whatever --stop-file you pass) → same effect.
    - To continue, run the same command with --resume-from <run_dir>.
      The ablation runner does this discovery automatically.

Perf knobs (set via env var so they apply to subprocesses too):
    MEMRL_TF32=0              disable TF32 matmul (3070 Ti is Ampere → default on)
    MEMRL_CUDNN_BENCHMARK=0   disable cudnn algorithm autotuning
    MEMRL_TORCH_THREADS=N     cap intra-op CPU threads (default 4; runner sets
                              max(1, 16 // parallel) so concurrent workers don't
                              oversubscribe the i9-11900K)

YAML schema:
    env_name:          string — gym env id, e.g. "popgym-RepeatPreviousEasy-v0"
    n_envs:            int    — number of parallel envs
    total_timesteps:   int    — total env steps to train for
    seed:              int    — base seed (env i seeded with seed + i)

    cell:
        name:          string — one of CELL_REGISTRY keys (GRU, LSTM, …)
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

import numpy as np
import yaml
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from memrl.cell import (
    DTHLMU,
    GRU,
    LMU,
    LSTM,
    SHM,
    DeltaNet,
    GatedDeltaNet,
    GatedLMU,
    LinearTransformer,
    Mamba2,
    MultiLayerGatedDeltaNet,
    RecurrentCell,
    RetNet,
    S4D,
    SelectiveLMU,
    mLSTM,
)
from memrl.envs import make_vec_env
from memrl.exploration import make_intrinsic
from memrl.ppo import MemPPO
from memrl.utils import (
    ResumableCheckpointCallback,
    apply_perf_defaults,
    load_checkpoint,
    maybe_compile_policy,
    resolve_device,
)


# Cell registry — keep in sync with `memrl.cell.__init__.__all__`.
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
    "GatedLMU": GatedLMU,
    "SelectiveLMU": SelectiveLMU,
    "DTHLMU": DTHLMU,
    "GatedDeltaNet": GatedDeltaNet,
    "MultiLayerGatedDeltaNet": MultiLayerGatedDeltaNet,
    "SHM": SHM,
}

# Per-cell default kwargs used when --cell is overridden via CLI.
# Keep these in sync with the cell signatures.
# `hidden_size` is set from cfg["encoder_dim"] automatically.
DEFAULT_CELL_KWARGS: dict[str, dict[str, Any]] = {
    "GRU":               {},
    "LSTM":              {},
    "LMU":               {"memory_size": 32, "theta": 64.0},
    "LinearTransformer": {"n_heads": 4},
    "S4D":               {"d_state": 64},
    "Mamba2":            {"n_heads": 4, "d_state": 64},
    "DeltaNet":          {"n_heads": 4},
    "RetNet":            {"n_heads": 4},
    "mLSTM":             {"n_heads": 4},
    "GatedLMU":          {"memory_size": 32, "theta": 100.0, "gate_type": "softsign_sum"},
    "SelectiveLMU":      {"memory_size": 32, "theta": 100.0, "gate_type": "softsign_sum",
                          "n_scales": 3, "scale_factor": 2.0, "readout_skip_scale": 0.1},
    "DTHLMU":            {"memory_size": 32, "theta": 100.0, "n_scales": 3,
                          "scale_factor": 2.0, "assoc_size": 64,
                          "hebbian_mode": "gated_delta_eps"},
    "GatedDeltaNet":     {"assoc_size": 64},   # single-layer, RL convention
    "MultiLayerGatedDeltaNet": {"n_layers": 2, "assoc_size": 64},
    "SHM":               {"L": 128},   # paper default for easy POPGym tasks
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


def _flat_obs_dim(env) -> int:
    """Discover the flattened observation dimension by reading the obs space.

    Handles Discrete / MultiDiscrete / Box. Used to size RND / E3B / etc.'s
    encoder networks.
    """
    import gymnasium as gym
    sp = env.observation_space
    if isinstance(sp, gym.spaces.Discrete):
        return 1                                 # represented as scalar int
    if isinstance(sp, gym.spaces.MultiDiscrete):
        return int(len(sp.nvec))
    if isinstance(sp, gym.spaces.Box):
        return int(np.prod(sp.shape))
    raise ValueError(f"Unsupported obs space for intrinsic encoder: {sp}")


def _n_actions(env) -> int:
    """Discover Discrete action vocab size (ICM needs this)."""
    import gymnasium as gym
    sp = env.action_space
    if isinstance(sp, gym.spaces.Discrete):
        return int(sp.n)
    if isinstance(sp, gym.spaces.MultiDiscrete):
        return int(np.prod(sp.nvec))
    return 0    # continuous — ICM not supported


def _build_intrinsic_module(cfg: dict, env):
    """Instantiate the configured exploration module.

    YAML schema:
        intrinsic: "none" | "rnd" | "e3b_rand" | "e3b_obs" | "e3b_innov" | "noveld" | "icm"
        intrinsic_kwargs: {...}   # forwarded to the module ctor (lambda_reg, hidden_dim, etc.)

    Returns None if "none" or unspecified.
    """
    name = cfg.get("intrinsic", "none")
    if name in (None, "none", "None"):
        return None
    kwargs = dict(cfg.get("intrinsic_kwargs", {}))
    return make_intrinsic(
        name,
        n_envs=cfg["n_envs"],
        obs_dim=_flat_obs_dim(env),
        n_actions=_n_actions(env),
        device="cpu",       # MemPPO moves it as needed via .to() on its modules
        **kwargs,
    )


def make_run_dir(
    config_path: str,
    cfg: dict,
    runs_dir: str,
    resume_from: Path | None = None,
) -> Path:
    """Create or reuse runs/<benchmark>/<cell>/seed_<n>_<timestamp>/.

    When ``resume_from`` is given, that directory is reused as-is — no new
    timestamped dir, no overwrite of the snapshotted config (the saved
    config is the source of truth for the resumed run).
    """
    if resume_from is not None:
        run_dir = Path(resume_from)
        if not run_dir.exists():
            raise FileNotFoundError(f"resume-from dir not found: {run_dir}")
        return run_dir

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help="override config total_timesteps",
    )
    parser.add_argument(
        "--cell", default=None,
        help="override config cell.name (must be a key in CELL_REGISTRY)",
    )
    parser.add_argument(
        "--theta", type=float, default=None,
        help="override LMU/GatedLMU theta hyperparameter (no effect on other cells)",
    )
    parser.add_argument("--runs-dir", default="runs", help="root dir for run outputs")
    parser.add_argument(
        "--resume-from", default=None,
        help="reuse this existing run dir; load latest.pt and continue training "
             "until total_timesteps (no new timestamped dir is created)",
    )
    parser.add_argument(
        "--save-freq-steps", type=int, default=50_000,
        help="env-step interval between resumable-checkpoint saves "
             "(default 50_000 ≈ <1 min of work on a 3070 Ti)",
    )
    parser.add_argument(
        "--stop-file", default=None,
        help="additional path the trainer polls for a graceful-stop signal. "
             "Always also polls <run_dir>/STOP. Touch any of these files to "
             "trigger a final checkpoint + clean exit.",
    )
    parser.add_argument(
        "--device", default=None,
        help="device override: 'cuda', 'mps', 'cpu', or 'auto'. Default "
             "honors MEMRL_DEVICE env var, then falls back to 'auto' which "
             "picks cuda → mps → cpu. (SB3's built-in 'auto' skips MPS, so "
             "use this on Macs.)",
    )
    args = parser.parse_args()

    # Apply GPU/CPU perf defaults (TF32, cudnn benchmark, thread cap). These
    # are no-ops if the corresponding MEMRL_* env vars disable them.
    perf = apply_perf_defaults()
    device = resolve_device(args.device)
    print(f"  perf: device={device} ({perf['device_name']}) "
          f"tf32={perf['tf32']} cudnn_benchmark={perf['cudnn_benchmark']} "
          f"torch_threads={perf['torch_threads']}")

    resume = args.resume_from is not None
    if resume:
        # When resuming, the snapshotted config inside the run dir is the
        # source of truth. CLI overrides for --seed/--cell/--theta are
        # ignored to keep the resumed run faithful; --total-timesteps still
        # works because the user might want to extend a run.
        resume_dir = Path(args.resume_from)
        cfg_path_to_load = resume_dir / "config.yaml"
        if not cfg_path_to_load.exists():
            raise FileNotFoundError(
                f"resume-from dir has no config.yaml: {cfg_path_to_load}"
            )
        with open(cfg_path_to_load) as f:
            cfg: dict[str, Any] = yaml.safe_load(f)
        if args.total_timesteps is not None:
            cfg["total_timesteps"] = args.total_timesteps
        if any(x is not None for x in (args.seed, args.cell, args.theta)):
            print("  [resume] ignoring --seed/--cell/--theta — using saved config.")
    else:
        with open(args.config) as f:
            cfg: dict[str, Any] = yaml.safe_load(f)

        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.total_timesteps is not None:
            cfg["total_timesteps"] = args.total_timesteps
        if args.cell is not None:
            if args.cell not in CELL_REGISTRY:
                raise ValueError(
                    f"Unknown --cell {args.cell!r}. Available: {sorted(CELL_REGISTRY)}"
                )
            cfg["cell"]["name"] = args.cell
            # When the user overrides the cell, replace the cell.kwargs with this
            # cell's defaults — the YAML's kwargs are for the YAML's cell, not
            # the override.
            cfg["cell"]["kwargs"] = DEFAULT_CELL_KWARGS[args.cell].copy()
        if args.theta is not None:
            # Apply theta override (LMU/GatedLMU). Silently no-op for cells
            # that don't accept theta in their kwargs.
            if "theta" in cfg["cell"].get("kwargs", {}):
                cfg["cell"]["kwargs"]["theta"] = args.theta
            else:
                # Cell doesn't take theta — warn but don't crash so the same
                # launcher script can be reused across cells.
                print(f"  [warn] --theta {args.theta} ignored: {cfg['cell']['name']} doesn't accept theta")

    run_dir = make_run_dir(
        args.config, cfg, args.runs_dir,
        resume_from=Path(args.resume_from) if resume else None,
    )
    print(f"Run dir: {run_dir}{' (resume)' if resume else ''}")

    env = make_vec_env(
        env_name=cfg["env_name"],
        n_envs=cfg["n_envs"],
        seed=cfg["seed"],
    )

    factory = make_cell_factory(
        cell_name=cfg["cell"]["name"],
        cell_kwargs=cfg["cell"].get("kwargs", {}),
        hidden_size=cfg["encoder_dim"],   # cell hidden_size == encoder_dim by default
    )

    # Optional linear LR decay (Engstrom et al. 2020; standard PPO trick).
    # SB3 PPO accepts a Schedule (callable: progress_remaining → lr).
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
        lambda_intrinsic=cfg.get("lambda_intrinsic", 0.0),
        intrinsic_module=_build_intrinsic_module(cfg, env),
        intrinsic_source=cfg.get("intrinsic_source", "eps_mem"),
        encoder_dim=cfg["encoder_dim"],
        encoder_hidden=cfg.get("encoder_hidden", 128),
        shared_backbones=cfg.get("shared_backbones", False),   # default = Option A
        chunk_len=cfg["chunk_len"],
        n_chunks_per_batch=cfg["n_chunks_per_batch"],
        tensorboard_log=str(run_dir),
        verbose=1,
        seed=cfg["seed"],
        device=device,
    )

    # Note: torch.compile (MEMRL_COMPILE=1) is applied AFTER load_checkpoint
    # below, not here. Compiled wrappers add an `_orig_mod.` prefix to state
    # dict keys; loading an un-compiled checkpoint into a pre-compiled
    # policy would mismatch. Loading first, then compiling, keeps both
    # cold-start and resume paths working.

    # EvalCallback: periodic deterministic-policy eval on a held-out env.
    # eval_freq is in vec-env steps (one rollout = n_steps vec steps), so this
    # runs eval once per ~`cfg["eval_every_rollouts"]` rollouts. Default: every
    # rollout. Eval env is seeded differently from training so we measure
    # generalization, not memorization.
    eval_env = make_vec_env(
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

    # Resumable checkpoint: writes <run_dir>/latest.pt every save_freq_steps,
    # and on SIGINT / <run_dir>/STOP (or any --stop-file) flushes a final
    # save and signals SB3 to unwind cleanly.
    extra_stop = [args.stop_file] if args.stop_file else []
    ckpt_cb = ResumableCheckpointCallback(
        save_path=run_dir,
        save_freq_steps=args.save_freq_steps,
        extra_stop_files=extra_stop,
        verbose=1,
    )

    callbacks = CallbackList([eval_cb, ckpt_cb])

    if resume:
        loaded_steps = load_checkpoint(run_dir, model)
        # Compile AFTER load so state-dict keys match the un-compiled save.
        maybe_compile_policy(model.policy)
        if loaded_steps >= cfg["total_timesteps"]:
            print(f"  [resume] loaded @ step {loaded_steps:,}; already at/past "
                  f"total_timesteps ({cfg['total_timesteps']:,}). Marking DONE.")
            (run_dir / "DONE").touch()
            return 0
        # Reset envs + cell state on resume.
        #
        # Why: we faithfully restore the LEARNING state (policy weights,
        # optimizer moments, num_timesteps, RNG) so PPO continues improving
        # from where it left off. We do NOT try to resume in-progress
        # episodes — VecEnv child envs can't be snapshotted across
        # processes, and the SB3 Monitor wrapper enforces reset-before-step
        # anyway. So resume creates a clean episode boundary in each env;
        # cell state zeros out and the next rollout rebuilds context. For
        # PPO over POPGym (episodes ≤1200 steps, rollouts of 8192) this
        # costs at most one in-progress episode of throughput per pause —
        # negligible vs. a multi-hour run.
        model._last_obs            = model.env.reset()
        model._last_episode_starts = np.ones((model.env.num_envs,), dtype=bool)
        model._cell_state          = model.policy.initial_state(
            model.env.num_envs, model.device,
        )
        print(f"  [resume] loaded @ step {loaded_steps:,}; envs + cell state "
              f"reset (clean episode boundary).")

        # Train the REMAINING steps. SB3 with reset_num_timesteps=False
        # interprets total_timesteps as a delta added to num_timesteps —
        # so passing (target - loaded) gets us the right stop condition.
        remaining = cfg["total_timesteps"] - loaded_steps
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            reset_num_timesteps=False,
        )
    else:
        # Fresh start: compile (if enabled) before learn(). The first
        # iteration will be slow as Inductor traces and codegens; subsequent
        # ones reuse the cached kernels.
        maybe_compile_policy(model.policy)
        model.learn(total_timesteps=cfg["total_timesteps"], callback=callbacks)

    # If we exited because of a pause signal, don't overwrite the meaningful
    # final_model save — `latest.pt` is the resumable artifact. Exit code 0
    # signals "paused cleanly, resume me" to the runner.
    if ckpt_cb._stop_requested:
        print(f"  [paused] checkpoint saved to {run_dir / 'latest.pt'}. "
              f"Re-run with --resume-from {run_dir} to continue.")
        return 0

    model.save(str(run_dir / "final_model"))
    print(f"Done. Final model saved to {run_dir / 'final_model.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
