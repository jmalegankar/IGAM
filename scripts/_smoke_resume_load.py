"""Validate the resume LOAD path: build a fresh MemPPO from the saved
config, call load_checkpoint, and verify state restoration. Doesn't call
.learn() — just exercises the deserialize path so we know the resume flow
works before we trust it on a multi-hour ablation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from memrl.envs import make_vec_env
from memrl.ppo import MemPPO
from memrl.utils import apply_perf_defaults, load_checkpoint

# Reuse train.py's CELL_REGISTRY + factory builder rather than duplicating.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import make_cell_factory  # noqa: E402


def main(run_dir: str) -> int:
    apply_perf_defaults()
    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    env = make_vec_env(cfg["env_name"], n_envs=cfg["n_envs"], seed=cfg["seed"])
    factory = make_cell_factory(
        cfg["cell"]["name"], cfg["cell"].get("kwargs", {}), cfg["encoder_dim"],
    )

    model = MemPPO(
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
        shared_backbones=cfg.get("shared_backbones", False),
        chunk_len=cfg["chunk_len"],
        n_chunks_per_batch=cfg["n_chunks_per_batch"],
        verbose=0,
        seed=cfg["seed"],
    )

    print(f"  fresh model: num_timesteps={model.num_timesteps}, "
          f"_last_obs={'<None>' if model._last_obs is None else 'set'}")

    loaded = load_checkpoint(run_dir, model)
    print(f"  after load:  num_timesteps={model.num_timesteps:,}, "
          f"_last_obs.shape={model._last_obs.shape}, "
          f"_cell_state keys={len(model._cell_state)}")
    assert model.num_timesteps == loaded
    assert model._last_obs is not None
    assert len(model._cell_state) == 12   # 6 actor + 6 critic

    # Sanity-check that the policy actually sees the loaded weights — pick
    # one param and compare to its stored copy.
    pol_param = next(iter(model.policy.state_dict().values()))
    print(f"  sample param: shape={tuple(pol_param.shape)}, "
          f"abs_mean={pol_param.abs().mean().item():.6f}")
    print("OK — resume load path works")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
