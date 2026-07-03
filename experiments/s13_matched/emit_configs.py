"""Emit the reward- and metric-matched S13 grid (matched to memory-gym MysteryPath).

The problem this fixes: MiniGrid-MemoryS13's native reward is horizon-DISCOUNTED
(1 - 0.9*t/T on success, 0 otherwise), while MysteryPath's sparse arm is flat +1
(reward_step=0). Comparing amplify (MysteryPath) vs equalize (S13) across those two
reward shapes / metrics is confounded. This emits S13 under MATCHED arms via
MemoryRewardWrapper (memrl/envs/minigrid_wrappers.py), reporting binary success_rate:

  flat   : env_kwargs {agent_view_size: V, reward_mode: flat}            -> +1 / 0  (== MPG sparse)
  freeze : + {move_penalty: 1/T_max}  (unconditional per-move cost, free nop=done)  (== MPG penalty)
  native : {agent_view_size: V}       -> as-shipped 1-0.9t/T             (reference / current runs)

Grid: 6 cells x {none, e3b_idm, noveld, pbim_e3b_idm} x view {7,3} x arm {flat, freeze, native}.
Run:  python experiments/s13_matched/emit_configs.py   (writes ./configs/*.yaml)
"""
import os, yaml

OUT = os.path.join(os.path.dirname(__file__), "configs")
os.makedirs(OUT, exist_ok=True)
T_MAX = 5 * 13 ** 2  # 845 = MemoryS13 max_steps (5*size^2)

CELLS = {"GRU": {}, "LSTM": {}, "RetNet": {"n_heads": 4},
         "GatedDeltaNet": {"assoc_size": 64}, "Mamba2": {}, "Memoryless": {}}
BONUSES = ["none", "e3b_idm", "noveld", "pbim_e3b_idm"]

def base(cell, kw, intr):
    c = dict(
        env_name="MiniGrid-MemoryS13-v0", n_envs=16, total_timesteps=20_000_000, seed=0,
        cell={"name": cell, **({"kwargs": kw} if kw else {})},
        encoder_dim=128, encoder_hidden=256,
        lr=3e-4, n_steps=512, n_epochs=4, gamma=0.999, gae_lambda=0.98, clip_range=0.2,
        ent_coef=0.008, vf_coef=1.0, max_grad_norm=0.5, target_kl=0.05,
        chunk_len=32, n_chunks_per_batch=32,
        eval_every_rollouts=10, n_eval_episodes=100, wandb=False, intrinsic=intr,
    )
    if intr == "e3b_idm":
        c["lambda_intrinsic"] = 0.03; c["intrinsic_kwargs"] = {"normalize_phi": True}
    if intr == "pbim_e3b_idm":
        c["lambda_intrinsic"] = 0.03
    return c

def emit(cell, kw, intr, view, arm):
    c = base(cell, kw, intr)
    ek = {"agent_view_size": view}
    if arm in ("flat", "freeze"):
        ek["reward_mode"] = "flat"
    if arm == "freeze":
        ek["move_penalty"] = 1.0 / T_MAX
    c["env_kwargs"] = ek
    name = f"S13_{arm}_v{view}_{cell}_{intr}"
    c["run_name"] = name
    yaml.safe_dump(c, open(f"{OUT}/{name}.yaml", "w"), sort_keys=False)
    return name

if __name__ == "__main__":
    n = 0
    for cell, kw in CELLS.items():
        for intr in BONUSES:
            for view in (7, 3):
                for arm in ("flat", "freeze", "native"):
                    emit(cell, kw, intr, view, arm); n += 1
    print(f"emitted {n} configs to {OUT}  (T_max={T_MAX}, move_penalty=1/T_max={1/T_MAX:.6f})")
