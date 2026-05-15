from .minigrid_wrappers import make_minigrid_vec_env
from .popgym_wrappers import make_popgym_vec_env


def make_vec_env(env_name: str, n_envs: int = 8, seed: int = 0):
    """Dispatch to the right factory based on env_name prefix.

    - "popgym-*"      → make_popgym_vec_env
    - "MiniGrid-*"    → make_minigrid_vec_env
    """
    if env_name.startswith("popgym-"):
        return make_popgym_vec_env(env_name, n_envs=n_envs, seed=seed)
    if env_name.startswith("MiniGrid-"):
        return make_minigrid_vec_env(env_name, n_envs=n_envs, seed=seed)
    raise ValueError(
        f"Unknown env prefix in {env_name!r}. Expected 'popgym-' or 'MiniGrid-'."
    )


__all__ = ["make_popgym_vec_env", "make_minigrid_vec_env", "make_vec_env"]
