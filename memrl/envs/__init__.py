"""Environment factories for POMDP / memory benchmarks.

``make_vec_env`` dispatches on the env name to the right factory. The
``popgym`` and ``minigrid`` factories are imported eagerly because their
dependencies are core to Phase A. The ``memory-gym`` and ``popgym-arcade``
factories are imported lazily — their upstream packages (and JAX, in the
case of popgym-arcade) are optional extras.
"""

from .classic_wrappers import is_classic_env
from .minigrid_wrappers import make_minigrid_vec_env
from .popgym_wrappers import make_popgym_vec_env


def make_vec_env(env_name: str, n_envs: int = 8, seed: int = 0, **kwargs):
    """Dispatch to the right factory based on ``env_name``.

    Routing:
        ``popgym-arcade-*``       → popgym-arcade (gymnax/JAX backend)
        ``popgym-*``              → popgym (gymnasium)
        ``MiniGrid-*``            → minigrid
        ``TMaze-*``               → T-Maze (Passive/Active, Ni et al. 2023)
        ``POMDP-*`` / classic ids → classic-control (CartPole-v1, masked POMDP)
        memory-gym ids (e.g.      → memory-gym
            ``Endless-*``,
            ``MortarMayhem*``,
            ``MysteryPath*``,
            ``SearingSpotlights*``)

    Extra ``kwargs`` are forwarded to the per-factory call (e.g.
    ``reset_options=`` for memory-gym, ``partial_obs=``/``obs_size=`` for
    popgym-arcade, ``corridor_length=`` for T-Maze). Unrecognized kwargs raise
    at the factory level.
    """
    if env_name.startswith("popgym-arcade-"):
        from .popgym_arcade_wrappers import make_popgym_arcade_vec_env
        return make_popgym_arcade_vec_env(env_name, n_envs=n_envs, seed=seed, **kwargs)
    if env_name.startswith("popgym-"):
        return make_popgym_vec_env(env_name, n_envs=n_envs, seed=seed, **kwargs)
    if env_name.startswith("TinyReproduce"):
        from .tiny_reproduce import make_tiny_reproduce_vec_env
        return make_tiny_reproduce_vec_env(env_name, n_envs=n_envs, seed=seed, **kwargs)
    if env_name.startswith("MiniWorld-"):
        from .miniworld_wrappers import make_miniworld_vec_env
        return make_miniworld_vec_env(env_name, n_envs=n_envs, seed=seed, **kwargs)
    if env_name.startswith("MiniGrid-"):
        return make_minigrid_vec_env(env_name, n_envs=n_envs, seed=seed, **kwargs)
    if env_name.startswith("TMaze-"):
        from .tmaze_wrappers import make_tmaze_vec_env
        return make_tmaze_vec_env(env_name, n_envs=n_envs, seed=seed, **kwargs)
    if is_classic_env(env_name):
        from .classic_wrappers import make_classic_vec_env
        return make_classic_vec_env(env_name, n_envs=n_envs, seed=seed, **kwargs)
    if _is_memory_gym(env_name):
        from .memory_gym_wrappers import make_memory_gym_vec_env
        return make_memory_gym_vec_env(env_name, n_envs=n_envs, seed=seed, **kwargs)
    raise ValueError(
        f"Unknown env id {env_name!r}. Expected one of: "
        f"'popgym-arcade-*', 'popgym-*', 'MiniGrid-*', 'TMaze-*', a classic id "
        f"('CartPole-v1', 'POMDP-CartPole-v1', ...), or a memory-gym id "
        f"('Endless-*', 'MortarMayhem*', 'MysteryPath*', 'SearingSpotlights*')."
    )


# memory-gym env ids are duplicated here (not imported from memory_gym_wrappers)
# so dispatch recognizes the id even when the optional memory-gym package
# isn't installed — the user gets a clear ImportError from the factory call
# rather than a misleading "Unknown env id".
_MEMORY_GYM_IDS = frozenset({
    "MortarMayhem-v0",
    "MortarMayhem-Grid-v0",
    "MortarMayhemB-v0",
    "MortarMayhemB-Grid-v0",
    "MysteryPath-v0",
    "MysteryPath-Grid-v0",
    "SearingSpotlights-v0",
    "Endless-MortarMayhem-v0",
    "Endless-MysteryPath-v0",
    "Endless-SearingSpotlights-v0",
})


def _is_memory_gym(env_name: str) -> bool:
    return env_name in _MEMORY_GYM_IDS


__all__ = [
    "make_popgym_vec_env",
    "make_minigrid_vec_env",
    "make_vec_env",
    "is_classic_env",
]
