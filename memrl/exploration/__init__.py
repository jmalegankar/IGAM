"""Intrinsic-reward modules for sparse-reward POMDP exploration.

All modules conform to `IntrinsicRewardModule` (see base.py) and are
instantiated via `make_intrinsic(name, ...)` so MemPPO doesn't need
per-method branches.

Available modules:
    "none"       — NoBonus baseline (control)
    "rnd"        — RND (Burda 2019)        — lifelong novelty
    "e3b_rand"   — E3B with φ^rand          — episodic, thesis winner
    "e3b_obs"    — E3B with φ^obs           — episodic, simpler
    "e3b_innov"  — E3B with φ^eps_mem       — episodic, Agency-Principle counter
    "noveld"     — NovelD (Zhang 2021)      — hybrid (RND diff + first-visit gate)
    "icm"        — ICM (Pathak 2017)        — lifelong, IDF features

For E3B + a different feature source, build via:
    >>> from memrl.exploration import E3B, RandomPhi
    >>> phi = RandomPhi(obs_dim=128)
    >>> module = E3B(n_envs=16, phi_source=phi, ...)
"""

from .base import IntrinsicRewardModule
from .e3b import EllipticalEpisodicBonus, RunningStd      # thesis implementation
from .e3b_module import E3B
from .icm import ICM
from .none import NoBonus
from .noveld import NovelD
from .phi_sources import CellInnovationPhi, ObsPhi, PhiSource, RandomPhi
from .rnd import RND


_REGISTRY = {
    "none":       NoBonus,
    "rnd":        RND,
    "noveld":     NovelD,
    "icm":        ICM,
}


def make_intrinsic(
    name: str,
    *,
    n_envs: int,
    obs_dim: int,
    n_actions: int | None = None,
    device: str = "cpu",
    **kwargs,
) -> IntrinsicRewardModule:
    """Factory: instantiate an intrinsic module by name.

    E3B variants are special — they need a PhiSource. The "e3b_*" names
    pick the source. For custom sources, instantiate `E3B(phi_source=...)` directly.

    Args:
        name:       "none" / "rnd" / "e3b_rand" / "e3b_obs" / "e3b_innov" / "noveld" / "icm"
        n_envs:     parallel envs
        obs_dim:    flattened observation dimension
        n_actions:  needed for ICM (Discrete action vocab)
        device:     torch device
        **kwargs:   forwarded to the module constructor
    """
    name = name.lower()
    if name in _REGISTRY:
        cls = _REGISTRY[name]
        if name == "icm":
            if n_actions is None:
                raise ValueError("ICM requires n_actions=<Discrete vocab>")
            return cls(n_envs=n_envs, obs_dim=obs_dim, n_actions=n_actions,
                       device=device, **kwargs)
        if name == "none":
            return cls(n_envs=n_envs, device=device, **kwargs)
        return cls(n_envs=n_envs, obs_dim=obs_dim, device=device, **kwargs)

    if name.startswith("e3b_"):
        source_name = name.split("_", 1)[1]
        if source_name == "rand":
            phi = RandomPhi(obs_dim=obs_dim, device=device)
        elif source_name == "obs":
            phi = ObsPhi(obs_dim=obs_dim, device=device)
        elif source_name == "innov":
            phi = CellInnovationPhi(side_key="eps_mem", device=device)
        else:
            raise ValueError(f"Unknown E3B feature source '{source_name}'. "
                             "Choose from: rand, obs, innov.")
        return E3B(n_envs=n_envs, phi_source=phi, device=device, **kwargs)

    raise ValueError(
        f"Unknown intrinsic module '{name}'. "
        f"Available: {sorted(list(_REGISTRY.keys()) + ['e3b_rand', 'e3b_obs', 'e3b_innov'])}"
    )


__all__ = [
    "IntrinsicRewardModule",
    "make_intrinsic",
    "NoBonus",
    "RND",
    "E3B",
    "NovelD",
    "ICM",
    "PhiSource", "RandomPhi", "ObsPhi", "CellInnovationPhi",
    "EllipticalEpisodicBonus", "RunningStd",
]
