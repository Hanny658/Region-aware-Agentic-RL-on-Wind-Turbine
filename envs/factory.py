"""Build envs / episode lists from a small experiment spec so scripts share one code path."""
from __future__ import annotations

import os
from pathlib import Path

from envs.base_env import EnvConfig, EpisodeSpec, default_config

WIND_DIR = os.path.expanduser(os.environ.get("WTRL_WIND", "~/wtrl/wind"))
WTRL = os.path.expanduser(os.environ.get("WTRL_HOME", "~/wtrl"))


def wind_path(mean: float, ti: float = 8.0, seed: int = 1) -> str:
    return f"{WIND_DIR}/U{mean:g}_TI{ti:g}_S{seed}.bts"


def episode_list(means, seeds=(1,), ti=8.0, episode_s=150.0, warmup_s=20.0) -> list[EpisodeSpec]:
    eps = []
    for s in seeds:
        for u in means:
            p = wind_path(u, ti, s)
            if not Path(p).exists():
                raise FileNotFoundError(p)
            eps.append(EpisodeSpec(wind_file=p, mean_wind=u, episode_s=episode_s, warmup_s=warmup_s))
    return eps


def make_env(backend: str, episodes: list[EpisodeSpec], cfg: EnvConfig | None = None,
             port: int = 5600, work_tag: str = "work", seed: int = 0, keep_outputs: bool = False):
    cfg = cfg or default_config()
    if backend == "toy":
        from envs.toy_env import ToyTurbineEnv
        return ToyTurbineEnv(cfg, episodes, f"{WTRL}/rosco_install/lib/libdiscon.so",
                             f"{WTRL}/runs/toy_discon", seed=seed)
    if backend == "openfast":
        from envs.openfast_env import OpenFASTEnv
        return OpenFASTEnv(cfg, episodes, f"{WTRL}/runs/template_5mw", f"{WTRL}/runs/{work_tag}",
                           port=port, seed=seed, keep_outputs=keep_outputs)
    raise ValueError(backend)


def baseline_dir(backend: str) -> str:
    return f"{WTRL}/baselines/{backend}"
