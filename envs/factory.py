"""Build envs / episode lists from a small experiment spec so scripts share one code path."""
from __future__ import annotations

import os
from pathlib import Path

from envs.base_env import EnvConfig, EpisodeSpec, default_config

WIND_DIR = os.path.expanduser(os.environ.get("WTRL_WIND", "~/wtrl/wind"))
WTRL = os.path.expanduser(os.environ.get("WTRL_HOME", "~/wtrl"))
# OpenFAST case template; override to evaluate a differently configured ROSCO (e.g. the
# tower-damper baseline template_5mw_td) without touching the canonical one
TEMPLATE = os.path.expanduser(os.environ.get("WTRL_TEMPLATE", f"{WTRL}/runs/template_5mw"))


def ti_label(ti) -> str:
    """Turbulence label of a wind file: the intensity in percent (8 -> "8") or an IEC class ("B")."""
    return ti.upper() if isinstance(ti, str) else f"{ti:g}"


def parse_ti(text):
    """argparse type: a turbulence intensity in percent, or an IEC turbulence class A / B / C."""
    t = str(text).strip().upper()
    return t if t in ("A", "B", "C") else float(text)


def wind_path(mean: float, ti=8.0, seed: int = 1) -> str:
    return f"{WIND_DIR}/U{mean:g}_TI{ti_label(ti)}_S{seed}.bts"


def episode_list(means, seeds=(1,), ti=8.0, episode_s=150.0, warmup_s=20.0) -> list[EpisodeSpec]:
    eps = []
    for s in seeds:
        for u in means:
            p = wind_path(u, ti, s)
            if not Path(p).exists():
                raise FileNotFoundError(p)
            eps.append(EpisodeSpec(wind_file=p, mean_wind=u, episode_s=episode_s, warmup_s=warmup_s))
    return eps


def mpc_base_kw(cp_scale: float = 1.0, ftower_scale: float = 1.0, mass_scale: float = 1.0,
                adapt: str = "none", tau_adapt: float = 5.0, notch_3p_q: float = 0.0) -> dict:
    """The J-selected LPV-MPC (cost scale v2, N=20, r=0.3, qt=3, wc_v=0.35) as a base controller,
    with optional mismatches of the MPC's internal model (aerodynamic coefficients, first tower
    frequency, tower modal mass); the plant is never changed."""
    return dict(horizon=20, ts=0.1, q=1.0, r=0.3, qt=3.0, wc_v=0.35, err_ref=0.005, dbeta_ref=0.002,
                cp_scale=float(cp_scale), ftower_scale=float(ftower_scale), mass_scale=float(mass_scale),
                adapt=str(adapt), tau_adapt=float(tau_adapt), notch_3p_q=float(notch_3p_q))


def make_env(backend: str, episodes: list[EpisodeSpec], cfg: EnvConfig | None = None,
             port: int = 5600, work_tag: str = "work", seed: int = 0, keep_outputs: bool = False):
    cfg = cfg or default_config()
    if backend == "toy":
        from envs.toy_env import ToyTurbineEnv
        return ToyTurbineEnv(cfg, episodes, f"{WTRL}/rosco_install/lib/libdiscon.so",
                             f"{WTRL}/runs/toy_discon", seed=seed)
    if backend == "openfast":
        from envs.openfast_env import OpenFASTEnv
        return OpenFASTEnv(cfg, episodes, TEMPLATE, f"{WTRL}/runs/{work_tag}",
                           port=port, seed=seed, keep_outputs=keep_outputs)
    raise ValueError(backend)


def baseline_dir(backend: str) -> str:
    return f"{WTRL}/baselines/{backend}"
