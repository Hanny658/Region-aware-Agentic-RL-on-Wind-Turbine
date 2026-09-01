"""Ground-truth fitness F that no learner or supervisor can modify (decision 2026-08-29, option 2:
constraint form).

Relative to the paired GSPI baseline (same wind file, same seed):
    objective     maximise DEL reduction [%] of the target load: 'tower' (tower-base fore-aft,
                  TwrBsMyt from .outb, decision 2026-08-30; OpenFAST only) or 'blade' (RootMoop
                  from the env log; the toy twin can only do blade)
    constraint 1  energy loss <= 1 %                       (all episodes)
    constraint 2  R3 generator-speed std not worse than GSPI (episodes with >= 50 % R3 steps)

    F = DEL_red_pct - 20 * max(0, energy_loss_pct - 1.0) - 20 * max(0, 100 * (speed_std_ratio - 1))

i.e. every percentage point of constraint violation costs 20 points of DEL reduction. F is
evaluated on deterministic evaluation episodes, never on exploratory training episodes.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from agents.rollout import episode_metrics
from envs.base_env import EpisodeSpec

ENERGY_TOL_PCT = 1.0
SPEED_TOL_RATIO = 1.0
PENALTY = 20.0


def baseline_metrics(baseline_dir: str, episodes: list[EpisodeSpec], dt: float, wg_rated: float) -> dict:
    """{wind_file: metrics} computed from the zero-residual npz logs with the same code path."""
    out = {}
    for ep in episodes:
        p = Path(os.path.expanduser(baseline_dir)) / f"{Path(ep.wind_file).stem}.npz"
        d = np.load(p)
        L = {k: d[k] for k in d.files if not k.startswith("outb_")}
        outb = {k[5:]: d[k] for k in d.files if k.startswith("outb_")} or None
        out[ep.wind_file] = episode_metrics(L, dt, wg_rated, ep.warmup_s, outb)
    return out


def fitness(results: list[dict], base: dict, target: str = "blade") -> dict:
    """results: outputs of deterministic run_episode (one per eval episode).
    target: 'blade' (RootMoop DEL from the env log) or 'tower' (TwrBsMyt DEL from .outb, OpenFAST)."""
    e_new = sum(r["metrics"]["energy_MWh"] for r in results)
    e_base = sum(base[r["wind_file"]]["energy_MWh"] for r in results)
    energy_loss_pct = 100.0 * (1.0 - e_new / e_base)
    key = "TwrBsMyt_DEL_MNm" if target == "tower" else "RootMoop_DEL_MNm"
    if target == "tower" and not all(key in r["metrics"] and key in base[r["wind_file"]] for r in results):
        raise ValueError("fitness target 'tower' needs .outb metrics (OpenFAST backend)")
    del_red = [100.0 * (1.0 - r["metrics"][key] / base[r["wind_file"]][key]) for r in results]
    del_red_pct = float(np.mean(del_red))
    r3 = [(r["metrics"]["gen_speed_std_R3"], base[r["wind_file"]]["gen_speed_std_R3"])
          for r in results if r["metrics"]["frac_R3"] >= 0.5]
    if r3:
        speed_std_ratio = float(np.mean([a / b for a, b in r3]))
    else:
        speed_std_ratio = 1.0
    # R3 tracking MAE (reporting only; the paper's speed/power-tracking metric)
    mae = [(r["metrics"].get("gen_speed_mae_R3"), base[r["wind_file"]].get("gen_speed_mae_R3"),
            r["metrics"].get("power_mae_R3"))
           for r in results if r["metrics"]["frac_R3"] >= 0.5]
    mae = [(a, b, c) for a, b, c in mae if a == a and b == b and b]
    speed_mae_ratio_R3 = float(np.mean([a / b for a, b, _ in mae])) if mae else float("nan")
    gen_speed_mae_R3 = float(np.mean([a for a, _, _ in mae])) if mae else float("nan")
    power_mae_R3 = float(np.mean([c for _, _, c in mae if c == c])) if mae else float("nan")
    pen_e = PENALTY * max(0.0, energy_loss_pct - ENERGY_TOL_PCT)
    pen_s = PENALTY * max(0.0, 100.0 * (speed_std_ratio - SPEED_TOL_RATIO))
    F = del_red_pct - pen_e - pen_s
    # tiered reporting (decision 2026-08-30):
    #   strict   : constraints met as defined (speed std <= GSPI, energy loss <= 1 %)  -> F == F_strict
    #   tol2     : engineering-equivalent (speed std <= 1.02 x GSPI, energy loss <= 1 %) -> F_tol2
    #   degraded : neither
    pen_s2 = PENALTY * max(0.0, 100.0 * (speed_std_ratio - 1.02))
    F_tol2 = del_red_pct - pen_e - pen_s2
    tier = "strict" if (pen_e == 0 and pen_s == 0) else ("tol2" if (pen_e == 0 and pen_s2 == 0) else "degraded")
    extra = {"F_strict": float(F), "F_tol2": float(F_tol2), "tier": tier, "target": target,
             "speed_mae_ratio_R3": speed_mae_ratio_R3, "gen_speed_mae_R3": gen_speed_mae_R3,
             "power_mae_R3": power_mae_R3}
    # paper-style paired MSE reductions (Wang et al. Fig. 4 analogues, R3 steps, vs rated)
    for key, name in (("power_mse_R3", "power_mse_red_pct"), ("gen_speed_mse_R3", "gen_speed_mse_red_pct")):
        pairs = [(r["metrics"].get(key), base[r["wind_file"]].get(key))
                 for r in results if r["metrics"]["frac_R3"] >= 0.5]
        pairs = [(a, b) for a, b in pairs if a == a and b == b and b]
        extra[name] = float(np.mean([100.0 * (1 - a / b) for a, b in pairs])) if pairs else float("nan")
    if all("RootMyc1_DEL_MNm" in r["metrics"] for r in results) and \
            all("RootMyc1_DEL_MNm" in base[r["wind_file"]] for r in results):
        extra["RootMyc1_DEL_red_pct"] = float(np.mean(
            [100 * (1 - r["metrics"]["RootMyc1_DEL_MNm"] / base[r["wind_file"]]["RootMyc1_DEL_MNm"]) for r in results]))
        extra["TwrBsMyt_DEL_red_pct"] = float(np.mean(
            [100 * (1 - r["metrics"]["TwrBsMyt_DEL_MNm"] / base[r["wind_file"]]["TwrBsMyt_DEL_MNm"]) for r in results]))
    return {
        "F": float(F), "del_red_pct": del_red_pct, "energy_loss_pct": float(energy_loss_pct),
        "speed_std_ratio": speed_std_ratio, "constraints_ok": bool(pen_e == 0 and pen_s == 0),
        "terminated_any": any(r["terminated"] for r in results),
        "per_episode": [{"mean_wind": r["mean_wind"], "del_red_pct": d,
                         "energy_MWh": r["metrics"]["energy_MWh"],
                         "energy_base_MWh": base[r["wind_file"]]["energy_MWh"],
                         "gen_speed_std_R3": r["metrics"]["gen_speed_std_R3"],
                         "gen_speed_mae_R3": r["metrics"].get("gen_speed_mae_R3"),
                         "pitch_travel_deg": r["metrics"]["pitch_travel_deg"],
                         "pitch_travel_base_deg": base[r["wind_file"]]["pitch_travel_deg"],
                         "dbeta_abs_mean_deg": r["metrics"]["dbeta_abs_mean_deg"]}
                        for r, d in zip(results, del_red)],
        **extra,
    }
