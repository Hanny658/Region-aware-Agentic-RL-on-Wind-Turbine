"""Evaluation metrics: DEL (rainflow, fatpack), pitch travel / ADC, tracking errors."""
from __future__ import annotations

import numpy as np


def del_rainflow(signal: np.ndarray, dt: float, m: float, neq_per_s: float = 1.0) -> float:
    """Damage-equivalent load with Woehler exponent m, referenced to neq_per_s cycles per second
    over the signal duration (so DELs of equal-length signals are directly comparable).
    Blade (composite): m=10; steel tower: m=4."""
    import fatpack
    ranges = fatpack.find_rainflow_ranges(np.asarray(signal, dtype=float))
    if len(ranges) == 0:
        return 0.0
    T = len(signal) * dt
    neq = neq_per_s * T
    return float((np.sum(ranges ** m) / neq) ** (1.0 / m))


def pitch_travel(beta: np.ndarray) -> float:
    """Total pitch travel [rad]: proxy for actuator duty."""
    return float(np.sum(np.abs(np.diff(beta))))


def adc(beta: np.ndarray, dt: float, max_rate: float) -> float:
    """Actuator duty cycle: mean |beta_dot| / max_rate over the window."""
    return float(np.mean(np.abs(np.diff(beta)) / dt) / max_rate)


def summary(t: np.ndarray, P: np.ndarray, gen_speed: np.ndarray, beta: np.ndarray,
            M_root: np.ndarray, M_twr: np.ndarray | None, P_rated: float, wg_rated: float,
            max_rate: float, warmup_s: float = 30.0) -> dict:
    dt = float(t[1] - t[0])
    k = int(warmup_s / dt)
    sl = slice(k, None)
    out = {
        "energy_MWh": float(np.trapezoid(P[sl], t[sl]) / 3.6e9),
        "P_mean_MW": float(P[sl].mean() / 1e6),
        "P_mse_norm": float(np.mean(((P[sl] - P_rated) / P_rated) ** 2)),
        "gen_speed_std_rads": float(gen_speed[sl].std()),
        "gen_speed_mse_norm": float(np.mean(((gen_speed[sl] - wg_rated) / wg_rated) ** 2)),
        "gen_speed_max_rel": float(gen_speed[sl].max() / wg_rated),
        "pitch_travel_rad": pitch_travel(beta[sl]),
        "adc": adc(beta[sl], dt, max_rate),
        "RootMyc_DEL_m10": del_rainflow(M_root[sl], dt, 10),
        "RootMyc_mean_abs": float(np.abs(M_root[sl]).mean()),
    }
    if M_twr is not None:
        out["TwrBsMyt_DEL_m4"] = del_rainflow(M_twr[sl], dt, 4)
    return out
