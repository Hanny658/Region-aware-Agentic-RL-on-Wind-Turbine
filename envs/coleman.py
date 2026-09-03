"""Coleman (multi-blade / d-q) transform, matching ROSCO v2.10.5 Functions.f90 exactly
(ColemanTransform / ColemanTransformInverse, harmonic 1, aziOffset 0 — an RL policy acting on
both axes learns any phase lead implicitly). Blade i phase: psi_i = azimuth + 2*pi*(i-1)/3.

Used by the IPC action channel (base_env): the agent outputs quasi-static (theta_d, theta_q);
the inverse transform turns them into three azimuth-phased per-blade pitch offsets. The forward
transform turns the three measured blade-root OoP moments into the (M_d, M_q) observation.
"""
from __future__ import annotations

import numpy as np

_PHI = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])


def coleman(moop3, azimuth: float) -> tuple[float, float]:
    """(M1,M2,M3), azimuth [rad] -> (M_d tilt, M_q yaw). ROSCO Functions.f90:351."""
    a = azimuth + _PHI
    m = np.asarray(moop3, dtype=float)
    return (float(2.0 / 3.0 * np.sum(np.cos(a) * m)),
            float(2.0 / 3.0 * np.sum(np.sin(a) * m)))


def coleman_inverse(theta_d: float, theta_q: float, azimuth: float) -> np.ndarray:
    """(theta_d, theta_q), azimuth [rad] -> per-blade offsets (3,). ROSCO Functions.f90:373."""
    a = azimuth + _PHI
    return np.cos(a) * theta_d + np.sin(a) * theta_q
