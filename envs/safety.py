"""Residual post-processing shared by every method:
    raw agent output -> damper (paper eq. 2) -> region mask -> absolute bounds vs ROSCO cmd.
ROSCO v2.10.5 rate-limits the *total* pitch command after adding the ZMQ offset, so the
physical pitch-rate limit is enforced inside ROSCO; the damper here keeps the residual smooth.
dbeta_max is per region (supervisor knobs dbeta_max_R2 / dbeta_max_R3).
"""
from __future__ import annotations

import numpy as np

from controllers.router import R2, R3


class SecondOrderDamper:
    """Critically damped 2nd-order low-pass  y'' + 2 zeta wn y' + wn^2 y = wn^2 u,
    integrated with an exact zero-order-hold discretisation of the state-space form."""

    def __init__(self, omega_n: float, zeta: float, dt: float):
        from scipy.linalg import expm
        A = np.array([[0.0, 1.0], [-omega_n ** 2, -2.0 * zeta * omega_n]])
        B = np.array([[0.0], [omega_n ** 2]])
        M = np.zeros((3, 3))
        M[:2, :2] = A * dt
        M[:2, 2:] = B * dt
        E = expm(M)
        self.Ad, self.Bd = E[:2, :2], E[:2, 2:]
        self.x = np.zeros((2, 1))

    def reset(self, y0: float = 0.0):
        self.x = np.array([[y0], [0.0]])

    def step(self, u: float) -> float:
        self.x = self.Ad @ self.x + self.Bd * u
        return float(self.x[0, 0])


class ResidualSafety:
    """Turns a raw residual into the pitch offset actually sent to ROSCO."""

    def __init__(self, dbeta_max, min_pitch: float, max_pitch: float,
                 damper: SecondOrderDamper | None):
        self.dbeta_max = {R2: float(dbeta_max), R3: float(dbeta_max)} if not isinstance(dbeta_max, dict) \
            else {R2: float(dbeta_max["R2"]), R3: float(dbeta_max["R3"])}
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.damper = damper

    def set_knobs(self, knobs: dict):
        if "dbeta_max_R2" in knobs:
            self.dbeta_max[R2] = float(knobs["dbeta_max_R2"])
        if "dbeta_max_R3" in knobs:
            self.dbeta_max[R3] = float(knobs["dbeta_max_R3"])

    def knobs(self) -> dict:
        return {"dbeta_max_R2": self.dbeta_max[R2], "dbeta_max_R3": self.dbeta_max[R3]}

    def reset(self):
        if self.damper is not None:
            self.damper.reset()

    def apply(self, action_unit: float, region: int, beta_native: float, min_pit_now: float) -> float:
        """action_unit: agent output in [-1, 1] (scaled here by the region's dbeta_max).
        beta_native: ROSCO collective pitch command before the offset.
        min_pit_now: ROSCO's current lower pitch limit (fine pitch or peak-shaving limit)."""
        d = float(np.clip(action_unit, -1.0, 1.0)) * self.dbeta_max[region]
        if region == R2:                       # decision: R2 residual is non-negative
            d = max(d, 0.0)
        if self.damper is not None:
            d = self.damper.step(d)
        lo = max(self.min_pitch, min_pit_now)
        total = float(np.clip(beta_native + d, lo, self.max_pitch))
        return total - beta_native
