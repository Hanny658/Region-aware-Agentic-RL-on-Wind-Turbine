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
                 damper: SecondOrderDamper | None, dtau_max_nm: float = 0.0,
                 tq_min_nm: float = 0.0, tq_max_nm: float = 0.0,
                 tau_damper: SecondOrderDamper | None = None):
        self.dbeta_max = {R2: float(dbeta_max), R3: float(dbeta_max)} if not isinstance(dbeta_max, dict) \
            else {R2: float(dbeta_max["R2"]), R3: float(dbeta_max["R3"])}
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.damper = damper
        # torque residual channel (R2 only, decision 2026-09-02); disabled when dtau_max_nm == 0
        self.dtau_max = float(dtau_max_nm)
        self.tq_min, self.tq_max = float(tq_min_nm), float(tq_max_nm)
        self.tau_damper = tau_damper
        self.tq_speed_cut = 0.0            # set by the env: 0.98 * rated gen speed [rad/s]
        # IPC channel (dq-frame cyclic pitch, R3 only; 2026-09-03): amplitude cap + one damper
        # per axis, all in the quasi-static dq domain (smoothness matters there, not per blade)
        self.ipc_max = 0.0
        self.ipc_dampers: tuple | None = None

    def set_knobs(self, knobs: dict):
        if "dbeta_max_R2" in knobs:
            self.dbeta_max[R2] = float(knobs["dbeta_max_R2"])
        if "dbeta_max_R3" in knobs:
            self.dbeta_max[R3] = float(knobs["dbeta_max_R3"])
        if "ipc_max" in knobs and self.ipc_max > 0.0:
            self.ipc_max = float(knobs["ipc_max"])

    def knobs(self) -> dict:
        out = {"dbeta_max_R2": self.dbeta_max[R2], "dbeta_max_R3": self.dbeta_max[R3]}
        if self.ipc_max > 0.0:
            out["ipc_max"] = self.ipc_max
        return out

    def reset(self):
        if self.damper is not None:
            self.damper.reset()
        if self.tau_damper is not None:
            self.tau_damper.reset()
        if self.ipc_dampers is not None:
            for d in self.ipc_dampers:
                d.reset()

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

    def apply_tau(self, action_unit: float, region: int, tq_native: float,
                  gen_speed: float = 0.0) -> float:
        """Torque residual [Nm]: R2 only AND truly below rated speed, +-dtau_max, smoothed,
        total clamped to the hard torque limits (ROSCO saturates again Fortran-side).
        The speed gate (< tq_speed_cut) breaks the overspeed reward exploit found 2026-09-03:
        negative dtau accelerates the rotor past rated while the oracle label can stay R2."""
        if self.dtau_max <= 0.0:
            return 0.0
        gate = region == R2 and (self.tq_speed_cut <= 0.0 or gen_speed < self.tq_speed_cut)
        d = float(np.clip(action_unit, -1.0, 1.0)) * self.dtau_max if gate else 0.0
        if self.tau_damper is not None:
            d = self.tau_damper.step(d)
        total = float(np.clip(tq_native + d, self.tq_min, self.tq_max))
        return total - tq_native

    def apply_ipc(self, ud: float, uq: float, region: int) -> tuple[float, float]:
        """dq-frame cyclic-pitch amplitudes [rad]: R3 only, each axis clipped to +-ipc_max and
        smoothed; ROSCO rate-limits and saturates the total per-blade command Fortran-side."""
        if self.ipc_max <= 0.0:
            return 0.0, 0.0
        gate = region == R3
        td = float(np.clip(ud, -1.0, 1.0)) * self.ipc_max if gate else 0.0
        tq = float(np.clip(uq, -1.0, 1.0)) * self.ipc_max if gate else 0.0
        if self.ipc_dampers is not None:
            td = self.ipc_dampers[0].step(td)
            tq = self.ipc_dampers[1].step(tq)
        return td, tq
