"""1-DOF NREL-5MW toy backend driven by the *real* ROSCO DISCON (ctypes), for pipeline
debugging and cheap hyper-parameter sweeps before OpenFAST.

Dynamics
    J_lss * dOmega/dt = T_aero(lambda, beta, v) - N * T_gen
    T_aero = 0.5 rho pi R^2 Cp(lambda, beta) v^3 / Omega,   F_T = 0.5 rho pi R^2 Ct v^2
    blade-1 root OoP moment = static share of thrust (F_T/3 * 0.65 R) passed through a lightly
    damped first-flap mode (0.67 Hz, zeta 0.05) + 1P modulation from the rotationally sampled
    rotor-averaged wind. Pitch actuator: rate limit only (ROSCO already rate-limits its part).
Cp/Ct tables: ROSCO Cp_Ct_Cq.NREL5MW.txt (pitch deg x TSR).
Residual is added in Python after ROSCO (DISCON.IN copy must have ZMQ_Mode = 0).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from controllers.rosco_ctypes import RoscoDiscon
from envs.base_env import EnvConfig, EpisodeSpec, ResidualPitchEnv
from envs.wind import load_bts, rotor_average_series, sample_at


def read_cp_ct_tables(path: str | Path):
    """Parse ROSCO performance file -> (pitch_deg, tsr, Cp[tsr, pitch], Ct[tsr, pitch])."""
    lines = Path(path).read_text().splitlines()
    vec = []
    mats = []
    cur = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            if cur:
                mats.append(np.array(cur))
                cur = []
            continue
        row = np.array([float(x) for x in s.split()])
        if len(vec) < 3:
            vec.append(row)
        else:
            cur.append(row)
    if cur:
        mats.append(np.array(cur))
    pitch, tsr = vec[0], vec[1]
    cp, ct = mats[0], mats[1]
    assert cp.shape == (len(tsr), len(pitch)), cp.shape
    return pitch, tsr, cp, ct


def read_ps_table(discon_in: str | Path):
    """PS_WindSpeeds / PS_BldPitchMin from DISCON.IN (used to mimic ROSCO's peak-shaving lower limit)."""
    ws = bp = None
    for ln in Path(discon_in).read_text().splitlines():
        if "! PS_WindSpeeds" in ln:
            ws = np.array([float(x) for x in ln.split("!")[0].split()])
        elif "! PS_BldPitchMin" in ln and "_N" not in ln:
            bp = np.array([float(x) for x in ln.split("!")[0].split()])
    return ws, bp


class _Bilinear:
    def __init__(self, x, y, z):        # z[y, x]
        self.x, self.y, self.z = x, y, z

    def __call__(self, xv, yv):
        xv = np.clip(xv, self.x[0], self.x[-1])
        yv = np.clip(yv, self.y[0], self.y[-1])
        i = np.clip(np.searchsorted(self.x, xv) - 1, 0, len(self.x) - 2)
        j = np.clip(np.searchsorted(self.y, yv) - 1, 0, len(self.y) - 2)
        fx = (xv - self.x[i]) / (self.x[i + 1] - self.x[i])
        fy = (yv - self.y[j]) / (self.y[j + 1] - self.y[j])
        z = self.z
        return ((1 - fx) * (1 - fy) * z[j, i] + fx * (1 - fy) * z[j, i + 1]
                + (1 - fx) * fy * z[j + 1, i] + fx * fy * z[j + 1, i + 1])


class ToyTurbineEnv(ResidualPitchEnv):
    BACKEND = "toy"
    def __init__(self, cfg: EnvConfig, episodes: list[EpisodeSpec], lib_path: str, discon_dir: str,
                 seed: int = 0, flap_hz: float = 0.67, flap_zeta: float = 0.05, one_p_amp: float = 0.08):
        super().__init__(cfg, episodes, seed)
        self.lib_path = os.path.expanduser(lib_path)
        self.discon_dir = Path(os.path.expanduser(discon_dir))
        self.discon_in = self.discon_dir / "DISCON.IN"
        pitch, tsr, cp, ct = read_cp_ct_tables(self.discon_dir / "Cp_Ct_Cq.NREL5MW.txt")
        self.Cp = _Bilinear(pitch, tsr, np.maximum(cp, 0.0))
        self.Ct = _Bilinear(pitch, tsr, np.maximum(ct, 0.0))
        self.ps_ws, self.ps_bp = read_ps_table(self.discon_in)
        tb = self.tb
        self.R = float(tb["rotor_radius_m"])
        self.A = np.pi * self.R ** 2
        self.rho = float(tb["rho_air"])
        self.J = float(tb["drivetrain_inertia_lss_kgm2"])
        self.N = float(tb["gearbox_ratio"])
        self.eta = float(tb["gen_efficiency"])
        self.rate_max = float(tb["max_pitch_rate_rads"])
        self.tq_max = float(tb["max_gen_torque_nm"])
        self.flap_w = 2 * np.pi * flap_hz
        self.flap_z = flap_zeta
        self.one_p_amp = one_p_amp
        self.rosco: RoscoDiscon | None = None
        self._wind_cache: dict[str, tuple[np.ndarray, float]] = {}

    # ---------------------------------------------------------------- physics
    def _wind(self, spec: EpisodeSpec):
        if spec.wind_file not in self._wind_cache:
            w = load_bts(spec.wind_file)
            self._wind_cache[spec.wind_file] = (rotor_average_series(w, self.R), w.dt)
        return self._wind_cache[spec.wind_file]

    def _aero(self, v, omega, beta_rad):
        lam = omega * self.R / max(v, 0.5)
        cp = self.Cp(np.rad2deg(beta_rad), lam)
        ct = self.Ct(np.rad2deg(beta_rad), lam)
        P = 0.5 * self.rho * self.A * cp * v ** 3
        T = P / max(omega, 0.05)
        F = 0.5 * self.rho * self.A * ct * v ** 2
        return T, F

    def _min_pit(self, v):
        if self.ps_ws is None:
            return float(self.tb["fine_pitch_rad"])
        return float(max(np.interp(v, self.ps_ws, self.ps_bp), self.tb["fine_pitch_rad"]))

    def _sim_reset(self, spec: EpisodeSpec) -> dict:
        self.series, self.wind_dt = self._wind(spec)
        if self.rosco is not None:
            self.rosco.finish()
        self.rosco = RoscoDiscon(self.lib_path, self.discon_in, self.dt,
                                 sim_name=str(self.discon_dir / "toy"))
        v0 = float(self.series[0])
        tsr = float(self.tb["tsr_opt"])
        omega_rated = float(self.tb["rated_rotor_speed_rpm"]) * 2 * np.pi / 60
        self.omega = min(tsr * v0 / self.R, omega_rated)
        self.beta = self._min_pit(v0) if v0 < self.tb["rated_wind_ms"] else np.deg2rad(2.0 * (v0 - 11.4) + 1)
        self.beta = float(np.clip(self.beta, 0, 0.5))
        self.azimuth = 0.0
        self.Tg = 0.0
        self.t = 0.0
        self.flap_x = np.zeros(2)          # [M_dyn, M_dyn_dot]
        self.rosco.init(self.beta, self.omega * self.N, self.omega, v0, self.Tg)
        self.beta_native, self.Tg = self.rosco.step(0.0, self.beta, self.omega * self.N, self.omega,
                                                    self.Tg, self.eta, v0)
        self.Tg_app = self.Tg
        return self._measure(v0, 0.0)

    def _sim_step(self, pitch_offset: float, tq_offset: float = 0.0, ipc3=None) -> dict:
        assert ipc3 is None, "the 1-DOF toy twin cannot represent per-blade pitch (IPC is OpenFAST-only)"
        v = sample_at(self.series, self.wind_dt, self.t)
        # actuator: total command = ROSCO native + residual, rate-limited
        cmd = self.beta_native + pitch_offset
        d = np.clip(cmd - self.beta, -self.rate_max * self.dt, self.rate_max * self.dt)
        self.beta = float(self.beta + d)
        # torque: native + residual, hard-limited (mirrors the patched avrSWAP(47) write)
        self.Tg_app = float(np.clip(self.Tg + tq_offset, 0.0, self.tq_max))
        # rotor
        T_aero, F_T = self._aero(v, self.omega, self.beta)
        self.omega += self.dt * (T_aero - self.N * self.Tg_app) / self.J
        self.omega = max(self.omega, 0.05)
        self.azimuth = (self.azimuth + self.omega * self.dt) % (2 * np.pi)
        # blade root flap load: static + mode + 1P
        M_static = (F_T / 3.0) * 0.65 * self.R
        x, xd = self.flap_x
        xdd = self.flap_w ** 2 * (M_static - x) - 2 * self.flap_z * self.flap_w * xd
        xd += self.dt * xdd
        x += self.dt * xd
        self.flap_x[:] = (x, xd)
        self.M_oop = x * (1.0 + self.one_p_amp * np.sin(self.azimuth))
        self.t += self.dt
        # controller (measurements after the physics update; DISCON sees the *applied* torque,
        # as OpenFAST would, while its own PI state stays native — the Fortran patch guarantees that)
        self.beta_native, self.Tg = self.rosco.step(self.t, self.beta, self.omega * self.N, self.omega,
                                                    self.Tg_app, self.eta, v, root_oop=(self.M_oop,) * 3)
        return self._measure(v, pitch_offset)

    def _measure(self, v, pitch_offset) -> dict:
        wg = self.omega * self.N
        tq = getattr(self, "Tg_app", self.Tg)
        return {
            "t": self.t, "P": wg * tq * self.eta, "gen_speed": wg, "rot_speed": self.omega,
            "gen_torque": tq, "v_hub": v, "v_est": v, "M_oop": getattr(self, "M_oop", 0.0),
            "beta_meas": self.beta, "beta_native": self.beta_native, "min_pit": self._min_pit(v),
            "beta_applied": self.beta, "offset": pitch_offset, "fa_acc": 0.0,
        }

    def _sim_close(self):
        if self.rosco is not None:
            self.rosco.finish()
            self.rosco = None
