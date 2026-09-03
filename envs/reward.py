"""Unified region-conditional reward (configs/reward.yaml). Shared by all methods.

    r = 1[R2] * w_P * (P/P_base - 1) + 1[R3] * w_w * exp(-|d_wg|/tau)
        - lambda_L[region] * load_proxy_t - lambda_A * (dbeta/kappa)^2

Load proxies (configs/reward.yaml `load_proxy`), all normalised so that the term averages about
-lambda_L per step under the GSPI baseline of the backend (scripts/dev/calib_dM_ref.py):
    increment : |M_t - M_{t-1}| / dM_ref                       (weak proxy for the m=10 DEL)
    range_inc : max(0, range_t - range_{t-1}) * N_win / R_ref   range over the trailing window
                (window sum == window peak-to-peak, which correlates 0.83-1.0 with the DEL)
    ema_std   : ema_std_t / S_ref                               EMA standard deviation of M
History: the |M| level penalty (first campaign) drove load-feedback pitching that excited the
flap mode; `increment` (campaign 3) rewarded suppressing small high-frequency wiggles and was
unstable at lambda >= 3. See scripts/dev/del_anatomy.py for the evidence.
lambda_load may be a scalar or {"R2": x, "R3": y}. Supervisor knobs are hot-swappable.
"""
from __future__ import annotations

import numpy as np

from controllers.router import R2, R3

PROXIES = ("increment", "range_inc", "ema_std")


def _per_region(v) -> dict:
    if isinstance(v, dict):
        return {R2: float(v["R2"]), R3: float(v["R3"])}
    return {R2: float(v), R3: float(v)}


def _by_backend(v, backend: str) -> float:
    return float(v[backend]) if isinstance(v, dict) else float(v)


class RegionReward:
    def __init__(self, cfg: dict, rated_gen_speed: float, backend: str, dt: float):
        self.w_P = float(cfg["w_power"])
        self.w_w = float(cfg["w_speed"])
        self.tau = float(cfg["tau_speed_err"])
        self.lam_L = _per_region(cfg["lambda_load"])
        self.lam_A = float(cfg["lambda_act"])
        self.kappa = float(cfg["kappa_beta_rad"])
        # torque-residual penalty scale [Nm]; mirrors kappa_beta (kappa = 2x the default residual
        # bound, so equal fractional actuation costs the same in both channels)
        self.kappa_tau = float(cfg.get("kappa_tau_nm", 4000.0))
        self.P_rated = float(cfg.get("rated_power_w", 5.0e6))
        self.kappa_ipc = float(cfg.get("kappa_ipc_rad", 0.035))   # 2x the default 1 deg IPC bound
        # kinetic-energy-exact power accounting (torque residual, 2026-09-03): credit
        # P + eta * d(0.5*J*omega^2)/dt so draining/storing rotor KE is reward-neutral — the
        # per-step P/P_base reward otherwise pays for draining (rotor recovery ~20-60 s >> the
        # gamma=0.998 5 s horizon; tq_on2kg_s0 learned constant +dtau, Eloss ~8%). EMA tau 1 s
        # tames the per-step variance; the residual exploit window (~1 s) stays unprofitable.
        self._ke_alpha = min(1.0, dt / 1.0)
        self._ke_ema = 0.0
        self.wg_rated = rated_gen_speed
        self.proxy = cfg.get("load_proxy", "increment")
        assert self.proxy in PROXIES, self.proxy
        self.signal = cfg.get("load_signal", "M_oop")
        if self.signal == "fa_acc":
            self.dM_ref = _by_backend(cfg["fa_dM_ref"], backend)
            self.R_ref = _by_backend(cfg["fa_R_ref"], backend)
            self.S_ref = _by_backend(cfg["fa_S_ref"], backend)
            assert self.R_ref > 0 and self.dM_ref > 0, "fa_acc references not calibrated (scripts/dev/calib_fa_ref.py)"
        else:
            self.dM_ref = _by_backend(cfg["dM_ref_nm"], backend)
            self.R_ref = _by_backend(cfg.get("R_ref_nm", 1.0), backend)
            self.S_ref = _by_backend(cfg.get("S_ref_nm", 1.0), backend)
        self.N_win = max(1, int(round(float(cfg.get("window_s", 10.0)) / dt)))
        self.ema_a = dt / float(cfg.get("ema_tau_s", 5.0))
        self.reset()

    def reset(self):
        self._buf = np.zeros(self.N_win)
        self._n = 0
        self._range_prev = 0.0
        self._m1 = self._m2 = None
        self._ke_ema = 0.0

    def set_knobs(self, knobs: dict):
        if "w_power" in knobs:
            self.w_P = float(knobs["w_power"])
        if "w_speed" in knobs:
            self.w_w = float(knobs["w_speed"])
        if "lambda_load_R2" in knobs:
            self.lam_L[R2] = float(knobs["lambda_load_R2"])
        if "lambda_load_R3" in knobs:
            self.lam_L[R3] = float(knobs["lambda_load_R3"])

    def knobs(self) -> dict:
        return {"w_power": self.w_P, "w_speed": self.w_w,
                "lambda_load_R2": self.lam_L[R2], "lambda_load_R3": self.lam_L[R3]}

    def load_proxy(self, M: float, M_prev: float) -> float:
        """Unit-free per-step load penalty (>= 0), ~1 on average under GSPI."""
        if self.proxy == "increment":
            return abs(M - M_prev) / self.dM_ref
        if self.proxy == "range_inc":
            self._buf[self._n % self.N_win] = M
            self._n += 1
            seg = self._buf if self._n >= self.N_win else self._buf[:self._n]
            rng = float(seg.max() - seg.min())
            inc = max(0.0, rng - self._range_prev)
            self._range_prev = rng
            return inc * self.N_win / self.R_ref
        # ema_std
        if self._m1 is None:
            self._m1, self._m2 = M, M * M
        else:
            self._m1 = (1 - self.ema_a) * self._m1 + self.ema_a * M
            self._m2 = (1 - self.ema_a) * self._m2 + self.ema_a * M * M
        return float(np.sqrt(max(self._m2 - self._m1 ** 2, 0.0))) / self.S_ref

    def __call__(self, region: int, P: float, P_base: float, gen_speed: float,
                 M_oop: float, M_prev: float, dbeta: float, dtau: float = 0.0,
                 ke_dot: float | None = None, dipc: float = 0.0) -> tuple[float, dict]:
        d_wg = (gen_speed - self.wg_rated) / self.wg_rated
        r_task = 0.0
        if region == R2:
            if ke_dot is None:               # torque channel off: original formula, bit-identical
                r_task = self.w_P * (P / max(P_base, 1.0) - 1.0)
            else:
                # torque channel on — two exploit guards (2026-09-03):
                # (1) cap credited power at rated: in constant-torque mode P = tau_rated * omega
                #     grows with overspeed and the agent farmed that while the label stayed R2;
                # (2) add the rotor-KE flux so draining/storing kinetic energy is reward-neutral.
                self._ke_ema += self._ke_alpha * (ke_dot - self._ke_ema)
                r_task = self.w_P * (min(P + self._ke_ema, self.P_rated) / max(P_base, 1.0) - 1.0)
        elif region == R3:
            r_task = self.w_w * float(np.exp(-abs(d_wg) / self.tau))
        r_load = -self.lam_L[region] * self.load_proxy(M_oop, M_prev)
        r_act = -self.lam_A * ((dbeta / self.kappa) ** 2 + (dtau / self.kappa_tau) ** 2
                               + (dipc / self.kappa_ipc) ** 2)
        r = r_task + r_load + r_act
        return r, {"r_task": r_task, "r_load": r_load, "r_act": r_act, "d_wg": d_wg}
