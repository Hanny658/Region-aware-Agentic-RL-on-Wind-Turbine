"""LPV-MPC collective-pitch baseline (2026-09-03, baselines = GSPI + MPC).

Three-state model — rotor + first tower fore-aft mode — the minimum the literature uses for
pitch MPC (a 1-state rotor model is provably unstable here: it wins on the 1-DOF toy twin,
speed std 0.03 vs GSPI 0.48, but on OpenFAST it pumps the unmodeled ~0.32 Hz tower mode,
tower DEL +100..240 %; see roadmap §16):

    J  dw/dt   = T_aero(v - xd, w, beta) - T_gen(w)
    m  d2x/dt2 = F_thrust(v - xd, w, beta) - c xd - k x

Linearised at the CURRENT operating point (w, beta, v_est, tower state estimate) from the ROSCO
Cp/Ct tables every update (Ts = 0.1 s), discretised exactly (ZOH via expm), condensed QP (OSQP):

    min sum q ((w_k - w_rated)/w_rated)^2 + qt (xd_k / 0.2)^2 + r (dbeta_k / 0.1)^2
    s.t. beta in [pitch floor (incl. peak shaving), beta_max], |dbeta| <= rate_max * Ts

Tower states are estimated from the measured tower-top acceleration by leaky integration
(leak 0.03 Hz << the 0.324 Hz mode). Torque stays native ROSCO, so below rated the optimum
rides the pitch floor. Wind over the horizon is held at ROSCO's estimate (no preview).
"""
from __future__ import annotations

import numpy as np

from envs.toy_env import _Bilinear, read_cp_ct_tables

F1_TOWER_HZ = 0.324          # NREL 5 MW first tower fore-aft frequency
ZETA_STRUCT = 0.01           # structural damping ratio (aero damping enters via dF/dv)
M_MODAL = 4.37e5             # kg: rotor+nacelle (350 t) + 0.25 x tower mass (347.5 t)


class LPVMPC:
    def __init__(self, tb: dict, cp_table_path: str, horizon: int = 20, ts: float = 0.1,
                 q: float = 1.0, r: float = 1.0, qt: float = 0.0, wc_v: float = 0.25):
        import osqp
        import scipy.sparse as sp
        from scipy.linalg import expm
        self._osqp, self._sp, self._expm = osqp, sp, expm
        pitch, tsr, cp, ct = read_cp_ct_tables(cp_table_path)
        self.Cp = _Bilinear(pitch, tsr, np.maximum(cp, 0.0))
        self.Ct = _Bilinear(pitch, tsr, np.maximum(ct, 0.0))
        self.R = float(tb["rotor_radius_m"])
        self.rho = float(tb["rho_air"])
        self.J = float(tb["drivetrain_inertia_lss_kgm2"])
        self.w_rated = float(tb["rated_rotor_speed_rpm"]) * 2 * np.pi / 60.0     # LSS rad/s
        self.tq_rated_lss = float(tb["rated_gen_torque_nm"]) * float(tb["gearbox_ratio"])
        self.beta_max = float(tb["max_pitch_rad"])
        self.rate_max = float(tb["max_pitch_rate_rads"])
        self.N, self.Ts = int(horizon), float(ts)
        self.q, self.r, self.qt = float(q), float(r), float(qt)
        self.k_t = M_MODAL * (2 * np.pi * F1_TOWER_HZ) ** 2
        self.c_t = 2 * ZETA_STRUCT * M_MODAL * (2 * np.pi * F1_TOWER_HZ)
        # tower state estimator (leaky double integration of measured fa_acc)
        self.leak = 2 * np.pi * 0.03
        self.xd_hat = 0.0
        self.x_hat = 0.0
        # rotor-speed low-pass (ROSCO feeds its PC loop GenSpeedF, ~1 rad/s corner; raw OpenFAST
        # speed carries 3P/drivetrain/tower content that a 10 Hz MPC otherwise chases)
        self.wc_speed = 50.0        # near-raw; heavier speed filtering only adds destabilising lag
        # wind input low-pass (rotor-effective wind / dynamic-inflow timescale). Without it the
        # v-driven affine equilibrium acts as an overconfident static-Cp feedforward and the loop
        # hunts at ~0.15 Hz (diagnosed 2026-09-03; true hub wind hunts too, so it is not the WSE)
        self.wc_v = float(wc_v)
        self.w_f: float | None = None
        self.v_f: float | None = None

    def reset(self):
        self.xd_hat = self.x_hat = 0.0
        self.w_f = self.v_f = None

    def observe(self, fa_acc: float, dt: float, w_meas: float | None = None,
                v_meas: float | None = None):
        self.xd_hat += dt * (fa_acc - self.leak * self.xd_hat)
        self.x_hat += dt * (self.xd_hat - self.leak * self.x_hat)
        if w_meas is not None:
            self.w_f = w_meas if self.w_f is None else self.w_f + dt * self.wc_speed * (w_meas - self.w_f)
        if v_meas is not None:
            self.v_f = v_meas if self.v_f is None else self.v_f + dt * self.wc_v * (v_meas - self.v_f)

    def _aero(self, v, w, beta_rad):
        v = max(v, 0.5)
        lam = np.clip(w * self.R / v, 1e-3, 25.0)
        A = 0.5 * self.rho * np.pi * self.R ** 2
        cp = self.Cp(np.rad2deg(beta_rad), lam)
        ct = self.Ct(np.rad2deg(beta_rad), lam)
        return A * cp * v ** 3 / max(w, 0.05), A * ct * v ** 2

    def solve(self, w: float, beta: float, v_est: float, floor: float) -> float:
        """-> collective pitch target [rad] for the next Ts. Call observe() every sim step."""
        if self.w_f is not None:
            w = self.w_f
        if self.v_f is not None:
            v_est = self.v_f
        v_rel = max(v_est - self.xd_hat, 0.5)
        d = 1e-3
        T0, F0 = self._aero(v_rel, w, beta)
        Tw, Fw = [(a - b) / (2 * d) for a, b in zip(self._aero(v_rel, w + d, beta),
                                                    self._aero(v_rel, w - d, beta))]
        Tb, Fb = [(a - b) / (2 * d) for a, b in zip(self._aero(v_rel, w, beta + d),
                                                    self._aero(v_rel, w, beta - d))]
        Tv, Fv = [(a - b) / (2 * d) for a, b in zip(self._aero(v_rel + d, w, beta),
                                                    self._aero(v_rel - d, w, beta))]
        K_lss = self.tq_rated_lss / self.w_rated ** 2
        Tg0 = min(K_lss * w * w, self.tq_rated_lss)
        Tgw = 2 * K_lss * w if w < 0.98 * self.w_rated else 0.0
        # states s = [w, x, xd] (absolute), input beta:  ds/dt = Ac s + Bc beta + cc
        J, m = self.J, M_MODAL
        Ac = np.array([[(Tw - Tgw) / J, 0.0, -Tv / J],
                       [0.0, 0.0, 1.0],
                       [Fw / m, -self.k_t / m, -(self.c_t + Fv) / m]])
        Bc = np.array([[Tb / J], [0.0], [Fb / m]])
        cc = np.array([(T0 - Tg0 - Tw * w - Tb * beta + Tv * self.xd_hat) / J + Tgw * w / J,
                       0.0,
                       (F0 - Fw * w - Fb * beta + Fv * self.xd_hat) / m
                       + self.k_t * self.x_hat / m + self.c_t * self.xd_hat / m])
        # note: cc is built so that Ac s0 + Bc beta0 + cc reproduces the nonlinear derivatives at s0
        M = np.zeros((5, 5))
        M[:3, :3] = Ac * self.Ts
        M[:3, 3:4] = Bc * self.Ts
        M[:3, 4] = cc * self.Ts
        E = self._expm(M)
        Ad, Bd, cd = E[:3, :3], E[:3, 3], E[:3, 4]
        # rollout s_k = F_k s0 + G U + h
        N = self.N
        s0 = np.array([w, self.x_hat, self.xd_hat])
        Fmats = []
        P_ = np.eye(3)
        for _ in range(N):
            P_ = Ad @ P_
            Fmats.append(P_.copy())
        G = np.zeros((3 * N, N))
        h = np.zeros(3 * N)
        acc_h = np.zeros(3)
        for k_ in range(N):
            acc_h = Ad @ acc_h + cd
            h[3 * k_:3 * k_ + 3] = acc_h
            for j in range(k_ + 1):
                blk = Bd if j == k_ else Fmats[k_ - j - 1] @ Bd
                G[3 * k_:3 * k_ + 3, j] = blk
        Fs0 = np.concatenate([Fm @ s0 for Fm in Fmats])
        # cost selection: w rows (0,3,6..), xd rows (2,5,8..)
        iw = np.arange(N) * 3
        ix = iw + 2
        Wq = np.zeros(3 * N)
        Wq[iw] = self.q / self.w_rated ** 2
        Wq[ix] = self.qt / 0.2 ** 2
        ref = np.zeros(3 * N)
        ref[iw] = self.w_rated
        rr = self.r / 0.1 ** 2
        D = np.eye(N) - np.eye(N, k=-1)
        d0 = np.zeros(N); d0[0] = beta
        e = Fs0 + h - ref
        P = 2 * ((G.T * Wq) @ G + rr * D.T @ D)
        qv = 2 * ((G.T * Wq) @ e - rr * D.T @ d0)
        A = np.vstack([np.eye(N), D])
        lb = np.concatenate([np.full(N, max(floor, 0.0)), -np.full(N, self.rate_max * self.Ts) + d0])
        ub = np.concatenate([np.full(N, self.beta_max), np.full(N, self.rate_max * self.Ts) + d0])
        prob = self._osqp.OSQP()
        prob.setup(P=self._sp.csc_matrix(P), q=qv, A=self._sp.csc_matrix(A), l=lb, u=ub,
                   verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=2000)
        res = prob.solve()
        if res.x is None or not np.isfinite(res.x[0]):
            return float(np.clip(beta, floor, self.beta_max))
        return float(np.clip(res.x[0], floor, self.beta_max))
