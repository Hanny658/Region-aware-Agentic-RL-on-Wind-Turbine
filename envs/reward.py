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

import math

import numpy as np

from controllers.router import R2, R3

PROXIES = ("increment", "range_inc", "ema_std")

# --------------------------------------------------------------------- LLM-written reward (v3)
# `llm_reward` supervision lets the LLM replace the per-step reward with one expression over a
# fixed, already-normalised set of scalars. The expression is compiled once, evaluated with no
# builtins and a whitelisted namespace, and validated on a grid before it is allowed to run: a
# reward that is non-finite or absurdly scaled would otherwise destroy a whole training wave.
# The ground-truth fitness never uses the reward, so a self-gaming expression is caught by
# evaluation rather than by trusting the model.
_SAFE = {"exp": math.exp, "log": math.log, "sqrt": math.sqrt, "tanh": math.tanh,
         "abs": abs, "min": min, "max": max, "pi": math.pi}
_GRID = [(reg, d, p, l, a)
         for reg in (0, 1) for d in (-0.05, -0.005, 0.0, 0.005, 0.05)
         for p in (0.9, 1.0, 1.05) for l in (0.0, 1.0, 5.0) for a in (0.0, 0.1, 1.0)]
REWARD_CODE_VARS = ("region", "d_wg", "p_ratio", "load", "act")
REWARD_CODE_VARS_V2 = ("region", "d_wg", "p_ratio", "load_t", "load_b", "act")
REWARD_CODE_VARS_V3 = ("region", "region_w", "d_wg", "p_ratio", "load_t", "load_b", "act")


class _LoadProxy:
    """One stateful per-step load proxy (same three variants as RegionReward.load_proxy), so that
    reward v2 can score the tower and the blade signal simultaneously."""

    def __init__(self, kind: str, dM_ref: float, R_ref: float, S_ref: float, N_win: int, ema_a: float):
        self.kind, self.dM_ref, self.R_ref, self.S_ref, self.N_win, self.ema_a = kind, dM_ref, R_ref, S_ref, N_win, ema_a
        self.reset()

    def reset(self):
        self._buf = np.zeros(self.N_win)
        self._n = 0
        self._range_prev = 0.0
        self._m1 = self._m2 = None

    def __call__(self, M: float, M_prev: float) -> float:
        if self.kind == "increment":
            return abs(M - M_prev) / self.dM_ref
        if self.kind == "range_inc":
            self._buf[self._n % self.N_win] = M
            self._n += 1
            seg = self._buf if self._n >= self.N_win else self._buf[:self._n]
            rng = float(seg.max() - seg.min())
            inc = max(0.0, rng - self._range_prev)
            self._range_prev = rng
            return inc * self.N_win / self.R_ref
        if self._m1 is None:
            self._m1, self._m2 = M, M * M
        else:
            self._m1 = (1 - self.ema_a) * self._m1 + self.ema_a * M
            self._m2 = (1 - self.ema_a) * self._m2 + self.ema_a * M * M
        return float(np.sqrt(max(self._m2 - self._m1 ** 2, 0.0))) / self.S_ref


def compile_reward_code(src: str, limit: float = 1.0e4, version: str = "v1"):
    """Compile an LLM-written reward expression; raises ValueError if it is unusable.
    v1 exposes (region, d_wg, p_ratio, load, act); v2 exposes (region, d_wg, p_ratio, load_t,
    load_b, act) — the tower and blade proxies separately, because both DELs are in the objective."""
    if not isinstance(src, str) or len(src) > 400:
        raise ValueError("reward code must be a string under 400 characters")
    code = compile(src, "<llm_reward>", "eval")
    ns = dict(_SAFE)
    ns["__builtins__"] = {}

    if version == "v3":
        def f(region, region_w, d_wg, p_ratio, load_t, load_b, act):
            return float(eval(code, ns, {"region": region, "region_w": region_w, "d_wg": d_wg, "p_ratio": p_ratio,
                                         "load_t": load_t, "load_b": load_b, "act": act}))
        grid = [(reg, rw, d, p, lt, lb, a) for reg, d, p, lt, a in _GRID for lb in (0.0, 2.0) for rw in (0, 1)]
    elif version == "v2":
        def f(region, d_wg, p_ratio, load_t, load_b, act):
            return float(eval(code, ns, {"region": region, "d_wg": d_wg, "p_ratio": p_ratio,
                                         "load_t": load_t, "load_b": load_b, "act": act}))
        grid = [(reg, d, p, lt, lb, a) for reg, d, p, lt, a in _GRID for lb in (0.0, 2.0)]
    else:
        def f(region, d_wg, p_ratio, load, act):
            return float(eval(code, ns, {"region": region, "d_wg": d_wg, "p_ratio": p_ratio,
                                         "load": load, "act": act}))
        grid = _GRID

    for args in grid:
        try:
            v = f(*args)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"reward code failed on {args}: {type(e).__name__}: {e}") from None
        if not math.isfinite(v) or abs(v) > limit:
            raise ValueError(f"reward code gives {v} on {args} (must be finite, |r| <= {limit:g})")
    return f


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
        self.code_fn, self.code_src = None, None
        self.N_win = max(1, int(round(float(cfg.get("window_s", 10.0)) / dt)))
        # ---- reward v2 (2026-09-11): MSE-matched speed term + tower AND blade load proxies.
        # The objective J carries both DELs and the R3 speed MSE; v1 could only serve one load
        # signal and its exp(-|dw|/0.02) term is nearly flat at the typical error of 0.005.
        self.version = str(cfg.get("version", "v1"))
        # v3 (2026-09-11, fairness step 1): the speed term is gated by the WIND label (v_hub > rated),
        # the same subset the objective's MSE terms use, instead of the router's region — under v2
        # the transition steps that J scores as R3 carried no speed penalty at all (roadmap v2 §9)
        self.speed_by_wind = self.version == "v3"
        self.err_ref = float(cfg.get("speed_err_ref", 0.005))
        self.lam_T = float(cfg.get("lambda_tower", cfg.get("lambda_load", 1.0) if not isinstance(cfg.get("lambda_load"), dict) else 1.0))
        self.lam_B = float(cfg.get("lambda_blade", self.lam_T))
        self._px_t = self._px_b = None
        if self.version in ("v2", "v3"):
            ema_a = dt / float(cfg.get("ema_tau_s", 5.0))
            kind = cfg.get("load_proxy", "range_inc")
            fa = {k: (float(cfg[k].get(backend, 1.0)) if isinstance(cfg.get(k), dict) else float(cfg.get(k) or 1.0))
                  for k in ("fa_dM_ref", "fa_R_ref", "fa_S_ref")}   # toy twin: no fa_acc, unit scale
            self._px_t = _LoadProxy(kind, fa["fa_dM_ref"], fa["fa_R_ref"], fa["fa_S_ref"], self.N_win, ema_a)
            self._px_b = _LoadProxy(kind, _by_backend(cfg["dM_ref_nm"], backend),
                                    _by_backend(cfg.get("R_ref_nm", 1.0), backend),
                                    _by_backend(cfg.get("S_ref_nm", 1.0), backend), self.N_win, ema_a)
        self.ema_a = dt / float(cfg.get("ema_tau_s", 5.0))
        self.reset()

    def reset(self):
        if self._px_t is not None:
            self._px_t.reset(); self._px_b.reset()
        self._buf = np.zeros(self.N_win)
        self._n = 0
        self._range_prev = 0.0
        self._m1 = self._m2 = None
        self._ke_ema = 0.0

    def set_code(self, src: str | None):
        """Install an LLM-written reward expression (None restores the built-in formula)."""
        if src is None or not str(src).strip():
            self.code_fn, self.code_src = None, None
            return
        self.code_fn = compile_reward_code(str(src), version=self.version)
        self.code_src = str(src).strip()

    def set_knobs(self, knobs: dict):
        if "reward_code" in knobs:
            self.set_code(knobs["reward_code"])
        if "w_power" in knobs:
            self.w_P = float(knobs["w_power"])
        if "w_speed" in knobs:
            self.w_w = float(knobs["w_speed"])
        if "lambda_load_R2" in knobs:
            self.lam_L[R2] = float(knobs["lambda_load_R2"])
        if "lambda_load_R3" in knobs:
            self.lam_L[R3] = float(knobs["lambda_load_R3"])
        if "lambda_tower" in knobs:
            self.lam_T = float(knobs["lambda_tower"])
        if "lambda_blade" in knobs:
            self.lam_B = float(knobs["lambda_blade"])

    def knobs(self) -> dict:
        if self.version in ("v2", "v3"):
            k = {"w_power": self.w_P, "w_speed": self.w_w,
                 "lambda_tower": self.lam_T, "lambda_blade": self.lam_B}
        else:
            k = {"w_power": self.w_P, "w_speed": self.w_w,
                 "lambda_load_R2": self.lam_L[R2], "lambda_load_R3": self.lam_L[R3]}
        if getattr(self, "code_src", None):
            k["reward_code"] = self.code_src
        return k

    def _call_v2(self, region, P, P_base, d_wg, dbeta, dtau, dipc, aux, region_w=None) -> tuple[float, dict]:
        """aux = (fa_acc, fa_acc_prev, M_oop, M_oop_prev). Tower proxy is 0 on a backend without
        fa_acc (the toy twin), which keeps the twin usable for smoke tests. `region_w` = wind label
        (v3): the speed term follows it; the power term always follows the router's region."""
        fa, fa_p, mo, mo_p = aux if aux is not None else (0.0, 0.0, 0.0, 0.0)
        load_t = self._px_t(fa, fa_p) if fa == fa else 0.0
        load_b = self._px_b(mo, mo_p)
        act = ((dbeta / self.kappa) ** 2 + (dtau / self.kappa_tau) ** 2 + (dipc / self.kappa_ipc) ** 2)
        p_ratio = P / max(P_base, 1.0)
        if region_w is None:
            region_w = region
        if self.code_fn is not None:
            if self.version == "v3":
                r = self.code_fn(region, region_w, d_wg, p_ratio, load_t, load_b, act)
            else:
                r = self.code_fn(region, d_wg, p_ratio, load_t, load_b, act)
            return r, {"r_task": r, "r_load": -(self.lam_T * load_t + self.lam_B * load_b),
                       "r_act": -self.lam_A * act, "d_wg": d_wg}
        r_task = 0.0
        if region == R2:
            r_task += self.w_P * (p_ratio - 1.0)
        if (region_w if self.speed_by_wind else region) == R3:
            r_task -= self.w_w * (d_wg / self.err_ref) ** 2
        r_load = -(self.lam_T * load_t + self.lam_B * load_b)
        r_act = -self.lam_A * act
        return r_task + r_load + r_act, {"r_task": r_task, "r_load": r_load, "r_act": r_act, "d_wg": d_wg}

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
                 ke_dot: float | None = None, dipc: float = 0.0,
                 aux: tuple | None = None, region_w: int | None = None) -> tuple[float, dict]:
        d_wg = (gen_speed - self.wg_rated) / self.wg_rated
        if self.version in ("v2", "v3"):
            return self._call_v2(region, P, P_base, d_wg, dbeta, dtau, dipc, aux, region_w=region_w)
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
        load = self.load_proxy(M_oop, M_prev)
        act = ((dbeta / self.kappa) ** 2 + (dtau / self.kappa_tau) ** 2
               + (dipc / self.kappa_ipc) ** 2)
        if self.code_fn is not None:        # LLM-written reward: it replaces the whole expression
            r = self.code_fn(region, d_wg, P / max(P_base, 1.0), load, act)
            return r, {"r_task": r, "r_load": -self.lam_L[region] * load, "r_act": -self.lam_A * act,
                       "d_wg": d_wg}
        r_load = -self.lam_L[region] * load
        r_act = -self.lam_A * act
        r = r_task + r_load + r_act
        return r, {"r_task": r_task, "r_load": r_load, "r_act": r_act, "d_wg": d_wg}
