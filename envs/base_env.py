"""Common residual-pitch environment: observation, safety layer, oracle region, unified reward.

Backends (toy / OpenFAST) only implement `_sim_reset` and `_sim_step`; everything an RL method
sees or is scored on lives here so that all methods share identical interfaces.

Measurement dict returned by backends (all SI, HSS speeds in rad/s):
    t, P (electrical W), gen_speed, rot_speed, gen_torque, v_hub, v_est,
    M_oop (blade-1 root out-of-plane moment, Nm), beta_meas, beta_native (ROSCO cmd before
    offset), min_pit (ROSCO current lower pitch limit), beta_applied, M_twr (optional)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

from controllers.router import R2, R3, OracleRegionRouter
from envs.reward import RegionReward
from envs.coleman import coleman, coleman_inverse
from envs.safety import ResidualSafety, SecondOrderDamper

PROJ = Path(__file__).resolve().parents[1]


def load_yaml(p: str | Path) -> dict:
    with open(p) as f:
        return yaml.safe_load(f)


@dataclass
class EpisodeSpec:
    wind_file: str               # path to .bts
    mean_wind: float             # nominal mean [m/s] (for initial conditions / logging)
    episode_s: float = 150.0
    warmup_s: float = 20.0       # agent inactive, no reward, lets ROSCO/turbine settle


@dataclass
class EnvConfig:
    turbine: dict
    reward: dict
    dbeta_max: float = 0.05
    region_flag_in_obs: bool = False
    use_damper: bool = True
    damper_zeta: float = 1.0
    damper_omega_mult: float = 6.0
    baseline_dir: str | None = None      # cache of paired GSPI trajectories (P_base)
    obs_fa_acc: bool = False             # append tower-top fore-aft acceleration to the observation
    dtau_max_nm: float = 0.0             # R2 torque residual bound [Nm]; 0 disables (action stays 1-D)
    ipc_max_rad: float = 0.0             # R3 dq-frame cyclic-pitch bound [rad]; 0 disables (+2 act, +2 obs)
    ipc_hold_s: float = 0.0              # >0: rotation-held IPC (rollout samples the dq action every
                                         # ipc_hold_s from a separate slow actor; Coquelet-style)
    obs_scales: dict = field(default_factory=lambda: {
        "dwg_dot": 0.05,     # normalised gen accel scale [1/s]
        "v": 25.0,
        "M": None,           # None -> reward M_ref
    })


class ResidualPitchEnv(gym.Env):
    metadata = {"render_modes": []}
    BACKEND = "toy"          # overridden by backends; selects dM_ref

    def __init__(self, cfg: EnvConfig, episodes: list[EpisodeSpec], seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.tb = cfg.turbine
        self.dt = float(self.tb["dt_ctrl_s"])
        self.episodes = episodes
        self.rng = np.random.default_rng(seed)
        self._ep_idx = -1

        self.wg_rated = float(self.tb["rated_gen_speed_rads"])
        self.reward_fn = RegionReward({**cfg.reward, "rated_power_w": self.tb["rated_power_w"]},
                                      self.wg_rated, self.BACKEND, self.dt)
        rr = cfg.reward["region_rule"]
        self.router = OracleRegionRouter(float(self.tb["fine_pitch_rad"]),
                                         np.deg2rad(rr["pitch_threshold_deg"]), rr["hold_s"], self.dt)
        damper = None
        if cfg.use_damper:
            T_rot = 60.0 / float(self.tb["rated_rotor_speed_rpm"])
            wn = cfg.damper_omega_mult * 4.0 * 9.23 / T_rot
            damper = SecondOrderDamper(wn, cfg.damper_zeta, self.dt)
        tau_damper = None
        if cfg.use_damper and cfg.dtau_max_nm > 0.0:
            tau_damper = SecondOrderDamper(wn, cfg.damper_zeta, self.dt)
        ipc_dampers = None
        if cfg.use_damper and cfg.ipc_max_rad > 0.0:
            ipc_dampers = (SecondOrderDamper(wn, cfg.damper_zeta, self.dt),
                           SecondOrderDamper(wn, cfg.damper_zeta, self.dt))
        self.safety = ResidualSafety(cfg.dbeta_max, float(self.tb["min_pitch_rad"]),
                                     float(self.tb["max_pitch_rad"]), damper,
                                     dtau_max_nm=cfg.dtau_max_nm, tq_min_nm=0.0,
                                     tq_max_nm=float(self.tb["max_gen_torque_nm"]),
                                     tau_damper=tau_damper)
        self.safety.tq_speed_cut = 0.98 * self.wg_rated
        self.safety.ipc_max = float(cfg.ipc_max_rad)
        self.safety.ipc_dampers = ipc_dampers
        self._J_lss = float(self.tb["drivetrain_inertia_lss_kgm2"])
        self._gen_eff = float(self.tb["gen_efficiency"])
        # dq-moment observation filter (EMA tau 0.5 s) for the IPC channel
        self._dq_alpha = min(1.0, self.dt / 0.5)
        self._dq = np.zeros(2)

        self.load_key = "fa_acc" if cfg.reward.get("load_signal", "M_oop") == "fa_acc" else "M_oop"
        n_obs = (5 + (2 if cfg.region_flag_in_obs else 0) + (1 if cfg.obs_fa_acc else 0)
                 + (2 if cfg.ipc_max_rad > 0.0 else 0))
        n_act = 1 + (1 if cfg.dtau_max_nm > 0.0 else 0) + (2 if cfg.ipc_max_rad > 0.0 else 0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n_act,), np.float32)

        self.M_scale = cfg.obs_scales["M"] or float(cfg.reward["M_ref_nm"])
        self._baseline = None
        self.log: dict[str, list] = {}

    # ------------------------------------------------------------------ backend hooks
    def _sim_reset(self, spec: EpisodeSpec) -> dict:
        raise NotImplementedError

    def _sim_step(self, pitch_offset: float, tq_offset: float = 0.0, ipc3=None) -> dict:
        raise NotImplementedError

    def _sim_close(self):
        pass

    # ------------------------------------------------------------------ helpers
    def _load_baseline(self, spec: EpisodeSpec):
        self._baseline = None
        if self.cfg.baseline_dir is None:
            return
        name = Path(spec.wind_file).stem
        p = Path(os.path.expanduser(self.cfg.baseline_dir)) / f"{name}.npz"
        if p.exists():
            d = np.load(p)
            self._baseline = (d["t"], d["P"])

    def _P_base(self, t: float, P_now: float) -> float:
        if self._baseline is None:
            return P_now          # r_task(R2) == 0 until a paired baseline exists
        tb, Pb = self._baseline
        return float(np.interp(t, tb, Pb))

    def _obs(self, m: dict, region: int) -> np.ndarray:
        d_wg = (m["gen_speed"] - self.wg_rated) / self.wg_rated
        d_wg_dot = (d_wg - self._prev_dwg) / self.dt if self._prev_dwg is not None else 0.0
        self._prev_dwg = d_wg
        o = [d_wg, d_wg_dot / self.cfg.obs_scales["dwg_dot"], m["beta_meas"],
             m["v_hub"] / self.cfg.obs_scales["v"], m["M_oop"] / self.M_scale]
        if self.cfg.obs_fa_acc:
            o.append(m.get("fa_acc", 0.0))
        if self.cfg.ipc_max_rad > 0.0:
            o += [self._dq[0] / self.M_scale, self._dq[1] / self.M_scale]
        if self.cfg.region_flag_in_obs:
            o += [1.0 if region == R2 else 0.0, 1.0 if region == R3 else 0.0]
        return np.asarray(o, dtype=np.float32)

    def _log(self, m: dict, region: int, dbeta: float, r: float, info: dict):
        for k, v in m.items():
            self.log.setdefault(k, []).append(v)
        self.log.setdefault("region", []).append(region)
        self.log.setdefault("dbeta", []).append(dbeta)
        self.log.setdefault("reward", []).append(r)
        for k, v in info.items():
            self.log.setdefault(k, []).append(v)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and "episode_index" in options:
            self._ep_idx = int(options["episode_index"])
        else:
            self._ep_idx = (self._ep_idx + 1) % len(self.episodes)
        self.spec_ep = self.episodes[self._ep_idx]
        self._load_baseline(self.spec_ep)
        self.log = {}
        self._prev_dwg = None
        self._dq = np.zeros(2)
        self.safety.reset()
        self.reward_fn.reset()
        m = self._sim_reset(self.spec_ep)
        self.router.reset(R3 if self.spec_ep.mean_wind > self.tb["rated_wind_ms"] else R2)
        self.region = self.router.update(m["beta_native"], m["min_pit"])
        self.t_end = self.spec_ep.episode_s
        # warm-up: zero residual, no reward, not part of the RL trajectory
        while m["t"] < self.spec_ep.warmup_s - 1e-9:
            m_prev = m
            m = self._sim_step(0.0)
            region = self.region
            self.region = self.router.update(m["beta_native"], m["min_pit"])
            r, info = self.reward_fn(region, m["P"], self._P_base(m["t"], m["P"]), m["gen_speed"],
                                     m[self.load_key], m_prev[self.load_key], 0.0)
            if self.cfg.ipc_max_rad > 0.0:
                md, mq = coleman((m["M_oop"], m["M_oop2"], m["M_oop3"]), m["azimuth"])
                self._dq += self._dq_alpha * (np.array([md, mq]) - self._dq)
            self._log(m, region, 0.0, r, {**info, "warmup": 1, "dtau": 0.0,
                                          "theta_d": 0.0, "theta_q": 0.0})
        self._m = m
        return self._obs(m, self.region), {"region": self.region}

    def step(self, action):
        m_prev = self._m
        region = self.region
        dbeta = self.safety.apply(float(action[0]), region, m_prev["beta_native"], m_prev["min_pit"])
        i = 1
        dtau = 0.0
        if self.cfg.dtau_max_nm > 0.0:
            dtau = self.safety.apply_tau(float(action[i]), region, m_prev["gen_torque"],
                                         m_prev["gen_speed"])
            i += 1
        ipc3 = None
        theta_d = theta_q = 0.0
        if self.cfg.ipc_max_rad > 0.0:
            theta_d, theta_q = self.safety.apply_ipc(float(action[i]), float(action[i + 1]), region)
            # apply at the azimuth of THIS step (measurement is one step old: advance by omega*dt)
            psi = m_prev["azimuth"] + m_prev["rot_speed"] * self.dt
            ipc3 = coleman_inverse(theta_d, theta_q, psi)
        m = self._sim_step(dbeta, dtau, ipc3)
        # exact rotor-KE flux (electrical-equivalent) so the torque channel cannot profit from
        # draining/storing kinetic energy; None when the channel is off (pitch-only runs unchanged)
        ke_dot = None
        if self.cfg.dtau_max_nm > 0.0:
            ke_dot = 0.5 * self._J_lss * (m["rot_speed"] ** 2 - m_prev["rot_speed"] ** 2) / self.dt \
                     * self._gen_eff
        dipc = float(np.hypot(theta_d, theta_q))
        r, info = self.reward_fn(region, m["P"], self._P_base(m["t"], m["P"]), m["gen_speed"],
                                 m[self.load_key], m_prev[self.load_key], dbeta, dtau, ke_dot, dipc)
        if self.cfg.ipc_max_rad > 0.0:      # dq-moment observation (EMA), from the fresh measurement
            md, mq = coleman((m["M_oop"], m["M_oop2"], m["M_oop3"]), m["azimuth"])
            self._dq += self._dq_alpha * (np.array([md, mq]) - self._dq)
        self._log(m, region, dbeta, r, {**info, "warmup": 0, "dtau": dtau,
                                        "theta_d": theta_d, "theta_q": theta_q})
        # region for the *next* decision uses the fresh ROSCO command
        self.region = self.router.update(m["beta_native"], m["min_pit"])
        self._m = m
        terminated = bool(m["gen_speed"] > 1.3 * self.wg_rated)      # overspeed guard
        truncated = bool(m["t"] >= self.t_end - 1e-9)
        if terminated:
            r -= 100.0
        return self._obs(m, self.region), float(r), terminated, truncated, {"region": region, **info}

    def close(self):
        self._sim_close()

    # ------------------------------------------------------------------ supervisor knobs
    KNOB_NAMES = ("lambda_load_R2", "lambda_load_R3", "w_power", "w_speed", "dbeta_max_R2", "dbeta_max_R3")

    def set_knobs(self, knobs: dict):
        self.reward_fn.set_knobs(knobs)
        self.safety.set_knobs(knobs)

    def knobs(self) -> dict:
        return {**self.reward_fn.knobs(), **self.safety.knobs()}

    def log_arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.log.items()}


def default_config(**over) -> EnvConfig:
    tb = load_yaml(PROJ / "configs" / "turbine" / "nrel5mw.yaml")
    rw = load_yaml(PROJ / "configs" / "reward.yaml")
    ppo = load_yaml(PROJ / "configs" / "ppo.yaml")
    cfg = EnvConfig(turbine=tb, reward=rw, dbeta_max=float(ppo["dbeta_max_rad"]),
                    damper_zeta=float(ppo["damper"]["zeta"]),
                    damper_omega_mult=float(ppo["damper"]["omega_n_mult"]))
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg
