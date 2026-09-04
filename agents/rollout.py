"""Episode-level rollout workers.

Each worker owns one environment (toy or OpenFAST, its own ZMQ port) and runs *whole episodes*
with a frozen policy snapshot, returning the trajectory plus episode metrics. The learner
process updates after every batch of episodes (the paper updates once per episode).

A "policy set" maps region -> actor snapshot (or None = zero residual):
    mono / mono_flag : {R2: A, R3: A}
    specialised      : {R2: A2, R3: A3}
    R3-only          : {R2: None, R3: A3}
    GSPI             : {R2: None, R3: None}
Jobs may carry `knobs` (supervisor parameters) which are hot-applied to the worker env.
"""
from __future__ import annotations

import multiprocessing as mp
import traceback

import numpy as np
import torch

from agents.ppo import Actor, RunningMeanStd
from controllers.router import R2, R3
from envs.base_env import EpisodeSpec, default_config
from envs.factory import make_env
from eval.metrics import del_rainflow


def episode_metrics(L: dict, dt: float, wg_rated: float, warmup_s: float, outb: dict | None = None) -> dict:
    """Metrics of one episode from the env log (works for training, evaluation and the GSPI
    baseline npz alike). Regions are the oracle labels stored in the log."""
    act = L["warmup"] == 0
    r2 = act & (L["region"] == R2)
    r3 = act & (L["region"] == R3)
    out = {
        "reward_sum": float(L["reward"][act].sum()),
        "reward_mean": float(L["reward"][act].mean()),
        "frac_R3": float(r3.sum() / max(act.sum(), 1)),
        "energy_MWh": float(L["P"][act].sum() * dt / 3.6e9),
        "gen_speed_std": float(L["gen_speed"][act].std()),
        "gen_speed_std_R3": float(L["gen_speed"][r3].std()) if r3.sum() > 100 else float("nan"),
        "gen_speed_mae_R3": float(np.abs(L["gen_speed"][r3] - wg_rated).mean() / wg_rated) if r3.sum() > 100 else float("nan"),
        "power_mae_R3": float(np.abs(L["P"][r3] - 5.0e6).mean() / 5.0e6) if r3.sum() > 100 else float("nan"),
        "gen_speed_mse_R3": float((((L["gen_speed"][r3] - wg_rated) / wg_rated) ** 2).mean()) if r3.sum() > 100 else float("nan"),
        "power_mse_R3": float((((L["P"][r3] - 5.0e6) / 5.0e6) ** 2).mean()) if r3.sum() > 100 else float("nan"),
        "gen_speed_max_rel": float(L["gen_speed"][act].max() / wg_rated),
        "pitch_travel_deg": float(np.rad2deg(np.abs(np.diff(L["beta_meas"][act])).sum())),
        "dbeta_abs_mean_deg": float(np.rad2deg(np.abs(L["dbeta"][act]).mean())),
        "M_oop_mean_MNm": float(L["M_oop"][act].mean() / 1e6),
        "RootMoop_DEL_MNm": float(del_rainflow(L["M_oop"][act], dt, 10) / 1e6),
    }
    if "theta_d" in L:      # IPC telemetry: mean dq amplitude actually used (supervisor diagnostic)
        out["ipc_amp_deg"] = float(np.rad2deg(np.hypot(L["theta_d"], L["theta_q"])[act].mean()))
    # spectral side-information for the supervisor (optimize_anything: actionable feedback beats score-only):
    # share of pitch-rate power in the tower fore-aft band (0.25-0.40 Hz, 5 MW ~0.32 Hz) and tower-top accel RMS
    try:
        b = L["beta_meas"][act]
        if len(b) > 2048:
            rate = np.diff(b) / dt
            f = np.fft.rfftfreq(len(rate), dt)
            pw = np.abs(np.fft.rfft(rate - rate.mean())) ** 2
            tot = pw[(f > 0.05) & (f < 5.0)].sum()
            out["pitch_rate_power_tower_band_frac"] = float(pw[(f >= 0.25) & (f <= 0.40)].sum() / max(tot, 1e-12))
            out["pitch_rate_power_3P_band_frac"] = float(pw[(f >= 0.5) & (f <= 0.75)].sum() / max(tot, 1e-12))
        if "fa_acc" in L:
            out["fa_acc_rms"] = float(np.sqrt(np.mean(L["fa_acc"][act] ** 2)))
    except Exception:  # noqa: BLE001
        pass
    for sel, name in ((r2, "R2"), (r3, "R3")):
        if sel.any():
            out[f"r_task_{name}"] = float(L["r_task"][sel].mean())
            out[f"r_load_{name}"] = float(L["r_load"][sel].mean())
            out[f"n_{name}"] = int(sel.sum())
    if outb is not None and "TwrBsMyt" in outb:
        k = outb["Time"] >= warmup_s
        dto = float(outb["Time"][1] - outb["Time"][0])
        out["TwrBsMyt_DEL_MNm"] = float(del_rainflow(outb["TwrBsMyt"][k], dto, 4) / 1e3)
        out["RootMyc1_DEL_MNm"] = float(del_rainflow(outb["RootMyc1"][k], dto, 10) / 1e3)
    return out


def _build_actors(policy_set: dict, obs_dim: int, hidden, act_dim: int = 1) -> dict:
    actors = {}
    for reg, snap in policy_set.items():
        if snap is None:
            actors[reg] = None
            continue
        # act_dim per snapshot from its own weights: regional and slow-IPC actors differ
        ad = int(snap["actor"]["log_std"].shape[0])
        a = Actor(obs_dim, ad, hidden)
        a.load_state_dict(snap["actor"])
        a.eval()
        rms = RunningMeanStd((obs_dim,))
        rms.load(snap["obs_rms"])
        actors[reg] = (a, rms)
    return actors


def run_episode(env, policy_set: dict, hidden, episode_index: int, deterministic: bool,
                seed: int | None = None, knobs: dict | None = None, keep_log: bool = False) -> dict:
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    actors = _build_actors(policy_set, obs_dim, hidden, act_dim)
    if knobs:
        env.set_knobs(knobs)
    if seed is not None:
        torch.manual_seed(seed)
    obs, info = env.reset(options={"episode_index": episode_index})
    O, A, LP, RW, RG = [], [], [], [], []
    # rotation-held IPC (Coquelet-style train-slow): a separate slow actor decides (theta_d,
    # theta_q) once per `ipc_hold_s` and the value is held; its macro-transitions are returned
    # under "ipc" for a dedicated slow learner. Without an "IPC" snapshot behaviour is unchanged.
    slow = actors.get("IPC")
    hold = int(round(float(getattr(env.cfg, "ipc_hold_s", 0.0)) / env.dt)) if slow else 0
    iO, iA, iLP, iRW = [], [], [], []
    held = np.zeros(2, np.float32)
    step_i = 0
    done, terminated = False, False
    while not done:
        region = env.region
        pol = actors.get(region)
        if pol is None:
            a, lp = np.zeros(act_dim, np.float32), 0.0
        else:
            actor, rms = pol
            a, lp = actor.act(rms.normalize(obs).astype(np.float32), deterministic)
        if hold:
            if step_i % hold == 0:
                s_actor, s_rms = slow
                held, slp = s_actor.act(s_rms.normalize(obs).astype(np.float32), deterministic)
                iO.append(obs); iA.append(held); iLP.append(slp); iRW.append([])
            a = np.concatenate([np.asarray(a, np.float32).reshape(-1)[:act_dim - 2], held])
        nobs, r, terminated, truncated, info = env.step(a)
        if hold:
            iRW[-1].append(r)
        O.append(obs); A.append(a[:act_dim - 2] if hold else a); LP.append(lp); RW.append(r); RG.append(region)
        obs = nobs
        step_i += 1
        done = terminated or truncated
    L = env.log_arrays()
    metrics = episode_metrics(L, env.dt, env.wg_rated, env.spec_ep.warmup_s, getattr(env, "outb", None))
    out = {
        "obs": np.asarray(O, np.float32), "act": np.asarray(A, np.float32).reshape(len(A), -1),
        "logp": np.asarray(LP, np.float32), "rew": np.asarray(RW, np.float32),
        "region": np.asarray(RG, np.int8), "last_obs": obs.astype(np.float32),
        "terminated": bool(terminated), "episode_index": episode_index,
        "mean_wind": env.spec_ep.mean_wind, "wind_file": env.spec_ep.wind_file,
        "metrics": metrics, "knobs": env.knobs(), "log": L if keep_log else None,
    }
    if hold:
        out["ipc"] = {"obs": np.asarray(iO, np.float32), "act": np.asarray(iA, np.float32),
                      "logp": np.asarray(iLP, np.float32),
                      "rew_win": [np.asarray(w, np.float32) for w in iRW],
                      "last_obs": obs.astype(np.float32), "terminal": bool(terminated)}
    return out


def _worker(idx, backend, episodes, cfg_over, port, conn, hidden, tag: str = "work", max_retries: int = 2):
    """One env per worker. A failed episode (e.g. OpenFAST/ZMQ timeout) is retried with a freshly
    built env up to `max_retries` times before the error is propagated to the learner."""
    torch.set_num_threads(1)
    env = None
    try:
        cfg = default_config(**cfg_over)
        env = make_env(backend, episodes, cfg=cfg, port=port, work_tag=f"{tag}_w{idx}", seed=idx)
        while True:
            msg = conn.recv()
            if msg is None:
                break
            last = None
            for attempt in range(max_retries + 1):
                try:
                    res = run_episode(env, msg["policy_set"], hidden, msg["episode_index"], msg["deterministic"],
                                      msg.get("seed"), msg.get("knobs"), msg.get("keep_log", False))
                    if attempt:
                        res["retries"] = attempt
                    break
                except Exception:  # noqa: BLE001
                    last = traceback.format_exc()
                    try:
                        env.close()
                    except Exception:  # noqa: BLE001
                        pass
                    env = make_env(backend, episodes, cfg=cfg, port=port, work_tag=f"{tag}_w{idx}", seed=idx)
                    res = None
            if res is None:
                conn.send({"error": f"episode {msg['episode_index']} failed {max_retries + 1} times:\n{last}"})
                continue
            conn.send(res)
        env.close()
    except Exception:
        conn.send({"error": traceback.format_exc()})


class WorkerPool:
    def __init__(self, n: int, backend: str, episodes: list[EpisodeSpec], cfg_over: dict,
                 hidden=(64, 64), port0: int = 5800, tag: str = "work"):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        self.backend = backend
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(i, backend, episodes, cfg_over, port0 + i, b, hidden, tag),
                            daemon=True)
            p.start()
            self.conns.append(a); self.procs.append(p)

    def run(self, jobs: list[dict]) -> list[dict]:
        """jobs: list of {'policy_set','episode_index','deterministic','seed','knobs'}; processed in
        waves of len(workers)."""
        results = []
        for i in range(0, len(jobs), len(self.conns)):
            wave = jobs[i:i + len(self.conns)]
            for c, j in zip(self.conns, wave):
                c.send(j)
            for c, _ in zip(self.conns, wave):
                r = c.recv()
                if "error" in r:
                    raise RuntimeError("worker failed:\n" + r["error"])
                results.append(r)
        return results

    def close(self):
        for c in self.conns:
            try:
                c.send(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
