"""LPV-MPC pitch baseline: run deterministic episodes and write evaluate.py-compatible
eval_<tag>.{json,csv} into a run directory, so heldout_table / paper_table pick it up.

    # tuning sweep on the supervisor winds
    python scripts/mpc_baseline.py --means 8 12.5 15 --seeds 1 2 --r 0.3 1 3 10 --tag tune_s12 \
        --out ~/wtrl/exp/mpc --port0 6000 --jobs 4
    # held-out evaluation of one config
    python scripts/mpc_baseline.py --means 8 12.5 15 --seeds 3 4 5 6 --r 1 --tag heldout_s3456 \
        --out ~/wtrl/exp/mpc --port0 6000 --jobs 4

The MPC drives the plant through the same ZMQ residual channel as the RL agents: the sent offset
is (beta_target - beta_native), the safety clamp bounds the total to [current pitch floor,
beta_max], the residual damper is OFF and ROSCO's own 10 deg/s rate limit provides the actuator
physicality. Torque stays native ROSCO everywhere.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import yaml

from agents.rollout import episode_metrics
from controllers.mpc import LPVMPC
from envs.base_env import PROJ, default_config
from envs.factory import baseline_dir, episode_list, make_env
from eval.fitness import baseline_metrics, fitness

DBETA_MAX = 0.35     # residual channel wide open: MPC expresses the full (target - native) offset


def run_episode_mpc(ep_index: int, episodes, tb, cp_path, mpc_kw, port, tag, residual: bool = False) -> dict:
    # `residual` (fairness step 0, 2026-09-11): the MPC's target goes through exactly the RL agents'
    # channel — |target - native| <= the PPO residual bound with the same second-order damper — so
    # the comparison no longer hands the MPC 20 deg of authority against the RL's 2.9 deg.
    if residual:
        cfg = default_config(baseline_dir=baseline_dir("openfast"), use_damper=True)
        dbeta_max = cfg.dbeta_max
    else:
        cfg = default_config(baseline_dir=baseline_dir("openfast"), dbeta_max=DBETA_MAX, use_damper=False)
        dbeta_max = DBETA_MAX
    env = make_env("openfast", episodes, cfg, port=port, work_tag=tag)
    mpc = LPVMPC(tb, cp_path, **mpc_kw)
    hold = max(1, int(round(mpc.Ts / env.dt)))
    obs, _ = env.reset(options={"episode_index": ep_index})
    mpc.reset()
    done, terminated, k, target = False, False, 0, None
    t_solve = 0.0
    while not done:
        m = env._m
        mpc.observe(m.get("fa_acc", 0.0), env.dt, w_meas=m["rot_speed"], v_meas=m["v_est"])
        if k % hold == 0:
            t0 = time.time()
            target = mpc.solve(m["rot_speed"], m["beta_meas"], m["v_est"], m["min_pit"])
            t_solve += time.time() - t0
        a = np.clip((target - m["beta_native"]) / dbeta_max, -1.0, 1.0)
        obs, r, terminated, truncated, info = env.step(np.array([a], np.float32))
        done = terminated or truncated
        k += 1
    L = env.log_arrays()
    # region labels for metrics: the oracle rule keys off ROSCO's native command, which the MPC
    # override distorts (U15 frac_R3 collapsed to 0.4-0.5 and the speed constraint went blind);
    # for a non-ROSCO controller "R3" = above rated wind, plain and controller-independent
    L["region"] = (L["v_hub"] > float(tb["rated_wind_ms"])).astype(np.int8)
    metrics = episode_metrics(L, env.dt, env.wg_rated, env.spec_ep.warmup_s, getattr(env, "outb", None))
    spec = env.spec_ep
    env.close()
    return {"metrics": metrics, "wind_file": spec.wind_file, "mean_wind": spec.mean_wind,
            "terminated": bool(terminated), "wall_solve_s": t_solve, "n_solves": k // hold + 1}


def _worker(args):
    return run_episode_mpc(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--means", nargs="+", type=float, default=[8, 12.5, 15])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--ti", type=float, default=8.0, help="turbulence intensity [%] of the wind bank")
    ap.add_argument("--episode_s", type=float, default=150.0)
    ap.add_argument("--horizon", nargs="+", type=int, default=[20], help="prediction horizon(s) in steps; >1 value = sweep")
    ap.add_argument("--scale", default="v1", choices=["v1", "v2"],
                    help="cost scaling: v1 = original references (err_ref 1, dbeta_ref 0.1 rad); "
                         "v2 = O(1) terms at typical values (err_ref 0.005, dbeta_ref 0.002 rad) so qt can act")
    ap.add_argument("--ts", type=float, default=0.1)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--qt", nargs="+", type=float, default=[0.0], help="tower-velocity weight(s); >1 value = sweep")
    ap.add_argument("--wc_v", nargs="+", type=float, default=[0.25], help="wind LPF corner(s) [rad/s]")
    ap.add_argument("--r", nargs="+", type=float, default=[1.0], help="pitch-rate weight(s); >1 value = sweep")
    ap.add_argument("--fitness_target", default="tower", choices=["tower", "blade"])
    ap.add_argument("--mm_cp", type=float, default=1.0, help="model mismatch: Cp/Ct scale in the controller model")
    ap.add_argument("--mm_ft", type=float, default=1.0, help="model mismatch: tower frequency scale in the controller model")
    ap.add_argument("--mm_m", type=float, default=1.0, help="model mismatch: tower modal-mass scale in the controller model")
    ap.add_argument("--residual", action="store_true",
                    help="drive the plant through the RL agents' bounded, damped residual channel "
                         "(same authority as the RL) instead of the wide-open channel")
    ap.add_argument("--out", default="~/wtrl/exp/mpc")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--port0", type=int, default=6000)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    tb = yaml.safe_load(open(PROJ / "configs" / "turbine" / "nrel5mw.yaml"))
    cp_path = os.path.expanduser("~/wtrl/runs/toy_discon/Cp_Ct_Cq.NREL5MW.txt")
    episodes = episode_list(args.means, args.seeds, ti=args.ti, episode_s=args.episode_s)
    # pair against GSPI baselines RELABELED by wind (v_hub > rated), matching the MPC's labels —
    # oracle-labeled baseline R3 subsets differ at 12.5 m/s and corrupt the R3 MSE/std ratios
    base = {}
    for ep in episodes:
        d = np.load(Path(os.path.expanduser(baseline_dir("openfast"))) / (Path(ep.wind_file).stem + ".npz"))
        L = {k: d[k] for k in d.files if not k.startswith("outb_")}
        L["region"] = (L["v_hub"] > float(tb["rated_wind_ms"])).astype(np.int8)
        outb = {k[5:]: d[k] for k in d.files if k.startswith("outb_")} or None
        base[ep.wind_file] = episode_metrics(L, float(tb["dt_ctrl_s"]), float(tb["rated_gen_speed_rads"]),
                                             ep.warmup_s, outb)
    if not (out / "config.json").exists():
        json.dump({"method": "mpc", "reward": {"fitness_target": args.fitness_target}},
                  open(out / "config.json", "w"), indent=1)

    from itertools import product
    scale_kw = dict(err_ref=0.005, dbeta_ref=0.002) if args.scale == "v2" else {}
    for N_h, r_w, qt_w, wcv in product(args.horizon, args.r, args.qt, args.wc_v):
        mpc_kw = dict(horizon=N_h, ts=args.ts, q=args.q, r=r_w, qt=qt_w, wc_v=wcv, **scale_kw,
                      cp_scale=args.mm_cp, ftower_scale=args.mm_ft, mass_scale=args.mm_m)
        mm = "" if (args.mm_cp, args.mm_ft, args.mm_m) == (1.0, 1.0, 1.0) else f"_cp{args.mm_cp:g}ft{args.mm_ft:g}m{args.mm_m:g}"
        rtag = f"{args.tag}_N{N_h}q{args.q:g}r{r_w:g}qt{qt_w:g}w{wcv:g}{mm}"
        if (out / f"eval_{rtag}.json").exists():      # grids are resumable: a finished point is kept
            print(f"mpc [{rtag}] skip (exists)", flush=True)
            continue
        jobs = [(i, episodes, tb, cp_path, mpc_kw, args.port0 + 7 * i,
                 f"work_mpc{i}", args.residual) for i in range(len(episodes))]
        with mp.Pool(min(args.jobs, len(jobs))) as pool:
            res = pool.map(_worker, jobs)
        fit = fitness(res, base, target=args.fitness_target)
        with open(out / f"eval_{rtag}.csv", "w", newline="") as f:
            keys = ["mean_wind", "wind_file", "terminated"]
            for rr in res:
                keys += [kk for kk in rr["metrics"] if kk not in keys]
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for rr in res:
                w.writerow({"mean_wind": rr["mean_wind"], "wind_file": Path(rr["wind_file"]).stem,
                            "terminated": int(rr["terminated"]), **rr["metrics"]})
        json.dump(fit, open(out / f"eval_{rtag}.json", "w"), indent=1, default=float)
        ws = sum(rr["wall_solve_s"] for rr in res) / max(sum(rr["n_solves"] for rr in res), 1)
        print(f"mpc [{rtag}] target={fit.get('target')} F_strict={fit['F']:.2f} "
              f"F_tol2={fit.get('F_tol2', float('nan')):.2f} tier={fit.get('tier')} "
              f"DELred={fit['del_red_pct']:.2f}% Eloss={fit['energy_loss_pct']:.2f}% "
              f"spd_ratio={fit['speed_std_ratio']:.3f} J={fit['J']:.2f} "
              f"[P {fit['J_power_mse_red_pct']:+.1f} w {fit['J_gen_speed_mse_red_pct']:+.1f} "
              f"T {fit['J_TwrBsMyt_DEL_red_pct']:+.1f} B {fit['J_RootMyc1_DEL_red_pct']:+.1f}] "
              f"(avg solve {1e3 * ws:.1f} ms)", flush=True)


if __name__ == "__main__":
    main()
