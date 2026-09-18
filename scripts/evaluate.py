"""Deterministic evaluation of a trained run (ckpt + final knobs) on given wind means / seeds,
paired against the GSPI baselines of the same backend. Writes <out>/eval_<tag>.csv and prints
the fitness summary. Baselines for every (mean, seed) must exist (scripts/make_baselines.py).

    python scripts/evaluate.py --run ~/wtrl/exp/toy_s3_llm --backend toy --means 8 12.5 15 --seeds 1
    python scripts/evaluate.py --run ~/wtrl/exp/toy_s3_llm --backend openfast --means 8 12.5 15 --seeds 1 2 3 --workers 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from agents.rollout import WorkerPool
from controllers.router import R2, R3
from envs.base_env import PROJ
from envs.factory import baseline_dir, episode_list, parse_ti, ti_label
from eval.fitness import baseline_metrics, fitness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    ap.add_argument("--backend", default="toy", choices=["toy", "openfast"])
    ap.add_argument("--means", nargs="+", type=float, default=[8, 12.5, 15])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--ti", type=parse_ti, default=8.0,
                    help="turbulence intensity [%] selecting the wind bank / baseline files "
                         "(U<mean>_TI<ti>_S<seed>); the stress classes use 14 and 22")
    ap.add_argument("--episode_s", type=float, default=150.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--port0", type=int, default=5900)
    ap.add_argument("--gspi", action="store_true",
                    help="evaluate the zero-residual base controller (GSPI, or the MPC base of an MPC-base run) "
                         "instead of the checkpoint")
    ap.add_argument("--base_mm_cp", type=float, default=None, help="override the MPC base's Cp/Ct model scale")
    ap.add_argument("--base_mm_ft", type=float, default=None, help="override the MPC base's tower-frequency model scale")
    ap.add_argument("--base_mm_m", type=float, default=None, help="override the MPC base's modal-mass model scale")
    ap.add_argument("--base_mpc_json", default=None,
                    help="JSON dict merged into the MPC base's LPVMPC keyword arguments (e.g. '{\"qt\": 0}')")
    ap.add_argument("--out", default=None, help="directory for eval_<tag>.{csv,json} and logs_<tag>/ (default: the run)")
    ap.add_argument("--relabel_wind", action="store_true",
                    help="score R3 by v_hub > rated on BOTH sides instead of the oracle rule; use it for "
                         "reference controllers that alter ROSCO's own pitch command (tower damper, MPC)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dump_log", action="store_true",
                    help="also write every episode's per-step log (env log + OpenFAST channels) to "
                         "<run>/logs_<tag>/<wind>.npz, the layout of the GSPI baseline files (trajectory figures)")
    ap.add_argument("--fitness_target", default=None, choices=["blade", "tower"])
    args = ap.parse_args()

    run = Path(os.path.expanduser(args.run))
    cfg_run = json.load(open(run / "config.json"))
    summ = json.load(open(run / "summary.json")) if (run / "summary.json").exists() else {}
    knobs = summ.get("final_knobs", cfg_run.get("knobs0"))
    if cfg_run.get("objective") == "J" and not args.relabel_wind:
        # J-trained runs were selected with wind-labelled R3 subsets (roadmap v2, D3): score them
        # the same way so the held-out J is comparable to the training-time J
        args.relabel_wind = True
        print("objective J run: R3 subsets labelled by wind on both sides (--relabel_wind)")
    ppo_yaml = yaml.safe_load(open(PROJ / "configs" / "ppo.yaml"))
    hidden = tuple(ppo_yaml["hidden"])
    flag = cfg_run["method"] == "mono_flag"
    cfg_over = {"baseline_dir": baseline_dir(args.backend), "region_flag_in_obs": flag, "reward": cfg_run["reward"],
                "obs_fa_acc": bool(cfg_run.get("obs_fa_acc", False)),
                "dtau_max_nm": float(cfg_run.get("dtau_max", 0.0) or 0.0),
                "ipc_max_rad": float(cfg_run.get("ipc_max", 0.0) or 0.0),
                "ipc_hold_s": float(cfg_run.get("ipc_hold", 0.0) or 0.0),
                "region_label_by_wind": bool(args.relabel_wind)}
    if cfg_run.get("base") == "mpc":
        from envs.factory import mpc_base_kw
        mm = dict(cp_scale=float(cfg_run.get("base_mm_cp", 1.0)), ftower_scale=float(cfg_run.get("base_mm_ft", 1.0)),
                  mass_scale=float(cfg_run.get("base_mm_m", 1.0)),
                  adapt=str(cfg_run.get("base_adapt", "none")), tau_adapt=float(cfg_run.get("base_tau_adapt", 5.0)))
        for k, v in (("cp_scale", args.base_mm_cp), ("ftower_scale", args.base_mm_ft), ("mass_scale", args.base_mm_m)):
            if v is not None:
                mm[k] = float(v)
        mpc_kw = mpc_base_kw(**mm)
        if args.base_mpc_json:
            mpc_kw.update(json.loads(args.base_mpc_json))
        cfg_over |= {"base_ctrl": "mpc", "mpc_kw": mpc_kw, "obs_base": True}
        print(f"base controller: LPV-MPC in the worker (model scales Cp/Ct {mm['cp_scale']:g}, "
              f"tower f {mm['ftower_scale']:g}, modal mass {mm['mass_scale']:g})")

    if args.gspi:
        ps = {R2: None, R3: None}
    else:
        st = torch.load(run / args.ckpt, weights_only=False)
        if "state" in st:                      # ckpt_best.pt: {"episode","F","knobs","state"}
            knobs = st["knobs"]
            print(f"using best checkpoint from episode {st['episode']} "
                  f"(supervisor-wind {st.get('objective', 'F')}={st['F']:.2f}) with its knobs")
            st = st["state"]
        if "shared" in st:                     # spec_sc: {"shared": {"actors": {r: ...}, "obs_rms": ...}}
            sh = st["shared"]
            ps = {r: {"actor": sh["actors"][r], "obs_rms": sh["obs_rms"]} for r in (R2, R3)}
        else:
            ps = {r: (None if st[r] is None else {"actor": st[r]["actor"], "obs_rms": st[r]["obs_rms"]}) for r in (R2, R3)}
            if "IPC" in st:      # rotation-held IPC: separate slow actor rides along
                ps["IPC"] = {"actor": st["IPC"]["actor"], "obs_rms": st["IPC"]["obs_rms"]}

    episodes = episode_list(args.means, args.seeds, ti=args.ti, episode_s=args.episode_s)
    tb = yaml.safe_load(open(PROJ / "configs" / "turbine" / "nrel5mw.yaml"))
    base = baseline_metrics(baseline_dir(args.backend), episodes, float(tb["dt_ctrl_s"]),
                            float(tb["rated_gen_speed_rads"]),
                            relabel_wind=float(tb["rated_wind_ms"]) if args.relabel_wind else None)
    import re as _re
    pool = WorkerPool(min(args.workers, len(episodes)), args.backend, episodes, cfg_over, hidden=hidden,
                      port0=args.port0, tag="ev_" + _re.sub(r"[^A-Za-z0-9_.-]", "_", run.name)[:40])
    try:
        res = pool.run([{"policy_set": ps, "episode_index": i, "deterministic": True, "seed": 777, "knobs": knobs,
                         "keep_log": bool(args.dump_log)} for i in range(len(episodes))])
    finally:
        pool.close()
    tgt = args.fitness_target or cfg_run.get("reward", {}).get("fitness_target", "blade")
    fit = fitness(res, base, target=tgt)
    # default tag unchanged at TI 8 (the historical bank) so old campaign scripts keep their filenames
    ti_tag = "" if args.ti == 8.0 else f"_ti{ti_label(args.ti)}"
    tag = args.tag or (f"{args.backend}{ti_tag}_s{'-'.join(map(str, args.seeds))}"
                       + ("_gspi" if args.gspi else ""))
    out_dir = Path(os.path.expanduser(args.out)) if args.out else run
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"eval_{tag}.csv", "w", newline="") as f:
        keys = ["mean_wind", "wind_file", "terminated"]
        for r in res:
            keys += [k for k in r["metrics"] if k not in keys]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in res:
            w.writerow({"mean_wind": r["mean_wind"], "wind_file": Path(r["wind_file"]).stem,
                        "terminated": int(r["terminated"]), **r["metrics"]})
    json.dump(fit, open(out_dir / f"eval_{tag}.json", "w"), indent=1, default=float)
    if args.dump_log:
        ld = out_dir / f"logs_{tag}"
        ld.mkdir(exist_ok=True)
        for r in res:
            extra = {f"outb_{k}": v for k, v in (r.get("outb") or {}).items()}
            np.savez(ld / f"{Path(r['wind_file']).stem}.npz", **r["log"], **extra)
        print(f"per-step logs -> {ld}")
    print(f"{run.name} [{tag}] target={fit.get('target')} F_strict={fit['F']:.2f} F_tol2={fit.get('F_tol2', float('nan')):.2f} "
          f"tier={fit.get('tier')}  DELred={fit['del_red_pct']:.2f}%  Eloss={fit['energy_loss_pct']:.2f}%  "
          f"spd_ratio={fit['speed_std_ratio']:.3f}")
    print(f"   J={fit['J']:.2f} (mean {fit['J_metric_mean']:.2f}, energy pen {fit['J_energy_penalty']:.2f}) "
          f"powerMSE {fit['J_power_mse_red_pct']:+.1f}%  speedMSE {fit['J_gen_speed_mse_red_pct']:+.1f}%  "
          f"towerDEL {fit['J_TwrBsMyt_DEL_red_pct']:+.1f}%  bladeDEL {fit['J_RootMyc1_DEL_red_pct']:+.1f}%"
          + ("" if fit["energy_ok"] else "  [ENERGY > 1 %]"))
    print(f"   C={fit['C']:.2f} (regulation {fit['C_goal']:.2f}, violation {fit['C_violation_pct']:.2f} %)   "
          f"CT={fit['CT']:.2f} (tower {fit['CT_goal']:.2f}, violation {fit['CT_violation_pct']:.2f} %)")
    for pe in fit["per_episode"]:
        print(f"   U{pe['mean_wind']:g}: DELred {pe['del_red_pct']:6.2f}%  E {pe['energy_MWh']:.4f}/{pe['energy_base_MWh']:.4f} MWh  "
              f"pitch travel {pe['pitch_travel_deg']:.0f}/{pe['pitch_travel_base_deg']:.0f} deg  |dbeta| {pe['dbeta_abs_mean_deg']:.2f} deg")


if __name__ == "__main__":
    main()
