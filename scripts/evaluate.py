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

import torch
import yaml

from agents.rollout import WorkerPool
from controllers.router import R2, R3
from envs.base_env import PROJ
from envs.factory import baseline_dir, episode_list
from eval.fitness import baseline_metrics, fitness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    ap.add_argument("--backend", default="toy", choices=["toy", "openfast"])
    ap.add_argument("--means", nargs="+", type=float, default=[8, 12.5, 15])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--episode_s", type=float, default=150.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--port0", type=int, default=5900)
    ap.add_argument("--gspi", action="store_true", help="evaluate the zero-residual GSPI instead of the checkpoint")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fitness_target", default=None, choices=["blade", "tower"])
    args = ap.parse_args()

    run = Path(os.path.expanduser(args.run))
    cfg_run = json.load(open(run / "config.json"))
    summ = json.load(open(run / "summary.json")) if (run / "summary.json").exists() else {}
    knobs = summ.get("final_knobs", cfg_run.get("knobs0"))
    ppo_yaml = yaml.safe_load(open(PROJ / "configs" / "ppo.yaml"))
    hidden = tuple(ppo_yaml["hidden"])
    flag = cfg_run["method"] == "mono_flag"
    cfg_over = {"baseline_dir": baseline_dir(args.backend), "region_flag_in_obs": flag, "reward": cfg_run["reward"],
                "obs_fa_acc": bool(cfg_run.get("obs_fa_acc", False)),
                "dtau_max_nm": float(cfg_run.get("dtau_max", 0.0) or 0.0),
                "ipc_max_rad": float(cfg_run.get("ipc_max", 0.0) or 0.0)}

    if args.gspi:
        ps = {R2: None, R3: None}
    else:
        st = torch.load(run / args.ckpt, weights_only=False)
        if "state" in st:                      # ckpt_best.pt: {"episode","F","knobs","state"}
            knobs = st["knobs"]
            print(f"using best checkpoint from episode {st['episode']} (toy F={st['F']:.2f}) with its knobs")
            st = st["state"]
        if "shared" in st:                     # spec_sc: {"shared": {"actors": {r: ...}, "obs_rms": ...}}
            sh = st["shared"]
            ps = {r: {"actor": sh["actors"][r], "obs_rms": sh["obs_rms"]} for r in (R2, R3)}
        else:
            ps = {r: (None if st[r] is None else {"actor": st[r]["actor"], "obs_rms": st[r]["obs_rms"]}) for r in (R2, R3)}

    episodes = episode_list(args.means, args.seeds, episode_s=args.episode_s)
    tb = yaml.safe_load(open(PROJ / "configs" / "turbine" / "nrel5mw.yaml"))
    base = baseline_metrics(baseline_dir(args.backend), episodes, float(tb["dt_ctrl_s"]), float(tb["rated_gen_speed_rads"]))
    import re as _re
    pool = WorkerPool(min(args.workers, len(episodes)), args.backend, episodes, cfg_over, hidden=hidden,
                      port0=args.port0, tag="ev_" + _re.sub(r"[^A-Za-z0-9_.-]", "_", run.name)[:40])
    try:
        res = pool.run([{"policy_set": ps, "episode_index": i, "deterministic": True, "seed": 777, "knobs": knobs}
                        for i in range(len(episodes))])
    finally:
        pool.close()
    tgt = args.fitness_target or cfg_run.get("reward", {}).get("fitness_target", "blade")
    fit = fitness(res, base, target=tgt)
    tag = args.tag or f"{args.backend}_s{'-'.join(map(str, args.seeds))}" + ("_gspi" if args.gspi else "")
    with open(run / f"eval_{tag}.csv", "w", newline="") as f:
        keys = ["mean_wind", "wind_file", "terminated"]
        for r in res:
            keys += [k for k in r["metrics"] if k not in keys]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in res:
            w.writerow({"mean_wind": r["mean_wind"], "wind_file": Path(r["wind_file"]).stem,
                        "terminated": int(r["terminated"]), **r["metrics"]})
    json.dump(fit, open(run / f"eval_{tag}.json", "w"), indent=1, default=float)
    print(f"{run.name} [{tag}] target={fit.get('target')} F_strict={fit['F']:.2f} F_tol2={fit.get('F_tol2', float('nan')):.2f} "
          f"tier={fit.get('tier')}  DELred={fit['del_red_pct']:.2f}%  Eloss={fit['energy_loss_pct']:.2f}%  "
          f"spd_ratio={fit['speed_std_ratio']:.3f}")
    for pe in fit["per_episode"]:
        print(f"   U{pe['mean_wind']:g}: DELred {pe['del_red_pct']:6.2f}%  E {pe['energy_MWh']:.4f}/{pe['energy_base_MWh']:.4f} MWh  "
              f"pitch travel {pe['pitch_travel_deg']:.0f}/{pe['pitch_travel_base_deg']:.0f} deg  |dbeta| {pe['dbeta_abs_mean_deg']:.2f} deg")


if __name__ == "__main__":
    main()
