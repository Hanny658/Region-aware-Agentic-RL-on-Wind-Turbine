"""Paired GSPI baselines: zero-residual rollout per wind file -> {baseline_dir}/{wind}.npz
(t, P, plus the full env log). Needed by the R2 reward term P/P_base and by every evaluation
(all improvements are reported relative to the same-seed GSPI run).

    python scripts/make_baselines.py --backend toy --means 8 12.5 15 --seeds 1
    python scripts/make_baselines.py --backend openfast --means 8 12.5 15 --seeds 1 --jobs 3
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from envs.factory import baseline_dir, episode_list, make_env


def run_one(args):
    backend, ep, out, port = args
    env = make_env(backend, [ep], port=port, work_tag="work_base")
    env.reset(options={"episode_index": 0})
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(np.zeros(1, dtype=np.float32))
        done = term or trunc
    L = env.log_arrays()
    extra = {}
    if getattr(env, "outb", None) is not None:
        extra = {f"outb_{k}": v for k, v in env.outb.items()}
    env.close()
    np.savez(out, **L, **extra)
    act = L["warmup"] == 0
    return (f"{Path(ep.wind_file).stem:>16}  P={L['P'][act].mean() / 1e6:5.2f} MW  "
            f"pitch={np.rad2deg(L['beta_meas'][act].mean()):5.2f} deg  M={L['M_oop'][act].mean() / 1e6:5.2f} MNm  "
            f"R3={L['region'][act].mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["toy", "openfast"])
    ap.add_argument("--means", nargs="+", type=float, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--ti", type=float, default=8.0)
    ap.add_argument("--episode_s", type=float, default=150.0)
    ap.add_argument("--warmup_s", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--port0", type=int, default=5700)
    args = ap.parse_args()

    eps = episode_list(args.means, args.seeds, ti=args.ti, episode_s=args.episode_s, warmup_s=args.warmup_s)
    out_dir = Path(baseline_dir(args.backend))
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.backend, ep, str(out_dir / f"{Path(ep.wind_file).stem}.npz"), args.port0 + i)
            for i, ep in enumerate(eps)]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            for line in ex.map(run_one, jobs):
                print(line)
    else:
        for j in jobs:
            print(run_one(j))
    print("baselines written to", out_dir)
