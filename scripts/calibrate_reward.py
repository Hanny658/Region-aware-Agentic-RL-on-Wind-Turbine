"""Reward-scale calibration: how much does each reward term *vary* under random residuals in
each region?  PPO only cares about advantage scale, so we match the per-region std of the
task term (R2: w_P*(P/P_base-1); R3: w_w*exp(-|d|/tau)) rather than their means.

Runs random-action (uniform in [-1,1], held for `hold_s`) episodes on the given backend with
the paired baselines loaded and prints the std / mean of every term per region, plus the w_P
that would equalise the task-term std across regions.

    python scripts/calibrate_reward.py --backend toy --means 8 12.5 15 --episodes 3
"""
from __future__ import annotations

import argparse

import numpy as np

from controllers.router import R2, R3
from envs.base_env import default_config
from envs.factory import baseline_dir, episode_list, make_env

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="toy", choices=["toy", "openfast"])
ap.add_argument("--means", nargs="+", type=float, default=[8, 12.5, 15])
ap.add_argument("--seeds", nargs="+", type=int, default=[1])
ap.add_argument("--episodes", type=int, default=3, help="passes over the episode list")
ap.add_argument("--hold_s", type=float, default=2.0, help="random action held for this long")
ap.add_argument("--episode_s", type=float, default=150.0)
args = ap.parse_args()

cfg = default_config(baseline_dir=baseline_dir(args.backend))
eps = episode_list(args.means, args.seeds, episode_s=args.episode_s)
env = make_env(args.backend, eps, cfg=cfg, port=5790, work_tag="work_calib")
rng = np.random.default_rng(0)
hold = int(args.hold_s / env.dt)

rows = []
for k in range(args.episodes * len(eps)):
    env.reset(options={"episode_index": k % len(eps)})
    done, n, a = False, 0, 0.0
    while not done:
        if n % hold == 0:
            a = rng.uniform(-1, 1)
        _, r, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        done = term or trunc
        n += 1
    L = env.log_arrays()
    act = L["warmup"] == 0
    for key in ("r_task", "r_load", "r_act", "reward"):
        for reg, name in ((R2, "R2"), (R3, "R3")):
            sel = act & (L["region"] == reg)
            if sel.sum() > 100:
                rows.append((name, key, L[key][sel].mean(), L[key][sel].std(), sel.sum()))
    if args.backend == "openfast":
        print(f"episode {k} done")
env.close()

import collections
agg = collections.defaultdict(list)
for reg, key, mu, sd, n in rows:
    agg[(reg, key)].append((mu, sd, n))
print(f"\n{'region':6} {'term':8} {'mean':>9} {'std':>9} {'n':>8}")
stats = {}
for (reg, key), v in sorted(agg.items()):
    v = np.array(v)
    mu = np.average(v[:, 0], weights=v[:, 2]); sd = np.average(v[:, 1], weights=v[:, 2])
    stats[(reg, key)] = (mu, sd)
    print(f"{reg:6} {key:8} {mu:9.3f} {sd:9.3f} {int(v[:, 2].sum()):8d}")
if ("R2", "r_task") in stats and ("R3", "r_task") in stats:
    w_P = float(cfg.reward["w_power"])
    sd2, sd3 = stats[("R2", "r_task")][1], stats[("R3", "r_task")][1]
    print(f"\ncurrent w_power={w_P:g}: std(R2 task)={sd2:.3f} vs std(R3 task)={sd3:.3f}")
    if sd2 > 0:
        print(f"w_power that equalises task-term std: {w_P * sd3 / sd2:.1f}")
