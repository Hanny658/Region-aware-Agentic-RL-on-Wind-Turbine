"""Power-tracking MAE in R3 (the user's original J3 metric: MAE(P, P_rated)/P_rated) for the
held-out evaluations, absolute and relative to the paired GSPI baselines."""
import glob
import os
import sys

import pandas as pd
import yaml

from envs.base_env import PROJ
from envs.factory import baseline_dir, episode_list
from eval.fitness import baseline_metrics

tb = yaml.safe_load(open(PROJ / "configs" / "turbine" / "nrel5mw.yaml"))
eps = episode_list([8, 12.5, 15], [3, 4, 5, 6], episode_s=150.0)
base = baseline_metrics(baseline_dir("openfast"), eps, float(tb["dt_ctrl_s"]), float(tb["rated_gen_speed_rads"]))
base_by_stem = {os.path.basename(k).replace(".bts", ""): v for k, v in base.items()}

print(f"{'run':>18} {'P_MAE_R3 %':>11} {'GSPI %':>7} {'ratio':>6} | {'spd_MAE ratio':>13}")
rows = []
for pat in sys.argv[1:] or ["~/wtrl/exp/n1_*"]:
    for d in sorted(glob.glob(os.path.expanduser(pat))):
        f = os.path.join(d, "eval_heldout_s3456_ckpt_best.csv")
        if not os.path.exists(f):
            continue
        p = pd.read_csv(f)
        p = p[p["frac_R3"] >= 0.5]
        pol, bas, spol, sbas = [], [], [], []
        for _, r in p.iterrows():
            b = base_by_stem[r["wind_file"]]
            if r["power_mae_R3"] == r["power_mae_R3"] and b.get("power_mae_R3", float("nan")) == b.get("power_mae_R3"):
                pol.append(r["power_mae_R3"]); bas.append(b["power_mae_R3"])
                spol.append(r["gen_speed_mae_R3"]); sbas.append(b["gen_speed_mae_R3"])
        if not pol:
            continue
        pm = 100 * sum(pol) / len(pol); bm = 100 * sum(bas) / len(bas)
        ratio = (sum(a / b for a, b in zip(pol, bas)) / len(pol))
        sratio = (sum(a / b for a, b in zip(spol, sbas)) / len(spol))
        print(f"{os.path.basename(d):>18} {pm:11.3f} {bm:7.3f} {ratio:6.3f} | {sratio:13.3f}")
