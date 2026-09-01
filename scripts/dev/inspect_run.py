"""Per-wind training statistics over episode windows for one run dir (diagnosing collapses)."""
import os
import sys

import pandas as pd

run = os.path.expanduser(sys.argv[1])
p = pd.read_csv(f"{run}/episodes.csv")
cols = [c for c in ["RootMoop_DEL_MNm", "gen_speed_std", "dbeta_abs_mean_deg", "pitch_travel_deg",
                    "r_task_R2", "r_load_R2", "r_task_R3", "r_load_R3", "reward_mean"] if c in p]
n = int(p.episode.max()) + 1
for lo in range(0, n, max(n // 5, 1)):
    hi = min(lo + max(n // 5, 1), n)
    t = p[(p.episode >= lo) & (p.episode < hi)]
    print(f"\nepisodes {lo}-{hi - 1}:")
    print(t.groupby("mean_wind")[cols].mean().round(3).to_string())
std_cols = [c for c in p.columns if c.endswith("/std")]
if std_cols:
    print("\npolicy std over training:")
    print(p[["episode"] + std_cols].iloc[::max(n // 8, 1)].round(3).to_string(index=False))
e = pd.read_csv(f"{run}/evals.csv")
print("\nevals:")
print(e[["episode", "F", "del_red_pct", "energy_loss_pct", "speed_std_ratio"]].round(2).to_string(index=False))
