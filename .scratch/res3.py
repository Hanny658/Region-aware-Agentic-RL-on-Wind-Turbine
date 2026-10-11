"""Resolution of each constraint on the 14-episode design set, from paired stratified bootstraps."""
import json, glob, os
import numpy as np
fs = sorted(glob.glob(os.path.expanduser("~/wtrl/exp/roscosearch/*/cache/eval_*.json")))
d = json.load(open(fs[0]))
pe = d.get("per_episode", [])
print(f"{len(fs)} cached evaluations; per_episode keys:")
print("  ", sorted(pe[0]) if pe else "none")

COLS = {"tower": "TwrBsMyt_DEL_red_pct", "blade": "RootMyc1_DEL_red_pct",
        "speed": "gen_speed_mse_red_pct", "power": "power_mse_red_pct"}
rng = np.random.default_rng(0)
half = {k: [] for k in COLS}
nshown = 0
for f in fs:
    d = json.load(open(f))
    pe = d.get("per_episode", [])
    if len(pe) < 14 or d.get("failed"):
        continue
    u = np.array([p["mean_wind"] for p in pe], float)
    for k, col in COLS.items():
        v = np.array([p.get(col, np.nan) for p in pe], float)
        if not np.isfinite(v).all():
            continue
        draws = []
        for _ in range(400):                       # stratified by mean wind speed, as the paper's bootstraps are
            idx = np.concatenate([rng.choice(np.where(u == s)[0], (u == s).sum(), replace=True) for s in np.unique(u)])
            draws.append(v[idx].mean())
        lo, hi = np.percentile(draws, [2.5, 97.5])
        half[k].append((hi - lo) / 2)
    nshown += 1
print(f"\nusable evaluations: {nshown}")
print("95 % paired-bootstrap half-width over the 14 design episodes, in points:")
for k in COLS:
    h = np.array(half[k])
    print(f"  {k:<6} median {np.median(h):6.2f}   quartiles {np.percentile(h,25):5.2f}-{np.percentile(h,75):5.2f}   max {h.max():6.2f}")
