import json, glob, os
import numpy as np
fs = sorted(glob.glob(os.path.expanduser("~/wtrl/exp/roscosearch/*/cache/eval_*.json")))
rng = np.random.default_rng(0); half = []
for f in fs:
    d = json.load(open(f)); pe = d.get("per_episode", [])
    if len(pe) < 14 or d.get("failed"): continue
    u = np.array([p["mean_wind"] for p in pe], float)
    en = np.array([p["energy_MWh"] for p in pe], float)
    eb = np.array([p["energy_base_MWh"] for p in pe], float)
    draws = []
    for _ in range(400):
        idx = np.concatenate([rng.choice(np.where(u == s)[0], (u == s).sum(), replace=True) for s in np.unique(u)])
        draws.append(100.0 * (1 - en[idx].sum() / eb[idx].sum()))
    lo, hi = np.percentile(draws, [2.5, 97.5]); half.append((hi - lo) / 2)
h = np.array(half)
print(f"energy loss, 95 % paired-bootstrap half-width over the 14 design episodes, in points:")
print(f"  median {np.median(h):.3f}   quartiles {np.percentile(h,25):.3f}-{np.percentile(h,75):.3f}   max {h.max():.3f}")
