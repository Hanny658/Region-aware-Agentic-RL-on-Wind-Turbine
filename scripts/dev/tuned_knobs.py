"""Derive the J-tuned default knob vector (fairness step 2) from the weight-search runs of step 1:
per knob, the geometric median (= median of the logs) of the best-checkpoint knobs across seeds,
so no single seed's excursion sets the default. Writes a JSON for train.py --knobs_json.

    python scripts/dev/tuned_knobs.py "~/wtrl/exp/jwf3_s*" --out configs/knobs_j_v3_tuned.json
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("pattern")
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = {}
for d in sorted(glob.glob(os.path.expanduser(a.pattern))):
    if not os.path.exists(f"{d}/ckpt_best.pt"):
        continue
    ck = torch.load(f"{d}/ckpt_best.pt", weights_only=False)
    kn = {k: float(v) for k, v in ck["knobs"].items() if isinstance(v, (int, float))}
    rows[os.path.basename(d)] = (ck.get("F"), ck.get("episode"), kn)
    print(f"{os.path.basename(d):>10} best {ck.get('objective', 'F')}={ck['F']:.2f} @ep {ck['episode']}: "
          + ", ".join(f"{k}={v:g}" for k, v in kn.items()))
keys = sorted(set(k for _, _, kn in rows.values() for k in kn))
out = {}
for k in keys:
    v = np.array([kn[k] for _, _, kn in rows.values() if k in kn], float)
    out[k] = float(np.exp(np.median(np.log(v)))) if np.all(v > 0) else float(np.median(v))
print("tuned default (per-knob geometric median over %d runs):" % len(rows))
print("   " + ", ".join(f"{k}={v:.4g}" for k, v in out.items()))
os.makedirs(os.path.dirname(os.path.expanduser(a.out)) or ".", exist_ok=True)
json.dump({**out, "_source": sorted(rows), "_rule": "geometric median of best-checkpoint knobs"},
          open(os.path.expanduser(a.out), "w", encoding="utf-8"), indent=1)
print("->", a.out)
