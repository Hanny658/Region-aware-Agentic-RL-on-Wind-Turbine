"""Best seed of every arm by its held-out objective (the key named in summary.json: J, C or CT) on
eval_heldout_s3456_ckpt_best.json; prints the run names on one line (campaign_agent_rt.sh, stage 3)."""
from __future__ import annotations

import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser("~/wtrl/exp"))
ap.add_argument("--arms", nargs="+", required=True)
ap.add_argument("--eval", default="eval_heldout_s3456_ckpt_best.json")
a = ap.parse_args()
out = []
for arm in a.arms:
    best = None
    for d in sorted(glob.glob(f"{a.exp}/{arm}_s*")):
        s, e = f"{d}/summary.json", f"{d}/{a.eval}"
        if not (os.path.exists(s) and os.path.exists(e)):
            continue
        key = json.load(open(s)).get("objective", "J")
        v = json.load(open(e)).get(key)
        if v is not None and (best is None or v > best[0]):
            best = (v, os.path.basename(d))
    if best:
        out.append(best[1])
print(" ".join(out))
