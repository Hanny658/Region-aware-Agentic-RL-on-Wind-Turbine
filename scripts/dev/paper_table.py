"""Paper-style comparison table (Wang et al. Fig. 4 metric set) from eval_paper_s3456.json files:
reductions vs paired GSPI in % — Power MSE (R3, vs rated), GenSpeed MSE (R3, vs rated),
TwrBsMyt DEL, RootMyc1 DEL — plus energy loss and the tier, aggregated per method (mean ± std)."""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

runs = defaultdict(list)
for pat in sys.argv[1:] or ["~/wtrl/exp/n1_*"]:
    for f in sorted(glob.glob(os.path.expanduser(pat + "/eval_paper_s3456.json"))):
        name = os.path.basename(os.path.dirname(f))
        method = re.sub(r"_s\d+$", "", name)
        runs[method].append((name, json.load(open(f))))

cols = [("power_mse_red_pct", "Power MSE red%"), ("gen_speed_mse_red_pct", "GenSpd MSE red%"),
        ("TwrBsMyt_DEL_red_pct", "TwrBsMyt DEL red%"), ("RootMyc1_DEL_red_pct", "RootMyc1 DEL red%"),
        ("energy_loss_pct", "Energy loss%")]
print(f"{'method':>14} " + " ".join(f"{h:>18}" for _, h in cols) + f" {'tiers':>22}")
for m, items in sorted(runs.items()):
    vals = {k: [j.get(k, float('nan')) for _, j in items] for k, _ in cols}
    tiers = ",".join(j.get("tier", "?")[:4] for _, j in items)
    cells = []
    for k, _ in cols:
        v = np.array(vals[k], float)
        cells.append(f"{np.nanmean(v):8.2f} ±{np.nanstd(v):5.2f}")
    print(f"{m:>14} " + " ".join(f"{c:>18}" for c in cells) + f" {tiers:>22}")
print("\nper-run detail:")
for m, items in sorted(runs.items()):
    for name, j in items:
        print(f"  {name:>16} tier={j.get('tier','?'):>8} " +
              " ".join(f"{k.split('_')[0][:5]}={j.get(k, float('nan')):+6.2f}" for k, _ in cols))
