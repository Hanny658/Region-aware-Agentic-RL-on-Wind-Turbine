"""Paper-style comparison table (Wang et al. Fig. 4 metric set) from held-out evaluation JSONs:
reductions vs paired GSPI in % — Power MSE (R3, vs rated), GenSpeed MSE (R3, vs rated),
TwrBsMyt DEL, RootMyc1 DEL — plus energy loss and the tier, aggregated per method (mean ± std).

    python scripts/dev/paper_table.py "~/wtrl/exp/tq_off_*"
    python scripts/dev/paper_table.py --eval eval_robust_ti14.json "~/wtrl/exp/tq_off_*"

Per run the first existing file of `--eval` (default: eval_paper_s3456.json, then
eval_heldout_s3456_ckpt_best.json) is used — the two carry the same fields; only the
dedicated paper-metric re-evaluation of the retired machine used the first name."""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

args = sys.argv[1:]
names = ["eval_paper_s3456.json", "eval_heldout_s3456_ckpt_best.json"]
if "--eval" in args:
    i = args.index("--eval")
    names = [args[i + 1]]
    args = args[:i] + args[i + 2:]

runs = defaultdict(list)
for pat in args or ["~/wtrl/exp/n1_*"]:
    for d in sorted(glob.glob(os.path.expanduser(pat))):
        f = next((os.path.join(d, n) for n in names if os.path.exists(os.path.join(d, n))), None)
        if f is None:
            continue
        name = os.path.basename(d.rstrip("/"))
        method = re.sub(r"_s\d+$", "", name)
        runs[method].append((name, json.load(open(f, encoding="utf-8"))))

cols = [("J", "J (metric-set obj)"), ("power_mse_red_pct", "Power MSE red%"), ("gen_speed_mse_red_pct", "GenSpd MSE red%"),
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
