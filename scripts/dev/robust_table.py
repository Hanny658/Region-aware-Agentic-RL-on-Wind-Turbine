"""Aggregate the stress-class evaluations (eval_robust_*.json + the TI8 held-out reference) per arm.

    python scripts/dev/robust_table.py "~/wtrl/exp/tq_off_*" "~/wtrl/exp/sched2_ep_*" ...

Classes: heldout_s3456_ckpt_best (TI8 reference), robust_ti14, robust_ti22, robust_u18.
Reports mean +- std of the target DEL reduction over the RL seeds of each arm, plus the tier count
and the mean speed-std ratio, i.e. the table the manuscript's robustness section needs.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

CLASSES = [("heldout_s3456_ckpt_best", "TI8 (held-out ref)"), ("robust_ti14", "TI14 @ 8/12.5/15"),
           ("robust_ti22", "TI22 @ 15"), ("robust_u18", "U18 @ TI8 (OOD)")]


def main():
    pats = sys.argv[1:] or ["~/wtrl/exp/tq_off_*", "~/wtrl/exp/sched2_ep_*", "~/wtrl/exp/n1_*"]
    data = defaultdict(lambda: defaultdict(list))     # arm -> class -> [(del_red, tier, spd)]
    for pat in pats:
        for d in sorted(glob.glob(os.path.expanduser(pat))):
            arm = re.sub(r"_s\d+$", "", os.path.basename(d.rstrip("/")))
            for tag, _ in CLASSES:
                f = os.path.join(d, f"eval_{tag}.json")
                if os.path.exists(f):
                    j = json.load(open(f, encoding="utf-8"))
                    data[arm][tag].append((j["del_red_pct"], j["tier"], j["speed_std_ratio"]))
    print(f"{'arm':>16} {'class':>22} {'n':>3} {'DELred%':>16} {'strict':>8} {'spd':>7}")
    for arm in sorted(data):
        for tag, label in CLASSES:
            v = data[arm].get(tag)
            if not v:
                continue
            d = np.array([x[0] for x in v], float)
            s = sum(x[1] == "strict" for x in v)
            spd = float(np.mean([x[2] for x in v]))
            sd = f"± {d.std(ddof=1):4.2f}" if len(d) > 1 else "      "
            print(f"{arm:>16} {label:>22} {len(d):3d} {d.mean():9.2f} {sd:>6} {f'{s}/{len(v)}':>8} {spd:7.3f}")
    print("\n(target DEL = the run's own fitness target: tower for the tower-objective arms.)")


if __name__ == "__main__":
    main()
