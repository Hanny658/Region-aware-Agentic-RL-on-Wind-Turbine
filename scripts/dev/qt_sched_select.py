"""Select among wind-scheduled tower-weight candidates on the selection winds (campaign_next1.sh, stage A).

Reads eval_<tag><suffix>.json files of one directory, computes J and the per-wind terms, and applies the per-wind
load rule of roadmap s28: a candidate is feasible when no term is below -1 % at any mean wind speed (power / speed
MSE only where defined). Feasible candidates are ranked by J, infeasible ones by their worst per-wind term.
Prints the chosen tags (space separated) to stdout; the table goes to stderr.

    ~/wtrl/run.sh python scripts/dev/qt_sched_select.py --dir ~/wtrl/exp/mpcsearch600/qtsched --suffix _s12 --n 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--suffix", default="_s12")
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--exclude", nargs="*", default=["nominal", "ref"], help="tags listed for comparison only")
ap.add_argument("--tol", type=float, default=-1.0)
a = ap.parse_args()
T = ("power_mse_red_pct", "gen_speed_mse_red_pct", "TwrBsMyt_DEL_red_pct", "RootMyc1_DEL_red_pct")
rows = []
for p in sorted(glob.glob(f"{a.dir}/eval_*{a.suffix}.json")):
    tag = os.path.basename(p)[len("eval_"):-len(a.suffix) - len(".json")]
    d = json.load(open(p))
    per = d["per_episode"]
    by = {}
    for q in per:
        by.setdefault(q["mean_wind"], []).append(q)
    worst, pw = 0.0, {}
    for u in sorted(by):
        for t in T:
            v = [q[t] for q in by[u] if q.get(t) is not None and q[t] == q[t]]
            if v:
                m = float(np.mean(np.clip(v, -100, 100)))
                pw[(u, t)] = m
                worst = min(worst, m)
    trav = np.mean([q["pitch_travel_deg"] for q in per]) / np.mean([q["pitch_travel_base_deg"] for q in per])
    feasible = worst >= a.tol
    rows.append({"tag": tag, "J": d["J"], "worst": worst, "feasible": feasible, "travel": trav, "pw": pw,
                 "terms": [d[f"J_{t}"] for t in T]})
rows.sort(key=lambda r: (not r["feasible"], -(r["J"] if r["feasible"] else r["worst"])))
winds = sorted({u for r in rows for (u, _) in r["pw"]})
print(f"{'tag':>22} {'J':>6} {'feas':>5} {'worst':>6} {'xGSPI':>6} | power / speed / tower / blade | tower term per wind " + " ".join(f"{u:g}" for u in winds), file=sys.stderr)
for r in rows:
    tw = " ".join(f"{r['pw'].get((u, T[2]), float('nan')):5.1f}" for u in winds)
    print(f"{r['tag']:>22} {r['J']:6.2f} {str(r['feasible']):>5} {r['worst']:6.1f} {r['travel']:6.2f} | " + " / ".join(f"{x:5.1f}" for x in r["terms"]) + " | " + tw, file=sys.stderr)
chosen = [r["tag"] for r in rows if r["feasible"] and r["tag"] not in a.exclude][:a.n]
if not chosen:
    chosen = [r["tag"] for r in rows if r["tag"] not in a.exclude][:a.n]
    print("no feasible candidate: taking the least infeasible", file=sys.stderr)
print(" ".join(chosen))
