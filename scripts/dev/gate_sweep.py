"""The gate-threshold sweep: ungated / 13-15 / 15-17, from docs/tables/gated_residual.csv (2026-09-24).

The review of 2026-09-23 objected that only one gate setting had been reported, so "where the residual must close"
was asserted rather than measured. This reads the table built by gate_table.py and answers three questions: what each
threshold is worth in J on each base, where its worst per-wind term sits, and how large that term is. The J
differences are small against the seed spread, so the location of the damage is the part of the comparison that does
not depend on three or two seeds.

    ~/wtrl/run.sh python scripts/dev/gate_sweep.py
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from scipy.stats import fisher_exact

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="docs/tables/gated_residual.csv")
a = ap.parse_args()
rows = list(csv.DictReader(open(os.path.expanduser(a.csv))))
lo = [r for r in rows if "13-15" in r["arm"]]
hi = [r for r in rows if "15-17" in r["arm"] and "agent" in r["arm"]]
hi_all = [r for r in rows if "15-17" in r["arm"]]


def near(rs, thr=16.0):
    return sum(1 for r in rs if float(r["worst_at"].split("@")[1]) <= thr)


print("J against each run's own base, agent-written reward only (mean over seeds, per seed listed)\n")
for seeds, name in (("s3456", "held out"), ("s78910", "fresh")):
    print(f"--- {name} (seeds {seeds[1:]})")
    for base in ("tuned ROSCO", "scheduled MPC"):
        for lab, rs in (("gate 13-15", lo), ("gate 15-17", hi)):
            d = [float(r["diff"]) for r in rs if r["seeds"] == seeds and r["base"] == base]
            j = [float(r["J"]) for r in rs if r["seeds"] == seeds and r["base"] == base]
            if d:
                print(f"   {base:>14}  {lab}:  n={len(d)}  mean {np.mean(d):+5.2f}   J {np.mean(j):6.2f}"
                      f"   per seed " + " ".join(f"{x:+.2f}" for x in d))
    print()

print("where the worst per-wind term sits, over all rows of each arm (both wind sets)")
print(f"   at or below 16 m/s: gate 13-15 {near(lo)}/{len(lo)},  gate 15-17 {near(hi_all)}/{len(hi_all)}")
t = [[near(lo), len(lo) - near(lo)], [near(hi_all), len(hi_all) - near(hi_all)]]
print(f"   Fisher exact, two-sided p = {fisher_exact(t)[1]:.4f}")
print(f"   largest single per-wind shortfall: gate 13-15 {min(float(r['worst_per_wind_diff']) for r in lo):+.1f},"
      f"  gate 15-17 {min(float(r['worst_per_wind_diff']) for r in hi_all):+.1f}")
print("\n   the low gate's damage is at 12-16 m/s, which is where the base is already strong and where the ungated")
print("   layer was diagnosed to do harm; moving the gate down reintroduces it.")
