"""Does the language model propose better reward structures than a random draw from its own vocabulary? (2026-09-24)

The one statistical question about the agent that survived the referee readings of s45. The level is not in question:
a searched reward structure buys 3.5-4 points over the tuned PI base whoever proposes it. The question is the
objective's own per-wind load rule -- both fatigue loads within 1 % of the base at EVERY mean wind speed on the 600 s
held-out set -- where the agent held 4 of 5 and the random structural control 0 of 3, which Fisher could not separate
(two-sided p = 0.14). `campaign_rand_seeds.sh` took that control to n = 8.

Reads docs/tables/range_residual.csv (built by range_residual_table.py) and reports, per arm, the level and the
load-rule count, then the tests the paper will quote.

    ~/wtrl/run.sh python scripts/dev/proposer_test.py
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from scipy.stats import fisher_exact

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="docs/tables/range_residual.csv")
ap.add_argument("--tol", type=float, default=-1.0, help="per-wind tolerance in points")
a = ap.parse_args()
rows = list(csv.DictReader(open(os.path.expanduser(a.csv))))


def held(r):
    return (float(r["range600_worst_tower_diff"]) >= a.tol) and (float(r["range600_worst_blade_diff"]) >= a.tol)


arms: dict[str, list] = {}
for r in rows:
    arms.setdefault(r["arm"], []).append(r)

print(f"{'arm':>44}   n   loads held   mean diff to the tuned ROSCO   s.d.")
for arm, rs in arms.items():
    d = [float(r["range600_diff_tuned"]) for r in rs]
    print(f"{arm:>44}  {len(rs):2d}     {sum(map(held, rs))}/{len(rs)}          {np.mean(d):+6.2f}"
          f"                {np.std(d, ddof=1) if len(d) > 1 else 0.0:5.2f}")

A = [r for r in rows if r["arm"].startswith("agent-written reward") and "gated" not in r["arm"]]
R = [r for r in rows if "random reward" in r["arm"]]
F = [r for r in rows if "fixed reward" in r["arm"]]
print()
for lab, C in (("random structure", R), ("fixed rewards", F), ("pooled controls", R + F)):
    ha, hc = sum(map(held, A)), sum(map(held, C))
    t = [[ha, len(A) - ha], [hc, len(C) - hc]]
    two = fisher_exact(t)[1]
    one = fisher_exact(t, alternative="greater")[1]
    print(f"per-wind load rule: agent {ha}/{len(A)} vs {lab} {hc}/{len(C)}   "
          f"two-sided p = {two:.4f}   one-sided p = {one:.4f}")
print("\nLevel and load rule are separate questions: the first is bought by searching the reward's structure,")
print("the second is where the proposers are compared.")
