"""The gate comparison with the trained POLICY as the statistical unit (2026-09-24).

The first version of this test counted rows of docs/tables/gated_residual.csv, which are trained-policy x
evaluation-set pairs. The two rows of one policy are not independent observations: the same weights meet two wind
sets, and where a policy degrades is a property of the policy. Counting them separately is pseudo-replication. This
recomputes the comparison with one observation per trained policy, under two definitions of the outcome, and also
reports the descriptive form, which is what we would rather rely on at these sample sizes.

    ~/wtrl/run.sh python scripts/dev/gate_policy_test.py
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

from scipy.stats import fisher_exact

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="docs/tables/gated_residual.csv")
ap.add_argument("--thr", type=float, default=16.0, help="wind speed at or below which a shortfall counts as near rated")
ap.add_argument("--agent_only", action="store_true", help="exclude the fixed-reward gated runs from the high-gate arm")
a = ap.parse_args()
rows = list(csv.DictReader(open(os.path.expanduser(a.csv))))

pol: dict[str, dict] = defaultdict(dict)
for r in rows:
    if "13-15" in r["arm"]:
        g = "13-15"
    elif "15-17" in r["arm"]:
        g = "15-17"
    else:
        continue
    if a.agent_only and "agent" not in r["arm"]:
        continue
    d = pol[r["run"]]
    d["gate"], d["base"] = g, r["base"]
    d.setdefault("at", {})[r["seeds"]] = (float(r["worst_at"].split("@")[1]), float(r["worst_per_wind_diff"]))

print(f"{'policy':>14} {'gate':>7} {'base':>14} | worst-term wind speed and shortfall, per wind set")
for run in sorted(pol, key=lambda k: (pol[k]["gate"], k)):
    d = pol[run]
    cells = "  ".join(f"{k[1:]:>6}: {v[0]:4.0f} m/s ({v[1]:+5.1f})" for k, v in sorted(d["at"].items()))
    print(f"{run:>14} {d['gate']:>7} {d['base']:>14} | {cells}")

for name, rule in (("either wind set", any), ("both wind sets", all)):
    tab = {"13-15": [0, 0], "15-17": [0, 0]}
    for d in pol.values():
        near = rule(v[0] <= a.thr for v in d["at"].values())
        tab[d["gate"]][0 if near else 1] += 1
    lo, hi = tab["13-15"], tab["15-17"]
    p = fisher_exact([lo, hi])[1]
    print(f"\nworst term at or below {a.thr:g} m/s on {name}, one observation per trained policy:")
    print(f"   gate 13-15: {lo[0]}/{sum(lo)}   gate 15-17: {hi[0]}/{sum(hi)}   Fisher exact two-sided p = {p:.4f}")

worst = {"13-15": [], "15-17": []}
for d in pol.values():
    worst[d["gate"]].append(min(v[1] for v in d["at"].values()))
print("\nlargest per-wind shortfall of each policy, in points:")
for g, v in worst.items():
    print(f"   gate {g}: " + ", ".join(f"{x:+.1f}" for x in sorted(v)))
