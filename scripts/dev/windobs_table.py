"""Is the true hub wind in the policy's observation load-bearing? (2026-09-24, campaign_windobs.sh)

The MPC and the gate read the controller's own wind-speed estimate, but the PPO observation carried the simulator's
true hub wind. This compares each frozen policy evaluated twice on the same fresh-seed episodes, once as trained
(true hub wind) and once with the estimate substituted into that observation slot, no retraining. The difference is
paired episode by episode, so it is bootstrapped over episodes stratified by mean wind speed.

    ~/wtrl/run.sh python scripts/dev/windobs_table.py [--csv docs/tables/windobs.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser("~/wtrl/exp"))
ap.add_argument("--n_boot", type=int, default=10000)
ap.add_argument("--csv", default=None)
a = ap.parse_args()
T = ("power_mse_red_pct", "gen_speed_mse_red_pct", "TwrBsMyt_DEL_red_pct", "RootMyc1_DEL_red_pct")


def load(p):
    p = os.path.expanduser(p)
    return json.load(open(p))["per_episode"] if os.path.exists(p) else None


def terms(recs):
    out = []
    for m in T:
        x = np.array([np.nan if r.get(m) is None else r[m] for r in recs], float)
        out.append(float(np.clip(x[np.isfinite(x)], -100, 100).mean()))
    return out


def J(recs):
    t = terms(recs)
    e, eb = sum(r["energy_MWh"] for r in recs), sum(r["energy_base_MWh"] for r in recs)
    return float(np.mean(t) - 20 * max(0.0, 100 * (1 - e / eb) - 1.0))


def travel(recs):
    return float(np.mean([r["pitch_travel_deg"] for r in recs]) / np.mean([r["pitch_travel_base_deg"] for r in recs]))


rng = np.random.default_rng(0)
rows = []
print(f"{'policy':>14} {'base':>14} | {'J (true wind)':>13} {'J (estimate)':>13} {'difference':>22} | {'travel true':>11} {'est':>6}")
for d in sorted(glob.glob(f"{a.exp}/trwCwG3t_s?")) + sorted(glob.glob(f"{a.exp}/mrwCwG3t_s?")):
    A = load(f"{d}/eval_range_TIB_s78910.json")
    B = load(f"{d}/eval_range_TIB_s78910_vest.json")
    if not A or not B:
        continue
    assert len(A) == len(B) and all(x["mean_wind"] == y["mean_wind"] for x, y in zip(A, B)), d
    strata: dict[float, list] = {}
    for i, r in enumerate(A):
        strata.setdefault(r["mean_wind"], []).append(i)
    dd = np.empty(a.n_boot)
    for i in range(a.n_boot):
        pick = [k for ks in strata.values() for k in rng.choice(ks, size=len(ks), replace=True)]
        dd[i] = J([B[k] for k in pick]) - J([A[k] for k in pick])
    lo, hi = np.percentile(dd, [2.5, 97.5])
    name = os.path.basename(d)
    base = "tuned ROSCO" if name.startswith("t") else "scheduled MPC"
    print(f"{name:>14} {base:>14} | {J(A):13.2f} {J(B):13.2f} {J(B) - J(A):+8.2f} [{lo:+.2f}, {hi:+.2f}] |"
          f" {travel(A):11.2f} {travel(B):6.2f}")
    rows.append({"run": name, "base": base, "J_true_wind": round(J(A), 2), "J_wind_estimate": round(J(B), 2),
                 "diff": round(J(B) - J(A), 3), "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
                 **{f"{t}_est": round(x, 2) for t, x in zip(T, terms(B))},
                 "travel_true": round(travel(A), 3), "travel_est": round(travel(B), 3)})

if rows:
    d = np.array([r["diff"] for r in rows])
    for base in ("tuned ROSCO", "scheduled MPC"):
        x = np.array([r["diff"] for r in rows if r["base"] == base])
        if len(x):
            print(f"\n{base}: mean change {x.mean():+.2f} points over {len(x)} policies "
                  f"(range {x.min():+.2f} to {x.max():+.2f})")
    print(f"all policies: mean change {d.mean():+.2f} points, worst {d.min():+.2f}")

if a.csv and rows:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("->", a.csv)
