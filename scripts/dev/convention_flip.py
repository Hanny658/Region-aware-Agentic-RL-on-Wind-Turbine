"""Does the ranking of two controllers reverse between reporting conventions, or merely vanish? (2026-09-24)

The same 28 fresh-seed episodes, the same two controllers, three aggregations that are all in use in the wind-energy
literature:

  per-episode   mean over episodes of the per-episode percentage reduction          (what J and the figures use)
  level ratio   1 - (mean level of the controller) / (mean level of the baseline)   (what a DEL comparison requires)
  Rayleigh      the same ratio with IEC Rayleigh bin weights, renormalised over the evaluated bins

For each convention this reports the reduction of each controller against the paired shipped GSPI, the difference
between the two controllers, and a paired stratified bootstrap interval of that difference over episodes. A reversal
is only a reversal if the interval on the difference excludes zero in both conventions with opposite signs.

    ~/wtrl/run.sh python scripts/dev/convention_flip.py [--n_boot 20000]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser("~/wtrl/exp"))
ap.add_argument("--vave", type=float, default=10.0)
ap.add_argument("--bin", type=float, default=2.0)
ap.add_argument("--n_boot", type=int, default=20000)
a = ap.parse_args()
R6 = f"{a.exp}/mpcsearch600"
PAIRS = [("tuned ROSCO", f"{R6}/fresh/eval_rosco_tuned_s78910"),
         ("scheduled MPC", f"{R6}/fresh/eval_sched_mpc_s78910")]
QTY = {"power MSE": ("power_mse_R3", "power_mse_red_pct", 1.0),
       "speed MSE": ("gen_speed_mse_R3", "gen_speed_mse_red_pct", 1.0),
       "tower DEL": ("TwrBsMyt_DEL_MNm", "TwrBsMyt_DEL_red_pct", 4.0),
       "blade DEL": ("RootMyc1_DEL_MNm", "RootMyc1_DEL_red_pct", 10.0)}


def load(stem):
    per = json.load(open(os.path.expanduser(stem + ".json")))["per_episode"]
    lev = list(csv.DictReader(open(os.path.expanduser(stem + ".csv"))))
    assert len(per) == len(lev)
    return per, lev


D = {}
for lab, stem in PAIRS:
    per, lev = load(stem)
    d = {"wind": np.array([r["mean_wind"] for r in per], float)}
    for q, (col, red, _m) in QTY.items():
        x = np.array([float(r[col]) for r in lev])
        p = np.array([r[red] for r in per], float)
        d[q] = x
        d[q + "_red"] = np.clip(p, -100, 100)
        d[q + "_base"] = x / (1.0 - p / 100.0)          # the paired GSPI level, from the reported reduction
    D[lab] = d

w0, w1 = D[PAIRS[0][0]], D[PAIRS[1][0]]
assert np.allclose(w0["wind"], w1["wind"])
for q in QTY:
    rel = np.abs(w0[q + "_base"] - w1[q + "_base"]) / np.maximum(1e-30, np.abs(w0[q + "_base"]))
    print(f"paired-baseline agreement, {q}: max relative difference {rel.max():.2e}")
winds = sorted(set(w0["wind"]))


def rayleigh(winds, vave, width):
    def cdf(v):
        return 1.0 - math.exp(-math.pi / 4.0 * (v / vave) ** 2)
    w = {u: cdf(u + width / 2) - cdf(u - width / 2) for u in winds}
    s = sum(w.values())
    return {u: x / s for u, x in w.items()}


W = rayleigh(winds, a.vave, a.bin)


def _agg(v, u, m, weights):
    """mean the levels within each wind bin, then take the weighted power mean across bins (m = 1 is a weighted mean)."""
    bins = sorted(set(u))
    per_bin = np.array([v[u == uu].mean() for uu in bins])
    wt = np.array([weights[uu] for uu in bins], float)
    wt = wt / wt.sum()
    return float(((wt * per_bin ** m).sum()) ** (1.0 / m))


def stat(d, q, m, kind, idx):
    """reduction of this controller against its paired baseline under one convention, on the episodes `idx`."""
    if kind == "per-episode":
        return float(np.mean(d[q + "_red"][idx]))
    x, b, u = d[q][idx], d[q + "_base"][idx], d["wind"][idx]
    weights = {uu: 1.0 for uu in set(u)} if kind == "level ratio" else W
    return 100.0 * (1.0 - _agg(x, u, m, weights) / _agg(b, u, m, weights))


strata: dict[float, list] = {}
for i, u in enumerate(w0["wind"]):
    strata.setdefault(u, []).append(i)
rng = np.random.default_rng(0)
boots = [np.array([k for ks in strata.values() for k in rng.choice(ks, size=len(ks), replace=True)])
         for _ in range(a.n_boot)]
allidx = np.arange(len(w0["wind"]))

print(f"\nfresh seeds 7-10, {len(allidx)} episodes; difference = scheduled MPC minus tuned ROSCO, in points of"
      f" reduction against the same paired GSPI\n")
print(f"{'quantity':>10} {'convention':>13} | {'tuned ROSCO':>12} {'scheduled MPC':>14} | {'difference':>11}  95 % interval")
for q, (_c, _r, m) in QTY.items():
    for kind in ("per-episode", "level ratio", "Rayleigh"):
        A = stat(w0, q, m, kind, allidx)
        B = stat(w1, q, m, kind, allidx)
        dd = np.array([stat(w1, q, m, kind, ix) - stat(w0, q, m, kind, ix) for ix in boots])
        lo, hi = np.percentile(dd, [2.5, 97.5])
        flag = "   <-- interval contains zero" if lo <= 0 <= hi else ""
        print(f"{q:>10} {kind:>13} | {A:11.1f}% {B:13.1f}% | {B - A:+10.1f}  [{lo:+.1f}, {hi:+.1f}]{flag}")
    print()
