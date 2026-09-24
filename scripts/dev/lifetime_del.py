"""Wind-distribution-weighted equivalent fatigue loads from the per-episode damage-equivalent loads (no simulations).

The results tables report the mean percentage reduction over equally weighted mean wind speeds. Fatigue work in wind
energy aggregates over the site wind distribution instead: IEC 61400-1 uses a Rayleigh distribution (Weibull with
shape 2) whose mean is 0.2 x the reference wind speed of the turbine class. Damage adds linearly (Palmgren-Miner) and
a damage-equivalent load of Wohler exponent m scales as DEL^m, so the weighted equivalent load over a set of wind bins
is

    DEL_eq = ( sum_i w_i * DEL_i^m / sum_i w_i )^(1/m),   w_i = probability of bin i,

with the bin probability from the Rayleigh CDF over the bin edges. This script reads the per-episode DEL values in
MN m from the evaluation CSVs, averages them per mean wind speed, and reports DEL_eq for each controller together
with the reduction relative to a reference controller.

Scope. The evaluations cover 12-24 m/s. The weights are therefore renormalised over that range, and the numbers are
"above-rated equivalent loads": they answer "over the above-rated part of the wind distribution, by how much does the
controller change the equivalent load", not "lifetime damage including below-rated winds". Both controllers of a
comparison are weighted identically, so the RATIO is the meaningful quantity. Below 15 m/s the gated residual is
inactive by construction, so for those rows the below-rated contribution is unchanged by assumption; for the MPC rows
that is an assumption, not a measurement, and must be stated as such.

    ~/wtrl/run.sh python scripts/dev/lifetime_del.py [--csv docs/tables/lifetime_del.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser("~/wtrl/exp"))
ap.add_argument("--vave", type=float, nargs="+", default=[10.0, 8.5, 7.5],
                help="annual mean wind speeds of the Rayleigh distribution (IEC class I / II / III = 10 / 8.5 / 7.5 m/s)")
ap.add_argument("--bin", type=float, default=2.0, help="width of the mean-wind bins [m/s]")
ap.add_argument("--csv", default=None)
a = ap.parse_args()
M = {"tower": 4.0, "blade": 10.0}       # Wohler exponents: welded steel tower, composite blade
COL = {"tower": "TwrBsMyt_DEL_MNm", "blade": "RootMyc1_DEL_MNm"}
# the regulation metrics add linearly over the distribution (a mean square error, not a damage sum)
REG = {"power MSE": "power_mse_R3", "speed MSE": "gen_speed_mse_R3"}
R6 = f"{a.exp}/mpcsearch600"

CONTROLLERS = [
    # label, csv path, per-episode JSON with the reductions (to back out the GSPI baseline), seed set
    ("tuned ROSCO", f"{R6}/fresh/eval_rosco_tuned_s78910.csv", f"{R6}/fresh/eval_rosco_tuned_s78910.json", "s78910"),
    ("scheduled MPC", f"{R6}/fresh/eval_sched_mpc_s78910.csv", f"{R6}/fresh/eval_sched_mpc_s78910.json", "s78910"),
]
for d in sorted(glob.glob(f"{a.exp}/trwCwG3t_s?")) + sorted(glob.glob(f"{a.exp}/mrwCwG3t_s?")):
    n = os.path.basename(d)
    base = "tuned ROSCO" if n.startswith("t") else "scheduled MPC"
    CONTROLLERS.append((f"{base} + gated residual ({n[-2:]})", f"{d}/eval_range_TIB_s78910.csv", f"{d}/eval_range_TIB_s78910.json", "s78910"))


def per_wind_del(csv_path, json_path):
    """-> {quantity: {mean_wind: DEL in MN m}} for the controller, and the same for its paired GSPI baseline."""
    rows = list(csv.DictReader(open(csv_path)))
    red = {}
    if os.path.exists(json_path):
        for pe in json.load(open(json_path))["per_episode"]:
            red.setdefault(pe["mean_wind"], []).append(pe)
    ctrl, base = {}, {}
    for q, col in {**COL, **REG}.items():
        by_u, by_u_b = {}, {}
        for i, r in enumerate(rows):
            u = float(r["mean_wind"])
            d = float(r[col])
            by_u.setdefault(u, []).append(d)
            # the paired GSPI baseline DEL follows from the reported reduction: d = d_base * (1 - red/100)
            pe = red.get(u, [])
            k = sum(1 for rr in rows[:i] if float(rr["mean_wind"]) == u)
            key = {"tower": "TwrBsMyt_DEL_red_pct", "blade": "RootMyc1_DEL_red_pct",
                   "power MSE": "power_mse_red_pct", "speed MSE": "gen_speed_mse_red_pct"}[q]
            if k < len(pe) and pe[k].get(key) is not None and pe[k][key] == pe[k][key]:
                by_u_b.setdefault(u, []).append(d / max(1e-12, 1.0 - float(pe[k][key]) / 100.0))
        ctrl[q] = {u: float(np.mean(v)) for u, v in by_u.items() if v}
        base[q] = {u: float(np.mean(v)) for u, v in by_u_b.items() if v}
    return ctrl, base


def rayleigh_weights(winds, vave, width):
    """Probability of each mean-wind bin under a Rayleigh distribution, renormalised over the evaluated bins."""
    def cdf(v):
        return 1.0 - math.exp(-math.pi / 4.0 * (v / vave) ** 2)
    w = {u: cdf(u + width / 2) - cdf(u - width / 2) for u in winds}
    s = sum(w.values())
    return {u: x / s for u, x in w.items()}


def del_eq(per_u, w, m):
    num = sum(w[u] * per_u[u] ** m for u in w if u in per_u)
    return num ** (1.0 / m)


ref = {}
rows_out = []
for label, cp, jp, _ in CONTROLLERS:
    if not os.path.exists(cp):
        continue
    ctrl, base = per_wind_del(cp, jp)
    winds = sorted(ctrl["tower"])
    for vave in a.vave:
        w = rayleigh_weights(winds, vave, a.bin)
        rec = {"controller": label, "Vave": vave}
        for q in list(COL) + list(REG):
            m = M.get(q, 1.0)                      # exponent 1 for the regulation metrics: a weighted mean
            uu = {u: x for u, x in w.items() if u in ctrl[q] and u in base[q]}
            if not uu:
                continue
            de_c, de_b = del_eq(ctrl[q], uu, m), del_eq(base[q], uu, m)
            rec[f"{q}_DEL_eq_MNm"] = float(f"{de_c:.6g}")   # 6 significant digits: the MSE aggregates are O(1e-4)
            rec[f"{q}_vs_GSPI_pct"] = round(100.0 * (1.0 - de_c / de_b), 2)
            # the equally weighted column must differ from the Rayleigh one ONLY in the weights, so it is the
            # same power mean with uniform weights, not an arithmetic mean of the bin levels
            uu_eq = {u: 1.0 / len(uu) for u in uu}
            rec[f"{q}_equal_mean_pct"] = round(100.0 * (1.0 - del_eq(ctrl[q], uu_eq, m) / del_eq(base[q], uu_eq, m)), 2)
            ref.setdefault((q, vave), {})[label] = de_c
        rows_out.append(rec)

print("Rayleigh-weighted equivalent fatigue loads over the evaluated above-rated bins (12-24 m/s), fresh seeds 7-10")
print("reduction vs the paired GSPI baseline; 'equal' = the same aggregation with uniform weights, so the columns differ only in the weights\n")
for vave in a.vave:
    print(f"--- IEC Rayleigh, annual mean {vave:g} m/s"
          + ("   (class I)" if vave == 10 else "   (class II)" if vave == 8.5 else "   (class III)" if vave == 7.5 else ""))
    print(f"{'controller':>44} | {'power MSE':>21} | {'speed MSE':>21} | {'tower DEL':>21} | {'blade DEL':>21}")
    print(f"{'':>44} | {'weighted':>10} {'equal':>9} | {'weighted':>10} {'equal':>9} | {'weighted':>10} {'equal':>9} | {'weighted':>10} {'equal':>9}")
    for r in [r for r in rows_out if r["Vave"] == vave]:
        cells = "".join(f" | {r.get(q + '_vs_GSPI_pct', float('nan')):9.1f}% {r.get(q + '_equal_mean_pct', float('nan')):8.1f}%"
                        for q in ("power MSE", "speed MSE", "tower", "blade"))
        print(f"{r['controller']:>44}" + cells)
    # relative to the tuned ROSCO, the honest industrial reference
    print(f"{'  relative to the tuned ROSCO:':>44}")
    for r in [r for r in rows_out if r["Vave"] == vave and r["controller"] != "tuned ROSCO"]:
        out = []
        for q in ("power MSE", "speed MSE", "tower", "blade"):
            base_eq, own = ref[(q, vave)].get("tuned ROSCO"), ref[(q, vave)].get(r["controller"])
            if base_eq and own:
                out.append(f"{q} {100.0 * (1.0 - own / base_eq):+6.1f}%")
        print(f"{r['controller']:>44} | " + "  ".join(out))
    print()

if a.csv and rows_out:
    keys = list(dict.fromkeys(k for r in rows_out for k in r))
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows_out)
    print("->", a.csv)
