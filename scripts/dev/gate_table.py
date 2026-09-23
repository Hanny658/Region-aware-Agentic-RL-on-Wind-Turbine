"""Wind-gated residual (campaign_gate.sh): 600 s rows against the original GSPI, on the held-out range seeds 3-6 and on
the fresh TurbSim seeds 7-10 that no training, selection or earlier report has seen.

Every row is compared with its own base controller on the same episodes (paired bootstrap over episodes, stratified by
mean wind): the scheduled MPC of roadmap s33 for the mrw/mg arms, the tuned ROSCO of s29 for the trw arm. Reports J,
the four terms, pitch travel, the per-wind worst term and the difference to the base.

    ~/wtrl/run.sh python scripts/dev/gate_table.py [--csv docs/tables/gated_residual.csv]
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
ap.add_argument("--csv", default=None)
ap.add_argument("--n_boot", type=int, default=10000)
a = ap.parse_args()
T = ("power_mse_red_pct", "gen_speed_mse_red_pct", "TwrBsMyt_DEL_red_pct", "RootMyc1_DEL_red_pct")
ARMS = {"mrwCwG3t": ("agent reward, gated", "mpc"), "mgCwG3t": ("fixed reward, gated", "mpc"),
        "trwCwG3t": ("agent reward, gated", "rosco")}
R6 = f"{a.exp}/mpcsearch600"
BASES = {
    ("mpc", "s3456"): (f"{R6}/qtsched/eval_off_q6_18_22_s3456.json", None, "scheduled MPC"),
    ("mpc", "s78910"): (f"{R6}/fresh/eval_sched_mpc_s78910.json", None, "scheduled MPC"),
    ("rosco", "s3456"): (f"{R6}/range/eval_rosco_tuned.json", slice(14, 42), "tuned ROSCO"),
    ("rosco", "s78910"): (f"{R6}/fresh/eval_rosco_tuned_s78910.json", None, "tuned ROSCO"),
}


def J(recs):
    t = []
    for m in T:
        x = np.array([np.nan if r.get(m) is None else r[m] for r in recs], float)
        x = np.clip(x[np.isfinite(x)], -100, 100)
        t.append(x.mean())
    e, eb = sum(r["energy_MWh"] for r in recs), sum(r["energy_base_MWh"] for r in recs)
    return float(np.mean(t) - 20 * max(0.0, 100 * (1 - e / eb) - 1.0)), t


def load(p, sl=None):
    if not os.path.exists(p):
        return None
    per = json.load(open(p))["per_episode"]
    return per[sl] if sl else per


rng = np.random.default_rng(0)
rows = []
for seeds in ("s3456", "s78910"):
    print(f"\n=== 600 s range set, TurbSim seeds {seeds[1:]} ({'held-out' if seeds == 's3456' else 'FRESH, unseen by any selection or report'}), vs the original GSPI")
    printed_base = set()
    for arm, (desc, base_kind) in ARMS.items():
        bp, bsl, bname = BASES[(base_kind, seeds)]
        base = load(bp, bsl)
        if base is None:
            continue
        jb, tb = J(base)
        strata = {}
        for i, r in enumerate(base):
            strata.setdefault(r["mean_wind"], []).append(i)
        if bname not in printed_base:
            trav = np.mean([r["pitch_travel_deg"] for r in base]) / np.mean([r["pitch_travel_base_deg"] for r in base])
            print(f"{bname + ' alone':>34} J {jb:6.2f}  travel x{trav:.2f} | " + " / ".join(f"{x:5.1f}" for x in tb))
            printed_base.add(bname)
        for d in sorted(glob.glob(f"{a.exp}/{arm}_s?")):
            A = load(f"{d}/eval_range_TIB_{seeds}.json")
            if A is None:
                continue
            assert len(A) == len(base) and all(x["mean_wind"] == y["mean_wind"] and abs(x["energy_base_MWh"] - y["energy_base_MWh"]) < 1e-9
                                               for x, y in zip(A, base)), f"{d} {seeds}"
            ja, ta = J(A)
            trav = np.mean([r["pitch_travel_deg"] for r in A]) / np.mean([r["pitch_travel_base_deg"] for r in A])
            dd = np.empty(a.n_boot)
            for i in range(a.n_boot):
                pick = [k for ks in strata.values() for k in rng.choice(ks, size=len(ks), replace=True)]
                dd[i] = J([A[k] for k in pick])[0] - J([base[k] for k in pick])[0]
            lo, hi = np.percentile(dd, [2.5, 97.5])
            worst = (0.0, "")
            for u, ks in strata.items():
                da = np.array(J([A[k] for k in ks])[1]) - np.array(J([base[k] for k in ks])[1])
                for nm, v in zip(("power", "speed", "tower", "blade"), da):
                    if np.isfinite(v) and v < worst[0]:
                        worst = (float(v), f"{nm}@{u:g}")
            print(f"{os.path.basename(d):>16} {desc:>22.22} J {ja:6.2f} {ja - jb:+5.2f} [{lo:+.2f},{hi:+.2f}] travel x{trav:.2f} | "
                  + " / ".join(f"{x:5.1f}" for x in ta) + f" | worst per wind vs base {worst[0]:+5.1f} {worst[1]}")
            rows.append({"run": os.path.basename(d), "arm": desc, "base": bname, "seeds": seeds, "J": round(ja, 2),
                         "base_J": round(jb, 2), "diff": round(ja - jb, 3), "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
                         **{t: round(x, 2) for t, x in zip(T, ta)}, "travel_x_gspi": round(float(trav), 3),
                         "worst_per_wind_diff": round(worst[0], 2), "worst_at": worst[1]})
if a.csv and rows:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("\n->", a.csv)
