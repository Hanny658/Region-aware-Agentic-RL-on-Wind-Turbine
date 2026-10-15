"""Does the fatigue conclusion depend on the Wohler exponent? (2026-09-23, reads campaign_mexp.sh's evaluations.)

The paper reports tower-base DEL at m = 4 and blade-root DEL at m = 10, which are the conventional values, and a
referee will ask whether the conclusion survives the neighbouring ones. agents/rollout.py records the tower DEL at
m = 3, 4, 5 and the blade DEL at m = 8, 10, 12 in the same rainflow pass, so the controller side comes from the
re-evaluations of campaign_mexp.sh. The paired GSPI baseline is recomputed here from the raw channels stored in the
baseline .npz, so no baseline simulation is repeated either.

    ~/wtrl/run.sh python scripts/dev/wohler_sensitivity.py [--csv docs/tables/wohler.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from eval.metrics import del_rainflow  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser("~/wtrl/exp"))
ap.add_argument("--home", default=os.path.expanduser("~/wtrl600"))
ap.add_argument("--warmup_s", type=float, default=20.0)
ap.add_argument("--csv", default=None)
a = ap.parse_args()
R6 = f"{a.exp}/mpcsearch600"
TOWER, BLADE = (3, 4, 5), (8, 10, 12)

# Every evaluation written after 2026-09-23 records the extra exponents in its ordinary CSV, so the native
# baseline (s52) and the agent's leg-1 rows (s55) need no re-evaluation; the hand-scaled row of s29 is not a
# tuning file and is kept out. Rows evaluated before that date fall back to campaign_mexp.sh's re-evaluation.
CACHE = f"{a.exp}/roscosearch_native/_verify/cache"
CTRL = [("tuned baseline, native (s52)", f"{R6}/fresh/eval_rosco_tuned_native_s78910.csv"),
        ("agent one-line tuning (corner 3.0)", f"{CACHE}/eval_ddcdd3521ca6_s78910_90365b.csv"),
        ("agent tuning, corner 6.5", f"{CACHE}/eval_52645b613ce6_s78910_90365b.csv"),
        ("scheduled MPC", f"{R6}/mexp/eval_sched_mpc_mexp.csv")]
for d in sorted(glob.glob(f"{a.exp}/tnrwCwG3t_s?")) + sorted(glob.glob(f"{a.exp}/mrwCwG3t_s?")):
    n = os.path.basename(d)
    base = "native baseline" if n.startswith("t") else "scheduled MPC"
    cands = [f"{d}/eval_range_TIB_s78910_mexp.csv", f"{d}/eval_range_TIB_s78910.csv"]
    CTRL.append((f"{base} + gated layer ({n[-2:]})", next((c for c in cands if os.path.exists(c)), cands[0])))


def baseline_del():
    """{wind_file: {('tower'|'blade', m): DEL}} of the paired GSPI, recomputed from the stored raw channels."""
    out: dict[str, dict] = {}
    for p in sorted(glob.glob(f"{a.home}/baselines/openfast/*TIB*.npz")):
        name = os.path.basename(p)[:-4]
        try:
            with np.load(p) as z:
                t = z["outb_Time"]
                k = t >= a.warmup_s
                dt = float(t[1] - t[0])
                d = {}
                for m in TOWER:
                    d[("tower", m)] = float(del_rainflow(z["outb_TwrBsMyt"][k], dt, m) / 1e3)
                for m in BLADE:
                    d[("blade", m)] = float(del_rainflow(z["outb_RootMyc1"][k], dt, m) / 1e3)
            out[name] = d
        except Exception as e:  # noqa: BLE001 - a truncated baseline is reported, not fatal
            print(f"  ! {name}: {type(e).__name__}")
    return out


BASE = baseline_del()
if not BASE:
    sys.exit(f"no baselines under {a.home}")
print(f"{len(BASE)} paired baseline episodes recomputed at m = {TOWER} (tower) and {BLADE} (blade)\n")

COL = {("tower", 3): "TwrBsMyt_DEL_m3_MNm", ("tower", 4): "TwrBsMyt_DEL_MNm", ("tower", 5): "TwrBsMyt_DEL_m5_MNm",
       ("blade", 8): "RootMyc1_DEL_m8_MNm", ("blade", 10): "RootMyc1_DEL_MNm", ("blade", 12): "RootMyc1_DEL_m12_MNm"}
rows = []
for lab, cp in CTRL:
    cp = os.path.expanduser(cp)
    if not os.path.exists(cp):
        continue
    eps = list(csv.DictReader(open(cp)))
    if COL[("tower", 3)] not in eps[0]:
        print(f"{lab:>34}  (evaluated before the extra exponents were recorded; re-run campaign_mexp.sh)")
        continue
    rec = {"controller": lab, "n_episodes": len(eps)}
    for q, ms in (("tower", TOWER), ("blade", BLADE)):
        for m in ms:
            red = []
            for r in eps:
                b = BASE.get(r["wind_file"])
                if b and COL[(q, m)] in r:
                    red.append(100.0 * (1.0 - float(r[COL[(q, m)]]) / b[(q, m)]))
            if red:
                rec[f"{q}_m{m}"] = round(float(np.mean(red)), 2)
    rows.append(rec)

if not rows:
    sys.exit("no re-evaluated controllers found; campaign_mexp.sh has not finished")
print(f"{'controller':>34} | " + " | ".join(f"tower m={m}" for m in TOWER) + " | " + " | ".join(f"blade m={m}" for m in BLADE))
for r in rows:
    print(f"{r['controller']:>34} | " + " | ".join(f"{r.get(f'tower_m{m}', float('nan')):9.1f}%" for m in TOWER)
          + " | " + " | ".join(f"{r.get(f'blade_m{m}', float('nan')):9.1f}%" for m in BLADE))
print("\nmean of the per-episode percentage reductions against the paired shipped GSPI, fresh seeds 7-10.")
print("The conclusion is exponent-independent if the ordering of the controllers is the same in every column.")

if a.csv:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("->", a.csv)
