"""Checks demanded by the review of 2026-09-23 (no simulations; reads the evaluation artefacts).

1. Is the power term of J the same information as the speed term? Above rated the torque is constant, so P = T*omega
   and the two normalised errors should be proportional. Reports the correlation over episodes and the rank
   correlation of the two reductions, and recomputes J without the power term to see whether any ranking moves.
2. Table 3 against Fig. 1: the weighting table aggregates DEL LEVELS (a ratio of means) while J averages per-episode
   percentage reductions. Prints both conventions side by side for every controller so the difference is stated
   rather than implied.
3. Paired intervals: recomputes the gated layer's interval against its base on the two held-out sets MERGED
   (eight seeds per wind speed, 56 episodes) instead of four, and prints the spread across training seeds next to
   the within-wind interval.

    ~/wtrl/run.sh python scripts/dev/review_checks.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

EXP = os.path.expanduser("~/wtrl/exp")
R6 = f"{EXP}/mpcsearch600"
T = ("power_mse_red_pct", "gen_speed_mse_red_pct", "TwrBsMyt_DEL_red_pct", "RootMyc1_DEL_red_pct")
NAMES = ("power MSE", "speed MSE", "tower DEL", "blade DEL")


def load(p, sl=None):
    p = os.path.expanduser(p)
    if not os.path.exists(p):
        return None
    per = json.load(open(p))["per_episode"]
    return per[sl] if sl else per


def terms(recs, drop=()):
    out = []
    for m in T:
        if m in drop:
            continue
        x = np.array([np.nan if r.get(m) is None else r[m] for r in recs], float)
        out.append(float(np.clip(x[np.isfinite(x)], -100, 100).mean()))
    return out


def J(recs, drop=()):
    t = terms(recs, drop)
    e, eb = sum(r["energy_MWh"] for r in recs), sum(r["energy_base_MWh"] for r in recs)
    return float(np.mean(t) - 20 * max(0.0, 100 * (1 - e / eb) - 1.0))


CTRL = {
    "tuned ROSCO": (f"{R6}/fresh/eval_rosco_tuned_s78910.json", f"{R6}/range/eval_rosco_tuned.json", slice(14, 42)),
    "scheduled MPC": (f"{R6}/fresh/eval_sched_mpc_s78910.json", f"{R6}/qtsched/eval_off_q6_18_22_s3456.json", None),
}
for d in sorted(glob.glob(f"{EXP}/trwCwG3t_s?")) + sorted(glob.glob(f"{EXP}/mrwCwG3t_s?")):
    n = os.path.basename(d)
    base = "tuned ROSCO" if n.startswith("t") else "scheduled MPC"
    CTRL[f"{base} + gated ({n[-2:]})"] = (f"{d}/eval_range_TIB_s78910.json", f"{d}/eval_range_TIB_s3456.json", None)

print("=" * 108)
print("1. Is the power term the same information as the speed term? (fresh seeds, per episode)")
print("=" * 108)
for lab, (fp, _, _) in CTRL.items():
    per = load(fp)
    if not per:
        continue
    p = np.array([r["power_mse_red_pct"] for r in per], float)
    w = np.array([r["gen_speed_mse_red_pct"] for r in per], float)
    k = np.isfinite(p) & np.isfinite(w)
    r = float(np.corrcoef(p[k], w[k])[0, 1])
    rs = float(np.corrcoef(np.argsort(np.argsort(p[k])), np.argsort(np.argsort(w[k])))[0, 1])
    print(f"{lab:>34}  Pearson r {r:6.4f}   Spearman {rs:6.4f}   mean gap {np.mean(w[k] - p[k]):+5.2f} points")

print("\n   J with all four terms vs J without the power term (fresh seeds), and the ranking under each")
rows = []
for lab, (fp, _, _) in CTRL.items():
    per = load(fp)
    if per:
        rows.append((lab, J(per), J(per, drop=("power_mse_red_pct",))))
r4 = {lab: i for i, (lab, _, _) in enumerate(sorted(rows, key=lambda x: -x[1]))}
r3 = {lab: i for i, (lab, _, _) in enumerate(sorted(rows, key=lambda x: -x[2]))}
for lab, j4, j3 in rows:
    flag = "" if r4[lab] == r3[lab] else f"   RANK MOVES {r4[lab]} -> {r3[lab]}"
    print(f"{lab:>34}  J(4 terms) {j4:6.2f}   J(no power) {j3:6.2f}{flag}")

print("\n" + "=" * 108)
print("2. Two aggregation conventions for the same episodes (fresh seeds): mean of per-episode reductions")
print("   (what J and Fig. 1 use) vs 1 - ratio of the means of the underlying quantity (what the weighting table uses)")
print("=" * 108)
COL = {"power MSE": "power_mse_R3", "speed MSE": "gen_speed_mse_R3",
       "tower DEL": "TwrBsMyt_DEL_MNm", "blade DEL": "RootMyc1_DEL_MNm"}
print(f"{'controller':>34} | " + " | ".join(f"{n:>19}" for n in NAMES))
print(f"{'':>34} | " + " | ".join(f"{'per-ep':>9} {'ratio':>9}" for _ in NAMES))
import csv as _csv
for lab, (fp, _, _) in CTRL.items():
    per = load(fp)
    cp = os.path.expanduser(fp).replace(".json", ".csv")
    if not per or not os.path.exists(cp):
        continue
    lev = list(_csv.DictReader(open(cp)))        # the raw levels live in the CSV, the reductions in the JSON
    cells = []
    for name, m in zip(NAMES, T):
        x = np.array([r[m] for r in per], float)
        per_ep = float(np.clip(x[np.isfinite(x)], -100, 100).mean())
        c = np.array([float(r[COL[name]]) for r in lev], float)
        b = c / (1.0 - x / 100.0)                      # the paired baseline level, from the reported reduction
        k = np.isfinite(c) & np.isfinite(b)
        ratio = 100.0 * (1.0 - c[k].mean() / b[k].mean())
        cells.append(f"{per_ep:8.1f}% {ratio:8.1f}%")
    print(f"{lab:>34} | " + " | ".join(cells))

print("\n   Are the two regulation metrics the same quantity? (raw levels per episode, constant torque above rated)")
for lab, (fp, _, _) in CTRL.items():
    cp = os.path.expanduser(fp).replace(".json", ".csv")
    if not os.path.exists(cp):
        continue
    lev = list(_csv.DictReader(open(cp)))
    pm = np.array([float(r["power_mse_R3"]) for r in lev])
    wm = np.array([float(r["gen_speed_mse_R3"]) for r in lev])
    k = np.isfinite(pm) & np.isfinite(wm) & (wm > 0)
    rat = pm[k] / wm[k]
    print(f"{lab:>34}  corr(levels) {np.corrcoef(pm[k], wm[k])[0, 1]:6.4f}   "
          f"power/speed ratio {rat.mean():5.2f} +- {rat.std():4.2f}  (min {rat.min():4.2f}, max {rat.max():4.2f})")

print("\n" + "=" * 108)
print("3. Paired intervals on the two held-out sets MERGED (8 seeds per wind speed, 56 episodes)")
print("   and the spread across training seeds, which the within-wind interval does not contain")
print("=" * 108)
rng = np.random.default_rng(0)
BASE = {"tuned ROSCO": (f"{R6}/range/eval_rosco_tuned.json", slice(14, 42), f"{R6}/fresh/eval_rosco_tuned_s78910.json"),
        "scheduled MPC": (f"{R6}/qtsched/eval_off_q6_18_22_s3456.json", None, f"{R6}/fresh/eval_sched_mpc_s78910.json")}
for base_lab, (bh, bsl, bf) in BASE.items():
    base = (load(bh, bsl) or []) + (load(bf) or [])
    if not base:
        continue
    strata = {}
    for i, r in enumerate(base):
        strata.setdefault(r["mean_wind"], []).append(i)
    print(f"\n{base_lab}: merged base J {J(base):6.2f} over {len(base)} episodes, {len(strata[min(strata)])} seeds per wind speed")
    seed_js = []
    for lab, (fp, hp, _) in CTRL.items():
        if not lab.startswith(base_lab) or "gated" not in lab:
            continue
        A = (load(hp) or []) + (load(fp) or [])
        if len(A) != len(base):
            continue
        d = J(A) - J(base)
        seed_js.append(J(A))
        dd = np.empty(4000)
        for i in range(4000):
            pick = [k for ks in strata.values() for k in rng.choice(ks, size=len(ks), replace=True)]
            dd[i] = J([A[k] for k in pick]) - J([base[k] for k in pick])
        lo, hi = np.percentile(dd, [2.5, 97.5])
        print(f"{lab:>34}  J {J(A):6.2f}  diff {d:+5.2f}  [{lo:+.2f}, {hi:+.2f}]  (width {hi - lo:.2f})")
    if len(seed_js) > 1:
        print(f"{'across training seeds':>34}  J {np.mean(seed_js):6.2f} +- {np.std(seed_js, ddof=1):.2f} (s.d. over {len(seed_js)} seeds),"
              f" range {max(seed_js) - min(seed_js):.2f} points")
