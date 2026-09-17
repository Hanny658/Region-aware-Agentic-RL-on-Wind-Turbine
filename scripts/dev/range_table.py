"""Controllers over the above-rated wind range (no simulations): table, per-wind breakdown, paired bootstrap.

Reads the evaluations of scripts/wsl/campaign_must1_range.sh: 12-24 m/s in 2 m/s steps, IEC normal turbulence
class B, 600 s, TurbSim seeds 1-6, 42 episodes, none of them used to select anything. Every controller ran as the
environment's base with a zero residual, so every row is deterministic given the wind file and the uncertainty of
a difference between two controllers is the wind realisation. Differences are bootstrapped over the 42 episodes,
both controllers resampled with the same indices, stratified by mean wind speed; the objective is recomputed on
every resample exactly as eval/fitness.py does (per-episode reductions clipped to +-100, mean per term, mean of
the four terms, energy penalty from the summed energies).

    ~/wtrl/run.sh python scripts/dev/range_table.py [--csv docs/tables/range] [--extra tag ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.expanduser("~/wtrl/exp/mpcsearch600/range"))
ap.add_argument("--csv", default=None, help="prefix: writes <prefix>_controllers.csv, <prefix>_per_wind.csv, <prefix>_paired.csv")
ap.add_argument("--extra", nargs="*", default=[], help="additional eval tags to list (e.g. rosco_tuned)")
ap.add_argument("--pairs", nargs="*", default=[], help="additional comparisons A:B")
ap.add_argument("--n_boot", type=int, default=20000)
ap.add_argument("--scored_s", type=float, default=580.0)
a = ap.parse_args()
TERMS = ("power_mse_red_pct", "gen_speed_mse_red_pct", "TwrBsMyt_DEL_red_pct", "RootMyc1_DEL_red_pct")
SHORT = ("power", "speed", "tower", "blade")
NAMES = {"identity_gspi": "GSPI (identity check)", "towerdamper": "ROSCO + tower damper", "rosco_tuned": "tuned ROSCO",
         "nominal_cp1": "nominal MPC", "offset_cp1": "offset-free MPC", "jselected_cp1": "J-selected MPC",
         "knee3_cp1": "duty knee 3", "knee2_cp1": "duty knee 2", "nominal_cp0.95": "nominal MPC, Cp/Ct x0.95",
         "offset_cp0.95": "offset-free MPC, Cp/Ct x0.95", "knee2_cp0.95": "duty knee 2, Cp/Ct x0.95"}
ORDER = ["identity_gspi", "towerdamper", "rosco_tuned", "nominal_cp1", "offset_cp1", "jselected_cp1", "knee3_cp1", "knee2_cp1",
         "nominal_cp0.95", "offset_cp0.95", "knee2_cp0.95"] + list(a.extra)
PAIRS = [("offset_cp1", "nominal_cp1"), ("offset_cp0.95", "nominal_cp0.95"), ("nominal_cp0.95", "nominal_cp1"),
         ("offset_cp0.95", "offset_cp1"), ("knee2_cp1", "offset_cp1"), ("knee3_cp1", "offset_cp1"), ("jselected_cp1", "offset_cp1"),
         ("knee2_cp1", "jselected_cp1"), ("knee2_cp0.95", "knee2_cp1"), ("towerdamper", "identity_gspi"),
         ("rosco_tuned", "identity_gspi"), ("offset_cp1", "rosco_tuned"), ("knee2_cp1", "rosco_tuned")] + [tuple(p.split(":")) for p in a.pairs]


def load(tag):
    p = f"{a.dir}/eval_{tag}.json"
    return json.load(open(p))["per_episode"] if os.path.exists(p) else None


def peak_speed(tag):
    """largest generator speed over all episodes, relative to rated (from the per-episode CSV of the evaluation)"""
    q = f"{a.dir}/eval_{tag}.csv"
    if not os.path.exists(q):
        return float("nan")
    return max(float(r["gen_speed_max_rel"]) for r in csv.DictReader(open(q)))


def terms_of(recs):
    out = []
    for t in TERMS:
        v = np.array([np.nan if r.get(t) is None else r[t] for r in recs], float)
        v = np.clip(v[np.isfinite(v)], -100, 100)
        out.append(v.mean() if len(v) else np.nan)
    return out


def objective(recs):
    t = terms_of(recs)
    e, eb = sum(r["energy_MWh"] for r in recs), sum(r["energy_base_MWh"] for r in recs)
    loss = 100 * (1 - e / eb)
    return float(np.nanmean(t) - 20 * max(0.0, loss - 1.0)), t, loss


rows, wind_rows, pair_rows = [], [], []
print(f"{'controller':>30} {'J':>6} {'duty':>6} {'xGSPI':>6} {'energy':>7} {'peak w':>6} | power / speed / tower / blade | J by mean wind")
for tag in ORDER:
    per = load(tag)
    if per is None:
        continue
    J, t, loss = objective(per)
    trav = np.mean([r["pitch_travel_deg"] for r in per]); base = np.mean([r["pitch_travel_base_deg"] for r in per])
    by = {}
    for r in per:
        by.setdefault(r["mean_wind"], []).append(r)
    jw = []
    for u in sorted(by):
        ju, tu, _ = objective(by[u])
        jw.append(ju)
        wind_rows.append({"controller": NAMES.get(tag, tag), "mean_wind": u, "n": len(by[u]), "J": round(ju, 2),
                          **{s: round(x, 2) for s, x in zip(SHORT, tu)},
                          "duty_x_gspi": round(np.mean([r["pitch_travel_deg"] for r in by[u]]) / np.mean([r["pitch_travel_base_deg"] for r in by[u]]), 3)})
    print(f"{NAMES.get(tag, tag):>30} {J:6.2f} {trav / a.scored_s:6.3f} {trav / base:6.2f} {-loss:+6.2f}% {peak_speed(tag):6.3f} | "
          + " / ".join(f"{x:5.1f}" for x in t) + " | " + " ".join(f"{x:5.1f}" for x in jw))
    rows.append({"controller": NAMES.get(tag, tag), "n_episodes": len(per), "J": round(J, 2), **{s: round(x, 2) for s, x in zip(SHORT, t)},
                 "energy_change_pct": round(-loss, 3), "duty_deg_per_s": round(trav / a.scored_s, 4), "duty_x_gspi": round(trav / base, 3), "peak_gen_speed_rel": round(peak_speed(tag), 4),
                 **{f"J_U{u:g}": round(x, 2) for u, x in zip(sorted(by), jw)}})

rng = np.random.default_rng(0)
print(f"\npaired bootstrap over the episodes (stratified by mean wind, {a.n_boot} resamples): A - B, 95 % interval")
print(f"{'A':>30}   {'B':<30} {'J':>17} | " + " ".join(f"{s:>17}" for s in SHORT))
for ta, tb in PAIRS:
    A, B = load(ta), load(tb)
    if A is None or B is None:
        continue
    assert len(A) == len(B) and all(x["mean_wind"] == y["mean_wind"] and abs(x["energy_base_MWh"] - y["energy_base_MWh"]) < 1e-9
                                    for x, y in zip(A, B)), f"episode order differs: {ta} vs {tb}"
    strata = {}
    for i, r in enumerate(A):
        strata.setdefault(r["mean_wind"], []).append(i)
    ja, tta, _ = objective(A); jb, ttb, _ = objective(B)
    d = np.empty((a.n_boot, 5))
    for i in range(a.n_boot):
        pick = [k for ks in strata.values() for k in rng.choice(ks, size=len(ks), replace=True)]
        oa, ob = objective([A[k] for k in pick]), objective([B[k] for k in pick])
        d[i, 0] = oa[0] - ob[0]; d[i, 1:] = np.array(oa[1]) - np.array(ob[1])
    lo, hi = np.nanpercentile(d, [2.5, 97.5], axis=0)
    point = [ja - jb] + list(np.array(tta) - np.array(ttb))
    print(f"{NAMES.get(ta, ta):>30} - {NAMES.get(tb, tb):<30} " + " | ".join(f"{p:+5.2f} [{l:+5.2f},{h:+5.2f}]" for p, l, h in zip(point, lo, hi)))
    rec = {"A": NAMES.get(ta, ta), "B": NAMES.get(tb, tb), "n_episodes": len(A)}
    for s, p, l, h in zip(("J",) + SHORT, point, lo, hi):
        rec[f"{s}_diff"], rec[f"{s}_lo"], rec[f"{s}_hi"] = round(p, 3), round(l, 3), round(h, 3)
    pair_rows.append(rec)

if a.csv:
    for name, rr in (("controllers", rows), ("per_wind", wind_rows), ("paired", pair_rows)):
        if rr:
            keys = list(dict.fromkeys(k for r in rr for k in r))
            with open(f"{a.csv}_{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rr)
            print("->", f"{a.csv}_{name}.csv")
