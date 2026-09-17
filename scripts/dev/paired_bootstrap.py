"""Paired statistics for deterministic controllers over held-out wind realisations (no simulations).

An MPC evaluation is deterministic given the wind file, so the uncertainty of "controller A vs controller B" is
the wind realisation, not a training seed. Every controller here was evaluated on the same held-out episodes
(two sets of 12 at 600 s: 8 / 12.5 / 15 m/s x TurbSim seeds 3-6 and 7-10), so the comparison is paired by
episode. For each pair this script recomputes the objective exactly as eval/fitness.py does (per-episode
reductions clipped to +-100, averaged per term over the episodes where the term is defined, mean of the four
terms, energy penalty from the summed energies) on bootstrap resamples of the 24 episodes, resampling
A and B with the same indices, stratified by mean wind speed so every resample keeps the wind mix.

Also reports, per controller, the quantities the literature scan says belong next to any load claim: actuator
duty, peak generator speed, energy change, and the per-wind-speed breakdown of the four terms.

    WTRL_HOME=~/wtrl600 ~/wtrl/run.sh python scripts/dev/paired_bootstrap.py [--csv docs/tables/paired_bootstrap.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.mpc_param_search import INCUMBENT0, key_of  # noqa: E402
from scripts.rosco_tune import clip as rosco_clip, key_of as rosco_key  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser("~/wtrl/exp"))
ap.add_argument("--csv", default=None)
ap.add_argument("--n_boot", type=int, default=20000)
a = ap.parse_args()
R = f"{a.exp}/mpcsearch600"
TERMS = ("power_mse_red_pct", "gen_speed_mse_red_pct", "TwrBsMyt_DEL_red_pct", "RootMyc1_DEL_red_pct")
JSEL = {"horizon": 20, "r": 0.255, "qt": 3.25, "wc_v": 0.38, "tau_adapt": 5.2, "adapt": "rls"}


def files(kind, cp="cp1"):
    k0, kj = key_of(INCUMBENT0), key_of(JSEL)
    rb = f"{a.exp}/roscotune600/best.json"
    kr = rosco_key(rosco_clip(json.load(open(rb))["params"])) if os.path.exists(rb) else "none"
    m = {"offset-free reference": [f"{R}/cache/eval_{k0}_{cp}_{s}.json" for s in ("s3456", "s78910")],
         "J-selected point": [f"{R}/cache/eval_{kj}_{cp}_{s}.json" for s in ("s3456", "s78910")],
         "knee2 (2.9x GSPI travel)": [f"{R}/refs/eval_knee2_{cp}_{s}.json" for s in ("s3456", "s78910")],
         "knee3 (4.0x)": [f"{R}/refs/eval_knee3_{cp}_{s}.json" for s in ("s3456", "s78910")],
         "knee1 (2.4x)": [f"{R}/refs/eval_knee1_{cp}_{s}.json" for s in ("s3456", "s78910")],
         "nominal MPC": [f"{R}/refs/eval_nominal_{cp}_{s}.json" for s in ("s3456", "s78910")],
         "adaptive (RLS) reference": [f"{R}/refs/eval_rls_{cp}_{s}.json" for s in ("s3456", "s78910")],
         "offset-free + 3P notch": [f"{R}/refs/eval_offset_notch5_{cp}_{s}.json" for s in ("s3456", "s78910")],
         "ROSCO + tower damper": [f"{R}/refs/eval_towerdamper_{s}.json" for s in ("s3456", "s78910")],
         "tuned ROSCO": [f"{a.exp}/roscotune600/cache/eval_{kr}_{s}.json" for s in ("s3456", "s78910")]}
    return m[kind]


def episodes(paths):
    """-> dict wind_file -> per-episode record (both held-out sets merged), or None if a file is missing."""
    out = {}
    for p in paths:
        if not os.path.exists(p):
            return None
        for pe in json.load(open(p))["per_episode"]:
            out[pe.get("wind_file") or f"{pe['mean_wind']}_{len(out)}"] = pe
    return out


def objective(recs):
    terms = []
    for t in TERMS:
        v = np.array([r.get(t, np.nan) for r in recs], float)
        v = np.clip(v[np.isfinite(v)], -100, 100)
        terms.append(v.mean() if len(v) else np.nan)
    e, eb = sum(r["energy_MWh"] for r in recs), sum(r["energy_base_MWh"] for r in recs)
    loss = 100 * (1 - e / eb)
    return float(np.nanmean(terms) - 20 * max(0.0, loss - 1.0)), terms, loss


rng = np.random.default_rng(0)
rows = []


def compare(a_name, b_name, cp="cp1"):
    A, B = episodes(files(a_name, cp)), episodes(files(b_name, cp))
    if not A or not B:
        print(f"  {a_name} vs {b_name} [{cp}]: missing files")
        return
    keys = sorted(set(A) & set(B))
    strata = {}
    for k in keys:
        strata.setdefault(A[k]["mean_wind"], []).append(k)
    ja, jb = objective([A[k] for k in keys])[0], objective([B[k] for k in keys])[0]
    diffs = np.empty(a.n_boot)
    for i in range(a.n_boot):
        pick = [k for ks in strata.values() for k in rng.choice(ks, size=len(ks), replace=True)]
        diffs[i] = objective([A[k] for k in pick])[0] - objective([B[k] for k in pick])[0]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_gt = float((diffs > 0).mean())
    print(f"  {a_name:>26} - {b_name:<24} [{cp:>6}] n={len(keys)}: {ja:6.2f} - {jb:6.2f} = {ja - jb:+5.2f}   "
          f"95 % CI [{lo:+.2f}, {hi:+.2f}]   P(A > B) = {p_gt:.3f}")
    rows.append({"A": a_name, "B": b_name, "model": cp, "n_episodes": len(keys), "J_A": round(ja, 3), "J_B": round(jb, 3),
                 "diff": round(ja - jb, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "p_A_gt_B": round(p_gt, 4)})


print("paired bootstrap over the 24 held-out 600 s episodes (stratified by mean wind, 20000 resamples)")
for cp in ("cp1", "cp0.95"):
    compare("knee2 (2.9x GSPI travel)", "offset-free reference", cp)
compare("knee3 (4.0x)", "offset-free reference")
compare("knee1 (2.4x)", "offset-free reference")
compare("knee2 (2.9x GSPI travel)", "J-selected point")
compare("J-selected point", "offset-free reference")
compare("offset-free reference", "nominal MPC")
compare("offset-free reference", "nominal MPC", "cp0.95")
compare("adaptive (RLS) reference", "offset-free reference")
compare("offset-free + 3P notch", "offset-free reference")
compare("ROSCO + tower damper", "nominal MPC")
compare("nominal MPC", "tuned ROSCO")
compare("offset-free reference", "tuned ROSCO")
compare("J-selected point", "tuned ROSCO")
compare("knee2 (2.9x GSPI travel)", "tuned ROSCO")

print("\nper controller, both held-out sets (exact model): J, actuator duty, peak generator speed, energy change")
print(f"{'controller':>28} {'J':>6} {'duty':>6} {'xGSPI':>6} {'peak speed':>10} {'energy':>7}   per mean wind: P / w / T / B")
for name in ("nominal MPC", "offset-free reference", "J-selected point", "knee3 (4.0x)", "knee2 (2.9x GSPI travel)",
             "knee1 (2.4x)", "ROSCO + tower damper", "tuned ROSCO"):
    E = episodes(files(name))
    if not E:
        continue
    recs = list(E.values())
    J, _, loss = objective(recs)
    trav = np.mean([r["pitch_travel_deg"] for r in recs]); base = np.mean([r["pitch_travel_base_deg"] for r in recs])
    peak = max(r.get("gen_speed_max_rel", np.nan) for r in recs) if any("gen_speed_max_rel" in r for r in recs) else np.nan
    by = {}
    for r in recs:
        by.setdefault(r["mean_wind"], []).append(r)
    parts = []
    for u in sorted(by):
        t = objective(by[u])[1]
        parts.append(f"U{u:g}: " + "/".join("-" if not np.isfinite(x) else f"{x:.0f}" for x in t))
    print(f"{name:>28} {J:6.2f} {trav / 580:6.3f} {trav / base:6.2f} {peak:10.3f} {-loss:+6.2f}%   " + "   ".join(parts))

if a.csv and rows:
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("->", a.csv)
