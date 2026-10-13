"""Who writes the reward, under the gate, on the baseline that is a real tuning file (2026-10-13, roadmap s58).

s49 settled the proposer question on the UNGATED tuned-PI base under a rule stated there and implemented in
scripts/dev/proposer_test.py: a run HOLDS when both fatigue loads stay within one point of ITS OWN BASE at every
mean wind speed of the 600 s set. This table applies that rule unchanged to the gated layer on the native tuned
baseline - the agent arm of s56 (tnrwCwG3t) against its matched random-structure control (tnrrCwG3t,
campaign_gate_native_rand.sh), three seeds each, paired seed by seed.

Two cautions about the measurement, both of which this script exists to remove. The per-wind number s56 reported
(violation 0.00 %) is the evaluator's own per-wind Cw against the SHIPPED controller, which is a weaker test than
the s49 rule; and the four published reductions are against the shipped controller, so a difference between two
rows of this table is a difference of differences.

Aggregation follows the one rule of the project: bin mean first, then the difference between bins, then the
worst bin. The level is reported beside the rule because s49 found the two searched arms identical in level, so
a level difference under the gate would be a new finding.

    ~/wtrl/run.sh python scripts/dev/gate_proposer_control.py [--csv docs/tables/gate_proposer_control.csv]
"""
from __future__ import annotations

import argparse
import csv as csvmod
import json
import os

import numpy as np
from scipy.stats import fisher_exact

EXP = os.path.expanduser("~/wtrl/exp")
TERMS = (("power MSE", "power_mse_red_pct"), ("speed MSE", "gen_speed_mse_red_pct"),
         ("tower DEL", "TwrBsMyt_DEL_red_pct"), ("blade DEL", "RootMyc1_DEL_red_pct"))
LOADS = ("TwrBsMyt_DEL_red_pct", "RootMyc1_DEL_red_pct")
SCORED_S, RATE_LIMIT = 580.0, 10.0

ap = argparse.ArgumentParser()
ap.add_argument("--n_boot", type=int, default=20000)
ap.add_argument("--tol", type=float, default=-1.0, help="per-wind load tolerance in points, as in s49")
ap.add_argument("--csv", default="docs/tables/gate_proposer_control.csv")
a = ap.parse_args()

# (label, arm, path); the base of every learned row is the first entry, and rows are keyed by path, never by
# label, because the two arms carry the same three seed numbers (the defect of s56.4)
SETS = {}
for seeds in ("s78910", "s3456"):
    base = f"{EXP}/mpcsearch600/fresh/eval_rosco_tuned_native_{seeds}.json"
    rows = [("native tuned baseline (s52)", "base", base)]
    for s in (0, 1, 2):
        rows.append((f"agent-written reward, seed {s}", "agent",
                     f"{EXP}/tnrwCwG3t_s{s}/eval_range_TIB_{seeds}.json"))
    for s in (0, 1, 2):
        rows.append((f"random reward structure, seed {s}", "random",
                     f"{EXP}/tnrrCwG3t_s{s}/eval_range_TIB_{seeds}.json"))
    SETS[seeds] = ("fresh seeds 7-10" if seeds == "s78910" else "held-out seeds 3-6", base, rows)


def load(path: str) -> dict | None:
    """Per-episode records of one evaluation, keyed by the wind file its CSV records beside it."""
    if not os.path.exists(path) or not os.path.exists(path[:-5] + ".csv"):
        return None
    pe = json.load(open(path))["per_episode"]
    files = [r["wind_file"] for r in csvmod.DictReader(open(path[:-5] + ".csv"))]
    return {f: r for f, r in zip(files, pe)} if len(files) == len(pe) else None


out_rows, counts = [], {}
for seeds, (what, base, rows) in SETS.items():
    data, missing = {}, []
    for name, arm, path in rows:
        d = load(path)
        if d:
            data[path] = d
        else:
            missing.append((name, path))
    if base not in data:
        print(f"{seeds}: the base row is missing, nothing to compare")
        continue
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    u = np.array([float(f.split("_")[0][1:]) for f in common])
    winds = np.unique(u)
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(b, size=(a.n_boot, len(b)), replace=True)
                          for b in (np.where(u == s)[0] for s in winds)], axis=1)

    def col(path, key):
        return np.array([data[path][f].get(key, np.nan) for f in common], float)

    def ci(v):
        d = v[idx].mean(axis=1)
        return (float(np.mean(v)), *(float(x) for x in np.percentile(d, [2.5, 97.5])))

    print()
    print(f"=== {what}, {len(common)} episodes, every row the same episodes")
    for n, p in missing:
        print(f"  not available: {n}  ({p})")
    print("reductions against the paired SHIPPED controller; the two rightmost columns are the s49 rule - the")
    print("worst single-wind load term against this run's own base, where -1.00 is the tolerance")
    print(f"{'row':>32} " + " ".join(f"{t:>20}" for t, _ in TERMS) +
          f" {'travel':>7} {'ADC':>6} {'worst twr':>10} {'worst bld':>10}  holds")
    for name, arm, path in rows:
        if path not in data:
            continue
        cells = [f"{m:+6.1f} [{lo:+5.1f},{hi:+5.1f}]" for m, lo, hi in (ci(col(path, k)) for _, k in TERMS)]
        tr = col(path, "pitch_travel_deg").sum() / col(path, "pitch_travel_base_deg").sum()
        adc = float(np.mean(col(path, "pitch_travel_deg") / SCORED_S / RATE_LIMIT))
        worst = [0.0, 0.0]
        if arm != "base":
            for i, k in enumerate(LOADS):
                dv = [float(np.mean(col(path, k)[u == s]) - np.mean(col(base, k)[u == s])) for s in winds]
                worst[i] = min(0.0, min(dv))
        holds = arm != "base" and all(w >= a.tol for w in worst)
        flag = "-" if arm == "base" else ("yes" if holds else "NO")
        print(f"{name:>32} " + " ".join(f"{c:>20}" for c in cells) +
              f" {tr:6.3f}x {adc:6.4f} {worst[0]:+10.2f} {worst[1]:+10.2f}  {flag:>5}")
        if arm == "base":
            continue
        counts.setdefault(seeds, {}).setdefault(arm, []).append(holds)
        rec = {"seeds": seeds, "arm": arm, "run": name, "n_episodes": len(common),
               "travel_x_shipped": round(float(tr), 3), "adc": round(adc, 4),
               "worst_perwind_tower_vs_base": round(worst[0], 2),
               "worst_perwind_blade_vs_base": round(worst[1], 2), "holds_s49_rule": bool(holds)}
        for t, k in TERMS:
            m, lo, hi = ci(col(path, k))
            rec[t.replace(" ", "_")] = round(m, 2)
            m, lo, hi = ci(col(path, k) - col(base, k))
            rec["d_vs_base_" + t.replace(" ", "_")] = round(m, 2)
            rec["d_vs_base_" + t.replace(" ", "_") + "_lo"] = round(lo, 2)
            rec["d_vs_base_" + t.replace(" ", "_") + "_hi"] = round(hi, 2)
        out_rows.append(rec)

    print()
    print(f"  what the layer adds, paired against the base on the same {len(common)} episodes:")
    for name, arm, path in rows:
        if arm == "base" or path not in data:
            continue
        cells = []
        for t, k in TERMS:
            m, lo, hi = ci(col(path, k) - col(base, k))
            cells.append(f"{t} {m:+5.2f} [{lo:+5.2f},{hi:+5.2f}]" + ("" if lo > 0 or hi < 0 else "*"))
        trb = col(path, "pitch_travel_deg").sum() / col(base, "pitch_travel_deg").sum()
        print(f"    {name}: " + "  ".join(cells) + f"   travel {trb:.3f}x base")
    print("  (* the interval spans zero: the difference is not resolved)")

    print()
    print("  per wind speed, both load terms against the base - the rule is applied to the worst cell of this")
    print("  block, not to its mean, because a set mean is gamed (s32, s53.2, s55)")
    for k, lab in zip(LOADS, ("tower DEL", "blade DEL")):
        print(f"  {lab}: " + " ".join(f"{s:>7.0f}" for s in winds))
        for name, arm, path in rows:
            if arm == "base" or path not in data:
                continue
            dv = [float(np.mean(col(path, k)[u == s]) - np.mean(col(base, k)[u == s])) for s in winds]
            print(f"    {name:>30} " + " ".join(f"{x:+7.2f}" for x in dv))

for seeds, arms in counts.items():
    ag, rd = arms.get("agent", []), arms.get("random", [])
    if not ag or not rd:
        print()
        print(f"{seeds}: only one arm is on disk, the test waits for the other")
        continue
    ha, hr = sum(ag), sum(rd)
    t = [[ha, len(ag) - ha], [hr, len(rd) - hr]]
    two = fisher_exact(t)[1]
    one = fisher_exact(t, alternative="greater")[1]
    print()
    print(f"{seeds}: per-wind load rule under the gate, agent {ha}/{len(ag)} against the random structure "
          f"{hr}/{len(rd)}   two-sided p = {two:.4f}   one-sided p = {one:.4f}")
    print("        (three against three cannot establish the agent: the best possible outcome is p = 0.1. It can")
    print("         refute the extension of s49 to the gated configuration, which is what it was run for.)")

if a.csv and out_rows:
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=sorted({k for r in out_rows for k in r}))
        w.writeheader()
        w.writerows(out_rows)
    print()
    print(f"-> {a.csv}")
