"""Leg 2 of the main line: what the agentic design search is worth on an untuned MPC (2026-10-13, no simulations).

Roadmap s38 established that the llm and es proposers both discover the wind-scheduled tower weight in 24
simulations while random search finds nothing in 48, but it reported the discovery in composite points. This
table reports the same thing in the four published reductions, which is where it becomes legible: the schedule
*costs* regulation and *buys* tower-base fatigue, and the set means understate it - the whole repair is at 22
and 24 m/s, where the unscheduled controller is negative and the scheduled one is not.

Both rows are evaluations already on disk. They are joined on the wind file, so the comparison is paired
episode by episode and a file holding more seeds than asked for (the range evaluation carries seeds 1-6) is
subset by the join rather than by a slice. Intervals are stratified bootstraps over episodes with the mean wind
speed as the stratum and one resample index matrix shared by every row.

    ~/wtrl/run.sh python scripts/dev/mpc_leg_table.py [--seeds s78910] [--csv docs/tables/mpc_leg_fresh.csv]
"""
from __future__ import annotations

import argparse
import csv as csvmod
import json
import os

import numpy as np

EXP = os.path.expanduser("~/wtrl/exp")
TERMS = (("power MSE", "power_mse_red_pct"), ("speed MSE", "gen_speed_mse_red_pct"),
         ("tower DEL", "TwrBsMyt_DEL_red_pct"), ("blade DEL", "RootMyc1_DEL_red_pct"))
SCORED_S, RATE_LIMIT = 580.0, 10.0

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="s78910", choices=["s78910", "s3456"])
ap.add_argument("--n_boot", type=int, default=20000)
ap.add_argument("--csv", default=None)
a = ap.parse_args()

ROWS = {
    "s78910": [
        ("tuned baseline, native (s52)", f"{EXP}/mpcsearch600/fresh/eval_rosco_tuned_native_s78910.json"),
        ("offset-free LPV-MPC, UNSCHEDULED", f"{EXP}/mpcsearch600/fresh/eval_unsched_mpc_s78910.json"),
        ("offset-free LPV-MPC, agent-found schedule", f"{EXP}/mpcsearch600/fresh/eval_sched_mpc_s78910.json"),
    ],
    "s3456": [
        ("tuned baseline, native (s52)", f"{EXP}/mpcsearch600/fresh/eval_rosco_tuned_native_s3456.json"),
        # seeds 1-6 in one file; the wind-file join keeps the 3-6 episodes
        ("offset-free LPV-MPC, UNSCHEDULED", f"{EXP}/mpcsearch600/range/eval_offset_cp1.json"),
        ("offset-free LPV-MPC, agent-found schedule", f"{EXP}/mpcsearch600/qtsched/eval_off_q6_18_22_s3456.json"),
    ],
}[a.seeds]
UNSCHED, SCHED = ROWS[1][0], ROWS[2][0]


def load(path: str) -> dict | None:
    """Per-episode records of one evaluation, keyed by the wind file its CSV records beside it."""
    if not os.path.exists(path) or not os.path.exists(path[:-5] + ".csv"):
        return None
    pe = json.load(open(path))["per_episode"]
    files = [r["wind_file"] for r in csvmod.DictReader(open(path[:-5] + ".csv"))]
    return {f: r for f, r in zip(files, pe)} if len(files) == len(pe) else None


data, missing = {}, []
for name, path in ROWS:
    d = load(path)
    data[name] = d if d else missing.append((name, path))
data = {k: v for k, v in data.items() if v}
for n, p in missing:
    print(f"not available: {n}  ({p})")
if SCHED not in data or UNSCHED not in data:
    raise SystemExit("leg 2 needs both MPC rows")

common = sorted(set.intersection(*(set(d) for d in data.values())))
u = np.array([float(f.split("_")[0][1:]) for f in common])
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(b, size=(a.n_boot, len(b)), replace=True)
                      for b in (np.where(u == s)[0] for s in np.unique(u))], axis=1)


def col(name: str, key: str) -> np.ndarray:
    return np.array([data[name][f].get(key, np.nan) for f in common], float)


def ci(v: np.ndarray) -> tuple[float, float, float]:
    d = v[idx].mean(axis=1)
    return (float(np.mean(v)), *(float(x) for x in np.percentile(d, [2.5, 97.5])))


print(f"\n{len(common)} episodes of the {a.seeds[1:]} wind fields, every row the same episodes, every reduction "
      f"against the paired shipped controller on the same wind file.\n")
print(f"{'row':>42} " + " ".join(f"{t:>20}" for t, _ in TERMS) + f" {'energy':>7} {'travel':>7} {'ADC':>6}")
for name, _ in ROWS:
    if name not in data:
        continue
    cells = [f"{m:+6.1f} [{lo:+5.1f},{hi:+5.1f}]" for m, lo, hi in (ci(col(name, k)) for _, k in TERMS)]
    tr = col(name, "pitch_travel_deg").sum() / col(name, "pitch_travel_base_deg").sum()
    en = 100.0 * (col(name, "energy_MWh").sum() / col(name, "energy_base_MWh").sum() - 1.0)
    adc = np.mean(col(name, "pitch_travel_deg") / SCORED_S / RATE_LIMIT)
    print(f"{name:>42} " + " ".join(f"{c:>20}" for c in cells) + f" {en:+7.3f} {tr:7.3f}x {adc:6.4f}")

print(f"\nwhat the agent-found schedule changes, paired on the same {len(common)} episodes:")
for t, k in TERMS:
    m, lo, hi = ci(col(SCHED, k) - col(UNSCHED, k))
    print(f"   {t:>10} {m:+7.2f} [{lo:+6.2f}, {hi:+6.2f}]{'' if lo > 0 or hi < 0 else '   not resolved'}")

print("\nper wind speed, tower-base DEL reduction vs the shipped controller - the schedule was found to repair\n"
      "the high-wind end, and that is invisible in the set mean:")
print(f"{'U':>4} " + " ".join(f"{n[:30]:>30}" for n, _ in ROWS if n in data))
for s in np.unique(u):
    m = u == s
    print(f"{s:4.0f} " + " ".join(f"{np.mean(col(n, 'TwrBsMyt_DEL_red_pct')[m]):+30.2f}"
                                 for n, _ in ROWS if n in data))

if a.csv:
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    out = []
    for name, _ in ROWS:
        if name not in data:
            continue
        row = {"row": name, "seeds": a.seeds, "n_episodes": len(common),
               "travel_x_shipped": round(float(col(name, "pitch_travel_deg").sum()
                                              / col(name, "pitch_travel_base_deg").sum()), 3),
               "adc": round(float(np.mean(col(name, "pitch_travel_deg") / SCORED_S / RATE_LIMIT)), 4)}
        for t, k in TERMS:
            m, lo, hi = ci(col(name, k))
            row[t.replace(" ", "_")] = round(m, 2)
            row[t.replace(" ", "_") + "_lo"], row[t.replace(" ", "_") + "_hi"] = round(lo, 2), round(hi, 2)
            if name == SCHED:
                m, lo, hi = ci(col(SCHED, k) - col(UNSCHED, k))
                row["d_vs_unscheduled_" + t.replace(" ", "_")] = round(m, 2)
                row["d_vs_unscheduled_" + t.replace(" ", "_") + "_lo"] = round(lo, 2)
                row["d_vs_unscheduled_" + t.replace(" ", "_") + "_hi"] = round(hi, 2)
        for s in np.unique(u):
            row[f"tower_DEL_U{s:g}"] = round(float(np.mean(col(name, "TwrBsMyt_DEL_red_pct")[u == s])), 2)
        out.append(row)
    with open(a.csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=sorted({k for r in out for k in r}))
        w.writeheader()
        w.writerows(out)
    print(f"\n-> {a.csv}")
