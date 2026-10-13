"""The ladder in the four published reductions, no composite anywhere (2026-10-12, no simulations).

The ladder of the manuscript was written in points of a composite that averaged four percentage reductions, so
every rung carried one number and the arithmetic "32.5 = 15.9 + 14.5 + 2.1" was arithmetic on that composite.
The composite is gone from selection (eval/selection.py) and it has to go from reporting: what the baseline
paper publishes, and what this table reports, are the four reductions against the paired shipped controller,
each one on its own.

Every rung is an evaluation that already exists on disk, so this costs nothing. The rows are joined on the wind
file, which the per-episode CSV of every evaluation carries, so all rows are the same episodes of the same wind
fields and a difference between two rungs is paired episode by episode. Intervals are stratified bootstraps over
episodes with the strata the mean wind speed and the same resample indices for every row.

    ~/wtrl/run.sh python scripts/dev/ladder_published.py [--seeds s78910] [--csv docs/tables/ladder_published.csv]
"""
from __future__ import annotations

import argparse
import csv as csvmod
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.rosco_native import key_of  # noqa: E402

EXP = os.path.expanduser("~/wtrl/exp")
TERMS = (("power MSE", "power_mse_red_pct"), ("speed MSE", "gen_speed_mse_red_pct"),
         ("tower DEL", "TwrBsMyt_DEL_red_pct"), ("blade DEL", "RootMyc1_DEL_red_pct"))
# the stage-1 native tuning, roadmap s52: the standing baseline row
BASELINE = {"omega_pc": 0.604783, "zeta_pc": 0.737849, "lpf_corner": 6.52919,
            "fa_ki": 0.0331436, "fa_hpf": 0.692474}

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="s78910", choices=["s78910", "s3456"],
                help="s78910 = the fresh seeds the manuscript headlines, s3456 = the held-out set")
ap.add_argument("--gate_seed", default="s1", help="which agent seed of the gated layer to show")
ap.add_argument("--n_boot", type=int, default=20000)
ap.add_argument("--csv", default=None)
a = ap.parse_args()


def native_eval(params: dict, seeds: str) -> str | None:
    """The cached evaluation of a native ROSCO tuning on one seed set, wherever the campaign put it."""
    pat = f"eval_{key_of(params)}_{seeds}_*.json"
    for root in (f"{EXP}/roscosearch_native/_verify/cache", f"{EXP}/roscotune_native/*/cache",
                 f"{EXP}/roscosearch_native/*/cache"):
        hits = sorted(glob.glob(os.path.join(root, pat)))
        if hits:
            return hits[0]
    return None


ROWS = {
    "s78910": [
        ("tuned baseline, superseded (hand-scaled gains)", f"{EXP}/mpcsearch600/fresh/eval_rosco_tuned_s78910.json"),
        ("tuned baseline, native (s52)", native_eval(BASELINE, "s78910")),
        ("native baseline + gated layer", f"{EXP}/tnrwCwG3t_{a.gate_seed}/eval_range_TIB_s78910.json"),
        ("scheduled offset-free LPV-MPC", f"{EXP}/mpcsearch600/fresh/eval_sched_mpc_s78910.json"),
        ("MPC + wind-gated learned layer", f"{EXP}/mrwCwG3t_{a.gate_seed}/eval_range_TIB_s78910.json"),
        ("superseded baseline + gated layer", f"{EXP}/trwCwG3t_{a.gate_seed}/eval_range_TIB_s78910.json"),
    ],
    "s3456": [
        ("tuned baseline, native (s52)", native_eval(BASELINE, "s3456")),
        ("native baseline + gated layer", f"{EXP}/tnrwCwG3t_{a.gate_seed}/eval_range_TIB_s3456.json"),
        ("scheduled offset-free LPV-MPC", f"{EXP}/mpcsearch600/qtsched/eval_off_q6_18_22_s3456.json"),
        ("MPC + wind-gated learned layer", f"{EXP}/mrwCwG3t_{a.gate_seed}/eval_range_TIB_s3456.json"),
        ("superseded baseline + gated layer", f"{EXP}/trwCwG3t_{a.gate_seed}/eval_range_TIB_s3456.json"),
    ],
}[a.seeds]


def load(path: str | None) -> dict | None:
    """Per-episode records of one evaluation, keyed by wind file (from the CSV the evaluation writes beside it)."""
    if not path or not os.path.exists(path):
        return None
    pe = json.load(open(path))["per_episode"]
    q = path[:-5] + ".csv"
    if not os.path.exists(q):
        return None
    files = [r["wind_file"] for r in csvmod.DictReader(open(q))]
    if len(files) != len(pe):
        return None
    return {f: r for f, r in zip(files, pe)}


data, missing = {}, []
for name, path in ROWS:
    d = load(path)
    (data.setdefault(name, d) if d else missing.append((name, path or "no cached evaluation")))

common = sorted(set.intersection(*(set(d) for d in data.values()))) if data else []
if not common:
    sys.exit("no episodes in common" + (f"; missing: {missing}" if missing else ""))

u = np.array([float(f.split("_")[0][1:]) for f in common])
rng = np.random.default_rng(0)
blocks = [np.where(u == s)[0] for s in np.unique(u)]
idx = np.concatenate([rng.choice(b, size=(a.n_boot, len(b)), replace=True) for b in blocks], axis=1)


def col(name: str, key: str) -> np.ndarray:
    return np.array([data[name][f].get(key, np.nan) for f in common], float)


def ci(v: np.ndarray) -> tuple[float, float, float]:
    d = v[idx].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(v)), float(lo), float(hi)


def travel(name: str) -> float:
    return float(col(name, "pitch_travel_deg").sum() / col(name, "pitch_travel_base_deg").sum())


def energy(name: str) -> float:
    return float(100.0 * (col(name, "energy_MWh").sum() / col(name, "energy_base_MWh").sum() - 1.0))


names = [n for n, _ in ROWS if n in data]
print(f"{len(common)} episodes of the {a.seeds[1:]} wind fields, every row the same episodes, every reduction "
      f"against the paired shipped controller on the same wind file.")
if missing:
    for n, p in missing:
        print(f"  not available: {n}  ({p})")
print()
print(f"{'rung':>46} " + " ".join(f"{t:>20}" for t, _ in TERMS) + f" {'energy':>7} {'travel':>7}")
for n in names:
    cells = []
    for _, k in TERMS:
        m, lo, hi = ci(col(n, k))
        cells.append(f"{m:+6.1f} [{lo:+5.1f},{hi:+5.1f}]")
    print(f"{n:>46} " + " ".join(f"{c:>20}" for c in cells) + f" {energy(n):+7.3f} {travel(n):7.3f}x")

print("\nRung to rung, each term on its own, paired over the same episodes "
      "(a difference whose interval spans zero is not resolved):")
for lo_n, hi_n in zip(names, names[1:]):
    print(f"  {hi_n}  minus  {lo_n}")
    for t, k in TERMS:
        m, l, h = ci(col(hi_n, k) - col(lo_n, k))
        print(f"      {t:>10} {m:+7.2f} [{l:+6.2f}, {h:+6.2f}]{'' if l > 0 or h < 0 else '   not resolved'}")

if a.csv:
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    out = []
    for n in names:
        row = {"rung": n, "n_episodes": len(common), "seeds": a.seeds,
               "energy_change_pct": round(energy(n), 3), "travel_x_shipped": round(travel(n), 3)}
        for t, k in TERMS:
            m, lo, hi = ci(col(n, k))
            row[t.replace(" ", "_")] = round(m, 2)
            row[t.replace(" ", "_") + "_lo"], row[t.replace(" ", "_") + "_hi"] = round(lo, 2), round(hi, 2)
        out.append(row)
    with open(a.csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\n-> {a.csv}")
