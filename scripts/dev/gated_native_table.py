"""The wind-gated learned layer on the NATIVE tuned baseline, in the four published reductions (2026-10-13).

Roadmap s55 fixed the baseline row as the stage-1 native ROSCO tuning, which left the manuscript's PI-base
learned rung (s34 / s37 / s41) sitting on the superseded hand-scaled baseline: a gain over a controller that is
no longer the baseline, and not re-readable from the old tables because the residual was trained against the old
base's own rollouts. campaign_gate_native.sh retrained it (arm tnrwCwG3t, three agent seeds, gate 15-17 m/s).

This table reads the finished evaluations. Every row is an existing file, so it costs no simulation. Each run is
reported twice: the four reductions against the paired SHIPPED controller, which is what the baseline paper
publishes, and the paired difference against its OWN base on the same episodes, which is what the layer adds.
Rows are joined on the wind file so every comparison is paired episode by episode, and the intervals are
stratified bootstraps over episodes with the mean wind speed as the stratum and shared resample indices.

    ~/wtrl/run.sh python scripts/dev/gated_native_table.py [--csv docs/tables/gated_native.csv]
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
SCORED_S = 580.0   # 600 s simulated, 20 s discarded
RATE_LIMIT = 10.0  # deg/s, the pitch actuator limit the ADC is taken against

ap = argparse.ArgumentParser()
ap.add_argument("--n_boot", type=int, default=20000)
ap.add_argument("--csv", default="docs/tables/gated_native.csv")
a = ap.parse_args()

# (label, path, the label of the base this row is paired against) - rows are keyed by path, not by label,
# because two bases carry the same three seed labels and a label collision would silently compare the wrong pair
SETS = {
    "s78910": ("fresh seeds 7-10", [
        ("native tuned baseline (s52)", f"{EXP}/mpcsearch600/fresh/eval_rosco_tuned_native_s78910.json", None),
        ("native + gated layer, seed 0", f"{EXP}/tnrwCwG3t_s0/eval_range_TIB_s78910.json", 0),
        ("native + gated layer, seed 1", f"{EXP}/tnrwCwG3t_s1/eval_range_TIB_s78910.json", 0),
        ("native + gated layer, seed 2", f"{EXP}/tnrwCwG3t_s2/eval_range_TIB_s78910.json", 0),
        ("superseded tuned baseline", f"{EXP}/mpcsearch600/fresh/eval_rosco_tuned_s78910.json", None),
        ("superseded + gated layer, seed 0", f"{EXP}/trwCwG3t_s0/eval_range_TIB_s78910.json", 4),
        ("superseded + gated layer, seed 1", f"{EXP}/trwCwG3t_s1/eval_range_TIB_s78910.json", 4),
        ("superseded + gated layer, seed 2", f"{EXP}/trwCwG3t_s2/eval_range_TIB_s78910.json", 4),
    ]),
    "s3456": ("held-out seeds 3-6", [
        ("native tuned baseline (s52)", f"{EXP}/mpcsearch600/fresh/eval_rosco_tuned_native_s3456.json", None),
        ("native + gated layer, seed 0", f"{EXP}/tnrwCwG3t_s0/eval_range_TIB_s3456.json", 0),
        ("native + gated layer, seed 1", f"{EXP}/tnrwCwG3t_s1/eval_range_TIB_s3456.json", 0),
        ("native + gated layer, seed 2", f"{EXP}/tnrwCwG3t_s2/eval_range_TIB_s3456.json", 0),
        # seeds 1-6 in one file; the wind-file join keeps the 3-6 episodes
        ("superseded tuned baseline", f"{EXP}/mpcsearch600/range/eval_rosco_tuned.json", None),
        ("superseded + gated layer, seed 0", f"{EXP}/trwCwG3t_s0/eval_range_TIB_s3456.json", 4),
        ("superseded + gated layer, seed 1", f"{EXP}/trwCwG3t_s1/eval_range_TIB_s3456.json", 4),
        ("superseded + gated layer, seed 2", f"{EXP}/trwCwG3t_s2/eval_range_TIB_s3456.json", 4),
    ]),
}


def load(path: str) -> dict | None:
    """Per-episode records of one evaluation, keyed by the wind file its CSV records beside it."""
    if not os.path.exists(path) or not os.path.exists(path[:-5] + ".csv"):
        return None
    pe = json.load(open(path))["per_episode"]
    files = [r["wind_file"] for r in csvmod.DictReader(open(path[:-5] + ".csv"))]
    return {f: r for f, r in zip(files, pe)} if len(files) == len(pe) else None


out_rows = []
for seeds, (what, rows) in SETS.items():
    data, missing = {}, []
    for name, path, _ in rows:
        d = load(path)
        if d:
            data[path] = d
        else:
            missing.append((name, path))
    if not data:
        print(f"{seeds}: nothing on disk")
        continue
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    u = np.array([float(f.split("_")[0][1:]) for f in common])
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(b, size=(a.n_boot, len(b)), replace=True)
                          for b in (np.where(u == s)[0] for s in np.unique(u))], axis=1)

    def col(path, key):
        return np.array([data[path][f].get(key, np.nan) for f in common], float)

    def ci(v):
        d = v[idx].mean(axis=1)
        return (float(np.mean(v)), *(float(x) for x in np.percentile(d, [2.5, 97.5])))

    print(f"\n=== {what}, {len(common)} episodes, every row the same episodes; reductions against the paired "
          f"SHIPPED controller")
    for n, p in missing:
        print(f"  not available: {n}  ({p})")
    print(f"{'row':>33} " + " ".join(f"{t:>20}" for t, _ in TERMS) + f" {'energy':>7} {'travel':>7} {'ADC':>6}")
    for name, path, _ in rows:
        if path not in data:
            continue
        cells = [f"{m:+6.1f} [{lo:+5.1f},{hi:+5.1f}]" for m, lo, hi in (ci(col(path, k)) for _, k in TERMS)]
        tr = col(path, "pitch_travel_deg").sum() / col(path, "pitch_travel_base_deg").sum()
        en = 100.0 * (col(path, "energy_MWh").sum() / col(path, "energy_base_MWh").sum() - 1.0)
        adc = np.mean(col(path, "pitch_travel_deg") / SCORED_S / RATE_LIMIT)
        print(f"{name:>33} " + " ".join(f"{c:>20}" for c in cells) +
              f" {en:+7.3f} {tr:7.3f}x {adc:6.4f}")

    print(f"\n  what the layer adds, paired against its own base on the same {len(common)} episodes:")
    for name, path, bi in rows:
        if bi is None or path not in data:
            continue
        bname, bpath, _ = rows[bi]
        if bpath not in data:
            continue
        cells = []
        for t, k in TERMS:
            m, lo, hi = ci(col(path, k) - col(bpath, k))
            cells.append(f"{t} {m:+5.2f} [{lo:+5.2f},{hi:+5.2f}]{'' if lo > 0 or hi < 0 else '*'}")
        trb = col(path, "pitch_travel_deg").sum() / col(bpath, "pitch_travel_deg").sum()
        print(f"    {name}: " + "  ".join(cells) + f"   travel {trb:.3f}x base")
        row = {"seeds": seeds, "base": bname, "run": name, "n_episodes": len(common),
               "travel_x_base": round(float(trb), 3)}
        for t, k in TERMS:
            m, lo, hi = ci(col(path, k))
            row[t.replace(" ", "_")] = round(m, 2)
            m, lo, hi = ci(col(path, k) - col(bpath, k))
            row["d_" + t.replace(" ", "_")] = round(m, 2)
            row["d_" + t.replace(" ", "_") + "_lo"] = round(lo, 2)
            row["d_" + t.replace(" ", "_") + "_hi"] = round(hi, 2)
        out_rows.append(row)
    print("  (* the interval spans zero: the difference is not resolved)")

if a.csv and out_rows:
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n-> {a.csv}")
