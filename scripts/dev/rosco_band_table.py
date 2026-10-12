"""The verified baseline band, in the four published reductions (2026-10-12, no simulations).

`rosco_verify_band.py` measured the band off the design winds and reported the objective and the two load
constraints, which is what the selection rule needs. A baseline row in the manuscript needs all four published
reductions, the energy and the actuator duty, on the same episodes, so this reads the same cached evaluations
and reports them that way: the members that hold their loads at every wind speed off the design winds, plus the
stage-1 tuning that already held it, each against the paired shipped controller, and each against the stage-1
tuning episode by episode.

    ~/wtrl/run.sh python scripts/dev/rosco_band_table.py [--seeds s78910] [--csv docs/tables/rosco_band_rows.csv]
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
from scripts.rosco_native import fmt, key_of  # noqa: E402

EXP = os.path.expanduser("~/wtrl/exp")
TERMS = (("power MSE", "power_mse_red_pct"), ("speed MSE", "gen_speed_mse_red_pct"),
         ("tower DEL", "TwrBsMyt_DEL_red_pct"), ("blade DEL", "RootMyc1_DEL_red_pct"))
KEYS = ("omega_pc", "zeta_pc", "lpf_corner", "fa_ki", "fa_hpf")
STAGE1 = {"omega_pc": 0.604783, "zeta_pc": 0.737849, "lpf_corner": 6.52919,
          "fa_ki": 0.0331436, "fa_hpf": 0.692474}
RATE_LIMIT = 10.0           # deg/s, the limiter the controller runs with
SCORED_S = 580.0            # 600 s simulated, the first 20 s discarded

ap = argparse.ArgumentParser()
ap.add_argument("--verified", default="docs/tables/rosco_band_verified.csv")
ap.add_argument("--seeds", default="s78910", choices=["s78910", "s3456"])
ap.add_argument("--n_boot", type=int, default=20000)
ap.add_argument("--csv", default=None)
a = ap.parse_args()


def cached(params: dict, seeds: str) -> str | None:
    for root in (f"{EXP}/roscosearch_native/_verify/cache", f"{EXP}/roscotune_native/*/cache",
                 f"{EXP}/roscosearch_native/*/cache"):
        hits = sorted(glob.glob(os.path.join(root, f"eval_{key_of(params)}_{seeds}_*.json")))
        if hits:
            return hits[0]
    return None


def load(path: str | None) -> dict | None:
    """Per-episode records keyed by wind file, so rows can be paired episode by episode."""
    if not path or not os.path.exists(path) or not os.path.exists(path[:-5] + ".csv"):
        return None
    pe = json.load(open(path))["per_episode"]
    files = [r["wind_file"] for r in csvmod.DictReader(open(path[:-5] + ".csv"))]
    return {f: r for f, r in zip(files, pe)} if len(files) == len(pe) else None


rows = [("stage 1 (s52), damper on", STAGE1, float("nan"))]
for r in csvmod.DictReader(open(a.verified)):
    if r["holds_per_wind"] == "True":
        p = {k: float(r[k]) for k in KEYS}
        rows.append((r["from"], p, float(r["travel_ratio"])))

data, missing = {}, []
for name, p, _ in rows:
    d = load(cached(p, a.seeds))
    data[name] = d if d else missing.append((name, fmt(p)))
data = {k: v for k, v in data.items() if v}
common = sorted(set.intersection(*(set(d) for d in data.values())))
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


print(f"{len(common)} episodes of the {a.seeds[1:]} wind fields; every member holds both load terms at every "
      f"wind speed off the design winds. Reductions against the paired shipped controller.")
for n, p in missing:
    print(f"  not available: {n}  ({p})")
print()
print(f"{'member':>14} " + " ".join(f"{t:>20}" for t, _ in TERMS) + f" {'energy':>7} {'travel':>7} {'ADC':>7}")
out = []
for name, p, _ in rows:
    if name not in data:
        continue
    travel = float(col(name, "pitch_travel_deg").sum() / col(name, "pitch_travel_base_deg").sum())
    adc = float(np.mean(col(name, "pitch_travel_deg") / SCORED_S / RATE_LIMIT))
    energy = float(100.0 * (col(name, "energy_MWh").sum() / col(name, "energy_base_MWh").sum() - 1.0))
    cells, rec = [], {"member": name, "tuning": fmt(p), "seeds": a.seeds, "n_episodes": len(common),
                      "energy_change_pct": round(energy, 3), "travel_x_shipped": round(travel, 3),
                      "adc": round(adc, 4), **{k: p[k] for k in KEYS}}
    for t, k in TERMS:
        m, lo, hi = ci(col(name, k))
        cells.append(f"{m:+6.1f} [{lo:+5.1f},{hi:+5.1f}]")
        rec[t.replace(" ", "_")], rec[t.replace(" ", "_") + "_lo"], rec[t.replace(" ", "_") + "_hi"] = (
            round(m, 2), round(lo, 2), round(hi, 2))
    print(f"{name:>14} " + " ".join(f"{c:>20}" for c in cells) + f" {energy:+7.3f} {travel:7.3f}x {adc:7.4f}")
    out.append(rec)

ref = rows[0][0]
print(f"\nAgainst the stage-1 tuning, paired over the same episodes "
      f"(an interval spanning zero means the two are not distinguished):")
for name, p, _ in rows[1:]:
    if name not in data:
        continue
    print(f"  {name}  minus  {ref}   ({fmt(p)})")
    for t, k in TERMS:
        m, lo, hi = ci(col(name, k) - col(ref, k))
        print(f"      {t:>10} {m:+7.2f} [{lo:+6.2f}, {hi:+6.2f}]{'' if lo > 0 or hi < 0 else '   not resolved'}")

if a.csv and out:
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\n-> {a.csv}")
