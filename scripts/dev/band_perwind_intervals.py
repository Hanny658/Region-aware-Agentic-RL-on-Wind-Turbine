"""Per-wind load terms of the verified ladder with intervals (roadmap s63, review round 1; no simulation).

Table 2 of the manuscript prints each row's worst single-wind load term on the held-out and on the fresh set as a
point estimate over four realisations per wind speed. This script reads the same verification caches and gives,
for each row and each wind speed, the tower-base and blade-root DEL reductions as bin means over the eight
episodes of both sets pooled with a 95 % bootstrap interval over those episodes, and - the quantity the rule
turns on - the share of episode resamples (within each wind bin, both sets pooled) in which the worst single-wind
load term stays inside the published tolerance of -1 point and above zero.

    WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 ~/wtrl/run.sh python scripts/dev/band_perwind_intervals.py \
        [--csv docs/tables/band_perwind_intervals.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))
from scripts.rosco_native import START, canon, key_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--caches", nargs="+", default=[os.path.expanduser("~/wtrl/exp/roscosearch_native/_verify/cache"),
                                                os.path.expanduser("~/wtrl/exp/roscosearch_rule6/_verify/cache"),
                                                os.path.expanduser("~/wtrl/exp/roscosearch_margin/_verify/cache")])
ap.add_argument("--csv", default=str(PROJ / "docs" / "tables" / "band_perwind_intervals.csv"))
ap.add_argument("--n_boot", type=int, default=4000)
ap.add_argument("--tol", type=float, default=1.0)
a = ap.parse_args()

# the rows of Table 2, by their exact tuning as the band table and the tuned-baseline table carry it (the key every
# cache file carries is the hash of the canonical parameters, so the values must be the evaluated ones)
import csv as _csv
_band = {r["from"]: r for r in _csv.DictReader(open(PROJ / "docs" / "tables" / "rosco_band_verified.csv"))}
_tuned = next(_csv.DictReader(open(PROJ / "docs" / "tables" / "rosco_tuned_baseline.csv")))
def _p(r, pre=""):
    return {k: float(r[pre + k]) for k in ("omega_pc", "zeta_pc", "lpf_corner", "fa_ki", "fa_hpf")}
ROWS = [
    ("agent, one line (corner 3.0)", _p(_band["llm_r1#9"])),
    ("agent, the same line at 6.5", _p(_band["llm_r1#11"])),
    ("stage 1 (fixed recipe)", _p(_tuned, "best_")),
    ("local search, sub-shipped duty", _p(_band["local_r0#22"])),
]
TERMS = (("tower", "TwrBsMyt_DEL_red_pct"), ("blade", "RootMyc1_DEL_red_pct"))


def find(key, seeds):
    for c in a.caches + sorted(glob.glob(os.path.expanduser("~/wtrl/exp/*/_verify/cache")))             + sorted(glob.glob(os.path.expanduser("~/wtrl/exp/*/cache"))) + sorted(glob.glob(os.path.expanduser("~/wtrl/exp/*/*/cache"))):
        hits = glob.glob(f"{c}/eval_{key}_s{seeds}_*.json")
        if hits:
            return hits[0]
    return None


def cells(fit):
    """{wind: {term: [per-episode reductions]}}"""
    by = {}
    for e in fit.get("per_episode", []):
        if e.get("mean_wind") is None:
            continue
        d = by.setdefault(float(e["mean_wind"]), {})
        for nm, col in TERMS:
            v = e.get(col)
            if v is not None and v == v:
                d.setdefault(nm, []).append(float(v))
    return by


rng = np.random.default_rng(0)
out_rows = []
print(f"{'row':<32} {'wind':>5} {'tower (8 ep)':>22} {'blade (8 ep)':>22}")
for name, p in ROWS:
    k = key_of(canon(p))
    files = {s: find(k, s) for s in ("3456", "78910")}
    if not all(files.values()):
        # the stage-1 row may carry a different parameter rounding in its own template; try the band table's values
        print(f"! {name}: cache not found for key {k} ({files}); skipped")
        continue
    pooled = {}
    per_set = {}
    for s, f in files.items():
        c = cells(json.load(open(f)))
        per_set[s] = c
        for u, d in c.items():
            for nm, v in d.items():
                pooled.setdefault(u, {}).setdefault(nm, []).extend(v)
    winds = sorted(pooled)
    # bootstrap: resample episodes within each wind bin, recompute every cell, take the worst
    B = a.n_boot
    worst = np.empty(B)
    boots = {u: {nm: np.empty(B) for nm, _ in TERMS} for u in winds}
    for b in range(B):
        w = np.inf
        for u in winds:
            for nm, _ in TERMS:
                v = np.asarray(pooled[u][nm])
                m = float(v[rng.integers(0, len(v), len(v))].mean())
                boots[u][nm][b] = m
                w = min(w, m)
        worst[b] = w
    share_tol = float((worst >= -a.tol).mean())
    share_zero = float((worst >= 0.0).mean())
    w_point = min(float(np.mean(pooled[u][nm])) for u in winds for nm, _ in TERMS)
    w_ho = min(float(np.mean(v)) for d in per_set["3456"].values() for v in d.values())
    w_fr = min(float(np.mean(v)) for d in per_set["78910"].values() for v in d.values())
    for u in winds:
        rec = {"row": name, "wind": u}
        line = f"{name:<32} {u:>5g}"
        for nm, _ in TERMS:
            v = np.asarray(pooled[u][nm])
            lo, hi = np.percentile(boots[u][nm], [2.5, 97.5])
            rec[f"{nm}_mean"] = round(float(v.mean()), 2)
            rec[f"{nm}_lo"] = round(float(lo), 2)
            rec[f"{nm}_hi"] = round(float(hi), 2)
            rec[f"{nm}_heldout"] = round(float(np.mean(per_set["3456"][u][nm])), 2)
            rec[f"{nm}_fresh"] = round(float(np.mean(per_set["78910"][u][nm])), 2)
            line += f" {v.mean():+6.2f} [{lo:+5.1f},{hi:+5.1f}]"
        rec.update({"worst_pooled": round(w_point, 2), "worst_heldout": round(w_ho, 2), "worst_fresh": round(w_fr, 2),
                    "share_worst_within_tol": round(share_tol, 3), "share_worst_above_zero": round(share_zero, 3)})
        out_rows.append(rec)
        print(line)
    print(f"{'':<32} worst cell: pooled {w_point:+.2f}, held out {w_ho:+.2f}, fresh {w_fr:+.2f}; "
          f"resamples with the worst cell >= -{a.tol:g}: {share_tol:.1%}, >= 0: {share_zero:.1%}\n")

os.makedirs(os.path.dirname(a.csv), exist_ok=True)
with open(a.csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)
print(f"-> {a.csv}")
