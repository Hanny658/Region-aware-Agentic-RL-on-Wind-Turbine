"""Re-rank every tuning the search has evaluated under a per-wind load constraint (2026-10-12).

Stage 2 produced a front rather than a winner: the tuning with the largest regulation reduction holds the
tower-base load on the set mean, inside the resolution of that measurement, while losing several points of it at
one wind speed; the tuning with the largest tower-base reduction gives up 33 points of regulation. Roadmap s32
already measured why that happens -- a set-mean constraint is gamed across wind speeds -- so the baseline that
goes into the manuscript has to hold its loads at every wind speed, which is the rule `per_wind_rank_key`
applies.

This costs no simulations. Every history record carries the per-wind terms of its own candidate, so every
tuning the campaign has ever evaluated can be re-ranked from the files on disk; only verifying what comes out
on winds it was not designed on costs anything.

What it prints: how many of the evaluated tunings survive the per-wind screen, the best of them by regulation,
what the screen costs in regulation against the set-mean rule, and for each run whether the screen changes what
that run would have selected.

    ~/wtrl/run.sh python scripts/dev/rosco_perwind_select.py [--tol 1.0] [--top 12] [--csv <path>]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from eval.selection import PER_WIND_TOL, per_wind_rank_key, rank_key, worst_per_wind  # noqa: E402
from scripts.rosco_native import fmt, key_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--roots", nargs="+",
                default=[os.path.expanduser("~/wtrl/exp/roscotune_native"),
                         os.path.expanduser("~/wtrl/exp/roscosearch_native")])
ap.add_argument("--tol", type=float, default=PER_WIND_TOL,
                help="points of load reduction allowed to be negative at a single wind speed")
ap.add_argument("--top", type=int, default=12)
ap.add_argument("--csv", default=None)
a = ap.parse_args()

runs: dict[str, list] = {}
for root in a.roots:
    for h in sorted(glob.glob(f"{root}/*/history.jsonl")):
        name = os.path.basename(os.path.dirname(h))
        if "_r" not in name:
            continue                                   # _superseded and _verify are not runs
        runs[name] = [json.loads(line) for line in open(h)]

if not runs:
    sys.exit(f"no histories under {' '.join(a.roots)}")

# one entry per distinct tuning, with every run that evaluated it
seen: dict[str, dict] = {}
for name, hist in runs.items():
    for rec in hist:
        k = key_of(rec["params"])
        e = seen.setdefault(k, {"rec": rec, "runs": []})
        e["runs"].append(f"{name}#{rec['i']}")

cands = [e["rec"] | {"from": ", ".join(e["runs"])} for e in seen.values()]
usable = [r for r in cands if r["assess"].get("constraints")]
set_ok = [r for r in usable if r["assess"]["feasible"]]
pw_ok = [r for r in set_ok if per_wind_rank_key(r, a.tol)[0] == 1]

print(f"{len(runs)} runs, {len(cands)} distinct tunings evaluated ({len(cands) - len(usable)} the plant could "
      f"not run), {len(set_ok)} hold the constraints on the set mean, {len(pw_ok)} also hold both loads at "
      f"every wind speed within {a.tol:.1f} point.")
print(f"The design set carries two episodes per wind speed, so per wind this is a screen, not a measurement: "
      f"what it selects has to be confirmed off the design winds.\n")


def row(r: dict) -> dict:
    asm, c = r["assess"], r["assess"]["constraints"]
    w, at, which = worst_per_wind(r)
    return {"regulation": round(asm["objective"], 2), "power": round(asm.get("power", float("nan")), 2),
            "tower": round(c["tower"]["value"], 2), "blade": round(c["blade"]["value"], 2),
            "energy": round(c["energy"]["value"], 3), "worst_per_wind": round(w, 1),
            "worst_at": at, "worst_which": which, "travel_ratio": r.get("travel_ratio", float("nan")),
            "adc": r.get("adc", float("nan")), "from": r["from"],
            **{k: v for k, v in r["params"].items()}}


hdr = (f"{'regulation':>10} {'power':>7} {'tower':>7} {'blade':>7} {'energy':>7} {'worst/wind':>11} {'at':>5} "
       f"{'which':>6} {'travel':>7} {'ADC':>7}  tuning")
print(f"The {min(a.top, len(pw_ok))} best tunings that hold both loads at every wind speed, by regulation:")
print(hdr)
rows_pw = sorted((row(r) for r in pw_ok), key=lambda d: -d["regulation"])
for d in rows_pw[:a.top]:
    print(f"{d['regulation']:>10.2f} {d['power']:>7.2f} {d['tower']:>7.2f} {d['blade']:>7.2f} "
          f"{d['energy']:>7.3f} {d['worst_per_wind']:>11.1f} {d['worst_at']:>5} {d['worst_which']:>6} "
          f"{d['travel_ratio']:>7.3f} {d['adc']:>7.4f}  "
          f"{fmt({k: d[k] for k in ('omega_pc', 'zeta_pc', 'lpf_corner', 'fa_ki', 'fa_hpf')})}")

best_set = max(set_ok, key=rank_key) if set_ok else None
best_pw = max(pw_ok, key=lambda r: per_wind_rank_key(r, a.tol)) if pw_ok else None
if best_set is not None and best_pw is not None:
    ws = worst_per_wind(best_set)
    print(f"\nWhat the per-wind requirement costs: the set-mean rule selects {best_set['assess']['objective']:+.2f} "
          f"points of regulation with {ws[0]:+.1f} at {ws[1]} m/s on {ws[2]}; the per-wind rule selects "
          f"{best_pw['assess']['objective']:+.2f} with "
          f"{worst_per_wind(best_pw)[0]:+.1f} as its worst single wind speed. "
          f"Difference {best_set['assess']['objective'] - best_pw['assess']['objective']:+.2f} points.")
    print(f"  set mean rule: {fmt(best_set['params'])}   ({best_set['from']})")
    print(f"  per wind rule: {fmt(best_pw['params'])}   ({best_pw['from']})")

print(f"\nPer run, what each rule would select from that run alone:")
print(f"{'run':>12} {'set-mean rule':>14} {'worst/wind':>11}   {'per-wind rule':>14} {'worst/wind':>11}  changed")
for name in sorted(runs):
    hist = runs[name]
    bs, bp = max(hist, key=rank_key), max(hist, key=lambda r: per_wind_rank_key(r, a.tol))
    print(f"{name:>12} {bs['assess']['objective']:>14.2f} {worst_per_wind(bs)[0]:>11.1f}   "
          f"{bp['assess']['objective']:>14.2f} {worst_per_wind(bp)[0]:>11.1f}  "
          f"{'yes' if bs['i'] != bp['i'] else '-'}")

if a.csv and rows_pw:
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_pw[0]))
        w.writeheader()
        w.writerows(rows_pw)
    print(f"\n-> {a.csv}")
