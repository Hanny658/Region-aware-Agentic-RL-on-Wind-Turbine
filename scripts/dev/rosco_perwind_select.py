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

`--verify` is the part that does cost simulations, and it exists because stage 3 of the campaign does not cover
it: that stage re-scores each run's SET-MEAN selection off the design winds, so a run whose per-wind selection
is a different candidate is left unmeasured (s59.5: local r4 #10 against #21, local r5 #18 against #9). With
the flag, every run's per-wind selection is evaluated on held-out seeds 3-6 and fresh seeds 7-10, two
evaluations of 28 episodes each, into the same parameter-hash cache the rest of the campaign reads, so a
selection another stage has already measured costs nothing and `rosco_proposer_test.py` picks the new rows up.

    ~/wtrl/run.sh python scripts/dev/rosco_perwind_select.py [--tol 1.0] [--top 12] [--csv <path>]
    ~/wtrl/run.sh python scripts/dev/rosco_perwind_select.py --verify [--port0 8300] [--workers 7]
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
ap.add_argument("--verify", action="store_true",
                help="evaluate each run's per-wind selection off the design winds (seeds 3-6 and 7-10)")
ap.add_argument("--verify_root", default=os.path.expanduser("~/wtrl/exp/roscosearch_native"),
                help="the search root whose _verify cache the evaluations are written to and read from")
ap.add_argument("--workers", type=int, default=7)
ap.add_argument("--port0", type=int, default=8300)
ap.add_argument("--means", nargs="+", type=float, default=[12, 14, 16, 18, 20, 22, 24])
ap.add_argument("--ti", default="B")
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


# ----------------------------------------------- the per-wind selections off the design winds (costs simulations)
if a.verify:
    from eval.selection import assess  # noqa: E402
    from scripts.rosco_native import Evaluator  # noqa: E402

    class VArgs:
        root = a.verify_root + "/_verify"
        template = os.path.expanduser("~/wtrl/runs/template_5mw")
        config_run = os.path.expanduser("~/wtrl/exp/jg3t_s0")
        seeds = [3, 4, 5, 6]
        episode_s = 600.0
        workers = a.workers
        port0 = a.port0

    ev = Evaluator(VArgs())
    ev.extra_args = ["--means", *[f"{m:g}" for m in a.means], "--ti", a.ti]

    def worst_measured(fit: dict) -> float:
        """The worst load reduction at any single wind speed of an evaluation: bin mean first, then the worst bin."""
        by: dict[float, dict[str, list]] = {}
        for e in fit.get("per_episode", []):
            if e.get("mean_wind") is None:
                continue
            d = by.setdefault(e["mean_wind"], {})
            for nm, col in (("tower", "TwrBsMyt_DEL_red_pct"), ("blade", "RootMyc1_DEL_red_pct")):
                v = e.get(col)
                if v is not None and v == v:
                    d.setdefault(nm, []).append(float(v))
        vals = [sum(v) / len(v) for d in by.values() for v in d.values()]
        return min(vals) if vals else float("nan")

    print()
    print(f"Each run's per-wind selection measured off the design winds, tolerance {a.tol:.1f} point "
          f"at any single wind speed on BOTH seed sets:")
    print(f"{'run':>12} {'candidate':>10} {'design':>8} {'held out':>9} {'fresh':>8}   {'design w/w':>11} "
          f"{'heldout w/w':>12} {'fresh w/w':>10}  ships")
    for name in sorted(runs):
        hist = runs[name]
        bp = max(hist, key=lambda r: per_wind_rank_key(r, a.tol))
        if bp["i"] == 0 or per_wind_rank_key(bp, a.tol)[0] != 1:
            print(f"{name:>12} {'-':>10}   no candidate holds both loads at every wind speed on the design set")
            continue
        ho, fr = ev.run(bp["params"], 0, seeds=[3, 4, 5, 6]), ev.run(bp["params"], 1, seeds=[7, 8, 9, 10])
        w_ho, w_fr = worst_measured(ho), worst_measured(fr)
        ships = w_ho == w_ho and w_fr == w_fr and min(w_ho, w_fr) >= -a.tol
        print(f"{name:>12} {'#' + str(bp['i']):>10} {bp['assess']['objective']:>8.2f} "
              f"{assess(ho).get('objective', float('nan')):>9.2f} "
              f"{assess(fr).get('objective', float('nan')):>8.2f}   {worst_per_wind(bp)[0]:>11.2f} "
              f"{w_ho:>12.2f} {w_fr:>10.2f}  {'yes' if ships else 'NO'}")
    print()
    print("The design column is the 14-episode screen; the other two are 28 episodes each and are the "
          "measurement.")
    print("Run scripts/dev/rosco_proposer_test.py for the delivery counts these rows feed.")
