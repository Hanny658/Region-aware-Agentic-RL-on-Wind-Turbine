"""Proposer comparison on the shipped ROSCO pitch loop (2026-10-09, redesigned 2026-10-11).

Four proposers search the same five ROSCO tuning quantities from the same shipped start under the same
simulation budget and the same constrained selection rule. This reads their histories and answers the three
questions a design tool is judged on:

  how much      the regulation the selected tuning buys, and what it does to the loads it must not harm
  how reliably  the spread over repeats, because one run cannot separate a proposer from a lucky draw
  how fast      the simulations spent before the first feasible tuning that improves on the shipped one

Everything is reported in the quantities the measurement produces -- percentage reductions against the paired
shipped controller, energy, pitch travel, actuator duty -- with no composite anywhere. Selection is
`eval/selection.py`: one objective, three constraints at "no worse than shipped", constraint domination.

With --verify each selected tuning is re-scored on the held-out and fresh wind seeds, which is the check
roadmap s38 failed: a design search is only useful if the design winds rank candidates the way held-out winds
do.

    ~/wtrl/run.sh python scripts/dev/rosco_search_table.py [--csv docs/tables/rosco_search.csv] [--verify]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from eval.selection import rank_key  # noqa: E402
from scripts.rosco_native import FA_INT_SAT, fmt  # noqa: E402

HPF_INTEGRAL_M_S = 1.076      # median max |integral of high-passed nacelle fore-aft acceleration|, from the
                              # shipped baselines (scripts/dev/search_box_bounds.py); used only to say whether
                              # the damper's pitch contribution would reach its 10 deg limiter

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.expanduser("~/wtrl/exp/roscosearch_native"))
ap.add_argument("--csv", default=None)
ap.add_argument("--verify", action="store_true", help="re-score each selected tuning on held-out and fresh seeds")
ap.add_argument("--workers", type=int, default=7)
ap.add_argument("--port0", type=int, default=7700)
ap.add_argument("--means", nargs="+", type=float, default=[12, 14, 16, 18, 20, 22, 24])
ap.add_argument("--ti", default="B")
a = ap.parse_args()

runs: dict[str, dict[int, list]] = {}
for h in sorted(glob.glob(f"{a.root}/*/history.jsonl")):
    name = os.path.basename(os.path.dirname(h))
    if "_r" not in name:
        continue
    arm, rep = name.rsplit("_r", 1)
    hist = [json.loads(line) for line in open(h)]
    if len(hist) > 1:
        runs.setdefault(arm, {})[int(rep)] = hist
if not runs:
    sys.exit(f"no histories under {a.root}")

ORDER = [k for k in ("random", "local", "bayes", "llm", "tune") if k in runs] + \
        [k for k in runs if k not in ("random", "local", "bayes", "llm", "tune")]

print("Each run evaluates the shipped controller first, free of budget; its objective must read exactly 0.")
print("A tuning is selected only if tower-base load, blade-root load and energy capture are all no worse than")
print("shipped; 'sims to first' counts simulations to the first such tuning that also improves regulation.\n")
hdr = (f"{'proposer':>9} {'rep':>4} {'budget':>7} {'identity':>9} {'feasible':>9} {'sims to first':>14} "
       f"{'regulation':>11} {'power MSE':>10} {'tower DEL':>10} {'blade DEL':>10} {'energy':>7} "
       f"{'travel':>7} {'ADC':>6}")
print(hdr)
rows = []
for arm in ORDER:
    for rep in sorted(runs[arm]):
        hist = runs[arm][rep]
        start, cand = hist[0], hist[1:]
        best = max(cand, key=rank_key)
        asm = best["assess"]
        c = asm.get("constraints", {})
        n_feas = sum(1 for h in cand if h["assess"].get("feasible"))
        first = next((h["i"] for h in cand
                      if h["assess"].get("feasible") and h["assess"]["objective"] > 0), None)
        dem = np.rad2deg(best["params"]["fa_ki"] * HPF_INTEGRAL_M_S)
        r = {"proposer": arm, "repeat": rep, "budget": len(cand),
             "identity_objective": round(start["assess"]["objective"], 3),
             "n_feasible": n_feas, "sims_to_first_feasible_gain": first if first else "",
             "selected_feasible": bool(asm.get("feasible")),
             "regulation": round(asm.get("objective", float("nan")), 2),
             "power": round(asm.get("power", float("nan")), 2),
             "tower": round(c.get("tower", {}).get("value", float("nan")), 2),
             "tower_lo": round(c.get("tower", {}).get("lo", float("nan")), 2),
             "tower_hi": round(c.get("tower", {}).get("hi", float("nan")), 2),
             "blade": round(c.get("blade", {}).get("value", float("nan")), 2),
             "energy": round(c.get("energy", {}).get("value", float("nan")), 3),
             "travel_ratio": best.get("travel_ratio", float("nan")),
             "adc": best.get("adc", float("nan")),
             "damper_demand_deg": round(float(dem), 2),
             "damper_limiter_reached": bool(dem > np.rad2deg(FA_INT_SAT)),
             **{f"best_{k}": v for k, v in best["params"].items()}}
        rows.append(r)
        print(f"{arm:>9} {rep:>4} {len(cand):>7} {r['identity_objective']:>9.3f} {n_feas:>4}/{len(cand):<4} "
              f"{str(first or '-'):>14} {r['regulation']:>11.2f} {r['power']:>10.2f} {r['tower']:>10.2f} "
              f"{r['blade']:>10.2f} {r['energy']:>7.3f} {r['travel_ratio']:>7.3f} {r['adc']:>6.4f}"
              + ("" if asm.get("feasible") else "   <- no feasible tuning found; least-violating shown"))

print("\nThe selected tunings, in ROSCO's own quantities:")
for r in rows:
    p = {k[len("best_"):]: v for k, v in r.items() if k.startswith("best_")}
    note = ""
    if p["fa_ki"] > 0:
        note = (f"   damper demand about {r['damper_demand_deg']:.1f} deg"
                + ("  (reaches the 10 deg limiter)" if r["damper_limiter_reached"] else ""))
    print(f"{r['proposer']:>9} {r['repeat']:>4}   {fmt(p)}{note}")

print(f"\n{'proposer':>9} {'runs':>5} {'regulation: mean':>17} {'sd':>6} {'min':>7} {'max':>7} "
      f"{'runs with a feasible gain':>26} {'median sims to it':>18}")
for arm in ORDER:
    rs = [r for r in rows if r["proposer"] == arm]
    ok = [r for r in rs if r["selected_feasible"]]
    b = np.array([r["regulation"] for r in ok], float) if ok else np.array([np.nan])
    f = [r["sims_to_first_feasible_gain"] for r in rs if r["sims_to_first_feasible_gain"] != ""]
    print(f"{arm:>9} {len(rs):>5} {b.mean():>17.2f} {(b.std(ddof=1) if len(b) > 1 else 0.0):>6.2f} "
          f"{b.min():>7.2f} {b.max():>7.2f} {len(ok):>17d} of {len(rs):<6} "
          f"{(np.median(f) if f else float('nan')):>18.1f}")

print("\nBest feasible regulation so far, against simulations spent (the whole trace, so no threshold is needed):")
marks = [2, 4, 6, 8, 12, 16, 20, 24]
print(f"{'proposer':>9} {'rep':>4} " + " ".join(f"{m:>7}" for m in marks))
for arm in ORDER:
    for rep in sorted(runs[arm]):
        cand = runs[arm][rep][1:]
        cells = []
        for m in marks:
            seen = [h["assess"]["objective"] for h in cand
                    if h["i"] <= m and h["assess"].get("feasible")]
            cells.append(f"{max(seen):>7.2f}" if seen else f"{'-':>7}")
        print(f"{arm:>9} {rep:>4} " + " ".join(cells))

# ------------------------------------------------------------------ held-out verification of the selected tunings
if a.verify:
    from eval.selection import assess  # noqa: E402
    from scripts.rosco_native import Evaluator  # noqa: E402

    class Args:
        root = a.root + "/_verify"
        template = os.path.expanduser("~/wtrl/runs/template_5mw")
        config_run = os.path.expanduser("~/wtrl/exp/jg3t_s0")
        seeds = [3, 4, 5, 6]
        episode_s = 600.0
        workers = a.workers
        port0 = a.port0

    ev = Evaluator(Args())
    ev.extra_args = ["--means", *[f"{m:g}" for m in a.means], "--ti", a.ti]
    print("\nThe selected tunings on wind seeds they were not designed on, same box, same protocol.")
    print(f"{'proposer':>9} {'rep':>4} {'design':>8} {'held-out':>9} {'fresh':>8} {'gap':>7}   "
          f"{'held-out tower':>15} {'fresh tower':>12}  feasible")
    for r in rows:
        p = {k[len("best_"):]: v for k, v in r.items() if k.startswith("best_")}
        ho = assess(ev.run(p, 0, seeds=[3, 4, 5, 6]))
        fr = assess(ev.run(p, 1, seeds=[7, 8, 9, 10]))
        r["heldout_regulation"] = round(ho["objective"], 2)
        r["fresh_regulation"] = round(fr["objective"], 2)
        r["heldout_tower"] = round(ho["constraints"].get("tower", {}).get("value", float("nan")), 2)
        r["fresh_tower"] = round(fr["constraints"].get("tower", {}).get("value", float("nan")), 2)
        r["heldout_feasible"] = bool(ho["feasible"])
        r["fresh_feasible"] = bool(fr["feasible"])
        print(f"{r['proposer']:>9} {r['repeat']:>4} {r['regulation']:>8.2f} {r['heldout_regulation']:>9.2f} "
              f"{r['fresh_regulation']:>8.2f} {r['regulation'] - r['heldout_regulation']:>+7.2f}   "
              f"{r['heldout_tower']:>15.2f} {r['fresh_tower']:>12.2f}  "
              f"{'held-out yes' if ho['feasible'] else 'held-out NO'}, "
              f"{'fresh yes' if fr['feasible'] else 'fresh NO'}")
    d = np.array([r["regulation"] - r["heldout_regulation"] for r in rows], float)
    print(f"\nOptimism of the design score: mean {d.mean():+.2f} points, worst {d.max():+.2f}.")
    print(f"Selected tunings that stay feasible off the design winds: "
          f"{sum(r['heldout_feasible'] for r in rows)} of {len(rows)} held out, "
          f"{sum(r['fresh_feasible'] for r in rows)} of {len(rows)} fresh.")
    if len(rows) > 2:
        x = [r["regulation"] for r in rows]
        y = [r["heldout_regulation"] for r in rows]
        c = np.corrcoef(x, y)[0, 1]
        rk = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]
        print(f"Design against held-out over the {len(rows)} selected tunings: correlation {c:+.2f}, "
              f"rank correlation {rk:+.2f} (roadmap s38 measured -0.9 for the model-based design search).")

if a.csv and rows:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\n->", a.csv)
