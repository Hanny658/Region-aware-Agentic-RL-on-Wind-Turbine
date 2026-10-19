"""The one-knob sweep read the three ways the search arms are read (roadmap s63, review round 1).

The sweep run (`rosco_search.py --proposer sweep --budget 8`) evaluates the generator-speed filter corner alone on a
fixed grid, shipped gains, damper off, on the 14-episode design set. This script reads its history under the
set-mean rule the searches used, under the per-wind screen and under the completed rule, measures every distinct
selection on the 28 held-out and the 28 fresh episodes through the same evaluator and cache as every other arm,
and writes one CSV row per grid point plus the three selections.

    WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 ~/wtrl/run.sh python scripts/dev/rosco_sweep_report.py \
        --run ~/wtrl/exp/roscosearch_sweep/sweep_r0 --verify_root ~/wtrl/exp/roscosearch_sweep \
        --csv docs/tables/rosco_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))
from eval.selection import assess, per_wind_rank_key, rank_key, worst_per_wind  # noqa: E402
from scripts.rosco_native import Evaluator, fmt, key_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=os.path.expanduser("~/wtrl/exp/roscosearch_sweep/sweep_r0"))
ap.add_argument("--verify_root", default=os.path.expanduser("~/wtrl/exp/roscosearch_sweep"))
ap.add_argument("--csv", default=None)
ap.add_argument("--ship_tol", type=float, default=1.0)
ap.add_argument("--workers", type=int, default=7)
ap.add_argument("--port0", type=int, default=8300)
ap.add_argument("--means", nargs="+", type=float, default=[12, 14, 16, 18, 20, 22, 24])
ap.add_argument("--ti", default="B")
ap.add_argument("--no_verify", action="store_true", help="read the design set only")
a = ap.parse_args()

hist = [json.loads(line) for line in open(os.path.join(a.run, "history.jsonl"))]
cands = hist[1:]
print(f"{len(cands)} grid points evaluated on the design set ({a.run})")


def terms(r):
    s = r["assess"]
    c = s.get("constraints") or {}
    g = lambda k: c.get(k, {}).get("value", float("nan"))  # noqa: E731
    return s["objective"], g("tower"), g("blade"), g("energy")


# ---------------------------------------------------------------- the three readings
sel_set = max(hist, key=rank_key)
sel_pw = max(hist, key=lambda r: per_wind_rank_key(r, 0.0))
sel_c6 = max(hist, key=lambda r: per_wind_rank_key(r, 0.0, 6.0))
first_gain = next((r["i"] for r in cands if r["assess"].get("feasible") and r["assess"]["objective"] > 0), None)
first_pw = next((r["i"] for r in cands if per_wind_rank_key(r, 0.0)[0] == 1 and r["assess"]["objective"] > 0), None)
print(f"first feasible gain on the set mean: candidate #{first_gain}; first holding the per-wind screen: #{first_pw}")
print(f"{'#':>3} {'corner':>7} {'speed':>7} {'tower':>7} {'blade':>7} {'energy':>7} {'travel':>7} {'worst/wind':>11}  set-mean  per-wind  completed")
for r in cands:
    o, t, b, e = terms(r)
    w, at, which = worst_per_wind(r)
    flags = ("ok" if r["assess"].get("feasible") else "infeasible",
             "holds" if per_wind_rank_key(r, 0.0)[0] == 1 else "-",
             "holds" if per_wind_rank_key(r, 0.0, 6.0)[0] == 1 else "-")
    print(f"{r['i']:>3} {r['params']['lpf_corner']:>7.2f} {o:>+7.2f} {t:>+7.2f} {b:>+7.2f} {e:>+7.3f} "
          f"{r.get('travel_ratio', float('nan')):>7.3f} {w:>+6.1f} ({which} {at:g})  {flags[0]:>8}  {flags[1]:>8}  {flags[2]:>9}")

selections = {"set-mean rule": sel_set, "per-wind screen": sel_pw, "completed rule": sel_c6}
for name, r in selections.items():
    print(f"selection under the {name}: " + ("the shipped tuning" if r["i"] == 0 else
          f"#{r['i']} corner {r['params']['lpf_corner']:g}, regulation {r['assess']['objective']:+.2f} on the design set"))

# ---------------------------------------------------------------- off the design winds
rows = []
if not a.no_verify:
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

    def worst_measured(fit):
        by = {}
        for e in fit.get("per_episode", []):
            if e.get("mean_wind") is None:
                continue
            d = by.setdefault(e["mean_wind"], {})
            for nm, col in (("tower", "TwrBsMyt_DEL_red_pct"), ("blade", "RootMyc1_DEL_red_pct")):
                v = e.get(col)
                if v is not None and v == v:
                    d.setdefault(nm, []).append(float(v))
        cells = [(sum(v) / len(v), nm, u) for u, d in by.items() for nm, v in d.items()]
        return min(cells) if cells else (float("nan"), "", float("nan"))

    print(f"\nthe selections off the design winds (28 held-out + 28 fresh episodes; ships if both loads within "
          f"{a.ship_tol:.1f} at every wind speed on both sets):")
    done = {}
    for name, r in selections.items():
        if r["i"] == 0:
            print(f"  {name:>16}: the shipped tuning, nothing to measure")
            continue
        k = key_of(r["params"])
        if k not in done:
            ho = ev.run(r["params"], 0, seeds=[3, 4, 5, 6])
            fr = ev.run(r["params"], 1, seeds=[7, 8, 9, 10])
            done[k] = (ho, fr)
        ho, fr = done[k]
        aho, afr = assess(ho), assess(fr)
        wh, wf = worst_measured(ho), worst_measured(fr)
        ships = wh[0] == wh[0] and wf[0] == wf[0] and min(wh[0], wf[0]) >= -a.ship_tol
        print(f"  {name:>16}: #{r['i']} corner {r['params']['lpf_corner']:g}  design {r['assess']['objective']:+.2f}  "
              f"held out {aho.get('objective', float('nan')):+.2f} (worst {wh[0]:+.2f} {wh[1]} {wh[2]:g})  "
              f"fresh {afr.get('objective', float('nan')):+.2f} (worst {wf[0]:+.2f} {wf[1]} {wf[2]:g})  "
              f"{'SHIPS' if ships else 'does not ship'}")
        rows.append({"selection": name, "candidate": r["i"], "corner": r["params"]["lpf_corner"], "key": k,
                     "design_regulation": round(r["assess"]["objective"], 2),
                     "design_worst": round(worst_per_wind(r)[0], 2),
                     "heldout_regulation": round(aho.get("objective", float("nan")), 2), "heldout_worst": round(wh[0], 2),
                     "heldout_worst_cell": f"{wh[1]} {wh[2]:g}",
                     "fresh_regulation": round(afr.get("objective", float("nan")), 2), "fresh_worst": round(wf[0], 2),
                     "fresh_worst_cell": f"{wf[1]} {wf[2]:g}",
                     "fresh_power": round(afr.get("power", float("nan")), 2),
                     "fresh_tower": round((afr.get("constraints") or {}).get("tower", {}).get("value", float("nan")), 2),
                     "fresh_blade": round((afr.get("constraints") or {}).get("blade", {}).get("value", float("nan")), 2),
                     "travel": round(r.get("travel_ratio", float("nan")), 3), "ships": ships})

if a.csv:
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        cols = ["i", "corner", "tuning", "design_regulation", "design_power", "design_tower", "design_blade",
                "design_energy", "travel", "adc", "design_worst", "design_worst_cell", "feasible_set_mean",
                "holds_per_wind", "holds_completed", "first_feasible_gain", "first_per_wind"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in cands:
            o, t, b, e = terms(r)
            ww, at, which = worst_per_wind(r)
            w.writerow({"i": r["i"], "corner": r["params"]["lpf_corner"], "tuning": fmt(r["params"]),
                        "design_regulation": round(o, 2), "design_power": round(r["assess"].get("power", float("nan")), 2),
                        "design_tower": round(t, 2), "design_blade": round(b, 2), "design_energy": round(e, 3),
                        "travel": round(r.get("travel_ratio", float("nan")), 3), "adc": round(r.get("adc", float("nan")), 4),
                        "design_worst": round(ww, 2), "design_worst_cell": f"{which} {at:g}",
                        "feasible_set_mean": bool(r["assess"].get("feasible")),
                        "holds_per_wind": per_wind_rank_key(r, 0.0)[0] == 1,
                        "holds_completed": per_wind_rank_key(r, 0.0, 6.0)[0] == 1,
                        "first_feasible_gain": first_gain, "first_per_wind": first_pw})
    if rows:
        sel_csv = a.csv.replace(".csv", "_selections.csv")
        with open(sel_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"-> {sel_csv}")
    print(f"-> {a.csv}")
