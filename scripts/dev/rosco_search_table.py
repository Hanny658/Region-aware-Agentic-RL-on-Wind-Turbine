"""Proposer comparison on the shipped ROSCO pitch loop (2026-10-09; campaign_rosco_search.sh).

Four proposers search the same four-parameter box from the same shipped start under the same simulation budget. This
reads their histories and answers the three questions a design tool is judged on:

  how much      the best J each proposer reaches, and its four terms
  how reliably  the spread over repeats, because one run cannot separate a proposer from a lucky draw
  how fast      the number of simulations before the first candidate above a useful threshold

With --verify it also re-scores each arm's winner on the held-out and fresh wind seeds, which is the check roadmap
s38 failed: a design search is only useful if the design winds rank candidates the way held-out winds do.

    ~/wtrl/run.sh python scripts/dev/rosco_search_table.py [--csv docs/tables/rosco_search.csv] [--verify]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.expanduser("~/wtrl/exp/roscosearch"))
ap.add_argument("--thresh", type=float, default=10.0, help="J a candidate must exceed to count as a useful find")
ap.add_argument("--csv", default=None)
ap.add_argument("--verify", action="store_true", help="re-score each arm's winner on held-out and fresh seeds")
ap.add_argument("--workers", type=int, default=7)
ap.add_argument("--port0", type=int, default=7700)
a = ap.parse_args()

runs = {}
for h in sorted(glob.glob(f"{a.root}/*/history.jsonl")):
    name = os.path.basename(os.path.dirname(h))
    arm, rep = name.rsplit("_r", 1)
    hist = [json.loads(line) for line in open(h)]
    if len(hist) > 1:
        runs.setdefault(arm, {})[int(rep)] = hist
if not runs:
    sys.exit(f"no histories under {a.root}")

ORDER = [k for k in ("random", "local", "bayes", "llm") if k in runs] + \
        [k for k in runs if k not in ("random", "local", "bayes", "llm")]

print("Each run evaluates the shipped controller first, free of budget; the selection score is measured against the shipped")
print(f"baselines, so a positive score is better than shipped. 'first >{a.thresh:g}' counts simulations, not wall clock.\n")
print(f"{'proposer':>9} {'rep':>4} {'budget':>7} {'best score':>11} {'at sim':>7} {f'first >{a.thresh:g}':>10} "
      f"{'positive':>9} {'median score':>13}   best parameters")
rows = []
for arm in ORDER:
    for rep in sorted(runs[arm]):
        hist = runs[arm][rep]
        start, cand = hist[0], hist[1:]
        js = np.array([h["result"]["J"] for h in cand], float)
        best_i = int(np.argmax(js))
        first = next((i + 1 for i, v in enumerate(js) if v > a.thresh), None)
        p = cand[best_i]["params"]
        rows.append({"proposer": arm, "repeat": rep, "budget": len(cand), "start_J": round(start["result"]["J"], 2),
                     "best_J": round(float(js[best_i]), 2), "best_at": best_i + 1,
                     f"first_above_{a.thresh:g}": first if first else "",
                     "n_positive": int((js > 0).sum()), "median_J": round(float(np.median(js)), 2),
                     **{f"best_{k}": v for k, v in p.items()},
                     **{t: round(float(cand[best_i]["result"].get(t, float("nan"))), 2)
                        for t in ("power", "speed", "tower", "blade")},
                     "energy_loss_pct": round(float(cand[best_i]["result"].get("energy_loss_pct", 0.0)), 3),
                     "worst_term": round(min(float(cand[best_i]["result"].get(t, 0.0))
                                             for t in ("power", "speed", "tower", "blade")), 2)})
        print(f"{arm:>9} {rep:>4} {len(cand):>7} {js[best_i]:>11.2f} {best_i + 1:>7} {str(first or '-'):>10} "
              f"{int((js > 0).sum()):>4}/{len(cand):<4} {np.median(js):>13.2f}   "
              f"kp {p['kp_scale']:.3g} ki {p['ki_scale']:.3g} corner {p['lpf_corner']:.3g} damper {p['fa_ki']:.4g}")

print(f"\n{'proposer':>9} {'runs':>5} {'best score: mean':>17} {'sd':>6} {'min':>7} {'max':>7} "
      f"{'runs that found >' + f'{a.thresh:g}':>22} {'median sims to it':>18}")
for arm in ORDER:
    b = np.array([r["best_J"] for r in rows if r["proposer"] == arm], float)
    f = [r[f"first_above_{a.thresh:g}"] for r in rows if r["proposer"] == arm]
    hit = [x for x in f if x != ""]
    print(f"{arm:>9} {len(b):>5} {b.mean():>17.2f} {b.std(ddof=1) if len(b) > 1 else 0:>6.2f} "
          f"{b.min():>7.2f} {b.max():>7.2f} {len(hit):>13d} of {len(b):<6} "
          f"{(np.median(hit) if hit else float('nan')):>18.1f}")

print("\nWhat each arm's best candidate actually delivers: reductions against the shipped controller, in points")
print("(negative = worse than shipped; the selection score only averages the four, it does not forbid a loss):")
print(f"{'proposer':>9} {'rep':>4} {'power MSE':>10} {'speed MSE':>10} {'tower DEL':>10} {'blade DEL':>10} "
      f"{'energy':>7} {'worst':>7}")
for r in rows:
    print(f"{r['proposer']:>9} {r['repeat']:>4} {r['power']:>10.1f} {r['speed']:>10.1f} "
          f"{r['tower']:>10.1f} {r['blade']:>10.1f} {r['energy_loss_pct']:>7.2f} {r['worst_term']:>7.1f}")

print("\nBest candidates whose every term stays within 1 point of the shipped controller:")
for arm in ORDER:
    rs = [r for r in rows if r["proposer"] == arm]
    print(f"{arm:>9} {sum(1 for r in rs if r['worst_term'] >= -1.0):>2} of {len(rs)}")

# ------------------------------------------------------------------ held-out verification of the winners
if a.verify:
    from scripts.rosco_tune import Evaluator, clip  # noqa: E402
    from scripts.mpc_param_search import summarise  # noqa: E402

    class Args:
        root = a.root + "/_verify"
        template = os.path.expanduser("~/wtrl/runs/template_5mw")
        config_run = os.path.expanduser("~/wtrl/exp/jg3t_s0")
        seeds = [3, 4, 5, 6]
        episode_s = 600.0
        workers = a.workers
        port0 = a.port0

    ev = Evaluator(Args())
    ev.extra_args = ["--means", "12", "14", "16", "18", "20", "22", "24", "--ti", "B"]
    print("\nheld-out (seeds 3-6) and fresh (seeds 7-10) score of each arm's winner, same box, same protocol")
    print(f"{'proposer':>9} {'rep':>4} {'design score':>13} {'held-out':>9} {'fresh':>8}   (design minus held-out)")
    for r in rows:
        p = clip({k[len('best_'):]: v for k, v in r.items() if k.startswith("best_") and k != "best_J" and k != "best_at"})
        ho = summarise(ev.run(p, 0, seeds=[3, 4, 5, 6]))["J"]
        fr = summarise(ev.run(p, 1, seeds=[7, 8, 9, 10]))["J"]
        r["heldout_J"], r["fresh_J"] = round(ho, 2), round(fr, 2)
        print(f"{r['proposer']:>9} {r['repeat']:>4} {r['best_J']:>13.2f} {ho:>9.2f} {fr:>8.2f}   {r['best_J'] - ho:>+8.2f}")
    d = np.array([r["best_J"] - r["heldout_J"] for r in rows], float)
    print(f"\noptimism of the design score: mean {d.mean():+.2f} points, worst {d.max():+.2f}")
    if len(rows) > 2:
        c = np.corrcoef([r["best_J"] for r in rows], [r["heldout_J"] for r in rows])[0, 1]
        print(f"correlation of the design score with the held-out score over the {len(rows)} winners: {c:+.2f}")

if a.csv and rows:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("\n->", a.csv)
