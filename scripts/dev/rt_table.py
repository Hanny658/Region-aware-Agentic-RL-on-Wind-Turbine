"""Residual on the tuned ROSCO under the constrained objectives (campaign_agent_rt.sh): one table, no simulations.

Per run: the objective the run selected on (C or CT), the in-training trace (init row = the tuned ROSCO alone,
best episode), the held-out rows against the TUNED ROSCO (the gain on top of the baseline: objective, violation,
four terms), the absolute row against the ORIGINAL GSPI (wind seeds 3-6) and, where evaluated, the 600 s range set.

    ~/wtrl/run.sh python scripts/dev/rt_table.py [--csv docs/tables/tuned_rosco_residual.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser("~/wtrl/exp"))
ap.add_argument("--arms", nargs="+", default=["trwCw3t", "tgCw3L10", "trwTw3t", "tgTw3L10", "tgC3t", "trwC3t"])
ap.add_argument("--csv", default=None)
a = ap.parse_args()
DESC = {"trwCw3t": "agent-written reward, objective Cw (per-wind constraints)", "tgCw3L10": "fixed reward, tower weight 10, objective Cw",
        "trwTw3t": "agent-written reward, objective CTw (per-wind constraints)", "tgTw3L10": "fixed reward, tower weight 10, objective CTw",
        "tgC3t": "fixed reward (J-tuned, tower weight 1), objective C (set-mean)", "trwC3t": "agent-written reward, objective C (set-mean)",
        "tgC3L10": "fixed reward, tower weight 10, objective C", "tgT3L10": "fixed reward, tower weight 10, objective CT",
        "trwT3t": "agent-written reward, objective CT", "tgT3t": "fixed reward (J-tuned, tower weight 1), objective CT"}
T = ("J_power_mse_red_pct", "J_gen_speed_mse_red_pct", "J_TwrBsMyt_DEL_red_pct", "J_RootMyc1_DEL_red_pct")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def terms(d):
    return " / ".join(f"{d[t]:5.1f}" for t in T) if d else "-"


def duty(d):
    if not d or not d.get("per_episode"):
        return float("nan")
    per = d["per_episode"]
    return np.mean([q["pitch_travel_deg"] for q in per]) / np.mean([q["pitch_travel_base_deg"] for q in per])


rows = []
print(f"{'run':>11} {'obj':>3} {'best ep':>7} {'train best':>10} | held-out vs tuned ROSCO (s3-6): obj  viol | P / w / T / B   | (s7-10): obj | vs GSPI s3-6: J | P / w / T / B | range: J  xGSPI")
for arm in a.arms:
    for d in sorted(glob.glob(f"{a.exp}/{arm}_s?")):
        run = os.path.basename(d)
        summ = load(f"{d}/summary.json")
        if not os.path.exists(f"{d}/evals.csv"):
            continue
        obj = (summ or {}).get("objective") or ("Cw" if "Cw3" in run else "CTw" if "Tw3" in run else "C" if "C3" in run else "CT")
        ev = [r for r in csv.DictReader(open(f"{d}/evals.csv")) if r["tag"] in ("init", "eval")]
        best = max(ev, key=lambda r: float(r[obj]))
        h1, h2 = load(f"{d}/eval_heldout_s3456_ckpt_best.json"), load(f"{d}/eval_heldout2_s78910.json")
        ab, rg = load(f"{d}/eval_abs_gspi_s3456.json"), load(f"{d}/eval_range_TIB.json")
        o1 = f"{h1[obj]:6.2f} {h1[obj + '_violation_pct']:5.2f}" if h1 else "     -     -"
        o2 = f"{h2[obj]:6.2f}" if h2 else "     -"
        line = (f"{run:>11} {obj:>3} {int(best['episode']):7d} {float(best[obj]):10.2f} | {o1} | {terms(h1):>25} | {o2} | "
                f"{(f'{ab[chr(74)]:6.2f}' if ab else '     -')} | {terms(ab):>25} | {(f'{rg[chr(74)]:6.2f} {duty(rg):5.2f}' if rg else '-')}")
        print(line)
        rows.append({"run": run, "arm": arm, "description": DESC.get(arm, arm), "objective": obj, "done": bool(summ),
                     "best_episode": int(best["episode"]), "train_best_objective": round(float(best[obj]), 3),
                     "heldout_s3456_objective": round(h1[obj], 3) if h1 else None,
                     "heldout_s3456_violation_pct": round(h1[obj + "_violation_pct"], 3) if h1 else None,
                     **{f"heldout_s3456_{t}": round(h1[t], 3) for t in T if h1},
                     "heldout_s78910_objective": round(h2[obj], 3) if h2 else None,
                     **{f"heldout_s78910_{t}": round(h2[t], 3) for t in T if h2},
                     "abs_gspi_s3456_J": round(ab["J"], 3) if ab else None,
                     **{f"abs_gspi_s3456_{t}": round(ab[t], 3) for t in T if ab},
                     "range_J": round(rg["J"], 3) if rg else None, "range_duty_x_gspi": round(duty(rg), 3) if rg else None,
                     **{f"range_{t}": round(rg[t], 3) for t in T if rg}})
print("\nreference: the tuned ROSCO alone is the init row of every run (objective 0 by construction against its own baselines);"
      " against the original GSPI it scores J 14.20 (19.4 / 33.2 / 2.0 / 2.2) on wind seeds 3-6 at 150 s and 15.39 on the range set.")
if a.csv and rows:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("->", a.csv)
