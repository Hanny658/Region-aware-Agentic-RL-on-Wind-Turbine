"""J against actuator duty over every MPC setting evaluated by the 600 s parameter search (no simulations).

Roadmap s22 found the MPC buys its J with ~7x the GSPI pitch travel; s25 that removing the 3P line does not reduce
the travel. The 600 s search cache holds 120+ MPC settings on the supervisor winds, each with J and per-episode
pitch travel, so the trade-off between J and actuator duty is already measured. This script draws it, marks the
Pareto-efficient settings, and places the reference controllers and the learned-residual arms (supervisor-wind
scores) on the same axes.

    ~/wtrl/run.sh python scripts/dev/pareto_duty.py [--fig docs/figures/pareto_duty.png] [--csv docs/tables/pareto_duty.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.mpc_param_search import key_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser("~/wtrl/exp"))
ap.add_argument("--fig", default=None)
ap.add_argument("--csv", default=None)
a = ap.parse_args()
E = a.exp
R = f"{E}/mpcsearch600"


def duty(j, window):
    per = [p["pitch_travel_deg"] for p in j.get("per_episode", []) if "pitch_travel_deg" in p]
    base = [p["pitch_travel_base_deg"] for p in j.get("per_episode", []) if "pitch_travel_base_deg" in p]
    return (mean(per) / window if per else float("nan")), (mean(per) / mean(base) if per and base else float("nan"))


pts = []  # (duty deg/s, J, label, kind, params)
for arm in ("llm", "es", "random"):
    hp = f"{R}/{arm}/history.jsonl"
    if not os.path.exists(hp):
        continue
    for l in open(hp):
        h = json.loads(l)
        p = f"{R}/cache/eval_{key_of(h['params'])}_cp1_s12.json"
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        if j.get("failed"):
            continue
        d, ratio = duty(j, 580.0)
        pts.append((d, j["J"], f"{arm} r={h['params']['r']:g} qt={h['params']['qt']:g}", "search", h["params"]))
refs = []
for lab, p, window in (("nominal MPC", f"{R}/refs/eval_nominal_cp1_s12.json", 580.0),
                       ("offset-free MPC", f"{R}/refs/eval_offset_cp1_s12.json", 580.0),
                       ("adaptive MPC", f"{R}/refs/eval_rls_cp1_s12.json", 580.0),
                       ("offset-free + 3P notch Q5", f"{R}/refs/eval_offset_notch5_cp1_s12.json", 580.0),
                       ("ROSCO + tower damper", f"{R}/refs/eval_towerdamper_s12.json", 580.0)):
    if os.path.exists(p):
        j = json.load(open(p)); d, _ = duty(j, window)
        refs.append((d, j["J"], lab))
# learned residual arms on the supervisor winds (150 s training evaluations, best checkpoint = the decisions.jsonl best)
arms = []
for arm, lab in (("jg3t", "GSPI + fixed-reward residual"), ("jrw3t", "GSPI + agent-reward residual"),
                 ("mg3t", "MPC + fixed-reward residual"), ("mrw3t", "MPC + agent-reward residual")):
    ds, js = [], []
    for d in sorted(glob.glob(f"{E}/{arm}_s*")):
        hp = f"{d}/eval_heldout_s3456_ckpt_best.json"     # held-out 150 s; the supervisor-wind duty is not stored per run
        if os.path.exists(hp):
            j = json.load(open(hp)); dd, _ = duty(j, 130.0); ds.append(dd); js.append(j["J"])
    if ds:
        arms.append((mean(ds), mean(js), f"{lab} (n={len(ds)}, held-out 150 s)"))

# Pareto front of the search points (max J, min duty)
srt = sorted(pts, key=lambda t: t[0])
front, best = [], -1e9
for t in srt:
    if t[1] > best:
        front.append(t); best = t[1]
print(f"search settings with a valid evaluation: {len(pts)}; Pareto-efficient: {len(front)}")
print(f"{'duty [deg/s]':>13} {'J':>7}  setting")
for d, J, lab, _, p in front:
    print(f"{d:13.3f} {J:7.2f}  {lab}   {p}")
print("\nreferences (supervisor winds, 600 s):")
for d, J, lab in refs:
    print(f"{d:13.3f} {J:7.2f}  {lab}")
print("\nlearned residual arms (held-out 150 s, duty on the 130 s window):")
for d, J, lab in arms:
    print(f"{d:13.3f} {J:7.2f}  {lab}")
gspi_duty = None
for p in glob.glob(f"{R}/refs/eval_nominal_cp1_s12.json"):
    j = json.load(open(p)); b = [q["pitch_travel_base_deg"] for q in j["per_episode"]]; gspi_duty = mean(b) / 580.0
if gspi_duty:
    print(f"\nGSPI itself: duty {gspi_duty:.3f} deg/s, J = 0 by definition")

if a.csv:
    with open(a.csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["kind", "duty_deg_per_s", "J", "label", "params"])
        for d, J, lab, k, p in pts:
            w.writerow([k, round(d, 4), round(J, 3), lab, json.dumps(p)])
        for d, J, lab in refs:
            w.writerow(["reference", round(d, 4), round(J, 3), lab, ""])
        for d, J, lab in arms:
            w.writerow(["residual_arm", round(d, 4), round(J, 3), lab, ""])
    print("->", a.csv)
if a.fig:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter([t[0] for t in pts], [t[1] for t in pts], s=12, color="0.7", label="MPC settings tried by the search (600 s, supervisor winds)")
    ax.plot([t[0] for t in front], [t[1] for t in front], "-o", ms=4, color="#4C72B0", label="Pareto front (max J, min duty)")
    for d, J, lab in refs:
        ax.scatter([d], [J], s=40, marker="s", color="#C44E52"); ax.annotate(lab, (d, J), textcoords="offset points", xytext=(5, 3), fontsize=7)
    for d, J, lab in arms:
        ax.scatter([d], [J], s=40, marker="^", color="#55A868"); ax.annotate(lab.split(" (")[0], (d, J), textcoords="offset points", xytext=(5, -9), fontsize=7)
    if gspi_duty:
        ax.scatter([gspi_duty], [0], s=40, marker="*", color="k"); ax.annotate("GSPI", (gspi_duty, 0), textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("actuator duty: pitch travel per scored second [deg/s]"); ax.set_ylabel("J")
    ax.set_ylim(bottom=max(-5, min(t[1] for t in pts) - 2)); ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(a.fig, dpi=180); print("->", a.fig)
