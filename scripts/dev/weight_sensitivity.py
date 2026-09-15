"""Sensitivity of the controller ranking to the objective's weights (2026-09-15, zero simulations).

J is the equal-weight mean of four per-episode-clipped % reductions (power MSE, generator-speed MSE, tower-base
fatigue, blade-root fatigue) minus the energy penalty. The stored aggregate terms are already clipped per
episode, so a weighted objective J_w = sum_m w_m term_m - penalty is exact from the evaluation files.
Two one-parameter families: the tower-fatigue weight w_T from 0 to 0.7 with the rest shared equally, and the
regulation share (power + speed) from 0 to 1 with the fatigue share split equally. Every checkpoint was selected
under equal weights, so this re-weights the same controllers; it does not produce the optimum for other weights.

    ~/wtrl/run.sh python scripts/dev/weight_sensitivity.py [--csv docs/tables/weight_sensitivity.csv] [--fig docs/figures/weight_sensitivity.png]
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
ap.add_argument("--csv", default=None)
ap.add_argument("--fig", default=None)
a = ap.parse_args()
E = a.exp
K = ("J_power_mse_red_pct", "J_gen_speed_mse_red_pct", "J_TwrBsMyt_DEL_red_pct", "J_RootMyc1_DEL_red_pct")
SETS = {"S3-6": "eval_heldout_s3456_ckpt_best.json", "S7-10": "eval_heldout2_s78910.json"}


def terms(j):
    return np.array([j[k] for k in K], float), float(j.get("J_energy_penalty", 0.0))


rows = {}   # label -> list of (terms, penalty) over runs x both sets
arms = [("jg3t", "GSPI base: fixed reward"), ("jrw3t", "GSPI base: agent reward"), ("jrr3t", "GSPI base: random reward structure"),
        ("mg3t", "MPC base: fixed-reward residual"), ("mrw3t", "MPC base: agent residual")]
for arm, label in arms:
    for d in sorted(glob.glob(f"{E}/{arm}_s*")):
        for fn in SETS.values():
            p = f"{d}/{fn}"
            if os.path.exists(p):
                rows.setdefault(label, []).append(terms(json.load(open(p))))
for tag_a, tag_b, label in (("base0_s3456", "base0_s78910", "MPC, nominal"),
                            ("base0_offset_cp1_s3456", "base0_offset_cp1_s78910", "MPC, offset-free"),
                            ("base0_rls_cp1_s3456", "base0_rls_cp1_s78910", "MPC, adaptive")):
    for t in (tag_a, tag_b):
        p = f"{E}/mpc/eval_{t}.json"
        if os.path.exists(p):
            rows.setdefault(label, []).append(terms(json.load(open(p))))


def score(label, w):
    return float(np.mean([tt @ w - pen for tt, pen in rows[label]]))


out = []
print("tower-fatigue weight sweep (others share the rest equally); mean over runs and both held-out sets")
wts = np.round(np.arange(0.0, 0.71, 0.1), 2)
print(f"{'controller':>36} " + " ".join(f"wT={x:.1f}" for x in wts))
fam1 = {}
for label in rows:
    vals = []
    for wT in wts:
        w = np.array([(1 - wT) / 3, (1 - wT) / 3, wT, (1 - wT) / 3])
        vals.append(score(label, w))
        out.append({"family": "tower_weight", "x": wT, "controller": label, "J_w": vals[-1], "n": len(rows[label])})
    fam1[label] = vals
    print(f"{label:>36} " + " ".join(f"{v:6.2f}" for v in vals))
print("\nregulation share sweep (power and speed share x equally, tower and blade share 1-x equally)")
xs = np.round(np.arange(0.0, 1.01, 0.125), 3)
print(f"{'controller':>36} " + " ".join(f"{x:5.2f}" for x in xs))
fam2 = {}
for label in rows:
    vals = []
    for x in xs:
        w = np.array([x / 2, x / 2, (1 - x) / 2, (1 - x) / 2])
        vals.append(score(label, w))
        out.append({"family": "regulation_share", "x": x, "controller": label, "J_w": vals[-1], "n": len(rows[label])})
    fam2[label] = vals
    print(f"{label:>36} " + " ".join(f"{v:5.1f}" for v in vals))
# rank of the strongest controller at every weight
for name, fam, grid in (("tower weight", fam1, wts), ("regulation share", fam2, xs)):
    best = [max(fam, key=lambda l: fam[l][i]) for i in range(len(grid))]
    print(f"\nbest controller per {name}: " + ", ".join(f"{g:g}: {b}" for g, b in zip(grid, best)))

if a.csv:
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print("->", a.csv)
if a.fig:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, fam, grid, xl in ((axes[0], fam1, wts, "tower-fatigue weight (others share the rest equally)"),
                              (axes[1], fam2, xs, "regulation share (power + speed); fatigue share = 1 - x")):
        for label, vals in fam.items():
            ls = "-" if label.startswith("MPC,") else "--"
            ax.plot(grid, vals, ls, marker="o", ms=3, label=label)
        ax.axhline(0, color="k", lw=0.5); ax.set_xlabel(xl); ax.set_ylabel("weighted objective (held-out)")
    axes[1].axvline(0.5, color="0.6", lw=0.6, ls=":")
    axes[0].axvline(0.25, color="0.6", lw=0.6, ls=":")
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(a.fig, dpi=180)
    print("->", a.fig)
