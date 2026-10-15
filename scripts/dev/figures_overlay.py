"""Before and after, as levels: the untuned controller and the agent-tuned one on the same episodes (2026-10-15).

Figure F7 of `docs/paper/NARRATIVE.md`. Every other figure reports percentage reductions; this one overlays
the per-episode quantities those reductions are computed from, per mean wind speed, so a reader sees the two
curves the reduction is the distance between. Two rows: the shipped ROSCO pitch loop against the two
agent-proposed tuning files that hold every load term at every wind speed (leg 1), and the untuned offset-free
LPV-MPC against the agent-found wind schedule (leg 2), with the shipped controller drawn in both rows as the
common reference.

No simulation is run. Each evaluation on disk carries the absolute per-episode level of the controller it
evaluated (its CSV) and the percentage reduction against the paired shipped controller on the same wind file
(its JSON), so the shipped level is the controller's level divided by one minus the reduction - and because
every evaluation pairs against the same shipped rollout, that derived level has to agree across evaluations
file by file, which the script checks and reports.

    ~/wtrl/run.sh python scripts/dev/figures_overlay.py [--out docs/manuscript/figures]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP = os.path.expanduser("~/wtrl/exp")
CACHE = f"{EXP}/roscosearch_native/_verify/cache"

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(REPO, "docs", "manuscript", "figures"))
ap.add_argument("--seeds", default="s78910")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# (name, path stem): the CSV beside each JSON carries the absolute per-episode level
LEG1 = [
    ("agent-tuned ROSCO: one line changed (corner 1.57 -> 3.0), 1.00x travel", f"{CACHE}/eval_ddcdd3521ca6_{a.seeds}_90365b"),
    ("agent-tuned ROSCO: the same line at 6.5, 1.65x travel", f"{CACHE}/eval_52645b613ce6_{a.seeds}_90365b"),
]
LEG2 = [
    ("untuned offset-free LPV-MPC (no schedule)", f"{EXP}/mpcsearch600/fresh/eval_unsched_mpc_{a.seeds}"),
    ("agent-scheduled offset-free LPV-MPC", f"{EXP}/mpcsearch600/fresh/eval_sched_mpc_{a.seeds}"),
]
# column in the CSV (the level), key in the JSON (the reduction), panel title, y label
TERMS = [
    ("power_mse_R3", "power_mse_red_pct", "power MSE", "per-episode power MSE, as scored"),
    ("gen_speed_mse_R3", "gen_speed_mse_red_pct", "generator-speed MSE", "per-episode speed MSE, as scored"),
    ("TwrBsMyt_DEL_MNm", "TwrBsMyt_DEL_red_pct", "tower-base DEL", "DEL, MN m"),
    ("RootMyc1_DEL_MNm", "RootMyc1_DEL_red_pct", "blade-root DEL", "DEL, MN m"),
]


def load(stem: str) -> dict[str, dict]:
    """Per-episode records keyed by wind file: the CSV's level columns and the JSON's reduction columns."""
    if not (os.path.exists(stem + ".csv") and os.path.exists(stem + ".json")):
        sys.exit(f"missing evaluation: {stem}.{{csv,json}}")
    rows = list(csv.DictReader(open(stem + ".csv")))
    pe = json.load(open(stem + ".json"))["per_episode"]
    if len(rows) != len(pe):
        sys.exit(f"{stem}: {len(rows)} csv rows against {len(pe)} json records")
    out = {}
    for r, e in zip(rows, pe):
        d = {"mean_wind": float(r["mean_wind"])}
        for lvl, red, _, _ in TERMS:
            d[lvl] = float(r[lvl])
            d[red] = float(e[red])
            d["shipped_" + lvl] = d[lvl] / (1.0 - d[red] / 100.0)
        out[r["wind_file"]] = d
    return out


data = {n: load(p) for n, p in LEG1 + LEG2}
common = sorted(set.intersection(*(set(d) for d in data.values())))
if not common:
    sys.exit("no episodes in common")
winds = sorted({data[LEG1[0][0]][f]["mean_wind"] for f in common})

# the shipped level derived from every evaluation must be the same rollout, file by file
worst = 0.0
ref = LEG1[0][0]
for n in data:
    for f in common:
        for lvl, _, _, _ in TERMS:
            x, y = data[ref][f]["shipped_" + lvl], data[n][f]["shipped_" + lvl]
            worst = max(worst, abs(x - y) / max(abs(x), 1e-12))
print(f"{len(common)} episodes in common; the shipped level derived from the {len(data)} evaluations agrees "
      f"to a relative {worst:.1e} across all wind files and terms"
      + ("" if worst < 1e-6 else "  <- NOT the same paired rollout, do not use this figure"))


def bin_mean(n: str, key: str) -> list[float]:
    return [float(np.mean([data[n][f][key] for f in common if data[n][f]["mean_wind"] == u])) for u in winds]


plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "figure.dpi": 150, "savefig.dpi": 220, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.4), sharex=True)
SHIP = dict(color="#333333", lw=1.4, ls="--", marker="s", ms=3.0, label="shipped ROSCO, the paired reference")
# the two MSE terms span two decades between 12 m/s (near rated, where the loop saturates) and 18 m/s, so they
# are drawn on a log axis; the two DELs are within a factor of two and stay linear
for row, (leg, colours, styles) in enumerate(((LEG1, ("#55A868", "#2E7D4F"), ("-", "-.")),
                                              (LEG2, ("#E0A070", "#C44E52"), ("--", "-")))):
    for j, (ax, (lvl, _, title, ylab)) in enumerate(zip(axes[row], TERMS)):
        ax.plot(winds, bin_mean(leg[0][0], "shipped_" + lvl), **SHIP)
        for (n, _), c, ls in zip(leg, colours, styles):
            ax.plot(winds, bin_mean(n, lvl), color=c, lw=1.5, ls=ls, marker="o", ms=3.2, label=n)
        if j < 2:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        if row == 0:
            ax.set_title(title)
        else:
            ax.set_xlabel("mean wind speed, m/s")
        ax.set_ylabel(ylab)
axes[0][0].annotate("leg 1: the shipped pitch loop, tuned by the agent", (0.0, 1.16), xycoords="axes fraction",
                    fontsize=8.5, fontweight="bold", ha="left")
axes[1][0].annotate("leg 2: the untuned MPC, scheduled by the agent", (0.0, 1.06), xycoords="axes fraction",
                    fontsize=8.5, fontweight="bold", ha="left")
h1, l1 = axes[0][0].get_legend_handles_labels()
h2, l2 = axes[1][0].get_legend_handles_labels()
fig.legend(h1 + h2[1:], l1 + l2[1:], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07))
fig.suptitle(f"the levels the reductions are computed from: bin means over {len(common)} fresh-seed episodes "
             f"({a.seeds[1:]}), every curve on the same wind files", fontsize=8.5, y=1.03)
fig.tight_layout()
fig.savefig(f"{a.out}/f5_overlay.png", bbox_inches="tight")
print(f"-> {a.out}/f5_overlay.png")

# the per-wind levels, printed so the figure can be checked against the tables by eye
for lvl, red, title, _ in TERMS:
    print(f"\n{title}: bin means, shipped then each controller")
    print(f"{'U':>4} " + f"{'shipped':>12} " + " ".join(f"{n[:28]:>28}" for n, _ in LEG1 + LEG2))
    ship = bin_mean(LEG1[0][0], "shipped_" + lvl)
    cols = [bin_mean(n, lvl) for n, _ in LEG1 + LEG2]
    for i, u in enumerate(winds):
        print(f"{u:>4.0f} {ship[i]:>12.5g} " + " ".join(f"{c[i]:>28.5g}" for c in cols))
