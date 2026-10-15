"""Figure F6: the same controllers under an equally weighted and a site-weighted aggregation (2026-10-15).

Reads `docs/tables/lifetime_del.csv` as `scripts/dev/lifetime_del.py --csv` writes it, so the bars are the
numbers in the committed table and nothing else. (a) the IEC Rayleigh weights over the evaluated 12-24 m/s
bins at three annual means; (b) tower-base DEL and (c) generator-speed MSE reductions against the paired
shipped controller, equally weighted and Rayleigh-weighted, for the rungs the paper reports. Replaces the
weighting panel of figures_v5.py, which was keyed on the hand-scaled baseline that s52 voided.

    ~/wtrl/run.sh python scripts/dev/figures_weighting.py [--out docs/manuscript/figures]
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=os.path.join(REPO, "docs", "tables", "lifetime_del.csv"))
ap.add_argument("--out", default=os.path.join(REPO, "docs", "manuscript", "figures"))
ap.add_argument("--gate_seed", default="s1", help="which gated seed to show, as ladder_published.py does")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "figure.dpi": 150, "savefig.dpi": 220, "axes.spines.top": False,
                     "axes.spines.right": False})
WINDS = np.arange(12, 25, 2)
VAVES = (10.0, 8.5, 7.5)
# label in the CSV -> (tick label, colour); the colours follow the ladder figure
PICK = {"tuned baseline, native (s52)": ("tuned\nbaseline", "#4C72B0"),
        "agent one-line tuning (corner 3.0)": ("agent,\none line", "#8FBF9F"),
        "scheduled MPC": ("scheduled\nMPC", "#DD8452"),
        f"native baseline + gated residual ({a.gate_seed})": ("baseline\n+ layer", "#55A868"),
        f"scheduled MPC + gated residual ({a.gate_seed})": ("MPC\n+ layer", "#C44E52")}

rows = list(csv.DictReader(open(a.csv)))
have = [k for k in PICK if any(r["controller"] == k for r in rows)]
missing = [k for k in PICK if k not in have]
if missing:
    print("not in the table, skipped:", missing)


def val(ctrl: str, col: str, vave: float) -> float:
    return next(float(r[col]) for r in rows if r["controller"] == ctrl and float(r["Vave"]) == vave)


fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.0))
ax = axes[0]
for vave, ls in zip(VAVES, ("-", "--", ":")):
    cdf = lambda v: 1 - np.exp(-np.pi / 4 * (v / vave) ** 2)      # noqa: E731
    w = np.array([cdf(u + 1) - cdf(u - 1) for u in WINDS])
    w = w / w.sum()
    ax.plot(WINDS, 100 * w, ls, color="0.25", lw=1.4, marker="o", ms=2.8, label=f"$V_{{ave}}$ = {vave:g} m/s")
ax.set_xlabel("mean wind speed, m/s")
ax.set_ylabel("weight over 12-24 m/s, %")
ax.set_title("(a) IEC Rayleigh weights, renormalised")
ax.legend(frameon=False)

for ax, q, name in ((axes[1], "tower", "(b) tower-base DEL"), (axes[2], "speed MSE", "(c) generator-speed MSE")):
    x = np.arange(len(have))
    eq = [val(k, f"{q}_equal_mean_pct", 10.0) for k in have]
    ax.bar(x - 0.30, eq, 0.20, color="0.78", label="equally weighted")
    for j, (vave, alpha) in enumerate(zip(VAVES, (1.0, 0.70, 0.45))):
        v = [val(k, f"{q}_vs_GSPI_pct", vave) for k in have]
        ax.bar(x - 0.08 + 0.20 * j, v, 0.19, color=[PICK[k][1] for k in have], alpha=alpha)
    ax.set_xticks(x)
    ax.set_xticklabels([PICK[k][0] for k in have], fontsize=7.5)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_title(name)
    if q == "tower":
        ax.set_ylabel("reduction vs the shipped controller, %")
        h = [plt.Rectangle((0, 0), 1, 1, color="0.78")]
        h += [plt.Rectangle((0, 0), 1, 1, color="0.35", alpha=al) for al in (1.0, 0.70, 0.45)]
        ax.legend(h, ["equally weighted", "Rayleigh, $V_{ave}$ = 10", "8.5", "7.5"], frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(f"{a.out}/f5_weighting.png", bbox_inches="tight")
print(f"-> {a.out}/f5_weighting.png")
for k in have:
    print(f"{k:>44}  tower eq {val(k, 'tower_equal_mean_pct', 10):5.1f}  Rayleigh "
          + " / ".join(f"{val(k, 'tower_vs_GSPI_pct', v):5.1f}" for v in VAVES)
          + f"   speed eq {val(k, 'speed MSE_equal_mean_pct', 10):5.1f}  Rayleigh "
          + " / ".join(f"{val(k, 'speed MSE_vs_GSPI_pct', v):5.1f}" for v in VAVES))
