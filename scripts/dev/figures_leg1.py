"""Leg-1 figures: the tuning front against actuator duty, and the search traces (2026-10-15).

Figures F2 and F3 of `docs/paper/NARRATIVE.md`, the two the manuscript needs and did not have. Both are built
from the campaign's own history files, so every point is a tuning file the controller can take and every
number is the one the tables were made from - nothing is re-evaluated and no simulation is run.

f5_front.png   (a) the design screen: every evaluated tuning as regulation reduction against pitch-travel
               ratio, split into the ones the plant could not run, the ones holding the constraints on the
               set mean, and the ones also holding both loads at every wind speed; the shipped controller at
               the origin of both axes by construction. (b) the same x axis, but the y axis is the regulation
               measured on FRESH seeds for the ten band members s55 verified off the design winds, filled
               where the member holds both loads at every wind speed on 28 held-out AND 28 fresh episodes.
               Panel (a) is a screen on 14 episodes; panel (b) is the measurement.
f5_search.png  one panel per proposer: cumulative best feasible regulation against the simulation index, one
               line per run, the first feasible gain marked. The random arm never arrives, which is the
               honest form of "fewer simulations".

    ~/wtrl/run.sh python scripts/dev/figures_leg1.py [--out docs/manuscript/figures]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from eval.selection import PER_WIND_TOL, per_wind_rank_key, worst_per_wind  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.expanduser("~/wtrl/exp/roscosearch_native"))
ap.add_argument("--band", default=os.path.join(REPO, "docs", "tables", "rosco_band_verified.csv"))
ap.add_argument("--out", default=os.path.join(REPO, "docs", "manuscript", "figures"))
ap.add_argument("--tol", type=float, default=PER_WIND_TOL)
ap.add_argument("--budget", type=int, default=24)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "figure.dpi": 150, "savefig.dpi": 220, "axes.spines.top": False,
                     "axes.spines.right": False})
ARMS = ("random", "local", "bayes", "llm")
ARM_N = {"random": "random draws", "local": "local search", "bayes": "Bayesian optimisation",
         "llm": "LLM agent"}
CLR = {"random": "#8C8C8C", "local": "#4C72B0", "bayes": "#DD8452", "llm": "#C44E52"}

# ------------------------------------------------------------------ the histories, as the tables read them
runs: dict[str, list[dict]] = {}
for h in sorted(glob.glob(f"{a.root}/*/history.jsonl")):
    name = os.path.basename(os.path.dirname(h))
    if "_r" not in name:                                   # _superseded and _verify are not runs
        continue
    runs[name] = [json.loads(line) for line in open(h)]
if not runs:
    sys.exit(f"no histories under {a.root}")


def arm_of(name: str) -> str:
    return name.rsplit("_r", 1)[0]


def usable(rec: dict) -> bool:
    return bool(rec["assess"].get("constraints"))


# one entry per distinct tuning: the same tuning proposed by two runs is one point on the front
seen: dict[str, dict] = {}
for name, hist in runs.items():
    for rec in hist:
        if rec["i"] == 0:                                  # the shipped tuning itself, the identity check
            continue
        seen.setdefault(json.dumps(rec["params"], sort_keys=True), rec)
cands = list(seen.values())

ran = [r for r in cands if usable(r)]
set_ok = [r for r in ran if r["assess"]["feasible"]]
pw_ok = [r for r in set_ok if per_wind_rank_key(r, a.tol)[0] == 1]
set_only = [r for r in set_ok if r not in pw_ok]
infeasible = [r for r in ran if not r["assess"]["feasible"]]

print(f"{len(runs)} runs, {len(cands)} distinct PROPOSED tunings ({len(cands) + 1} including the shipped "
      f"tuning itself, which every run evaluates as its identity check and which the tables count), "
      f"{len(cands) - len(ran)} the plant could not run, {len(set_ok)} hold the set mean, {len(pw_ok)} also "
      f"hold both loads at every wind speed.")

# ------------------------------------------------------------------ F2: the front against duty
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 3.6))


def xy(recs):
    return ([r.get("travel_ratio", float("nan")) for r in recs],
            [r["assess"]["objective"] for r in recs])


axa.axhline(0.0, lw=0.7, color="#999999", zorder=1)
axa.axvline(1.0, lw=0.7, ls=":", color="#999999", zorder=1)
axa.scatter(*xy(infeasible), s=9, c="#CCCCCC", marker="o", lw=0, zorder=2,
            label=f"loses a constraint on the set mean ({len(infeasible)})")
axa.scatter(*xy(set_only), s=15, facecolors="none", edgecolors="#4C72B0", lw=0.7, zorder=3,
            label=f"holds it on the set mean ({len(set_only)})")
axa.scatter(*xy(pw_ok), s=20, c="#55A868", marker="o", lw=0, zorder=4,
            label=f"holds both loads at every wind speed ({len(pw_ok)})")
axa.scatter([1.0], [0.0], s=52, marker="*", c="black", zorder=6, label="shipped controller")
# the box is real: half of what the proposers can write is far worse than shipped, so both axes carry their
# full range - log in duty, symmetric-log in regulation, linear within +-50 points where the front lives
axa.set_xscale("log")
axa.set_yscale("symlog", linthresh=50)
axa.set_xlim(0.3, 22)
axa.set_ylim(-3e4, 60)
axa.set_xticks([0.5, 1, 2, 5, 10, 20])
axa.set_xticklabels(["0.5x", "1x", "2x", "5x", "10x", "20x"])
axa.set_yticks([-1e4, -1e3, -1e2, 0, 50])
axa.set_yticklabels(["-10000", "-1000", "-100", "0", "+50"])
axa.set_xlabel("pitch travel, ratio to the shipped controller")
axa.set_ylabel("generator-speed MSE reduction, %")
axa.set_title(f"(a) the design screen: {len(cands)} distinct proposed tunings, 14 episodes each")
axa.legend(loc="lower left", frameon=False, labelspacing=0.6, borderpad=0.2)
axa.annotate(f"{len(cands) - len(ran)} more the plant could not run at all",
             (0.98, 0.04), xycoords="axes fraction", fontsize=7, color="#777777", ha="right")

# ------------------------------------------------------------------ the verified members, measured off design
band = []
if os.path.exists(a.band):
    with open(a.band) as f:
        band = list(csv.DictReader(f))
else:
    print(f"! {a.band} missing - panel (b) will be empty")


def fnum(row, k):
    try:
        return float(row[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def truthy(row, k):
    return str(row.get(k, "")).strip().lower() == "true"


holds = [r for r in band if truthy(r, "holds_per_wind")]
fails = [r for r in band if not truthy(r, "holds_per_wind")]
axb.axhline(0.0, lw=0.7, color="#999999", zorder=1)
axb.axvline(1.0, lw=0.7, ls=":", color="#999999", zorder=1)
axb.scatter([fnum(r, "travel_ratio") for r in fails], [fnum(r, "fresh_regulation") for r in fails],
            s=36, facecolors="none", edgecolors="#C44E52", lw=1.2, zorder=3,
            label=f"loses a load at one wind speed ({len(fails)})")
axb.scatter([fnum(r, "travel_ratio") for r in holds], [fnum(r, "fresh_regulation") for r in holds],
            s=36, c="#55A868", marker="o", lw=0, zorder=4,
            label=f"holds both loads at every wind speed ({len(holds)})")
axb.scatter([1.0], [0.0], s=52, marker="*", c="black", zorder=6, label="shipped controller")

# the fixed-recipe baseline row, which is not an agent arm and is the reference the rule selects
base = os.path.join(REPO, "docs", "tables", "rosco_tuned_baseline.csv")
if os.path.exists(base):
    with open(base) as f:
        for row in csv.DictReader(f):
            bx, by = fnum(row, "travel_ratio"), fnum(row, "fresh_regulation")
            axb.scatter([bx], [by], s=44, marker="D", c="#333333", zorder=5,
                        label="stage 1, the fixed recipe")
            axb.annotate("stage 1 (fixed recipe,\nnot an agent arm)", (bx, by), textcoords="offset points",
                         xytext=(13, 9), fontsize=7, color="#333333", ha="left",
                         arrowprops=dict(arrowstyle="-", lw=0.6, color="#777777"))

LBL = {"llm_r1#9": ("one line of the shipped file\n(corner 1.57 -> 3.0), 1.00x travel", (13, -3), "left"),
       "llm_r1#11": ("the same line at 6.5", (-13, -13), "right"),
       "local_r0#22": ("sub-shipped duty, 0.97x", (13, -2), "left"),
       "llm_r1#12": ("the most regulating row:\n-1.17 at 24 m/s", (-13, -5), "right")}
for r in band:
    key = r.get("from", "")
    if key in LBL:
        txt, off, ha = LBL[key]
        axb.annotate(txt, (fnum(r, "travel_ratio"), fnum(r, "fresh_regulation")), textcoords="offset points",
                     xytext=off, fontsize=7, color="#333333", ha=ha,
                     arrowprops=dict(arrowstyle="-", lw=0.6, color="#777777"))
axb.set_xlim(0.78, 2.9)
axb.set_xlabel("pitch travel, ratio to the shipped controller")
axb.set_ylabel("generator-speed MSE reduction, fresh seeds, %")
axb.set_title("(b) off the design winds: 28 held-out + 28 fresh episodes")
axb.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(f"{a.out}/f5_front.png", bbox_inches="tight")
print(f"-> {a.out}/f5_front.png")

# ------------------------------------------------------------------ F3: the search traces
def trace(hist: list[dict]) -> tuple[list[int], list[float], int | None]:
    """Cumulative best FEASIBLE regulation against the simulation index, and the first feasible gain."""
    xs, ys, first, best = [], [], None, 0.0
    for rec in sorted(hist, key=lambda r: r["i"]):
        if rec["i"] == 0:
            continue
        o = rec["assess"].get("objective", float("nan"))
        if usable(rec) and rec["assess"].get("feasible") and o == o and o > best:
            best = o
            if first is None:
                first = rec["i"]
        xs.append(rec["i"])
        ys.append(best)
    return xs, ys, first

fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.9), sharex=True, sharey=True)
summary = {}
for ax, arm in zip(axes, ARMS):
    names = sorted(n for n in runs if arm_of(n) == arm)
    firsts = []
    for n in names:
        xs, ys, first = trace(runs[n])
        ax.step(xs, ys, where="post", lw=1.1, color=CLR[arm], alpha=0.85)
        if first is not None:
            k = xs.index(first)
            ax.plot([first], [ys[k]], "o", ms=4.0, color=CLR[arm], zorder=4)
        firsts.append(first if first is not None else a.budget + 1)
    got = sum(1 for f in firsts if f <= a.budget)
    med = statistics.median(firsts) if firsts else float("nan")
    summary[arm] = (got, len(names), med)
    # a censored run enters the median at budget + 1, as s59.1 fixed in advance; say so rather than printing
    # a median that reads like a run that arrived
    if got == 0:
        sub = f"{got} of {len(names)} runs, no feasible gain in {a.budget} simulations"
    else:
        cens = "" if got == len(names) else f", {len(names) - got} censored"
        sub = f"{got} of {len(names)} runs, median {med:g} simulations{cens}"
    ax.set_title(f"{ARM_N[arm]}\n{sub}")
    ax.set_xlabel("simulations")
    ax.set_xlim(0, a.budget + 1)
axes[0].set_ylabel("best feasible speed-MSE reduction, %")
for ax in axes:
    ax.axhline(0.0, lw=0.7, color="#999999")
fig.tight_layout()
fig.savefig(f"{a.out}/f5_search.png", bbox_inches="tight")
print(f"-> {a.out}/f5_search.png")

# ------------------------------------------------------------------ self-check against the published table
print("\nFirst feasible gain per run, against docs/tables/rosco_proposer_test.csv "
      "(a mismatch means the figure and the table disagree and the figure is wrong):")
pub = {}
pt = os.path.join(REPO, "docs", "tables", "rosco_proposer_test.csv")
if os.path.exists(pt):
    with open(pt) as f:
        for row in csv.DictReader(f):
            pub[f"{row['proposer']}_r{row['repeat']}"] = row["setmean_sims_to_first"]
bad = 0
for n in sorted(runs):
    _, _, first = trace(runs[n])
    mine = "" if first is None else str(first)
    theirs = pub.get(n, "?")
    ok = mine == theirs
    bad += 0 if ok else 1
    print(f"  {n:>12}  figure {mine or 'censored':>8}  table {theirs or 'censored':>8}  "
          f"{'ok' if ok else 'MISMATCH'}")
PUBLISHED = {"llm": (5, 10.0), "bayes": (5, 18.0), "local": (4, 14.0), "random": (0, 25.0)}  # s59.4
for arm in ARMS:
    got, n, med = summary[arm]
    p_got, p_med = PUBLISHED[arm]
    ok = (got == p_got) and abs(med - p_med) < 1e-9
    bad += 0 if ok else 1
    print(f"{ARM_N[arm]:>22}: {got} of {n} runs reached a feasible gain, median {med:g} simulations "
          f"(censored runs enter at {a.budget + 1}) - s59.4 says {p_got}, {p_med:g}  "
          f"{'ok' if ok else 'MISMATCH'}")
print("MISMATCHES:", bad)
if bad:
    sys.exit("the figures disagree with the published table; do not use them")
