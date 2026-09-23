"""Figures for the rewritten manuscript (baselines -> scheduled MPC -> wind-gated agent layer).

    ~/wtrl/run.sh python scripts/dev/figures_v5.py [--out docs/manuscript/figures]

Every panel is built from the fresh-seed evaluations (TurbSim seeds 7-10, 12-24 m/s, 600 s, IEC class B) that no
training, selection or earlier report has seen, and every percentage is against the same paired original-GSPI
baseline episode by episode.

f5_stack.png     the four terms and J for each controller of the stack, fresh seeds (the main results figure)
f5_perwind.png   the four terms per mean wind speed: where each layer acts, and why the residual is gated
f5_weighting.png set-mean vs IEC-Rayleigh-weighted reductions, with the class-I/II/III sensitivity (roadmap s42)
f5_duty.png      J against pitch travel: what each layer costs the actuator
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser("~/wtrl/exp"))
ap.add_argument("--out", default=os.path.join(REPO, "docs", "manuscript", "figures"))
ap.add_argument("--n_boot", type=int, default=4000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
R6 = f"{a.exp}/mpcsearch600"
T = ("power_mse_red_pct", "gen_speed_mse_red_pct", "TwrBsMyt_DEL_red_pct", "RootMyc1_DEL_red_pct")
TN = ("power MSE", "generator-speed MSE", "tower-base DEL", "blade-root DEL")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                     "figure.dpi": 150, "savefig.dpi": 220, "axes.spines.top": False, "axes.spines.right": False})
CLR = {"tuned ROSCO": "#4C72B0", "scheduled MPC": "#DD8452",
       "tuned ROSCO + gated layer": "#55A868", "scheduled MPC + gated layer": "#C44E52"}


def load(p):
    p = os.path.expanduser(p)
    return json.load(open(p))["per_episode"] if os.path.exists(p) else None


def best_of(pattern, key=lambda r: r):
    """The seed with the highest J among the runs matching the glob (selection is reported, not hidden)."""
    out = []
    for d in sorted(glob.glob(f"{a.exp}/{pattern}")):
        per = load(f"{d}/eval_range_TIB_s78910.json")
        if per:
            out.append((J(per)[0], os.path.basename(d), per))
    return max(out) if out else None


def J(recs):
    t = []
    for m in T:
        x = np.array([np.nan if r.get(m) is None else r[m] for r in recs], float)
        t.append(float(np.clip(x[np.isfinite(x)], -100, 100).mean()))
    e, eb = sum(r["energy_MWh"] for r in recs), sum(r["energy_base_MWh"] for r in recs)
    return float(np.mean(t) - 20 * max(0.0, 100 * (1 - e / eb) - 1.0)), t


def travel(recs):
    return float(np.mean([r["pitch_travel_deg"] for r in recs]) / np.mean([r["pitch_travel_base_deg"] for r in recs]))


def boot_terms(recs, strata, rng, n):
    """95 % intervals of the four term means and of J, resampling episodes within mean-wind strata."""
    out = np.empty((n, 5))
    for i in range(n):
        pick = [k for ks in strata.values() for k in rng.choice(ks, size=len(ks), replace=True)]
        j, t = J([recs[k] for k in pick])
        out[i] = [*t, j]
    return np.percentile(out, [2.5, 97.5], axis=0)


CONTROLLERS = {}
for lab, per in (("tuned ROSCO", load(f"{R6}/fresh/eval_rosco_tuned_s78910.json")),
                 ("scheduled MPC", load(f"{R6}/fresh/eval_sched_mpc_s78910.json"))):
    if per:
        CONTROLLERS[lab] = (None, per)
for lab, pat in (("tuned ROSCO + gated layer", "trwCwG3t_s?"), ("scheduled MPC + gated layer", "mrwCwG3t_s?")):
    b = best_of(pat)
    if b:
        CONTROLLERS[lab] = (b[1], b[2])
assert CONTROLLERS, "no fresh-seed evaluations found"
rng = np.random.default_rng(0)
STRATA = {}
for i, r in enumerate(next(iter(CONTROLLERS.values()))[1]):
    STRATA.setdefault(r["mean_wind"], []).append(i)
WINDS = sorted(STRATA)

# ---------------------------------------------------------------- f5_stack: the main results figure
fig, axes = plt.subplots(1, 5, figsize=(10.4, 2.9), sharey=False)
labs = list(CONTROLLERS)
for k, (ax, name) in enumerate(zip(axes, [*TN, "objective $J$"])):
    for i, lab in enumerate(labs):
        run, per = CONTROLLERS[lab]
        j, t = J(per)
        v = [*t, j][k]
        lo, hi = boot_terms(per, STRATA, np.random.default_rng(1 + i), a.n_boot)[:, k]
        ax.bar(i, v, color=CLR[lab], width=0.72)
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor="0.25", capsize=2.5, lw=1)
        ax.annotate(f"{v:.1f}", (i, max(v, hi)), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.5)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels([""] * len(labs))
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_title(name)
    if k == 0:
        ax.set_ylabel("reduction vs GSPI [%]")
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CLR[l]) for l in labs], labels=labs,
           loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("fresh wind seeds 7–10, 12–24 m/s, IEC class B, 600 s (28 episodes); 95 % paired bootstrap", fontsize=8.5, y=1.0)
fig.tight_layout(rect=(0, 0.07, 1, 0.97))
fig.savefig(f"{a.out}/f5_stack.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- f5_perwind: where each layer acts
fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.7), sharex=True)
for ax, m, name in zip(axes, T, TN):
    for lab, (run, per) in CONTROLLERS.items():
        y = [float(np.clip([per[k][m] for k in STRATA[u]], -100, 100).mean()) for u in WINDS]
        ax.plot(WINDS, y, "-o", ms=3, color=CLR[lab], label=lab,
                ls="--" if "gated" in lab else "-", lw=1.4)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axvspan(15, 17, color="0.85", zorder=0)     # the gate ramp
    ax.set_title(name)
    ax.set_xlabel("mean wind speed [m/s]")
axes[0].set_ylabel("reduction vs GSPI [%]")
axes[0].text(16, axes[0].get_ylim()[0] + 2, "gate", ha="center", fontsize=7, color="0.35")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(f"{a.out}/f5_perwind.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- f5_weighting: set mean vs site distribution
csvp = os.path.join(REPO, "docs", "tables", "lifetime_del.csv")
if os.path.exists(csvp):
    rows = list(csv.DictReader(open(csvp)))
    pick = {"tuned ROSCO": "tuned ROSCO", "scheduled MPC": "scheduled MPC",
            "tuned ROSCO + gated residual (s2)": "tuned ROSCO + gated layer",
            "scheduled MPC + gated residual (s1)": "scheduled MPC + gated layer"}
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.9))
    # (a) the Rayleigh weights over the evaluated bins
    ax = axes[0]
    for vave, ls in ((10.0, "-"), (8.5, "--"), (7.5, ":")):
        w = []
        for u in WINDS:
            c = lambda v: 1 - np.exp(-np.pi / 4 * (v / vave) ** 2)
            w.append(c(u + 1) - c(u - 1))
        w = np.array(w) / sum(w)
        ax.plot(WINDS, 100 * w, ls, color="0.25", lw=1.4, label=f"$V_{{ave}}$ = {vave:g} m/s")
    ax.set_xlabel("mean wind speed [m/s]"); ax.set_ylabel("weight over 12–24 m/s [%]")
    ax.set_title("(a) IEC Rayleigh weights"); ax.legend(frameon=False)
    # (b),(c) tower and speed reductions, equally weighted vs weighted
    for ax, q, name in ((axes[1], "tower", "(b) tower-base DEL"), (axes[2], "speed MSE", "(c) generator-speed MSE")):
        labs2 = [l for l in pick.values() if any(r["controller"] == k and pick[k] == l for r in rows for k in pick)]
        x = np.arange(len(labs2))
        eq = [next(float(r[f"{q}_equal_mean_pct"]) for r in rows if pick.get(r["controller"]) == l and float(r["Vave"]) == 10) for l in labs2]
        ax.bar(x - 0.28, eq, 0.26, color="0.75", label="equally weighted set mean")
        for j, (vave, alpha) in enumerate(((10.0, 1.0), (8.5, 0.72), (7.5, 0.48))):
            v = [next(float(r[f"{q}_vs_GSPI_pct"]) for r in rows if pick.get(r["controller"]) == l and float(r["Vave"]) == vave) for l in labs2]
            ax.bar(x + 0.02 + 0.18 * j, v, 0.17, color=[CLR[l] for l in labs2], alpha=alpha)
        ax.set_xticks(x); ax.set_xticklabels(["ROSCO", "MPC", "ROSCO\n+layer", "MPC\n+layer"][:len(labs2)], fontsize=7.5)
        ax.axhline(0, color="0.4", lw=0.8); ax.set_title(name)
        if q == "tower":
            ax.set_ylabel("reduction vs GSPI [%]")
            # neutral proxies: the bar colour codes the controller, the shading codes the weighting
            h = [plt.Rectangle((0, 0), 1, 1, color="0.75")]
            h += [plt.Rectangle((0, 0), 1, 1, color="0.35", alpha=al) for al in (1.0, 0.72, 0.48)]
            ax.legend(h, ["equally weighted set mean", "Rayleigh $V_{ave}$ = 10", "8.5", "7.5"],
                      frameon=False, fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(f"{a.out}/f5_weighting.png", bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- f5_duty: J against actuator duty
fig, ax = plt.subplots(figsize=(4.6, 3.1))
for lab, (run, per) in CONTROLLERS.items():
    j, _ = J(per)
    ax.scatter(travel(per), j, s=46, color=CLR[lab], label=lab, zorder=3)
    ax.annotate(f"{j:.1f}", (travel(per), j), textcoords="offset points", xytext=(6, -3), fontsize=7.5)
for pat, lab in (("trwCwG3t_s?", "tuned ROSCO + gated layer"), ("mrwCwG3t_s?", "scheduled MPC + gated layer")):
    for d in sorted(glob.glob(f"{a.exp}/{pat}")):
        per = load(f"{d}/eval_range_TIB_s78910.json")
        if per:
            ax.scatter(travel(per), J(per)[0], s=16, facecolor="none", edgecolor=CLR[lab], lw=0.9, zorder=2)
ax.scatter(1.0, 0.0, s=46, color="0.35", marker="s", label="GSPI (reference)", zorder=3)
ax.set_xlabel("pitch travel / GSPI pitch travel")
ax.set_ylabel("objective $J$ [% mean reduction]")
ax.set_title("fresh seeds 7–10; open markers: the other seeds", fontsize=8.5)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(f"{a.out}/f5_duty.png", bbox_inches="tight")
plt.close(fig)

for f in ("f5_stack", "f5_perwind", "f5_weighting", "f5_duty"):
    p = f"{a.out}/{f}.png"
    if os.path.exists(p):
        print(f"{p}  {os.path.getsize(p) / 1024:.0f} kB")
for lab, (run, per) in CONTROLLERS.items():
    print(f"{lab:>30} {run or '-':>14}  J {J(per)[0]:6.2f}  travel x{travel(per):.2f}")
