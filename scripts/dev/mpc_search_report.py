"""Report of the verified MPC-parameter search on 600 s episodes (scripts/mpc_param_search.py).

(1) search curves: best supervisor-wind J so far against verified candidates, per proposer
(2) the selected candidates (each arm's top 3) and the references on both held-out sets, exact model and
    Cp/Ct x0.95 in the MPC's model; the four terms on wind seeds 3-6
(3) does the supervisor-wind ranking survive on held-out winds? (Spearman over every candidate evaluated on both)

    WTRL_HOME=~/wtrl600 ~/wtrl/run.sh python scripts/dev/mpc_search_report.py [--csv ...] [--fig ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.mpc_param_search import INCUMBENT0, key_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.expanduser("~/wtrl/exp/mpcsearch600"))
ap.add_argument("--csv", default=None)
ap.add_argument("--fig", default=None)
a = ap.parse_args()
R = Path(a.root)
K = ("J_power_mse_red_pct", "J_gen_speed_mse_red_pct", "J_TwrBsMyt_DEL_red_pct", "J_RootMyc1_DEL_red_pct")


def load(p):
    try:
        return json.load(open(p))
    except FileNotFoundError:
        return None


def cached(params, cp, seeds):
    return load(R / "cache" / f"eval_{key_of(params)}_cp{cp:g}_s{''.join(map(str, seeds))}.json")


def fmtJ(j):
    return f"{j['J']:7.2f}" if j else "      -"


def fmt_terms(j):
    return " / ".join(f"{j[k]:5.1f}" for k in K) if j else "-"


arms = ("llm", "es", "random")
hist = {arm: [json.loads(l) for l in open(R / arm / "history.jsonl")] for arm in arms if (R / arm / "history.jsonl").exists()}

print("(1) best supervisor-wind J after n verified candidates (600 s, seeds 1-2)")
curves = {}
for arm, H in hist.items():
    best, c = -1e9, []
    for h in H[1:]:
        best = max(best, h["result"]["J"], H[0]["result"]["J"])
        c.append(best)
    curves[arm] = c
    marks = [c[i - 1] for i in (8, 16, 24, 32, 40) if len(c) >= i]
    print(f"  {arm:>7}: " + "  ".join(f"n={n}: {v:.2f}" for n, v in zip((8, 16, 24, 32, 40), marks))
          + f"   failed/unstable candidates (J <= 0): {sum(h['result']['J'] <= 0 for h in H[1:])}/{len(H) - 1}")

rows = []
print("\n(2) held-out, 600 s (J: supervisor winds | seeds 3-6 | seeds 7-10 | x0.95 seeds 3-6 | x0.95 seeds 7-10; terms on seeds 3-6)")
refs = {"nominal MPC": "nominal", "adaptive MPC (reference weights)": "rls"}
for label, tag in refs.items():
    s12 = load(R / "refs" / f"eval_{tag}_cp1_s12.json")
    h1, h2 = load(R / "refs" / f"eval_{tag}_cp1_s3456.json"), load(R / "refs" / f"eval_{tag}_cp1_s78910.json")
    m1, m2 = load(R / "refs" / f"eval_{tag}_cp0.95_s3456.json"), load(R / "refs" / f"eval_{tag}_cp0.95_s78910.json")
    print(f"  {label:>44}: {fmtJ(s12)} | {fmtJ(h1)} | {fmtJ(h2)} | {fmtJ(m1)} | {fmtJ(m2)}   {fmt_terms(h1)}")
    rows.append({"row": label, "sup": s12 and s12["J"], "S3-6": h1 and h1["J"], "S7-10": h2 and h2["J"],
                 "x0.95_S3-6": m1 and m1["J"], "x0.95_S7-10": m2 and m2["J"]})
cands = [("offset-free MPC (search start)", INCUMBENT0)]
for arm, H in hist.items():
    top = sorted(H, key=lambda h: -h["result"]["J"])[:3]
    for i, h in enumerate(top):
        cands.append((f"{arm} #{i + 1} {h['params']}", h["params"]))
seen = set()
for label, p in cands:
    k = key_of(p)
    if k in seen:
        continue
    seen.add(k)
    s12, h1, h2 = cached(p, 1.0, [1, 2]), cached(p, 1.0, [3, 4, 5, 6]), cached(p, 1.0, [7, 8, 9, 10])
    m1, m2 = cached(p, 0.95, [3, 4, 5, 6]), cached(p, 0.95, [7, 8, 9, 10])
    print(f"  {label[:44]:>44}: {fmtJ(s12)} | {fmtJ(h1)} | {fmtJ(h2)} | {fmtJ(m1)} | {fmtJ(m2)}   {fmt_terms(h1)}")
    rows.append({"row": label, "params": json.dumps(p), "sup": s12 and s12["J"], "S3-6": h1 and h1["J"],
                 "S7-10": h2 and h2["J"], "x0.95_S3-6": m1 and m1["J"], "x0.95_S7-10": m2 and m2["J"]})

both = [r for r in rows if r.get("params") and r["sup"] is not None and r["S3-6"] is not None and r["S7-10"] is not None]
if len(both) >= 5:
    from scipy import stats
    sup = [r["sup"] for r in both]
    ho = [(r["S3-6"] + r["S7-10"]) / 2 for r in both]
    rho = stats.spearmanr(sup, ho)
    print(f"\n(3) supervisor-wind J vs mean held-out J over {len(both)} selected candidates: Spearman {rho.statistic:+.2f} (p {rho.pvalue:.3f});"
          f" mean optimism {np.mean(np.array(sup) - np.array(ho)):+.2f}")

if a.csv:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("->", a.csv)
if a.fig and curves:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    names = {"llm": "LLM proposals", "es": "local search around the incumbent", "random": "uniform random"}
    for arm, c in curves.items():
        ax.plot(range(1, len(c) + 1), c, marker="o", ms=2.5, label=names.get(arm, arm))
    j0 = hist[next(iter(hist))][0]["result"]["J"]
    ax.axhline(j0, color="0.5", lw=0.7, ls="--", label="search start (offset-free MPC)")
    ax.set_xlabel("verified candidates"); ax.set_ylabel("best J on the supervisor winds (600 s)")
    ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(a.fig, dpi=180)
    print("->", a.fig)
