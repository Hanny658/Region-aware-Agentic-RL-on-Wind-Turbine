"""Proposer comparison of the verified MPC design search (scripts/mpc_design_search.py): one table, no simulations.

Selection side (range seeds 1-2 at 12 / 16 / 20 / 24 m/s, 600 s): per arm-run the incumbent S after 8 / 16 / 24 verified
candidates, how many candidates beat the hand-set reference, how many failed, and the best parameters.
Held-out side (range seeds 3-6 at all seven wind speeds, 28 episodes, never seen by a proposer): J, per-wind worst
term, pitch travel, the paired difference to the hand-set unscheduled reference and to the hand-designed scheduled
controller of roadmap s33, next to the nominal MPC and the tuned ROSCO on the same episodes.

    ~/wtrl/run.sh python scripts/dev/design_search_table.py [--csv docs/tables/design_search.csv]
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
ap.add_argument("--n_boot", type=int, default=10000)
a = ap.parse_args()
ROOT = f"{a.exp}/mpcdesign"
T = ("power_mse_red_pct", "gen_speed_mse_red_pct", "TwrBsMyt_DEL_red_pct", "RootMyc1_DEL_red_pct")


def J(recs):
    t = []
    for m in T:
        x = np.array([np.nan if r.get(m) is None else r[m] for r in recs], float)
        x = np.clip(x[np.isfinite(x)], -100, 100)
        t.append(x.mean())
    e, eb = sum(r["energy_MWh"] for r in recs), sum(r["energy_base_MWh"] for r in recs)
    return float(np.mean(t) - 20 * max(0.0, 100 * (1 - e / eb) - 1.0)), t


print("selection (S = J - 4 V on range seeds 1-2, 8 episodes)")
print(f"{'arm-run':>10} {'ref S':>6} | incumbent S after 8 / 16 / 24 | beat ref | failed | best: J, V, travel, parameters")
sel = {}
for hp in sorted(glob.glob(f"{ROOT}/*_r?/history.jsonl")):
    name = os.path.basename(os.path.dirname(hp))
    h = [json.loads(l) for l in open(hp) if l.strip()]
    ref, cand = h[0]["result"]["S"], h[1:]
    inc = [max([ref] + [c["result"]["S"] for c in cand[:k]]) for k in (8, 16, 24)]
    b = max(h, key=lambda r: r["result"]["S"])
    q, r = b["params"], b["result"]
    sel[name] = {"arm_run": name, "reference_S": ref, "S_after_8": inc[0], "S_after_16": inc[1], "S_after_24": inc[2], "n_candidates": len(cand),
                 "n_beat_reference": sum(c["result"]["S"] > ref for c in cand), "n_failed": sum(c["result"]["J"] <= -99.9 for c in cand),
                 "best_J_sel": r["J"], "best_V_sel": r["V"], "best_travel_sel": r["travel_x_baseline"], **{f"p_{k}": v for k, v in q.items()}}
    print(f"{name:>10} {ref:6.2f} | {inc[0]:6.2f} / {inc[1]:6.2f} / {inc[2]:6.2f} | {sel[name]['n_beat_reference']:3d}/{len(cand):<3d} | {sel[name]['n_failed']:3d} | "
          f"J {r['J']:.2f} V {r['V']:.2f} x{r['travel_x_baseline']:.2f}  N{q['horizon']} r{q['r']} qt{q['qt']} x{q['qt_ratio']}@{q['v_lo']}+{q['v_span']} wc{q['wc_v']} {q['adapt']} tau{q['tau_adapt']}")

# ---------------------------------------------------------------- held-out
rows = {}
for p in sorted(glob.glob(f"{ROOT}/heldout/eval_*_s3456.json")):
    rows[os.path.basename(p)[5:-11]] = json.load(open(p))["per_episode"]
R6 = f"{a.exp}/mpcsearch600"
refs = {"hand-set reference (unscheduled offset-free)": (f"{R6}/range/eval_offset_cp1.json", slice(14, 42)),
        "hand-designed schedule (s33)": (f"{R6}/qtsched/eval_off_q6_18_22_s3456.json", None),
        "nominal MPC": (f"{R6}/range/eval_nominal_cp1.json", slice(14, 42)),
        "tuned ROSCO": (f"{R6}/range/eval_rosco_tuned.json", slice(14, 42))}
for k, (p, sl) in refs.items():
    if os.path.exists(p):
        per = json.load(open(p))["per_episode"]
        rows[k] = per[sl] if sl else per
if rows:
    base = rows["hand-set reference (unscheduled offset-free)"]
    winds = sorted({r["mean_wind"] for r in base})
    strata = {}
    for i, r in enumerate(base):
        strata.setdefault(r["mean_wind"], []).append(i)
    rng = np.random.default_rng(0)

    def paired(A, B):
        d = np.empty(a.n_boot)
        for i in range(a.n_boot):
            pick = [k for ks in strata.values() for k in rng.choice(ks, size=len(ks), replace=True)]
            d[i] = J([A[k] for k in pick])[0] - J([B[k] for k in pick])[0]
        return J(A)[0] - J(B)[0], np.percentile(d, [2.5, 97.5])

    print("\nheld-out (range seeds 3-6, seven wind speeds, 28 episodes, 600 s)")
    print(f"{'controller':>46} {'J':>6} {'xGSPI':>6} | power / speed / tower / blade | tower per wind | worst term | J - hand-set ref [95 %] | J - s33 schedule [95 %]")
    out = []
    for k, v in rows.items():
        assert len(v) == 28 and all(x["mean_wind"] == y["mean_wind"] and abs(x["energy_base_MWh"] - y["energy_base_MWh"]) < 1e-9 for x, y in zip(v, base)), k
        j, t = J(v)
        trav = np.mean([r["pitch_travel_deg"] for r in v]) / np.mean([r["pitch_travel_base_deg"] for r in v])
        tw, worst, short = [], (0.0, ""), 0.0
        for u in winds:
            tu = J([r for r in v if r["mean_wind"] == u])[1]
            tw.append(tu[2])
            short += sum(max(0.0, -1.0 - x) for x in tu if np.isfinite(x))
            for nm, val in zip(("power", "speed", "tower", "blade"), tu):
                if np.isfinite(val) and val < worst[0]:
                    worst = (val, f"{nm}@{u:g}")
        V = short / len(winds)                      # the per-wind load rule of the design objective, on the held-out episodes
        d1, (l1, h1) = paired(v, base)
        d2, (l2, h2) = paired(v, rows["hand-designed schedule (s33)"]) if "hand-designed schedule (s33)" in rows else (float("nan"), (float("nan"),) * 2)
        print(f"{k:>46} {j:6.2f} {trav:6.2f} S {j - 4 * V:6.2f} V {V:4.2f} | " + " / ".join(f"{x:5.1f}" for x in t) + " | " + " ".join(f"{x:5.1f}" for x in tw) +
              f" | {worst[0]:5.1f} {worst[1]:<9} | {d1:+5.2f} [{l1:+.2f}, {h1:+.2f}] | {d2:+5.2f} [{l2:+.2f}, {h2:+.2f}]")
        out.append({"controller": k, "heldout_S": round(j - 4 * V, 2), "heldout_V": round(V, 3), "heldout_J": round(j, 2), "travel_x_gspi": round(float(trav), 3), **{f"heldout_{m}": round(x, 2) for m, x in zip(T, t)},
                    **{f"tower_U{u:g}": round(x, 2) for u, x in zip(winds, tw)}, "worst_term": round(worst[0], 2), "worst_at": worst[1],
                    "diff_handset": round(d1, 2), "diff_handset_lo": round(float(l1), 2), "diff_handset_hi": round(float(h1), 2),
                    "diff_s33": round(d2, 2), "diff_s33_lo": round(float(l2), 2), "diff_s33_hi": round(float(h2), 2), **sel.get(k, {})})
    if a.csv:
        keys = list(dict.fromkeys(kk for r in out for kk in r))
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
        print("->", a.csv)
