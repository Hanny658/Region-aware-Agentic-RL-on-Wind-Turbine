"""Residual on the tuned ROSCO trained on the above-rated range winds (roadmap s34, s37): one table, no simulations.

Per run: the selected episode, the held-out objective Cw at 150 s against the tuned ROSCO (range winds, TurbSim seeds
3-6, 28 episodes) with its per-wind violation and the four set-mean terms, and the 600 s row against the ORIGINAL GSPI
on the same seeds with the paired difference to the tuned ROSCO alone (bootstrap over episodes, stratified by wind).
Per arm: how many seeds are positive on held-out Cw, the mean, and a one-sided Fisher test of the agent arm against
the pooled controls.

    ~/wtrl/run.sh python scripts/dev/range_residual_table.py [--csv docs/tables/range_residual.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from math import comb

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser("~/wtrl/exp"))
ap.add_argument("--csv", default=None)
ap.add_argument("--n_boot", type=int, default=10000)
ap.add_argument("--base", default="tuned", choices=["tuned", "mpc"],
                help="tuned: residual on the tuned ROSCO (s34, s37); mpc: residual on the scheduled MPC (s39, s40)")
a = ap.parse_args()
if a.base == "mpc":
    ARMS = {"mrwCwR3t": "agent-written reward on the scheduled MPC", "mrrCwR3t": "random reward structure on the scheduled MPC",
            "mgCwR3t": "fixed reward (J-tuned) on the scheduled MPC", "mrwCwG3t": "agent-written reward, wind-gated, on the scheduled MPC",
            "mgCwG3t": "fixed reward, wind-gated, on the scheduled MPC"}
else:
    ARMS = {"trwCwR3t": "agent-written reward", "tgCwR3L10": "fixed reward, tower weight 10",
            "tgCwR3t": "fixed reward, J-tuned weights", "trrCwR3t": "random reward structure", "trwCwG3t": "agent-written reward, wind-gated"}
T = ("power_mse_red_pct", "gen_speed_mse_red_pct", "TwrBsMyt_DEL_red_pct", "RootMyc1_DEL_red_pct")


def J(recs):
    t = []
    for m in T:
        x = np.array([np.nan if r.get(m) is None else r[m] for r in recs], float)
        x = np.clip(x[np.isfinite(x)], -100, 100)
        t.append(x.mean())
    e, eb = sum(r["energy_MWh"] for r in recs), sum(r["energy_base_MWh"] for r in recs)
    return float(np.mean(t) - 20 * max(0.0, 100 * (1 - e / eb) - 1.0)), t


if a.base == "mpc":
    ref = json.load(open(f"{a.exp}/mpcsearch600/qtsched/eval_off_q6_18_22_s3456.json"))["per_episode"]        # scheduled MPC, seeds 3-6
else:
    ref = json.load(open(f"{a.exp}/mpcsearch600/range/eval_rosco_tuned.json"))["per_episode"][14:42]   # tuned ROSCO, seeds 3-6
REF = "scheduled MPC alone" if a.base == "mpc" else "tuned ROSCO alone"
jref, tref = J(ref)
rng = np.random.default_rng(0)
strata = {}
for i, r in enumerate(ref):
    strata.setdefault(r["mean_wind"], []).append(i)


def paired(A):
    d = np.empty(a.n_boot)
    for i in range(a.n_boot):
        pick = [k for ks in strata.values() for k in rng.choice(ks, size=len(ks), replace=True)]
        d[i] = J([A[k] for k in pick])[0] - J([ref[k] for k in pick])[0]
    return np.percentile(d, [2.5, 97.5])


rows, by_arm, ci_arm = [], {}, {}
print(f"{REF}, 600 s range set seeds 3-6: J {jref:.2f} (" + " / ".join(f"{x:.1f}" for x in tref) + ")")
print(f"{'run':>13} {'ep':>4} | 150 s held-out vs the base: {'Cw':>7} {'viol':>5} | P / w / T / B | travel | 600 s vs GSPI: J, diff to {REF} [95 %], P / w / T / B, travel xGSPI, worst tower / blade diff per wind")
for arm, desc in ARMS.items():
    for d in sorted(glob.glob(f"{a.exp}/{arm}_s?")):
        run = os.path.basename(d)
        h = f"{d}/eval_heldout_s3456_ckpt_best.json"
        if not os.path.exists(h):
            continue
        H = json.load(open(h))
        per = H["per_episode"]
        trav = np.mean([q["pitch_travel_deg"] for q in per]) / np.mean([q["pitch_travel_base_deg"] for q in per])
        ep = None
        if os.path.exists(f"{d}/ckpt_best.pt"):
            import torch
            ep = torch.load(f"{d}/ckpt_best.pt", weights_only=False, map_location="cpu").get("episode")
        rec = {"run": run, "arm": desc, "selected_episode": ep, "heldout_Cw": round(H["Cw"], 3), "heldout_violation_pct": round(H["Cw_violation_pct"], 3),
               **{f"heldout_{t}": round(H["J_" + t], 2) for t in T}, "heldout_travel_x_tuned": round(float(trav), 3)}
        by_arm.setdefault(arm, []).append(H["Cw"])
        line = (f"{run:>13} {str(ep):>4} | {H['Cw']:7.2f} {H['Cw_violation_pct']:5.2f} | " + " / ".join(f"{H['J_' + t]:5.1f}" for t in T) + f" | {trav:5.2f} | ")
        r6 = f"{d}/eval_range_TIB_s3456.json"
        if os.path.exists(r6):
            A = json.load(open(r6))["per_episode"]
            assert len(A) == 28 and all(x["mean_wind"] == y["mean_wind"] and abs(x["energy_base_MWh"] - y["energy_base_MWh"]) < 1e-9 for x, y in zip(A, ref)), run
            ja, ta = J(A)
            lo, hi = paired(A)
            tr6 = np.mean([q["pitch_travel_deg"] for q in A]) / np.mean([q["pitch_travel_base_deg"] for q in A])
            wt = wb = 0.0
            for u, ks in strata.items():
                da = np.array(J([A[k] for k in ks])[1]) - np.array(J([ref[k] for k in ks])[1])
                wt, wb = min(wt, da[2]), min(wb, da[3])
            line += (f"{ja:6.2f} {ja - jref:+5.2f} [{lo:+.2f}, {hi:+.2f}] | " + " / ".join(f"{x:5.1f}" for x in ta) + f" | {tr6:4.2f} | {wt:+.1f} / {wb:+.1f}")
            ci_arm.setdefault(arm, []).append((float(lo) > 0, ja - jref))
            rec.update({"range600_J": round(ja, 2), "range600_diff_tuned": round(ja - jref, 2), "range600_ci_lo": round(float(lo), 2), "range600_ci_hi": round(float(hi), 2),
                        **{f"range600_{t}": round(x, 2) for t, x in zip(T, ta)}, "range600_travel_x_gspi": round(float(tr6), 3),
                        "range600_worst_tower_diff": round(float(wt), 2), "range600_worst_blade_diff": round(float(wb), 2)})
        print(line)
        rows.append(rec)
print()
for arm, v in by_arm.items():
    c = ci_arm.get(arm, [])
    extra = f"; 600 s: interval above zero in {sum(x for x, _ in c)}/{len(c)}, mean diff {np.mean([d for _, d in c]):+.2f}" if c else ""
    print(f"{ARMS[arm]:>52}: held-out Cw positive in {sum(x > 0 for x in v)}/{len(v)} seeds, mean {np.mean(v):+.2f}, median {np.median(v):+.2f}{extra}")
AG = "mrwCwR3t" if a.base == "mpc" else "trwCwR3t"
ag = by_arm.get(AG, [])
ct = [x for k, v in by_arm.items() if k != AG and not k.endswith("G3t") for x in v]
if ag and ct:
    k, n, K, N = sum(x > 0 for x in ag), len(ag), sum(x > 0 for x in ag + ct), len(ag) + len(ct)
    p = sum(comb(K, i) * comb(N - K, n - i) for i in range(k, min(n, K) + 1)) / comb(N, n)
    print(f"agent {k}/{n} vs pooled controls {sum(x > 0 for x in ct)}/{len(ct)} positive: one-sided Fisher exact p = {p:.3f}")
if a.csv and rows:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("->", a.csv)
