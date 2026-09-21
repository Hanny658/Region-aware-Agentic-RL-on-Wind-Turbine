"""GO / NO-GO for extending the MPC-stacking probe (campaign_probe_mpc.sh) with more seeds. The rule was fixed on
2026-09-21 22:20, before any result of the probe existed (user instruction: more seeds if the effect is good, otherwise
examine the agent design):

  GO  iff  (a) the held-out Cw at 150 s against the MPC is positive in at least 2 of the 3 agent-reward seeds, AND
           (b) on the 600 s range set (TurbSim seeds 3-6, 28 episodes) the mean difference in J between the agent runs
               and the MPC alone exceeds +0.5, with the paired 95 % interval above zero in at least 2 of the 3 seeds.

Prints GO or NO-GO on the last line, writes <exp>/probe_decision.json with the numbers.

    ~/wtrl/run.sh python scripts/dev/probe_decision.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser("~/wtrl/exp"))
ap.add_argument("--arm", default="mrwCwR3t")
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--n_boot", type=int, default=10000)
a = ap.parse_args()
T = ("power_mse_red_pct", "gen_speed_mse_red_pct", "TwrBsMyt_DEL_red_pct", "RootMyc1_DEL_red_pct")


def J(recs):
    t = []
    for m in T:
        x = np.array([np.nan if r.get(m) is None else r[m] for r in recs], float)
        x = np.clip(x[np.isfinite(x)], -100, 100)
        t.append(x.mean())
    e, eb = sum(r["energy_MWh"] for r in recs), sum(r["energy_base_MWh"] for r in recs)
    return float(np.mean(t) - 20 * max(0.0, 100 * (1 - e / eb) - 1.0)), t


ref = json.load(open(f"{a.exp}/mpcsearch600/qtsched/eval_off_q6_18_22_s3456.json"))["per_episode"]
strata = {}
for i, r in enumerate(ref):
    strata.setdefault(r["mean_wind"], []).append(i)
rng = np.random.default_rng(0)
out = {"rule": "GO iff >= 2/3 held-out Cw > 0 and mean 600 s J difference > 0.5 with >= 2/3 paired intervals above zero", "runs": {}}
n_cw, n_ci, diffs = 0, 0, []
for s in a.seeds:
    d = f"{a.exp}/{a.arm}_s{s}"
    rec = {}
    h = f"{d}/eval_heldout_s3456_ckpt_best.json"
    if os.path.exists(h):
        H = json.load(open(h))
        rec.update(heldout_Cw=round(H["Cw"], 3), heldout_violation_pct=round(H["Cw_violation_pct"], 3),
                   heldout_terms=[round(H["J_" + t], 2) for t in T])
        n_cw += H["Cw"] > 0
    r6 = f"{d}/eval_range_TIB_s3456.json"
    if os.path.exists(r6):
        A = json.load(open(r6))["per_episode"]
        assert len(A) == len(ref) and all(x["mean_wind"] == y["mean_wind"] and abs(x["energy_base_MWh"] - y["energy_base_MWh"]) < 1e-9 for x, y in zip(A, ref))
        boot = np.empty(a.n_boot)
        for i in range(a.n_boot):
            pick = [k for ks in strata.values() for k in rng.choice(ks, size=len(ks), replace=True)]
            boot[i] = J([A[k] for k in pick])[0] - J([ref[k] for k in pick])[0]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        diff = J(A)[0] - J(ref)[0]
        rec.update(range600_J=round(J(A)[0], 2), diff_to_mpc=round(diff, 3), ci=[round(float(lo), 3), round(float(hi), 3)],
                   range600_terms=[round(x, 2) for x in J(A)[1]])
        diffs.append(diff)
        n_ci += lo > 0
    out["runs"][f"{a.arm}_s{s}"] = rec
    print(f"{a.arm}_s{s}: {rec}")
mean_diff = float(np.mean(diffs)) if diffs else float("nan")
go = bool(n_cw >= 2 and len(diffs) >= 2 and mean_diff > 0.5 and n_ci >= 2)
out.update(mpc_alone_J=round(J(ref)[0], 2), n_heldout_positive=int(n_cw), n_interval_above_zero=int(n_ci), mean_diff_to_mpc=round(mean_diff, 3),
           decision="GO" if go else "NO-GO")
json.dump(out, open(f"{a.exp}/probe_decision.json", "w"), indent=1)
print(f"MPC alone J {out['mpc_alone_J']}; held-out Cw positive {n_cw}/{len(a.seeds)}; mean 600 s difference {mean_diff:+.2f}; intervals above zero {n_ci}/{len(diffs)}")
print("GO" if go else "NO-GO")
