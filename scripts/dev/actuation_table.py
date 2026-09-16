"""Actuator cost of every controller, from the evaluation CSVs already on disk (no simulations).

The literature scan (docs/literature_2026-09-16.md, items 12 and 16) reports fatigue reductions jointly with
pitch travel / actuator duty cycle, and treats a load gain bought with actuation as suspect. Every eval_*.csv
already carries pitch_travel_deg per episode and the paired GSPI baseline carries pitch_travel_base_deg in the
eval_*.json per_episode records, so the actuator columns come for free:

    pitch travel   : total |d beta| over the scored window [deg], and the ratio to the paired GSPI run
    duty cycle     : pitch travel per scored second [deg/s], the normalised measure used in the IPC literature
    residual usage : mean |d beta| of the learned residual itself [deg] (0 for the reference controllers)

    ~/wtrl/run.sh python scripts/dev/actuation_table.py [--csv docs/tables/actuation.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from statistics import mean

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser("~/wtrl/exp"))
ap.add_argument("--csv", default=None)
a = ap.parse_args()
E = a.exp

ROWS = [  # label, path glob of the eval json (one per run), scored window [s]
    ("GSPI (pairing reference)", [], 130.0),
    ("LPV-MPC, nominal", [f"{E}/mpc/eval_base0_s3456.json"], 130.0),
    ("LPV-MPC, offset-free", [f"{E}/mpc/eval_base0_offset_cp1_s3456.json"], 130.0),
    ("LPV-MPC, adaptive (RLS)", [f"{E}/mpc/eval_base0_rls_cp1_s3456.json"], 130.0),
    ("LPV-MPC, nominal, Cp x0.95", [f"{E}/mpc/eval_base0_cp0.95_s3456.json"], 130.0),
    ("LPV-MPC, offset-free, Cp x0.95", [f"{E}/mpc/eval_base0_offset_cp0.95_s3456.json"], 130.0),
    ("GSPI base + fixed-reward residual", sorted(glob.glob(f"{E}/jg3t_s*/eval_heldout_s3456_ckpt_best.json")), 130.0),
    ("GSPI base + agent-reward residual", sorted(glob.glob(f"{E}/jrw3t_s*/eval_heldout_s3456_ckpt_best.json")), 130.0),
    ("MPC base + fixed-reward residual", sorted(glob.glob(f"{E}/mg3t_s*/eval_heldout_s3456_ckpt_best.json")), 130.0),
    ("MPC base + agent-reward residual", sorted(glob.glob(f"{E}/mrw3t_s*/eval_heldout_s3456_ckpt_best.json")), 130.0),
    ("x0.95 MPC base + fixed-reward residual", sorted(glob.glob(f"{E}/mgC3t_s*/eval_heldout_s3456_ckpt_best.json")), 130.0),
    ("x0.95 MPC base + agent-reward residual", sorted(glob.glob(f"{E}/mrwC3t_s*/eval_heldout_s3456_ckpt_best.json")), 130.0),
    ("x0.95 offset-free base + fixed-reward residual", sorted(glob.glob(f"{E}/moC3t_s*/eval_heldout_s3456_ckpt_best.json")), 130.0),
    ("x0.95 offset-free base + agent-reward residual", sorted(glob.glob(f"{E}/morC3t_s*/eval_heldout_s3456_ckpt_best.json")), 130.0),
    ("offset-free base + fixed-reward residual", sorted(glob.glob(f"{E}/mo3t_s*/eval_heldout_s3456_ckpt_best.json")), 130.0),
]


def stats(path):
    """-> (pitch travel [deg], baseline pitch travel [deg], residual |dbeta| [deg], J) averaged over episodes."""
    j = json.load(open(path))
    per = j.get("per_episode") or []
    trav = [p["pitch_travel_deg"] for p in per if "pitch_travel_deg" in p]
    base = [p["pitch_travel_base_deg"] for p in per if "pitch_travel_base_deg" in p]
    dres = [p.get("dbeta_abs_mean_deg") for p in per if p.get("dbeta_abs_mean_deg") is not None]
    if not trav:
        return None
    return mean(trav), (mean(base) if base else float("nan")), (mean(dres) if dres else float("nan")), j.get("J")


out = []
print(f"{'controller':>46} {'n':>2} {'pitch travel':>13} {'vs GSPI':>8} {'duty':>9} {'|dbeta|':>8} {'J':>7}")
print(f"{'':>46} {'':>2} {'[deg/episode]':>13} {'':>8} {'[deg/s]':>9} {'[deg]':>8}")
for label, paths, window in ROWS:
    vals = [stats(p) for p in paths if os.path.exists(p)]
    vals = [v for v in vals if v]
    if not vals and label.startswith("GSPI ("):
        # the pairing reference: its travel is the baseline column of any evaluation
        any_row = next((stats(p) for _, ps, _ in ROWS for p in ps if os.path.exists(p) and stats(p)), None)
        if any_row:
            b = any_row[1]
            print(f"{label:>46} {'-':>2} {b:13.1f} {1.0:8.2f} {b / window:9.3f} {0.0:8.3f} {0.0:7.2f}")
            out.append({"controller": label, "n": 0, "pitch_travel_deg": round(b, 2), "ratio_vs_gspi": 1.0,
                        "duty_deg_per_s": round(b / window, 4), "residual_dbeta_deg": 0.0, "J": 0.0})
        continue
    if not vals:
        continue
    t, b, d, J = (mean(v[0] for v in vals), mean(v[1] for v in vals),
                  mean(v[2] for v in vals), mean(v[3] for v in vals if v[3] is not None))
    print(f"{label:>46} {len(vals):2d} {t:13.1f} {t / b:8.2f} {t / window:9.3f} {d:8.3f} {J:7.2f}")
    out.append({"controller": label, "n": len(vals), "pitch_travel_deg": round(t, 2), "ratio_vs_gspi": round(t / b, 3),
                "duty_deg_per_s": round(t / window, 4), "residual_dbeta_deg": round(d, 4), "J": round(J, 2)})

print("\nPitch travel is the summed |d beta| over the scored window of one episode, averaged over the 12 held-out")
print("episodes (wind seeds 3-6 at 8 / 12.5 / 15 m/s) and over seeds; the ratio is against the paired GSPI run.")
if a.csv:
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print("->", a.csv)
