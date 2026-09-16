"""Is the MPC's pitch travel useful motion or re-solve chatter? (2026-09-16, no simulations)

Roadmap s22 found that every MPC row buys its gains with ~7.6x the GSPI pitch travel. Part of that may be
chatter from re-solving the QP every 0.1 s rather than load-driven motion. Every episode already stores two
spectral shares of the pitch-RATE power (scripts/../agents/rollout.py: episode_metrics):

    pitch_rate_power_tower_band_frac : share in 0.25-0.40 Hz, the first tower fore-aft mode (useful damping)
    pitch_rate_power_3P_band_frac    : share in 0.50-0.75 Hz, the 3P band (blade passing)

and the rest of the 0.05-5 Hz band is neither. A controller whose travel is load-driven puts its pitch-rate
power in the tower band; a chattering one spreads it over the remaining band. This script reports, per
controller, the pitch travel, the two band shares and the residual "other" share, from the evaluation CSVs.

    ~/wtrl/run.sh python scripts/dev/pitch_spectrum_table.py [--csv docs/tables/pitch_spectrum.csv]
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
BASE = os.path.expanduser("~/wtrl/baselines/openfast")

ROWS = [
    ("LPV-MPC, nominal", [f"{E}/mpc/eval_base0_s3456.csv"]),
    ("LPV-MPC, offset-free", [f"{E}/mpc/eval_base0_offset_cp1_s3456.csv"]),
    ("LPV-MPC, adaptive (RLS)", [f"{E}/mpc/eval_base0_rls_cp1_s3456.csv"]),
    ("LPV-MPC, offset-free, Cp x0.95", [f"{E}/mpc/eval_base0_offset_cp0.95_s3456.csv"]),
    ("GSPI + fixed-reward residual", sorted(glob.glob(f"{E}/jg3t_s*/eval_heldout_s3456_ckpt_best.csv"))),
    ("GSPI + agent-reward residual", sorted(glob.glob(f"{E}/jrw3t_s*/eval_heldout_s3456_ckpt_best.csv"))),
    ("MPC + fixed-reward residual", sorted(glob.glob(f"{E}/mg3t_s*/eval_heldout_s3456_ckpt_best.csv"))),
    ("MPC + agent-reward residual", sorted(glob.glob(f"{E}/mrw3t_s*/eval_heldout_s3456_ckpt_best.csv"))),
    ("x0.95 offset-free base + fixed reward", sorted(glob.glob(f"{E}/moC3t_s*/eval_heldout_s3456_ckpt_best.csv"))),
    ("x0.95 offset-free base + agent reward", sorted(glob.glob(f"{E}/morC3t_s*/eval_heldout_s3456_ckpt_best.csv"))),
]


def read(paths):
    trav, tower, p3, n = [], [], [], 0
    for p in paths:
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            if not r.get("pitch_rate_power_tower_band_frac"):
                continue
            trav.append(float(r["pitch_travel_deg"]))
            tower.append(float(r["pitch_rate_power_tower_band_frac"]))
            p3.append(float(r["pitch_rate_power_3P_band_frac"]))
            n += 1
    if not trav:
        return None
    return mean(trav), mean(tower), mean(p3), n


# the GSPI baseline's own spectrum, from the paired baseline npz files (same metric function)
gspi = None
try:
    import numpy as np
    import yaml
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from agents.rollout import episode_metrics
    from envs.base_env import PROJ
    tb = yaml.safe_load(open(PROJ / "configs" / "turbine" / "nrel5mw.yaml"))
    tr, tw, p3 = [], [], []
    for f in sorted(glob.glob(f"{BASE}/U*_TI8_S[3-6].npz")):
        d = np.load(f)
        L = {k: d[k] for k in d.files if not k.startswith("outb_")}
        L["region"] = (L["v_hub"] > float(tb["rated_wind_ms"])).astype(np.int8)
        outb = {k[5:]: d[k] for k in d.files if k.startswith("outb_")} or None
        m = episode_metrics(L, float(tb["dt_ctrl_s"]), float(tb["rated_gen_speed_rads"]), 20.0, outb)
        if "pitch_rate_power_tower_band_frac" in m:
            tr.append(m["pitch_travel_deg"]); tw.append(m["pitch_rate_power_tower_band_frac"]); p3.append(m["pitch_rate_power_3P_band_frac"])
    if tr:
        gspi = (mean(tr), mean(tw), mean(p3), len(tr))
except Exception as e:  # noqa: BLE001
    print("GSPI baseline spectrum unavailable:", type(e).__name__, e)

out = []
print(f"{'controller':>40} {'travel':>8} {'tower band':>11} {'3P band':>9} {'other':>7} {'n ep':>5}")
print(f"{'':>40} {'[deg]':>8} {'0.25-0.40Hz':>11} {'0.5-0.75':>9} {'':>7}")
rows = ([("GSPI (pairing reference)", gspi)] if gspi else []) + [(lab, read(ps)) for lab, ps in ROWS]
for lab, v in rows:
    if not v:
        continue
    trav, tw, p3v, n = v
    print(f"{lab:>40} {trav:8.1f} {tw:11.3f} {p3v:9.3f} {1 - tw - p3v:7.3f} {n:5d}")
    out.append({"controller": lab, "pitch_travel_deg": round(trav, 2), "tower_band_frac": round(tw, 4),
                "band_3P_frac": round(p3v, 4), "other_frac": round(1 - tw - p3v, 4), "episodes": n})
print("\nShares are of the pitch-rate power in 0.05-5 Hz. 'other' is everything outside the tower and 3P bands:")
print("a high 'other' share with high travel is chatter, not load-driven motion.")
if a.csv and out:
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print("->", a.csv)
