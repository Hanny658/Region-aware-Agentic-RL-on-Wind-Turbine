"""Is the 20 s discarded window long enough? (2026-09-23, no simulations.)

The evaluation protocol simulates 600 s, discards the first 20 s and scores the remaining 580 s. Published wind-energy
work discards 100-300 s and scores 600 s, so a reviewer will ask whether our scored window still contains the start-up
transient. The GSPI baseline .npz files keep the raw OpenFAST channels, so the question can be answered from data
already on disk: this script reports, per baseline episode, the RMS of the tower-base moment and the generator-speed
error in consecutive 20 s windows, and the damage-equivalent load and the two regulation metrics recomputed with the
discard set to 20, 60 and 100 s. If the early windows are not outliers and the metrics are flat in the discard length,
20 s is enough on this plant and the paper can say so with numbers.

    ~/wtrl/run.sh python scripts/dev/transient_check.py [--home ~/wtrl600] [--n 8]
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from eval.metrics import del_rainflow  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--home", default="~/wtrl600")
ap.add_argument("--n", type=int, default=8, help="how many baseline episodes to read")
ap.add_argument("--discards", type=float, nargs="+", default=[20.0, 60.0, 100.0])
a = ap.parse_args()

files = sorted(glob.glob(os.path.expanduser(f"{a.home}/baselines/openfast/*TIB*.npz")))[: a.n]
assert files, f"no baselines under {a.home}"
print(f"{len(files)} baseline episodes from {a.home}\n")

print("RMS of the tower-base moment [MN m] in consecutive 20 s windows, first 200 s (mean over episodes)")
win, rows = 20.0, []
for p in files:
    with np.load(p) as z:
        t, M = z["outb_Time"], z["outb_TwrBsMyt"] / 1e3
    rows.append([float(np.sqrt(np.mean(M[(t >= k * win) & (t < (k + 1) * win)] ** 2))) for k in range(10)])
m = np.array(rows).mean(0)
print("  window  " + "".join(f"{int(k * win):>7d}" for k in range(10)))
print("  RMS     " + "".join(f"{x:7.2f}" for x in m))
print(f"  first window vs the mean of windows 5-9: {100 * (m[0] / m[5:].mean() - 1):+.1f} %"
      f" | second window {100 * (m[1] / m[5:].mean() - 1):+.1f} %\n")

print("metrics of the same episodes recomputed with a longer discard (mean over episodes, % change vs 20 s)")
base = {}
for d in a.discards:
    tw, ge, pm = [], [], []
    for p in files:
        with np.load(p) as z:
            t, M = z["outb_Time"], z["outb_TwrBsMyt"]
            dt = float(t[1] - t[0])
            k = t >= d
            tw.append(del_rainflow(M[k], dt, 4) / 1e3)
            tt, w, P = z["t"], z["gen_speed"], z["P"]
            r3 = (z["region"] == 1) & (tt >= d)      # the log stores the region as a 0/1 above-rated flag
            if r3.sum() > 100:
                ge.append(float((((w[r3] - w[r3].mean()) / w[r3].mean()) ** 2).mean()))
                pm.append(float((((P[r3] - 5.0e6) / 5.0e6) ** 2).mean()))
    v = (float(np.mean(tw)), float(np.mean(ge)), float(np.mean(pm)))
    if not base:
        base = v
        print(f"  discard {d:5.0f} s: tower DEL {v[0]:8.3f} MN m | speed var {v[1]:.3e} | power MSE {v[2]:.3e}   (reference)")
    else:
        print(f"  discard {d:5.0f} s: tower DEL {v[0]:8.3f} MN m ({100 * (v[0] / base[0] - 1):+5.1f} %)"
              f" | speed var {v[1]:.3e} ({100 * (v[1] / base[1] - 1):+5.1f} %)"
              f" | power MSE {v[2]:.3e} ({100 * (v[2] / base[2] - 1):+5.1f} %)")
print("\nA controller is compared with its baseline on the SAME window, so a common bias cancels in every ratio;"
      "\nwhat would not cancel is a transient large enough to dominate one episode's rainflow count.")
