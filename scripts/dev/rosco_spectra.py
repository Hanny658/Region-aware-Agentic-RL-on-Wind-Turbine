"""Why the filter corner matters: the loop's own frequencies, measured (roadmap s63, review round 1).

Three tuning files that differ in one line, F_LPFCornerFreq = 1.5708 (shipped), 3.0 and 6.5 rad/s, are run on the
same seven fresh wind fields (seed 7, 12 to 24 m/s, class B, 600 s) with the per-step logs kept, and the power
spectral densities of the generator-speed error, the tower-base fore-aft moment and the pitch rate are compared,
averaged over the seven wind speeds. Beside them, the first-order filter's magnitude and phase at the plant's
landmarks, which is the analytic form of the same statement: a corner at 1.57 rad/s lags the pitch loop at its
crossover and still passes most of the tower mode.

    WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 ~/wtrl/run.sh python scripts/dev/rosco_spectra.py \
        [--seeds 7] [--out docs/manuscript/figures] [--figure_only]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))
from scripts.rosco_native import START, canon, key_of, make_template  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.expanduser("~/wtrl/exp/rosco_spectra"))
ap.add_argument("--template", default=os.path.expanduser("~/wtrl/runs/template_5mw"))
ap.add_argument("--config_run", default=os.path.expanduser("~/wtrl/exp/jg3t_s0"))
ap.add_argument("--seeds", nargs="+", type=int, default=[7])
ap.add_argument("--means", nargs="+", type=float, default=[12, 14, 16, 18, 20, 22, 24])
ap.add_argument("--ti", default="B")
ap.add_argument("--episode_s", type=float, default=600.0)
ap.add_argument("--workers", type=int, default=7)
ap.add_argument("--port0", type=int, default=8300)
ap.add_argument("--out", default=str(PROJ / "docs" / "manuscript" / "figures"))
ap.add_argument("--csv", default=str(PROJ / "docs" / "tables" / "rosco_spectra.csv"))
ap.add_argument("--figure_only", action="store_true")
a = ap.parse_args()

ROWS = [("shipped, corner 1.57", 1.5708, "#333333", "--"),
        ("one line: corner 3.0", 3.0, "#55A868", "-"),
        ("the same line at 6.5", 6.5, "#2E7D4F", "-.")]
TOWER, THREE_P, CROSS = 2.04, 3.80, 0.6          # rad/s: first tower fore-aft mode, 3P at rated, the loop's omega_pc
root = Path(a.root)
root.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- the three rollouts, logs kept
for name, corner, _, _ in ROWS:
    p = canon({**START, "lpf_corner": corner})
    tag = f"spec_{key_of(p)}_s{''.join(map(str, a.seeds))}"
    logs = root / f"logs_{tag}"
    if a.figure_only or (logs.exists() and len(glob.glob(str(logs / "*.npz"))) >= len(a.means) * len(a.seeds)):
        continue
    tpl = root / "templates" / key_of(p)
    tpl.parent.mkdir(parents=True, exist_ok=True)
    make_template(Path(a.template), tpl, p)
    cmd = [sys.executable, str(PROJ / "scripts" / "evaluate.py"), "--run", a.config_run, "--gspi", "--backend",
           "openfast", "--seeds", *map(str, a.seeds), "--episode_s", str(a.episode_s), "--workers", str(a.workers),
           "--port0", str(a.port0), "--tag", tag, "--out", str(root), "--relabel_wind", "--dump_log",
           "--means", *[f"{m:g}" for m in a.means], "--ti", a.ti]
    print(f"== {name}: {' '.join(cmd[-14:])}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "WTRL_TEMPLATE": str(tpl)})
    shutil.rmtree(tpl, ignore_errors=True)
    print(r.stdout[-1500:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], flush=True)
        sys.exit(f"evaluation failed for {name}")

# ---------------------------------------------------------------- spectra
from scipy.signal import welch  # noqa: E402


def load_runs(corner):
    p = canon({**START, "lpf_corner": corner})
    tag = f"spec_{key_of(p)}_s{''.join(map(str, a.seeds))}"
    files = sorted(glob.glob(str(root / f"logs_{tag}" / "*.npz")))
    if not files:
        sys.exit(f"no logs for corner {corner}: run without --figure_only first")
    return [np.load(f) for f in files]


def series(d, which):
    """(t, x) for one quantity, from the OpenFAST channels where they exist, else from the 10 ms env log."""
    keys = set(d.files)
    if which == "speed":
        if "outb_GenSpeed" in keys:
            return d["outb_Time"], d["outb_GenSpeed"]
        return None, d["gen_speed"]
    if which == "tower":
        return d["outb_Time"], d["outb_TwrBsMyt"]
    if which == "pitch":
        for k in ("outb_BldPitch1", "outb_PtchPMzc1"):
            if k in keys:
                return d["outb_Time"], d[k]
        for k in ("pitch", "beta", "pitch_cmd"):
            if k in keys:
                return None, d[k]
    raise KeyError(which)


def psd(d, which, warm_s=20.0, dt_env=0.01):
    t, x = series(d, which)
    x = np.asarray(x, dtype=float)
    if t is None:
        dt = dt_env
        t = np.arange(len(x)) * dt
    else:
        t = np.asarray(t, dtype=float)
        dt = float(t[1] - t[0])
    k = t >= warm_s
    x = x[k]
    if which == "speed":
        x = x - x.mean()                            # the regulation error about its own mean
    if which == "pitch":
        x = np.gradient(x, dt)                      # pitch rate, deg/s
    f, P = welch(x, fs=1.0 / dt, nperseg=min(len(x), 8192), detrend="constant")
    return f, P


once = load_runs(ROWS[0][1])[0]
print("log keys:", sorted(once.files)[:40])
quant = [("speed", "generator speed about its mean", "(rpm)$^2$/Hz"),
         ("tower", "tower-base fore-aft moment", "(kN m)$^2$/Hz"),
         ("pitch", "collective pitch rate", "(deg/s)$^2$/Hz")]
spectra = {}
for name, corner, _, _ in ROWS:
    runs = load_runs(corner)
    for q, _, _ in quant:
        acc = None
        for d in runs:
            f, P = psd(d, q)
            acc = P if acc is None else acc + P
        spectra[(corner, q)] = (f, acc / len(runs))

# band powers, for the text and the CSV: the share of each spectrum below the shipped corner, around the tower
# mode (0.25 to 0.40 Hz) and around 3P (0.50 to 0.75 Hz)
def band(f, P, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.trapz(P[m], f[m])) if m.sum() > 1 else float("nan")


rows_out = []
for name, corner, _, _ in ROWS:
    rec = {"row": name, "corner": corner}
    for q, _, _ in quant:
        f, P = spectra[(corner, q)]
        tot = band(f, P, 0.0, 5.0)
        rec[f"{q}_total"] = tot
        rec[f"{q}_below_0p25Hz"] = band(f, P, 0.0, 0.25)
        rec[f"{q}_tower_band"] = band(f, P, 0.25, 0.40)
        rec[f"{q}_3P_band"] = band(f, P, 0.50, 0.75)
    rows_out.append(rec)
w = 2 * np.pi
for corner in (1.5708, 3.0, 6.5):
    lag = np.degrees(np.arctan(CROSS / corner))
    gt, g3 = 1 / np.sqrt(1 + (TOWER / corner) ** 2), 1 / np.sqrt(1 + (THREE_P / corner) ** 2)
    print(f"corner {corner:g}: phase lag at the loop's {CROSS} rad/s {lag:.1f} deg; passes {gt:.0%} of the tower mode "
          f"({TOWER} rad/s) and {g3:.0%} of 3P ({THREE_P} rad/s)")
import csv  # noqa: E402
os.makedirs(os.path.dirname(a.csv), exist_ok=True)
with open(a.csv, "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
    wr.writeheader()
    wr.writerows(rows_out)
for rec in rows_out:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
print(f"-> {a.csv}")

# ---------------------------------------------------------------- the figure
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "figure.dpi": 150, "savefig.dpi": 220, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.9))
ax = axes[0]
ww = np.logspace(-1.3, 1.5, 400)
for name, corner, c, ls in ROWS:
    H = 1 / np.sqrt(1 + (ww / corner) ** 2)
    ax.plot(ww, 20 * np.log10(H), color=c, ls=ls, lw=1.4, label=name)
ax.set_xscale("log")
ax.set_xlabel("frequency, rad/s")
ax.set_ylabel("filter gain, dB")
ax.set_title("the speed filter, first order")
ax.set_ylim(-24, 1)
for x, lab in ((CROSS, "loop"), (TOWER, "tower"), (THREE_P, "3P")):
    ax.axvline(x, color="#999999", lw=0.7, ls=":")
    ax.text(x, -23, lab, fontsize=7, color="#777777", ha="center", va="bottom")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.08))
for ax, (q, title, unit) in zip(axes[1:], quant):
    for name, corner, c, ls in ROWS:
        f, P = spectra[(corner, q)]
        m = (f > 0.02) & (f <= 3.0)
        ax.plot(f[m] * w, P[m], color=c, ls=ls, lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency, rad/s")
    ax.set_ylabel(f"PSD, {unit}")
    ax.set_title(title)
    for x in (TOWER, THREE_P):
        ax.axvline(x, color="#999999", lw=0.7, ls=":")
fig.suptitle(f"the three tuning files on the same {len(a.means) * len(a.seeds)} fresh episodes "
             f"(seed{'s' if len(a.seeds) > 1 else ''} {' '.join(map(str, a.seeds))}, 12 to 24 m/s), spectra averaged over wind speed",
             fontsize=8.5, y=1.02)
fig.tight_layout()
out = Path(a.out) / "f5_spectra.png"
fig.savefig(out, bbox_inches="tight")
print(f"-> {out}")
