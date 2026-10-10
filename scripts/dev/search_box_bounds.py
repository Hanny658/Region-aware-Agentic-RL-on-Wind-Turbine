"""Where the parameter-search box ends, read off the plant instead of chosen (2026-10-11).

Every limit of the search box in `docs/design_2026-10-11_rosco-native-search.md` is either a controller
setting or something measured on the shipped controller's own baselines. This script produces all of them, so
the box is reproducible rather than asserted, and it checks the one piece of algebra the redesign rests on:
that ROSCO's gain-schedule formula can be inverted exactly from the shipped tables.

    ~/wtrl/run.sh python scripts/dev/search_box_bounds.py
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

TEMPLATE = os.path.expanduser("~/wtrl/runs/template_5mw")
BASELINES = os.path.expanduser("~/wtrl600/baselines/openfast")
ABOVE_RATED = (12, 14, 16, 18, 20, 22, 24)
ZETA_SHIPPED, OMEGA_SHIPPED = 0.7, 0.6      # ROSCO/Examples/Tune_Cases/NREL5MW.yaml
DLL_DT = 0.01                               # ServoDyn, s
FA_HPF_SHIPPED = 0.172                       # the corner the damper template used, for the integral below

# ---------------------------------------------------------------- 1. invert ROSCO's gain schedule
txt = open(f"{TEMPLATE}/DISCON.IN").read().splitlines()


def disc_row(name: str) -> np.ndarray:
    ln = next(l for l in txt if name in l)
    return np.array([float(x) for x in re.split(r"\s+", ln.split("!")[0].strip())])


kp, ki = disc_row("PC_GS_KP"), disc_row("PC_GS_KI")
# second_order_PI:  Kp = (2 zeta omega + A) / B,  Ki = omega^2 / B
B = OMEGA_SHIPPED ** 2 / ki
A = kp * B - 2 * ZETA_SHIPPED * OMEGA_SHIPPED
print(f"ROSCO gain schedule inverted from the shipped tables (zeta {ZETA_SHIPPED}, omega {OMEGA_SHIPPED} rad/s)")
print(f"  B = d(speed rate)/d(pitch): {B.min():9.3f} .. {B.max():9.3f}   monotone "
      f"{bool(np.all(np.diff(B) < 0) or np.all(np.diff(B) > 0))}")
print(f"  A = d(speed rate)/d(speed): {A.min():9.5f} .. {A.max():9.5f}")
print(f"  round trip error: {np.abs((2 * ZETA_SHIPPED * OMEGA_SHIPPED + A) / B - kp).max():.1e}")


def gains(zeta: float, omega: float) -> tuple[np.ndarray, np.ndarray]:
    """The tables ROSCO's toolbox would write for this tuning."""
    return (2 * zeta * omega + A) / B, omega ** 2 / B


def implied_zeta(kp_scale: float, ki_scale: float) -> np.ndarray:
    """The damping ratio a pair of table multipliers implies, point by point along the schedule.

    A ROSCO tuning has one zeta_pc, so a spread here means the scaled tables are unreachable by any tuning."""
    omega = OMEGA_SHIPPED * np.sqrt(ki_scale)
    return (kp_scale * 2 * ZETA_SHIPPED * OMEGA_SHIPPED + (kp_scale - 1) * A) / (2 * omega)


print("\nAre scalar multipliers on the tables a ROSCO tuning?")
for a_, b_ in ((1.13, 1.02), (1.15, 1.40), (1.27, 1.64), (0.887, 1.21)):
    z = implied_zeta(a_, b_)
    print(f"  kp x{a_:<6.3f} ki x{b_:<6.3f} -> omega_pc {OMEGA_SHIPPED * np.sqrt(b_):.4f} rad/s, implied zeta_pc "
          f"{z.min():.4f}..{z.max():.4f} ({100 * (z.max() - z.min()) / z.mean():.1f}% of its mean)")
kp_hi, ki_hi = gains(2.0, 2.037)
print(f"  far corner of the new box (zeta 2.0, omega 2.04): Kp {kp_hi.max() / kp.max():.1f}x and "
      f"Ki {ki_hi.max() / ki.max():.1f}x the shipped tables")

# ---------------------------------------------------------------- 2. measure the plant
fs = [f for f in sorted(glob.glob(f"{BASELINES}/*.npz"))
      if any(f"U{u}_" in os.path.basename(f) for u in ABOVE_RATED)]
print(f"\n{len(fs)} above-rated 600 s baselines of the shipped controller")

twr, rot, acc_sd, hp_int, trav = [], [], [], [], []
spec_sum, spec_f = None, None
for f in fs:
    d = np.load(f)
    t = d["outb_Time"]
    dt = float(np.median(np.diff(t)))
    m = t >= 20.0

    def spectrum(ch: str) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(d[ch], float)[m]
        x = x - x.mean()
        return np.fft.rfftfreq(len(x), dt), np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2

    F, P = spectrum("outb_TwrBsMyt")
    s = (F > 0.15) & (F < 0.60)
    twr.append(float(F[s][np.argmax(P[s])]))
    F, P = spectrum("outb_GenSpeed")
    spec_f, spec_sum = F, P if spec_sum is None else spec_sum + P
    rot.append(float(np.mean(d["outb_RotSpeed"][m])))

    a = np.asarray(d["outb_NcIMUTAxs"], float)[m]
    acc_sd.append(float(a.std()))
    # the damper integrates the high-passed acceleration; its unsaturated pitch demand is FA_KI times this
    al = 1.0 / (1.0 + FA_HPF_SHIPPED * dt)
    y, xp = 0.0, a[0]
    yh = np.empty_like(a)
    for i, x in enumerate(a):
        y = al * (y + x - xp)
        xp = x
        yh[i] = y
    hp_int.append(float(np.abs(np.cumsum(yh) * dt).max()))

    b = np.asarray(d["outb_BldPitch1"], float)[m]
    trav.append(float(np.abs(np.diff(b)).sum() / (t[m][-1] - t[m][0])))

cum = np.cumsum(spec_sum) / spec_sum.sum()
f9999 = float(spec_f[int(np.searchsorted(cum, 0.9999))])
w_twr = 2 * np.pi * float(np.median(twr))
rpm = float(np.median(rot))
hp_med, hp_max = float(np.median(hp_int)), float(max(hp_int))

print(f"  first tower fore-aft mode        {np.median(twr):8.4f} Hz = {w_twr:7.3f} rad/s")
print(f"  rotor speed / 3P                 {rpm:8.3f} rpm, 3P {3 * rpm / 60:.4f} Hz = {2 * np.pi * rpm / 20:.3f} rad/s")
print(f"  99.99 % of generator-speed power below {f9999:7.4f} Hz = {2 * np.pi * f9999:.3f} rad/s")
print(f"  nacelle fore-aft accel s.d.      {np.median(acc_sd):8.4f} m/s2 (worst {max(acc_sd):.3f})")
print(f"  max |integral of its high pass|  {hp_med:8.4f} m/s  (worst {hp_max:.3f})")
print(f"  shipped pitch travel             {np.median(trav):8.4f} deg/s = "
      f"{100 * np.median(trav) / 10.0:.2f}% of the 10 deg/s rate limit")
print(f"  controller step {DLL_DT} s -> Nyquist {np.pi / DLL_DT:.1f} rad/s")

print("\nBox limits that follow:")
print(f"  omega_pc         0.10 .. {w_twr:5.2f} rad/s   upper = the tower fore-aft mode the loop must not drive")
print(f"  zeta_pc          0.10 .. 2.00            nearly undamped to heavily overdamped")
print(f"  F_LPFCornerFreq  0.20 .. {2 * np.pi * f9999:5.1f} rad/s   upper = where the filter passes 99.99 % of the signal")
print(f"  FA_KI            0.00 .. 0.20  rad s/m   upper: demand {0.20 * hp_med:.3f} rad = "
      f"{np.rad2deg(0.20 * hp_med):.1f} deg on a median episode")
print(f"       the old ceiling 0.004 allowed {np.rad2deg(0.004 * hp_med):.2f} deg, so the damper could do nothing")
print(f"  FA_HPFCornerFreq 0.05 .. 1.20  rad/s     upper: the tower mode still passes within "
      f"{np.rad2deg(np.arctan(1.20 / w_twr)):.0f} deg of phase")
