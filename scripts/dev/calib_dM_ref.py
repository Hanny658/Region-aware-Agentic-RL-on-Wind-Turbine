"""Reference scales for the load-term proxies from the GSPI baselines (per backend):
    increment : mean |M_t - M_{t-1}| per step                       -> dM_ref_nm
    range_inc : mean 10 s window range (max - min)                  -> R_ref_nm  (per-step scale = R_ref / N_win)
    ema_std   : mean EMA std of M with time constant tau            -> S_ref_nm
"""
import glob
import os

import numpy as np

WIN_S, TAU_S, DT = 10.0, 5.0, 0.01

for backend in ("toy", "openfast"):
    inc, rng, ema = [], [], []
    for p in sorted(glob.glob(os.path.expanduser(f"~/wtrl/baselines/{backend}/*.npz"))):
        d = np.load(p)
        M = d["M_oop"][d["warmup"] == 0]
        inc.append(np.abs(np.diff(M)).mean())
        win = int(WIN_S / DT)
        rng.append(np.mean([np.ptp(M[i:i + win]) for i in range(0, len(M) - win, win // 2)]))
        a = DT / TAU_S
        m1 = m2 = None
        s = []
        for x in M:
            m1 = x if m1 is None else (1 - a) * m1 + a * x
            m2 = x * x if m2 is None else (1 - a) * m2 + a * x * x
            s.append(np.sqrt(max(m2 - m1 * m1, 0.0)))
        ema.append(np.mean(s[int(2 * TAU_S / DT):]))
        print(f"{backend:>8} {os.path.basename(p):>18}: |dM| {inc[-1] / 1e3:6.2f} kNm  range10s {rng[-1] / 1e6:5.2f} MNm  emastd {ema[-1] / 1e6:5.3f} MNm")
    print(f"  => {backend}: dM_ref_nm = {np.mean(inc):.0f}   R_ref_nm = {np.mean(rng):.0f}   S_ref_nm = {np.mean(ema):.0f}")
