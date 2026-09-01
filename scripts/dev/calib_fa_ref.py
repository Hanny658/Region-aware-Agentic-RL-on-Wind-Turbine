"""Reference scales for load_signal = fa_acc (tower-top fore-aft acceleration) from the OpenFAST
GSPI baselines, same definitions as calib_dM_ref.py. Prints the yaml lines to paste."""
import glob
import os

import numpy as np

WIN_S, TAU_S, DT = 10.0, 5.0, 0.01
inc, rng, ema = [], [], []
for p in sorted(glob.glob(os.path.expanduser("~/wtrl/baselines/openfast/*.npz"))):
    d = np.load(p)
    if "fa_acc" not in d.files:
        continue
    x = d["fa_acc"][d["warmup"] == 0]
    inc.append(np.abs(np.diff(x)).mean())
    win = int(WIN_S / DT)
    rng.append(np.mean([np.ptp(x[i:i + win]) for i in range(0, len(x) - win, win // 2)]))
    a = DT / TAU_S
    m1 = m2 = None
    s = []
    for v in x:
        m1 = v if m1 is None else (1 - a) * m1 + a * v
        m2 = v * v if m2 is None else (1 - a) * m2 + a * v * v
        s.append(np.sqrt(max(m2 - m1 * m1, 0.0)))
    ema.append(np.mean(s[int(2 * TAU_S / DT):]))
    print(f"{os.path.basename(p):>18}: |dx| {inc[-1]:.5f}  range10s {rng[-1]:.4f}  emastd {ema[-1]:.4f}  (m/s^2)  std {x.std():.4f}")
print(f"fa_dM_ref: {{openfast: {np.mean(inc):.6f}}}")
print(f"fa_R_ref:  {{openfast: {np.mean(rng):.6f}}}")
print(f"fa_S_ref:  {{openfast: {np.mean(ema):.6f}}}")
