"""Which cycles make the m=10 DEL? Rainflow anatomy of the GSPI baseline blade-root moment, plus
correlation of candidate per-step load proxies with the DEL over 10 s windows."""
import glob
import os

import fatpack
import numpy as np

for backend in ("toy", "openfast"):
    print(f"\n### {backend}")
    for p in sorted(glob.glob(os.path.expanduser(f"~/wtrl/baselines/{backend}/*.npz"))):
        d = np.load(p)
        act = d["warmup"] == 0
        M = d["M_oop"][act] / 1e6
        ranges = fatpack.find_rainflow_ranges(M)
        r = np.sort(ranges)[::-1]
        w = r ** 10
        cum = np.cumsum(w) / w.sum()
        n10 = int(np.searchsorted(cum, 0.9)) + 1
        print(f"{os.path.basename(p):>18}: {len(r)} cycles, range max {r[0]:.2f} MNm, median {np.median(r):.3f}; "
              f"90% of sum(range^10) comes from the largest {n10} cycles (top range >= {r[n10 - 1]:.2f} MNm)")
        # window-level proxies vs window DEL
        dt = 0.01
        win = int(10 / dt)
        dels, tv, tv2, rng, std = [], [], [], [], []
        for i in range(0, len(M) - win, win):
            seg = M[i:i + win]
            rr = fatpack.find_rainflow_ranges(seg)
            dels.append((np.sum(rr ** 10)) ** 0.1 if len(rr) else 0)
            dM = np.abs(np.diff(seg))
            tv.append(dM.mean()); tv2.append((dM ** 2).mean()); rng.append(np.ptp(seg)); std.append(seg.std())
        dels = np.array(dels)
        for name, x in (("mean|dM|", tv), ("mean dM^2", tv2), ("10s range", rng), ("10s std", std)):
            print(f"      corr(window DEL, {name:>9}) = {np.corrcoef(dels, np.array(x))[0, 1]:.3f}")
