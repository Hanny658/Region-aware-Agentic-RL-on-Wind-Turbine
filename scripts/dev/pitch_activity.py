"""Pitch activity of the GSPI baselines (toy vs OpenFAST) and of a run's best toy evaluation."""
import glob
import json
import os
import sys

import numpy as np

for backend in ("toy", "openfast"):
    for p in sorted(glob.glob(os.path.expanduser(f"~/wtrl/baselines/{backend}/*.npz"))):
        d = np.load(p)
        act = d["warmup"] == 0
        b = np.rad2deg(d["beta_meas"][act])
        bn = np.rad2deg(d["beta_native"][act])
        db = np.abs(np.diff(b))
        print(f"{backend:>8} {os.path.basename(p):>18}: pitch travel {db.sum():7.1f} deg  max step {db.max():.3f} deg  "
              f"pitch mean {b.mean():5.2f} std {b.std():5.2f}  native-cmd travel {np.abs(np.diff(bn)).sum():7.1f}  "
              f"gen_speed std {d['gen_speed'][act].std():.3f}")
if len(sys.argv) > 1:
    for r in sys.argv[1:]:
        recs = [json.loads(l) for l in open(os.path.expanduser(f"~/wtrl/exp/{r}/decisions.jsonl"))]
        recs = [x for x in recs if "per_episode" in x]
        best = max(recs, key=lambda x: x["fit"]["F"])
        print(f"\n{r} best toy eval @ep {best['episode']} F={best['fit']['F']:.2f}")
        for e in best["per_episode"]:
            print(f"   U{e['mean_wind']:g}: pitch travel {e['pitch_travel_deg']:.0f} (GSPI {e['pitch_travel_base_deg']:.0f}) deg  |dbeta| {e['dbeta_abs_mean_deg']:.2f} deg  DELred {e['del_red_pct']:.1f}%")
