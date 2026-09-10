"""Training-time trajectory of the metric-set objective for one or more runs: at every decision the
measured J (before any rollback masking), its four terms, the speed-std ratio, energy, and the
per-episode speed-MSE reductions — the quickest way to see *why* J moved (a mean-offset blow-up of
the MSE terms looks very different from a load regression).

    python scripts/dev/j_trajectory.py ~/wtrl/exp/jg2_s*
"""
import glob
import json
import os
import sys

for pat in sys.argv[1:] or ["~/wtrl/exp/jg2_s*"]:
    for d in sorted(glob.glob(os.path.expanduser(pat))):
        p = os.path.join(d, "decisions.jsonl")
        if not os.path.exists(p):
            continue
        print("==", os.path.basename(d))
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            f = r["fit"]
            if "J" not in f:
                continue
            tag = " ROLLBACK" if "rollback" in r else (" fork->%s" % r["fork"]["chosen"] if "fork" in r else "")
            pe = [p.get("gen_speed_mse_red_pct") for p in r.get("per_episode", [])]
            pe = ["  ." if v is None or v != v else f"{v:4.0f}" for v in pe]
            print(f"ep {r['episode']:3d}  J {f.get('J_measured', f['J']):7.2f}  "
                  f"P {f['J_power_mse_red_pct']:6.1f}  w {f['J_gen_speed_mse_red_pct']:6.1f}  "
                  f"T {f['J_TwrBsMyt_DEL_red_pct']:5.1f}  B {f['J_RootMyc1_DEL_red_pct']:5.1f}  "
                  f"spd {f['speed_std_ratio']:.3f}  E {f['energy_loss_pct']:5.2f}  w/ep [{' '.join(pe)}]{tag}")
