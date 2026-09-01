"""Table of held-out evaluations (eval_*.json) for runs matching the given globs."""
import glob
import json
import os
import sys

pats = sys.argv[1:] or ["~/wtrl/exp/of*"]
print(f"{'run':>26} {'tag':>24} {'tgt':>5} {'F':>7} {'F_tol2':>7} {'tier':>8} {'blade':>6} {'tower':>6} {'Eloss':>6} {'spd':>6} {'spdMAE':>7}")
for pat in pats:
    for d in sorted(glob.glob(os.path.expanduser(pat))):
        for f in sorted(glob.glob(os.path.join(d, "eval_*.json"))):
            j = json.load(open(f))
            tag = os.path.basename(f)[5:-5]
            if "tier" not in j:      # older evals: derive the tier from the stored constraint numbers
                e_ok = j['energy_loss_pct'] <= 1.0
                j["tier"] = "strict" if (e_ok and j['speed_std_ratio'] <= 1.0) else ("tol2" if (e_ok and j['speed_std_ratio'] <= 1.02) else "degraded")
            blade = j.get('RootMyc1_DEL_red_pct', j['del_red_pct'])
            tower = j.get('TwrBsMyt_DEL_red_pct', float('nan'))
            print(f"{os.path.basename(d):>26} {tag:>24} {str(j.get('target', 'blade'))[:5]:>5} {j['F']:7.2f} "
                  f"{j.get('F_tol2', float('nan')):7.2f} {j.get('tier', '?'):>8} {blade:6.2f} {tower:6.2f} "
                  f"{j['energy_loss_pct']:6.2f} {j['speed_std_ratio']:6.3f} {j.get('speed_mae_ratio_R3', float('nan')):7.3f}")
