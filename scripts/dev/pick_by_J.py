"""Model selection under the metric-set objective: among eval_*.json files matching a glob, print the
config with the best J subject to energy_ok (best J overall if none satisfies it). J is recomputed from
the stored metrics for evaluations written before 2026-09-11, so the historical sweeps can be re-picked
without re-simulating.

    python scripts/dev/pick_by_J.py "~/wtrl/exp/mpc/eval_tuneJ_s12_*.json" --regex "r([0-9.]+)qt([0-9.]+)w([0-9.]+)"
    python scripts/dev/pick_by_J.py "~/wtrl/exp/gspi_td/eval_tune_*.json" --regex "ki([0-9.]+)_hpf([0-9.]+)"

Prints the table on stderr and the regex groups of the winner (space-separated) on stdout, so a
shell can `read -r A B <<< "$(python ... 2>/dev/null)"`.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eval.fitness import objective_J  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("pattern")
ap.add_argument("--regex", required=True, help="groups = the config parameters to print")
args = ap.parse_args()

rows = []
for f in sorted(glob.glob(os.path.expanduser(args.pattern))):
    m = re.search(args.regex, os.path.basename(f))
    if not m:
        continue
    j = json.load(open(f, encoding="utf-8"))
    if "J" not in j:
        j["J"], parts = objective_J(j, float(j["energy_loss_pct"]))
        j.update(parts)
        j["energy_ok"] = float(j["energy_loss_pct"]) <= 1.0
    rows.append((bool(j["energy_ok"]), float(j["J"]), m.groups(), j))

if not rows:
    print("no evaluations match", file=sys.stderr)
    sys.exit(1)
rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
print(f"{'config':>28} {'J':>7} {'E_ok':>5} {'P_mse':>7} {'w_mse':>7} {'twrDEL':>7} {'bldDEL':>7} {'Eloss':>6} {'F':>7}", file=sys.stderr)
for ok, J, g, j in rows:
    print(f"{'/'.join(g):>28} {J:7.2f} {str(ok):>5} {j.get('J_power_mse_red_pct', float('nan')):7.1f} "
          f"{j.get('J_gen_speed_mse_red_pct', float('nan')):7.1f} {j.get('J_TwrBsMyt_DEL_red_pct', float('nan')):7.1f} "
          f"{j.get('J_RootMyc1_DEL_red_pct', float('nan')):7.1f} {j['energy_loss_pct']:6.2f} {j.get('F', float('nan')):7.2f}",
          file=sys.stderr)
print("best:", "/".join(rows[0][2]), f"J={rows[0][1]:.2f}", "" if rows[0][0] else "(NO config keeps energy <= 1 %)", file=sys.stderr)
print(" ".join(rows[0][2]))
