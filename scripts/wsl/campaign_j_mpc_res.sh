#!/bin/bash
# Fairness step 0 (2026-09-11): the LPV-MPC (cost scale v2) driven through the RL agents' residual
# channel — |target − ROSCO command| ≤ the PPO residual bound (0.05 rad) with the same second-order
# damper — tuned under J on S1+S2 with the same rule as every arm and evaluated on both held-out
# sets. Tells how much of the wide-open MPC's J (12.4 / 14.1) is actuation authority.
#
# Grid: N=20, r ∈ {0.1, 0.3, 1} × qt ∈ {0.3, 1, 3, 10} × wc_v ∈ {0.25, 0.35} = 24 points.
# Usage: ~/wtrl/run.sh bash scripts/wsl/campaign_j_mpc_res.sh
set -u
EXP=~/wtrl/exp
FILT='mpc \[\|Traceback\|Error'
echo "=== step 0: LPV-MPC through the residual channel, under J  $(date) ==="
python scripts/mpc_baseline.py --means 8 12.5 15 --seeds 1 2 --scale v2 --residual --horizon 20 --q 1 \
   --r 0.1 0.3 1 --qt 0.3 1 3 10 --wc_v 0.25 0.35 --jobs 8 --port0 6800 \
   --out $EXP/mpc --tag tuneJ2r_s12 2>&1 | grep "$FILT" | grep -v "skip (exists)"
read -r R QT W <<< "$(python scripts/dev/pick_by_J.py "$EXP/mpc/eval_tuneJ2r_s12_*.json" --regex 'N20q1r([0-9.]+)qt([0-9.]+)w([0-9.]+)\.json$' 2>"$EXP/mpc/pick_J2r.txt")"
cat "$EXP/mpc/pick_J2r.txt"
echo "=== MPC (residual channel) selected under J: r=$R qt=$QT wc_v=$W ==="
for spec in "heldoutJ2r_s3456:3 4 5 6" "heldoutJ2r_2W:7 8 9 10"; do
  tag=${spec%%:*}; seeds=${spec#*:}
  [ -f "$EXP/mpc/eval_${tag}_N20q1r${R}qt${QT}w${W}.json" ] && { echo "skip mpc $tag"; continue; }
  python scripts/mpc_baseline.py --means 8 12.5 15 --seeds $seeds --scale v2 --residual --horizon 20 --q 1 \
     --r "$R" --qt "$QT" --wc_v "$W" --jobs 8 --port0 6800 --out $EXP/mpc --tag "$tag" 2>&1 | grep "$FILT"
done
echo "{\"scale\": \"v2\", \"residual\": true, \"N\": 20, \"r\": $R, \"qt\": $QT, \"wc_v\": $W, \"objective\": \"J\"}" > "$EXP/mpc/selected_J2r.json"
echo "--- MPC variants on both held-out sets (wide-open v2 vs residual-channel v2 vs regulation-only):"
python scripts/dev/pick_by_J.py "$EXP/mpc/eval_heldout*_N*.json" --regex 'eval_(heldout[A-Za-z0-9_]*)_(N[0-9]+q1r[0-9.]+qt[0-9.]+w[0-9.]+)\.json$' 2>&1 >/dev/null | grep -v "^best"
echo "STEP0_DONE $(date)"
