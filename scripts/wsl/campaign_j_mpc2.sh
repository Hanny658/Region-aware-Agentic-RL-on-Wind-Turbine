#!/bin/bash
# Stage 2b of roadmap v2 (2026-09-11): the LPV-MPC with the cost re-scaled so that its tower term
# can act (controllers/mpc.py, `--scale v2`), tuned under J on S1+S2 with the same rule as every
# other arm, then evaluated on both held-out wind sets.
#
# Grid: N x r x qt x wc_v = 2 x 4 x 5 x 2 = 80 points (q = 1). In the v2 scale the historical
# optimum (v1 r = 0.02) corresponds to r ~ 0.3; qt in {0.1 .. 3} spans 5 % .. 170 % of the speed
# term at typical values, where v1 could only reach "off" or "feather".
#
# Resumable: finished grid points and evaluations are skipped.
# Usage: ~/wtrl/run.sh bash scripts/wsl/campaign_j_mpc2.sh
set -u
EXP=~/wtrl/exp
FILT='mpc \[\|Traceback\|Error'
echo "=== stage 2b: LPV-MPC (cost scale v2) under J  $(date) ==="
python scripts/mpc_baseline.py --means 8 12.5 15 --seeds 1 2 --scale v2 --horizon 20 40 --q 1 \
   --r 0.1 0.3 1 3 --qt 0 0.1 0.3 1 3 --wc_v 0.25 0.35 --jobs 8 --port0 6800 \
   --out $EXP/mpc --tag tuneJ2_s12 2>&1 | grep "$FILT" | grep -v "skip (exists)"
read -r N R QT W <<< "$(python scripts/dev/pick_by_J.py "$EXP/mpc/eval_tuneJ2_s12_*.json" --regex 'N([0-9]+)q1r([0-9.]+)qt([0-9.]+)w([0-9.]+)\.json$' 2>"$EXP/mpc/pick_J2.txt")"
cat "$EXP/mpc/pick_J2.txt"
echo "=== MPC (scale v2) selected under J: N=$N r=$R qt=$QT wc_v=$W ==="
for spec in "heldoutJ2_s3456:3 4 5 6" "heldoutJ2_2W:7 8 9 10"; do
  tag=${spec%%:*}; seeds=${spec#*:}
  [ -f "$EXP/mpc/eval_${tag}_N${N}q1r${R}qt${QT}w${W}.json" ] && { echo "skip mpc $tag"; continue; }
  python scripts/mpc_baseline.py --means 8 12.5 15 --seeds $seeds --scale v2 --horizon "$N" --q 1 --r "$R" \
     --qt "$QT" --wc_v "$W" --jobs 8 --port0 6800 --out $EXP/mpc --tag "$tag" 2>&1 | grep "$FILT"
done
echo "{\"scale\": \"v2\", \"N\": $N, \"r\": $R, \"qt\": $QT, \"wc_v\": $W, \"objective\": \"J\"}" > "$EXP/mpc/selected_J2.json"
echo "--- v1 (regulation-only) vs v2 (tower term active) on both held-out sets:"
python scripts/dev/pick_by_J.py "$EXP/mpc/eval_heldout*_N*.json" --regex 'eval_(heldout[A-Za-z0-9_]*)_(N[0-9]+q1r[0-9.]+qt[0-9.]+w[0-9.]+)\.json$' 2>&1 >/dev/null | grep -v "^best"
echo "STAGE2B_DONE $(date)"
