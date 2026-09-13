#!/bin/bash
# Robustness of the MPC-v2 reference (2026-09-13): does "the correctly scaled MPC is the strongest
# controller on J" survive a controller model that is wrong, and wind classes it was not tuned on?
# All rows use the J-selected configuration (N=20, q=1, r=0.3, qt=3, wc_v=0.35) through the RL's
# residual channel; the plant never changes, only the MPC's internal model.
#
#   mismatch: Cp/Ct x0.85 / x1.15, tower frequency x0.9 / x1.1, modal mass x0.8 / x1.2  (S3-S6)
#   stress:   TI 14 (S3-S4, three means), TI 22 (U15, S3-S4), U18 (S3-S4)                (RL rows: roadmap 08-30 s17)
#
# Eval-only, resumable (finished rows are skipped). Usage: ~/wtrl/run.sh bash scripts/wsl/campaign_j_mpc_robust.sh
set -u
EXP=~/wtrl/exp
FILT='mpc \[\|Traceback\|Error'
CFG="--scale v2 --residual --horizon 20 --q 1 --r 0.3 --qt 3 --wc_v 0.35 --jobs 8 --port0 6800 --out $EXP/mpc"
echo "=== MPC-v2 robustness  $(date) ==="
for mm in "1.0 1.0 1.0" "0.85 1.0 1.0" "1.15 1.0 1.0" "1.0 0.9 1.0" "1.0 1.1 1.0" "1.0 1.0 0.8" "1.0 1.0 1.2"; do
  set -- $mm
  python scripts/mpc_baseline.py --means 8 12.5 15 --seeds 3 4 5 6 $CFG --mm_cp $1 --mm_ft $2 --mm_m $3 --tag mmJ2r_s3456 2>&1 | grep "$FILT"
done
python scripts/mpc_baseline.py --means 8 12.5 15 --seeds 3 4 --ti 14 $CFG --tag robustJ2r_ti14 2>&1 | grep "$FILT"
python scripts/mpc_baseline.py --means 15        --seeds 3 4 --ti 22 $CFG --tag robustJ2r_ti22 2>&1 | grep "$FILT"
python scripts/mpc_baseline.py --means 18        --seeds 3 4 --ti 8  $CFG --tag robustJ2r_u18  2>&1 | grep "$FILT"
echo "--- summary:"
python scripts/dev/pick_by_J.py "$EXP/mpc/eval_mmJ2r_s3456_*.json" --regex 'eval_mmJ2r_s3456_N20q1r0.3qt3w0.35(_cp[0-9.]+ft[0-9.]+m[0-9.]+)?\.json$' 2>&1 >/dev/null | grep -v "^best"
python scripts/dev/pick_by_J.py "$EXP/mpc/eval_robustJ2r_*.json" --regex 'eval_(robustJ2r_[a-z0-9]+)_' 2>&1 >/dev/null | grep -v "^best"
echo "MPC_ROBUST_DONE $(date)"
