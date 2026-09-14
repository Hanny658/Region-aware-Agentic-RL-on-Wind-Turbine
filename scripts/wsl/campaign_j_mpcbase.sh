#!/bin/bash
# Residual RL on the LPV-MPC base (2026-09-14): does the supervised residual add to the strongest
# controller, and does it repair the MPC's aerodynamic-model error?
#   reference rows: wide-open MPC alone, exact model (exists) and with Cp/Ct x0.95 (evaluated here)
#   arms x 3 seeds: mg3t (fixed reward on the exact MPC), mrw3t (LLM reward expression on the exact MPC),
#                   mgC3t / mrwC3t (same two on the MPC whose model has Cp/Ct x0.95)
# Usage: ~/wtrl/run.sh bash scripts/wsl/campaign_j_mpcbase.sh
set -u
EXP=~/wtrl/exp
echo "=== MPC-base campaign  $(date) ==="
CFG="--scale v2 --horizon 20 --q 1 --r 0.3 --qt 3 --wc_v 0.35 --jobs 8 --port0 6800 --out $EXP/mpc --mm_cp 0.95"
python scripts/mpc_baseline.py --means 8 12.5 15 --seeds 3 4 5 6 $CFG --tag heldoutJ2 2>&1 | grep "mpc \[\|Traceback" | cut -c1-170
python scripts/mpc_baseline.py --means 8 12.5 15 --seeds 7 8 9 10 $CFG --tag heldoutJ2_2W 2>&1 | grep "mpc \[\|Traceback" | cut -c1-170
KNOBS=configs/knobs_j_v3_tuned.json ARMS="mg3t mrw3t mgC3t mrwC3t" bash scripts/wsl/campaign_j_core.sh 0 1 2 2>&1 | tee ~/wtrl/exp/j_mpcbase.log
echo "MPCBASE_DONE $(date)"
