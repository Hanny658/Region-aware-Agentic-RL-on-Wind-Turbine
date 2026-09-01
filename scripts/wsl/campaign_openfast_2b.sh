#!/bin/bash
# E1b: speed-term scale. tau_speed_err=0.02 makes exp(-|d|/tau) nearly flat for OpenFAST-size errors
# (|d| ~ 0.005): reward ~ constant -> advantage = noise -> policy random-walks and speed std grows
# (E1 base: 1.06 -> 1.27). Try tau matched to the actual error magnitude. Waits for campaign_of2.sh.
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|done in\|Traceback\|Error\|retries'
while ps aux | grep -q "[c]ampaign_of2.sh$"; do sleep 60; done
run() {
  name=$1; shift
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name" ~/wtrl/runs/work_w*
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --workers 8 --supervise_every 24 --episodes 96 --seeds 1 \
     --method r3only --means 15 --eval_means 15 --lambda_load 0 --supervisor none --seed 0 \
     --out "$EXP/$name" "$@" 2>&1 | grep "$FILT"
}
run "of2b_r3_lam0_tau005"        --tau_speed_err 0.005
run "of2b_r3_lam0_tau005_std02"  --tau_speed_err 0.005 --log_std_init -1.6
run "of2b_r3_lam0_tau0025_std02" --tau_speed_err 0.0025 --log_std_init -1.6
echo "CAMPAIGN_OF2B_DONE $(date)"
