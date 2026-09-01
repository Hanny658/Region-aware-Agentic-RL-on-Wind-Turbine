#!/bin/bash
# E4 (autonomous, 2026-08-30): horizon / regulariser hypothesis for the R3 residual on OpenFAST.
# E1/E1b showed a pure speed reward makes PPO learn fast pitching that excites the tower (pitch travel
# 39 -> 51-57 deg, tower DEL +20-26 %, speed std worse) -- myopic with gamma=0.99 (1 s at 10 ms) versus
# the tower period (~3 s). Test a longer horizon and the load term as regulariser. Waits for campaign 3.
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|done in\|Traceback\|Error\|retries'
while ps aux | grep -q "[c]ampaign_of[23].sh"; do sleep 60; done
run() {
  name=$1; shift
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name" ~/wtrl/runs/work_w*
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --workers 8 --supervise_every 24 --episodes 96 --seeds 1 \
     --method r3only --means 15 --eval_means 15 --supervisor none --seed 0 \
     --out "$EXP/$name" "$@" 2>&1 | grep "$FILT"
}
run "of4_r3_lam0_g998"            --lambda_load 0 --gamma 0.998 --gae_lambda 0.98
run "of4_r3_lam1_g998"            --lambda_load 1 --gamma 0.998 --gae_lambda 0.98
run "of4_r3_lam1_g99"             --lambda_load 1
run "of4_r3_lam1tower_g998"       --lambda_load 1 --gamma 0.998 --gae_lambda 0.98 --load_signal fa_acc --fitness_target tower --obs_fa_acc
echo "CAMPAIGN_OF4_DONE $(date)"
