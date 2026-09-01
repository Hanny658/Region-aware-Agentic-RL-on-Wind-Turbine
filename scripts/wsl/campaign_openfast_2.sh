#!/bin/bash
# Exploration E1 (autonomous, 2026-08-30): can the residual PPO improve ROSCO's Region-3 speed
# regulation on OpenFAST at all?  r3only on 15 m/s, lambda=0 (pure speed term), no guardrail,
# 100 episodes, with exploration / action-scale variants. Waits for campaign 1 to finish.
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|done in\|Traceback\|Error\|retries'
while ps aux | grep -q "[c]ampaign_of1.sh"; do sleep 60; done
run() {
  name=$1; shift
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name" ~/wtrl/runs/work_w*
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --workers 8 --supervise_every 24 --episodes 96 --seeds 1 \
     --method r3only --means 15 --eval_means 15 --lambda_load 0 --supervisor none --seed 0 \
     --out "$EXP/$name" "$@" 2>&1 | grep "$FILT"
}
run "of2_r3_lam0_base"                                            # paper-like: std 0.37, dbeta_max 0.05
run "of2_r3_lam0_std02"     --log_std_init -1.6                   # less exploration noise (std 0.20)
run "of2_r3_lam0_db002"     --dbeta_max 0.02                      # smaller residual authority (1.1 deg)
run "of2_r3_lam0_std02_db002" --log_std_init -1.6 --dbeta_max 0.02
echo "CAMPAIGN_OF2_DONE $(date)"
