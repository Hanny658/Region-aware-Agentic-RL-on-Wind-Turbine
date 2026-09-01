#!/bin/bash
# Toy campaign, stage 1+3 (run inside WSL via ~/wtrl/run.sh bash scripts/wsl/campaign_toy_stage1.sh)
#   stage 1: lambda_load sweep with fixed knobs, spec method            -> ~/wtrl/exp/toy_s1_lam{L}
#   stage 3: supervisors on spec (none / llm / random), seed 0            -> ~/wtrl/exp/toy_s3_{sup}
set -u
EXP=~/wtrl/exp
EP=300
W=9
run() {  # name, extra args...
  name=$1; shift
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name"
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend toy --method spec --episodes $EP --workers $W --out "$EXP/$name" "$@" \
     2>&1 | grep -v "ROSCO\|^ *$\|\*\*\*\|Developed\|Delft\|-----\|Check WE_Op\|The filtered\|A wind turbine" \
     | grep "init\|eval\|sup \|done\|Traceback\|Error\|/300\]" | grep -v "^\[ *[0-9]*/300\] roll"
}
for L in 1 3 10 30; do
  run "toy_s1_lam$L" --lambda_load $L --supervisor none --supervise_every 30
done
run "toy_s3_none"   --supervisor none   --supervise_every 30 --seed 0
run "toy_s3_llm"    --supervisor llm    --supervise_every 30 --seed 0
run "toy_s3_random" --supervisor random --supervise_every 30 --seed 0
echo "CAMPAIGN_DONE $(date)"
