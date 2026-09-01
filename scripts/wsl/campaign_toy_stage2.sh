#!/bin/bash
# Stage 2 control: fixed knobs + guardrail only (spec, seed 0). Waits for any running campaign first.
set -u
EXP=~/wtrl/exp
while pgrep -f "[s]cripts/train.py" > /dev/null; do sleep 30; done
name=toy_s3_guard
if [ ! -f "$EXP/$name/summary.json" ]; then
  rm -rf "$EXP/$name"; echo "=== $name  $(date) ==="
  python scripts/train.py --backend toy --method spec --episodes 300 --workers 9 --out "$EXP/$name" \
     --supervisor guard --supervise_every 30 --seed 0 2>&1 \
     | grep "init\|eval\|sup \|done\|Traceback\|Error"
fi
echo "CAMPAIGN2_DONE $(date)"
