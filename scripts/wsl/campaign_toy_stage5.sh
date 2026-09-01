#!/bin/bash
# Stage 5: unguarded variants of the method comparison (mono / mono_flag / spec with --supervisor none),
# so the method ranking does not depend on the rollback guardrail. Waits for any running campaign.
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error'
while pgrep -f "[s]cripts/train.py" > /dev/null || pgrep -f "[c]ampaign4.sh" > /dev/null; do sleep 30; done
run() {
  name=$1; shift
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name"; echo "=== $name  $(date) ==="
  python scripts/train.py --backend toy --workers 9 --supervise_every 30 --out "$EXP/$name" "$@" 2>&1 | grep "$FILT"
}
run "toy_c5_mono_none"      --method mono      --episodes 300 --lambda_load 1 --supervisor none --seed 0
run "toy_c5_mono_flag_none" --method mono_flag --episodes 300 --lambda_load 1 --supervisor none --seed 0
run "toy_c5_spec_none"      --method spec      --episodes 300 --lambda_load 1 --supervisor none --seed 0
echo "CAMPAIGN5_DONE $(date)"
