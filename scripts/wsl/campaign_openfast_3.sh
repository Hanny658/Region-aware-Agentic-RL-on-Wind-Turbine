#!/bin/bash
# Exploration E2/E3 (autonomous, 2026-08-30), queued behind campaign_of2.
#   E2  of3_r3all_lam1     : R3 specialist only (R2 residual = 0) on all winds, blade target  -> does removing the
#                            harmful R2 agent beat mono?
#   E3  of3_spec_tower     : spec + guard, load signal = tower-top fore-aft accel, fitness = tower-base DEL
#       of3_mono_tower     : mono + guard, same tower objective
#       of3_spec_llm_tower : spec + LLM,   same tower objective
# All: train on S1 (300 ep, 8 workers), then held-out evaluation on S2/S3 with the run's fitness target.
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error\|retries'
while ps aux | grep -q "[c]ampaign_of[12].sh"; do sleep 60; done
run() {
  name=$1; shift
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name" ~/wtrl/runs/work_w*
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --workers 8 --supervise_every 30 --episodes 300 --seeds 1 --seed 0 \
     --out "$EXP/$name" "$@" 2>&1 | grep "$FILT"
}
TOWER="--load_signal fa_acc --fitness_target tower --obs_fa_acc"
run "of3_r3all_lam1"      --method r3only --lambda_load 1 --supervisor guard
run "of3_spec_tower"      --method spec   --lambda_load 1 --supervisor guard $TOWER
run "of3_mono_tower"      --method mono   --lambda_load 1 --supervisor guard $TOWER
run "of3_spec_llm_tower"  --method spec   --lambda_load 1 --supervisor llm   $TOWER
echo "=== held-out evaluation (seeds 2,3) $(date) ==="
i=0
for r in of3_r3all_lam1 of3_spec_tower of3_mono_tower of3_spec_llm_tower; do
  for ck in ckpt_best.pt ckpt_last.pt; do
    [ -f "$EXP/$r/$ck" ] || continue
    echo "--- $r $ck ---"
    python scripts/evaluate.py --run "$EXP/$r" --ckpt $ck --backend openfast --means 8 12.5 15 --seeds 2 3 --workers 6 \
       --port0 $((6300 + 20 * i)) --tag "heldout_s23_${ck%.pt}" 2>&1 | grep "F=\|U8:\|U12.5:\|U15:\|Traceback\|Error"
    i=$((i + 1))
  done
done
echo "CAMPAIGN_OF3_DONE $(date)"
