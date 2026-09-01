#!/bin/bash
# E3 rerun of spec+LLM with the tower objective (first attempt crashed: the toy twin pool could not
# build the fa_acc reward). Waits for campaigns 2-5.
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error\|retries'
while ps aux | grep -q "[c]ampaign_of[2345].sh"; do sleep 60; done
name=of3_spec_llm_tower
if [ ! -f "$EXP/$name/summary.json" ]; then
  rm -rf "$EXP/$name" ~/wtrl/runs/work_w*; echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --workers 8 --supervise_every 30 --episodes 300 --seeds 1 --seed 0 \
     --method spec --lambda_load 1 --supervisor llm --load_signal fa_acc --fitness_target tower --obs_fa_acc \
     --out "$EXP/$name" 2>&1 | grep "$FILT"
  for ck in ckpt_best.pt ckpt_last.pt; do
    [ -f "$EXP/$name/$ck" ] || continue
    echo "--- $name $ck ---"
    python scripts/evaluate.py --run "$EXP/$name" --ckpt $ck --backend openfast --means 8 12.5 15 --seeds 2 3 --workers 6 \
       --port0 6700 --tag "heldout_s23_${ck%.pt}" 2>&1 | grep "F=\|U8:\|U12.5:\|U15:\|Traceback\|Error"
  done
fi
echo "CAMPAIGN_OF6_DONE $(date)"
