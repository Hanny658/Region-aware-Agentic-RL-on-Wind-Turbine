#!/bin/bash
# E5' (autonomous, 2026-08-30 06:50): core comparison on OpenFAST with the E4 fix (gamma 0.998 / gae 0.98,
# now the default in configs/ppo.yaml). Tower objective (load signal = tower-top fore-aft accel,
# fitness = tower-base DEL) unless noted. Train on S1, held-out S2/S3. Waits for campaign 4.
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error\|retries'
while ps aux | grep -q "[c]ampaign_of[234].sh"; do sleep 60; done
run() {
  name=$1; shift
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name" ~/wtrl/runs/work_w*
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --workers 8 --supervise_every 30 --episodes 300 --seeds 1 --seed 0 \
     --lambda_load 1 --out "$EXP/$name" "$@" 2>&1 | grep "$FILT"
}
TOWER="--load_signal fa_acc --fitness_target tower --obs_fa_acc"
run "of5_spec_tower_g998"      --method spec    --supervisor guard $TOWER
run "of5_mono_tower_g998"      --method mono    --supervisor guard $TOWER
run "of5_specsc_tower_g998"    --method spec_sc --supervisor guard $TOWER
run "of5_spec_llm_tower_g998"  --method spec    --supervisor llm   $TOWER
run "of5_spec_blade_g998"      --method spec    --supervisor guard
run "of5_mono_blade_g998"      --method mono    --supervisor guard
echo "=== held-out evaluation (seeds 2,3) $(date) ==="
i=0
for r in of5_spec_tower_g998 of5_mono_tower_g998 of5_specsc_tower_g998 of5_spec_llm_tower_g998 of5_spec_blade_g998 of5_mono_blade_g998; do
  for ck in ckpt_best.pt ckpt_last.pt; do
    [ -f "$EXP/$r/$ck" ] || continue
    echo "--- $r $ck ---"
    python scripts/evaluate.py --run "$EXP/$r" --ckpt $ck --backend openfast --means 8 12.5 15 --seeds 2 3 --workers 6 \
       --port0 $((6500 + 20 * i)) --tag "heldout_s23_${ck%.pt}" 2>&1 | grep "F=\|U8:\|U12.5:\|U15:\|Traceback\|Error"
    i=$((i + 1))
  done
done
echo "CAMPAIGN_OF5_DONE $(date)"
