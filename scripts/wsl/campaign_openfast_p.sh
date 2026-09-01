#!/bin/bash
# P-campaign (2026-08-30 afternoon): P3 r3only-all-winds (R2 residual 0), P4 tier-aware guard,
# P2 LLM with supervisor F on seeds 1+2 (held-out S3). All: gamma 0.998 default, tower objective,
# tower-accel reward signal, 300 ep, 8 workers, train S1.
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error\|retries'
TOWER="--load_signal fa_acc --fitness_target tower --obs_fa_acc"
run() {
  name=$1; shift
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name" ~/wtrl/runs/work_w*
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --workers 8 --supervise_every 30 --episodes 300 --seeds 1 --seed 0 \
     --lambda_load 1 $TOWER --out "$EXP/$name" "$@" 2>&1 | grep "$FILT"
}
ev() {  # run, seeds..., port
  r=$1; shift; port=$1; shift
  for ck in ckpt_best.pt ckpt_last.pt; do
    [ -f "$EXP/$r/$ck" ] || continue
    echo "--- $r $ck heldout $* ---"
    python scripts/evaluate.py --run "$EXP/$r" --ckpt $ck --backend openfast --means 8 12.5 15 --seeds "$@" --workers 6 \
       --port0 $port --tag "heldout_s$(echo "$@" | tr -d ' ')_${ck%.pt}" 2>&1 | grep "F_strict\|Traceback\|Error"
    port=$((port + 20))
  done
}
run "p3_r3all_tower"        --method r3only --supervisor guard
run "p4_spec_tower_viol"    --method spec   --supervisor guard --rollback_on violation
run "p2_spec_llm_tower_ev2" --method spec   --supervisor llm   --eval_seeds 1 2 --rollback_on violation
echo "=== held-out evaluation $(date) ==="
ev p3_r3all_tower        6800 2 3
ev p4_spec_tower_viol    6850 2 3
ev p2_spec_llm_tower_ev2 6900 3
ev of5_spec_tower_g998   6950 3
ev of5_mono_tower_g998   6970 3
echo "CAMPAIGN_P_DONE $(date)"
