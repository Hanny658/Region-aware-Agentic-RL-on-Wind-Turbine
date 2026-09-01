#!/bin/bash
# P5 (2026-08-30): LLM v2 = fork-verified candidate supervision (RHyVE-style short-horizon forks, K=3
# diverse candidates, 2-seed evaluation, tier-aware guard) vs its random-candidate control and the
# fixed-schedule replay of the E5 LLM trajectory. Tower objective, gamma 0.998. Waits for campaign_p.
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error\|retries'
TOWER="--load_signal fa_acc --fitness_target tower --obs_fa_acc"
while ps aux | grep -q "[c]ampaign_p.sh"; do sleep 60; done
run() {
  name=$1; shift
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name" ~/wtrl/runs/work_w*
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --workers 8 --supervise_every 30 --episodes 300 --seeds 1 --seed 0 \
     --method spec --lambda_load 1 $TOWER --eval_seeds 1 2 --rollback_on violation --out "$EXP/$name" "$@" 2>&1 | grep "$FILT"
}
ev() {
  r=$1; shift; port=$1; shift
  for ck in ckpt_best.pt ckpt_last.pt; do
    [ -f "$EXP/$r/$ck" ] || continue
    echo "--- $r $ck heldout $* ---"
    python scripts/evaluate.py --run "$EXP/$r" --ckpt $ck --backend openfast --means 8 12.5 15 --seeds "$@" --workers 6 \
       --port0 $port --tag "heldout_s$(echo "$@" | tr -d ' ')_${ck%.pt}" 2>&1 | grep "F_strict\|Traceback\|Error"
    port=$((port + 20))
  done
}
run "p5_spec_llmfork_tower"   --supervisor llm_fork    --n_candidates 3
run "p5_spec_randfork_tower"  --supervisor random_fork --n_candidates 3
run "p5_spec_schedule_tower"  --supervisor schedule --knob_schedule ~/wtrl/exp/schedule_of5_llm.json
echo "=== held-out evaluation (seed 3) $(date) ==="
ev p5_spec_llmfork_tower  7000 3
ev p5_spec_randfork_tower 7050 3
ev p5_spec_schedule_tower 7100 3
echo "CAMPAIGN_P5_DONE $(date)"
