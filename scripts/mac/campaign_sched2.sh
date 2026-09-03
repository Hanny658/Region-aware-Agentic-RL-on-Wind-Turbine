#!/bin/bash
# sched2 (2026-09-01, mac): episode-indexed vs competence-indexed replay of the SAME distilled
# curriculum (donor n1_llmfork_s3, ~/wtrl/exp/schedule_n2_llmfork_s3.json with F_gate), 5 RL seeds
# per arm. Protocol otherwise = night1/night2 (spec, tower, gamma 0.998, train S1, supervisor F on
# S1+S2, violation-only rollback, 300 ep, 8 training workers). Held-out S3-S6 evaluation runs
# interleaved between training runs at 4 workers (thermal breather, decision 2026-09-01).
# Resumable: summary.json / eval json skip. Tests litreview item: IPBT's RL schedule-replay
# negative result + the novel competence re-indexing.
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error\|retries'
SCHED=~/wtrl/exp/schedule_n2_llmfork_s3.json
[ -f "$SCHED" ] || { echo "missing $SCHED"; exit 1; }

run() {  # name supervisor seed extra...
  local name=$1 supv=$2 seed=$3; shift 3
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name"
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --method spec --workers 8 --supervise_every 30 --episodes 300 \
     --seeds 1 --eval_seeds 1 2 --lambda_load 1 --rollback_on violation \
     --load_signal fa_acc --fitness_target tower --obs_fa_acc \
     --supervisor "$supv" --knob_schedule "$SCHED" --seed "$seed" \
     --port0 5800 --out "$EXP/$name" "$@" 2>&1 | grep "$FILT"
}

I=0
ev() {  # name — held-out S3-S6 at 4 workers, both ckpts
  local r=$1
  for ck in ckpt_best.pt ckpt_last.pt; do
    [ -f "$EXP/$r/$ck" ] || continue
    tag="heldout_s3456_${ck%.pt}"
    [ -f "$EXP/$r/eval_${tag}.json" ] && { echo "skip $r $tag"; continue; }
    echo "--- eval $r $ck  $(date) ---"
    python scripts/evaluate.py --run "$EXP/$r" --ckpt "$ck" --backend openfast --means 8 12.5 15 \
       --seeds 3 4 5 6 --workers 4 --port0 $((6400 + 20 * (I % 8))) --tag "$tag" 2>&1 \
       | grep "F_strict\|Traceback\|Error"
    I=$((I + 1))
  done
}

for s in 0 1 2 3 4; do
  run "sched2_ep_s$s"   schedule      "$s"
  ev  "sched2_ep_s$s"
  run "sched2_comp_s$s" schedule_comp "$s"
  ev  "sched2_comp_s$s"
done
echo "=== summary ==="
python scripts/dev/heldout_table.py "~/wtrl/exp/sched2_*" || true
echo "CAMPAIGN_SCHED2_DONE $(date)"
