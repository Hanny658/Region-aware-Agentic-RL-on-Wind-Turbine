#!/bin/bash
# night2 (2026-09-01, mac): extend night1 to 5 RL seeds — seeds 3,4 for the three supervised
# variants. Same protocol as campaign_night1.sh: spec, tower objective, gamma 0.998, train S1,
# supervisor F on S1+S2, violation-only guard, 300 ep, held-out S3-S6, best checkpoint.
# Wind bank: local mac bank (regenerated 2026-09-01; NOT seed-paired with the old WSL bank).
# schedule runs are skipped until ~/wtrl/exp/schedule_of5_llm.json is copied from the old laptop
# (re-run this script afterwards; finished runs are skipped via summary.json / eval json).
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error\|retries'
SCHED=~/wtrl/exp/schedule_of5_llm.json

run() {  # name port0 extra...
  local name=$1 port=$2; shift 2
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name"
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --method spec --workers 8 --supervise_every 30 --episodes 300 \
     --seeds 1 --eval_seeds 1 2 --lambda_load 1 --rollback_on violation \
     --load_signal fa_acc --fitness_target tower --obs_fa_acc \
     --port0 "$port" --out "$EXP/$name" "$@" 2>&1 | grep "$FILT"
}

for s in 3 4; do
  run "n1_llmfork_s$s"  5800 --supervisor llm_fork    --seed "$s" --n_candidates 3
  run "n1_randfork_s$s" 5800 --supervisor random_fork --seed "$s" --n_candidates 3
  if [ -f "$SCHED" ]; then
    run "n1_sched_s$s"  5800 --supervisor schedule --seed "$s" --knob_schedule "$SCHED"
  else
    echo "SCHED_MISSING: $SCHED not found — skipping n1_sched_s$s"
  fi
done
echo "=== training done $(date) ==="

echo "=== held-out evaluation S3-S6 $(date) ==="
i=0
for r in n1_llmfork_s3 n1_llmfork_s4 n1_randfork_s3 n1_randfork_s4 n1_sched_s3 n1_sched_s4; do
  for ck in ckpt_best.pt ckpt_last.pt; do
    [ -f "$EXP/$r/$ck" ] || continue
    tag="heldout_s3456_${ck%.pt}"
    [ -f "$EXP/$r/eval_${tag}.json" ] && { echo "skip $r $tag"; continue; }
    echo "--- $r $ck  $(date) ---"
    python scripts/evaluate.py --run "$EXP/$r" --ckpt "$ck" --backend openfast --means 8 12.5 15 \
       --seeds 3 4 5 6 --workers 8 --port0 $((6400 + 20 * (i % 8))) --tag "$tag" 2>&1 \
       | grep "F_strict\|Traceback\|Error"
    i=$((i + 1))
  done
done
echo "=== summary ==="
python scripts/dev/heldout_table.py "~/wtrl/exp/n1_*" || true
echo "CAMPAIGN_NIGHT2_DONE $(date)"
