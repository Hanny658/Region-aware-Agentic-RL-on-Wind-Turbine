#!/bin/bash
# C7 (2026-09-10): clean re-run of the competence-indexed schedule replay.
#
# The original `sched2_comp` verdict ("competence re-indexing adds nothing") is contaminated: after
# a guardrail rollback the trainer masks F with the historical best, and the competence gate read
# that masked value, so aggressive curriculum entries were applied straight into a crash
# (s2: 4 of 5 entries applied inside rollback decisions, s3: 1 of 5; s0/s1/s4 clean). The gate was
# fixed 2026-09-02 to read `F_measured` (llm/supervisor.py) — this re-runs the arm with the fix.
#
# The episode-indexed arm `sched2_ep_s0-4` is NOT affected by that bug and is reused as the control.
# Same curriculum for both (distilled from n1_llmfork_s3), same protocol as night1/night2:
# spec, tower objective, gamma 0.998, train S1, supervisor F on S1+S2, violation-only rollback,
# 300 episodes, held-out S3-S6 on both checkpoints. Two lanes x 8 workers, resumable.
# Usage: ~/wtrl/run.sh bash scripts/wsl/campaign_c7_sched_comp_clean.sh
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error\|retries'
SCHED=~/wtrl/schedules/schedule_n2_llmfork_s3.json
[ -f "$SCHED" ] || { echo "missing curriculum $SCHED"; exit 1; }

run() {  # name port seed
  local name=$1 port=$2 seed=$3
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name"
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --method spec --workers 8 --supervise_every 30 \
     --episodes 300 --seeds 1 --eval_seeds 1 2 --lambda_load 1 --rollback_on violation \
     --load_signal fa_acc --fitness_target tower --obs_fa_acc \
     --supervisor schedule_comp --knob_schedule "$SCHED" --seed "$seed" \
     --port0 "$port" --out "$EXP/$name" 2>&1 | grep "$FILT"
}

lane() {  # port seed...
  local port=$1; shift
  for s in "$@"; do run "sched3_comp_s$s" "$port" "$s"; done
}

echo "=== C7 clean competence-indexed replay, start $(date) ==="
lane 5800 0 2 4 > ~/wtrl/exp/c7_laneA.log 2>&1 &
PID_A=$!
lane 6100 1 3   > ~/wtrl/exp/c7_laneB.log 2>&1 &
PID_B=$!
wait $PID_A $PID_B
echo "=== training done $(date) ==="

echo "=== held-out evaluation S3-S6 $(date) ==="
i=0
for s in 0 1 2 3 4; do
  r="sched3_comp_s$s"
  for ck in ckpt_best.pt ckpt_last.pt; do
    [ -f "$EXP/$r/$ck" ] || continue
    tag="heldout_s3456_${ck%.pt}"
    [ -f "$EXP/$r/eval_${tag}.json" ] && { echo "skip $r $tag"; continue; }
    echo "--- $r $ck  $(date +%H:%M:%S) ---"
    python scripts/evaluate.py --run "$EXP/$r" --ckpt "$ck" --backend openfast --means 8 12.5 15 \
       --seeds 3 4 5 6 --workers 8 --port0 $((6400 + 20 * (i % 8))) --tag "$tag" 2>&1 \
       | grep "F_strict\|Traceback\|Error"
    i=$((i + 1))
  done
done

echo "=== second held-out wind set S7-S10 (same treatment as every other arm, roadmap S22) ==="
for s in 0 1 2 3 4; do
  r="sched3_comp_s$s"
  [ -f "$EXP/$r/eval_heldout2_s78910.json" ] && { echo "skip $r"; continue; }
  python scripts/evaluate.py --run "$EXP/$r" --ckpt ckpt_best.pt --backend openfast \
     --means 8 12.5 15 --seeds 7 8 9 10 --workers 8 --port0 6750 --tag heldout2_s78910 2>&1 \
     | grep "F_strict\|Traceback\|Error"
done

echo "=== verdict: clean competence-indexed vs episode-indexed replay ==="
python scripts/dev/stats_table.py --a "~/wtrl/exp/sched3_comp_s*" --b "~/wtrl/exp/sched2_ep_s*" \
   --label "competence-indexed (clean) vs episode-indexed replay" || true
python scripts/dev/stats_table.py --eval eval_heldout2_s78910.json \
   --a "~/wtrl/exp/sched3_comp_s*" --b "~/wtrl/exp/sched2_ep_s*" \
   --label "same, on the second held-out wind set" || true
echo "--- for reference, the contaminated arm:"
python scripts/dev/heldout_table.py "~/wtrl/exp/sched2_comp_s*" "~/wtrl/exp/sched3_comp_s*" || true
echo "C7_DONE $(date)"
