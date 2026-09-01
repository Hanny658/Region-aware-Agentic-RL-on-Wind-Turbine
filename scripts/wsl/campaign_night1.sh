#!/bin/bash
# Overnight campaign (2026-08-30): 5 supervisors x 3 RL seeds on OpenFAST, tower objective,
# gamma 0.998, train S1, supervisor F on S1+S2, held-out S3-S6. Two lanes run concurrently
# (2 x 8 workers); every step is resumable (existing summary.json / eval json => skipped).
# Usage: ~/wtrl/run.sh bash ~/wtrl/campaign_night1.sh
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error\|retries'
SCHED=~/wtrl/exp/schedule_of5_llm.json

echo "=== stage 0: wind S4-S6 + baselines  $(date) ==="
python scripts/gen_wind.py --means 8 12.5 15 --seeds 4 5 6 --ti 8 --time 200 --out ~/wtrl/wind --jobs 3 2>&1 | tail -3
python scripts/make_baselines.py --backend toy --means 8 12.5 15 --seeds 4 5 6 2>&1 | grep "written"
python scripts/make_baselines.py --backend openfast --means 8 12.5 15 --seeds 4 5 6 --jobs 6 --port0 5700 2>&1 | grep "written"

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

lane() {  # lane_id port0 run-name list...
  local lane=$1 port=$2; shift 2
  for name in "$@"; do
    case "$name" in
      n1_guard_s*)    run "$name" "$port" --supervisor guard    --seed "${name##*_s}" ;;
      n1_mono_s*)     run "$name" "$port" --supervisor guard    --seed "${name##*_s}" --method mono ;;
      n1_llmfork_s*)  run "$name" "$port" --supervisor llm_fork --seed "${name##*_s}" --n_candidates 3 ;;
      n1_randfork_s*) run "$name" "$port" --supervisor random_fork --seed "${name##*_s}" --n_candidates 3 ;;
      n1_sched_s*)    run "$name" "$port" --supervisor schedule --seed "${name##*_s}" --knob_schedule "$SCHED" ;;
    esac
  done
}

# lane A (ports 5800+), lane B (ports 6100+) — interleaved so both lanes have similar total time
lane A 5800 n1_guard_s0 n1_llmfork_s0 n1_randfork_s0 n1_sched_s1 n1_mono_s1 n1_llmfork_s2 n1_randfork_s2 \
  > ~/wtrl/exp/night1_laneA.log 2>&1 &
PID_A=$!
lane B 6100 n1_mono_s0 n1_sched_s0 n1_guard_s1 n1_llmfork_s1 n1_randfork_s1 n1_guard_s2 n1_mono_s2 n1_sched_s2 \
  > ~/wtrl/exp/night1_laneB.log 2>&1 &
PID_B=$!
wait $PID_A $PID_B
echo "=== training done $(date) ==="

echo "=== held-out evaluation S3-S6 $(date) ==="
i=0
for r in n1_guard_s0 n1_guard_s1 n1_guard_s2 n1_mono_s0 n1_mono_s1 n1_mono_s2 \
         n1_llmfork_s0 n1_llmfork_s1 n1_llmfork_s2 n1_randfork_s0 n1_randfork_s1 n1_randfork_s2 \
         n1_sched_s0 n1_sched_s1 n1_sched_s2; do
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
python scripts/dev/heldout_table.py "~/wtrl/exp/n1_*"
echo "CAMPAIGN_NIGHT1_DONE $(date)"
