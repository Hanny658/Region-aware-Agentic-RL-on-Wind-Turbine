#!/bin/bash
# Seed extension of the supervisor comparison (2026-09-09, user decision).
#
# After fixing caveat C1 (one canonical wind bank) the 5-seed result reversed the earlier "the
# supervisor's identity does not matter" conclusion: llm_fork beat random_fork in 5/5 seeds
# (+5.44 F, paired t p = 0.092) and guard in 5/5 (+3.48, p = 0.041). At n = 5 the *exact*
# sign-flip test cannot go below p = 0.0625, so the central claim is capped by resolution, not by
# the data. This extension takes it to n = 7 and B1 to n = 5:
#
#   n1_llmfork_s5,  n1_llmfork_s6     llm_fork   -> 7 seeds
#   n1_randfork_s5, n1_randfork_s6    random_fork-> 7 seeds  (paired with the above)
#   n1_llmsingle_s3, n1_llmsingle_s4  single-proposal LLM (no fork, no dry run) -> 5 seeds
#
# Protocol identical to campaign_a1b1_seeds.sh (= night1/night2): spec, tower objective,
# gamma 0.998, train S1, supervisor F on S1+S2, violation-only guard, 300 episodes, held-out
# S3-S6, both checkpoints. Two lanes x 8 workers, resumable.
# Usage: ~/wtrl/run.sh bash scripts/wsl/campaign_a1b1_ext_seeds.sh
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error\|retries'

run() {  # name port0 extra...
  local name=$1 port=$2; shift 2
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name"
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --workers 8 --supervise_every 30 --episodes 300 \
     --seeds 1 --eval_seeds 1 2 --lambda_load 1 --rollback_on violation \
     --load_signal fa_acc --fitness_target tower --obs_fa_acc \
     --port0 "$port" --out "$EXP/$name" "$@" 2>&1 | grep "$FILT"
}

lane() {  # port0 name...
  local port=$1; shift
  for name in "$@"; do
    local s=${name##*_s}
    case "$name" in
      n1_llmfork_s*)   run "$name" "$port" --method spec --supervisor llm_fork    --seed "$s" --n_candidates 3 ;;
      n1_randfork_s*)  run "$name" "$port" --method spec --supervisor random_fork --seed "$s" --n_candidates 3 ;;
      n1_llmsingle_s*) run "$name" "$port" --method spec --supervisor llm         --seed "$s" --no_dry_run ;;
    esac
  done
}

echo "=== seed extension start $(date) ==="
lane 5800 n1_llmfork_s5 n1_randfork_s6 n1_llmsingle_s3 > ~/wtrl/exp/ext_laneA.log 2>&1 &
PID_A=$!
lane 6100 n1_randfork_s5 n1_llmfork_s6 n1_llmsingle_s4 > ~/wtrl/exp/ext_laneB.log 2>&1 &
PID_B=$!
wait $PID_A $PID_B
echo "=== training done $(date) ==="

echo "=== held-out evaluation S3-S6 $(date) ==="
i=0
for r in n1_llmfork_s5 n1_llmfork_s6 n1_randfork_s5 n1_randfork_s6 n1_llmsingle_s3 n1_llmsingle_s4; do
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
echo "=== statistics ==="
python scripts/dev/stats_table.py || true
echo "EXT_DONE $(date)"
