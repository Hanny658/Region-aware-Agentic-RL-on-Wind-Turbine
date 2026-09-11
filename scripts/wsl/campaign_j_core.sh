#!/bin/bash
# Stage 3 of roadmap v2 (docs/roadmap_2026-09-11_metric-set-objective.md): the core arms trained and
# checkpoint-selected under the metric-set objective J, all with the fixed critic (--value_norm).
#
#   arm            prefix   supervisor      reward  question
#   guard-v2       jg2      guard           v2      the hand-designed default aligned with J
#   llm_hparam     jhp      llm_hparam      v2      LLM tunes the learner (PPO hyper-parameters)
#   random_hparam  jrhp     random_hparam   v2      same fork search, random candidates (proposer vs search)
#   guard-v1       jg1      guard           v1      the historical reward under J (decision D6)
#   llm_reward     jrw      llm_reward      v2      LLM rewrites the reward expression
#
# Runs are ordered by information value (guard-v2 -> llm_hparam -> random_hparam -> guard-v1 ->
# llm_reward) and interleaved over two lanes (ports 5800 / 6100, 8 workers each).
#
# Pausable / resumable: train.py writes resume_*.pt every --resume_every_s seconds (5 kept); a run
# with resume files and no summary.json is continued with --resume, a finished run is skipped.
# Use scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} on a running campaign.
#
# Usage: ~/wtrl/run.sh bash scripts/wsl/campaign_j_core.sh [seeds...]       (default 0 1 2)
#        ARMS="jg2 jhp" ~/wtrl/run.sh bash scripts/wsl/campaign_j_core.sh 0 1 2
set -u
EXP=~/wtrl/exp
FILT='init\|resume\]\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error\|retries'
SEEDS=${*:-0 1 2}
ARMS=${ARMS:-jg2 jhp jrhp jg1 jrw}
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_j.sid"

arm_args() {  # prefix -> train.py arguments
  case "$1" in
    jg2)  echo "--supervisor guard         --reward v2" ;;
    jhp)  echo "--supervisor llm_hparam    --reward v2 --n_candidates 3" ;;
    jrhp) echo "--supervisor random_hparam --reward v2 --n_candidates 3" ;;
    jg1)  echo "--supervisor guard         --reward v1" ;;
    jrw)  echo "--supervisor llm_reward    --reward v2 --n_candidates 3" ;;
    # fairness step 1 (2026-09-11): reward v3 = speed term gated by the wind label; the weight
    # namespace searched under J by the same fork verification the MPC's grid amounts to
    jg3)  echo "--supervisor guard         --reward v3" ;;
    jwf3) echo "--supervisor random_fork   --reward v3 --n_candidates 3" ;;
    jlf3) echo "--supervisor llm_fork      --reward v3 --n_candidates 3" ;;
    jhp3) echo "--supervisor llm_hparam    --reward v3 --n_candidates 3" ;;
    jrhp3) echo "--supervisor random_hparam --reward v3 --n_candidates 3" ;;
    jrw3) echo "--supervisor llm_reward    --reward v3 --n_candidates 3" ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

run() {  # name port seed extra...
  local name=$1 port=$2 seed=$3; shift 3
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  local resume=""
  if ls "$EXP/$name"/resume_*.pt >/dev/null 2>&1; then
    resume="--resume"; echo "=== $name  RESUME from $(ls "$EXP/$name"/resume_*.pt | tail -1 | xargs basename)  $(date) ==="
  else
    rm -rf "$EXP/$name"; echo "=== $name  $(date) ==="
  fi
  python scripts/train.py --backend openfast --method spec --workers 8 --supervise_every 30 \
     --episodes 300 --seeds 1 --eval_seeds 1 2 --lambda_load 1 --rollback_on drop \
     --load_signal fa_acc --fitness_target tower --obs_fa_acc --value_norm --objective J \
     --resume_every_s 300 --resume_keep 5 $resume "$@" \
     --seed "$seed" --port0 "$port" --out "$EXP/$name" 2>&1 | grep "$FILT"
}

lane() {  # lane-index port  : runs every job whose index == lane (mod 2), in order
  local lane=$1 port=$2 i=0
  for arm in $ARMS; do
    for s in $SEEDS; do
      if [ $((i % 2)) -eq "$lane" ]; then
        run "${arm}_s$s" "$port" "$s" $(arm_args "$arm")
      fi
      i=$((i + 1))
    done
  done
}

echo "=== stage 3 (J core): arms [$ARMS] seeds [$SEEDS]  start $(date) ==="
lane 0 5800 >> "$EXP/j_core_laneA.log" 2>&1 &
lane 1 6100 >> "$EXP/j_core_laneB.log" 2>&1 &
wait
echo "=== training done $(date) ==="

# ---------------------------------------------------------------- held-out evaluation (both wind sets)
j=0
for arm in $ARMS; do
  for s in $SEEDS; do
    r="${arm}_s$s"
    [ -f "$EXP/$r/ckpt_best.pt" ] || continue
    for spec in "heldout_s3456_ckpt_best:3 4 5 6" "heldout2_s78910:7 8 9 10"; do
      tag=${spec%%:*}; seeds=${spec#*:}
      [ -f "$EXP/$r/eval_${tag}.json" ] && continue
      echo "--- $r $tag $(date +%H:%M:%S)"
      python scripts/evaluate.py --run "$EXP/$r" --ckpt ckpt_best.pt --backend openfast --means 8 12.5 15 \
         --seeds $seeds --workers 8 --port0 $((6400 + 20 * (j % 8))) --tag "$tag" 2>&1 \
         | grep "F_strict\|   J=\|Traceback\|Error"
      j=$((j + 1))
    done
  done
done

echo "=== critic health ==="
for arm in $ARMS; do python scripts/dev/critic_health.py --runs "$EXP/${arm}_s*" || true; done
echo "=== paper table (held-out S3-S6, best-by-J checkpoints) ==="
python scripts/dev/paper_table.py --eval eval_heldout_s3456_ckpt_best.json $(for a in $ARMS; do echo "$EXP/${a}_s*"; done) || true
echo "STAGE3_DONE $(date)"
