#!/bin/bash
# A1 + B1 (2026-09-08): complete the supervisor comparison on ONE canonical wind bank.
#
# A1 fixes caveats C1 (two wind banks) and the missing mono evidence: not a single `mono` run
# survived the retirement of the first machine, so F1 ("region specialisation wins, mono is
# 0/9 strict") currently rests on a table whose artifacts are gone. Re-run here:
#     n1_mono_s0..s4      spec vs mono control, seed-paired with the guard arm (tq_off_s0..s4)
#     n1_llmfork_s0..s2   completes llm_fork to 5 seeds (s3, s4 already exist on this bank)
#     n1_randfork_s0..s2  completes random_fork to 5 seeds
#
# B1 adds the arm that isolates *verification* rather than the proposer:
#     n1_llmsingle_s0..s2 the same LLM, one proposal per decision, applied WITHOUT fork
#                         verification and WITHOUT the toy dry run (--no_dry_run), guardrail
#                         still active. Paired with n1_llmfork_s0..s2 (same RL seeds), this turns
#                         the single-seed P2/E5 anecdote ("unverified LLM supervision is harmful")
#                         into an n=3 paired comparison.
#
# Protocol = night1/night2 verbatim: spec (mono where stated), tower objective, gamma 0.998,
# train S1, supervisor F on S1+S2, violation-only guard, 300 episodes, held-out S3-S6, both ckpts.
# Two lanes x 8 workers (~2.9 GB each; measured 57 s per 8-episode wave at 8 workers).
# Resumable: a run with summary.json and an eval with its json are skipped.
# Usage: ~/wtrl/run.sh bash scripts/wsl/campaign_a1b1_seeds.sh
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
      n1_mono_s*)      run "$name" "$port" --method mono --supervisor guard       --seed "$s" ;;
      n1_llmfork_s*)   run "$name" "$port" --method spec --supervisor llm_fork    --seed "$s" --n_candidates 3 ;;
      n1_randfork_s*)  run "$name" "$port" --method spec --supervisor random_fork --seed "$s" --n_candidates 3 ;;
      n1_llmsingle_s*) run "$name" "$port" --method spec --supervisor llm         --seed "$s" --no_dry_run ;;
    esac
  done
}

echo "=== A1+B1 training start $(date) ==="
lane 5800 n1_mono_s0 n1_llmfork_s0 n1_randfork_s1 n1_llmsingle_s0 n1_mono_s3 n1_llmfork_s2 n1_randfork_s2 \
  > ~/wtrl/exp/a1b1_laneA.log 2>&1 &
PID_A=$!
lane 6100 n1_mono_s1 n1_randfork_s0 n1_llmfork_s1 n1_llmsingle_s1 n1_mono_s2 n1_llmsingle_s2 n1_mono_s4 \
  > ~/wtrl/exp/a1b1_laneB.log 2>&1 &
PID_B=$!
wait $PID_A $PID_B
echo "=== training done $(date) ==="

echo "=== held-out evaluation S3-S6 $(date) ==="
i=0
for r in n1_mono_s0 n1_mono_s1 n1_mono_s2 n1_mono_s3 n1_mono_s4 \
         n1_llmfork_s0 n1_llmfork_s1 n1_llmfork_s2 \
         n1_randfork_s0 n1_randfork_s1 n1_randfork_s2 \
         n1_llmsingle_s0 n1_llmsingle_s1 n1_llmsingle_s2; do
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
echo "=== summary ==="
python scripts/dev/heldout_table.py "~/wtrl/exp/n1_*" "~/wtrl/exp/tq_off_*" "~/wtrl/exp/sched2_ep_*" || true
echo "A1B1_DONE $(date)"
