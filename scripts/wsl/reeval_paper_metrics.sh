#!/bin/bash
# Re-run the night1 held-out evaluations with the paper-metric set (power/speed MSE + DELs).
set -u
EXP=~/wtrl/exp
i=0
for r in n1_guard_s0 n1_guard_s1 n1_guard_s2 n1_mono_s0 n1_mono_s1 n1_mono_s2 \
         n1_llmfork_s0 n1_llmfork_s1 n1_llmfork_s2 n1_randfork_s0 n1_randfork_s1 n1_randfork_s2 \
         n1_sched_s0 n1_sched_s1 n1_sched_s2; do
  [ -f "$EXP/$r/eval_paper_s3456.json" ] && { echo "skip $r"; continue; }
  echo "--- $r  $(date) ---"
  python scripts/evaluate.py --run "$EXP/$r" --ckpt ckpt_best.pt --backend openfast --means 8 12.5 15 \
     --seeds 3 4 5 6 --workers 8 --port0 $((6400 + 20 * (i % 8))) --tag paper_s3456 2>&1 | grep "F_strict\|Traceback\|Error"
  i=$((i + 1))
done
echo "REEVAL_PAPER_DONE $(date)"
