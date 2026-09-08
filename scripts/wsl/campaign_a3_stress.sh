#!/bin/bash
# A3 (2026-09-08): robustness stress sweep on the TOWER-objective arms.
# Fixes caveat C2: the published sweep (roadmap 17) ran on blade-objective runs, where the
# tower column is a by-product metric rather than the optimisation target.
# Pure evaluation, no training. Resumable (existing eval json => skipped).
# Usage: ~/wtrl/run.sh bash scripts/wsl/campaign_a3_stress.sh
set -u
EXP=~/wtrl/exp
RUNS="tq_off_s0 tq_off_s1 tq_off_s2 tq_off_s3 tq_off_s4
      sched2_ep_s0 sched2_ep_s1 sched2_ep_s2 sched2_ep_s3 sched2_ep_s4
      n1_llmfork_s3 n1_llmfork_s4 n1_randfork_s3 n1_randfork_s4"

ev() {  # run tag means... -- ti seeds...
  local run=$1 tag=$2; shift 2
  [ -d "$EXP/$run" ] || { echo "missing $run"; return; }
  [ -f "$EXP/$run/eval_${tag}.json" ] && { echo "skip $run $tag"; return; }
  echo "--- $run $tag  $(date +%H:%M:%S)"
  python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt --backend openfast \
     --workers 8 --port0 6400 --tag "$tag" "$@" 2>&1 | grep "F_strict\|Traceback\|Error"
}

for r in $RUNS; do
  ev "$r" robust_ti14 --means 8 12.5 15 --seeds 3 4 --ti 14
  ev "$r" robust_ti22 --means 15        --seeds 3 4 --ti 22
  ev "$r" robust_u18  --means 18        --seeds 3 4 --ti 8
done
echo "A3_DONE $(date)"
