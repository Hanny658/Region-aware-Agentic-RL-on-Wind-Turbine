#!/bin/bash
# Master runner for the pre-manuscript experiment programme (2026-09-08).
#   A3  robustness stress sweep on the tower-objective arms        (fixes C2, ~1 h, eval only)
#   A2  GSPI + ROSCO tower-damper reference controller             (fixes C4, ~0.5 h, eval only)
#   A1  mono x5 + llm_fork/random_fork s0-s2 on one wind bank      (fixes C1 + missing mono evidence)
#   B1  single-proposal LLM arm x3 (isolates fork verification)    (A1 and B1 share one campaign)
#   B2  600 s re-evaluation of the champions                       (fixes C5, ~2 h, eval only)
#   B3  paired statistics with the exact permutation floor         (fixes C6, seconds)
# Every stage is resumable; re-running skips finished runs and evaluations.
# Usage: ~/wtrl/run.sh bash scripts/wsl/campaign_manuscript.sh
set -u
cd "$(dirname "$0")/../.."
echo "########## MANUSCRIPT CAMPAIGN START $(date) ##########"

for stage in a3_stress "a2_towerdamper auto" a1b1_seeds b2_600s; do
  set -- $stage
  name=$1; shift
  echo; echo "########## stage $name  $(date) ##########"
  bash "scripts/wsl/campaign_$name.sh" "$@" || echo "STAGE_FAILED $name (continuing)"
done

echo; echo "########## B3 statistics + final tables  $(date) ##########"
python scripts/dev/stats_table.py || true
echo; echo "--- held-out table (all tower-objective arms) ---"
python scripts/dev/heldout_table.py "~/wtrl/exp/n1_*" "~/wtrl/exp/tq_off_*" "~/wtrl/exp/sched2_*" || true
echo; echo "--- paper metric table ---"
python scripts/dev/paper_table.py "~/wtrl/exp/n1_*" "~/wtrl/exp/tq_off_*" "~/wtrl/exp/sched2_ep_*" || true
echo "########## MANUSCRIPT CAMPAIGN DONE $(date) ##########"
