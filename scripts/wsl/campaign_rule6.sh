#!/bin/bash
# The completed rule, read on both leg-1 campaigns (2026-10-17, roadmap s62.4a - pre-registered before this ran)
#
# s62.3 showed that a per-wind screen with two episodes per wind is gamed by an optimiser that is told it: the
# agent drove every run to the screen's edge and five of six selections lost 1.6-2.2 points of tower-base DEL at
# 24 m/s on fresh seeds. The census of every tuning measured off the design winds on both seed sets (36 rows,
# s62.4) says what the screen cannot see at one wind speed the set-mean margin can: the 12 tunings with a
# design-set set-mean tower-base reduction of at least 6.0 points all ship, 2 of the 24 below it do. This pass
# re-selects every one of the 39 runs of s59 and s62.3 under the completed rule - feasible on the set mean, worst
# single-wind load term >= 0 AND set-mean tower-base reduction >= 6.0, all on the design set - evaluates the
# selections not already measured on held-out seeds 3-6 and fresh seeds 7-10, and decides shipping under the
# UNCHANGED published rule (both loads within 1.0 point at every wind speed on both sets). No search is re-run.
#
#   setsid nohup bash scripts/wsl/campaign_rule6.sh > ~/wtrl/exp/rule6.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_rule6
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_rule6.sid"

# the 600 s protocol lives in its own bank and its own paired baselines (s59.6)
export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600

A=$EXP/roscosearch_native
B=$EXP/roscosearch_margin
# one off-design cache for both campaigns: same protocol, same parameter hash
mkdir -p "$A/_verify/cache" "$B/_verify/cache"
cp -n "$B"/_verify/cache/eval_* "$A/_verify/cache/" 2>/dev/null

echo "=== s62.4a the completed rule (worst single wind >= 0 AND set-mean tower >= 6.0 on the design set): re-select all 39 runs and verify the new selections $(date +%H:%M)"
$RUN python scripts/dev/rosco_perwind_select.py --roots "$A" "$B" --verify_root "$A" \
  --verify --tol 0.0 --tower_margin 6.0 --ship_tol 1.0 --port0 8300
cp -n "$A"/_verify/cache/eval_* "$B/_verify/cache/" 2>/dev/null

echo ""
echo "=== the delivery counts under the completed rule, shipping rule unchanged: the s59 campaign $(date +%H:%M)"
$RUN python scripts/dev/rosco_proposer_test.py --root "$A" --screen_tol 0.0 --tower_margin 6.0 \
  --csv docs/tables/rosco_proposer_test_rule6_s59.csv

echo ""
echo "=== the same for the s62.3 campaign (every arm screened, the agent told the screen) $(date +%H:%M)"
$RUN python scripts/dev/rosco_proposer_test.py --root "$B" --screen_tol 0.0 --tower_margin 6.0 \
  --csv docs/tables/rosco_proposer_test_rule6_s62.csv

echo ""
echo "=== done $(date +%H:%M)"
