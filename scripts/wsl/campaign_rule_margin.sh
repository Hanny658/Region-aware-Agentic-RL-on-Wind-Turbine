#!/bin/bash
# The margin screen, read on the existing campaign (2026-10-15, roadmap s62.1 - pre-registered before this ran)
#
# s59.6 closed leg 1 with the screen, not the search, as the thing that fails: with two episodes per wind the
# 14-episode design set cannot tell -0.60 from -1.20, so the per-wind rule hands the user the highest row on the
# front and that row loses the tower-base load off the design winds. s55 found that the design-set margin
# separates the members that hold from the ones that do not (holding: worst single wind +0.2 to +0.7; failing:
# -0.4 to -0.9). This pass re-selects every one of the 21 runs under the screen "worst single-wind load term
# >= 0 on the design set" (one point of margin inside the published tolerance), evaluates the selections that
# stage 3 and the s59.6 pass have not already measured (three: bayes_r1#13, llm_r2#12, llm_r5#14) on held-out
# seeds 3-6 and fresh seeds 7-10, and decides shipping under the UNCHANGED published rule (both loads within
# 1.0 point at every wind speed on both sets). No search is re-run; no proposer sees anything new.
#
#   setsid nohup bash scripts/wsl/campaign_rule_margin.sh > ~/wtrl/exp/rule_margin.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_rule_margin
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_rule_margin.sid"

# the 600 s protocol lives in its own bank and its own paired baselines (s59.6: without this the evaluation
# looks for the 150 s fields and dies before it runs an episode)
export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600

echo "=== s62.1 the margin screen: re-select all 21 runs at worst >= 0 and verify the new selections $(date +%H:%M)"
$RUN python scripts/dev/rosco_perwind_select.py --roots "$EXP/roscosearch_native" \
  --verify --tol 0.0 --ship_tol 1.0 --port0 8300

echo ""
echo "=== the delivery counts under the margin screen, shipping rule unchanged $(date +%H:%M)"
$RUN python scripts/dev/rosco_proposer_test.py --root "$EXP/roscosearch_native" \
  --screen_tol 0.0 --csv docs/tables/rosco_proposer_test_margin.csv

echo ""
echo "=== done $(date +%H:%M)"
