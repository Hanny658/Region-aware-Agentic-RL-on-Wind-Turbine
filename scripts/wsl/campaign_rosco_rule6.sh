#!/bin/bash
# Leg 1 under the completed rule: a proposer that is told the whole rule and nothing else (2026-10-17, roadmap s62.4b)
#
# s62.3 ran the leg-1 search with the per-wind screen built into every arm and the agent told the screen: the
# agent drove every run to the screen's edge and shipped in 1 of 6, because a screen with two episodes per wind
# cannot see the 1.6-2.2 points of tower-base DEL that its selections lose at 24 m/s on wind realisations it has
# not seen. The census of s62.4 says the set-mean tower-base margin can: 12 of 12 verified tunings at >= 6.0
# points ship, 2 of 24 below. This campaign runs the same search with the rule COMPLETED for every arm -
#
#   feasible on the set mean, worst single-wind load term >= 0 AND set-mean tower-base reduction >= 6.0 points,
#   all on the 14 design episodes (scripts/rosco_search.py --screen_tol 0.0 --tower_margin 6.0): the local
#   search's incumbent, Bayesian optimisation's feasibility model (the tower GP thresholded at the margin), each
#   run's best.json, and the llm proposer's task text, which states the rule and the design set's resolution
#
# and with the prompt leak of s62.3 removed: the task text no longer says where earlier campaigns found the
# ladder (the filter corner, the damper) nor what held before. The agent is told the rule; no arm is told a result.
#
# Same box, same shipped start, same budget of 24 candidates, same 14-episode design set, same evaluation.
# Repeats 6-11 so that no arm replays an earlier trajectory (the local search of s62.3 replayed four of its six
# s59 runs because the screen never changed its incumbent). Arms llm, local, bayes; random is not re-run.
# The agent arm runs first so that it can be read on its own if the campaign is stopped.
#
# Endpoints are pre-registered in roadmap s62.4b BEFORE this runs: per arm, the number of runs whose selection
# ships under the unchanged rule, and the precision of the completed screen (selections that ship / selections).
# No statement that the agent beats a classical search follows from it.
#
#   setsid nohup bash scripts/wsl/campaign_rosco_rule6.sh > ~/wtrl/exp/rosco_rule6.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_rosco_rule6
#
# Cost: about 1 h 45 min per run without cache hits, so up to 32 h for the eighteen, then the per-wind
# verification of the selections that are not already cached (56 episodes each, about 20 min).
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
ROOT=$EXP/roscosearch_rule6
OLD1=$EXP/roscosearch_native
OLD2=$EXP/roscosearch_margin
BUDGET=${BUDGET:-24}
REPEATS=${REPEATS:-"6 7 8 9 10 11"}
ARMS=${ARMS:-"llm local bayes"}
MEANS="12 14 16 18 20 22 24"
PROT=(--means $MEANS --ti B --seeds 1 2 --episode_s 600 --workers 7 --batch 2 --parallel 2 --screen_tol 0.0 --tower_margin 6.0)
sid() { ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_rosco_rule6.sid"; }
sid

# the LLM arm needs outbound network; on this machine WSL sometimes has none and the call is relayed through Windows
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi

# the 600 s protocol lives in its own bank and its own paired baselines (s59.6)
export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600

# never compete with the s62.4a read for the seven workers
while ps aux | grep -q "[c]ampaign_rule6.sh"; do sleep 60; done

echo "=== leg 1 under the completed rule, start $(date)"
echo "    arms $ARMS, repeats $REPEATS, budget $BUDGET, rule: worst single-wind load term >= 0 AND set-mean tower >= 6.0 on the design set"

# warm the caches from both earlier campaigns: identical protocol and parameter hash, so a revisited tuning is free.
# The design-set cache is per RUN directory (Evaluator writes <run>/cache), the off-design one per root.
mkdir -p "$ROOT/_verify/cache"
cp -n "$OLD1"/_verify/cache/eval_* "$OLD2"/_verify/cache/eval_* "$ROOT/_verify/cache/" 2>/dev/null
echo "    off-design cache warmed: $(ls "$ROOT/_verify/cache" | wc -l) files"

for arm in $ARMS; do
  for r in $REPEATS; do
    if [ -f "$ROOT/${arm}_r$r/best.json" ]; then
      echo "--- $arm r$r skip (finished)"; continue
    fi
    mkdir -p "$ROOT/${arm}_r$r/cache"
    cp -n "$OLD1"/*/cache/eval_* "$OLD2"/*/cache/eval_* "$ROOT/${arm}_r$r/cache/" 2>/dev/null
    echo "--- $arm r$r: $BUDGET candidates $(date +%H:%M), design-set cache warmed with $(ls "$ROOT/${arm}_r$r/cache" | wc -l) files"
    $RUN python scripts/rosco_search.py --proposer "$arm" --repeat "$r" --budget "$BUDGET" "${PROT[@]}" \
        --root "$ROOT" --config_run "$EXP/jg3t_s0" --port0 7800 2>&1 | grep -vE "^\[eval|^ *$"
    sid
  done
done

echo ""
echo "=== every selection under the completed rule, measured off the design winds $(date +%H:%M)"
$RUN python scripts/dev/rosco_perwind_select.py --roots "$ROOT" --verify_root "$ROOT" \
  --verify --tol 0.0 --tower_margin 6.0 --ship_tol 1.0 --port0 8300
sid

echo ""
echo "=== the delivery counts, shipping rule unchanged $(date +%H:%M)"
$RUN python scripts/dev/rosco_proposer_test.py --root "$ROOT" --screen_tol 0.0 --tower_margin 6.0 \
  --csv docs/tables/rosco_proposer_test_rule6_campaign.csv

echo ""
echo "=== done $(date +%H:%M)"
