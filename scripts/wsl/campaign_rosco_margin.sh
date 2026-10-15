#!/bin/bash
# Leg 1 under the margin screen: a proposer that is told the rule that ships (2026-10-16, roadmap s62.2)
#
# s59.6 closed the first leg-1 campaign with the screen, not the search, as the thing that fails: the rule the
# proposers were told and selected under was the set-mean rule, the per-wind load terms were recorded but never
# shown to the model, and the agent did exactly what it was told - it walked up the regulation front and every
# selection lost the tower-base load at 24 m/s off the design winds. s62.1 read the existing candidates under a
# screen with a margin (worst single-wind load term >= 0 on the design set) and found the agent's own candidates
# deliver where the screen lets them. This campaign runs the same search with that screen built in:
#
#   every proposer selects under it (the local search's incumbent, Bayesian optimisation's feasibility model,
#   the run's own best.json), and the llm proposer is SHOWN the two load terms at each of the seven wind speeds
#   for every tuning tried and TOLD the rule that ships, the resolution of the design set and what has failed
#   before (scripts/rosco_search.py --screen_tol 0.0: SYSTEM_SCREEN, SHIP_TEXT, _per_wind_cells).
#
# Same box, same shipped start, same budget of 24 candidates, same 14-episode design set (two things are not
# changed at once: the design set's resolution is a separate lever), same evaluation. Arms llm, local, bayes,
# six repeats each; random is not re-run (0 of 3 with any feasible gain in s59, and it cannot be "told" a rule).
# The agent arm runs first so that it can be read on its own if the campaign is stopped.
#
# Endpoints are pre-registered in roadmap s62.2 BEFORE this runs: per arm, the number of runs whose selection
# holds both loads at every wind speed on 28 held-out AND 28 fresh episodes (the unchanged shipping rule), and
# the regulation of the ones that do. No statement that the agent beats a classical search follows from it.
#
# The evaluation cache of the s53/s59 campaign is copied in (same protocol, same parameter hash), so a tuning
# any proposer revisits costs nothing; a fresh root keeps the two campaigns' tables apart.
#
#   setsid nohup bash scripts/wsl/campaign_rosco_margin.sh > ~/wtrl/exp/rosco_margin.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_rosco_margin
#
# Cost from the s53 log: about 1 h 45 min per run, so about 32 h for the eighteen, then the per-wind
# verification of the selections that are not already cached (56 episodes each, about 20 min).
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
ROOT=$EXP/roscosearch_margin
OLD=$EXP/roscosearch_native
BUDGET=${BUDGET:-24}
REPEATS=${REPEATS:-"0 1 2 3 4 5"}
ARMS=${ARMS:-"llm local bayes"}
MEANS="12 14 16 18 20 22 24"
PROT=(--means $MEANS --ti B --seeds 1 2 --episode_s 600 --workers 7 --batch 2 --parallel 2 --screen_tol 0.0)
sid() { ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_rosco_margin.sid"; }
sid

# the LLM arm needs outbound network; on this machine WSL sometimes has none and the call is relayed through Windows
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi

# the 600 s protocol lives in its own bank and its own paired baselines (s59.6)
export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600

# never compete with the s62.1 verification pass for the seven workers
while ps aux | grep -q "[c]ampaign_rule_margin.sh"; do sleep 60; done

echo "=== leg 1 under the margin screen, start $(date)"
echo "    arms $ARMS, repeats $REPEATS, budget $BUDGET, screen: worst single-wind load term >= 0 on the design set"

# warm the caches from the first campaign: identical protocol and parameter hash, so a revisited tuning is free.
# The design-set cache is per RUN directory (Evaluator writes <run>/cache), the off-design one per root.
mkdir -p "$ROOT/_verify/cache"
cp -n "$OLD"/_verify/cache/eval_* "$ROOT/_verify/cache/" 2>/dev/null
echo "    off-design cache warmed: $(ls "$ROOT/_verify/cache" | wc -l) files"

for arm in $ARMS; do
  for r in $REPEATS; do
    if [ -f "$ROOT/${arm}_r$r/best.json" ]; then
      echo "--- $arm r$r skip (finished)"; continue
    fi
    mkdir -p "$ROOT/${arm}_r$r/cache"
    cp -n "$OLD"/*/cache/eval_* "$ROOT/${arm}_r$r/cache/" 2>/dev/null
    echo "--- $arm r$r: $BUDGET candidates $(date +%H:%M), design-set cache warmed with $(ls "$ROOT/${arm}_r$r/cache" | wc -l) files"
    $RUN python scripts/rosco_search.py --proposer "$arm" --repeat "$r" --budget "$BUDGET" "${PROT[@]}" \
        --root "$ROOT" --config_run "$EXP/jg3t_s0" --port0 7800 2>&1 | grep -vE "^\[eval|^ *$"
    sid
  done
done

echo ""
echo "=== every selection under the screen, measured off the design winds $(date +%H:%M)"
$RUN python scripts/dev/rosco_perwind_select.py --roots "$ROOT" --verify_root "$ROOT" \
  --verify --tol 0.0 --ship_tol 1.0 --port0 8300
sid

echo ""
echo "=== the delivery counts, shipping rule unchanged $(date +%H:%M)"
$RUN python scripts/dev/rosco_proposer_test.py --root "$ROOT" --screen_tol 0.0 \
  --csv docs/tables/rosco_proposer_test_margin_campaign.csv

echo ""
echo "=== done $(date +%H:%M)"
