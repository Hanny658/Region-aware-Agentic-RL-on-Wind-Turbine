#!/bin/bash
# SUPERSEDED on 2026-10-11 by campaign_rosco_native.sh. Kept because its 8 finished runs are on disk and
# keep one finding: an unconstrained mean of four reductions makes a parameter search buy regulation with
# tower fatigue. Its box is not usable - two of its four inputs are scalar multipliers on ROSCO's gain
# tables rather than ROSCO tuning quantities, and three of its limits were chosen rather than measured.
# See docs/design_2026-10-11_rosco-native-search.md. Do not start this script.
#
# Who should propose a classical controller's parameters? (2026-10-09)
#
# The agent is used here as a pre-deployment design tool rather than a layer in the loop: starting from the SHIPPED
# ROSCO settings, each proposer spends an identical simulation budget searching the same four-parameter box, and the
# question is how much of the available margin each one finds. Roadmap s29 already supplies the reason to ask: its
# random half found exactly one good point in twenty draws, and that single point was the winner.
#
# Arms, all on the range protocol (12-24 m/s, class B, 600 s, TurbSim seeds 1-2 = 14 episodes per candidate),
# scored by J against the ORIGINAL shipped-GSPI baselines, so J > 0 reads directly as "better than shipped":
#
#   random  r0 r1 r2    log-uniform draws from the box
#   local   r0 r1 r2    one random point, then a (1+lambda) local search around the incumbent
#   bayes   r0 r1 r2    GP surrogate with expected improvement
#   llm     r0 r1 r2    a reasoning model shown the box, the objective and the whole history
#
# Three repeats per arm because one run cannot separate a proposer from a lucky draw. Each run evaluates the shipped
# settings first, free of budget, which doubles as an identity check of the template-editing path (J should be ~0).
#
# Resumable at the candidate: every evaluation is cached by parameter hash and protocol, and each run appends to its
# own history.jsonl, so stopping and restarting the campaign loses at most the candidate in flight.
#
#   setsid nohup bash scripts/wsl/campaign_rosco_search.sh > ~/wtrl/exp/rosco_search.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_rosco_search
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
ROOT=$EXP/roscosearch
BUDGET=${BUDGET:-20}
REPEATS=${REPEATS:-"0 1 2"}
ARMS=${ARMS:-"random local bayes llm"}
MEANS="12 14 16 18 20 22 24"
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_rosco_search.sid"

# the LLM arm needs outbound network; on this machine WSL sometimes has none and the call is relayed through Windows
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi

export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600
echo "=== proposer comparison start $(date) (budget $BUDGET, repeats $REPEATS, arms $ARMS)"

port=7800
for rep in $REPEATS; do
  for arm in $ARMS; do
    d="$ROOT/${arm}_r${rep}"
    if [ -f "$d/best.json" ]; then echo "skip $arm r$rep (finished)"; continue; fi
    echo "--- $arm r$rep $(date +%H:%M)"
    $RUN python scripts/rosco_search.py --proposer "$arm" --repeat "$rep" --budget "$BUDGET" \
        --batch 2 --parallel 2 --workers 7 --means $MEANS --ti B --seeds 1 2 --episode_s 600 \
        --root "$ROOT" --config_run "$EXP/jg3t_s0" --port0 "$port" 2>&1 | grep -vE "^\[eval|^ *$"
    port=$((port + 40))
    [ "$port" -gt 8600 ] && port=7800
    ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_rosco_search.sid"
  done
done

echo
echo "=== held-out verification of each arm's winner $(date +%H:%M)"
$RUN python scripts/dev/rosco_search_table.py --csv docs/tables/rosco_search.csv --verify 2>&1 | tail -40
echo "=== proposer comparison done $(date) ==="
