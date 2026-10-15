#!/bin/bash
# Who should propose a classical controller's parameters, in ROSCO's own quantities (2026-10-11)
#
# Replaces campaign_rosco_search.sh, whose box contained two quantities ROSCO does not expose and three limits we
# had chosen rather than measured (docs/paper/design_2026-10-11_rosco-native-search.md). Here the searched quantities are
# omega_pc, zeta_pc, F_LPFCornerFreq, FA_KI and FA_HPFCornerFreq, the gain schedule is regenerated with ROSCO's own
# formula so every candidate is a controller that could ship in a tuning file, every box limit is read off the
# plant, and selection is constrained: maximise the generator-speed regulation reduction subject to tower-base
# load, blade-root load and energy capture being no worse than the shipped controller, ranked by constraint
# domination. No composite score anywhere.
#
# Stage 1  re-derive the paper's tuned baseline in the native parameterisation, with the recipe and budget of
#          roadmap s29 (40 candidates, half random then local), and emit its case template. Two statements in the
#          manuscript depend on this: the tuned baseline was a hand-scaled gain table no tuning file can produce,
#          and the tower damper was judged worthless inside a box that allowed it a quarter of a degree.
# Stage 2  the proposer comparison: random / (1+lambda) local / constrained Bayesian / reasoning model, three
#          repeats each, 24 candidates each, same box, same start, same budget, same evaluation.
# Stage 3  re-score every selected tuning on held-out seeds 3-6 and fresh seeds 7-10.
#
# Each run evaluates the shipped settings first, free of budget; its objective must read exactly 0.000, which is
# the identity check of the template-writing path. Resumable at the candidate: every evaluation is cached by
# parameter hash and protocol and each run appends to its own history.jsonl.
#
#   setsid nohup bash scripts/wsl/campaign_rosco_native.sh > ~/wtrl/exp/rosco_native.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_rosco_native
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
ROOT=$EXP/roscosearch_native
TUNE=$EXP/roscotune_native
BUDGET=${BUDGET:-24}
TUNE_BUDGET=${TUNE_BUDGET:-40}
REPEATS=${REPEATS:-"0 1 2"}
ARMS=${ARMS:-"random local bayes llm"}
STAGES=${STAGES:-"1 2 3"}
MEANS="12 14 16 18 20 22 24"
PROT=(--means $MEANS --ti B --seeds 1 2 --episode_s 600 --workers 7 --batch 2 --parallel 2)
sid() { ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_rosco_native.sid"; }
sid

# the LLM arm needs outbound network; on this machine WSL sometimes has none and the call is relayed through Windows
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi

export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600
echo "=== ROSCO-native parameter search start $(date)"
echo "    stages $STAGES; proposer budget $BUDGET, repeats $REPEATS, arms $ARMS; baseline budget $TUNE_BUDGET"

# ---------------------------------------------------------------- stage 1: the paper's tuned baseline, re-derived
if [[ " $STAGES " == *" 1 "* ]]; then
  if [ -f "$TUNE/tune_r0/best.json" ]; then
    echo "--- stage 1 skip (already finished)"
  else
    echo "--- stage 1: tuned baseline, $TUNE_BUDGET candidates $(date +%H:%M)"
    $RUN python scripts/rosco_search.py --proposer tune --repeat 0 --budget "$TUNE_BUDGET" \
        "${PROT[@]}" --root "$TUNE" --config_run "$EXP/jg3t_s0" --port0 7600 2>&1 | grep -vE "^\[eval|^ *$"
    sid
  fi
  if [ -f "$TUNE/tune_r0/best.json" ]; then
    $RUN python scripts/rosco_search.py --proposer tune --repeat 0 --root "$TUNE" \
        --emit_template "$HOME/wtrl/runs/template_5mw_rosco_tuned_native" 2>&1 | tail -2
    echo "--- stage 1: the re-derived baseline on held-out and fresh winds $(date +%H:%M)"
    $RUN python scripts/dev/rosco_search_table.py --root "$TUNE" --verify \
        --csv docs/tables/rosco_tuned_baseline.csv 2>&1 | tail -30
    sid
  fi
fi

# ---------------------------------------------------------------- stage 2: the proposer comparison
if [[ " $STAGES " == *" 2 "* ]]; then
  port=7800
  for rep in $REPEATS; do
    for arm in $ARMS; do
      d="$ROOT/${arm}_r${rep}"
      if [ -f "$d/best.json" ]; then echo "skip $arm r$rep (finished)"; continue; fi
      echo "--- $arm r$rep $(date +%H:%M)"
      $RUN python scripts/rosco_search.py --proposer "$arm" --repeat "$rep" --budget "$BUDGET" \
          "${PROT[@]}" --root "$ROOT" --config_run "$EXP/jg3t_s0" --port0 "$port" 2>&1 | grep -vE "^\[eval|^ *$"
      port=$((port + 40)); [ "$port" -gt 8600 ] && port=7800
      sid
    done
  done
fi

# ---------------------------------------------------------------- stage 3: off the design winds
if [[ " $STAGES " == *" 3 "* ]]; then
  echo
  echo "=== the selected tunings off the design winds $(date +%H:%M)"
  $RUN python scripts/dev/rosco_search_table.py --root "$ROOT" --csv docs/tables/rosco_search.csv --verify 2>&1 | tail -60
fi
echo "=== ROSCO-native parameter search done $(date) ==="
