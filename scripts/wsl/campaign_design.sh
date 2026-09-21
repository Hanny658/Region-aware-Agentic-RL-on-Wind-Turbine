#!/bin/bash
# Proposer comparison for the supervised MPC design loop on the range set (2026-09-21, user decision).
#
# Is the main-claim controller of roadmap s33 something the verified design loop produces by itself, and does the
# proposer matter? scripts/mpc_design_search.py: space = MPC parameters + tower-weight schedule, selection objective
# S = J - 4 * V (per-wind load rule as a soft constraint) on range seeds 1-2 at 12 / 16 / 20 / 24 m/s, 600 s; every arm
# starts from the hand-set UNSCHEDULED reference; budget 24 verified candidates; arms llm / es / random, two repeats,
# interleaved so that one complete set of three exists first. Then the best candidate of every arm-run on the held-out
# seeds 3-6 at all seven wind speeds (28 episodes, never seen by a proposer), next to the hand-designed controller of s33.
#
#   setsid nohup bash scripts/wsl/campaign_design.sh > ~/wtrl/exp/design.log 2>&1 < /dev/null &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
ROOT=$EXP/mpcdesign
export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_design.sid"
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi
BUDGET=${BUDGET:-24}
REPS=${REPS:-0 1}
echo "=== design search (proposer comparison on the range set) start $(date) ==="
for rep in $REPS; do
  for arm in llm es random; do
    [ -f "$ROOT/${arm}_r$rep/best.json" ] && { echo "skip ${arm}_r$rep (done)"; continue; }
    echo "--- ${arm}_r$rep $(date +%H:%M)"
    $RUN python scripts/mpc_design_search.py --arm "$arm" --rep "$rep" --budget "$BUDGET" --root "$ROOT" 2>&1 \
      | grep -E "^\[|Traceback|rror"
  done
done

echo "--- held-out: the best candidate of every arm-run on seeds 3-6, seven wind speeds $(date +%H:%M)"
mkdir -p "$ROOT/heldout"
held() {  # name port
  local name=$1 port=$2
  [ -f "$ROOT/$name/best.json" ] || { echo "missing $name"; return; }
  [ -f "$ROOT/heldout/eval_${name}_s3456.json" ] && { echo "skip $name held-out"; return; }
  local js
  js=$($RUN python scripts/dev/design_best_kw.py "$ROOT/$name/best.json")
  echo "== $name held-out $(date +%H:%M)  $js"
  $RUN python scripts/evaluate.py --run "$EXP/mg3t_s0" --gspi --backend openfast --means 12 14 16 18 20 22 24 --seeds 3 4 5 6 --ti B \
      --episode_s 600 --workers 6 --port0 "$port" --tag "${name}_s3456" --out "$ROOT/heldout" --base_mm_cp 1 --base_mpc_json "$js" \
      2>&1 | grep -E "J=|per-wind|Traceback|rror:"
}
A=""; B=""; i=0
for rep in $REPS; do for arm in llm es random; do
  if [ $((i % 2)) -eq 0 ]; then A="$A ${arm}_r$rep"; else B="$B ${arm}_r$rep"; fi; i=$((i + 1))
done; done
( for n in $A; do held "$n" 6600; done ) &
( for n in $B; do held "$n" 7000; done ) &
wait
echo "=== design search done $(date) ==="
