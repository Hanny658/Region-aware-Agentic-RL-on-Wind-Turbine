#!/bin/bash
# Must-do 4 and 3 (2026-09-17), run after campaign_must1_range.sh.
#
#   A  Tuned ROSCO baseline: the GSPI's own pitch loop (gain-schedule scales, speed-filter corner, tower-damper gain)
#      searched under J on the 600 s supervisor winds with the budget every other controller got (40 candidates),
#      then both 600 s held-out sets. J is measured against the ORIGINAL GSPI baselines, so the tuned ROSCO's J is
#      the part of every reported margin that a tuned baseline would take away.
#   B  Domain-randomised residual: the residual on the nominal MPC trained with the MPC's Cp/Ct scale drawn from
#      [0.85, 1.15] every episode (fixed and agent-written reward, 3 seeds each; 150 s bank like every residual arm),
#      then evaluated at fixed model errors 0.85 / 0.9 / 0.95 / 1.05 / 1.15 (wind seeds 3-6) for the transfer table
#      of roadmap s23. The campaign_j_core stage evaluates the exact model on both held-out sets.
#
#   setsid nohup bash scripts/wsl/campaign_must2.sh > ~/wtrl/exp/must2.log 2>&1 < /dev/null &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_must2.sid"
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi
echo "=== must-do 4 + 3 start $(date) ==="

echo "--- A: tuned ROSCO baseline (600 s) $(date +%H:%M)"
( export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600
  $RUN python scripts/rosco_tune.py --budget 40 --episode_s 600 --parallel 2 --workers 6 --port0 7600 --heldout 2>&1 | grep -E "^\[rosco\]|Traceback|rror:" )

echo "--- B1: domain-randomised residual, training $(date +%H:%M)"
KNOBS=configs/knobs_j_v3_tuned.json ARMS="mgR3t mrwR3t" $RUN bash scripts/wsl/campaign_j_core.sh 0 1 2
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_must2.sid"

echo "--- B2: transfer of the domain-randomised residuals across model error $(date +%H:%M)"
transfer() {   # run port
  local run=$1 port=$2
  [ -f "$EXP/$run/ckpt_best.pt" ] || { echo "missing $run"; return; }
  for cp in 0.85 0.9 0.95 1.05 1.15; do
    [ -f "$EXP/$run/eval_gen_cp${cp}_s3456.json" ] && { echo "skip $run cp$cp"; continue; }
    echo "== $run cp $cp $(date +%H:%M)"
    $RUN python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt --backend openfast --seeds 3 4 5 6 \
        --workers 6 --port0 "$port" --tag "gen_cp${cp}_s3456" --base_mm_cp "$cp" 2>&1 | grep -E "J=|Traceback|rror:"
  done
}
( for r in mgR3t_s0 mgR3t_s1 mgR3t_s2; do transfer "$r" 5900; done ) &
( for r in mrwR3t_s0 mrwR3t_s1 mrwR3t_s2; do transfer "$r" 6300; done ) &
wait
echo "=== must-do 4 + 3 done $(date) ==="
