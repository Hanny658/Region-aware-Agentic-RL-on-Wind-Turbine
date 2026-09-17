#!/bin/bash
# Control direction, stage 3 (2026-09-17): held-out confirmation of the knee of the J-vs-actuator-duty front.
#
# Roadmap s25: over the 123 MPC settings of the 600 s search, J against actuator duty forms a front on which the
# J-selected operating point (23.4 at 1.70 deg/s, ~7.9x GSPI) sits far up the actuation curve, while settings at
# 0.5-0.9 deg/s (2.4-4x GSPI) keep 19-21.5. Those knee points were scored on the supervisor winds only (one
# deterministic evaluation each). Here: three knee settings on both 600 s held-out sets with the exact model,
# and the middle one also with Cp/Ct x0.95.
#
#   setsid nohup bash scripts/wsl/campaign_control3.sh > ~/wtrl/exp/control3.log 2>&1 < /dev/null &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
REFS=$EXP/mpcsearch600/refs
export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_control3.sid"
echo "=== control stage 3 (duty knee, held-out) start $(date) ==="

ev() {   # tag port seeds cp json
  local tag=$1 port=$2 seeds=$3 cp=$4 js=$5
  [ -f "$REFS/eval_$tag.json" ] && { echo "skip $tag"; return; }
  echo "== $tag  $(date +%H:%M)"
  $RUN python scripts/evaluate.py --run "$EXP/mg3t_s0" --gspi --backend openfast --seeds $seeds --episode_s 600 \
      --workers 6 --port0 "$port" --tag "$tag" --out "$REFS" --base_mm_cp "$cp" --base_mpc_json "$js" \
      2>&1 | grep -E "J=|Traceback|rror:"
}
# knee settings from docs/tables/pareto_duty.csv (supervisor-wind duty / J): K1 0.516 / 18.91, K2 0.628 / 21.01, K3 0.860 / 21.54
K1='{"horizon": 15, "r": 0.527, "qt": 3.4, "wc_v": 0.618, "tau_adapt": 6.48, "adapt": "offset"}'
K2='{"horizon": 15, "r": 0.408, "qt": 3.68, "wc_v": 0.425, "tau_adapt": 6.76, "adapt": "offset"}'
K3='{"horizon": 15, "r": 0.294, "qt": 4.27, "wc_v": 0.41, "tau_adapt": 5.52, "adapt": "offset"}'
( ev knee1_cp1_s3456 6600 "3 4 5 6" 1 "$K1"; ev knee2_cp1_s3456 6600 "3 4 5 6" 1 "$K2"; ev knee3_cp1_s3456 6600 "3 4 5 6" 1 "$K3"; ev knee2_cp0.95_s3456 6600 "3 4 5 6" 0.95 "$K2" ) &
( ev knee1_cp1_s78910 7000 "7 8 9 10" 1 "$K1"; ev knee2_cp1_s78910 7000 "7 8 9 10" 1 "$K2"; ev knee3_cp1_s78910 7000 "7 8 9 10" 1 "$K3"; ev knee2_cp0.95_s78910 7000 "7 8 9 10" 0.95 "$K2" ) &
wait
echo "=== control stage 3 done $(date) ==="
