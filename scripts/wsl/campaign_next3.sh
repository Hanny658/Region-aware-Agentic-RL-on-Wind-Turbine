#!/bin/bash
# Follow-up 3 (2026-09-20), after campaign_next2.sh: model-error robustness of the wind-scheduled compensated MPC on the
# range set. The estimator's robustness over +-15 % was measured on the low-turbulence set only (roadmap s19, s23); the
# range set has x0.95 for the unscheduled rows (s28). Held-out seeds 3-6 (28 episodes, 600 s, class B):
#
#   scheduled offset-free MPC (qt 3 -> 6 over 18-22 m/s, selected in s33)   Cp/Ct x0.85, x0.95, x1.15
#   unscheduled offset-free MPC                                             x0.85, x1.15   (separates schedule from estimator)
#   nominal MPC                                                             x0.85, x1.15   (what the estimator protects against)
#
#   setsid nohup bash scripts/wsl/campaign_next3.sh > ~/wtrl/exp/next3.log 2>&1 < /dev/null &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
OUT=$EXP/mpcsearch600/qtsched
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_next3.sid"
MEANS="12 14 16 18 20 22 24"
echo "=== follow-up 3 (model error on the range set) start $(date) ==="
ev() {   # tag port cp json
  local tag=$1 port=$2 cp=$3 js=$4
  [ -f "$OUT/eval_$tag.json" ] && { echo "skip $tag"; return; }
  echo "== $tag  $(date +%H:%M)"
  WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 $RUN python scripts/evaluate.py --run "$EXP/mg3t_s0" --gspi --backend openfast \
      --means $MEANS --seeds 3 4 5 6 --ti B --episode_s 600 --workers 6 --port0 "$port" --tag "$tag" --out "$OUT" \
      --base_mm_cp "$cp" --base_mpc_json "$js" 2>&1 | grep -E "J=|per-wind|Traceback|rror:"
}
OFF='"horizon": 20, "r": 0.3, "qt": 3.0, "wc_v": 0.35, "adapt": "offset", "tau_adapt": 5.0'
SCH="{$OFF, \"qt_sched\": [6, 18, 22]}"
NOM='{"horizon": 20, "r": 0.3, "qt": 3.0, "wc_v": 0.35}'
( ev sched_cp0.95_s3456 6600 0.95 "$SCH"; ev sched_cp0.85_s3456 6600 0.85 "$SCH"; ev sched_cp1.15_s3456 6600 1.15 "$SCH"; ev nominal_cp0.85_s3456 6600 0.85 "$NOM" ) &
( ev offset_cp0.85_s3456 7000 0.85 "{$OFF}"; ev offset_cp1.15_s3456 7000 1.15 "{$OFF}"; ev nominal_cp1.15_s3456 7000 1.15 "$NOM" ) &
wait

echo "--- 600 s range rows (seeds 3-6, vs the original GSPI) of the remaining range-wind residual seeds (s34) $(date +%H:%M)"
export WTRL_TEMPLATE=$HOME/wtrl/runs/template_5mw_rosco_tuned
rng() {  # run port
  local run=$1 port=$2
  [ -f "$EXP/$run/ckpt_best.pt" ] || { echo "missing $run"; return; }
  [ -f "$EXP/$run/eval_range_TIB_s3456.json" ] && { echo "skip $run range"; return; }
  echo "== $run range $(date +%H:%M)"
  WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 $RUN python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt \n      --backend openfast --means $MEANS --seeds 3 4 5 6 --ti B --episode_s 600 --workers 6 --port0 "$port" \n      --tag range_TIB_s3456 2>&1 | grep -E "J=|per-wind|Traceback|rror:"
}
( rng trwCwR3t_s1 6600; rng tgCwR3L10_s0 6600 ) &
( rng trwCwR3t_s0 7000; rng tgCwR3L10_s1 7000 ) &
wait
echo "=== follow-up 3 done $(date) ==="
