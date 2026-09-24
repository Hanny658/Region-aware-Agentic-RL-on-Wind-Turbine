#!/bin/bash
# Deployment-realism check on the wind signal the policy reads (2026-09-24).
#
# The gate and the MPC both use the controller's own wind-speed estimate, but the PPO observation carried the
# simulator's TRUE hub wind (envs/base_env.py). On this plant the estimate is not a good stand-in for it: bias
# -0.35 m/s, error s.d. 1.42 m/s, correlation 0.73 over the baseline episodes. So the policies were trained with
# information a real turbine does not have, and a reviewer is right to ask how much of their margin depends on it.
#
# This re-evaluates the FROZEN policies with the estimate substituted into that observation slot (--obs_wind_est).
# No retraining: it is a robustness check, not a fair-retraining comparison, and it answers the narrower question
# "is the oracle load-bearing at run time". Fresh seeds 7-10, same 28 episodes, same paired baselines.
#
#   setsid nohup bash scripts/wsl/campaign_windobs.sh > ~/wtrl/exp/windobs.log 2>&1 < /dev/null &
#   scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_windobs
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
MEANS="12 14 16 18 20 22 24"
TUNED=$HOME/wtrl/runs/template_5mw_rosco_tuned
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_windobs.sid"
echo "=== wind-observation realism check start $(date) ==="

rng() {  # run port [template]
  local run=$1 port=$2 tpl=${3:-}
  [ -f "$EXP/$run/ckpt_best.pt" ] || { echo "missing $run"; return; }
  [ -f "$EXP/$run/eval_range_TIB_s78910_vest.json" ] && { echo "skip $run"; return; }
  echo "== $run $(date +%H:%M)"
  WTRL_TEMPLATE=${tpl:-$HOME/wtrl/runs/template_5mw} WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 \
    $RUN python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt --backend openfast --means $MEANS \
      --seeds 7 8 9 10 --ti B --episode_s 600 --workers 6 --port0 "$port" --obs_wind_est \
      --tag range_TIB_s78910_vest 2>&1 | grep -E "estimate|J=|Traceback|rror:"
}
( for r in trwCwG3t_s0 trwCwG3t_s2 mrwCwG3t_s1; do rng "$r" 6500 "$([ "${r:0:1}" = t ] && echo $TUNED)"; done ) &
( for r in trwCwG3t_s1 mrwCwG3t_s0 mrwCwG3t_s2; do rng "$r" 7100 "$([ "${r:0:1}" = t ] && echo $TUNED)"; done ) &
wait
echo "=== wind-observation realism check done $(date) ==="
