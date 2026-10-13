#!/bin/bash
# The UNSCHEDULED offset-free LPV-MPC on the fresh TurbSim seeds 7-10 (2026-10-13, roadmap s57).
#
# Why: leg 2 of the main line is what the agentic design search buys on an untuned MPC (roadmap s38: llm and es
# both discover the wind-scheduled tower weight in 24 simulations, random 0/48). Measured in the four published
# reductions it is a feasibility repair, not a level gain - the unscheduled controller loses tower-base fatigue
# at 22 and 24 m/s (-2.46 and -6.27) and the agent-found schedule turns that into +0.24 and +0.08 for -1.6
# points of regulation. That comparison exists only on held-out seeds 3-6, because the unscheduled reference was
# never run on the fresh seeds, while every other rung of the ladder is reported on 7-10. One evaluation fixes it.
#
# Identical to the scheduled row (campaign_gate.sh, ref sched_mpc_s78910) in every argument except qt_sched,
# which is the whole point: same horizon 20, r 0.3, qt 3.0, wc_v 0.35, offset-free with tau 5 s, exact model
# (--base_mm_cp 1), same wind protocol, same paired shipped-GSPI baselines in ~/wtrl600.
#
#   setsid nohup bash scripts/wsl/campaign_mpc_unsched_fresh.sh > ~/wtrl/exp/mpc_unsched_fresh.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_mpc_unsched_fresh
#
# Resumable: the evaluation is skipped if its eval_<tag>.json already exists.
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
MEANS="12 14 16 18 20 22 24"
OUT=$EXP/mpcsearch600/fresh
OFF='{"horizon": 20, "r": 0.3, "qt": 3.0, "wc_v": 0.35, "adapt": "offset", "tau_adapt": 5.0}'
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_mpc_unsched_fresh.sid"
mkdir -p "$OUT"
echo "=== unscheduled offset-free MPC on fresh seeds 7-10, start $(date) ==="

if [ -f "$OUT/eval_unsched_mpc_s78910.json" ]; then
  echo "skip: already measured"
else
  WTRL_TEMPLATE=$HOME/wtrl/runs/template_5mw WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 $RUN python scripts/evaluate.py --run "$EXP/mg3t_s0" --gspi --backend openfast --means $MEANS --seeds 7 8 9 10 --ti B --episode_s 600 --workers 7 --port0 6800 --tag unsched_mpc_s78910 --out "$OUT" --base_mm_cp 1 --base_mpc_json "$OFF" 2>&1 | grep -E "J=|per-wind|Traceback|rror:"
fi
echo "=== done $(date) ==="
