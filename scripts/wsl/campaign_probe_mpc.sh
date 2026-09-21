#!/bin/bash
# Probe (2026-09-21, user request): the agent-supervised residual STACKED ON the main-claim MPC of roadmap s33.
#
# Every earlier residual-on-MPC arm (s17-s24, s30) was on the low-turbulence 150 s bank, under J, before the defects of
# s32 were fixed. s34-s35 say the layer pays where the base leaves room; the scheduled MPC leaves room near rated
# (power MSE +2 to +40 % at 12-16 m/s) and in the fatigue terms at high wind. This probe repeats the s34 setup with the
# MPC as the base: above-rated range winds at 150 s (TurbSim seed 1 training, seeds 1-2 supervisor), per-wind objective
# Cw, and the paired baselines are THE MPC'S OWN zero-residual rollouts (separate home ~/wtrl_mpc_range), so that a
# positive Cw reads "regulation gained on top of the MPC with both loads held at every wind speed".
#
#   arms   mrwCwR3t  agent-written reward   seeds 0 1 2
#          mgCwR3t   fixed reward (J-tuned)  seeds 0 1
#   then   held-out Cw at 150 s (seeds 3-6) vs the MPC; every run on the 600 s range set (seeds 3-6) vs the original GSPI,
#          next to the MPC alone (28.95 on the same 28 episodes)
#
#   setsid nohup bash scripts/wsl/campaign_probe_mpc.sh > ~/wtrl/exp/probe_mpc.log 2>&1 < /dev/null &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
HOME_MPC=$HOME/wtrl_mpc_range
MEANS="12 14 16 18 20 22 24"
SCHED=@configs/mpc_sched_s33.json
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_probe_mpc.sid"
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi
echo "=== probe: agent residual on the scheduled MPC start $(date) ==="

echo "--- 0: paired baselines of the scheduled MPC (range winds, 150 s, seeds 1-6) $(date +%H:%M)"
mkdir -p "$HOME_MPC"
[ -e "$HOME_MPC/runs" ] || ln -s "$HOME/wtrl/runs" "$HOME_MPC/runs"
[ -e "$HOME_MPC/rosco_install" ] || ln -s "$HOME/wtrl/rosco_install" "$HOME_MPC/rosco_install"
( export WTRL_HOME=$HOME_MPC WTRL_WIND=$HOME/wtrl/wind600
  $RUN python scripts/make_baselines.py --backend openfast --means $MEANS --seeds 1 2 3 4 5 6 --ti B --episode_s 150 \
       --base mpc --base_mpc_json "$SCHED" --jobs 12 --port0 5700 2>&1 | grep -E "base      :|baselines :|written|skipping|Error|Traceback" | tail -n 4
  ls "$HOME_MPC/baselines/openfast"/*.npz | wc -l
  echo "--- identity check (zero residual on the MPC vs its own baselines; expect 0) $(date +%H:%M)"
  [ -f "$EXP/agent_rt_identity/eval_identity_mpc_range.json" ] || \
  $RUN python scripts/evaluate.py --run "$EXP/mg3t_s0" --gspi --backend openfast --means $MEANS --seeds 1 2 --ti B --episode_s 150 \
       --workers 8 --port0 6600 --tag identity_mpc_range --out "$EXP/agent_rt_identity" --base_mm_cp 1 --base_mpc_json "$SCHED" \
       2>&1 | grep -E "J=|per-wind|Traceback|rror:"

  echo "--- 1: training $(date +%H:%M)"
  HELD_MEANS="$MEANS" HELD_TI=B HELD2=0 KNOBS=configs/knobs_j_v3_tuned.json ARMS="mrwCwR3t" $RUN bash scripts/wsl/campaign_j_core.sh 0 1 2
  HELD_MEANS="$MEANS" HELD_TI=B HELD2=0 KNOBS=configs/knobs_j_v3_tuned.json ARMS="mgCwR3t" $RUN bash scripts/wsl/campaign_j_core.sh 0 1 )
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_probe_mpc.sid"

echo "--- 2: 600 s range rows (seeds 3-6, vs the original GSPI) $(date +%H:%M)"
rng() {  # run port
  local run=$1 port=$2
  [ -f "$EXP/$run/ckpt_best.pt" ] || { echo "missing $run"; return; }
  [ -f "$EXP/$run/eval_range_TIB_s3456.json" ] && { echo "skip $run range"; return; }
  echo "== $run range $(date +%H:%M)"
  WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 $RUN python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt \
      --backend openfast --means $MEANS --seeds 3 4 5 6 --ti B --episode_s 600 --workers 6 --port0 "$port" \
      --tag range_TIB_s3456 2>&1 | grep -E "J=|per-wind|Traceback|rror:"
}
( rng mrwCwR3t_s0 6600; rng mrwCwR3t_s2 6600; rng mgCwR3t_s1 6600 ) &
( rng mrwCwR3t_s1 7000; rng mgCwR3t_s0 7000 ) &
wait
echo "=== probe done $(date) ==="
