#!/bin/bash
# Wind-gated residual (2026-09-22): the design change that came out of the MPC-stacking probe.
#
# Probe finding: on the scheduled MPC the agent-reward residual adds +12...+16 % power / speed MSE reduction at 18-24 m/s
# with both loads held, and costs 5 % tower fatigue at 12-14 m/s (visible on the supervisor winds as well: every late
# evaluation of both seeds is damaged at 12-14 m/s only). With the residual gated off below 15 m/s (1 above 17 m/s, on a
# 10 s low-pass of the wind estimate) the SAME final checkpoint of one seed reads held-out Cw +8.65 with zero per-wind
# violation on top of the MPC - but that checkpoint and the gate were picked after looking at held-out seeds 3-6.
# This campaign is the clean version: train WITH the gate from scratch, select on the supervisor winds (seeds 1-2),
# report on seeds 3-6 AND on fresh TurbSim seeds 7-10 that nothing has seen (added to the separate 600 s bank
# ~/wtrl/wind600 only; the canonical bank is untouched).
#
#   mrwCwG3t  agent-written reward, scheduled MPC base, gate 15-17 m/s    seeds 0 1 2
#   mgCwG3t   fixed reward (J-tuned), same                                seeds 0 1
#   trwCwG3t  agent-written reward, TUNED ROSCO base, gate 15-17 m/s      seeds 0 1 2   (s34 lost power MSE at 12 m/s)
#
#   setsid nohup bash scripts/wsl/campaign_gate.sh > ~/wtrl/exp/gate.log 2>&1 < /dev/null &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
MEANS="12 14 16 18 20 22 24"
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_gate.sid"
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi
echo "=== wind-gated residual start $(date) ==="
grep -q "mrwCwG3t" scripts/wsl/campaign_j_core.sh || { echo "campaign_j_core.sh lacks the gated arms (gate_arms_patch.py)"; exit 1; }

echo "--- 1: gated residual on the scheduled MPC $(date +%H:%M)"
( export WTRL_HOME=$HOME/wtrl_mpc_range WTRL_WIND=$HOME/wtrl/wind600
  export HELD_MEANS="$MEANS" HELD_TI=B HELD2=0 KNOBS=configs/knobs_j_v3_tuned.json
  ARMS="mrwCwG3t" $RUN bash scripts/wsl/campaign_j_core.sh 0 1 2
  ARMS="mgCwG3t"  $RUN bash scripts/wsl/campaign_j_core.sh 0 1 )
echo "--- 2: gated residual on the tuned ROSCO $(date +%H:%M)"
( export WTRL_HOME=$HOME/wtrl_rt_range WTRL_WIND=$HOME/wtrl/wind600 WTRL_TEMPLATE=$HOME/wtrl/runs/template_5mw_rosco_tuned
  export HELD_MEANS="$MEANS" HELD_TI=B HELD2=0 KNOBS=configs/knobs_j_v3_tuned.json
  ARMS="trwCwG3t" $RUN bash scripts/wsl/campaign_j_core.sh 0 1 2 )
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_gate.sid"

echo "--- 3: fresh wind seeds 7-10 for the range bank and their GSPI baselines (600 s) $(date +%H:%M)"
( export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600
  $RUN python scripts/gen_wind.py --means $MEANS --seeds 7 8 9 10 --ti B --time 650 --out "$WTRL_WIND" --jobs 7 2>&1 | tail -n 2
  ls "$WTRL_WIND"/U*_TIB_S*.bts | wc -l
  $RUN python scripts/make_baselines.py --backend openfast --means $MEANS --seeds 7 8 9 10 --ti B --episode_s 600 --jobs 12 --port0 5700 \
       2>&1 | grep -E "baselines :|written|skipping|Error|Traceback" | tail -n 3 )

echo "--- 4: 600 s range rows vs the original GSPI: seeds 3-6 and the fresh seeds 7-10 $(date +%H:%M)"
rng() {  # run port seeds tag [template]
  local run=$1 port=$2 seeds=$3 tag=$4 tpl=${5:-}
  [ -f "$EXP/$run/ckpt_best.pt" ] || { echo "missing $run"; return; }
  [ -f "$EXP/$run/eval_$tag.json" ] && { echo "skip $run $tag"; return; }
  echo "== $run $tag $(date +%H:%M)"
  WTRL_TEMPLATE=${tpl:-$HOME/wtrl/runs/template_5mw} WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 \
    $RUN python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt --backend openfast --means $MEANS --seeds $seeds --ti B \
      --episode_s 600 --workers 6 --port0 "$port" --tag "$tag" 2>&1 | grep -E "gate|J=|per-wind|Traceback|rror:"
}
ref() {  # tag port seeds json template   (zero-residual references on the fresh seeds)
  local tag=$1 port=$2 seeds=$3 js=$4 tpl=${5:-}
  local out=$EXP/mpcsearch600/fresh; mkdir -p "$out"
  [ -f "$out/eval_$tag.json" ] && { echo "skip $tag"; return; }
  echo "== reference $tag $(date +%H:%M)"
  local run="$EXP/mg3t_s0" extra=(--base_mm_cp 1 --base_mpc_json "$js")
  if [ -n "$tpl" ]; then run="$EXP/jg3t_s0"; extra=(); fi
  WTRL_TEMPLATE=${tpl:-$HOME/wtrl/runs/template_5mw} WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 \
    $RUN python scripts/evaluate.py --run "$run" --gspi --backend openfast --means $MEANS --seeds $seeds --ti B --episode_s 600 \
      --workers 6 --port0 "$port" --tag "$tag" --out "$out" "${extra[@]}" 2>&1 | grep -E "J=|per-wind|Traceback|rror:"
}
TUNED=$HOME/wtrl/runs/template_5mw_rosco_tuned
( ref sched_mpc_s78910 6600 "7 8 9 10" @configs/mpc_sched_s33.json
  for r in mrwCwG3t_s0 mrwCwG3t_s2 mgCwG3t_s1 trwCwG3t_s1; do rng "$r" 6600 "3 4 5 6" range_TIB_s3456 "$([ "${r:0:1}" = t ] && echo $TUNED)"; rng "$r" 6600 "7 8 9 10" range_TIB_s78910 "$([ "${r:0:1}" = t ] && echo $TUNED)"; done ) &
( ref rosco_tuned_s78910 7000 "7 8 9 10" '{}' "$TUNED"
  for r in mrwCwG3t_s1 mgCwG3t_s0 trwCwG3t_s0 trwCwG3t_s2; do rng "$r" 7000 "3 4 5 6" range_TIB_s3456 "$([ "${r:0:1}" = t ] && echo $TUNED)"; rng "$r" 7000 "7 8 9 10" range_TIB_s78910 "$([ "${r:0:1}" = t ] && echo $TUNED)"; done ) &
wait
echo "=== wind-gated residual done $(date) ==="
