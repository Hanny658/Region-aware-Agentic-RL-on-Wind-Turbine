#!/bin/bash
# The gated learned layer re-trained on the NATIVE tuned baseline (2026-10-12, roadmap s55).
#
# Why: the PI-base rung of the manuscript (s34, s37, s41: +4.4 points, gate 15-17 m/s, three agent seeds) was
# trained and measured on the SUPERSEDED tuned baseline - a hand-scaled gain table no ROSCO tuning file can
# produce (s52). The band verification (s55) fixed the baseline row as the stage-1 native tuning, so that rung
# is a gain over a controller that is no longer the baseline. Re-reading tables cannot fix it: the residual was
# trained against the old base's rollouts. This re-trains it.
#
# Identical to stage 2 of campaign_gate.sh except for the base: same arm arguments (agent-written reward, gate
# 15-17 m/s, objective Cw per wind, 300 episodes of 150 s, three seeds), same wind protocol, same knob file. The
# base is chosen by WTRL_TEMPLATE and the paired rollouts by WTRL_HOME, which is why the arm needs a new name
# (tnrwCwG3t) rather than new arguments - the old runs stay where they are.
#
#   setsid nohup bash scripts/wsl/campaign_gate_native.sh > ~/wtrl/exp/gate_native.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_gate_native
#
# Resumable at every stage: baselines skip existing files, training resumes from its newest resume_*.pt (one
# every 5 min, 5 kept), evaluations skip an existing eval_<tag>.json.
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
MEANS="12 14 16 18 20 22 24"
BASE=$HOME/wtrl/runs/template_5mw_rosco_tuned_native
PAIRED=$HOME/wtrl_rtn_range
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_gate_native.sid"

# the LLM supervisor needs the network; relay through the Windows interpreter when WSL has no route out
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi
[ -d "$BASE" ] || { echo "no native baseline template at $BASE"; exit 1; }
echo "=== gated layer on the native tuned baseline, start $(date) ==="

echo "--- 1: the base's own rollouts, 150 s training protocol, seeds 1-6 $(date +%H:%M)"
mkdir -p "$PAIRED"
[ -e "$PAIRED/runs" ] || ln -s "$HOME/wtrl/runs" "$PAIRED/runs"
[ -e "$PAIRED/rosco_install" ] || ln -s "$HOME/wtrl/rosco_install" "$PAIRED/rosco_install"
( export WTRL_HOME=$PAIRED WTRL_WIND=$HOME/wtrl/wind600 WTRL_TEMPLATE=$BASE
  $RUN python scripts/make_baselines.py --backend openfast --means $MEANS --seeds 1 2 3 4 5 6 --ti B \
       --episode_s 150 --jobs 14 --port0 5600 2>&1 \
       | grep -E "baselines :|written|skipping|Error|Traceback" | tail -n 4
  ls "$PAIRED"/baselines/openfast/*.npz | wc -l
  # the base evaluated against its own rollouts must read exactly 0: it is the same controller, and that is what
  # says the paired tree belongs to THIS base and not to the superseded one
  echo "--- identity check, expect 0.000 $(date +%H:%M)"
  [ -f "$EXP/gate_native_identity/eval_identity_rtn_range.json" ] || \
  $RUN python scripts/evaluate.py --run "$EXP/jg3t_s0" --gspi --backend openfast --means $MEANS --seeds 1 2 \
       --ti B --episode_s 150 --workers 14 --port0 6600 --tag identity_rtn_range \
       --out "$EXP/gate_native_identity" 2>&1 | grep -E "J=|Cw=|Traceback|rror:" )

echo "--- 2: three agent seeds on that base, gate 15-17 m/s $(date +%H:%M)"
( export WTRL_HOME=$PAIRED WTRL_WIND=$HOME/wtrl/wind600 WTRL_TEMPLATE=$BASE
  export HELD_MEANS="$MEANS" HELD_TI=B HELD2=0 KNOBS=configs/knobs_j_v3_tuned.json
  ARMS="tnrwCwG3t" $RUN bash scripts/wsl/campaign_j_core.sh 0 1 2 )
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_gate_native.sid"

echo "--- 3: 600 s rows against the ORIGINAL shipped GSPI, held-out seeds 3-6 and fresh 7-10 $(date +%H:%M)"
rng() {  # run port seeds tag
  local run=$1 port=$2 seeds=$3 tag=$4
  [ -f "$EXP/$run/ckpt_best.pt" ] || { echo "missing $run"; return; }
  [ -f "$EXP/$run/eval_$tag.json" ] && { echo "skip $run $tag"; return; }
  echo "== $run $tag $(date +%H:%M)"
  WTRL_TEMPLATE=$BASE WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 \
    $RUN python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt --backend openfast --means $MEANS \
      --seeds $seeds --ti B --episode_s 600 --workers 7 --port0 "$port" --tag "$tag" 2>&1 \
      | grep -E "gate|J=|per-wind|Traceback|rror:"
}
ref() {  # tag port seeds   (the base itself with a zero residual, on the same episodes)
  local tag=$1 port=$2 seeds=$3
  local out=$EXP/mpcsearch600/fresh; mkdir -p "$out"
  [ -f "$out/eval_$tag.json" ] && { echo "skip $tag"; return; }
  echo "== reference $tag $(date +%H:%M)"
  WTRL_TEMPLATE=$BASE WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 \
    $RUN python scripts/evaluate.py --run "$EXP/jg3t_s0" --gspi --backend openfast --means $MEANS \
      --seeds $seeds --ti B --episode_s 600 --workers 7 --port0 "$port" --tag "$tag" --out "$out" 2>&1 \
      | grep -E "J=|per-wind|Traceback|rror:"
}
( ref rosco_tuned_native_s78910 7400 "7 8 9 10"
  for r in tnrwCwG3t_s0 tnrwCwG3t_s2; do
    rng "$r" 7400 "3 4 5 6" range_TIB_s3456; rng "$r" 7400 "7 8 9 10" range_TIB_s78910
  done ) &
( ref rosco_tuned_native_s3456 7700 "3 4 5 6"
  for r in tnrwCwG3t_s1; do
    rng "$r" 7700 "3 4 5 6" range_TIB_s3456; rng "$r" 7700 "7 8 9 10" range_TIB_s78910
  done ) &
wait
echo "=== gated layer on the native tuned baseline, done $(date) ==="
