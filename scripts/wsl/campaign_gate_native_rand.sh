#!/bin/bash
# The matched RANDOM-STRUCTURE control for the gated layer on the native tuned baseline (2026-10-13, roadmap s58).
#
# Why this is not bookkeeping: s49 settled "who writes the reward" at n = 8 on the UNGATED tuned-PI base - under the
# per-wind load rule the agent-written reward holds 4/5, a random structure from the same grammar with the same fork
# verification 1/8, a fixed reward 0/8 (Fisher two-sided p = 0.032 against the random structures). But the
# configuration the manuscript headlines is the WIND-GATED one, and it has never been run against this control on
# either base; s40 and s41 both record that as a limitation, and s56 re-trained only the agent arm on the baseline
# that is a real tuning file. So this is the one open experiment that can MOVE the claim of docs/MAIN_LINE.md rather
# than confirm it: if random structures also hold the loads under the gate, leg 3 must be scoped to the ungated base.
#
# Arm tnrrCwG3t is tnrwCwG3t with --supervisor random_reward and nothing else changed: same base
# (WTRL_TEMPLATE=template_5mw_rosco_tuned_native), same paired rollouts (WTRL_HOME=~/wtrl_rtn_range, the 42 files
# the s56 campaign generated and verified with an exactly 0.000 identity check), same winds, same knob file, same
# objective Cw, same 300 episodes of 150 s, and the same three training seeds 0 1 2 - so the three control runs pair
# with the three agent runs seed by seed.
#
# Statistical power, stated before the result (3 against 3): the best possible Fisher outcome, 3/3 against 0/3, is
# two-sided p = 0.1, so this cannot ESTABLISH the agent. It can refute the extension: if the random structures hold
# the loads, n = 3 is enough to say the proposer is not what buys feasibility under the gate. The informative branch
# is the cheap one, which is why 3 seeds run before any decision about 5.
#
#   setsid nohup bash scripts/wsl/campaign_gate_native_rand.sh > ~/wtrl/exp/gate_native_rand.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_gate_native_rand
#
# Resumable at every stage: training resumes from its newest resume_*.pt (one every 5 min, 5 kept), evaluations skip
# an existing eval_<tag>.json, and the 600 s reference rows of the base were already measured by s56.
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
MEANS="12 14 16 18 20 22 24"
BASE=$HOME/wtrl/runs/template_5mw_rosco_tuned_native
PAIRED=$HOME/wtrl_rtn_range
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_gate_native_rand.sid"

# no network is needed: the random-structure supervisor draws from a fixed grammar and never calls the model
[ -d "$BASE" ] || { echo "no native baseline template at $BASE"; exit 1; }
n=$(ls "$PAIRED"/baselines/openfast/*.npz 2>/dev/null | wc -l)
[ "$n" = "42" ] || { echo "the paired tree of the native base has $n of 42 rollouts: run campaign_gate_native.sh stage 1 first"; exit 1; }
echo "=== random-structure control on the native tuned baseline, start $(date) ==="

echo "--- 1: three control seeds on that base, gate 15-17 m/s $(date +%H:%M)"
( export WTRL_HOME=$PAIRED WTRL_WIND=$HOME/wtrl/wind600 WTRL_TEMPLATE=$BASE
  export HELD_MEANS="$MEANS" HELD_TI=B HELD2=0 KNOBS=configs/knobs_j_v3_tuned.json
  ARMS="tnrrCwG3t" $RUN bash scripts/wsl/campaign_j_core.sh 0 1 2 )
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_gate_native_rand.sid"

echo "--- 2: 600 s rows against the ORIGINAL shipped GSPI, held-out seeds 3-6 and fresh 7-10 $(date +%H:%M)"
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
( for r in tnrrCwG3t_s0 tnrrCwG3t_s2; do
    rng "$r" 7400 "3 4 5 6" range_TIB_s3456; rng "$r" 7400 "7 8 9 10" range_TIB_s78910
  done ) &
( for r in tnrrCwG3t_s1; do
    rng "$r" 7700 "3 4 5 6" range_TIB_s3456; rng "$r" 7700 "7 8 9 10" range_TIB_s78910
  done ) &
wait
echo "=== random-structure control on the native tuned baseline, done $(date) ==="
