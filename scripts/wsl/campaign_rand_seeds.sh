#!/bin/bash
# More control seeds for the one statistical question about the agent that is still open (2026-09-23).
#
# On the tuned PI base, without the gate, the agent-written reward holds both fatigue loads within 1 % at every wind
# speed in 4 of 5 seeds, the RANDOM reward structure with the same fork verification in 0 of 3, and the fixed rewards
# in 0 of 8. Reported separately, as they must be: against the fixed rewards Fisher gives two-sided p = 0.007, against
# the random structure at n = 3 it gives p = 0.14. So the claim "a searched reward structure holds the loads where a
# fixed one does not" is established and the claim "the language model proposes better structures than a random draw
# from its own vocabulary" is not. Five more random-structure seeds take that control to n = 8, which either settles it
# (4/5 vs 0/8 is p = 0.007) or kills it cleanly (any 2 of 8 holding takes p above 0.1 and the claim goes).
#
#   trrCwR3t   random reward structure, tuned ROSCO base, ungated   seeds 3 4 5 6 7
#
# Everything else is exactly stage 2 of campaign_next4.sh: same base, winds, budget, objective and selection, so the
# new seeds pool with the existing three without qualification.
#
# Runs AFTER campaign_mexp.sh, which runs after campaign_gate13.sh; the machine has 16 threads and 7 GB of RAM.
#
#   setsid nohup bash scripts/wsl/campaign_rand_seeds.sh > ~/wtrl/exp/rand_seeds.log 2>&1 < /dev/null &
#   scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_rand_seeds
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
MEANS="12 14 16 18 20 22 24"
export WTRL_TEMPLATE=$HOME/wtrl/runs/template_5mw_rosco_tuned
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_rand_seeds.sid"
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi

for w in mexp gate13; do
  [ -f "$EXP/$w.log" ] || continue
  echo "waiting for campaign_$w $(date +%H:%M)"
  while ! grep -qE "(Wohler-exponent sensitivity|low-gate residual) done" "$EXP/$w.log"; do sleep 120; done
done
echo "=== random-structure control seeds start $(date) ==="

echo "--- 1: training seeds 3-7 $(date +%H:%M)"
( export WTRL_HOME=$HOME/wtrl_rt_range WTRL_WIND=$HOME/wtrl/wind600
  export HELD_MEANS="$MEANS" HELD_TI=B HELD2=0 KNOBS=configs/knobs_j_v3_tuned.json
  ARMS="trrCwR3t" $RUN bash scripts/wsl/campaign_j_core.sh 3 4 5 6 7 )
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_rand_seeds.sid"

echo "--- 2: 600 s range rows on the held-out seeds 3-6, vs the original GSPI $(date +%H:%M)"
rng() {  # run port
  local run=$1 port=$2
  [ -f "$EXP/$run/ckpt_best.pt" ] || { echo "missing $run"; return; }
  [ -f "$EXP/$run/eval_range_TIB_s3456.json" ] && { echo "skip $run"; return; }
  echo "== $run $(date +%H:%M)"
  WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 $RUN python scripts/evaluate.py --run "$EXP/$run" \
      --ckpt ckpt_best.pt --backend openfast --means $MEANS --seeds 3 4 5 6 --ti B --episode_s 600 --workers 6 \
      --port0 "$port" --tag range_TIB_s3456 2>&1 | grep -E "J=|per-wind|Traceback|rror:"
}
( for r in trrCwR3t_s3 trrCwR3t_s5 trrCwR3t_s7; do rng "$r" 6400; done ) &
( for r in trrCwR3t_s4 trrCwR3t_s6; do rng "$r" 7400; done ) &
wait
echo "=== random-structure control seeds done $(date) ==="
