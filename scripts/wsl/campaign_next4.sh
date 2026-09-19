#!/bin/bash
# Follow-up 4 (2026-09-20), after campaign_next3.sh: make the positive result of roadmap s34 solid.
#
# s34: on the above-rated range winds, under per-wind Cw, on the tuned ROSCO, the agent-written reward is positive on
# held-out Cw in 3 of 3 seeds and the fixed reward (tower weight 10) negative in 3 of 3. n = 3 is thin and two
# controls are missing:
#
#   trwCwR3t   agent-written reward                 seeds 3 4      (n = 3 -> 5)
#   tgCwR3L10  fixed reward, tower weight 10        seeds 3 4      (n = 3 -> 5)
#   trrCwR3t   RANDOM reward structure (grammar)    seeds 0 1 2    (is it the LLM, or any structural search with the same
#                                                                   fork verification? cf. the random-vocabulary control of v2 s9)
#   tgCwR3t    fixed reward, J-tuned weights        seeds 0 1 2    (the default the agent starts from)
#
# Same base / winds / budget / objective / selection as stage B of campaign_next1.sh. Then every new run on the 600 s
# range set (seeds 3-6) against the original GSPI.
#
#   setsid nohup bash scripts/wsl/campaign_next4.sh > ~/wtrl/exp/next4.log 2>&1 < /dev/null &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
export WTRL_TEMPLATE=$HOME/wtrl/runs/template_5mw_rosco_tuned
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_next4.sid"
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi
MEANS="12 14 16 18 20 22 24"
echo "=== follow-up 4 (more seeds and two controls for s34) start $(date) ==="
grep -q "trrCwR3t" scripts/wsl/campaign_j_core.sh || { echo "campaign_j_core.sh lacks the follow-up 4 arms (next4_patch.py)"; exit 1; }

echo "--- 1: more seeds of the two s34 arms $(date +%H:%M)"
( export WTRL_HOME=$HOME/wtrl_rt_range WTRL_WIND=$HOME/wtrl/wind600
  HELD_MEANS="$MEANS" HELD_TI=B HELD2=0 KNOBS=configs/knobs_j_v3_tuned.json ARMS="trwCwR3t tgCwR3L10" \
    $RUN bash scripts/wsl/campaign_j_core.sh 3 4 )
echo "--- 2: random reward structure and the J-tuned fixed reward $(date +%H:%M)"
( export WTRL_HOME=$HOME/wtrl_rt_range WTRL_WIND=$HOME/wtrl/wind600
  HELD_MEANS="$MEANS" HELD_TI=B HELD2=0 KNOBS=configs/knobs_j_v3_tuned.json ARMS="trrCwR3t tgCwR3t" \
    $RUN bash scripts/wsl/campaign_j_core.sh 0 1 2 )
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_next4.sid"

echo "--- 3: 600 s range rows (seeds 3-6, vs the original GSPI) $(date +%H:%M)"
rng() {  # run port
  local run=$1 port=$2
  [ -f "$EXP/$run/ckpt_best.pt" ] || { echo "missing $run"; return; }
  [ -f "$EXP/$run/eval_range_TIB_s3456.json" ] && { echo "skip $run range"; return; }
  echo "== $run range $(date +%H:%M)"
  WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 $RUN python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt \
      --backend openfast --means $MEANS --seeds 3 4 5 6 --ti B --episode_s 600 --workers 6 --port0 "$port" \
      --tag range_TIB_s3456 2>&1 | grep -E "J=|per-wind|Traceback|rror:"
}
RUNS=$(cd "$EXP" && ls -d trwCwR3t_s? tgCwR3L10_s? trrCwR3t_s? tgCwR3t_s? 2>/dev/null)
i=0; A=""; B=""
for r in $RUNS; do [ -f "$EXP/$r/eval_range_TIB_s3456.json" ] && continue; if [ $((i % 2)) -eq 0 ]; then A="$A $r"; else B="$B $r"; fi; i=$((i + 1)); done
( for r in $A; do rng "$r" 6600; done ) &
( for r in $B; do rng "$r" 7000; done ) &
wait
echo "=== follow-up 4 done $(date) ==="
