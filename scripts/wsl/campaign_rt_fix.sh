#!/bin/bash
# Fix-up for campaign_agent_rt.sh (2026-09-19 07:30): the constrained objectives were evaluated with the oracle-rule
# above-rated subset (evaluate.py enabled the wind-labelled subset only for objective J). Re-evaluate, with the wind
# label on both sides, the runs whose selected checkpoint is not episode 0 (the others are the tuned ROSCO alone, 0 by
# construction) plus one episode-0 run as the consistency check against the tuned-ROSCO absolute row (J 14.20 on wind
# seeds 3-6 at 150 s). Rows: held-out vs the tuned ROSCO (both sets), absolute vs the original GSPI, the 600 s range set.
#
#   setsid nohup bash scripts/wsl/campaign_rt_fix.sh > ~/wtrl/exp/rt_fix.log 2>&1 < /dev/null &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
export WTRL_TEMPLATE=$HOME/wtrl/runs/template_5mw_rosco_tuned
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_rt_fix.sid"
RUNS="trwCw3t_s1 trwTw3t_s1 trwC3t_s0 trwC3t_s1 trwC3t_s2 tgC3t_s0"
echo "=== rt fix-up (wind-labelled subset for the constrained objectives) start $(date) ==="
for r in $RUNS; do
  for f in eval_heldout_s3456_ckpt_best eval_heldout2_s78910 eval_abs_gspi_s3456 eval_range_TIB; do
    [ -f "$EXP/$r/$f.json" ] && mv "$EXP/$r/$f.json" "$EXP/$r/${f}_oracle.json"
    [ -f "$EXP/$r/$f.csv" ] && mv "$EXP/$r/$f.csv" "$EXP/$r/${f}_oracle.csv"
  done
done
one() {  # run port
  local r=$1 port=$2
  for spec in "heldout_s3456_ckpt_best:3 4 5 6" "heldout2_s78910:7 8 9 10"; do
    tag=${spec%%:*}; seeds=${spec#*:}
    [ -f "$EXP/$r/eval_$tag.json" ] && continue
    echo "== $r $tag $(date +%H:%M)"
    WTRL_HOME=$HOME/wtrl_rt WTRL_WIND=$HOME/wtrl/wind $RUN python scripts/evaluate.py --run "$EXP/$r" --ckpt ckpt_best.pt --backend openfast \
        --means 8 12.5 15 --seeds $seeds --workers 6 --port0 "$port" --tag "$tag" 2>&1 | grep -E "J=|C=|per-wind|Traceback|rror:"
  done
  [ -f "$EXP/$r/eval_abs_gspi_s3456.json" ] || {
    echo "== $r abs $(date +%H:%M)"
    WTRL_HOME=$HOME/wtrl WTRL_WIND=$HOME/wtrl/wind $RUN python scripts/evaluate.py --run "$EXP/$r" --ckpt ckpt_best.pt --backend openfast \
        --seeds 3 4 5 6 --workers 6 --port0 "$port" --tag abs_gspi_s3456 2>&1 | grep -E "J=|C=|Traceback|rror:"; }
}
( for r in trwCw3t_s1 trwC3t_s0 trwC3t_s2; do one "$r" 7600; done ) &
( for r in trwTw3t_s1 trwC3t_s1 tgC3t_s0; do one "$r" 7900; done ) &
wait
echo "--- range set (600 s, seeds 1-6, vs the original GSPI) for the set-mean agent runs $(date +%H:%M)"
rng() {
  local r=$1 port=$2
  [ -f "$EXP/$r/eval_range_TIB.json" ] && { echo "skip $r"; return; }
  echo "== $r range $(date +%H:%M)"
  WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 $RUN python scripts/evaluate.py --run "$EXP/$r" --ckpt ckpt_best.pt --backend openfast \
      --means 12 14 16 18 20 22 24 --seeds 1 2 3 4 5 6 --ti B --episode_s 600 --workers 6 --port0 "$port" --tag range_TIB 2>&1 | grep -E "J=|per-wind|Traceback|rror:"
}
rng trwC3t_s2 7600 &
rng trwC3t_s0 7900 &
wait
echo "=== rt fix-up done $(date) ==="
