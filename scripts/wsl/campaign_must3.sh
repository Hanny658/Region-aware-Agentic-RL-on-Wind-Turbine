#!/bin/bash
# Follow-up to the must-do experiments (2026-09-18), run after campaign_must2.sh. Both stages use the above-rated
# range set of campaign_must1_range.sh (12-24 m/s, IEC class B, 600 s, TurbSim seeds 1-6, 42 episodes).
#
#   A  The tuned ROSCO of campaign_must2.sh (selected on the 8 / 12.5 / 15 m/s supervisor winds) on the range set,
#      so the tuned baseline has a row in the main table.
#   B  Diagnosis of roadmap s28 finding 3 (the tower term of every compensated MPC is negative from 20 m/s up):
#      the offset-free reference with its tower weight qt raised x2 / x4 / x8, and the duty knee with x3. This is a
#      sweep for the mechanism on winds that are held-out for every reported row; it selects nothing. If a larger
#      tower weight repairs 20-24 m/s at a cost at 12-16 m/s, a wind-scheduled weight is the remedy and has to be
#      selected on TurbSim seeds 1-2 and reported on seeds 3-6.
#
#   setsid nohup bash scripts/wsl/campaign_must3.sh > ~/wtrl/exp/must3.log 2>&1 < /dev/null &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
OUT=$EXP/mpcsearch600/range
export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_must3.sid"
MEANS="12 14 16 18 20 22 24"
SEEDS="1 2 3 4 5 6"
echo "=== must-do follow-up (range: tuned ROSCO, tower-weight sweep) start $(date) ==="

ev() {   # tag port cp json [template]
  local tag=$1 port=$2 cp=$3 js=$4 tpl=${5:-}
  [ -f "$OUT/eval_$tag.json" ] && { echo "skip $tag"; return; }
  echo "== $tag  $(date +%H:%M)"
  local run="$EXP/mg3t_s0" extra=(--base_mm_cp "$cp" --base_mpc_json "$js")
  if [ -n "$tpl" ]; then run="$EXP/jg3t_s0"; extra=(); fi
  WTRL_TEMPLATE=${tpl:-$HOME/wtrl/runs/template_5mw} $RUN python scripts/evaluate.py --run "$run" --gspi --backend openfast \
      --means $MEANS --seeds $SEEDS --ti B --episode_s 600 --workers 6 --port0 "$port" --tag "$tag" --out "$OUT" "${extra[@]}" \
      2>&1 | grep --line-buffered -E "J=|Traceback|rror:"
}

TPL=$HOME/wtrl/runs/template_5mw_rosco_tuned
if [ -f "$EXP/roscotune600/best.json" ]; then
  $RUN python scripts/rosco_tune.py --emit_template "$TPL" 2>&1 | grep -E "^\[rosco\]|Traceback|rror:"
else
  echo "no tuned ROSCO (roscotune600/best.json missing): stage A skipped"; TPL=""
fi

Q6='{"horizon": 20, "r": 0.3, "qt": 6.0, "wc_v": 0.35, "adapt": "offset", "tau_adapt": 5.0}'
Q12='{"horizon": 20, "r": 0.3, "qt": 12.0, "wc_v": 0.35, "adapt": "offset", "tau_adapt": 5.0}'
Q24='{"horizon": 20, "r": 0.3, "qt": 24.0, "wc_v": 0.35, "adapt": "offset", "tau_adapt": 5.0}'
K2Q='{"horizon": 15, "r": 0.408, "qt": 11.0, "wc_v": 0.425, "tau_adapt": 6.76, "adapt": "offset"}'
( [ -n "$TPL" ] && ev rosco_tuned 6600 1 '{}' "$TPL"; ev offset_qt12_cp1 6600 1 "$Q12"; ev knee2_qt11_cp1 6600 1 "$K2Q" ) &
( ev offset_qt6_cp1 7000 1 "$Q6"; ev offset_qt24_cp1 7000 1 "$Q24" ) &
wait
echo "=== must-do follow-up done $(date) ==="
