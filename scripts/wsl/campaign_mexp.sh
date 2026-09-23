#!/bin/bash
# Wohler-exponent sensitivity of the fatigue conclusion (2026-09-23).
#
# The conventional exponents are 4 (welded steel tower) and 10 (composite blade), and every table in the paper uses
# them. A reviewer will ask whether the conclusion survives the neighbouring values. The baseline .npz files store the
# raw OpenFAST channels, so the GSPI side of every ratio can be recomputed offline at any exponent; the controller side
# cannot, because only metrics are kept. agents/rollout.py now records the tower DEL at m = 3, 4, 5 and the blade DEL at
# m = 8, 10, 12 in one rainflow pass, so this campaign simply re-evaluates the four headline controllers on the fresh
# seeds with the new columns. No training, no new wind fields.
#
# Runs AFTER campaign_gate13.sh: the machine has 16 threads and 7 GB of RAM and the two would contend. The wait loop
# below watches the gate13 log for its final line rather than a process name.
#
#   setsid nohup bash scripts/wsl/campaign_mexp.sh > ~/wtrl/exp/mexp.log 2>&1 < /dev/null &
#   scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} mexp
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
MEANS="12 14 16 18 20 22 24"
TUNED=$HOME/wtrl/runs/template_5mw_rosco_tuned
OUT=$EXP/mpcsearch600/mexp; mkdir -p "$OUT"
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_mexp.sid"

if [ -f "$EXP/gate13.log" ]; then
  echo "waiting for the low-gate campaign to finish $(date +%H:%M)"
  while ! grep -q "low-gate residual done" "$EXP/gate13.log"; do sleep 120; done
fi
echo "=== Wohler-exponent sensitivity start $(date) ==="

ref() {  # tag port json template     (zero-residual reference through the same code path)
  local tag=$1 port=$2 js=$3 tpl=${4:-}
  [ -f "$OUT/eval_$tag.json" ] && { echo "skip $tag"; return; }
  echo "== reference $tag $(date +%H:%M)"
  local run="$EXP/mg3t_s0" extra=(--base_mm_cp 1 --base_mpc_json "$js")
  [ -n "$tpl" ] && { run="$EXP/jg3t_s0"; extra=(); }
  WTRL_TEMPLATE=${tpl:-$HOME/wtrl/runs/template_5mw} WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 \
    $RUN python scripts/evaluate.py --run "$run" --gspi --backend openfast --means $MEANS --seeds 7 8 9 10 --ti B \
      --episode_s 600 --workers 5 --port0 "$port" --tag "$tag" --out "$OUT" "${extra[@]}" 2>&1 | grep -E "J=|Traceback|rror:"
}
rng() {  # run port tag [template]
  local run=$1 port=$2 tag=$3 tpl=${4:-}
  [ -f "$EXP/$run/eval_$tag.json" ] && { echo "skip $run $tag"; return; }
  echo "== $run $tag $(date +%H:%M)"
  WTRL_TEMPLATE=${tpl:-$HOME/wtrl/runs/template_5mw} WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 \
    $RUN python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt --backend openfast --means $MEANS --seeds 7 8 9 10 \
      --ti B --episode_s 600 --workers 5 --port0 "$port" --tag "$tag" 2>&1 | grep -E "J=|Traceback|rror:"
}

( ref rosco_tuned_mexp 6900 '{}' "$TUNED"
  rng trwCwG3t_s2 6900 range_TIB_s78910_mexp "$TUNED" ) &
( ref sched_mpc_mexp 7300 @configs/mpc_sched_s33.json
  rng mrwCwG3t_s1 7300 range_TIB_s78910_mexp ) &
wait
echo "=== Wohler-exponent sensitivity done $(date) ==="
