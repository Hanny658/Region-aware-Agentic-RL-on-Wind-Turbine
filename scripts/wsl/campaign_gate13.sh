#!/bin/bash
# Low gate: the residual acts from 13 m/s instead of 15 m/s (2026-09-23).
#
# Motivation (roadmap s42). Re-weighting the fresh-seed results over an IEC Rayleigh wind distribution leaves the gated
# layer with ~0 weighted tower-DEL contribution on the tuned ROSCO base: the gate opens at 15-17 m/s, which is the tail
# of the distribution, while the near-rated bins carry most of the probability AND most of the tower damage. The
# ungated layer did reach those bins and damaged them (s39: -5 % tower at 12-14 m/s). This campaign tests the middle
# point: gate 13-15 m/s, everything else exactly as in campaign_gate.sh (same knobs, same objective Cw with its
# per-wind rule, selection on the supervisor winds only).
#
#   mrwCwGLt  agent-written reward, scheduled MPC base, gate 13-15 m/s   seeds 0 1
#   trwCwGLt  agent-written reward, tuned ROSCO base,   gate 13-15 m/s   seeds 0 1
#
# Report: the same 600 s range rows as s41 - held-out seeds 3-6 and the fresh seeds 7-10 - so the rows drop straight
# into gate_table.py and lifetime_del.py next to the 15-17 gate. The fresh wind fields and their baselines already
# exist in the separate 600 s bank (~/wtrl/wind600, ~/wtrl600); the canonical bank is not touched.
#
#   setsid nohup bash scripts/wsl/campaign_gate13.sh > ~/wtrl/exp/gate13.log 2>&1 < /dev/null &
#   scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} gate13
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
MEANS="12 14 16 18 20 22 24"
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_gate13.sid"
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi
echo "=== low-gate residual start $(date) ==="
grep -q "mrwCwGLt" scripts/wsl/campaign_j_core.sh || { echo "campaign_j_core.sh lacks the low-gate arms"; exit 1; }

echo "--- 1: low-gate residual on the scheduled MPC $(date +%H:%M)"
( export WTRL_HOME=$HOME/wtrl_mpc_range WTRL_WIND=$HOME/wtrl/wind600
  export HELD_MEANS="$MEANS" HELD_TI=B HELD2=0 KNOBS=configs/knobs_j_v3_tuned.json
  ARMS="mrwCwGLt" $RUN bash scripts/wsl/campaign_j_core.sh 0 1 )
echo "--- 2: low-gate residual on the tuned ROSCO $(date +%H:%M)"
( export WTRL_HOME=$HOME/wtrl_rt_range WTRL_WIND=$HOME/wtrl/wind600 WTRL_TEMPLATE=$HOME/wtrl/runs/template_5mw_rosco_tuned
  export HELD_MEANS="$MEANS" HELD_TI=B HELD2=0 KNOBS=configs/knobs_j_v3_tuned.json
  ARMS="trwCwGLt" $RUN bash scripts/wsl/campaign_j_core.sh 0 1 )
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_gate13.sid"

echo "--- 3: 600 s range rows vs the original GSPI: seeds 3-6 and the fresh seeds 7-10 $(date +%H:%M)"
TUNED=$HOME/wtrl/runs/template_5mw_rosco_tuned
rng() {  # run port seeds tag [template]
  local run=$1 port=$2 seeds=$3 tag=$4 tpl=${5:-}
  [ -f "$EXP/$run/ckpt_best.pt" ] || { echo "missing $run"; return; }
  [ -f "$EXP/$run/eval_$tag.json" ] && { echo "skip $run $tag"; return; }
  echo "== $run $tag $(date +%H:%M)"
  WTRL_TEMPLATE=${tpl:-$HOME/wtrl/runs/template_5mw} WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 \
    $RUN python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt --backend openfast --means $MEANS --seeds $seeds --ti B \
      --episode_s 600 --workers 6 --port0 "$port" --tag "$tag" 2>&1 | grep -E "gate|J=|per-wind|Traceback|rror:"
}
( for r in mrwCwGLt_s0 trwCwGLt_s0; do rng "$r" 6800 "3 4 5 6" range_TIB_s3456 "$([ "${r:0:1}" = t ] && echo $TUNED)"; rng "$r" 6800 "7 8 9 10" range_TIB_s78910 "$([ "${r:0:1}" = t ] && echo $TUNED)"; done ) &
( for r in mrwCwGLt_s1 trwCwGLt_s1; do rng "$r" 7200 "3 4 5 6" range_TIB_s3456 "$([ "${r:0:1}" = t ] && echo $TUNED)"; rng "$r" 7200 "7 8 9 10" range_TIB_s78910 "$([ "${r:0:1}" = t ] && echo $TUNED)"; done ) &
wait
echo "=== low-gate residual done $(date) ==="
