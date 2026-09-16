#!/bin/bash
# Control direction, stage 1 (2026-09-16). Two questions the literature scan left open
# (docs/literature_2026-09-16.md):
#
#   Q1  Does a learned residual still add anything once the MPC's model error has been removed by a
#       disturbance estimator? Nobody has measured observer-recovered vs residual-learned error on one
#       plant (the closest work reports only an ordering). Arms: the residual on an OFFSET-FREE MPC base,
#       with the exact model (mo3t) and with Cp/Ct x0.95 (moC3t fixed reward, morC3t agent reward).
#   Q2  How far does a residual trained at one model error transfer? The x0.95-trained residuals are
#       evaluated at x0.85 and x1.15 (x0.9 / x1.05 already exist, roadmap s19).
#
#   Plus the evaluation-protocol items the scan says a control journal expects:
#   Q3  ROSCO with its tower damper enabled, at 600 s, as a baseline row in the main table.
#
# Training stays on the canonical 150 s bank so the new arms are directly comparable with every existing
# residual arm; the 600 s bank is used for evaluation only.
#
#   setsid nohup bash scripts/wsl/campaign_control1.sh > ~/wtrl/exp/control1.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_j     # training stage
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_control1.sid"
echo "=== control stage 1 start $(date) ==="

# ---------------------------------------------------------------- Q2 + Q3 first (cheap, no training)
echo "--- Q2: transfer of the x0.95-trained residuals to x0.85 / x1.15 $(date +%H:%M)"
transfer() {   # run port
  local run=$1 port=$2
  for cp in 0.85 1.15; do
    [ -f "$EXP/$run/eval_gen_cp${cp}_s3456.json" ] && { echo "skip $run cp$cp"; continue; }
    echo "== $run cp $cp $(date +%H:%M)"
    $RUN python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt --backend openfast --seeds 3 4 5 6 \
        --workers 6 --port0 "$port" --tag "gen_cp${cp}_s3456" --base_mm_cp "$cp" 2>&1 | grep -E "J=|Traceback|rror:"
  done
}
( for r in mgC3t_s0 mgC3t_s1 mgC3t_s2; do transfer "$r" 5900; done ) &
( for r in mrwC3t_s0 mrwC3t_s1 mrwC3t_s2; do transfer "$r" 6300; done ) &
wait

echo "--- Q3: ROSCO with the tower damper, 600 s $(date +%H:%M)"
(
  export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 WTRL_TEMPLATE=$HOME/wtrl/runs/template_5mw_td
  mkdir -p "$EXP/mpcsearch600/refs"
  for spec in "1 2:s12" "3 4 5 6:s3456" "7 8 9 10:s78910"; do
    sd=${spec%%:*}; t=${spec#*:}
    [ -f "$EXP/mpcsearch600/refs/eval_towerdamper_$t.json" ] && { echo "skip towerdamper $t"; continue; }
    echo "== towerdamper $t $(date +%H:%M)"
    $RUN python scripts/evaluate.py --run "$EXP/jg3t_s0" --gspi --backend openfast --seeds $sd --episode_s 600 \
        --workers 6 --port0 6600 --tag "towerdamper_$t" --out "$EXP/mpcsearch600/refs" 2>&1 | grep -E "J=|Traceback|rror:"
  done
)

# ---------------------------------------------------------------- Q1: training on the compensated base
echo "--- Q1: residual on the offset-free MPC base, seeds 0 1 2 $(date +%H:%M)"
KNOBS=configs/knobs_j_v3_tuned.json ARMS="moC3t morC3t mo3t" $RUN bash scripts/wsl/campaign_j_core.sh 0 1 2
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_control1.sid"

# ---------------------------------------------------------------- 600 s held-out for the new arms
echo "--- 600 s held-out for the compensated-base arms $(date +%H:%M)"
(
  export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600
  for r in moC3t_s0 moC3t_s1 moC3t_s2 morC3t_s0 morC3t_s1 morC3t_s2 mo3t_s0 mo3t_s1 mo3t_s2; do
    [ -f "$EXP/$r/ckpt_best.pt" ] || { echo "missing $r"; continue; }
    for spec in "3 4 5 6:heldout600_s3456" "7 8 9 10:heldout600_s78910"; do
      sd=${spec%%:*}; t=${spec#*:}
      [ -f "$EXP/$r/eval_$t.json" ] && { echo "skip $r $t"; continue; }
      echo "== $r $t $(date +%H:%M)"
      $RUN python scripts/evaluate.py --run "$EXP/$r" --ckpt ckpt_best.pt --backend openfast --seeds $sd \
          --episode_s 600 --workers 6 --port0 6900 --tag "$t" 2>&1 | grep -E "J=|Traceback|rror:"
    done
  done
)
echo "=== control stage 1 done $(date) ==="
