#!/bin/bash
# The MPC *as the environment's base controller* with a zero residual (evaluate.py --gspi on an
# MPC-base run's config): the reference every MPC-base residual run is measured against, exact and
# with the model-mismatch / off-design variants of roadmap v2 s16, plus per-step logs of the trained
# residual runs for the trajectory figures. Two lanes of 8 OpenFAST workers.
#
#   setsid nohup bash scripts/wsl/campaign_mpcbase0.sh > ~/wtrl/exp/mpcbase0.log 2>&1 &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
CFG=$EXP/mg3t_s0          # any MPC-base run: only its config.json (base = mpc) is used with --gspi

# base0 <tag> <port0> <extra evaluate.py args...>   (skips finished rows)
base0() {
  local tag=$1 port=$2; shift 2
  if [ -f "$EXP/mpc/eval_$tag.json" ]; then echo "skip $tag"; return; fi
  echo "== $tag  $(date +%H:%M)"
  $RUN python scripts/evaluate.py --run "$CFG" --gspi --backend openfast --workers 8 --port0 "$port" \
      --tag "$tag" --out "$EXP/mpc" "$@" 2>&1 | grep -E "J=|ENERGY|Error|error"
}
# traj <run> <port0>: ckpt_best on wind seeds 3-6 with per-step logs
traj() {
  local run=$1 port=$2
  if [ -f "$EXP/$run/eval_traj_s3456.json" ]; then echo "skip traj $run"; return; fi
  echo "== traj $run  $(date +%H:%M)"
  $RUN python scripts/evaluate.py --run "$EXP/$run" --backend openfast --seeds 3 4 5 6 --workers 8 --port0 "$port" \
      --tag traj_s3456 --dump_log 2>&1 | grep -E "J=|ENERGY|Error|error|logs ->"
}

laneA() {
  base0 base0_s3456            5900 --seeds 3 4 5 6 --dump_log
  base0 base0_s78910           5900 --seeds 7 8 9 10
  base0 base0_cp0.95_s3456     5900 --seeds 3 4 5 6 --base_mm_cp 0.95 --dump_log
  base0 base0_cp0.95_s78910    5900 --seeds 7 8 9 10 --base_mm_cp 0.95
  base0 base0_cp0.85_s3456     5900 --seeds 3 4 5 6 --base_mm_cp 0.85
  base0 base0_cp1.05_s3456     5900 --seeds 3 4 5 6 --base_mm_cp 1.05
  base0 base0_cp1.15_s3456     5900 --seeds 3 4 5 6 --base_mm_cp 1.15
  base0 base0_ft0.9_s3456      5900 --seeds 3 4 5 6 --base_mm_ft 0.9
  base0 base0_ft1.1_s3456      5900 --seeds 3 4 5 6 --base_mm_ft 1.1
  base0 base0_m0.8_s3456       5900 --seeds 3 4 5 6 --base_mm_m 0.8
  base0 base0_m1.2_s3456       5900 --seeds 3 4 5 6 --base_mm_m 1.2
  base0 base0_ti14             5900 --means 8 12.5 15 --seeds 3 4 --ti 14
  base0 base0_ti22             5900 --means 15 --seeds 3 4 --ti 22
  base0 base0_u18              5900 --means 18 --seeds 3 4
  echo "lane A done $(date +%H:%M)"
}
laneB() {
  for r in mg3t_s0 mrw3t_s2 mrw3t_s1 mg3t_s1 mg3t_s2 mrw3t_s0 mgC3t_s2 mrwC3t_s1 jg3t_s0 jrw3t_s0; do
    traj "$r" 6400
  done
  echo "lane B done $(date +%H:%M)"
}
laneA > "$EXP/mpcbase0_laneA.log" 2>&1 &
laneB > "$EXP/mpcbase0_laneB.log" 2>&1 &
wait
echo "campaign done $(date +%H:%M)"
