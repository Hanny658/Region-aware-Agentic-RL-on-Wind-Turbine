#!/bin/bash
# Follow-up to campaign_mpcbase0.sh (run after its lane A has finished; same ports): the regulation-only
# MPC cost of the published verdict (unscaled references, no tower term) as the base controller.
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
CFG=$EXP/mg3t_s0
REG='{"qt": 0, "r": 0.02, "err_ref": 1.0, "dbeta_ref": 0.1}'
base0() {
  local tag=$1 port=$2; shift 2
  if [ -f "$EXP/mpc/eval_$tag.json" ]; then echo "skip $tag"; return; fi
  echo "== $tag  $(date +%H:%M)"
  $RUN python scripts/evaluate.py --run "$CFG" --gspi --backend openfast --workers 8 --port0 "$port" \
      --tag "$tag" --out "$EXP/mpc" "$@" 2>&1 | grep -E "J=|ENERGY|Error|error"
}
base0 base0_regonly_s3456   5900 --seeds 3 4 5 6 --base_mpc_json "$REG"
base0 base0_regonly_s78910  5900 --seeds 7 8 9 10 --base_mpc_json "$REG"
echo "follow-up done $(date +%H:%M)"
