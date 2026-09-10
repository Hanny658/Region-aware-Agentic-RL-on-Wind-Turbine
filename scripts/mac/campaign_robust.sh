#!/bin/bash
# Robustness evaluation (2026-09-06): existing champions on stress wind classes, eval-only.
# Classes: TI14@{8,12.5,15}, TI22@15, TI8@18 (seeds 3-4 each). Methods: lawrl_g_s0-4,
# ipc_off_s0-4 (RL-alone), law-alone, MPC (tuned config). No retraining.
set -u
EXP=~/wtrl/exp
SC=/private/tmp/claude-501/-Users-mac-Desktop-Region-aware-Agentic-RL-on-Wind-Turbine/3dcc4815-9a34-4aa5-8dd3-f065513f3e74/scratchpad
I=0
evc() {  # run means seeds ti tag
  local r=$1 means=$2 seeds=$3 ti=$4 tag=$5
  [ -f "$EXP/$r/eval_${tag}.json" ] && { echo "skip $r $tag"; return; }
  echo "--- $r $tag  $(date) ---"
  python scripts/evaluate.py --run "$EXP/$r" --ckpt ckpt_best.pt --backend openfast \
     --means $means --seeds $seeds --ti $ti --workers 4 --port0 $((6400 + 20 * (I % 8))) \
     --tag "$tag" 2>&1 | grep "F_strict\|Traceback\|Error"
  I=$((I + 1))
}
for r in lawrl_g_s0 lawrl_g_s1 lawrl_g_s2 lawrl_g_s3 lawrl_g_s4 ipc_off_s0 ipc_off_s1 ipc_off_s2 ipc_off_s3 ipc_off_s4; do
  evc "$r" "8 12.5 15" "3 4" 14 "robust_ti14"
  evc "$r" "15"        "3 4" 22 "robust_ti22"
  evc "$r" "18"        "3 4" 8  "robust_u18"
done
echo "=== law-alone ==="
python "$SC/robust_law.py" 8,12.5,15 3,4 14 robust_ti14
python "$SC/robust_law.py" 15 3,4 22 robust_ti22
python "$SC/robust_law.py" 18 3,4 8 robust_u18
echo "=== MPC ==="
python scripts/mpc_baseline.py --means 8 12.5 15 --seeds 3 4 --ti 14 --r 0.02 --wc_v 0.35 --tag robust_ti14 --out ~/wtrl/exp/mpc --port0 6000 --jobs 4 2>&1 | grep "^mpc"
python scripts/mpc_baseline.py --means 15 --seeds 3 4 --ti 22 --r 0.02 --wc_v 0.35 --tag robust_ti22 --out ~/wtrl/exp/mpc --port0 6000 --jobs 2 2>&1 | grep "^mpc"
python scripts/mpc_baseline.py --means 18 --seeds 3 4 --ti 8 --r 0.02 --wc_v 0.35 --tag robust_u18 --out ~/wtrl/exp/mpc --port0 6000 --jobs 2 2>&1 | grep "^mpc"
echo "CAMPAIGN_ROBUST_DONE $(date)"
