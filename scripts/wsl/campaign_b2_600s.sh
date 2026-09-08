#!/bin/bash
# B2 (2026-09-08): re-evaluate the champions on 600 s episodes.
# Fixes caveat C5: our episodes score 130 s (150 s minus 20 s warm-up) while the m = 10 blade DEL
# is set by the largest 1-5 cycles and IEC practice is 600 s. Training is untouched — this asks
# only whether the *reported* DEL reductions survive a ten-minute scoring window.
#
# Uses a separate wind bank and WTRL_HOME so the 600 s .bts / baselines cannot collide with the
# 150 s ones of the same name (they would silently overwrite each other).
# Usage: ~/wtrl/run.sh bash scripts/wsl/campaign_b2_600s.sh
set -u
EXP=~/wtrl/exp
RUNS600="tq_off_s0 tq_off_s1 tq_off_s2 tq_off_s3 tq_off_s4
         sched2_ep_s0 sched2_ep_s1 sched2_ep_s2 sched2_ep_s3 sched2_ep_s4
         n1_mono_s0 n1_mono_s1 n1_mono_s2 n1_mono_s3 n1_mono_s4"

echo "=== stage 1: 600 s wind bank $(date) ==="
python scripts/gen_wind.py --means 8 12.5 15 --seeds 3 4 5 6 --ti 8 --time 650 \
       --out ~/wtrl/wind600 --jobs 4 2>&1 | tail -3

mkdir -p ~/wtrl600/baselines
[ -e ~/wtrl600/rosco_install ] || ln -s ~/wtrl/rosco_install ~/wtrl600/rosco_install
[ -e ~/wtrl600/runs ]          || ln -s ~/wtrl/runs ~/wtrl600/runs
export WTRL_HOME=~/wtrl600 WTRL_WIND=~/wtrl/wind600

echo "=== stage 2: paired GSPI baselines at 600 s $(date) ==="
python scripts/make_baselines.py --backend openfast --means 8 12.5 15 --seeds 3 4 5 6 --ti 8 \
       --episode_s 600 --jobs 8 --port0 5700 2>&1 | grep "written\|Error" || true

echo "=== stage 3: identity check (GSPI vs its own 600 s baseline must be ~0) $(date) ==="
python scripts/evaluate.py --run "$EXP/tq_off_s0" --gspi --backend openfast --means 8 12.5 15 \
   --seeds 3 4 5 6 --episode_s 600 --workers 8 --port0 6700 --tag heldout600_gspi 2>&1 | grep "F_strict\|Error"

echo "=== stage 4: champions at 600 s $(date) ==="
for r in $RUNS600; do
  [ -d "$EXP/$r" ] || { echo "missing $r"; continue; }
  [ -f "$EXP/$r/eval_heldout600.json" ] && { echo "skip $r"; continue; }
  echo "--- $r  $(date +%H:%M:%S)"
  python scripts/evaluate.py --run "$EXP/$r" --ckpt ckpt_best.pt --backend openfast \
     --means 8 12.5 15 --seeds 3 4 5 6 --episode_s 600 --workers 8 --port0 6700 \
     --tag heldout600 2>&1 | grep "F_strict\|Traceback\|Error"
done

echo "=== stage 5: LPV-MPC row at 600 s (tuned config from roadmap 16) $(date) ==="
if [ ! -f ~/wtrl/exp/mpc600/eval_heldout600W_N20q1r0.02qt0w0.35.json ]; then
  python scripts/mpc_baseline.py --means 8 12.5 15 --seeds 3 4 5 6 --episode_s 600 \
     --horizon 20 --q 1 --r 0.02 --qt 0 --wc_v 0.35 --jobs 8 --port0 6800 \
     --out ~/wtrl/exp/mpc600 --tag heldout600W 2>&1 | tail -4
fi
echo "B2_DONE $(date)"
