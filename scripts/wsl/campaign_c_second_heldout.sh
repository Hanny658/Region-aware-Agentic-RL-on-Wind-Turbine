#!/bin/bash
# C (2026-09-09): a SECOND, disjoint held-out wind set.
#
# Every F in this project is measured on the same four TurbSim realisations (S3-S6). The central
# claim after the seed extension — fork-verified LLM supervision beats verified random search by
# +5.5 F (7/7 seeds, exact p = 0.016) — would collapse if that ranking were a property of those
# four winds. It is not a hypothetical risk: when the campaign moved between wind banks the sign
# of this very comparison flipped for two seeds (caveat C1). This evaluates every trained policy
# on four fresh realisations (S7-S10) it has never seen, training untouched.
#
# Pure evaluation, resumable. Usage: ~/wtrl/run.sh bash scripts/wsl/campaign_c_second_heldout.sh
set -u
EXP=~/wtrl/exp
SEEDS="7 8 9 10"
RUNS="tq_off_s0 tq_off_s1 tq_off_s2 tq_off_s3 tq_off_s4
      n1_mono_s0 n1_mono_s1 n1_mono_s2 n1_mono_s3 n1_mono_s4
      n1_llmfork_s0 n1_llmfork_s1 n1_llmfork_s2 n1_llmfork_s3 n1_llmfork_s4 n1_llmfork_s5 n1_llmfork_s6
      n1_randfork_s0 n1_randfork_s1 n1_randfork_s2 n1_randfork_s3 n1_randfork_s4 n1_randfork_s5 n1_randfork_s6
      n1_llmsingle_s0 n1_llmsingle_s1 n1_llmsingle_s2 n1_llmsingle_s3 n1_llmsingle_s4
      sched2_ep_s0 sched2_ep_s1 sched2_ep_s2 sched2_ep_s3 sched2_ep_s4"

echo "=== stage 1: wind S7-S10 (fresh realisations, same means/TI) $(date) ==="
python scripts/gen_wind.py --means 8 12.5 15 --seeds $SEEDS --ti 8 --time 200 \
       --out ~/wtrl/wind --jobs 4 2>&1 | tail -3

echo "=== stage 2: paired GSPI baselines for S7-S10 $(date) ==="
python scripts/make_baselines.py --backend openfast --means 8 12.5 15 --seeds $SEEDS --ti 8 \
       --jobs 8 --port0 5700 2>&1 | grep "written\|skipping\|Error" || true

echo "=== stage 3: identity check on the new set (must be ~0) $(date) ==="
python scripts/evaluate.py --run "$EXP/tq_off_s0" --gspi --backend openfast --means 8 12.5 15 \
   --seeds $SEEDS --workers 8 --port0 6750 --tag heldout2_gspi 2>&1 | grep "F_strict\|Error"

echo "=== stage 4: every trained policy on S7-S10 $(date) ==="
for r in $RUNS; do
  [ -d "$EXP/$r" ] || { echo "missing $r"; continue; }
  [ -f "$EXP/$r/eval_heldout2_s78910.json" ] && { echo "skip $r"; continue; }
  echo "--- $r  $(date +%H:%M:%S)"
  python scripts/evaluate.py --run "$EXP/$r" --ckpt ckpt_best.pt --backend openfast \
     --means 8 12.5 15 --seeds $SEEDS --workers 8 --port0 6750 --tag heldout2_s78910 2>&1 \
     | grep "F_strict\|Traceback\|Error"
done

echo "=== stage 5: LPV-MPC on the new set $(date) ==="
if [ ! -f ~/wtrl/exp/mpc/eval_heldout2W_N20q1r0.02qt0w0.35.json ]; then
  python scripts/mpc_baseline.py --means 8 12.5 15 --seeds $SEEDS --horizon 20 --q 1 --r 0.02 \
     --qt 0 --wc_v 0.35 --jobs 8 --port0 6800 --out ~/wtrl/exp/mpc --tag heldout2W 2>&1 | tail -3
fi

echo "=== stage 6: does the ranking survive the new evaluation wind? ==="
echo "--- original held-out S3-S6 ---"
python scripts/dev/stats_table.py || true
echo "--- SECOND held-out S7-S10 ---"
python scripts/dev/stats_table.py --eval eval_heldout2_s78910.json || true
echo "C_DONE $(date)"
