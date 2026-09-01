#!/bin/bash
# Zero-shot sim-to-sim check: toy-trained best checkpoints evaluated on OpenFAST (3 wind files, 150 s).
set -u
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
i=0
for r in toy_c4_llm_s0 toy_c4_llm_s2 toy_c4_guard_s2 toy_c4_lam1 toy_c4_mono_lam1 toy_c4_mono_flag_lam1; do
  echo "=== $r ==="
  python scripts/evaluate.py --run ~/wtrl/exp/$r --backend openfast --means 8 12.5 15 --seeds 1 --workers 3 --port0 $((6000 + 10 * i)) --tag openfast_zeroshot 2>&1 | grep -v "ROSCO\|^ *$\|\*\*\*\|Developed\|Delft\|-----\|Check WE_Op\|The filtered\|A wind turbine"
  i=$((i + 1))
done
echo "=== r3only (15 m/s only) ==="
python scripts/evaluate.py --run ~/wtrl/exp/toy_c4_r3only_lam1 --backend openfast --means 15 --seeds 1 --workers 1 --port0 6100 --tag openfast_zeroshot 2>&1 | grep -v "ROSCO\|^ *$\|\*\*\*\|Developed\|Delft\|-----\|Check WE_Op\|The filtered\|A wind turbine"
echo "ZEROSHOT_DONE"
