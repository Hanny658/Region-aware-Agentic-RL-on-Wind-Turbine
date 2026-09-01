#!/bin/bash
# OpenFAST minimal validation set (decision 2026-08-29 evening). Train on TurbSim seed 1 only,
# evaluate best checkpoints on held-out seeds 2 and 3. Run via ~/wtrl/run.sh bash <this>.
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error\|retries'
echo "=== wind bank seeds 2,3 + baselines  $(date) ==="
python scripts/gen_wind.py --means 8 12.5 15 --seeds 2 3 --ti 8 --time 200 --out ~/wtrl/wind --jobs 3 2>&1 | tail -6
python scripts/make_baselines.py --backend toy --means 8 12.5 15 --seeds 2 3 2>&1 | grep "P=\|written"
python scripts/make_baselines.py --backend openfast --means 8 12.5 15 --seeds 2 3 --jobs 6 --port0 5700 2>&1 | grep "P=\|written"
run() {  # name, extra args...
  name=$1; shift
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name" ~/wtrl/runs/work_w*
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --workers 8 --supervise_every 30 --episodes 300 --seeds 1 \
     --out "$EXP/$name" "$@" 2>&1 | grep "$FILT"
}
run "of1_spec_guard_lam1"   --method spec --lambda_load 1   --supervisor guard --seed 0
run "of1_spec_llm_lam1"     --method spec --lambda_load 1   --supervisor llm   --seed 0
run "of1_mono_guard_lam1"   --method mono --lambda_load 1   --supervisor guard --seed 0
run "of1_spec_guard_lam0.5" --method spec --lambda_load 0.5 --supervisor guard --seed 0
echo "=== held-out evaluation (seeds 2,3) $(date) ==="
i=0
for r in of1_spec_guard_lam1 of1_spec_llm_lam1 of1_mono_guard_lam1 of1_spec_guard_lam0.5; do
  for ck in ckpt_best.pt ckpt_last.pt; do
    [ -f "$EXP/$r/$ck" ] || continue
    echo "--- $r $ck ---"
    python scripts/evaluate.py --run "$EXP/$r" --ckpt $ck --backend openfast --means 8 12.5 15 --seeds 2 3 --workers 6 \
       --port0 $((6200 + 20 * i)) --tag "heldout_s23_${ck%.pt}" 2>&1 | grep "F=\|U8:\|U12.5:\|U15:\|Traceback\|Error"
    i=$((i + 1))
  done
done
echo "CAMPAIGN_OF1_DONE $(date)"
