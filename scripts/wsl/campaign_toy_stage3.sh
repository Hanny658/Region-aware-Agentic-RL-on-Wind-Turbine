#!/bin/bash
# Toy campaign 3 (fatigue-proxy load term, guardrail everywhere). Run via ~/wtrl/run.sh bash <this>.
#   A: lambda_load sweep {0.3,1,3,10}, spec, guard, seed 0            -> toy_c3_lam{L}
#   B: supervisors on spec from lambda=1: llm x3 seeds, random x3, guard x2 more seeds -> toy_c3_{sup}_s{seed}
#   C: method comparison at the best lambda from A: mono / mono_flag (300 ep) / r3only (100 ep @15 m/s)
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error'
run() {  # name, extra args...
  name=$1; shift
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name"
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend toy --workers 9 --supervise_every 30 --out "$EXP/$name" "$@" 2>&1 | grep "$FILT"
}
for L in 0.3 1 3 10; do
  run "toy_c3_lam$L" --method spec --episodes 300 --lambda_load $L --supervisor guard --seed 0
done
for S in 0 1 2; do
  run "toy_c3_llm_s$S"    --method spec --episodes 300 --lambda_load 1 --supervisor llm    --seed $S
  run "toy_c3_random_s$S" --method spec --episodes 300 --lambda_load 1 --supervisor random --seed $S
done
for S in 1 2; do
  run "toy_c3_guard_s$S"  --method spec --episodes 300 --lambda_load 1 --supervisor guard  --seed $S
done
# best lambda from A by best F
BEST=$(python - <<'PY'
import json, glob, os
best = None
for p in glob.glob(os.path.expanduser("~/wtrl/exp/toy_c3_lam*/summary.json")):
    js = json.load(open(p)); lam = js["best_knobs"]["lambda_load_R2"]
    if best is None or js["best_F"] > best[1]:
        best = (lam, js["best_F"])
print(f"{best[0]:g}")
PY
)
echo "best lambda from sweep: $BEST"
run "toy_c3_mono_lam$BEST"      --method mono      --episodes 300 --lambda_load $BEST --supervisor guard --seed 0
run "toy_c3_mono_flag_lam$BEST" --method mono_flag --episodes 300 --lambda_load $BEST --supervisor guard --seed 0
run "toy_c3_r3only_lam$BEST"    --method r3only    --episodes 100 --lambda_load $BEST --supervisor guard --seed 0 --means 15 --eval_means 15
echo "CAMPAIGN3_DONE $(date)"
