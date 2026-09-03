#!/bin/bash
# IPC campaign (2026-09-03): does a dq-frame cyclic-pitch residual (R3-only, +-1 deg/axis,
# ROSCO Coleman convention, learned phase via d/q mixing) reduce blade-root OoP DEL beyond the
# collective residual? Two arms x 5 RL seeds, seed-paired, blade objective (load_signal M_oop,
# fitness_target blade — the IPC axis; tower still reported via paper metrics):
#   ipc_on_s*  = spec + guard + --ipc_max 0.0175
#   ipc_off_s* = spec + guard                (collective-only control)
# Physical chain validated: static theta_d=+1deg -> blade1 DEL -14%; hand I-controller -> -22%
# (the classical reference point; wrong sign +37%). Lit anchor: field IPC 10-30% DEL
# (CART2/OSTI 1239055), RL-IPC ~19% (Coquelet 2022). Training 8 workers, held-out evals at
# 4 workers interleaved (thermal). Resumable.
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error\|retries'

run() {
  local name=$1; shift
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name"
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --method spec --workers 8 --supervise_every 30 --episodes 300 \
     --seeds 1 --eval_seeds 1 2 --lambda_load 1 --rollback_on violation \
     --load_signal M_oop --fitness_target blade --supervisor guard \
     --port0 5800 --out "$EXP/$name" "$@" 2>&1 | grep "$FILT"
}

I=0
ev() {
  local r=$1
  for ck in ckpt_best.pt ckpt_last.pt; do
    [ -f "$EXP/$r/$ck" ] || continue
    tag="heldout_s3456_${ck%.pt}"
    [ -f "$EXP/$r/eval_${tag}.json" ] && { echo "skip $r $tag"; continue; }
    echo "--- eval $r $ck  $(date) ---"
    python scripts/evaluate.py --run "$EXP/$r" --ckpt "$ck" --backend openfast --means 8 12.5 15 \
       --seeds 3 4 5 6 --workers 4 --port0 $((6400 + 20 * (I % 8))) --tag "$tag" 2>&1 \
       | grep "F_strict\|Traceback\|Error"
    I=$((I + 1))
  done
}

for s in 0 1 2 3 4; do
  run "ipc_on_s$s"  --seed "$s" --ipc_max 0.0175
  ev  "ipc_on_s$s"
  run "ipc_off_s$s" --seed "$s"
  ev  "ipc_off_s$s"
done
echo "=== summary ==="
python scripts/dev/heldout_table.py "~/wtrl/exp/ipc_*" || true
echo "CAMPAIGN_IPC_DONE $(date)"
