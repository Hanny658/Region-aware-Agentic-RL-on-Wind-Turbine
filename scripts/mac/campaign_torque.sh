#!/bin/bash
# torque campaign (2026-09-02, mac): does the R2 torque residual (new Controllers.f90 patch +
# 2nd action dim, ZMQ_TorqueOffset, dtau_max 2000 Nm ~4.6% rated) improve the energy-constrained
# tower objective? Two arms x 5 RL seeds, seed-paired on the local wind bank:
#   tq_on_s*  = spec + guard (fixed lambda=1) + --dtau_max 2000
#   tq_off_s* = spec + guard                  (control; night1 guard was 3 seeds on the OLD bank)
# Protocol otherwise = night1/night2/sched2. Training 8 workers; held-out S3-S6 evals at 4 workers
# interleaved between runs (thermal). Resumable via summary.json / eval json.
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error\|retries'

run() {  # name extra...
  local name=$1; shift
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name"
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --method spec --workers 8 --supervise_every 30 --episodes 300 \
     --seeds 1 --eval_seeds 1 2 --lambda_load 1 --rollback_on violation \
     --load_signal fa_acc --fitness_target tower --obs_fa_acc --supervisor guard \
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

# 2026-09-02 17:05: tq_on pilot at dtau 2000 failed catastrophically (all evals degraded,
# rollback-to-ep0 every wave; cold-start exploration noise ~±740 Nm wrecks R2/transition training
# data — the torque version of the lambda_L cliff). Control arm runs first; the tq_on arm is
# re-added once the toy scale sweep picks a cold-start dtau_max. tq_on_s0 (2000 Nm) kept as evidence.
for s in 0 1 2 3 4; do
  run "tq_off_s$s" --seed "$s"
  ev  "tq_off_s$s"
done
echo "=== summary ==="
python scripts/dev/heldout_table.py "~/wtrl/exp/tq_*" || true
echo "CAMPAIGN_TORQUE_DONE $(date)"
