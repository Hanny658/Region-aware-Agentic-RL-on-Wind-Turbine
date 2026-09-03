#!/bin/bash
# torque campaign part 3 (2026-09-03 ~04:15): THE actual fix — rollout.py packed the
# (T,2) action array as reshape(-1,1), misaligning actions with obs/logp for act_dim=2;
# every PPO update was garbage (all prior tq_on failures, incl. the toy sweep). Fixed to
# reshape(len(A),-1); toy retest healthy (dtau 500: strict F 5.9), pitch-only regression
# bit-identical. WSE patch + speed gate + KE-exact reward kept as principled guards.
# Prior header: part 2c gated + capped + KE-exact reward
# (third fix: R2 task term credits P + eta*d(0.5*J*w^2)/dt via 1 s EMA, so kinetic
# draining is priced; verified const +2000Nm@8m/s -> r_task -9.0 vs -0.3 before).
# Prior header: torque campaign part 2b: tq_on arm, gated + capped build
# (WSE fix + speed gate 0.98*rated on the residual + rated-power cap in the R2 reward,
# closing the overspeed reward exploit found in tq_on2k_s0). Original header:
# torque campaign part 2: tq_on arm with the FIXED build (WSE sees applied torque,
# Filters.f90/ControllerBlocks.f90 patches). dtau_max 2000 Nm. Names tq_on3_s* to distinguish
# from the pre-fix tq_on_s0 evidence run. Control arm tq_off_s{0..4} finished 2026-09-02 23:58.
# First run doubles as pilot — watch its [eval] trajectory; kinetic-draining signature would mean
# the reward needs the kinetic-energy correction next.
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

for s in 0 1 2 3 4; do
  run "tq_on3_s$s" --seed "$s" --dtau_max 2000
  ev  "tq_on3_s$s"
done
echo "=== summary ==="
python scripts/dev/heldout_table.py "~/wtrl/exp/tq_*" || true
echo "CAMPAIGN_TORQUE2_DONE $(date)"
