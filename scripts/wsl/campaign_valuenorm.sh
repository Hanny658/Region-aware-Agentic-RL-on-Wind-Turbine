#!/bin/bash
# Value-target normalisation: does fixing the critic change the results, and does agentic
# supervision still add anything on top of a critic that works? (2026-09-10)
#
# `scripts/dev/critic_health.py` showed the R3 critic collapses to a constant in every run of the
# main line: its second tanh layer is 100 % saturated and V_std = 0.00, while the true discounted
# return is 8115 +- 394. The cause is scale, not capacity — the R3 value target is O(1e3-1e4)
# (reward ~16/step at gamma 0.998) while the critic starts at xavier gain 0.1 under grad-norm
# clipping 0.5, so it cannot travel there inside 300 episodes (v_loss flat at ~2.3e5 throughout).
# `--value_norm` standardises the regression target; everything else is unchanged.
#
# First result (guard, 2 seeds): the critic recovers (0 % saturation, corr(V, -|dw|) +0.44/+0.58)
# and held-out F goes 13.87 -> 18.20, i.e. +4.33 seed-paired — the same size as the entire
# supervision advantage measured with the broken critic. That makes the 2x2 below the decisive
# experiment: supervision was compared against a handicapped baseline.
#
#   arm \ critic     broken (existing runs)        fixed (this campaign)
#   guard            tq_off_s*      13.93 +- 2.29  vnorm_s*
#   llm_fork         n1_llmfork_s*  17.89 +- 2.02  vnorm_llm_s*
#
# Usage: ~/wtrl/run.sh bash scripts/wsl/campaign_valuenorm.sh <guard|llm_fork> <seed> [seed...]
set -u
EXP=~/wtrl/exp
FILT='init\|\[eval\]\|\[sup \]\|done in\|Traceback\|Error\|retries'
SUPV=${1:-guard}; shift || true
SEEDS=${*:-0 1}
case "$SUPV" in
  guard)      PREFIX=vnorm;     EXTRA="" ;;
  llm_fork)   PREFIX=vnorm_llm; EXTRA="--n_candidates 3" ;;   # LLM tunes the reward weights
  llm_hparam) PREFIX=vnorm_hp;  EXTRA="--n_candidates 3" ;;   # LLM tunes the PPO hyper-parameters
  llm_reward) PREFIX=vnorm_rw;  EXTRA="--n_candidates 3" ;;   # LLM writes the reward expression
  *) echo "unknown supervisor $SUPV"; exit 1 ;;
esac

run() {  # name port seed
  local name=$1 port=$2 seed=$3
  if [ -f "$EXP/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  rm -rf "$EXP/$name"
  echo "=== $name  $(date) ==="
  python scripts/train.py --backend openfast --method spec --workers 8 --supervise_every 30 \
     --episodes 300 --seeds 1 --eval_seeds 1 2 --lambda_load 1 --rollback_on violation \
     --load_signal fa_acc --fitness_target tower --obs_fa_acc --supervisor "$SUPV" $EXTRA \
     --value_norm --seed "$seed" --port0 "$port" --out "$EXP/$name" 2>&1 | grep "$FILT"
}

echo "=== value-norm campaign: $SUPV, seeds $SEEDS, start $(date) ==="
i=0
for s in $SEEDS; do
  if [ $((i % 2)) -eq 0 ]; then
    run "${PREFIX}_s$s" 5800 "$s" >> ~/wtrl/exp/${PREFIX}_laneA.log 2>&1 &
    PID_A=$!
  else
    run "${PREFIX}_s$s" 6100 "$s" >> ~/wtrl/exp/${PREFIX}_laneB.log 2>&1 &
    PID_B=$!
    wait $PID_A $PID_B
  fi
  i=$((i + 1))
done
wait
echo "=== training done $(date) ==="

j=0
for s in $SEEDS; do
  r="${PREFIX}_s$s"
  for ck in ckpt_best.pt ckpt_last.pt; do
    [ -f "$EXP/$r/$ck" ] || continue
    tag="heldout_s3456_${ck%.pt}"
    [ -f "$EXP/$r/eval_${tag}.json" ] && continue
    echo "--- $r $ck $(date +%H:%M:%S)"
    python scripts/evaluate.py --run "$EXP/$r" --ckpt "$ck" --backend openfast --means 8 12.5 15 \
       --seeds 3 4 5 6 --workers 8 --port0 $((6400 + 20 * (j % 8))) --tag "$tag" 2>&1 \
       | grep "F_strict\|Traceback\|Error"
    j=$((j + 1))
  done
  [ -f "$EXP/$r/eval_heldout2_s78910.json" ] || \
    python scripts/evaluate.py --run "$EXP/$r" --ckpt ckpt_best.pt --backend openfast \
       --means 8 12.5 15 --seeds 7 8 9 10 --workers 8 --port0 6750 --tag heldout2_s78910 2>&1 \
       | grep "F_strict\|Error"
done

echo "=== critic health ==="
python scripts/dev/critic_health.py --runs "~/wtrl/exp/${PREFIX}_s*" || true
echo "=== the 2x2 (held-out S3-S6, F_tol2) ==="
python - <<'PY'
import glob, json, os, re
import statistics as st
cells = {"guard / broken": "~/wtrl/exp/tq_off_s*", "guard / fixed": "~/wtrl/exp/vnorm_s*",
         "llm_fork / broken": "~/wtrl/exp/n1_llmfork_s*", "weights / fixed": "~/wtrl/exp/vnorm_llm_s*",
         "hparam / fixed": "~/wtrl/exp/vnorm_hp_s*", "reward-code / fixed": "~/wtrl/exp/vnorm_rw_s*"}
got = {}
for name, pat in cells.items():
    v = {}
    for d in sorted(glob.glob(os.path.expanduser(pat))):
        f = os.path.join(d, "eval_heldout_s3456_ckpt_best.json")
        if os.path.exists(f):
            v[re.search(r"_s(\d+)$", os.path.basename(d)).group(1)] = json.load(open(f))["F_tol2"]
    got[name] = v
    if v:
        m = st.mean(v.values())
        s = st.stdev(v.values()) if len(v) > 1 else 0.0
        print(f"  {name:>20}  n={len(v)}  {m:6.2f} +- {s:4.2f}   " +
              " ".join(f"s{k}:{x:.2f}" for k, x in sorted(v.items())))
print("")
print("  agentic gain over the fixed-weight baseline, WITH a working critic:")
for arm in ("weights / fixed", "hparam / fixed", "reward-code / fixed"):
    common = sorted(set(got["guard / fixed"]) & set(got.get(arm, {})))
    if not common:
        continue
    d = [got[arm][k] - got["guard / fixed"][k] for k in common]
    print("    %18s  %+6.2f   per seed %s" % (arm.split(" /")[0], st.mean(d),
          " ".join("s%s:%+.2f" % (k, x) for k, x in zip(common, d))))
print("    (reward-weight supervision with the BROKEN critic was +3.48 over 5 paired seeds)")
PY
echo "VNORM_DONE $(date)"
