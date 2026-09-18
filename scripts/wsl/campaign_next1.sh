#!/bin/bash
# Follow-up campaign (2026-09-18 23:00, user: keep chaining the experiments worth running), after campaign_agent_rt.sh.
#
#   A  Wind-scheduled tower weight for the compensated MPC (roadmap s31 remedy for the negative tower term from
#      20 m/s up): candidates qt_sched = [qt_hi, v_lo, v_hi] on the offset-free reference (qt 3) and the duty knee,
#      SELECTED on the range bank's TurbSim seeds 1-2 (14 episodes, 600 s) by the per-wind load rule of s28 (no term
#      below -1 % at any wind speed) then J, REPORTED on seeds 3-6 (28 episodes). The reference and the nominal MPC
#      are re-run on seeds 1-2 for the same table.
#   B  The agent's best shot where the gap is: residual on the tuned ROSCO trained on the ABOVE-RATED range winds
#      (12-24 m/s class B, first 150 s of the 600 s fields, TurbSim seed 1 training / seeds 1-2 supervisor) under the
#      per-wind objective Cw, agent-written reward (3 seeds) and fixed reward with tower weight 10 (3 seeds); paired
#      tuned-ROSCO baselines at 150 s in ~/wtrl_rt_range (canonical banks untouched); held-out = seeds 3-6 at 150 s vs
#      the tuned ROSCO; then the best seed per arm on the 600 s range set vs the original GSPI.
#
#   setsid nohup bash scripts/wsl/campaign_next1.sh > ~/wtrl/exp/next1.log 2>&1 < /dev/null &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_next1.sid"
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi
MEANS="12 14 16 18 20 22 24"
echo "=== follow-up 1 start $(date) ==="

# ---------------------------------------------------------------- A: wind-scheduled tower weight
echo "--- A: wind-scheduled tower weight, selection on seeds 1-2 $(date +%H:%M)"
OUT=$EXP/mpcsearch600/qtsched; mkdir -p "$OUT"
ev() {   # tag port json seeds
  local tag=$1 port=$2 js=$3 seeds=$4
  [ -f "$OUT/eval_$tag.json" ] && { echo "skip $tag"; return; }
  echo "== $tag  $(date +%H:%M)"
  WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 $RUN python scripts/evaluate.py --run "$EXP/mg3t_s0" --gspi --backend openfast \
      --means $MEANS --seeds $seeds --ti B --episode_s 600 --workers 6 --port0 "$port" --tag "$tag" --out "$OUT" \
      --base_mm_cp 1 --base_mpc_json "$js" 2>&1 | grep -E "J=|per-wind|Traceback|rror:"
}
OFF='"horizon": 20, "r": 0.3, "qt": 3.0, "wc_v": 0.35, "adapt": "offset", "tau_adapt": 5.0'
K2='"horizon": 15, "r": 0.408, "qt": 3.68, "wc_v": 0.425, "tau_adapt": 6.76, "adapt": "offset"'
NOM='"horizon": 20, "r": 0.3, "qt": 3.0, "wc_v": 0.35'
( ev ref_s12 6600 "{$OFF}" "1 2"
  ev off_q6_16_20_s12 6600 "{$OFF, \"qt_sched\": [6, 16, 20]}" "1 2"
  ev off_q9_16_20_s12 6600 "{$OFF, \"qt_sched\": [9, 16, 20]}" "1 2"
  ev off_q12_18_22_s12 6600 "{$OFF, \"qt_sched\": [12, 18, 22]}" "1 2" ) &
( ev nominal_s12 7000 "{$NOM}" "1 2"
  ev off_q6_18_22_s12 7000 "{$OFF, \"qt_sched\": [6, 18, 22]}" "1 2"
  ev off_q9_18_22_s12 7000 "{$OFF, \"qt_sched\": [9, 18, 22]}" "1 2"
  ev knee2_q11_18_22_s12 7000 "{$K2, \"qt_sched\": [11, 18, 22]}" "1 2" ) &
wait
SEL=$($RUN python scripts/dev/qt_sched_select.py --dir "$OUT" --suffix _s12 --n 2 --exclude nominal ref)
echo "selected on seeds 1-2: $SEL"
js_of() {  # tag -> json
  case "$1" in
    off_q6_16_20) echo "{$OFF, \"qt_sched\": [6, 16, 20]}" ;;   off_q9_16_20) echo "{$OFF, \"qt_sched\": [9, 16, 20]}" ;;
    off_q12_18_22) echo "{$OFF, \"qt_sched\": [12, 18, 22]}" ;; off_q6_18_22) echo "{$OFF, \"qt_sched\": [6, 18, 22]}" ;;
    off_q9_18_22) echo "{$OFF, \"qt_sched\": [9, 18, 22]}" ;;   knee2_q11_18_22) echo "{$K2, \"qt_sched\": [11, 18, 22]}" ;;
    *) echo "" ;;
  esac
}
i=0
for tag in $SEL; do
  js=$(js_of "$tag"); [ -n "$js" ] || continue
  port=$((6600 + 400 * (i % 2))); i=$((i + 1))
  ev "${tag}_s3456" "$port" "$js" "3 4 5 6" &
done
wait
echo "--- A done $(date +%H:%M)"

# ---------------------------------------------------------------- B: residual on the tuned ROSCO, range winds, per-wind Cw
echo "--- B: residual on the tuned ROSCO on the above-rated range winds (150 s), per-wind Cw $(date +%H:%M)"
RTR=$HOME/wtrl_rt_range
mkdir -p "$RTR"
[ -e "$RTR/runs" ] || ln -s "$HOME/wtrl/runs" "$RTR/runs"
[ -e "$RTR/rosco_install" ] || ln -s "$HOME/wtrl/rosco_install" "$RTR/rosco_install"
export WTRL_TEMPLATE=$HOME/wtrl/runs/template_5mw_rosco_tuned
( export WTRL_HOME=$RTR WTRL_WIND=$HOME/wtrl/wind600
  $RUN python scripts/make_baselines.py --backend openfast --means $MEANS --seeds 1 2 3 4 5 6 --ti B --episode_s 150 \
       --jobs 12 --port0 5700 2>&1 | grep -E "baselines :|written|skipping|Error|Traceback" | tail -n 3
  ls "$RTR/baselines/openfast"/*.npz | wc -l
  echo "--- identity check (expect 0) $(date +%H:%M)"
  [ -f "$EXP/agent_rt_identity/eval_identity_rt_range.json" ] || \
  $RUN python scripts/evaluate.py --run "$EXP/jg3t_s0" --gspi --backend openfast --means $MEANS --seeds 1 2 --ti B --episode_s 150 \
       --workers 6 --port0 6600 --tag identity_rt_range --out "$EXP/agent_rt_identity" 2>&1 | grep -E "J=|Cw=|Traceback|rror:"
  HELD_MEANS="$MEANS" HELD_TI=B HELD2=0 KNOBS=configs/knobs_j_v3_tuned.json ARMS="trwCwR3t tgCwR3L10" \
    $RUN bash scripts/wsl/campaign_j_core.sh 0 1 2 )
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_next1.sid"

echo "--- B2: best seed per arm on the 600 s range set, seeds 3-6, vs the original GSPI $(date +%H:%M)"
BEST=$($RUN python scripts/dev/pick_best_seed.py --exp "$EXP" --arms trwCwR3t tgCwR3L10)
echo "range set for: $BEST"
rng() {  # run port
  local run=$1 port=$2
  [ -f "$EXP/$run/ckpt_best.pt" ] || { echo "missing $run"; return; }
  [ -f "$EXP/$run/eval_range_TIB_s3456.json" ] && { echo "skip $run range"; return; }
  echo "== $run range $(date +%H:%M)"
  WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 $RUN python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt \
      --backend openfast --means $MEANS --seeds 3 4 5 6 --ti B --episode_s 600 --workers 6 --port0 "$port" \
      --tag range_TIB_s3456 2>&1 | grep -E "J=|per-wind|Traceback|rror:"
}
i=0
for r in $BEST; do port=$((5900 + 400 * (i % 2))); i=$((i + 1)); rng "$r" "$port" & done
wait
echo "=== follow-up 1 done $(date) ==="
