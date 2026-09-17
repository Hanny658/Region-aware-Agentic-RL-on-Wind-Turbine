#!/bin/bash
# Must-do 1 (2026-09-17): wind coverage. Above-rated range 12-24 m/s in 2 m/s steps, IEC normal turbulence
# model class B (TurbSim applies TI(V): ~17 % at 12 m/s to ~14 % at 24 m/s), 600 s scored after a 20 s warm-up...
# note: our warm-up is 20 s and 580 s are scored; six realisations per wind speed (TurbSim seeds 1-6).
# None of these winds was used to select anything, so all 42 episodes are held-out for every controller.
#
#   0  wind fields into the separate 600 s bank ~/wtrl/wind600 (the canonical 150 s bank is not touched)
#   1  paired GSPI baselines at 600 s into the separate home ~/wtrl600
#   2  identity check (GSPI against its own baselines, J = 0)
#   3  controllers as the base with a zero residual: nominal MPC, offset-free reference, the duty knee (knee2,
#      knee3), the J-selected point, ROSCO with its tower damper; nominal / offset-free / knee2 also with Cp x0.95
#
#   setsid nohup bash scripts/wsl/campaign_must1_range.sh > ~/wtrl/exp/must1_range.log 2>&1 < /dev/null &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
OUT=$EXP/mpcsearch600/range
mkdir -p "$OUT"
export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_must1.sid"
MEANS="12 14 16 18 20 22 24"
SEEDS="1 2 3 4 5 6"
echo "=== must-do 1 (wind range, IEC class B) start $(date) ==="

echo "--- 0: wind $(date +%H:%M)"
$RUN python scripts/gen_wind.py --means $MEANS --seeds $SEEDS --ti B --time 650 --out "$WTRL_WIND" --jobs 7 2>&1 | tail -n 2
ls "$WTRL_WIND"/U*_TIB_S*.bts | wc -l

echo "--- 1: GSPI baselines $(date +%H:%M)"
$RUN python scripts/make_baselines.py --backend openfast --means $MEANS --seeds $SEEDS --ti B --episode_s 600 \
     --jobs 12 --port0 5700 2>&1 | grep -E "baselines :|written|skipping|Error|Traceback" | tail -n 5
ls "$WTRL_HOME/baselines/openfast"/U*_TIB_S*.npz | wc -l

ev() {   # tag port cp json [template]
  local tag=$1 port=$2 cp=$3 js=$4 tpl=${5:-}
  [ -f "$OUT/eval_$tag.json" ] && { echo "skip $tag"; return; }
  echo "== $tag  $(date +%H:%M)"
  local run="$EXP/mg3t_s0" extra=(--base_mm_cp "$cp" --base_mpc_json "$js")
  if [ -n "$tpl" ]; then run="$EXP/jg3t_s0"; extra=(); fi
  WTRL_TEMPLATE=${tpl:-$HOME/wtrl/runs/template_5mw} $RUN python scripts/evaluate.py --run "$run" --gspi --backend openfast \
      --means $MEANS --seeds $SEEDS --ti B --episode_s 600 --workers 6 --port0 "$port" --tag "$tag" --out "$OUT" "${extra[@]}" \
      2>&1 | grep -E "J=|Traceback|rror:"
}
echo "--- 2: identity check $(date +%H:%M)"
ev identity_gspi 6600 1 '{}' "$HOME/wtrl/runs/template_5mw"

NOM='{"horizon": 20, "r": 0.3, "qt": 3.0, "wc_v": 0.35}'
OFF='{"horizon": 20, "r": 0.3, "qt": 3.0, "wc_v": 0.35, "adapt": "offset", "tau_adapt": 5.0}'
K2='{"horizon": 15, "r": 0.408, "qt": 3.68, "wc_v": 0.425, "tau_adapt": 6.76, "adapt": "offset"}'
K3='{"horizon": 15, "r": 0.294, "qt": 4.27, "wc_v": 0.41, "tau_adapt": 5.52, "adapt": "offset"}'
JS='{"horizon": 20, "r": 0.255, "qt": 3.25, "wc_v": 0.38, "tau_adapt": 5.2, "adapt": "rls"}'
echo "--- 3: controllers over the range $(date +%H:%M)"
( ev offset_cp1 6600 1 "$OFF"; ev nominal_cp1 6600 1 "$NOM"; ev jselected_cp1 6600 1 "$JS"; ev offset_cp0.95 6600 0.95 "$OFF"; ev towerdamper 6600 1 '{}' "$HOME/wtrl/runs/template_5mw_td" ) &
( ev knee2_cp1 7000 1 "$K2"; ev knee3_cp1 7000 1 "$K3"; ev knee2_cp0.95 7000 0.95 "$K2"; ev nominal_cp0.95 7000 0.95 "$NOM" ) &
wait
echo "=== must-do 1 done $(date) ==="
