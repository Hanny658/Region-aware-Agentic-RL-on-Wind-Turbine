#!/bin/bash
# Control direction, stage 2 (2026-09-16): can the MPC's actuator cost be cut without losing performance?
#
# Roadmap s22/s24: the compensated MPC reaches J ~ 21 but spends ~7.6x the GSPI pitch travel, and a third of its
# pitch-rate power sits in the 3P band (0.5-0.75 Hz) against 4 % for the GSPI - i.e. it chases blade-passing
# ripple that collective pitch cannot reject. The MPC sees the speed signal near-raw (wc_speed = 50 rad/s)
# because low-passing it added lag. A speed-scheduled notch removes the 3P line without broadband lag.
#
#   A  supervisor winds (600 s, seeds 1-2): offset-free base, notch Q in {1, 2, 5}, exact and Cp/Ct x0.95
#   B  the best Q by J: both 600 s held-out sets, exact and x0.95
#   C  regenerate the actuator-cost and pitch-spectrum tables
#
#   setsid nohup bash scripts/wsl/campaign_control2.sh > ~/wtrl/exp/control2.log 2>&1 < /dev/null &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
REFS=$EXP/mpcsearch600/refs
mkdir -p "$REFS"
export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_control2.sid"
echo "=== control stage 2 (3P notch) start $(date) ==="

ev() {   # tag port seeds cp notchQ
  local tag=$1 port=$2 seeds=$3 cp=$4 q=$5
  [ -f "$REFS/eval_$tag.json" ] && { echo "skip $tag"; return; }
  echo "== $tag  $(date +%H:%M)"
  $RUN python scripts/evaluate.py --run "$EXP/mg3t_s0" --gspi --backend openfast --seeds $seeds --episode_s 600 \
      --workers 6 --port0 "$port" --tag "$tag" --out "$REFS" --base_mm_cp "$cp" \
      --base_mpc_json "{\"horizon\": 20, \"r\": 0.3, \"qt\": 3.0, \"wc_v\": 0.35, \"adapt\": \"offset\", \"tau_adapt\": 5.0, \"notch_3p_q\": $q}" \
      2>&1 | grep -E "J=|Traceback|rror:"
}

echo "--- A: notch sweep on the supervisor winds $(date +%H:%M)"
( for q in 1 2 5; do ev "offset_notch${q}_cp1_s12" 6600 "1 2" 1 "$q"; done ) &
( for q in 1 2 5; do ev "offset_notch${q}_cp0.95_s12" 7000 "1 2" 0.95 "$q"; done ) &
wait

BEST=$($RUN python - <<'PYEOF'
import json, os
R = os.path.expanduser("~/wtrl/exp/mpcsearch600/refs")
best, bq = None, None
for q in (1, 2, 5):
    v = []
    for cp in ("1", "0.95"):
        p = f"{R}/eval_offset_notch{q}_cp{cp}_s12.json"
        if os.path.exists(p):
            v.append(json.load(open(p))["J"])
    if len(v) == 2:
        m = sum(v) / 2
        print(f"notch Q={q}: mean J over exact / x0.95 = {m:.2f}  ({v[0]:.2f} / {v[1]:.2f})", flush=True)
        if best is None or m > best:
            best, bq = m, q
ref = []
for cp in ("1", "0.95"):
    p = f"{R}/eval_offset_cp{cp}_s12.json"
    if os.path.exists(p):
        ref.append(json.load(open(p))["J"])
if len(ref) == 2:
    print(f"no notch  : mean J over exact / x0.95 = {sum(ref) / 2:.2f}  ({ref[0]:.2f} / {ref[1]:.2f})", flush=True)
open(f"{R}/notch_best.txt", "w").write(str(bq or 2))
PYEOF
)
echo "$BEST"
Q=$(cat "$REFS/notch_best.txt" 2>/dev/null || echo 2)
echo "--- B: held-out with notch Q=$Q $(date +%H:%M)"
( ev "offset_notch${Q}_cp1_s3456" 6600 "3 4 5 6" 1 "$Q"; ev "offset_notch${Q}_cp0.95_s3456" 6600 "3 4 5 6" 0.95 "$Q" ) &
( ev "offset_notch${Q}_cp1_s78910" 7000 "7 8 9 10" 1 "$Q"; ev "offset_notch${Q}_cp0.95_s78910" 7000 "7 8 9 10" 0.95 "$Q" ) &
wait

echo "--- C: tables $(date +%H:%M)"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
$RUN python scripts/dev/actuation_table.py --csv docs/tables/actuation.csv 2>&1 | tail -n 20
$RUN python scripts/dev/pitch_spectrum_table.py --csv docs/tables/pitch_spectrum.csv 2>&1 | tail -n 16
echo "=== control stage 2 done $(date) ==="
