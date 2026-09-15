#!/bin/bash
# TCST revision campaign (2026-09-15). Every MPC row is the MPC as the environment's base controller with a
# zero residual (evaluate.py --gspi on an MPC-base run's config), as in roadmap v2 s18.
#
#   A  tune the model-error compensators on the supervisor winds (seeds 1-2, J), same rule as every arm:
#      adapt in {offset (offset-free MPC), rls (adaptive MPC)} x tau_adapt in {2, 5, 15} s x model Cp/Ct in {1, 0.95};
#      tau selected per compensator by the mean J over the two models.
#   B  held-out: the selected compensators with the exact and the x0.95 model on both held-out sets, and
#      the model-error sweep (x0.85 / 0.9 / 1.05 / 1.15) on wind seeds 3-6; the nominal MPC at x0.9 for the sweep.
#   C  residual generalisation: residuals trained on the x0.95 MPC evaluated on x0.9 / x1.05 MPC models,
#      residuals trained on the exact MPC evaluated on the x0.95 model (all wind seeds 3-6).
#   gate  if a compensated MPC alone repairs the x0.95 model as well as the fixed-reward residual (mean J
#         over both held-out sets >= 14.76), stop: the main line needs rethinking before more seeds.
#   D  seeds 3-7 for the four MPC-base arms (n = 3 -> 8), via campaign_j_core.sh (resumable, held-out evals included).
#
#   setsid nohup bash scripts/wsl/campaign_tcst.sh > ~/wtrl/exp/tcst.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_tcst
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
CFG=$EXP/mg3t_s0                 # any MPC-base run; only its config.json is used with --gspi
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_tcst.sid"
echo "=== TCST campaign start $(date) ==="

# base0 <tag> <port0> <extra evaluate.py args...>  : MPC as base, zero residual (skips finished rows)
base0() {
  local tag=$1 port=$2; shift 2
  if [ -f "$EXP/mpc/eval_$tag.json" ]; then echo "skip $tag"; return; fi
  echo "== $tag  $(date +%H:%M)"
  $RUN python scripts/evaluate.py --run "$CFG" --gspi --backend openfast --workers 8 --port0 "$port" \
      --tag "$tag" --out "$EXP/mpc" "$@" 2>&1 | grep -E "J=|ENERGY|Traceback|Error"
}
# resid <run> <tag> <port0> <extra...>  : a trained residual's best checkpoint on wind seeds 3-6
resid() {
  local run=$1 tag=$2 port=$3; shift 3
  if [ -f "$EXP/$run/eval_$tag.json" ]; then echo "skip $run $tag"; return; fi
  echo "== $run $tag  $(date +%H:%M)"
  $RUN python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt --backend openfast --seeds 3 4 5 6 \
      --workers 8 --port0 "$port" --tag "$tag" "$@" 2>&1 | grep -E "J=|ENERGY|Traceback|Error"
}
aj() { echo "{\"adapt\": \"$1\", \"tau_adapt\": $2}"; }

# ---------------------------------------------------------------- A: tuning on seeds 1-2
laneA_tune() {
  for cp in 1 0.95; do
    for tau in 2 5 15; do base0 "base0T_offset${tau}_cp${cp}_s12" 5900 --seeds 1 2 --base_mm_cp "$cp" --base_mpc_json "$(aj offset "$tau")"; done
  done
  base0 "base0T_none_cp1_s12" 5900 --seeds 1 2
}
laneB_tune() {
  for cp in 1 0.95; do
    for tau in 2 5 15; do base0 "base0T_rls${tau}_cp${cp}_s12" 6300 --seeds 1 2 --base_mm_cp "$cp" --base_mpc_json "$(aj rls "$tau")"; done
  done
  base0 "base0T_none_cp0.95_s12" 6300 --seeds 1 2 --base_mm_cp 0.95
}
echo "--- A: tuning $(date +%H:%M)"
laneA_tune > "$EXP/tcst_laneA.log" 2>&1 &
laneB_tune > "$EXP/tcst_laneB.log" 2>&1 &
wait
cat "$EXP/tcst_laneA.log" "$EXP/tcst_laneB.log"

$RUN python - <<'PYEOF'
import json, os
E = os.path.expanduser("~/wtrl/exp/mpc")
J = lambda t: json.load(open(f"{E}/eval_{t}.json"))["J"]
sel = {}
for ad in ("offset", "rls"):
    scores = {}
    for tau in (2, 5, 15):
        try:
            scores[tau] = (J(f"base0T_{ad}{tau}_cp1_s12") + J(f"base0T_{ad}{tau}_cp0.95_s12")) / 2
        except FileNotFoundError:
            pass
    best = max(scores, key=scores.get)
    sel[ad] = {"tau": best, "scores": scores}
    print(f"selected {ad}: tau {best} s  (mean J over exact / x0.95 on seeds 1-2: {scores})")
for cp in ("1", "0.95"):
    try:
        print(f"nominal MPC cp {cp} seeds 1-2: J {J(f'base0T_none_cp{cp}_s12'):.2f}")
    except FileNotFoundError:
        pass
json.dump(sel, open(f"{E}/tcst_selected.json", "w"), indent=1)
PYEOF
TAU_OFF=$(python3 -c "import json;print(json.load(open('$EXP/mpc/tcst_selected.json'))['offset']['tau'])")
TAU_RLS=$(python3 -c "import json;print(json.load(open('$EXP/mpc/tcst_selected.json'))['rls']['tau'])")
echo "selected tau: offset $TAU_OFF s, rls $TAU_RLS s"

# ---------------------------------------------------------------- B + C: held-out and generalisation
laneA_held() {
  for spec in "s3456:3 4 5 6" "s78910:7 8 9 10"; do
    t=${spec%%:*}; sd=${spec#*:}
    base0 "base0_offset_cp1_$t"    5900 --seeds $sd --base_mpc_json "$(aj offset "$TAU_OFF")"
    base0 "base0_offset_cp0.95_$t" 5900 --seeds $sd --base_mm_cp 0.95 --base_mpc_json "$(aj offset "$TAU_OFF")"
  done
  for cp in 0.85 0.9 1.05 1.15; do base0 "base0_offset_cp${cp}_s3456" 5900 --seeds 3 4 5 6 --base_mm_cp "$cp" --base_mpc_json "$(aj offset "$TAU_OFF")"; done
  base0 "base0_cp0.9_s3456" 5900 --seeds 3 4 5 6 --base_mm_cp 0.9
  for r in mgC3t_s0 mgC3t_s1 mgC3t_s2; do
    for cp in 0.9 1.05; do resid "$r" "gen_cp${cp}_s3456" 5900 --base_mm_cp "$cp"; done
  done
  for r in mg3t_s0 mg3t_s1 mg3t_s2; do resid "$r" "gen_cp0.95_s3456" 5900 --base_mm_cp 0.95; done
}
laneB_held() {
  for spec in "s3456:3 4 5 6" "s78910:7 8 9 10"; do
    t=${spec%%:*}; sd=${spec#*:}
    base0 "base0_rls_cp1_$t"    6300 --seeds $sd --base_mpc_json "$(aj rls "$TAU_RLS")"
    base0 "base0_rls_cp0.95_$t" 6300 --seeds $sd --base_mm_cp 0.95 --base_mpc_json "$(aj rls "$TAU_RLS")"
  done
  for cp in 0.85 0.9 1.05 1.15; do base0 "base0_rls_cp${cp}_s3456" 6300 --seeds 3 4 5 6 --base_mm_cp "$cp" --base_mpc_json "$(aj rls "$TAU_RLS")"; done
  for r in mrwC3t_s0 mrwC3t_s1 mrwC3t_s2; do
    for cp in 0.9 1.05; do resid "$r" "gen_cp${cp}_s3456" 6300 --base_mm_cp "$cp"; done
  done
  for r in mrw3t_s0 mrw3t_s1 mrw3t_s2; do resid "$r" "gen_cp0.95_s3456" 6300 --base_mm_cp 0.95; done
}
echo "--- B/C: held-out + generalisation $(date +%H:%M)"
laneA_held >> "$EXP/tcst_laneA.log" 2>&1 &
laneB_held >> "$EXP/tcst_laneB.log" 2>&1 &
wait
tail -n 60 "$EXP/tcst_laneA.log" "$EXP/tcst_laneB.log"

# ---------------------------------------------------------------- gate
$RUN python - <<'PYEOF'
import json, os
E = os.path.expanduser("~/wtrl/exp")
J = lambda p: json.load(open(p))["J"]
rows = {}
for ad in ("offset", "rls"):
    try:
        rows[ad] = (J(f"{E}/mpc/eval_base0_{ad}_cp0.95_s3456.json") + J(f"{E}/mpc/eval_base0_{ad}_cp0.95_s78910.json")) / 2
    except FileNotFoundError:
        pass
fixed = [(J(f"{E}/mgC3t_s{s}/eval_heldout_s3456_ckpt_best.json") + J(f"{E}/mgC3t_s{s}/eval_heldout2_s78910.json")) / 2 for s in (0, 1, 2)]
ref = sum(fixed) / len(fixed)
best = max(rows.values()) if rows else float("-inf")
print(f"gate: compensated MPC alone on the x0.95 model, mean J over both held-out sets {rows}; fixed-reward residual {ref:.2f}")
verdict = "STOP" if best >= ref else "GO"
print("gate verdict:", verdict)
open(f"{E}/mpc/tcst_gate.txt", "w").write(f"{verdict} {rows} residual_fixed {ref:.3f}\n")
PYEOF
if grep -q "^STOP" "$EXP/mpc/tcst_gate.txt"; then
  echo "=== gate STOP: a compensated MPC alone matches the residual's repair; no extra seeds launched $(date) ==="
  exit 0
fi

# ---------------------------------------------------------------- D: seeds 3-7 for the MPC-base arms
echo "--- D: seeds 3-7 $(date +%H:%M)"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
KNOBS=configs/knobs_j_v3_tuned.json ARMS="mg3t mrw3t mgC3t mrwC3t" $RUN bash scripts/wsl/campaign_j_core.sh 3 4 5 6 7
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_tcst.sid"
echo "=== TCST campaign done $(date) ==="
