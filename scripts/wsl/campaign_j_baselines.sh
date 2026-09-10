#!/bin/bash
# Stage 2 of roadmap v2 (docs/roadmap_2026-09-11_metric-set-objective.md): the reference controllers
# re-selected under the metric-set objective J, with the same rule as every RL arm (best J on the
# supervisor winds S1+S2 subject to energy loss <= 1 %; best J overall if nothing satisfies it).
#
#   1. GSPI identity: the zero-residual controller through the evaluation path must give J = 0.
#   2. LPV-MPC grid (r x qt x wc_v) on S1+S2 scored by J, then held-out S3-S6 and S7-S10 for the winner.
#   3. ROSCO tower damper: the existing gain sweep (campaign_a2, wind-labelled R3 on both sides, the
#      convention J uses) is re-picked by J without re-simulation; the winner gets both held-out sets.
#
# Every step skips work whose eval json already exists, so the script can be stopped and re-run.
# Usage: ~/wtrl/run.sh bash scripts/wsl/campaign_j_baselines.sh
set -u
EXP=~/wtrl/exp
FILT='F_strict\|   J=\|mpc \[\|Traceback\|Error'
echo "=== stage 2: baselines under J  $(date) ==="

# ---------------------------------------------------------------- 1. identity
CHK=$EXP/_j_checks
mkdir -p "$CHK"; cp -n $EXP/tq_off_s0/config.json $EXP/tq_off_s0/summary.json "$CHK/" 2>/dev/null
if [ ! -f "$CHK/eval_identity_J_s3456.json" ]; then
  python scripts/evaluate.py --run "$CHK" --gspi --relabel_wind --backend openfast --means 8 12.5 15 \
     --seeds 3 4 5 6 --workers 8 --port0 6600 --tag identity_J_s3456 2>&1 | grep "$FILT"
fi

# ---------------------------------------------------------------- 2. MPC re-tune by J
echo "--- MPC grid on S1+S2 (skips existing points)  $(date +%H:%M:%S)"
python scripts/mpc_baseline.py --means 8 12.5 15 --seeds 1 2 --horizon 20 --q 1 \
   --r 0.01 0.02 0.05 0.1 --qt 0 0.3 1 --wc_v 0.25 0.35 --jobs 8 --port0 6800 \
   --out $EXP/mpc --tag tuneJ_s12 2>&1 | grep "$FILT"
read -r R QT W <<< "$(python scripts/dev/pick_by_J.py "$EXP/mpc/eval_tuneJ_s12_*.json" --regex 'r([0-9.]+)qt([0-9.]+)w([0-9.]+)\.json$' 2>"$EXP/mpc/pick_J.txt")"
cat "$EXP/mpc/pick_J.txt"
echo "=== MPC selected under J: r=$R qt=$QT wc_v=$W ==="
for spec in "heldout_s3456:3 4 5 6" "heldout2W:7 8 9 10"; do
  tag=${spec%%:*}; seeds=${spec#*:}
  [ -f "$EXP/mpc/eval_${tag}_N20q1r${R}qt${QT}w${W}.json" ] && { echo "skip mpc $tag"; continue; }
  python scripts/mpc_baseline.py --means 8 12.5 15 --seeds $seeds --horizon 20 --q 1 --r "$R" --qt "$QT" \
     --wc_v "$W" --jobs 8 --port0 6800 --out $EXP/mpc --tag "$tag" 2>&1 | grep "$FILT"
done

# ---------------------------------------------------------------- 3. tower damper re-pick by J
RUN=$EXP/gspi_td
TPL=~/wtrl/runs/template_5mw_td
export WTRL_TEMPLATE=$TPL
[ -d "$TPL" ] || cp -r ~/wtrl/runs/template_5mw "$TPL"
set_td() {  # KI HPF SAT
  sed -i -E "s|^[0-9.eE+-]+([[:space:]]+! TD_Mode)|1\1|" "$TPL/DISCON.IN"
  sed -i -E "s|^[0-9.eE+-]+([[:space:]]+! FA_KI)|$1\1|"  "$TPL/DISCON.IN"
  sed -i -E "s|^[0-9.eE+-]+([[:space:]]+! FA_HPFCornerFreq)|$2\1|" "$TPL/DISCON.IN"
  sed -i -E "s|^[0-9.eE+-]+([[:space:]]+! FA_IntSat)|$3\1|" "$TPL/DISCON.IN"
  grep -E "! (TD_Mode|FA_KI|FA_HPFCornerFreq|FA_IntSat)" "$TPL/DISCON.IN" | tr '\n' ' '; echo
}
read -r KI HPF <<< "$(python scripts/dev/pick_by_J.py "$RUN/eval_tune_*.json" --regex 'ki([0-9.]+)_hpf([0-9.]+)\.json$' 2>"$RUN/pick_J.txt")"
cat "$RUN/pick_J.txt"
echo "=== tower damper selected under J: FA_KI=$KI FA_HPFCornerFreq=$HPF ==="
set_td "$KI" "$HPF" 0.0873
for spec in "J_heldout_s3456:3 4 5 6" "J_heldout2W:7 8 9 10"; do
  tag=${spec%%:*}; seeds=${spec#*:}
  [ -f "$RUN/eval_${tag}.json" ] && { echo "skip td $tag"; continue; }
  python scripts/evaluate.py --run "$RUN" --gspi --relabel_wind --backend openfast --means 8 12.5 15 \
     --seeds $seeds --workers 8 --port0 6600 --tag "$tag" 2>&1 | grep "$FILT"
done
echo "{\"KI\": \"$KI\", \"HPF\": \"$HPF\", \"objective\": \"J\"}" > "$RUN/selected_J.json"
echo "STAGE2_DONE $(date)"
