#!/bin/bash
# A2 (2026-09-08): GSPI + ROSCO's own fore-aft tower damper as a third reference controller.
# Fixes caveat C4: our GSPI baseline runs with every ROSCO load-mitigation feature off
# (TD_Mode=0, Fl_Mode=0, IPC_ControlMode=0; only PS_Mode=1), so "we cut tower-base DEL by 14 %"
# invites the question whether ROSCO's built-in damper does the same for free.
#
# Model selection is identical to the MPC baseline: sweep on the supervisor winds S1+S2, pick the
# only strict configuration with the best F, then report held-out S3-S6 and the stress classes.
# Region labels are wind-based on both sides (--relabel_wind): the damper changes ROSCO's own
# pitch command, which would otherwise move the R3 subset (roadmap 16, finding 4).
#
# Usage: ~/wtrl/run.sh bash scripts/wsl/campaign_a2_towerdamper.sh [sweep|final KI HPF SAT]
set -u
EXP=~/wtrl/exp
RUN=$EXP/gspi_td
TPL=~/wtrl/runs/template_5mw_td
export WTRL_TEMPLATE=$TPL

# ---------------------------------------------------------------- template + run dir
[ -d "$TPL" ] || cp -r ~/wtrl/runs/template_5mw "$TPL"
mkdir -p "$RUN"
cp -n $EXP/tq_off_s0/config.json $EXP/tq_off_s0/summary.json "$RUN/" 2>/dev/null

set_td() {  # KI HPF SAT   (TD_Mode=1 = translational nacelle acceleration feedback)
  sed -i -E "s|^[0-9.eE+-]+([[:space:]]+! TD_Mode)|1\1|" "$TPL/DISCON.IN"
  sed -i -E "s|^[0-9.eE+-]+([[:space:]]+! FA_KI)|$1\1|"  "$TPL/DISCON.IN"
  sed -i -E "s|^[0-9.eE+-]+([[:space:]]+! FA_HPFCornerFreq)|$2\1|" "$TPL/DISCON.IN"
  sed -i -E "s|^[0-9.eE+-]+([[:space:]]+! FA_IntSat)|$3\1|" "$TPL/DISCON.IN"
  grep -E "! (TD_Mode|FA_KI|FA_HPFCornerFreq|FA_IntSat)" "$TPL/DISCON.IN" | tr '\n' ' '; echo
}

ev() {  # tag means... -- extra
  local tag=$1; shift
  [ -f "$RUN/eval_${tag}.json" ] && { echo "skip $tag"; return; }
  python scripts/evaluate.py --run "$RUN" --gspi --relabel_wind --backend openfast \
     --workers "${WTRL_EVAL_WORKERS:-8}" --port0 6600 --tag "$tag" "$@" 2>&1 | grep "F_strict\|Traceback\|Error"
}

pick_best() {   # prints "KI HPF" of the best strict config (best F overall if none is strict)
  python - <<'PICK'
import glob, json, os, re
best_strict, best_any = None, None
for f in sorted(glob.glob(os.path.expanduser("~/wtrl/exp/gspi_td/eval_tune_*.json"))):
    j = json.load(open(f))
    m = re.search(r"tune_ki([0-9.]+)_hpf([0-9.]+)\.json$", os.path.basename(f))
    if not m:
        continue
    cand = (j["F"], m.group(1), m.group(2))
    best_any = cand if best_any is None or cand > best_any else best_any
    if j["tier"] == "strict":
        best_strict = cand if best_strict is None or cand > best_strict else best_strict
c = best_strict or best_any
if c:
    print(c[1], c[2])
PICK
}

case "${1:-auto}" in
auto)
  bash "$0" sweep
  read -r KI HPF <<< "$(pick_best)"
  if [ -z "${KI:-}" ]; then echo "A2: no sweep results to pick from"; exit 1; fi
  echo "=== A2 auto-selected FA_KI=$KI FA_HPFCornerFreq=$HPF (best strict by F; best F overall if none is strict) ==="
  bash "$0" final "$KI" "$HPF"
  ;;
sweep)
  echo "=== A2 sweep on S1+S2 (model selection by F, same rule as the MPC baseline) $(date) ==="
  SAT=0.0873          # 5 deg integrator saturation
  for KI in 0.001 0.003 0.01 0.03 0.10 0.30; do
    for HPF in 0.172 0.500; do
      tag="tune_ki${KI}_hpf${HPF}"
      [ -f "$RUN/eval_${tag}.json" ] && { echo "skip $tag"; continue; }
      echo "--- FA_KI=$KI FA_HPFCornerFreq=$HPF  $(date +%H:%M:%S)"
      set_td "$KI" "$HPF" "$SAT"
      ev "$tag" --means 8 12.5 15 --seeds 1 2
    done
  done
  echo "=== sweep table ==="
  python - <<'PY'
import glob, json, os
rows = []
for f in sorted(glob.glob(os.path.expanduser("~/wtrl/exp/gspi_td/eval_tune_*.json"))):
    j = json.load(open(f))
    rows.append((j["F"], os.path.basename(f)[10:-5], j["tier"], j["del_red_pct"],
                 j["energy_loss_pct"], j["speed_std_ratio"]))
print(f"{'config':>24} {'tier':>9} {'F':>8} {'towerDEL%':>10} {'Eloss%':>8} {'spd':>7}")
for F, name, tier, d, e, s in sorted(rows, reverse=True):
    print(f"{name:>24} {tier:>9} {F:8.2f} {d:10.2f} {e:8.3f} {s:7.4f}")
ok = [r for r in rows if r[2] == "strict"]
print("\nbest strict:", max(ok)[1] if ok else "NONE STRICT")
PY
  ;;
final)
  KI=$2; HPF=$3; SAT=${4:-0.0873}
  echo "=== A2 final config FA_KI=$KI HPF=$HPF SAT=$SAT  $(date) ==="
  set_td "$KI" "$HPF" "$SAT"
  ev heldout_s3456 --means 8 12.5 15 --seeds 3 4 5 6
  ev robust_ti14   --means 8 12.5 15 --seeds 3 4 --ti 14
  ev robust_ti22   --means 15        --seeds 3 4 --ti 22
  ev robust_u18    --means 18        --seeds 3 4 --ti 8
  # control: the same evaluation path with the damper OFF must return F = 0 (identity check)
  set_td 0.0 0.0 0.0
  sed -i -E "s|^[0-9.eE+-]+([[:space:]]+! TD_Mode)|0\1|" "$TPL/DISCON.IN"
  ev identity_td_off --means 8 12.5 15 --seeds 3 4 5 6
  ;;
esac
echo "A2_DONE $(date)"
