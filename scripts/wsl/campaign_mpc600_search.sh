#!/bin/bash
# Verified search over the compensated MPC's parameters on 600 s episodes (2026-09-15).
#
#   0  600 s wind bank: add TurbSim seeds 1-2 (supervisor winds) and 7-10 (second held-out set) at 8 / 12.5 / 15 m/s
#      to ~/wtrl/wind600 (650 s fields; seeds 3-6 exist). The canonical 150 s bank (~/wtrl/wind) is not touched.
#   1  paired GSPI baselines at 600 s in the separate home ~/wtrl600 (make_baselines skips existing files)
#   2  identity check: GSPI against its own 600 s baselines on seeds 1-2 must give J ~ 0
#   3  references at 600 s (MPC as base, zero residual): nominal, offset-free and adaptive MPC (s19 settings)
#   4  search arms random / es / llm, round-robin in budget steps of 8 up to 40 verified candidates each
#   5  held-out: each arm's top 3 (supervisor-wind J) and the references on wind seeds 3-6 and 7-10, exact model;
#      each arm's best and the references with Cp/Ct x0.95 in the MPC's model
#
#   setsid nohup bash scripts/wsl/campaign_mpc600_search.sh > ~/wtrl/exp/mpc600_search.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_mpc600
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
ROOT=$EXP/mpcsearch600
mkdir -p "$ROOT"
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_mpc600.sid"
# the 600 s home and wind bank for EVERY python call below (run.sh keeps a preset WTRL_HOME)
export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600
RUN=$HOME/wtrl/run.sh
echo "=== MPC 600 s search start $(date)  WTRL_HOME=$WTRL_HOME WTRL_WIND=$WTRL_WIND ==="

echo "--- 0: wind $(date +%H:%M)"
$RUN python scripts/gen_wind.py --means 8 12.5 15 --seeds 1 2 7 8 9 10 --ti 8 --time 650 --out "$WTRL_WIND" --jobs 6 2>&1 | tail -n 3

echo "--- 1: GSPI baselines at 600 s $(date +%H:%M)"
$RUN python scripts/make_baselines.py --backend openfast --means 8 12.5 15 --seeds 1 2 7 8 9 10 --ti 8 \
     --episode_s 600 --jobs 8 --port0 5700 2>&1 | grep -E "baselines :|written|skip|Error|Traceback" | tail -n 25
ls "$WTRL_HOME/baselines/openfast" | wc -l

echo "--- 2: identity check $(date +%H:%M)"
if [ ! -f "$ROOT/eval_identity_gspi_s12.json" ]; then
  $RUN python scripts/evaluate.py --run "$EXP/jg3t_s0" --gspi --backend openfast --seeds 1 2 --episode_s 600 \
       --workers 6 --port0 6600 --tag identity_gspi_s12 --out "$ROOT" 2>&1 | grep -E "J=|Error|Traceback"
fi

# base0 <tag> <port> <extra evaluate.py args...> : MPC as base, zero residual, 600 s, into $ROOT/refs
ref() {
  local tag=$1 port=$2; shift 2
  [ -f "$ROOT/refs/eval_$tag.json" ] && { echo "skip $tag"; return; }
  mkdir -p "$ROOT/refs"; echo "== $tag $(date +%H:%M)"
  $RUN python scripts/evaluate.py --run "$EXP/mg3t_s0" --gspi --backend openfast --episode_s 600 --workers 6 \
       --port0 "$port" --tag "$tag" --out "$ROOT/refs" "$@" 2>&1 | grep -E "J=|Error|Traceback"
}
NOM='{"horizon": 20, "r": 0.3, "qt": 3.0, "wc_v": 0.35}'
OFF='{"horizon": 20, "r": 0.3, "qt": 3.0, "wc_v": 0.35, "adapt": "offset", "tau_adapt": 5.0}'
RLS='{"horizon": 20, "r": 0.3, "qt": 3.0, "wc_v": 0.35, "adapt": "rls", "tau_adapt": 5.0}'
echo "--- 3: references $(date +%H:%M)"
( ref nominal_cp1_s12 7000 --seeds 1 2 --base_mpc_json "$NOM"; ref rls_cp1_s12 7000 --seeds 1 2 --base_mpc_json "$RLS" ) &
( ref offset_cp1_s12 7300 --seeds 1 2 --base_mpc_json "$OFF" ) &
wait

echo "--- 4: search $(date +%H:%M)"
for budget in 8 16 24 32 40; do
  for arm in llm es random; do
    echo ">>> $arm budget $budget $(date +%H:%M)"
    $RUN python scripts/mpc_param_search.py --arm "$arm" --budget "$budget" --root "$ROOT" --episode_s 600 \
         --parallel 2 --workers 6 --port0 7600 2>&1 | grep -E "^\[|Error|Traceback|rror:"
  done
done

echo "--- 5: held-out $(date +%H:%M)"
$RUN python - <<'PYEOF' > "$ROOT/heldout_plan.json"
import json, os
root = os.path.expanduser("~/wtrl/exp/mpcsearch600")
plan = {}
for arm in ("llm", "es", "random"):
    H = [json.loads(l) for l in open(f"{root}/{arm}/history.jsonl")]
    plan[arm] = [h["params"] for h in sorted(H, key=lambda h: -h["result"]["J"])[:3]]
print(json.dumps(plan))
PYEOF
cat "$ROOT/heldout_plan.json"
$RUN python - <<'PYEOF'
import json, os, sys, argparse
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.getcwd())
from scripts.mpc_param_search import Evaluator, INCUMBENT0, summarise
root = os.path.expanduser("~/wtrl/exp/mpcsearch600")
plan = json.load(open(f"{root}/heldout_plan.json"))
a = argparse.Namespace(root=root, seeds=[1, 2], episode_s=600.0, workers=6, port0=8200, config_run="~/wtrl/exp/mg3t_s0")
ev = Evaluator(a)
jobs = [(INCUMBENT0, 1.0, s) for s in ([3, 4, 5, 6], [7, 8, 9, 10])]
for arm, ps in plan.items():
    for i, p in enumerate(ps):
        for s in ([3, 4, 5, 6], [7, 8, 9, 10]):
            jobs.append((p, 1.0, s))
            if i == 0:
                jobs.append((p, 0.95, s))
for s in ([3, 4, 5, 6], [7, 8, 9, 10]):
    jobs.append((INCUMBENT0, 0.95, s))
seen, uniq = set(), []
for p, cp, s in jobs:
    k = (json.dumps(p, sort_keys=True), cp, tuple(s))
    if k not in seen:
        seen.add(k); uniq.append((p, cp, s))
print(f"held-out evaluations: {len(uniq)}", flush=True)
def one(i_job):
    i, (p, cp, s) = i_job
    r = summarise(ev.run(p, i % 2, cp=cp, seeds=s))
    print(f"  {p} cp {cp} seeds {s}: J {r['J']:.2f} ({r['power']} / {r['speed']} / {r['tower']} / {r['blade']})", flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(one, enumerate(uniq)))
PYEOF
# references on the held-out sets, exact and x0.95
( for s in "3 4 5 6:s3456" "7 8 9 10:s78910"; do sd=${s%%:*}; t=${s#*:}
    ref "nominal_cp1_$t" 7000 --seeds $sd --base_mpc_json "$NOM"; ref "nominal_cp0.95_$t" 7000 --seeds $sd --base_mm_cp 0.95 --base_mpc_json "$NOM"
    ref "rls_cp1_$t" 7000 --seeds $sd --base_mpc_json "$RLS"; ref "rls_cp0.95_$t" 7000 --seeds $sd --base_mm_cp 0.95 --base_mpc_json "$RLS"; done ) &
wait
echo "=== MPC 600 s search done $(date) ==="
