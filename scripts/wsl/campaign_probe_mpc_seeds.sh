#!/bin/bash
# More seeds and controls for the MPC-stacking probe, launched only when scripts/dev/probe_decision.py says GO
# (rule fixed before the probe's results existed; user instruction of 2026-09-21).
#
#   mrwCwR3t  agent-written reward on the scheduled MPC    seeds 3 4      (n = 3 -> 5)
#   mgCwR3t   fixed reward (J-tuned)                       seeds 2 3 4    (n = 2 -> 5)
#   mrrCwR3t  random reward structure, same verification   seeds 0 1 2    (is it the agent or any structural search?)
#
# Same base (configs/mpc_sched_s33.json), paired MPC baselines (~/wtrl_mpc_range), winds, budget and objective as
# campaign_probe_mpc.sh; then every new run on the 600 s range set (seeds 3-6) against the original GSPI.
#
#   setsid nohup bash scripts/wsl/campaign_probe_mpc_seeds.sh > ~/wtrl/exp/probe_mpc_seeds.log 2>&1 < /dev/null &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
MEANS="12 14 16 18 20 22 24"
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_probe_mpc_seeds.sid"
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi
echo "=== probe seeds (agent on the scheduled MPC, n = 5 and controls) start $(date) ==="
grep -q "mrrCwR3t" scripts/wsl/campaign_j_core.sh || { echo "campaign_j_core.sh lacks the arm mrrCwR3t (probe_seeds_patch.py)"; exit 1; }
( export WTRL_HOME=$HOME/wtrl_mpc_range WTRL_WIND=$HOME/wtrl/wind600
  export HELD_MEANS="$MEANS" HELD_TI=B HELD2=0 KNOBS=configs/knobs_j_v3_tuned.json
  echo "--- 1: agent reward, seeds 3 4 $(date +%H:%M)";            ARMS="mrwCwR3t" $RUN bash scripts/wsl/campaign_j_core.sh 3 4
  echo "--- 2: random reward structure, seeds 0 1 2 $(date +%H:%M)"; ARMS="mrrCwR3t" $RUN bash scripts/wsl/campaign_j_core.sh 0 1 2
  echo "--- 3: fixed reward, seeds 2 3 4 $(date +%H:%M)";           ARMS="mgCwR3t"  $RUN bash scripts/wsl/campaign_j_core.sh 2 3 4 )
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_probe_mpc_seeds.sid"

echo "--- 4: 600 s range rows (seeds 3-6, vs the original GSPI) $(date +%H:%M)"
rng() {  # run port
  local run=$1 port=$2
  [ -f "$EXP/$run/ckpt_best.pt" ] || { echo "missing $run"; return; }
  [ -f "$EXP/$run/eval_range_TIB_s3456.json" ] && { echo "skip $run range"; return; }
  echo "== $run range $(date +%H:%M)"
  WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 $RUN python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt \
      --backend openfast --means $MEANS --seeds 3 4 5 6 --ti B --episode_s 600 --workers 6 --port0 "$port" \
      --tag range_TIB_s3456 2>&1 | grep -E "J=|per-wind|Traceback|rror:"
}
RUNS=$(cd "$EXP" && ls -d mrwCwR3t_s? mgCwR3t_s? mrrCwR3t_s? 2>/dev/null)
i=0; A=""; B=""
for r in $RUNS; do [ -f "$EXP/$r/eval_range_TIB_s3456.json" ] && continue; if [ $((i % 2)) -eq 0 ]; then A="$A $r"; else B="$B $r"; fi; i=$((i + 1)); done
( for r in $A; do rng "$r" 6600; done ) &
( for r in $B; do rng "$r" 7000; done ) &
wait
echo "=== probe seeds done $(date) ==="
