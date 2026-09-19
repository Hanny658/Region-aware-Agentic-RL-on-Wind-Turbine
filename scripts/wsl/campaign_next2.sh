#!/bin/bash
# Follow-up 2 (2026-09-19 23:40), after campaign_next1.sh: the clean re-run of the residual on the tuned ROSCO.
#
# Every run of campaign_agent_rt.sh was TRAINED and selected with the oracle-rule above-rated subset (train.py enabled
# the wind-labelled subset only for objective J; fixed in a969ce9). Re-evaluated with the wind label, the set-mean agent
# runs are large losses at 12.5 m/s (power -45...-86 %, speed -139...-307 %) that their selection never saw. The negative
# conclusion "no per-wind one-sided gain" therefore needs runs whose selection saw the right subset:
#
#   trwCwF3t   agent-written reward, per-wind Cw, fixed labelling     3 seeds
#   tgCwF3L10  fixed reward (tower weight 10), per-wind Cw            3 seeds
#   trwTwF3t   agent-written reward, per-wind CTw (tower target)      3 seeds
#
# Same base (tuned ROSCO), same paired baselines (~/wtrl_rt), same 150 s bank, same budget as campaign_agent_rt.sh.
# First: the absolute rows of the episode-0 runs of campaign_agent_rt.sh that still carry the oracle label are redone.
#
#   setsid nohup bash scripts/wsl/campaign_next2.sh > ~/wtrl/exp/next2.log 2>&1 < /dev/null &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
export WTRL_TEMPLATE=$HOME/wtrl/runs/template_5mw_rosco_tuned
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_next2.sid"
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi
echo "=== follow-up 2 (clean re-run with the wind-labelled subset) start $(date) ==="

echo "--- 0: stale absolute rows (oracle label) of the episode-0 runs $(date +%H:%M)"
abs() {  # run port
  local r=$1 port=$2
  WTRL_HOME=$HOME/wtrl WTRL_WIND=$HOME/wtrl/wind $RUN python scripts/evaluate.py --run "$EXP/$r" --ckpt ckpt_best.pt --backend openfast \
      --seeds 3 4 5 6 --workers 6 --port0 "$port" --tag abs_gspi_s3456 2>&1 | grep -E "J=|Traceback|rror:"
}
STALE=""
for r in tgCw3L10_s0 tgCw3L10_s1 tgCw3L10_s2 tgTw3L10_s0 tgTw3L10_s1 tgTw3L10_s2 tgC3t_s1 tgC3t_s2; do
  f="$EXP/$r/eval_abs_gspi_s3456.json"
  [ -f "$f" ] && [ ! -f "$EXP/$r/eval_abs_gspi_s3456_oracle.json" ] && { mv "$f" "$EXP/$r/eval_abs_gspi_s3456_oracle.json"; STALE="$STALE $r"; }
done
i=0; A=""; B=""
for r in $STALE; do if [ $((i % 2)) -eq 0 ]; then A="$A $r"; else B="$B $r"; fi; i=$((i + 1)); done
( for r in $A; do abs "$r" 5900; done ) &
( for r in $B; do abs "$r" 6300; done ) &
wait

echo "--- 1: training with the fixed labelling $(date +%H:%M)"
( export WTRL_HOME=$HOME/wtrl_rt WTRL_WIND=$HOME/wtrl/wind
  KNOBS=configs/knobs_j_v3_tuned.json ARMS="trwCwF3t tgCwF3L10 trwTwF3t" $RUN bash scripts/wsl/campaign_j_core.sh 0 1 2 )
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_next2.sid"

echo "--- 2: absolute rows vs the ORIGINAL GSPI (wind seeds 3-6) $(date +%H:%M)"
RUNS=$(cd "$EXP" && ls -d trwCwF3t_s? tgCwF3L10_s? trwTwF3t_s? 2>/dev/null)
i=0; A=""; B=""
for r in $RUNS; do [ -f "$EXP/$r/eval_abs_gspi_s3456.json" ] && continue; if [ $((i % 2)) -eq 0 ]; then A="$A $r"; else B="$B $r"; fi; i=$((i + 1)); done
( for r in $A; do abs "$r" 5900; done ) &
( for r in $B; do abs "$r" 6300; done ) &
wait
echo "=== follow-up 2 done $(date) ==="
