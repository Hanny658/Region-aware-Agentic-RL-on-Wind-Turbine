#!/bin/bash
# Residual on the TUNED ROSCO under a constrained objective (2026-09-18, user decision after roadmap s27-s31).
#
# Question: can the agent-supervised residual add a one-sided gain on top of a tuned industrial baseline
# (regulation up with both fatigue loads not worse, or tower fatigue down with regulation and blade not worse)?
# Every earlier residual arm was selected under J, which lets one term pay for another; here selection,
# rollback and the supervisors' fitness use the constrained objectives C / CT of eval/fitness.py.
#
#   base            tuned ROSCO of roadmap s29 (WTRL_TEMPLATE = ~/wtrl/runs/template_5mw_rosco_tuned)
#   reference       paired baselines of the SAME tuned ROSCO on the canonical 150 s bank, in the separate home
#                   ~/wtrl_rt (the canonical wind bank is read, never regenerated), so every reduction is
#                   "on top of the tuned ROSCO"; the identity check (zero residual) must give C = CT = J = 0
#   arms            tgC3t   guard (fixed reward v3, J-tuned weights)  objective C   (regulation target, loads constrained)
#                   trwC3t  llm_reward                                objective C
#                   tgC3L10 guard, reward tower weight 10             objective C
#                   tgT3L10 guard, reward tower weight 10             objective CT  (tower target, regulation + blade constrained)
#                   trwT3t  llm_reward                                objective CT
#                   tgT3t   guard (J-tuned weights)                   objective CT
#                   (ordered by information value; the fixed J-tuned reward under C learns to buy regulation with tower
#                   fatigue and is rolled back to episode 0 every evaluation, seen at 13:30 on the first two runs)
#                   3 seeds each, 300 episodes, 150 s bank, the j_core stage evaluates both held-out sets vs the
#                   tuned ROSCO (WTRL_HOME=~/wtrl_rt)
#   afterwards      absolute rows vs the ORIGINAL GSPI on wind seeds 3-6 (WTRL_HOME=~/wtrl), and the 600 s range
#                   set (12-24 m/s class B, vs the original GSPI) for the extrapolation test
#
#   setsid nohup bash scripts/wsl/campaign_agent_rt.sh > ~/wtrl/exp/agent_rt.log 2>&1 < /dev/null &
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
RT=$HOME/wtrl_rt
export WTRL_TEMPLATE=$HOME/wtrl/runs/template_5mw_rosco_tuned
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_agent_rt.sid"
WINPY=/mnt/c/Users/hanny/AppData/Local/Programs/Python/Python313/python.exe
if ! curl -s -o /dev/null --max-time 10 https://example.com && [ -x "$WINPY" ]; then
  export WTRL_LLM_WINPY=$WINPY; echo "WSL has no outbound network: LLM calls relayed through Windows"
fi
[ -d "$WTRL_TEMPLATE" ] || { echo "missing tuned template $WTRL_TEMPLATE (rosco_tune.py --emit_template)"; exit 1; }
echo "=== residual on the tuned ROSCO, constrained objectives: start $(date) ==="

echo "--- 0: paired baselines of the tuned ROSCO (150 s bank, seeds 1-10) $(date +%H:%M)"
mkdir -p "$RT"
[ -e "$RT/runs" ] || ln -s "$HOME/wtrl/runs" "$RT/runs"
[ -e "$RT/rosco_install" ] || ln -s "$HOME/wtrl/rosco_install" "$RT/rosco_install"
( export WTRL_HOME=$RT WTRL_WIND=$HOME/wtrl/wind
  $RUN python scripts/make_baselines.py --backend openfast --means 8 12.5 15 --seeds 1 2 3 4 5 6 7 8 9 10 \
       --jobs 12 --port0 5700 2>&1 | grep -E "baselines :|written|skipping|Error|Traceback" | tail -n 5
  ls "$RT/baselines/openfast"/*.npz | wc -l
  echo "--- identity check (zero residual on the tuned ROSCO vs its own baselines; expect 0) $(date +%H:%M)"
  [ -f "$EXP/agent_rt_identity/eval_identity_rt.json" ] || \
  $RUN python scripts/evaluate.py --run "$EXP/jg3t_s0" --gspi --backend openfast --seeds 1 2 --workers 6 --port0 6600 \
       --tag identity_rt --out "$EXP/agent_rt_identity" 2>&1 | grep -E "J=|C=|Traceback|rror:" )

echo "--- 1: training $(date +%H:%M)"
( export WTRL_HOME=$RT WTRL_WIND=$HOME/wtrl/wind
  KNOBS=configs/knobs_j_v3_tuned.json ARMS="tgC3t trwC3t tgC3L10 tgT3L10 trwT3t tgT3t" $RUN bash scripts/wsl/campaign_j_core.sh 0 1 2 )
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_agent_rt.sid"

echo "--- 2: absolute rows vs the ORIGINAL GSPI (wind seeds 3-6) $(date +%H:%M)"
abs() {  # run port
  local run=$1 port=$2
  [ -f "$EXP/$run/ckpt_best.pt" ] || { echo "missing $run"; return; }
  [ -f "$EXP/$run/eval_abs_gspi_s3456.json" ] && { echo "skip $run abs"; return; }
  echo "== $run abs $(date +%H:%M)"
  WTRL_HOME=$HOME/wtrl WTRL_WIND=$HOME/wtrl/wind $RUN python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt \
      --backend openfast --seeds 3 4 5 6 --workers 6 --port0 "$port" --tag abs_gspi_s3456 2>&1 | grep -E "J=|C=|Traceback|rror:"
}
RUNS=$(cd "$EXP" && ls -d tgC3t_s? trwC3t_s? tgC3L10_s? tgT3L10_s? trwT3t_s? tgT3t_s? 2>/dev/null)
i=0; A=""; B=""
for r in $RUNS; do if [ $((i % 2)) -eq 0 ]; then A="$A $r"; else B="$B $r"; fi; i=$((i + 1)); done
( for r in $A; do abs "$r" 5900; done ) &
( for r in $B; do abs "$r" 6300; done ) &
wait

echo "--- 3: range set (12-24 m/s class B, 600 s, vs the original GSPI) $(date +%H:%M)"
rng() {  # run port
  local run=$1 port=$2
  [ -f "$EXP/$run/ckpt_best.pt" ] || { echo "missing $run"; return; }
  [ -f "$EXP/$run/eval_range_TIB.json" ] && { echo "skip $run range"; return; }
  echo "== $run range $(date +%H:%M)"
  WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600 $RUN python scripts/evaluate.py --run "$EXP/$run" --ckpt ckpt_best.pt \
      --backend openfast --means 12 14 16 18 20 22 24 --seeds 1 2 3 4 5 6 --ti B --episode_s 600 --workers 6 --port0 "$port" \
      --tag range_TIB 2>&1 | grep -E "J=|C=|Traceback|rror:"
}
# the best seed of every arm by its held-out objective vs the tuned ROSCO (C for the C arms, CT for the CT arms)
BEST=$($RUN python scripts/dev/pick_best_seed.py --exp "$EXP" --arms tgC3t trwC3t tgC3L10 tgT3L10 trwT3t tgT3t)
echo "range set for: $BEST"
i=0; A=""; B=""
for r in $BEST; do if [ $((i % 2)) -eq 0 ]; then A="$A $r"; else B="$B $r"; fi; i=$((i + 1)); done
( for r in $A; do rng "$r" 5900; done ) &
( for r in $B; do rng "$r" 6300; done ) &
wait
echo "=== residual on the tuned ROSCO done $(date) ==="
