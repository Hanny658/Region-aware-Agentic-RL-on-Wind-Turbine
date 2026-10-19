#!/bin/bash
# Review round 1 (2026-10-19, roadmap s63, pre-registered before this ran): two additions a reader of the leg-1
# result asks for, neither a search.
#
#  1. The one-knob control: the generator-speed filter corner alone, shipped gains, damper off, on the fixed grid
#     SWEEP_GRID of rosco_search.py (2.04, 2.5, 3.0, 3.8, 5.0, 6.5, 8.5, 11 rad/s), eight design-set evaluations -
#     the budget in which the agent found its one-line row - read under the set-mean rule, the per-wind screen and
#     the completed rule, each selection measured on held-out seeds 3-6 and fresh seeds 7-10 like every arm's.
#  2. The spectra: the shipped file, the one-line file (3.0) and the same line at 6.5 on the seven fresh wind
#     fields of seed 7 with the per-step logs kept; PSDs of the speed error, the tower-base moment and the pitch
#     rate, and the filter's gain at the plant's landmarks, in one figure (f5_spectra.png).
#
#   setsid nohup bash scripts/wsl/campaign_review1.sh > ~/wtrl/exp/review1.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_review1
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_review1.sid"
export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600

ROOT=$EXP/roscosearch_sweep
NAT=$EXP/roscosearch_native
mkdir -p "$ROOT/sweep_r0/cache" "$ROOT/_verify/cache"
# the same design-set and off-design caches every arm shares: a grid point another run already evaluated is a hit
cp -n "$NAT"/*/cache/eval_* "$ROOT/sweep_r0/cache/" 2>/dev/null
cp -n "$NAT"/_verify/cache/eval_* "$EXP"/roscosearch_rule6/_verify/cache/eval_* "$ROOT/_verify/cache/" 2>/dev/null
echo "=== s63.1 the one-knob sweep, 8 grid points on the design set $(date +%H:%M); caches warmed: $(ls "$ROOT/sweep_r0/cache" | wc -l) design, $(ls "$ROOT/_verify/cache" | wc -l) off-design"
if [ ! -f "$ROOT/sweep_r0/best.json" ]; then
  $RUN python scripts/rosco_search.py --proposer sweep --repeat 0 --budget 8 --batch 2 --parallel 2 --workers 7 \
    --means 12 14 16 18 20 22 24 --ti B --seeds 1 2 --episode_s 600 --root "$ROOT" --port0 8300 2>&1 | grep -v Warning
fi

echo ""
echo "=== s63.1 the sweep read three ways and its selections measured off the design winds $(date +%H:%M)"
$RUN python scripts/dev/rosco_sweep_report.py --run "$ROOT/sweep_r0" --verify_root "$ROOT" \
  --csv docs/tables/rosco_sweep.csv --port0 8300 2>&1 | grep -v Warning
cp -n "$ROOT"/_verify/cache/eval_* "$NAT/_verify/cache/" 2>/dev/null

echo ""
echo "=== s63.2 the spectra: three files on the seven fresh fields of seed 7, logs kept $(date +%H:%M)"
$RUN python scripts/dev/rosco_spectra.py --seeds 7 --port0 8300 2>&1 | grep -v Warning

echo ""
echo "=== done $(date +%H:%M)"
