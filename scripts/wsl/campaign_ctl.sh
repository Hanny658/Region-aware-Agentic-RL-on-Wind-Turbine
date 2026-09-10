#!/bin/bash
# Pause / resume / stop / inspect a running campaign launched with setsid (its session id is written
# to ~/wtrl/exp/<campaign>.sid by the campaign script).
#
#   pause   SIGSTOP every process of the session (train.py, its workers, the OpenFAST instances):
#           the machine goes idle at once; nothing is lost.
#   resume  SIGCONT the same processes; training continues where it stopped.
#   stop    SIGTERM the session. train.py keeps its rolling resume_*.pt files (one every 5 min, 5
#           kept); re-running the campaign script continues each unfinished run from its newest one.
#   status  what is running, the newest resume point of every unfinished run, the lane log tails.
#
# Usage: bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} [campaign_j]
set -u
EXP=~/wtrl/exp
NAME=${2:-campaign_j}
SIDF=$EXP/$NAME.sid
sid() { [ -f "$SIDF" ] && cat "$SIDF"; }

case "${1:-status}" in
  pause)
    S=$(sid); [ -n "$S" ] || { echo "no $SIDF"; exit 1; }
    pkill -STOP -s "$S" && echo "paused session $S ($(pgrep -s "$S" | wc -l) processes)" ;;
  resume)
    S=$(sid); [ -n "$S" ] || { echo "no $SIDF"; exit 1; }
    pkill -CONT -s "$S" && echo "resumed session $S" ;;
  stop)
    S=$(sid); [ -n "$S" ] || { echo "no $SIDF"; exit 1; }
    pkill -CONT -s "$S" 2>/dev/null; pkill -TERM -s "$S"; sleep 3; pkill -KILL -s "$S" 2>/dev/null
    echo "stopped session $S; unfinished runs resume from their newest resume_*.pt on relaunch" ;;
  status)
    S=$(sid)
    if [ -n "$S" ] && pgrep -s "$S" >/dev/null 2>&1; then
      st=$(ps -o stat= -p "$S" 2>/dev/null | tr -d ' ')
      echo "session $S: $(pgrep -s "$S" | wc -l) processes, state ${st:-?} (T = paused)"
      ps -s "$S" -o pid=,etime=,args= | grep -o "train.py.*--out [^ ]*" | sed 's/.*--supervisor \([a-z_]*\).*--seed \([0-9]*\).*--out .*\//  \1 seed \2  -> /' || true
    else
      echo "no running session for $NAME"
    fi
    for d in "$EXP"/*/; do
      [ -f "$d/summary.json" ] && continue
      r=$(ls "$d"/resume_*.pt 2>/dev/null | tail -1) || true
      [ -n "$r" ] && echo "  unfinished $(basename "$d"): newest $(basename "$r") ($(date -r "$r" +%H:%M))"
    done
    for l in "$EXP"/j_core_lane*.log; do
      [ -f "$l" ] && { echo "--- $(basename "$l")"; tail -n 3 "$l"; }
    done ;;
  *) echo "usage: $0 {pause|resume|stop|status} [campaign]"; exit 1 ;;
esac
