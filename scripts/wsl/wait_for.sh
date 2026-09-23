#!/bin/bash
# Block until a campaign log contains its completion line, then print the tail of every campaign log.
# Written as a file because inline `wsl.exe -- bash -c` mangles $(...) and loop variables on this setup.
#
#   bash scripts/wsl/wait_for.sh <logfile> <pattern> [max_minutes]
set -u
LOG=${1:?usage: wait_for.sh <logfile> <pattern> [max_minutes]}
PAT=${2:?}
MAX=${3:-900}
i=0
while [ "$i" -lt "$MAX" ]; do
  if [ -f "$LOG" ] && grep -q "$PAT" "$LOG"; then
    echo "DONE: '$PAT' found in $LOG after ${i} min"
    for f in "$HOME"/wtrl/exp/{gate13,mexp,rand_seeds}.log; do
      [ -f "$f" ] && { echo "--- $(basename "$f")"; tail -4 "$f"; }
    done
    exit 0
  fi
  sleep 60
  i=$((i + 1))
done
echo "TIMEOUT after ${MAX} min waiting for '$PAT' in $LOG"
for f in "$HOME"/wtrl/exp/{gate13,mexp,rand_seeds}.log; do
  [ -f "$f" ] && { echo "--- $(basename "$f")"; tail -3 "$f"; }
done
exit 1
