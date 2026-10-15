#!/bin/bash
# The per-wind selections stage 3 does not cover (2026-10-15, roadmap s59.5)
#
# Stage 3 of campaign_rosco_native.sh re-scores each run's SET-MEAN selection off the design winds. A run whose
# per-wind selection is a DIFFERENT candidate is therefore never measured where it was not designed, and the
# delivery count of s59.4 - how many runs per arm hand a user a tuning that holds both load terms at every wind
# speed - cannot be closed without those rows. Two runs are in that position: local r4 (#10 against its set-mean
# #21) and local r5 (#18 against #9). Four evaluations of 28 episodes, about 45 min.
#
# It waits for stage 3 to release the machine instead of competing with it for the seven workers, then evaluates
# into the same parameter-hash cache (so every selection stage 3 already measured costs nothing) and prints the
# closing delivery table under all three rules.
#
#   setsid nohup bash scripts/wsl/campaign_perwind_verify.sh > ~/wtrl/exp/perwind_verify.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_perwind_verify
set -u
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
RUN=$HOME/wtrl/run.sh
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_perwind_verify.sid"

# pgrep is unreliable on this setup (CLAUDE.md), so the wait is a plain ps | grep on stage 3's own command
while ps aux | grep -q "[r]osco_search_table.py --root $EXP/roscosearch_native"; do
  sleep 60
done

echo "=== the per-wind selections stage 3 does not cover $(date +%H:%M)"
$RUN python scripts/dev/rosco_perwind_select.py --roots "$EXP/roscosearch_native" --verify --port0 8300

echo ""
echo "=== the delivery counts under all three rules $(date +%H:%M)"
$RUN python scripts/dev/rosco_proposer_test.py --root "$EXP/roscosearch_native" \
  --csv docs/tables/rosco_proposer_test.csv

echo ""
echo "=== done $(date +%H:%M)"
