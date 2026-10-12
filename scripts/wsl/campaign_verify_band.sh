#!/bin/bash
# The last measurement of the ROSCO-native campaign: the band off the design winds (2026-10-12)
#
# Stage 3 verified the twelve tunings the twelve runs selected and three of the best failed the per-wind load
# check (roadmap s53.2). The band of s53.3 holds members with more regulation than the standing baseline that no
# run selected, plus the one-knob tuning that costs no actuator duty, and those have to be measured the same way
# before any of them is reported. This runs that measurement and applies the stated rule.
#
#   setsid nohup bash scripts/wsl/campaign_verify_band.sh > ~/wtrl/exp/verify_band.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_verify_band
#
# Resumable at the tuning: every evaluation is cached by parameter hash, protocol and seed set.
set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
EXP=$HOME/wtrl/exp
ps -o sid= -p $$ | tr -d ' ' > "$EXP/campaign_verify_band.sid"
export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600

echo "=== the band off the design winds, start $(date)"
$HOME/wtrl/run.sh python scripts/dev/rosco_verify_band.py \
    --workers "${WORKERS:-14}" --port0 "${PORT0:-7700}" \
    --csv docs/tables/rosco_band_verified.csv 2>&1 | grep -vE "^\[eval|^ *$"
echo "=== the band off the design winds, done $(date) ==="
