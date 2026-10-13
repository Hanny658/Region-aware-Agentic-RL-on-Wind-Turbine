#!/bin/bash
# Leg 1: the proposer comparison at n = 6 (2026-10-13, roadmap s59)
#
# Why this rather than the fixed-reward arm under the gate: s58 removed the one place where the LLM proposer was
# measurably better than a random draw from the same space, so the only proposer statement still standing is
# leg 1 - the agent arm bought a shippable gain in 3 of 3 runs and reached its first gain in a median of 8
# simulations against 14 for constrained Bayesian optimisation and 18 for the local search - and that statement
# rests on three runs per arm, where nothing can separate 3 of 3 from 2 of 3. This doubles the three arms that
# found anything: llm, bayes, local, repeats 3, 4, 5, same box, same start, same budget of 24 candidates, same
# constrained selection, same evaluation protocol. random stays at three runs. It selected the shipped tuning
# itself in all three, its role is only to show the space is not trivially searchable, and 0 of 3 against 6 of 6
# already reads p = 0.012.
#
# The endpoints are pre-registered in roadmap s59.1 BEFORE any of these runs exists, because s55 showed the
# number the search maximises is not the number that survives: every band member buying more regulation than
# the standing baseline failed the per-wind load rule off the design winds, and all three llm selections of s53
# are in that set. Primary endpoint: simulations to the first feasible gain, Mann-Whitney one-sided, llm against
# bayes. Secondary: how many runs per arm deliver a tuning that holds both loads at every wind speed off the
# design winds, and the regulation of the ones that do.
#
#   setsid nohup bash scripts/wsl/campaign_rosco_native_n6.sh > ~/wtrl/exp/rosco_native_n6.log 2>&1 < /dev/null &
#   bash scripts/wsl/campaign_ctl.sh {pause|resume|stop|status} campaign_rosco_native
#
# Cost from the s53 log: about 1 h 45 min per run, so about 16 h for the nine, then stage 3 re-scores only the
# new selections on held-out and fresh seeds because _verify caches by parameter hash (about 12 min each).
set -u
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine

# the twelve runs of s53 must be on disk: these repeats are only useful read beside them, and stage 3 rebuilds
# the whole table in one pass
n=$(ls "$HOME"/wtrl/exp/roscosearch_native/*/best.json 2>/dev/null | wc -l)
[ "$n" -ge 12 ] || { echo "only $n of the 12 runs of s53 are on disk: run campaign_rosco_native.sh first"; exit 1; }

export ARMS="llm bayes local" REPEATS="3 4 5" STAGES="2 3"
exec bash scripts/wsl/campaign_rosco_native.sh
