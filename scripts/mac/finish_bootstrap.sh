#!/bin/bash
set -euo pipefail
echo "== [1/5] wind bank =="
~/wtrl/run.sh python scripts/gen_wind.py --means 8 12.5 15 --seeds 1 2 3 4 5 6 --ti 8 --time 200 --out ~/wtrl/wind --jobs 4
echo "== [2/5] toy baselines =="
~/wtrl/run.sh python scripts/make_baselines.py --backend toy --means 8 12.5 15 --seeds 1 2 3 4 5 6
echo "== [3/5] openfast baselines =="
~/wtrl/run.sh python scripts/make_baselines.py --backend openfast --means 8 12.5 15 --seeds 1 2 3 4 5 6 --jobs 6 --port0 5700
echo "== [4/5] smoke_modules =="
~/wtrl/run.sh python scripts/dev/smoke_modules.py
echo "== [5/5] toy zero-residual =="
~/wtrl/run.sh python scripts/dev/run_toy_zero.py --episode_s 60 | tail -8
echo "BOOTSTRAP_DONE"
