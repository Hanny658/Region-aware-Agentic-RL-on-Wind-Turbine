#!/bin/bash
# Run a command inside the `wtrl` conda env from the project root (WSL side).
# Usage (from Windows):  wsl.exe -d Ubuntu-24.04 -- ~/wtrl/run.sh python scripts/foo.py
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export MAMBA_ROOT_PREFIX="$HOME/wtrl/mamba"
export WTRL_HOME="$HOME/wtrl"
export PROJ=/mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
export PYTHONPATH="$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
cd "$PROJ"
exec "$HOME/wtrl/bin/micromamba" run -n wtrl "$@"
