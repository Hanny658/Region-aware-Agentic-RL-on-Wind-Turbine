set -u
export TMPDIR=$HOME/wtrl/tmp; mkdir -p "$TMPDIR"
cd /mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
export WTRL_HOME=$HOME/wtrl600 WTRL_WIND=$HOME/wtrl/wind600
$HOME/wtrl/run.sh python scripts/rosco_search.py --proposer tune --repeat 0 --budget 0 \
  --means 12 14 16 18 20 22 24 --ti B --seeds 1 2 --episode_s 600 --workers 7 \
  --root $HOME/wtrl/exp/roscotune_native --config_run $HOME/wtrl/exp/jg3t_s0 --port0 7600
