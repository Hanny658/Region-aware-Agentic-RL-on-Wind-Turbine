#!/bin/bash
# Phase 0: micromamba + conda env for wind-turbine RL inside WSL (no sudo needed).
# Everything (compilers, cmake, zeromq, OpenFAST, TurbSim, PyTorch) comes from conda-forge.
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin   # drop Windows paths
WTRL=~/wtrl
mkdir -p "$WTRL" && cd "$WTRL"
if [ ! -x "$WTRL/bin/micromamba" ]; then
  echo "== installing micromamba =="
  # WSL Ubuntu image ships without bzip2 -> extract with python's tarfile instead of tar -j
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o micromamba.tar.bz2
  python3 -c "import tarfile; tarfile.open('micromamba.tar.bz2','r:bz2').extractall('.', members=[m for m in tarfile.open('micromamba.tar.bz2','r:bz2') if m.name=='bin/micromamba'])"
  rm -f micromamba.tar.bz2
  test -x bin/micromamba || { echo "micromamba extraction failed"; exit 1; }
fi
export MAMBA_ROOT_PREFIX="$WTRL/mamba"
eval "$("$WTRL/bin/micromamba" shell hook -s bash)"
if [ ! -d "$WTRL/mamba/envs/wtrl" ]; then
  echo "== creating env wtrl =="
  micromamba create -y -n wtrl -c conda-forge \
    python=3.11 openfast=4.2 \
    compilers gfortran cmake make pkg-config zeromq pyzmq \
    numpy scipy pandas matplotlib pyyaml \
    pytorch-cpu gymnasium stable-baselines3 tensorboard \
    fatpack tqdm
fi
micromamba activate wtrl
echo "== versions =="
python -c "import torch, gymnasium, stable_baselines3, fatpack, zmq; print('torch', torch.__version__, 'gym', gymnasium.__version__, 'sb3', stable_baselines3.__version__, 'zmq', zmq.zmq_version())"
for b in openfast turbsim gfortran cmake pkg-config; do printf "%-10s %s\n" "$b" "$(command -v $b || echo MISSING)"; done
openfast -v 2>&1 | grep -i "openfast-v\|compiled" | head -3 || true
pkg-config --modversion libzmq || echo "libzmq pkg-config MISSING"
echo "SETUP_DONE"
