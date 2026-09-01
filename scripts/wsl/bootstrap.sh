#!/bin/bash
# One-shot environment bootstrap for a fresh WSL Ubuntu-24.04 (no sudo needed).
# Usage: from the repo directory *inside WSL*, e.g.
#     cd /mnt/c/Users/<you>/.../Region-aware-Agentic-RL-on-Wind-Turbine
#     bash scripts/wsl/bootstrap.sh
# Builds: micromamba env `wtrl` (openfast 4.2, torch, ...), patched ROSCO libdiscon.so,
# toy DISCON dir, OpenFAST case template, ~/wtrl/run.sh, wind bank S1-S6 and GSPI baselines.
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
echo "repo: $PROJ"
mkdir -p ~/wtrl

# 1. conda env (idempotent)
tr -d '\r' < "$PROJ/scripts/wsl/setup_env.sh" > ~/wtrl/setup_env.sh
bash ~/wtrl/setup_env.sh

# 2. run.sh generated for THIS repo location
cat > ~/wtrl/run.sh <<EOF
#!/bin/bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export MAMBA_ROOT_PREFIX="\$HOME/wtrl/mamba"
export WTRL_HOME="\$HOME/wtrl"
export PROJ=$PROJ
export PYTHONPATH="\$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
cd "\$PROJ"
exec "\$HOME/wtrl/bin/micromamba" run -n wtrl "\$@"
EOF
chmod +x ~/wtrl/run.sh

# 3. ROSCO clone + patch + build
if [ ! -d ~/wtrl/ROSCO ]; then
  git clone --depth 1 --branch v2.10.5 https://github.com/NREL/ROSCO.git ~/wtrl/ROSCO
fi
tr -d '\r' < "$PROJ/scripts/wsl/build_rosco.sh" | sed "s|^PROJ=.*|PROJ=$PROJ|" > ~/wtrl/build_rosco.sh
bash ~/wtrl/build_rosco.sh

# 4. toy DISCON dir (ZMQ off, logging off, constant torque)
mkdir -p ~/wtrl/runs/toy_discon
cp ~/wtrl/ROSCO/Examples/Test_Cases/NREL-5MW/DISCON.IN \
   ~/wtrl/ROSCO/Examples/Test_Cases/NREL-5MW/Cp_Ct_Cq.NREL5MW.txt ~/wtrl/runs/toy_discon/
sed -i 's/^1                   ! LoggingLevel/0                   ! LoggingLevel/' ~/wtrl/runs/toy_discon/DISCON.IN
sed -i 's/^1                   ! VS_ConstPower/0                   ! VS_ConstPower/' ~/wtrl/runs/toy_discon/DISCON.IN

# 5. OpenFAST case template (reads configs/turbine/nrel5mw.yaml: ConstPower 0, ZMQ on, DT 0.01)
~/wtrl/run.sh python scripts/make_case_template.py --src ~/wtrl/ROSCO/Examples/Test_Cases/NREL-5MW \
   --dst ~/wtrl/runs/template_5mw --lib ~/wtrl/rosco_install/lib/libdiscon.so

# 6. wind bank + baselines (skips existing .bts / overwrites baselines; ~40 min total)
#    NOTE: to continue comparisons started on another machine, COPY its ~/wtrl/wind instead
#    (TurbSim seeds are reproducible in principle, but keep one canonical bank).
~/wtrl/run.sh python scripts/gen_wind.py --means 8 12.5 15 --seeds 1 2 3 4 5 6 --ti 8 --time 200 --out ~/wtrl/wind --jobs 4
~/wtrl/run.sh python scripts/make_baselines.py --backend toy --means 8 12.5 15 --seeds 1 2 3 4 5 6
~/wtrl/run.sh python scripts/make_baselines.py --backend openfast --means 8 12.5 15 --seeds 1 2 3 4 5 6 --jobs 6 --port0 5700

# 7. smoke tests
~/wtrl/run.sh python scripts/dev/smoke_modules.py
~/wtrl/run.sh python scripts/dev/run_toy_zero.py --episode_s 60 | tail -8
echo "BOOTSTRAP_DONE — try: ~/wtrl/run.sh python scripts/train.py --backend toy --method spec --episodes 6 --workers 3 --supervisor guard --supervise_every 6 --out ~/wtrl/exp/smoke --ckpt_every 6"
