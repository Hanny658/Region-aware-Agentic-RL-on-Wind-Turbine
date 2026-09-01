"""Build the OpenFAST case template used by envs.openfast_env (run inside WSL):
    python scripts/make_case_template.py --src ~/wtrl/ROSCO/Examples/Test_Cases/NREL-5MW \
        --dst ~/wtrl/runs/template_5mw --lib ~/wtrl/rosco_install/lib/libdiscon.so
Changes vs. the ROSCO test case: DT = dt_sim_s, InflowWind -> TurbSim .bts, ServoDyn -> patched
libdiscon.so, DISCON.IN -> ZMQ_Mode 1 / update period dt_ctrl_s / LoggingLevel 0.
Per-episode values (TMax, wind file, initial conditions, ZMQ address) are substituted at run time.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

from envs.fast_io import set_param

PROJ = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--lib", required=True)
ap.add_argument("--turbine", default=str(PROJ / "configs" / "turbine" / "nrel5mw.yaml"))
args = ap.parse_args()

src, dst = Path(os.path.expanduser(args.src)), Path(os.path.expanduser(args.dst))
tb = yaml.safe_load(open(args.turbine))
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)

fst = dst / "NREL-5MW.fst"
set_param(fst, "DT", f"{tb['dt_sim_s']}")
set_param(fst, "OutFileFmt", "2")           # binary only
set_param(fst, "SttsTime", "1000")          # quiet stdout

ifw = dst / "NRELOffshrBsline5MW_InflowWind.dat"
set_param(ifw, "WindType", "3")
set_param(ifw, "FileName_BTS", '"WIND_PLACEHOLDER.bts"')

svd = dst / "NRELOffshrBsline5MW_Onshore_ServoDyn.dat"
set_param(svd, "DLL_FileName", f'"{os.path.expanduser(args.lib)}"')
set_param(svd, "DLL_DT", f"{tb['dt_ctrl_s']}")

dsc = dst / "DISCON.IN"
r = tb["rosco"]
set_param(dsc, "LoggingLevel", str(r["LoggingLevel"]))
set_param(dsc, "ZMQ_Mode", str(r["ZMQ_Mode"]))
set_param(dsc, "ZMQ_UpdatePeriod", f"{r['ZMQ_UpdatePeriod']}")
set_param(dsc, "ZMQ_CommAddress", '"tcp://127.0.0.1:5555"')
for k in ("VS_ControlMode", "VS_ConstPower", "PC_ControlMode", "PS_Mode", "SS_Mode", "WE_Mode"):
    set_param(dsc, k, str(r[k]))
print("template written to", dst)
for f in (fst, ifw, svd, dsc):
    print("  ", f.name)
