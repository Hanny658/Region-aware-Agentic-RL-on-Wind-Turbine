"""Generate the TurbSim wind bank.  Run inside WSL (via scripts/wsl/run.sh):
    python scripts/gen_wind.py --means 8 11.4 15 --seeds 1 --ti 8 --time 200 --out ~/wtrl/wind
Output name: U{mean}_TI{ti}_S{seed}.bts  (referenced by both the toy env and OpenFAST InflowWind).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "data" / "wind" / "templates" / "turbsim_5mw.inp"


def wind_name(u: float, ti: float, seed: int) -> str:
    return f"U{u:g}_TI{ti:g}_S{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--means", nargs="+", type=float, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--ti", type=float, default=8.0, help="turbulence intensity in percent")
    ap.add_argument("--time", type=float, default=200.0, help="AnalysisTime [s]")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    tmpl = TEMPLATE.read_text()
    procs = []
    for u in args.means:
        for s in args.seeds:
            name = wind_name(u, args.ti, s)
            inp = out / f"{name}.inp"
            if (out / f"{name}.bts").exists():
                continue
            txt = (tmpl.replace("{{SEED}}", str(s)).replace("{{URef}}", f"{u:g}")
                   .replace("{{TI}}", f"{args.ti:g}").replace("{{TIME}}", f"{args.time:g}"))
            inp.write_text(txt)
            procs.append(subprocess.Popen(["turbsim", str(inp)], cwd=out,
                                          stdout=open(out / f"{name}.log", "w"), stderr=subprocess.STDOUT))
            if len(procs) >= args.jobs:
                for p in procs:
                    p.wait()
                procs = []
    for p in procs:
        p.wait()
    for f in sorted(out.glob("*.bts")):
        print(f, f.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    if shutil.which("turbsim") is None:
        raise SystemExit("turbsim not on PATH (run via scripts/wsl/run.sh)")
    main()
