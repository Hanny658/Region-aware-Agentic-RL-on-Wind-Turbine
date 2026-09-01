"""Print .bts header + basic statistics to validate envs/wind.py against TurbSim's own summary."""
import os
import struct
import sys

import numpy as np

from envs.wind import hub_series, load_bts, rotor_average_series

p = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/wtrl/wind/U8_TI8_S1.bts")
with open(p, "rb") as f:
    hdr = struct.unpack("<h4l", f.read(2 + 16))
    flt = struct.unpack("<6f", f.read(24))
    slope = struct.unpack("<3f", f.read(12))
    icpt = struct.unpack("<3f", f.read(12))
    (nchar,) = struct.unpack("<l", f.read(4))
    desc = f.read(nchar)
print("ID,nz,ny,ntwr,nt =", hdr)
print("dz,dy,dt,uhub,zhub,zbot =", [round(x, 3) for x in flt])
print("slope =", slope, " intercept =", icpt)
print("desc  =", desc[:80])
w = load_bts(p)
for i, c in enumerate("uvw"):
    print(f"{c}: mean {w.u[i].mean():7.3f}  std {w.u[i].std():6.3f}  min {w.u[i].min():7.2f} max {w.u[i].max():7.2f}")
h = hub_series(w)
ra = rotor_average_series(w, 63.0)
print(f"hub:   mean {h.mean():.3f}  TI {100 * h.std() / h.mean():.1f}%   rotor-avg: mean {ra.mean():.3f} TI {100 * ra.std() / ra.mean():.1f}%")
print("z grid:", w.z[0], "...", w.z[-1], " y grid:", w.y[0], "...", w.y[-1], " nt", w.nt, "dt", w.dt)
