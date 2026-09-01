"""TurbSim full-field (.bts) reader + hub / rotor-averaged wind series.

Only the pieces we need: no tower points, no interpolation in time. Format follows the
TurbSim user guide (binary FF, ID 7/8). The same file feeds both the toy env and OpenFAST
(InflowWind WindType=3) so both backends see identical wind.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BTSWind:
    u: np.ndarray        # (3, nt, ny, nz)  [m/s]  components u (along-wind), v, w
    y: np.ndarray        # (ny,)  lateral grid [m], centred at 0
    z: np.ndarray        # (nz,)  vertical grid [m], absolute height
    dt: float
    u_hub: float         # mean hub-height wind speed written by TurbSim
    z_hub: float

    @property
    def nt(self) -> int:
        return self.u.shape[1]

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.nt) * self.dt


def load_bts(path: str | Path) -> BTSWind:
    with open(path, "rb") as f:
        _id, nz, ny, ntwr, nt = struct.unpack("<h4l", f.read(2 + 4 * 4))
        dz, dy, dt, u_hub, z_hub, z_bottom = struct.unpack("<6f", f.read(6 * 4))
        # written per component as (slope, intercept) pairs, i.e. interleaved
        si = struct.unpack("<6f", f.read(24))
        slope = np.array(si[0::2])
        intercept = np.array(si[1::2])
        (nchar,) = struct.unpack("<l", f.read(4))
        f.read(nchar)  # description string
        # per time step: grid block (z, y, comp) followed by ntwr tower points (comp)
        per_t = 3 * ny * nz + 3 * ntwr
        raw = np.fromfile(f, dtype="<i2", count=per_t * nt).reshape(nt, per_t)[:, :3 * ny * nz]
    # file order: t, z, y, component
    raw = raw.reshape(nt, nz, ny, 3).astype(np.float64)
    u = (raw - intercept) / slope                      # broadcast over last axis
    u = np.transpose(u, (3, 0, 2, 1))                  # -> (3, nt, ny, nz)
    y = np.arange(ny) * dy - (ny - 1) * dy / 2.0
    z = z_bottom + np.arange(nz) * dz
    return BTSWind(u=u, y=y, z=z, dt=dt, u_hub=u_hub, z_hub=z_hub)


def hub_series(w: BTSWind) -> np.ndarray:
    """Along-wind speed at (y=0, z=z_hub) by bilinear interpolation, shape (nt,)."""
    iy = int(np.clip(np.searchsorted(w.y, 0.0) - 1, 0, len(w.y) - 2))
    iz = int(np.clip(np.searchsorted(w.z, w.z_hub) - 1, 0, len(w.z) - 2))
    fy = (0.0 - w.y[iy]) / (w.y[iy + 1] - w.y[iy])
    fz = (w.z_hub - w.z[iz]) / (w.z[iz + 1] - w.z[iz])
    u = w.u[0]
    return ((1 - fy) * (1 - fz) * u[:, iy, iz] + fy * (1 - fz) * u[:, iy + 1, iz]
            + (1 - fy) * fz * u[:, iy, iz + 1] + fy * fz * u[:, iy + 1, iz + 1])


def rotor_average_series(w: BTSWind, radius: float) -> np.ndarray:
    """Mean along-wind speed over grid points inside the rotor disk, shape (nt,).
    Used as the effective wind for the 1-DOF toy model (a single hub point is too spiky)."""
    yy, zz = np.meshgrid(w.y, w.z - w.z_hub, indexing="ij")
    mask = (yy ** 2 + zz ** 2) <= radius ** 2
    if mask.sum() == 0:
        return hub_series(w)
    return w.u[0][:, mask].mean(axis=1)


def sample_at(series: np.ndarray, dt_series: float, t: float) -> float:
    """Linear interpolation of a time series at time t (clamped to the series range)."""
    x = t / dt_series
    i = int(np.floor(x))
    if i >= len(series) - 1:
        return float(series[-1])
    if i < 0:
        return float(series[0])
    f = x - i
    return float((1 - f) * series[i] + f * series[i + 1])
