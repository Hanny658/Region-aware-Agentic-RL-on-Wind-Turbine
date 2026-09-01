"""Reader for OpenFAST binary output (.outb) -> dict of channel arrays.
FileID 1 = with time, 2 = without time, 4 = without time + channel-name length (OpenFAST >= 3.x default).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_outb(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    with open(path, "rb") as f:
        (file_id,) = struct.unpack("<h", f.read(2))
        len_name = struct.unpack("<h", f.read(2))[0] if file_id == 4 else 10
        n_ch, n_t = struct.unpack("<2i", f.read(8))
        if file_id == 1:
            time_scl, time_off = struct.unpack("<2d", f.read(16))
        elif file_id in (2, 4):
            time_out, time_incr = struct.unpack("<2d", f.read(16))
        else:
            raise ValueError(f"unsupported .outb FileID {file_id}")
        col_scl = np.frombuffer(f.read(4 * n_ch), dtype="<f4").astype(np.float64)
        col_off = np.frombuffer(f.read(4 * n_ch), dtype="<f4").astype(np.float64)
        (len_desc,) = struct.unpack("<i", f.read(4))
        f.read(len_desc)
        names = [f.read(len_name).decode("latin-1").strip() for _ in range(n_ch + 1)]
        units = [f.read(len_name).decode("latin-1").strip() for _ in range(n_ch + 1)]
        if file_id == 1:
            traw = np.frombuffer(f.read(4 * n_t), dtype="<i4").astype(np.float64)
            time = (traw - time_off) / time_scl
        else:
            time = time_out + time_incr * np.arange(n_t)
        raw = np.frombuffer(f.read(2 * n_t * n_ch), dtype="<i2").reshape(n_t, n_ch).astype(np.float64)
    data = (raw - col_off) / col_scl
    out = {"Time": np.asarray(time, dtype=float)}
    for i, nm in enumerate(names[1:]):
        out[nm] = data[:, i]
    return out, dict(zip(names, units))
