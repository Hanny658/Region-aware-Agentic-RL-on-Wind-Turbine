"""Integrity check of the data written around a disk-full or VM-restart event (2026-09-23).

A full disk leaves truncated files that the pipelines then skip-or-crash on: 0-byte TurbSim fields (gen_wind skips
existing names), unreadable baseline .npz (the denominators), and half-written evaluation JSONs. This checks every
artefact that later stages read, reports what is broken, and with --delete removes the broken files so the normal
"skip if it exists" logic regenerates them.

    ~/wtrl/run.sh python scripts/dev/integrity_check.py [--since "2026-09-23 06:00"] [--delete]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--homes", nargs="+", default=["~/wtrl", "~/wtrl600", "~/wtrl_rt", "~/wtrl_rt_range", "~/wtrl_mpc_range"])
ap.add_argument("--winds", nargs="+", default=["~/wtrl/wind", "~/wtrl/wind600"])
ap.add_argument("--exp", default="~/wtrl/exp")
ap.add_argument("--since", default=None, help="only report files modified after this time (YYYY-MM-DD HH:MM)")
ap.add_argument("--delete", action="store_true", help="delete the broken files so they are regenerated")
a = ap.parse_args()
since = dt.datetime.strptime(a.since, "%Y-%m-%d %H:%M").timestamp() if a.since else 0.0
broken: list[tuple[str, str]] = []


def recent(p):
    return os.path.getmtime(p) >= since


# ---- TurbSim fields: all fields of one mean wind / turbulence class have the same size
for w in a.winds:
    d = os.path.expanduser(w)
    if not os.path.isdir(d):
        continue
    by_kind: dict[str, list[str]] = {}
    for p in glob.glob(f"{d}/*.bts"):
        by_kind.setdefault(os.path.basename(p).rsplit("_S", 1)[0], []).append(p)
    for kind, ps in by_kind.items():
        sizes = [os.path.getsize(p) for p in ps]
        ref = max(set(sizes), key=sizes.count)
        for p, s in zip(ps, sizes):
            if s != ref and recent(p):
                broken.append((p, f"wind field {s} bytes, expected {ref}"))

# ---- baselines: every .npz must load and carry the channels fitness() reads
for h in a.homes:
    d = os.path.expanduser(f"{h}/baselines/openfast")
    if not os.path.isdir(d):
        continue
    for p in sorted(glob.glob(f"{d}/*.npz")):
        if not recent(p):
            continue
        try:
            with np.load(p) as z:
                miss = [k for k in ("P", "gen_speed", "beta_meas", "v_hub", "region") if k not in z.files]
                if miss:
                    broken.append((p, f"baseline missing channels {miss}"))
                elif len(z["P"]) < 100:
                    broken.append((p, f"baseline has only {len(z['P'])} samples"))
        except Exception as e:  # noqa: BLE001 - truncated zip, empty file, bad magic
            broken.append((p, f"baseline unreadable ({type(e).__name__})"))

# ---- evaluation JSONs and summaries: must parse and carry the objective
for p in sorted(glob.glob(os.path.expanduser(f"{a.exp}/*/eval_*.json")) + glob.glob(os.path.expanduser(f"{a.exp}/*/summary.json"))
                + glob.glob(os.path.expanduser(f"{a.exp}/*/*/eval_*.json"))):
    if not recent(p):
        continue
    try:
        d = json.load(open(p))
        if os.path.basename(p).startswith("eval_") and "J" not in d and "per_episode" not in d:
            broken.append((p, "evaluation JSON without J / per_episode"))
    except Exception as e:  # noqa: BLE001
        broken.append((p, f"JSON unreadable ({type(e).__name__})"))

# ---- torch checkpoints written in the window
for p in sorted(glob.glob(os.path.expanduser(f"{a.exp}/*/ckpt_*.pt"))):
    if not recent(p):
        continue
    if os.path.getsize(p) < 1000:
        broken.append((p, f"checkpoint {os.path.getsize(p)} bytes"))

if not broken:
    print("no broken artefacts found" + (f" (modified after {a.since})" if a.since else ""))
else:
    print(f"{len(broken)} broken artefact(s)" + (f" modified after {a.since}" if a.since else "") + ":")
    for p, why in broken:
        print(f"  {why:<44} {p}")
    if a.delete:
        for p, _ in broken:
            os.remove(p)
        print(f"deleted {len(broken)} file(s); rerun the campaign to regenerate them")
    else:
        print("pass --delete to remove them so the pipelines regenerate them")
