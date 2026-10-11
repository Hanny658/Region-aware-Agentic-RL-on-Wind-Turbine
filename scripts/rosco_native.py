"""The ROSCO pitch loop as ROSCO exposes it, plus the box limits the plant sets (2026-10-11).

The search of 2026-10-09 multiplied ROSCO's gain-schedule tables `PC_GS_KP` and `PC_GS_KI` by two scalars. A
ROSCO user does not set those tables: the toolbox generates them from two quantities in the tuning file,
`omega_pc` (the pitch loop's target natural frequency) and `zeta_pc` (its target damping ratio), through

    Kp(beta) = (2 zeta omega + A(beta)) / B(beta)
    Ki(beta) =  omega**2 / B(beta)                      (rosco/toolbox/controller.py::second_order_PI)

with A, B the linearised plant coefficients at that operating point. Scaling Ki by b is exactly a change of
omega to omega*sqrt(b), but scaling Kp by a demands 2 zeta' omega' = a 2 zeta omega + (a-1) A(beta), and A
varies over the schedule, so the implied damping ratio is not constant -- the scaled tables are unreachable by
any ROSCO tuning (6.8 to 12.8 % of spread for the designs that search produced). This module searches the
native quantities instead, and recovers A and B exactly from the shipped tables, which are known to come from
zeta_pc 0.7 and omega_pc 0.6 rad/s (ROSCO/Examples/Tune_Cases/NREL5MW.yaml). The recovery is self-checking:
that tuning has to return the shipped tables bit for bit, which `self_test()` asserts.

Every box limit is a controller setting or a measurement on the shipped controller's own baselines; none is
chosen. `scripts/dev/search_box_bounds.py` reproduces all of them, and
`docs/design_2026-10-11_rosco-native-search.md` gives the reasoning per limit.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

TEMPLATE = os.path.expanduser(os.environ.get("WTRL_ROSCO_TEMPLATE", "~/wtrl/runs/template_5mw"))
ZETA_SHIPPED, OMEGA_SHIPPED = 0.7, 0.6          # ROSCO/Examples/Tune_Cases/NREL5MW.yaml

# ----------------------------------------------------------------------------- the searched space
# limit                                        why this limit (measured values from scripts/dev/search_box_bounds.py)
#   omega_pc  [rad/s]  0.10 .. 2.04   upper: the measured first tower fore-aft mode (0.3241 Hz); a collective-pitch
#                                     loop whose bandwidth reaches it drives thrust straight into it.
#                                     lower: a 10 s loop time constant, below which the loop is not regulating.
#   zeta_pc   [-]      0.10 .. 2.00   nearly undamped to heavily overdamped; shipped 0.7.
#   lpf_corner[rad/s]  0.20 .. 19.2   upper: above 19.2 rad/s a first-order filter passes 99.99 % of the measured
#                                     generator-speed signal power and is indistinguishable from no filter.
#   fa_ki  [rad s/m]   0.00 .. 0.20   ROSCO's fore-aft tower damper gain, 0 = off (its default). At 0.20 the
#                                     unsaturated demand reaches 12.3 deg on a median above-rated episode,
#                                     comparable to the whole above-rated pitch range. The previous ceiling of
#                                     0.004 allowed the damper 0.25 deg, i.e. nothing.
#   fa_hpf    [rad/s]  0.05 .. 1.20   the damper must reject quasi-static thrust and pass the tower mode: at 1.20
#                                     the mode still passes within 31 deg of phase. Only acts when fa_ki > 0.
BOX = {"omega_pc": (0.10, 2.04), "zeta_pc": (0.10, 2.00), "lpf_corner": (0.20, 19.2),
       "fa_ki": (0.0, 0.20), "fa_hpf": (0.05, 1.20)}
START = {"omega_pc": OMEGA_SHIPPED, "zeta_pc": ZETA_SHIPPED, "lpf_corner": 1.5708, "fa_ki": 0.0, "fa_hpf": 0.172}
KEYS = list(BOX)

# A safety limiter rather than a tuning knob: the authority conceded to the damper is set by fa_ki, and 10 deg is
# generous enough not to bind. The table script reports how often it does.
FA_INT_SAT = 0.175

# Quantities that span decades and are positive-definite are searched on a log axis; the two of order one are
# searched linearly. The damper has a meaningful zero, so its log axis is shifted -- the shift is set so that the
# bottom of its axis is a gain worth under a tenth of a degree of pitch, i.e. "off".
LOG_KEYS = ("omega_pc", "lpf_corner", "fa_ki")
SHIFT = {"fa_ki": 1.0e-3}


# ----------------------------------------------------------------------------- ROSCO's own gain schedule
def _disc_rows(path: str) -> tuple[np.ndarray, np.ndarray]:
    txt = open(path).read().splitlines()

    def row(name: str) -> np.ndarray:
        ln = next(l for l in txt if name in l)
        return np.array([float(x) for x in re.split(r"\s+", ln.split("!")[0].strip())])

    return row("PC_GS_KP"), row("PC_GS_KI")


_KP0, _KI0 = _disc_rows(f"{TEMPLATE}/DISCON.IN")
B_PLANT = OMEGA_SHIPPED ** 2 / _KI0                                   # d(speed rate)/d(pitch)
A_PLANT = _KP0 * B_PLANT - 2 * ZETA_SHIPPED * OMEGA_SHIPPED           # d(speed rate)/d(speed)


def gains(zeta: float, omega: float) -> tuple[np.ndarray, np.ndarray]:
    """The gain-schedule tables ROSCO's toolbox would write for this tuning."""
    return (2.0 * zeta * omega + A_PLANT) / B_PLANT, omega ** 2 / B_PLANT


def self_test() -> None:
    kp, ki = gains(ZETA_SHIPPED, OMEGA_SHIPPED)
    e = max(float(np.abs(kp - _KP0).max()), float(np.abs(ki - _KI0).max()))
    assert e < 1e-12, f"the shipped tuning does not reproduce the shipped tables (error {e:.3e})"


self_test()


# ----------------------------------------------------------------------------- parameters
def canon(p: dict) -> dict:
    """Clip into the box and round to six significant figures.

    Six, not three: the previous three-significant-figure rounding wrote the shipped corner 1.5708 as 1.57,
    which is why the identity check used to read -0.15 instead of 0. Six digits is finer than the DISCON file's
    own format, so two parameter sets with the same canonical form produce the same input files."""
    out = {}
    for k, (lo, hi) in BOX.items():
        out[k] = float(f"{float(np.clip(p.get(k, START[k]), lo, hi)):.6g}")
    if out["fa_ki"] < 5.0e-6:
        # DISCON.IN carries FA_KI to five decimals, so anything below this writes as 0.00000: the damper is off,
        # and its filter corner is then not written either, which keeps one cache key per distinct controller
        out["fa_ki"] = 0.0
        out["fa_hpf"] = START["fa_hpf"]
    return out


def key_of(p: dict) -> str:
    return hashlib.sha1(json.dumps(canon(p), sort_keys=True).encode()).hexdigest()[:12]


def to_x(p: dict) -> np.ndarray:
    """The search coordinates: log for the three positive-definite spans, linear for the other two."""
    return np.array([np.log(p[k] + SHIFT.get(k, 0.0)) if k in LOG_KEYS else p[k] for k in KEYS])


def from_x(x: np.ndarray) -> dict:
    return canon({k: (float(np.exp(v) - SHIFT.get(k, 0.0)) if k in LOG_KEYS else float(v))
                  for k, v in zip(KEYS, x)})


def box_x() -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([np.log(BOX[k][0] + SHIFT.get(k, 0.0)) if k in LOG_KEYS else BOX[k][0] for k in KEYS])
    hi = np.array([np.log(BOX[k][1] + SHIFT.get(k, 0.0)) if k in LOG_KEYS else BOX[k][1] for k in KEYS])
    return lo, hi


def fmt(p: dict) -> str:
    return (f"omega {p['omega_pc']:.3f} zeta {p['zeta_pc']:.3f} corner {p['lpf_corner']:.3f} "
            f"damper {p['fa_ki']:.4f}" + (f" hpf {p['fa_hpf']:.3f}" if p["fa_ki"] > 0 else " (off)"))


# ----------------------------------------------------------------------------- writing the case
def make_template(src: Path, dst: Path, p: dict) -> None:
    """A copy of the OpenFAST case with DISCON.IN rewritten for this tuning, in ROSCO's own number formats."""
    p = canon(p)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    out = (dst / "DISCON.IN").read_text().splitlines(keepends=True)

    kp, ki = gains(p["zeta_pc"], p["omega_pc"])
    idx = next(i for i, ln in enumerate(out) if "PC_GS_n" in ln and "!" in ln)
    for off, tab in ((2, kp), (3, ki)):                 # rows after PC_GS_n: angles, KP, KI
        body, _, tail = out[idx + off].partition("!")
        assert len(body.split()) == len(tab), "gain-schedule length does not match the template"
        out[idx + off] = "   ".join(f"{v:.4e}" for v in tab) + "   !" + tail

    def setp(name: str, value: str) -> None:
        for i, ln in enumerate(out):
            if re.search(rf"!\s*{re.escape(name)}\b", ln):
                out[i] = f"{value:<20}! " + ln.split("!", 1)[1].lstrip()
                return
        raise KeyError(name)

    setp("F_LPFCornerFreq", f"{p['lpf_corner']:.5f}")
    if p["fa_ki"] > 0:
        setp("TD_Mode", "1")
        setp("FA_KI", f"{p['fa_ki']:.5f}")
        setp("FA_HPFCornerFreq", f"{p['fa_hpf']:.5f}")
        setp("FA_IntSat", f"{FA_INT_SAT:.5f}")
    (dst / "DISCON.IN").write_text("".join(out))


# ----------------------------------------------------------------------------- evaluation
class Evaluator:
    """One candidate -> the paired reductions against the shipped-GSPI baselines on the design winds.

    `--relabel_wind` is passed explicitly. It must be: evaluate.py only enables the wind-labelled above-rated
    subset by itself when the borrowed config run carries the old composite objective, and without it the
    subset reverts to a per-controller pitch rule, so the two sides would be scored on different samples."""

    def __init__(self, a):
        self.a = a
        self.root = Path(os.path.expanduser(a.root))
        (self.root / "cache").mkdir(parents=True, exist_ok=True)
        self.src = Path(os.path.expanduser(a.template))
        self.extra_args: list[str] = []          # a wind protocol the config_run does not carry

    def run(self, p: dict, slot: int, seeds=None, tag_extra: str = "") -> dict:
        seeds = seeds or self.a.seeds
        p = canon(p)
        args = ["--relabel_wind", *self.extra_args]
        prot = "_" + hashlib.sha1(" ".join(args).encode()).hexdigest()[:6]
        out = self.root / "cache" / f"eval_{key_of(p)}_s{''.join(map(str, seeds))}{prot}{tag_extra}.json"
        if out.exists():
            return json.load(open(out))
        tpl = self.root / "templates" / key_of(p)
        tpl.parent.mkdir(parents=True, exist_ok=True)
        make_template(self.src, tpl, p)
        cmd = [sys.executable, str(PROJ / "scripts" / "evaluate.py"),
               "--run", os.path.expanduser(self.a.config_run), "--gspi", "--backend", "openfast",
               "--seeds", *map(str, seeds), "--episode_s", str(self.a.episode_s),
               "--workers", str(self.a.workers), "--port0", str(self.a.port0 + 300 * slot),
               "--tag", out.stem[len("eval_"):], "--out", str(self.root / "cache"), *args]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           env={**os.environ, "WTRL_TEMPLATE": str(tpl)})
        shutil.rmtree(tpl, ignore_errors=True)
        if not out.exists():
            log = r.stdout[-4000:] + r.stderr[-4000:]
            # Two failures look the same from here and must not: a candidate the plant cannot run, which is a
            # result, and a broken machine, which is not. OpenFAST writes no .outb when it aborts, so the
            # episode-level complaint from fitness() identifies the first kind.
            diverged = "episodes produced none" in log or "needs .outb metrics" in log
            json.dump({"per_episode": [],
                       "failed": ("at least one episode did not run to completion (the plant diverged)"
                                  if diverged else "evaluation did not complete (not a plant failure)"),
                       "failure_kind": "diverged" if diverged else "infrastructure",
                       "log_tail": log[-2500:]}, open(out, "w"), indent=1)
        return json.load(open(out))


if __name__ == "__main__":
    self_test()
    print(f"shipped tuning reproduces the shipped tables exactly (zeta {ZETA_SHIPPED}, omega {OMEGA_SHIPPED})")
    print(f"A = {A_PLANT.min():.5f} .. {A_PLANT.max():.5f}   B = {B_PLANT.min():.3f} .. {B_PLANT.max():.3f}")
    for name, p in (("shipped", START),
                    ("far corner", {**START, "omega_pc": 2.04, "zeta_pc": 2.0, "lpf_corner": 19.2,
                                    "fa_ki": 0.2, "fa_hpf": 1.2})):
        kp, ki = gains(p["zeta_pc"], p["omega_pc"])
        print(f"  {name:<11} {fmt(canon(p))}  ->  Kp {kp.max():+.4f}..{kp.min():+.4f}  Ki {ki.max():+.5f}..{ki.min():+.5f}")
