"""Tune the ROSCO pitch controller under the same objective, budget and selection rule as every other controller
(must-do 4, 2026-09-17).

Every percentage in this repository is a reduction relative to the GSPI baseline, so a reviewer's first question
is whether that baseline was tuned. This script searches the baseline's own pitch loop:

    kp_scale, ki_scale   factors on ROSCO's gain-scheduled PC_GS_KP / PC_GS_KI tables (equivalent to moving the
                         loop's natural frequency and damping; KP ~ 2 zeta omega, KI ~ omega^2)
    lpf_corner           corner of the generator-speed low-pass filter F_LPFCornerFreq [rad/s]
    fa_ki                gain of ROSCO's fore-aft tower damper (0 = off; the damper's high-pass corner and
                         saturation are the values of the tower-damper template)

Each candidate is a copy of the OpenFAST case template with an edited DISCON.IN, evaluated deterministically on
the supervisor winds (TurbSim seeds 1-2) under J against the ORIGINAL GSPI baselines, so J > 0 means "better
than the untuned ROSCO". Proposer: random search for the first half of the budget, then (1+lambda) local search
around the incumbent. Candidates are cached by parameter hash; the search is resumable.

    WTRL_HOME=~/wtrl600 WTRL_WIND=~/wtrl/wind600 ~/wtrl/run.sh python scripts/rosco_tune.py --budget 40 --episode_s 600
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from scripts.mpc_param_search import summarise  # noqa: E402

BOX = {"kp_scale": (0.4, 3.0), "ki_scale": (0.4, 3.0), "lpf_corner": (0.6, 4.0), "fa_ki": (0.0, 0.004)}
START = {"kp_scale": 1.0, "ki_scale": 1.0, "lpf_corner": 1.5708, "fa_ki": 0.0}


def clip(p):
    out = {}
    for k, (lo, hi) in BOX.items():
        out[k] = float(f"{float(np.clip(p.get(k, START[k]), lo, hi)):.3g}")
    if out["fa_ki"] < 1e-4:
        out["fa_ki"] = 0.0
    return out


def key_of(p):
    return hashlib.sha1(json.dumps(clip(p), sort_keys=True).encode()).hexdigest()[:12]


def scale_row(line, f):
    nums = line.split()
    return "   ".join(f"{float(x) * f:.4e}" for x in nums) + "\n"


def make_template(src: Path, dst: Path, p: dict):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    lines = (dst / "DISCON.IN").read_text().splitlines(keepends=True)
    out = []
    for i, ln in enumerate(lines):
        if "! PC_GS_n" in ln:
            # the three rows after PC_GS_n are PC_GS_angles, PC_GS_KP, PC_GS_KI
            pass
        out.append(ln)
    idx = next(i for i, ln in enumerate(out) if "PC_GS_n" in ln and "!" in ln)
    # rows: idx+1 angles, idx+2 KP, idx+3 KI  (the numeric rows carry their label after "!")
    for off, f in ((2, p["kp_scale"]), (3, p["ki_scale"])):
        row = out[idx + off]
        body, _, tail = row.partition("!")
        out[idx + off] = "   ".join(f"{float(x) * f:.4e}" for x in body.split()) + "   !" + tail
    def setp(name, value):
        for i, ln in enumerate(out):
            if re.search(rf"!\s*{re.escape(name)}\b", ln):
                out[i] = f"{value:<20}! " + ln.split("!", 1)[1].lstrip()
                return
        raise KeyError(name)
    setp("F_LPFCornerFreq", f"{p['lpf_corner']:.5f}")
    if p["fa_ki"] > 0:
        setp("TD_Mode", "1"); setp("FA_KI", f"{p['fa_ki']:.5f}"); setp("FA_HPFCornerFreq", "0.172"); setp("FA_IntSat", "0.0873")
    (dst / "DISCON.IN").write_text("".join(out))


class Evaluator:
    def __init__(self, a):
        self.a = a
        self.root = Path(os.path.expanduser(a.root))
        (self.root / "cache").mkdir(parents=True, exist_ok=True)
        self.src = Path(os.path.expanduser(a.template))

    def run(self, p, slot, seeds=None, tag_extra=""):
        seeds = seeds or self.a.seeds
        p = clip(p)
        out = self.root / "cache" / f"eval_{key_of(p)}_s{''.join(map(str, seeds))}{tag_extra}.json"
        if out.exists():
            return json.load(open(out))
        tpl = self.root / "templates" / key_of(p)
        tpl.parent.mkdir(parents=True, exist_ok=True)
        make_template(self.src, tpl, p)
        env = {**os.environ, "WTRL_TEMPLATE": str(tpl)}
        cmd = [sys.executable, str(PROJ / "scripts" / "evaluate.py"), "--run", os.path.expanduser(self.a.config_run), "--gspi",
               "--backend", "openfast", "--seeds", *map(str, seeds), "--episode_s", str(self.a.episode_s), "--workers", str(self.a.workers),
               "--port0", str(self.a.port0 + 300 * slot), "--tag", out.stem[len("eval_"):], "--out", str(self.root / "cache")]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        shutil.rmtree(tpl, ignore_errors=True)
        if not out.exists():
            fail = {"J": -100.0, "J_power_mse_red_pct": -100.0, "J_gen_speed_mse_red_pct": -100.0, "J_TwrBsMyt_DEL_red_pct": -100.0,
                    "J_RootMyc1_DEL_red_pct": -100.0, "energy_loss_pct": float("nan"), "per_episode": [],
                    "failed": "evaluation did not complete", "log_tail": (r.stdout[-1200:] + r.stderr[-1200:])}
            json.dump(fail, open(out, "w"), indent=1)
        return json.load(open(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--episode_s", type=float, default=600.0)
    ap.add_argument("--root", default="~/wtrl/exp/roscotune600")
    ap.add_argument("--template", default="~/wtrl/runs/template_5mw")
    ap.add_argument("--config_run", default="~/wtrl/exp/jg3t_s0", help="any GSPI-base J run (its config.json)")
    ap.add_argument("--port0", type=int, default=7600)
    ap.add_argument("--heldout", action="store_true", help="evaluate the best candidate and the start on both held-out sets")
    ap.add_argument("--emit_template", default=None, metavar="DST",
                    help="write the case template of the best candidate (best.json under --root) to DST and exit")
    a = ap.parse_args()
    if a.emit_template:
        best = json.load(open(Path(os.path.expanduser(a.root)) / "best.json"))
        make_template(Path(os.path.expanduser(a.template)), Path(os.path.expanduser(a.emit_template)), clip(best["params"]))
        print(f"[rosco] template of {best['params']} (supervisor J {best['result']['J']:.2f}) -> {a.emit_template}")
        return
    ev = Evaluator(a)
    root = Path(os.path.expanduser(a.root))
    hp = root / "history.jsonl"
    hist = [json.loads(l) for l in open(hp)] if hp.exists() else []
    rng = np.random.default_rng([7, len(hist)])
    if not hist:
        r = summarise(ev.run(START, 0))
        hist.append({"round": 0, "params": clip(START), "result": r})
        open(hp, "a").write(json.dumps(hist[-1]) + "\n")
        print(f"[rosco] start (untuned GSPI against its own baseline) J {r['J']:.2f}", flush=True)
    seen = {key_of(h["params"]) for h in hist}
    n_done, rnd = len(hist) - 1, max(h["round"] for h in hist)
    while n_done < a.budget:
        rnd += 1
        n = min(a.batch, a.budget - n_done)
        cands = []
        for _ in range(200):
            if len(cands) >= n:
                break
            if n_done + len(cands) < a.budget // 2:
                p = {k: math.exp(rng.uniform(math.log(max(lo, 1e-4)), math.log(hi))) for k, (lo, hi) in BOX.items()}
                if rng.random() < 0.5:
                    p["fa_ki"] = 0.0
            else:
                inc = max(hist, key=lambda h: h["result"]["J"])["params"]
                p = {k: (inc[k] if inc[k] > 0 else 5e-4 * (rng.random() < 0.3)) * math.exp(rng.normal(0, 0.3)) for k in BOX}
            p = clip(p)
            if key_of(p) not in seen and key_of(p) not in {key_of(c) for c in cands}:
                cands.append(p)
        with ThreadPoolExecutor(max_workers=a.parallel) as pool:
            res = list(pool.map(lambda ic: ev.run(ic[1], ic[0] % a.parallel), enumerate(cands)))
        for p, fit in zip(cands, res):
            rec = {"round": rnd, "params": p, "result": summarise(fit)}
            hist.append(rec); seen.add(key_of(p)); n_done += 1
            open(hp, "a").write(json.dumps(rec) + "\n")
        best = max(hist, key=lambda h: h["result"]["J"])
        print(f"[rosco] round {rnd}: {n_done}/{a.budget}; batch J {[round(h['result']['J'], 2) for h in hist[-len(cands):]]}; "
              f"best J {best['result']['J']:.2f} {best['params']}", flush=True)
    best = max(hist, key=lambda h: h["result"]["J"])
    json.dump(best, open(root / "best.json", "w"), indent=1)
    print(f"[rosco] done: best J {best['result']['J']:.2f} {best['params']}", flush=True)
    if a.heldout:
        for seeds in ([3, 4, 5, 6], [7, 8, 9, 10]):
            for lab, p in (("tuned", best["params"]), ("start", START)):
                r = summarise(ev.run(p, 0, seeds=seeds))
                trav = ""
                print(f"[rosco] held-out seeds {seeds} {lab}: J {r['J']:.2f} ({r['power']} / {r['speed']} / {r['tower']} / {r['blade']}){trav}", flush=True)


if __name__ == "__main__":
    main()
