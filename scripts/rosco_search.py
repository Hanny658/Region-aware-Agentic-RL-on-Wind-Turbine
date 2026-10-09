"""Who should propose a classical controller's parameters? (2026-10-09)

Roadmap s29 tuned the shipped ROSCO pitch loop with a fixed recipe, half random draws and half a local search, and
the margin it found turned out to be half of everything the later rungs of the paper add. That makes the proposer
itself worth measuring: under a fixed simulation budget, starting from the shipped settings, how much of the
available margin does a language model find, and how much does random search or Bayesian optimisation find?

This runs the same search under four proposers, on the same box, from the same start, with the same budget and the
same evaluation, so the only thing that differs is who picks the next point.

    random   log-uniform draws from the box, with a coin flip for the damper-off corner
    local    one random point, then a (1+lambda) local search around the incumbent
    bayes    Gaussian-process surrogate with expected improvement, maximised over a dense random candidate set
    llm      a reasoning model, shown the box, the objective and the full history with its four terms

The search objective is J on the design winds against the ORIGINAL shipped-GSPI baselines, so J > 0 means better
than the shipped controller and the numbers are directly comparable with every other table in the repository. Each
arm is run with several repeats, because s29's random half found exactly one good point in twenty draws and one run
cannot tell that apart from skill.

    ~/wtrl/run.sh python scripts/rosco_search.py --proposer llm --repeat 0 --budget 20 \\
        --means 12 14 16 18 20 22 24 --seeds 1 2 --episode_s 600
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from scripts.rosco_tune import BOX, START, Evaluator, clip, key_of, make_template  # noqa: E402
from scripts.mpc_param_search import summarise  # noqa: E402

KEYS = list(BOX)
# the damper has a meaningful zero, so it is searched on a shifted log axis rather than a plain one
SHIFT = {"fa_ki": 2.0e-4}


def to_x(p):
    return np.array([math.log(p[k] + SHIFT.get(k, 0.0)) for k in KEYS])


def from_x(x):
    return clip({k: math.exp(v) - SHIFT.get(k, 0.0) for k, v in zip(KEYS, x)})


def box_x():
    lo = np.array([math.log(max(BOX[k][0], 1e-9) + SHIFT.get(k, 0.0)) for k in KEYS])
    hi = np.array([math.log(BOX[k][1] + SHIFT.get(k, 0.0)) for k in KEYS])
    return lo, hi


# ----------------------------------------------------------------------------- proposers
def propose_random(hist, n, rng, **_):
    out = []
    lo, hi = box_x()
    while len(out) < n:
        p = from_x(rng.uniform(lo, hi))
        if rng.random() < 0.5:
            p["fa_ki"] = 0.0
        out.append(clip(p))
    return out, "log-uniform draw from the box"


def propose_local(hist, n, rng, **_):
    if not hist:
        return propose_random(hist, n, rng)
    inc = max(hist, key=lambda h: h["result"]["J"])["params"]
    out = []
    while len(out) < n:
        p = {k: (inc[k] if inc[k] > 0 else 5e-4 * (rng.random() < 0.3)) * math.exp(rng.normal(0, 0.3)) for k in KEYS}
        out.append(clip(p))
    return out, "Gaussian step around the incumbent"


def _gp_posterior(X, y, Xs, length=0.9, noise=1.0):
    """Zero-mean GP with an RBF kernel; returns the posterior mean and standard deviation at Xs."""
    def k(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / length ** 2)
    ym = y.mean()
    K = k(X, X) + noise * np.eye(len(X))
    L = np.linalg.cholesky(K)
    al = np.linalg.solve(L.T, np.linalg.solve(L, y - ym))
    Ks = k(X, Xs)
    mu = ym + Ks.T @ al
    v = np.linalg.solve(L, Ks)
    var = np.clip(1.0 - (v ** 2).sum(0), 1e-9, None)
    return mu, np.sqrt(var)


def propose_bayes(hist, n, rng, **_):
    if len(hist) < 4:
        return propose_random(hist, n, rng)
    X = np.array([to_x(h["params"]) for h in hist])
    y = np.array([h["result"]["J"] for h in hist], float)
    y = np.clip(y, -50, None)                       # a failed candidate must not dominate the length scale
    s = y.std() or 1.0
    yn = (y - y.mean()) / s
    lo, hi = box_x()
    cand = rng.uniform(lo, hi, size=(4000, len(KEYS)))
    mu, sd = _gp_posterior(X, yn, cand, noise=0.1)
    best = yn.max()
    z = (mu - best - 0.01) / sd
    from math import erf, sqrt
    Phi = np.array([0.5 * (1 + erf(v / sqrt(2))) for v in z])
    phi = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    ei = (mu - best - 0.01) * Phi + sd * phi
    out, taken = [], []
    for i in np.argsort(-ei):
        p = from_x(cand[i])
        if key_of(p) in {key_of(q) for q in out}:
            continue
        if taken and min(np.linalg.norm(cand[i] - cand[j]) for j in taken) < 0.25:
            continue                                # keep a batch from collapsing onto one point
        out.append(p); taken.append(i)
        if len(out) >= n:
            break
    return out, "expected improvement under a GP surrogate"


SYSTEM = """You propose parameter sets for the pitch controller of a 5 MW wind turbine, to be tested in aeroelastic
simulation. You are given a box of four parameters, the objective, and every parameter set tried so far with its
measured score. Propose the next batch. You are being compared against random search and Bayesian optimisation on
an equal simulation budget, so spend each candidate deliberately: early on, resolve which parameter the objective
is actually sensitive to; later, exploit it. Answer with JSON only:
{"candidates": [{"kp_scale": x, "ki_scale": x, "lpf_corner": x, "fa_ki": x, "why": "one short sentence"}, ...]}"""


def propose_llm(hist, n, rng, client=None, objective="", **_):
    rows = []
    for h in sorted(hist, key=lambda r: -r["result"]["J"])[:40]:
        p, r = h["params"], h["result"]
        rows.append(f"  kp {p['kp_scale']:.3g}  ki {p['ki_scale']:.3g}  corner {p['lpf_corner']:.3g}  damper {p['fa_ki']:.4g}"
                    f"   ->  J {r['J']:+7.2f}   terms {r.get('J_power_mse_red_pct', float('nan')):+6.1f}"
                    f" {r.get('J_gen_speed_mse_red_pct', float('nan')):+6.1f}"
                    f" {r.get('J_TwrBsMyt_DEL_red_pct', float('nan')):+6.1f}"
                    f" {r.get('J_RootMyc1_DEL_red_pct', float('nan')):+6.1f}")
    box = "\n".join(f"  {k}: {v[0]} to {v[1]}" for k, v in BOX.items())
    user = (f"{objective}\n\nBox (a candidate outside it is clipped):\n{box}\n\n"
            f"Shipped controller: kp 1.0, ki 1.0, corner 1.5708 rad/s, damper 0 (its J is 0 by definition).\n\n"
            f"Tried so far, best first ({len(hist)} of a budget that is nearly spent if this list is long):\n"
            + ("\n".join(rows) if rows else "  (nothing yet)")
            + f"\n\nTerms are power MSE, generator-speed MSE, tower-base fatigue, blade-root fatigue, each a percentage "
              f"reduction against the shipped controller. Propose exactly {n} candidates.")
    out = client.ask_json(SYSTEM, user, tag=f"rosco_search_{len(hist):02d}")
    cands, why = [], []
    for c in (out.get("candidates") or [])[:n]:
        try:
            cands.append(clip({k: float(c[k]) for k in KEYS}))
            why.append(str(c.get("why", ""))[:160])
        except (KeyError, TypeError, ValueError):
            continue
    while len(cands) < n:                            # a malformed reply must not shrink the budget
        extra, _ = propose_random(hist, 1, rng)
        cands += extra; why.append("(filled randomly: the model returned fewer candidates than asked)")
    return cands, " | ".join(why)


PROPOSERS = {"random": propose_random, "local": propose_local, "bayes": propose_bayes, "llm": propose_llm}

OBJECTIVE = """Objective J: the mean of four percentage reductions against the shipped controller, measured on the
same wind fields - power mean-square error and generator-speed mean-square error above rated, tower-base fore-aft
damage-equivalent load, blade-root out-of-plane damage-equivalent load - minus 20 per percent of energy lost beyond
1 percent. J = 0 is the shipped controller. Higher is better. The parameters are multipliers on ROSCO's
gain-scheduled proportional and integral pitch gains, the corner frequency of the generator-speed low-pass filter in
rad/s (shipped 1.5708; the blade-passing frequency at rated speed is about 3.8), and the gain of the fore-aft tower
damper (0 turns it off)."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposer", choices=list(PROPOSERS), required=True)
    ap.add_argument("--repeat", type=int, default=0, help="independent run of the same proposer")
    ap.add_argument("--budget", type=int, default=20, help="candidates after the shipped start, which is free")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--means", nargs="+", type=float, default=None)
    ap.add_argument("--ti", default=None)
    ap.add_argument("--episode_s", type=float, default=600.0)
    ap.add_argument("--root", default="~/wtrl/exp/roscosearch")
    ap.add_argument("--template", default="~/wtrl/runs/template_5mw")
    ap.add_argument("--config_run", default="~/wtrl/exp/jg3t_s0")
    ap.add_argument("--port0", type=int, default=7800)
    a = ap.parse_args()

    a.root = os.path.expanduser(f"{a.root}/{a.proposer}_r{a.repeat}")
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    ev = Evaluator(a)
    if a.means:                                      # let the evaluator see a protocol the config_run does not carry
        ev.extra_args = ["--means", *[f"{m:g}" for m in a.means]] + (["--ti", a.ti] if a.ti else [])
    hp = root / "history.jsonl"
    hist = [json.loads(line) for line in open(hp)] if hp.exists() else []
    rng = np.random.default_rng([911, a.repeat, hash(a.proposer) % 10_000])
    for _ in range(len(hist)):
        rng.random()                                 # keep a resumed run on the stream it would have had

    client = None
    if a.proposer == "llm":
        from llm.client import LLMClient
        client = LLMClient(transcript_path=root / "llm_transcript.jsonl")

    if not hist:
        r = summarise(ev.run(START, 0))
        hist.append({"i": 0, "params": clip(START), "result": r, "why": "the shipped controller, free of budget"})
        open(hp, "a").write(json.dumps(hist[-1]) + "\n")
        print(f"[{a.proposer}/r{a.repeat}] shipped start J {r['J']:+.2f} (identity check: should be near 0)", flush=True)

    seen = {key_of(h["params"]) for h in hist}
    while len(hist) - 1 < a.budget:
        n = min(a.batch, a.budget - (len(hist) - 1))
        cands, why = PROPOSERS[a.proposer](hist, n, rng, client=client, objective=OBJECTIVE)
        cands = [c for c in cands if key_of(c) not in seen] or propose_random(hist, n, rng)[0]
        with ThreadPoolExecutor(max_workers=a.parallel) as pool:
            res = list(pool.map(lambda ic: ev.run(ic[1], ic[0] % a.parallel), enumerate(cands)))
        for p, fit in zip(cands, res):
            rec = {"i": len(hist), "params": p, "result": summarise(fit), "why": why}
            hist.append(rec); seen.add(key_of(p))
            open(hp, "a").write(json.dumps(rec) + "\n")
            b = max(h["result"]["J"] for h in hist)
            print(f"[{a.proposer}/r{a.repeat}] {len(hist) - 1:3d}/{a.budget}  "
                  f"kp {p['kp_scale']:.3g} ki {p['ki_scale']:.3g} corner {p['lpf_corner']:.3g} damper {p['fa_ki']:.4g}"
                  f"  ->  J {rec['result']['J']:+7.2f}   best so far {b:+7.2f}", flush=True)

    best = max(hist, key=lambda h: h["result"]["J"])
    json.dump(best, open(root / "best.json", "w"), indent=1)
    print(f"[{a.proposer}/r{a.repeat}] done. best J {best['result']['J']:+.2f} at {best['params']}", flush=True)


if __name__ == "__main__":
    main()
