"""Who should propose a classical controller's parameters? (2026-10-09, redesigned 2026-10-11)

Roadmap s29 tuned the shipped ROSCO pitch loop with a fixed recipe, and the margin it found turned out to be
half of everything the later rungs of the paper add. That makes the proposer itself worth measuring: under a
fixed simulation budget, starting from the shipped settings, how much of the available margin does a language
model find, and how much does random search or Bayesian optimisation find?

Four proposers, one box, one start, one budget, one evaluation, so the only thing that differs is who picks
the next point:

    random   draws from the box, log axis for the three spans of decades, linear for the two of order one
    local    one random point, then a (1+lambda) local search around the incumbent
    bayes    constrained Bayesian optimisation: a GP on the objective and one per constraint, with the
             acquisition expected improvement times the probability of feasibility (Gardner et al. 2014)
    llm      a reasoning model, shown the five quantities with their units and physical meaning, the objective,
             the constraints and the whole history
    tune     the recipe of roadmap s29, half random draws then a local search -- used to re-derive the paper's
             tuned baseline in the native parameterisation, not as an arm of the comparison

The searched quantities are ROSCO's own (`scripts/rosco_native.py`: omega_pc, zeta_pc, F_LPFCornerFreq, FA_KI,
FA_HPFCornerFreq), every box limit is read off the plant, and selection is the constrained rule of
`eval/selection.py` -- one objective, three constraints at "no worse than shipped", no weights. Several
repeats per arm, because s29's random half found exactly one good point in twenty draws and one run cannot
tell that apart from skill.

    ~/wtrl/run.sh python scripts/rosco_search.py --proposer llm --repeat 0 --budget 24 \\
        --means 12 14 16 18 20 22 24 --ti B --seeds 1 2 --episode_s 600
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
from eval.selection import (LOAD_CONSTRAINTS, REPORTED, assess, per_wind_rank_key,  # noqa: E402
                            rank_key, worst_per_wind)
from scripts.rosco_native import (BOX, KEYS, START, Evaluator, box_x, canon, fmt,  # noqa: E402
                                  from_x, key_of, to_x)

SCORED_S = 580.0            # 600 s simulated, the first 20 s discarded
# None: the set-mean rule of s51 (constraint domination on the set mean). A number t: the margin screen of
# roadmap s62 - the same rule with the worst single-wind load term required >= -t on the design set, which
# every proposer selects under and which the llm proposer is told about. Set by main() from --screen_tol.
SCREEN_TOL = None


def select_key(rec):
    """The key the run selects by: feasible first, then the objective - on the set mean, or per wind."""
    return rank_key(rec) if SCREEN_TOL is None else per_wind_rank_key(rec, SCREEN_TOL)
RATE_LIMIT_DEG_S = 10.0     # the controller's own pitch rate limit, for the actuator duty cycle


# ----------------------------------------------------------------------------- the record of one candidate
def record(fit: dict) -> tuple[dict, dict]:
    """The selection block and the reported quantities of one evaluation."""
    a = assess(fit)
    pe = [p for p in fit.get("per_episode", []) if p.get("mean_wind") is not None]
    per_wind: dict[str, dict] = {}
    for p in pe:
        d = per_wind.setdefault(f"{p['mean_wind']:g}", {})
        for name, col in REPORTED:
            v = p.get(col)
            if v is not None and v == v:
                d.setdefault(name, []).append(float(v))
    rep = {"per_wind": {u: {n: round(float(np.mean(v)), 1) for n, v in d.items()} for u, d in per_wind.items()}}
    tr = np.array([p.get("pitch_travel_deg", np.nan) for p in pe], float)
    tb = np.array([p.get("pitch_travel_base_deg", np.nan) for p in pe], float)
    if np.isfinite(tr).all() and np.isfinite(tb).all() and tb.sum() > 0:
        rep["travel_ratio"] = round(float(tr.sum() / tb.sum()), 3)
        rep["adc"] = round(float(np.mean(tr) / SCORED_S / RATE_LIMIT_DEG_S), 4)
        rep["adc_base"] = round(float(np.mean(tb) / SCORED_S / RATE_LIMIT_DEG_S), 4)
    return a, rep


def line(tag: str, i: int, budget: int, p: dict, a: dict, best: dict | None, rec: dict | None = None) -> str:
    c = a.get("constraints", {})
    cons = "  ".join(f"{k} {c[k]['value']:+6.2f}" for k in ("tower", "blade") if k in c)
    en = f"energy {c['energy']['value']:+6.3f}" if "energy" in c else ""
    flag = "ok " if a.get("feasible") else f"x{a.get('shortfall', 0):.1f}".ljust(3)
    if rec is not None and c:
        w, at, which = worst_per_wind(rec)
        flag += f"  worst/wind {w:+5.1f} ({which} @ {at})"
    bs = "none yet"
    if best is not None and select_key(best)[0] == 1 and best["i"] != 0:
        bs = f"{best['assess']['objective']:+.2f}"
    return (f"[{tag}] {i:3d}/{budget}  {fmt(p)}  ->  speed {a.get('objective', float('nan')):+7.2f}  "
            f"{cons}  {en}  {flag}  best under the rule {bs}")


# ----------------------------------------------------------------------------- proposers
def _uniform(n, rng):
    lo, hi = box_x()
    return [from_x(rng.uniform(lo, hi)) for _ in range(n)]


def propose_random(hist, n, rng, **_):
    return _uniform(n, rng), "draw from the box"


def propose_local(hist, n, rng, **_):
    """(1+lambda) local search whose incumbent starts at the shipped tuning, which is a feasible point at 0."""
    if not hist:
        return propose_random(hist, n, rng)
    inc = to_x(max(hist, key=select_key)["params"])
    lo, hi = box_x()
    step = 0.15 * (hi - lo)
    return [from_x(np.clip(inc + rng.normal(0, step), lo, hi)) for _ in range(n)], \
        "Gaussian step around the incumbent"


def propose_tune(hist, n, rng, budget=40, **_):
    """The recipe of roadmap s29: random draws for the first half of the budget, then a local search."""
    if len(hist) - 1 < budget // 2:
        return propose_random(hist, n, rng)
    return propose_local(hist, n, rng)


def _gp(X, y, Xs, length=0.35, noise=0.05):
    """Zero-mean GP with an RBF kernel on the unit cube; posterior mean and standard deviation at Xs."""
    def k(A, B):
        return np.exp(-0.5 * ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) / length ** 2)
    ym, ys = y.mean(), (y.std() or 1.0)
    L = np.linalg.cholesky(k(X, X) + noise * np.eye(len(X)))
    al = np.linalg.solve(L.T, np.linalg.solve(L, (y - ym) / ys))
    Ks = k(X, Xs)
    mu = Ks.T @ al
    v = np.linalg.solve(L, Ks)
    sd = np.sqrt(np.clip(1.0 - (v ** 2).sum(0), 1e-9, None))
    return ym + ys * mu, ys * sd


def _ncdf(z):
    from math import erf, sqrt
    return np.array([0.5 * (1.0 + erf(float(v) / sqrt(2.0))) for v in np.atleast_1d(z)])


def propose_bayes(hist, n, rng, **_):
    """Constrained Bayesian optimisation: EI on the objective times the probability that every constraint holds.

    Modelling the constraints separately rather than folding them into one score is what keeps this arm a fair
    strong baseline: a surrogate fitted to a discontinuous feasibility-ranked scalar would be handicapped."""
    done = [h for h in hist if h["assess"].get("constraints")]      # the shipped point too: it is data
    if len(done) < 5:
        return propose_random(hist, n, rng)
    lo, hi = box_x()
    span = hi - lo
    X = np.array([(to_x(h["params"]) - lo) / span for h in done])
    cand_x = rng.uniform(lo, hi, size=(4000, len(KEYS)))
    cand = (cand_x - lo) / span

    y = np.clip(np.array([h["assess"]["objective"] for h in done], float), -100.0, None)
    mu, sd = _gp(X, y, cand)
    feas = np.array([select_key(h)[0] == 1 for h in done])
    best = float(y[feas].max()) if feas.any() else None

    p_feas = np.ones(len(cand))
    for name, _col in LOAD_CONSTRAINTS + (("energy", None),):
        cv = np.array([h["assess"]["constraints"][name]["value"] for h in done], float)
        if not np.isfinite(cv).all():
            continue
        cm, cs = _gp(X, cv, cand)
        p_feas *= _ncdf(cm / np.maximum(cs, 1e-9))          # P(reduction >= 0)
    if SCREEN_TOL is not None:                               # the same screen the llm arm is told about
        wv = np.array([worst_per_wind(h)[0] for h in done], float)
        if np.isfinite(wv).all():
            wm, ws = _gp(X, wv, cand)
            p_feas *= _ncdf((wm + SCREEN_TOL) / np.maximum(ws, 1e-9))   # P(worst single wind >= -tol)

    if best is None:
        acq = p_feas                                        # nothing feasible yet: go looking for feasibility
        why = "probability of feasibility under GP surrogates (no feasible point yet)"
    else:
        z = (mu - best) / np.maximum(sd, 1e-9)
        acq = ((mu - best) * _ncdf(z) + sd * np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)) * p_feas
        why = "expected improvement times probability of feasibility under GP surrogates"

    out, taken = [], []
    for i in np.argsort(-acq):
        p = from_x(cand_x[i])
        if key_of(p) in {key_of(q) for q in out}:
            continue
        if taken and min(float(np.linalg.norm(cand[i] - cand[j])) for j in taken) < 0.15:
            continue                                        # keep a batch from collapsing onto one point
        out.append(p)
        taken.append(i)
        if len(out) >= n:
            break
    return out, why


SYSTEM = """You propose tunings for the pitch controller of a 5 MW wind turbine, to be tested in aeroelastic
simulation. You are given the tunable quantities with their units and physical meaning, the objective, the
constraints, and every tuning tried so far with what it measured. Propose the next batch. You are being
compared against random search and constrained Bayesian optimisation on an equal simulation budget, so spend
each candidate deliberately: early on, find out which quantity the objective and the constraints are actually
sensitive to; later, exploit it. Answer with JSON only:
{"candidates": [{"omega_pc": x, "zeta_pc": x, "lpf_corner": x, "fa_ki": x, "fa_hpf": x,
                 "why": "one short sentence"}, ...]}"""

# Under the margin screen (s62): the same task statement without the race framing, with what counts said
# plainly - a tuning that still holds both load terms at every wind speed on realisations the model never sees.
SYSTEM_SCREEN = """You propose tunings for the pitch controller of a 5 MW wind turbine, to be tested in
aeroelastic simulation. You are given the tunable quantities with their units and physical meaning, the
objective, the constraints, the rule under which one tuning is finally selected and shipped, and every tuning
tried so far with what it measured, including its load terms at each wind speed. What counts is a tuning that
still holds both load terms at every wind speed on wind realisations you will not see; a candidate that
improves regulation and loses a load term at one wind speed is worth nothing. Spend each candidate
deliberately: early on, find out which quantity the objective and the per-wind load terms are sensitive to;
later, exploit it with margin. Answer with JSON only:
{"candidates": [{"omega_pc": x, "zeta_pc": x, "lpf_corner": x, "fa_ki": x, "fa_hpf": x,
                 "why": "one short sentence"}, ...]}"""

OBJECTIVE_TEXT = """The controller is ROSCO's gain-scheduled PI collective-pitch loop on the NREL 5 MW turbine,
above rated, over 12 to 24 m/s in 2 m/s steps at IEC turbulence class B, 600 s per wind field.

Tunable quantities, all of them ROSCO's own; the gain schedule is regenerated from omega_pc and zeta_pc exactly
as ROSCO's toolbox does, so every candidate is a controller that could be shipped in a tuning file:
  omega_pc   [rad/s]   target natural frequency of the pitch loop. Shipped 0.6. The first tower fore-aft mode
                       is at 2.04 rad/s and a loop whose bandwidth reaches it drives thrust into that mode.
  zeta_pc    [-]       target damping ratio of the pitch loop. Shipped 0.7.
  lpf_corner [rad/s]   corner of the first-order low-pass filter on measured generator speed. Shipped 1.5708.
                       The blade-passing frequency at rated speed is 3.80 rad/s; above 19.2 rad/s the filter
                       passes essentially the whole signal.
  fa_ki   [rad s/m]    gain of ROSCO's fore-aft tower damper, which feeds high-passed nacelle fore-aft
                       acceleration to collective pitch. 0 turns it off, which is how the turbine ships. Its
                       pitch contribution is limited to 10 degrees.
  fa_hpf     [rad/s]   corner of the high pass on that acceleration signal; only acts when fa_ki > 0. It has to
                       reject quasi-static thrust and pass the 2.04 rad/s tower mode.

Objective, to maximise: the reduction of generator-speed regulation error (mean square deviation from rated
speed on above-rated samples) against the shipped controller on the same wind fields, in percent.

Constraints, all against the shipped controller on the same wind fields: tower-base fore-aft damage-equivalent
load, blade-root out-of-plane damage-equivalent load, and energy capture must each be no worse than shipped. A
candidate counts as infeasible only where the loss is larger than the measurement can resolve. A candidate that
improves regulation but loses tower fatigue is of no use."""

SHIP_TEXT = """
The rule that decides what is reported is stricter than the constraints above, and it is the rule you are
selected under. The selected tuning is the one with the largest regulation reduction whose tower-base AND
blade-root load terms are >= 0 at EVERY ONE of the seven wind speeds on these design episodes - one point of
margin inside the published tolerance of -1. It is then measured on eight further wind realisations per wind
speed that you will never see, and it ships only if both load terms stay within 1 point of shipped at every
wind speed there. The design episodes are two per wind speed, so a per-wind load term read here is uncertain
by about one point. In the campaigns so far every candidate that held the set-mean constraints but read below
zero at a single wind speed on the design episodes failed off them - always on the tower-base load at 22-24
m/s - and every candidate that read above zero at every wind speed held. A candidate at +30 regulation that
reads -0.5 at 24 m/s is worth nothing; a candidate at +23 that reads +0.5 at every wind speed ships. Turning
the tower damper off buys regulation and loses the tower-base load at the highest wind speeds; the filter
corner buys regulation nearly free up to about 3.4 rad/s."""


def _per_wind_cells(h: dict) -> str:
    """The two load terms at each of the seven wind speeds, and the worst of them - what the screen reads."""
    pw = h.get("per_wind") or {}
    us = sorted(pw, key=float)
    if not us:
        return ""
    tw = "/".join(f"{pw[u].get('tower', float('nan')):+.1f}" for u in us)
    bl = "/".join(f"{pw[u].get('blade', float('nan')):+.1f}" for u in us)
    w, at, which = worst_per_wind(h)
    return (f"\n        per wind {us[0]}..{us[-1]} m/s  tower {tw}  blade {bl}"
            f"  worst {w:+.1f} ({which} at {at} m/s)")


def propose_llm(hist, n, rng, client=None, **_):
    rows = []
    for h in sorted(hist[1:], key=select_key, reverse=True)[:40]:
        a, p = h["assess"], h["params"]
        c = a.get("constraints", {})
        if not c:
            rows.append(f"  {fmt(p)}   ->  did not run")
            continue
        if SCREEN_TOL is None:
            flag = "FEASIBLE" if a["feasible"] else "infeasible"
        elif select_key(h)[0] == 1:
            flag = "HOLDS AT EVERY WIND SPEED"
        else:
            flag = "set mean only - fails the rule" if a["feasible"] else "infeasible"
        rows.append(f"  omega {p['omega_pc']:.3f} zeta {p['zeta_pc']:.3f} corner {p['lpf_corner']:.3f} "
                    f"damper {p['fa_ki']:.4f} hpf {p['fa_hpf']:.3f}"
                    f"  ->  speed {a['objective']:+7.2f}  power {a.get('power', float('nan')):+7.2f}"
                    f"  tower {c['tower']['value']:+7.2f}  blade {c['blade']['value']:+6.2f}"
                    f"  energy {c['energy']['value']:+6.3f}"
                    f"  {flag}" + (_per_wind_cells(h) if SCREEN_TOL is not None else ""))
    box = "\n".join(f"  {k}: {v[0]} to {v[1]}" for k, v in BOX.items())
    task = OBJECTIVE_TEXT + (SHIP_TEXT if SCREEN_TOL is not None else "")
    user = (f"{task}\n\nBox (a candidate outside it is clipped):\n{box}\n\n"
            f"The shipped controller is omega_pc 0.6, zeta_pc 0.7, lpf_corner 1.5708, fa_ki 0 and scores 0 on "
            f"every quantity by definition.\n\nTried so far, best first under the rule "
            + ("that a tuning holding both loads at every wind speed beats one that does not, and a feasible one "
               "beats an infeasible one" if SCREEN_TOL is not None else "that a feasible tuning beats an "
               "infeasible one")
            + f" ({len(hist) - 1} used of the budget):\n"
            + ("\n".join(rows) if rows else "  (nothing yet)")
            + f"\n\nAll numbers are percentage reductions against the shipped controller; positive is better, and "
              f"for energy a negative value means energy was lost. Propose exactly {n} candidates.")
    out = client.ask_json(SYSTEM_SCREEN if SCREEN_TOL is not None else SYSTEM, user,
                          tag=f"rosco_search_{len(hist):02d}")
    cands, why = [], []
    for c in (out.get("candidates") or [])[:n]:
        try:
            cands.append(canon({k: float(c[k]) for k in KEYS}))
            why.append(str(c.get("why", ""))[:160])
        except (KeyError, TypeError, ValueError):
            continue
    while len(cands) < n:                            # a malformed reply must not shrink the budget
        cands += _uniform(1, rng)
        why.append("(filled randomly: the model returned fewer candidates than asked)")
    return cands, " | ".join(why)


PROPOSERS = {"random": propose_random, "local": propose_local, "bayes": propose_bayes,
             "llm": propose_llm, "tune": propose_tune}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposer", choices=list(PROPOSERS), required=True)
    ap.add_argument("--repeat", type=int, default=0, help="independent run of the same proposer")
    ap.add_argument("--budget", type=int, default=24, help="candidates after the shipped start, which is free")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--means", nargs="+", type=float, default=None)
    ap.add_argument("--ti", default=None)
    ap.add_argument("--episode_s", type=float, default=600.0)
    ap.add_argument("--root", default="~/wtrl/exp/roscosearch_native")
    ap.add_argument("--name", default=None, help="run directory name (default <proposer>_r<repeat>)")
    ap.add_argument("--template", default="~/wtrl/runs/template_5mw")
    ap.add_argument("--config_run", default="~/wtrl/exp/jg3t_s0")
    ap.add_argument("--port0", type=int, default=7800)
    ap.add_argument("--screen_tol", type=float, default=None,
                    help="select under the margin screen of roadmap s62: the worst single-wind load term must "
                         "be >= -screen_tol on the design set (0.0 = one point of margin). Every proposer "
                         "selects under it and the llm proposer is told about it. Default: the set-mean rule")
    ap.add_argument("--emit_template", default=None, metavar="DST",
                    help="write the case template of this run's selected design to DST and exit")
    a = ap.parse_args()
    global SCREEN_TOL
    SCREEN_TOL = a.screen_tol

    a.root = os.path.expanduser(f"{a.root}/{a.name or f'{a.proposer}_r{a.repeat}'}")
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    tag = f"{a.proposer}/r{a.repeat}"

    if a.emit_template:
        from scripts.rosco_native import make_template
        best = json.load(open(root / "best.json"))
        make_template(Path(os.path.expanduser(a.template)), Path(os.path.expanduser(a.emit_template)), best["params"])
        print(f"[{tag}] {fmt(best['params'])} -> {a.emit_template}")
        return

    ev = Evaluator(a)
    if a.means:
        ev.extra_args = ["--means", *[f"{m:g}" for m in a.means]] + (["--ti", a.ti] if a.ti else [])
    hp = root / "history.jsonl"
    hist = [json.loads(l) for l in open(hp)] if hp.exists() else []
    rng = np.random.default_rng([911, a.repeat, sum(map(ord, a.proposer))])
    for _ in range(len(hist)):
        rng.random()                                 # keep a resumed run on the stream it would have had

    client = None
    if a.proposer == "llm":
        from llm.client import LLMClient
        client = LLMClient(transcript_path=root / "llm_transcript.jsonl")

    if not hist:
        asm, rep = record(ev.run(START, 0))
        hist.append({"i": 0, "params": canon(START), "assess": asm, **rep,
                     "why": "the shipped controller, free of budget"})
        open(hp, "a").write(json.dumps(hist[-1]) + "\n")
        print(f"[{tag}] shipped start: objective {asm['objective']:+.3f} "
              f"(identity check, must be exactly 0.000)", flush=True)

    seen = {key_of(h["params"]) for h in hist}
    while len(hist) - 1 < a.budget:
        n = min(a.batch, a.budget - (len(hist) - 1))
        cands, why = PROPOSERS[a.proposer](hist, n, rng, client=client, budget=a.budget)
        cands = [c for c in cands if key_of(c) not in seen] or _uniform(n, rng)
        with ThreadPoolExecutor(max_workers=a.parallel) as pool:
            res = list(pool.map(lambda ic: ev.run(ic[1], ic[0] % a.parallel), enumerate(cands)))
        for p, fit in zip(cands, res):
            asm, rep = record(fit)
            hist.append({"i": len(hist), "params": p, "assess": asm, **rep, "why": why})
            seen.add(key_of(p))
            open(hp, "a").write(json.dumps(hist[-1]) + "\n")
            best = max(hist, key=select_key)      # the shipped tuning is always one of the options
            print(line(tag, len(hist) - 1, a.budget, p, asm, best, rec=hist[-1]), flush=True)

    if len(hist) < 2:                                 # --budget 0: the identity check on its own
        print(f"[{tag}] identity only, no candidates requested", flush=True)
        return
    best = max(hist, key=select_key)
    json.dump(best, open(root / "best.json", "w"), indent=1)
    a_ = best["assess"]
    if SCREEN_TOL is not None and best["i"] != 0 and select_key(best)[0] != 1:
        best = hist[0]                            # nothing held per wind: the shipped tuning is the selection
        json.dump(best, open(root / "best.json", "w"), indent=1)
        a_ = best["assess"]
    if best["i"] == 0:
        print(f"[{tag}] done. NO candidate improved on the shipped controller in {a.budget} simulations, so the "
              f"selection is the shipped tuning itself and this arm bought 0.00 points", flush=True)
    elif a_["feasible"]:
        print(f"[{tag}] done. selected {fmt(best['params'])}: regulation {a_['objective']:+.2f} points, "
              f"every constraint held", flush=True)
    else:
        print(f"[{tag}] done. NO FEASIBLE TUNING in {a.budget} simulations; least-violating candidate "
              f"{fmt(best['params'])} short by {a_['shortfall']:.2f} points", flush=True)


if __name__ == "__main__":
    main()
