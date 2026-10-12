"""Selection rule for a controller parameter search: one objective, physical constraints, no weights (2026-10-11).

Replaces the composite score that averaged four percentage reductions and charged invented penalties. That
composite chose the controller rather than measuring it: on the search of 2026-10-09 the top-scoring designs
paid for regulation with tower fatigue, up to 20 points of it, because a mean of four terms will trade one for
the others.

What a classical pitch-tuning exercise actually states is a constrained problem, so that is what this is:

    objective    the generator-speed regulation error reduction against the paired shipped controller.
                 One quantity -- the one the pitch loop's PI acts on. Power regulation is reported but not
                 optimised: above rated at constant torque the two carry much the same information
                 (correlation 0.94-0.95 on the model-based rows), so optimising both double-counts it.

    constraints  tower-base damage-equivalent load, blade-root damage-equivalent load and energy capture,
                 each no worse than the shipped controller.

A constraint needs a tolerance, and rather than choose one we use the resolution of the measurement itself: a
candidate is infeasible only where the loss is *resolvable* on its own episodes, i.e. where the upper end of
the 95 % paired stratified bootstrap interval of that reduction still lies below zero. Nothing is invented and
nothing is weighted. For reference, the measured median half-widths over the 14-episode design set are 1.30
points (tower), 1.00 (blade) and 0.031 (energy), so the rule is strict where the measurement is sharp.

Ranking is constraint domination (Deb 2000, the rule NSGA-II uses):

    a feasible candidate beats any infeasible one;
    between feasible candidates, the larger objective wins;
    between infeasible candidates, the smaller total resolved loss wins.

The shipped controller is itself one of the candidates: evaluated against itself it reads exactly zero on every
term with intervals of zero width, so it is feasible by construction and an objective of 0 is the floor of any
honest selection. A search that finds nothing selects it and reports a gain of exactly zero; it never reports a
controller worse than the one already in the field as its winner.
"""
from __future__ import annotations

import numpy as np

OBJECTIVE = ("speed", "gen_speed_mse_red_pct", "generator-speed regulation error")
REPORTED = (("power", "power_mse_red_pct"), ("speed", "gen_speed_mse_red_pct"),
            ("tower", "TwrBsMyt_DEL_red_pct"), ("blade", "RootMyc1_DEL_red_pct"))
LOAD_CONSTRAINTS = (("tower", "TwrBsMyt_DEL_red_pct"), ("blade", "RootMyc1_DEL_red_pct"))
N_BOOT = 2000
FAILED_SHORTFALL = 1.0e3        # an evaluation that did not complete ranks below every candidate that did


def _resample(strata: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Index matrix (n_boot, n_episodes) of a stratified bootstrap over episodes, strata = mean wind speed."""
    rng = np.random.default_rng(seed)
    blocks = [np.where(strata == s)[0] for s in np.unique(strata)]
    return np.concatenate([rng.choice(b, size=(n_boot, len(b)), replace=True) for b in blocks], axis=1)


def assess(fit: dict, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """The selection record of one evaluation: objective, constraints with their intervals, feasibility.

    `fit` is what evaluate.py writes: per-episode reductions against the paired shipped controller, plus the
    per-episode energy of both sides."""
    pe = [p for p in fit.get("per_episode", []) if p.get("mean_wind") is not None]
    if fit.get("failed") or len(pe) < 4:
        return {"objective": -FAILED_SHORTFALL, "feasible": False, "shortfall": FAILED_SHORTFALL,
                "n_episodes": len(pe), "constraints": {},
                "failure_kind": fit.get("failure_kind", "" if pe else "no episodes"),
                "note": fit.get("failed", "too few usable episodes")}

    u = np.array([p["mean_wind"] for p in pe], float)
    idx = _resample(u, n_boot, seed)

    def interval(v: np.ndarray) -> tuple[float, float, float]:
        d = v[idx].mean(axis=1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        return float(v.mean()), float(lo), float(hi)

    cols = {name: np.array([p.get(col, np.nan) for p in pe], float) for name, col in REPORTED}
    obj_name, obj_col, _ = OBJECTIVE
    ok = np.isfinite(cols[obj_name])
    objective = float(np.nanmean(cols[obj_name])) if ok.any() else float("nan")
    # the regulation metrics are defined only on episodes that are above rated for at least half their samples,
    # so say how many episodes carry the objective rather than leaving the reader to assume all of them
    obj_lo, obj_hi = float("nan"), float("nan")
    if ok.all():
        d = cols[obj_name][idx].mean(axis=1)
        obj_lo, obj_hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))

    cons: dict[str, dict] = {}
    for name, _col in LOAD_CONSTRAINTS:
        v = cols[name]
        if not np.isfinite(v).all():
            cons[name] = {"value": float("nan"), "lo": float("nan"), "hi": float("nan"),
                          "shortfall": FAILED_SHORTFALL}
            continue
        val, lo, hi = interval(v)
        cons[name] = {"value": val, "lo": lo, "hi": hi, "shortfall": max(0.0, -hi)}

    # energy capture: a ratio of sums, so it is resampled as a ratio rather than averaged per episode
    en = np.array([p.get("energy_MWh", np.nan) for p in pe], float)
    eb = np.array([p.get("energy_base_MWh", np.nan) for p in pe], float)
    if np.isfinite(en).all() and np.isfinite(eb).all() and eb.sum() > 0:
        d = 100.0 * (1.0 - en[idx].sum(axis=1) / eb[idx].sum(axis=1))      # loss, positive = energy lost
        lo, hi = np.percentile(-d, [2.5, 97.5])                            # as a reduction: positive = no loss
        val = float(100.0 * (en.sum() / eb.sum() - 1.0))
        cons["energy"] = {"value": val, "lo": float(lo), "hi": float(hi), "shortfall": max(0.0, -float(hi))}
    else:
        cons["energy"] = {"value": float("nan"), "lo": float("nan"), "hi": float("nan"),
                          "shortfall": FAILED_SHORTFALL}

    shortfall = float(sum(c["shortfall"] for c in cons.values()))
    return {"objective": objective, "objective_lo": obj_lo, "objective_hi": obj_hi,
            "feasible": shortfall == 0.0, "shortfall": shortfall,
            "n_episodes": len(pe), "n_objective": int(ok.sum()), "constraints": cons,
            **{name: float(np.nanmean(cols[name])) for name, _ in REPORTED}}


def rank_key(rec: dict) -> tuple[int, float]:
    """Constraint domination, as a key for max(): feasible first, then the objective, then the smaller loss."""
    a = rec.get("assess", rec)
    return (1 if a.get("feasible") else 0,
            a.get("objective", -FAILED_SHORTFALL) if a.get("feasible") else -a.get("shortfall", FAILED_SHORTFALL))


def best_of(records: list[dict]) -> dict | None:
    """The selected design among records carrying an `assess` block."""
    return max(records, key=rank_key) if records else None


# ------------------------------------------------------------------- the same constraints, at every wind speed
# A constraint on the set mean is gamed across wind speeds: roadmap s32 measured exactly that, and s52.2 saw it
# again, a tuning holding the mean tower-base load at -0.6 points while losing 5.9 of it at 18 m/s. A tuning that
# is to be reported as a baseline has to hold its loads where they are produced, so the load constraints are also
# required at every single mean wind speed. The tolerance is the one roadmap s33 already states for the
# manuscript, one point, not a new convention: the design set carries two episodes per wind speed, too few for a
# per-wind bootstrap, so per wind this is a screen on the design set and a measurement off the design winds,
# where there are four.
PER_WIND_TOL = 1.0


def worst_per_wind(rec: dict, keys=("tower", "blade")) -> tuple[float, str, str]:
    """The worst load reduction at any single mean wind speed of a history record, and where it happens."""
    worst, at, which = float("inf"), "", ""
    for u, d in (rec.get("per_wind") or {}).items():
        for k in keys:
            v = d.get(k)
            if v is not None and v == v and float(v) < worst:
                worst, at, which = float(v), u, k
    return (worst, at, which) if at else (float("nan"), "", "")


def per_wind_rank_key(rec: dict, tol: float = PER_WIND_TOL) -> tuple[int, float]:
    """Constraint domination with the load constraints required per wind speed as well as over the set."""
    a = rec.get("assess", rec)
    w = worst_per_wind(rec)[0]
    over = max(0.0, -tol - w) if w == w else FAILED_SHORTFALL
    ok = bool(a.get("feasible")) and over == 0.0
    return (1 if ok else 0,
            a.get("objective", -FAILED_SHORTFALL) if ok else -(a.get("shortfall", FAILED_SHORTFALL) + over))


def describe(a: dict) -> str:
    """One line: what this candidate does and whether it may be shipped."""
    if not a.get("constraints"):
        return f"unusable ({a.get('note', 'no episodes')})"
    c = a["constraints"]
    tag = "feasible" if a["feasible"] else f"infeasible by {a['shortfall']:.2f}"
    return (f"speed {a['objective']:+6.2f}  power {a.get('power', float('nan')):+6.2f}  "
            f"tower {c['tower']['value']:+6.2f} [{c['tower']['lo']:+.2f},{c['tower']['hi']:+.2f}]  "
            f"blade {c['blade']['value']:+6.2f} [{c['blade']['lo']:+.2f},{c['blade']['hi']:+.2f}]  "
            f"energy {c['energy']['value']:+6.3f}  {tag}")
