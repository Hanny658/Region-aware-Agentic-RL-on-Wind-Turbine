"""Measure the band off the design winds, which is what fixes the baseline row (2026-10-12).

Roadmap s53.2 settled how a tuning has to be judged: the load constraints hold where the loads are produced, at
every single mean wind speed, measured where there are four episodes per wind speed rather than two. Stage 3 did
that for the twelve tunings the twelve runs selected, and three of the best of them failed it. The band of
section 53.3 has members with more regulation than the verified baseline that no run selected, because selection
inside a run is per run; they have to be measured the same way before any of them can be reported.

Which tunings: every one in the pool that holds both load terms at every wind speed on the design set (the
screen) and either

    buys more regulation than the standing baseline (--above, default the stage-1 baseline, 28.3), so the
    stated rule -- the largest regulation reduction whose per-wind load terms hold off the design winds -- can
    be applied by walking down the list, or

    spends no more actuator duty than the shipped controller (--duty_below, default 1.01 times the shipped
    pitch travel) while buying regulation its own interval resolves, because a gain that costs no duty is
    reported whatever its size. The resolution clause is load-bearing: a slow loop uses less pitch travel than
    shipped by not acting at all, and this pool holds such tunings down to -1268 points of regulation.

Each one is then scored on held-out seeds 3-6 and on fresh seeds 7-10, 28 episodes each, same protocol as the
campaign. Resumable: every evaluation is cached by parameter hash, protocol and seed set, so a stopped run
repeats nothing.

    ~/wtrl/run.sh python scripts/dev/rosco_verify_band.py [--above 28.3] [--duty_below 1.01] [--csv <path>]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from eval.selection import PER_WIND_TOL, REPORTED, assess, per_wind_rank_key, worst_per_wind  # noqa: E402
from scripts.rosco_native import Evaluator, fmt, key_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--roots", nargs="+",
                default=[os.path.expanduser("~/wtrl/exp/roscotune_native"),
                         os.path.expanduser("~/wtrl/exp/roscosearch_native")])
ap.add_argument("--above", type=float, default=28.3, help="regulation of the standing baseline")
ap.add_argument("--duty_below", type=float, default=1.01, help="travel ratio under which a tuning is free")
ap.add_argument("--tol", type=float, default=PER_WIND_TOL)
ap.add_argument("--root", default=os.path.expanduser("~/wtrl/exp/roscosearch_native/_verify"))
ap.add_argument("--template", default="~/wtrl/runs/template_5mw")
ap.add_argument("--config_run", default="~/wtrl/exp/jg3t_s0")
ap.add_argument("--workers", type=int, default=14)
ap.add_argument("--port0", type=int, default=7700)
ap.add_argument("--means", nargs="+", type=float, default=[12, 14, 16, 18, 20, 22, 24])
ap.add_argument("--ti", default="B")
ap.add_argument("--episode_s", type=float, default=600.0)
ap.add_argument("--csv", default="docs/tables/rosco_band_verified.csv")
ap.add_argument("--dry", action="store_true", help="list what would be measured and what it costs, then stop")
a = ap.parse_args()

pool: dict[str, dict] = {}
for root in a.roots:
    for h in sorted(glob.glob(f"{root}/*/history.jsonl")):
        name = os.path.basename(os.path.dirname(h))
        if "_r" not in name:
            continue
        for line in open(h):
            rec = json.loads(line)
            pool.setdefault(key_of(rec["params"]), rec | {"from": f"{name}#{rec['i']}"})

screened = [r for r in pool.values()
            if r["assess"].get("constraints") and per_wind_rank_key(r, a.tol)[0] == 1]
want = [r for r in screened
        if r["assess"]["objective"] > a.above
        or (r.get("travel_ratio", 9.9) < a.duty_below and r["assess"].get("objective_lo", -9.9e9) > 0.0)]
want.sort(key=lambda r: -r["assess"]["objective"])

print(f"{len(pool)} tunings in the pool, {len(screened)} hold both loads at every wind speed on the design set, "
      f"{len(want)} of those buy more than {a.above:+.1f} points of regulation, or buy a regulation gain their "
      f"own interval resolves at under {a.duty_below:.2f}x the shipped pitch travel:")
for r in want:
    print(f"  {r['assess']['objective']:+7.2f}  tower {r['assess']['constraints']['tower']['value']:+6.2f}  "
          f"travel {r.get('travel_ratio', float('nan')):.3f}x  worst/wind {worst_per_wind(r)[0]:+.1f}  "
          f"{fmt(r['params'])}  ({r['from']})")
print(f"{len(want) * 56} episodes to run, 28 held out and 28 fresh each.\n", flush=True)
if a.dry:
    sys.exit(0)


class Args:
    root, template, config_run = a.root, a.template, a.config_run
    seeds, episode_s, workers, port0 = [3, 4, 5, 6], a.episode_s, a.workers, a.port0


ev = Evaluator(Args())
ev.extra_args = ["--means", *[f"{m:g}" for m in a.means], "--ti", a.ti]


def per_wind_of(fit: dict) -> dict:
    """The per-wind block of an evaluation, in the shape the history records carry."""
    by: dict[str, dict] = {}
    for e in fit.get("per_episode", []):
        if e.get("mean_wind") is None:
            continue
        d = by.setdefault(f"{e['mean_wind']:g}", {})
        for name, col in REPORTED:
            v = e.get(col)
            if v is not None and v == v:
                d.setdefault(name, []).append(float(v))
    return {u: {n: round(float(np.mean(v)), 2) for n, v in d.items()} for u, d in by.items()}


def one(fit: dict) -> dict:
    asm = assess(fit)
    if not asm.get("constraints"):
        return {"ok": False, "note": asm.get("note", "unusable")}
    w, at, which = worst_per_wind({"per_wind": per_wind_of(fit)})
    c = asm["constraints"]
    return {"ok": True, "regulation": asm["objective"], "tower": c["tower"]["value"],
            "blade": c["blade"]["value"], "energy": c["energy"]["value"], "set_feasible": asm["feasible"],
            "worst": w, "at": at, "which": which, "holds": asm["feasible"] and w >= -a.tol}


hdr = (f"{'regulation':>10} {'tower':>7} {'blade':>7} {'worst/wind':>11} {'at':>5} {'which':>6}")
rows = []
for r in want:
    p = r["params"]
    print(f"--- {fmt(p)}   design {r['assess']['objective']:+.2f}, travel "
          f"{r.get('travel_ratio', float('nan')):.3f}x  (from {r['from']})", flush=True)
    ho, fr = one(ev.run(p, 0, seeds=[3, 4, 5, 6])), one(ev.run(p, 1, seeds=[7, 8, 9, 10]))
    print(f"    {'set':>9} {hdr}")
    dw = worst_per_wind(r)
    print(f"    {'design':>9} {r['assess']['objective']:>10.2f} "
          f"{r['assess']['constraints']['tower']['value']:>7.2f} "
          f"{r['assess']['constraints']['blade']['value']:>7.2f} {dw[0]:>11.1f} {dw[1]:>5} {dw[2]:>6}")
    for nm, s in (("held out", ho), ("fresh", fr)):
        if not s["ok"]:
            print(f"    {nm:>9} unusable ({s['note']})")
            continue
        print(f"    {nm:>9} {s['regulation']:>10.2f} {s['tower']:>7.2f} {s['blade']:>7.2f} "
              f"{s['worst']:>11.2f} {s['at']:>5} {s['which']:>6}"
              f"{'' if s['holds'] else '   <- loses more than the tolerance at one wind speed'}")
    holds = bool(ho.get("holds") and fr.get("holds"))
    print(f"    verdict: {'HOLDS per wind off the design winds' if holds else 'does NOT hold'}", flush=True)
    rows.append({"from": r["from"], **{k: v for k, v in p.items()},
                 "design_regulation": round(r["assess"]["objective"], 2),
                 "design_tower": round(r["assess"]["constraints"]["tower"]["value"], 2),
                 "design_worst_per_wind": dw[0], "travel_ratio": r.get("travel_ratio", float("nan")),
                 "adc": r.get("adc", float("nan")),
                 **{f"heldout_{k}": (round(v, 2) if isinstance(v, float) else v) for k, v in ho.items()},
                 **{f"fresh_{k}": (round(v, 2) if isinstance(v, float) else v) for k, v in fr.items()},
                 "holds_per_wind": holds})

kept = [r for r in rows if r["holds_per_wind"]]
print(f"\n{len(kept)} of {len(rows)} hold both load terms at every wind speed on both sets.")
if kept:
    best = max(kept, key=lambda r: r["heldout_regulation"])
    print(f"The stated rule -- the largest regulation reduction whose per-wind load terms hold off the design "
          f"winds -- selects:")
    print(f"  {fmt({k: best[k] for k in ('omega_pc', 'zeta_pc', 'lpf_corner', 'fa_ki', 'fa_hpf')})}")
    print(f"  regulation {best['heldout_regulation']:+.2f} held out, {best['fresh_regulation']:+.2f} fresh; "
          f"tower {best['heldout_tower']:+.2f} / {best['fresh_tower']:+.2f}; "
          f"travel {best['travel_ratio']:.3f}x; worst per wind {best['heldout_worst']:+.2f} / "
          f"{best['fresh_worst']:+.2f}")
    print(f"  against the standing baseline at {a.above:+.1f} design, +28.32 held out, +29.75 fresh, "
          f"tower +23.93 / +25.32, travel 2.013x")
else:
    print("Nothing in the band beats the standing baseline under the per-wind rule; it stands.")

if rows:
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {a.csv}")
