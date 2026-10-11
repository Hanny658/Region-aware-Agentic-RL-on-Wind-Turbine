"""Exercise every proposer and the selection rule on synthetic evaluations: no OpenFAST, only plumbing."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from eval.selection import assess, rank_key, describe
from scripts.rosco_native import START, BOX, canon, key_of, to_x, from_x, box_x, fmt, gains, make_template
import scripts.rosco_search as rs

# round-trip of the search coordinates
rng = np.random.default_rng(0)
lo, hi = box_x()
for _ in range(400):
    p = from_x(rng.uniform(lo, hi))
    q = from_x(to_x(p))
    assert all(abs(p[k] - q[k]) <= 1e-9 + 1e-9 * abs(p[k]) for k in p), (p, q)
print("coordinate round trip exact over 400 random points")
assert from_x(to_x(canon(START))) == canon(START), "the shipped start must be a fixed point"
print("shipped start is a fixed point of the coordinates:", fmt(canon(START)))

# a fake evaluation: regulation improves with loop bandwidth and filter corner, tower load pays for bandwidth,
# the damper buys tower load back. Shaped only so that feasibility is sometimes reachable and sometimes not.
def fake(p, seed=0):
    r = np.random.default_rng(seed)
    pe = []
    for u in (12, 14, 16, 18, 20, 22, 24):
        for s in (1, 2):
            sp = 60 * np.tanh(1.4 * (p["omega_pc"] - 0.6)) + 6 * np.log(p["lpf_corner"] / 1.5708) + r.normal(0, 2)
            tw = -18 * (p["omega_pc"] - 0.6) + 40 * p["fa_ki"] + r.normal(0, 1)
            pe.append({"mean_wind": u, "gen_speed_mse_red_pct": sp, "power_mse_red_pct": 0.8 * sp,
                       "TwrBsMyt_DEL_red_pct": tw, "RootMyc1_DEL_red_pct": 0.3 * tw + 1.0,
                       "energy_MWh": 1.0 + 0.0001 * sp, "energy_base_MWh": 1.0,
                       "pitch_travel_deg": 300 * (1 + p["omega_pc"]), "pitch_travel_base_deg": 300})
    return {"per_episode": pe}

a0 = assess(fake(canon(START)))
print("identity on the fake evaluator:", describe(a0))

for name in ("random", "local", "bayes", "tune"):
    hist = [{"i": 0, "params": canon(START), "assess": assess(fake(canon(START))), "why": "start"}]
    rng = np.random.default_rng(1)
    for i in range(12):
        cands, why = rs.PROPOSERS[name](hist, 2, rng, client=None, budget=24)
        assert len(cands) == 2, (name, len(cands))
        for p in cands:
            assert all(BOX[k][0] - 1e-12 <= p[k] <= BOX[k][1] + 1e-12 for k in BOX), (name, p)
            hist.append({"i": len(hist), "params": p, "assess": assess(fake(p, seed=len(hist))), "why": why})
    best = max(hist[1:], key=rank_key)
    nf = sum(1 for h in hist[1:] if h["assess"]["feasible"])
    print(f"{name:>7}: 24 candidates, {nf:2d} feasible, selected {fmt(best['params'])} "
          f"-> {'feasible' if best['assess']['feasible'] else 'infeasible'} "
          f"regulation {best['assess']['objective']:+.2f}")

# constraint domination: a feasible candidate must beat a better-scoring infeasible one
feas = {"assess": {"feasible": True, "objective": 1.0, "shortfall": 0.0}}
infeas = {"assess": {"feasible": False, "objective": 99.0, "shortfall": 0.5}}
worse = {"assess": {"feasible": False, "objective": 99.0, "shortfall": 9.0}}
assert max([feas, infeas], key=rank_key) is feas
assert max([worse, infeas], key=rank_key) is infeas
print("constraint domination orders feasible > least-violating > more-violating")

# the DISCON writer
import tempfile, pathlib, re
src = pathlib.Path(os.path.expanduser("~/wtrl/runs/template_5mw"))
with tempfile.TemporaryDirectory() as d:
    dst = pathlib.Path(d) / "case"
    make_template(src, dst, canon(START))
    a = (src / "DISCON.IN").read_text().splitlines()
    b = (dst / "DISCON.IN").read_text().splitlines()
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    print(f"shipped tuning rewritten: {len(diff)} line(s) differ from the template")
    for x, y in diff:
        print("   template:", x.split('!')[0].strip()[:60], "| written:", y.split('!')[0].strip()[:60])
    dst2 = pathlib.Path(d) / "case2"
    make_template(src, dst2, canon({**START, "omega_pc": 0.9, "zeta_pc": 1.2, "lpf_corner": 6.0,
                                    "fa_ki": 0.05, "fa_hpf": 0.5}))
    c = (dst2 / "DISCON.IN").read_text().splitlines()
    ch = [y.split("!")[1].split("-")[0].strip() for x, y in zip(a, c) if x != y]
    print("a tuned candidate changes:", ", ".join(ch))
