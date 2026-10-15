"""Who should propose a classical controller's parameters, tested rather than tabulated (2026-10-13, roadmap s59).

`rosco_search_table.py` reports the proposer comparison run by run. This applies the two tests pre-registered in
s59.1 to it, so the arms are compared by something with a p-value instead of by a median over three runs:

  primary    the simulations spent before the first feasible gain, llm against each other arm, one-sided, by
             exact permutation of the arm labels (ties are everywhere here, so the permutation distribution of
             the rank-sum statistic is computed rather than approximated)
  secondary  how many runs of each arm delivered a gain at all, by Fisher's exact test

and it does both under the two selection rules the project has, because they disagree:

  set mean   the rule the campaigns searched under: the three load and energy constraints on the set mean
  per wind   the rule s32, s53.2 and s55 forced on the project: both load terms within one point at EVERY mean
             wind speed. On the design set this is a screen with two episodes per wind, not a measurement, so
             what it selects is confirmed off the design winds - and this script says which re-selected tunings
             still need that confirmation.

A run that never found a gain within its budget is censored, not dropped: it enters the primary test at
budget + 1, tied last, which is the convention stated in s59.1 before the n = 6 runs existed.

It also audits the llm arm. `propose_llm` fills the budget from the box when a reply is malformed and writes
`filled randomly` into the history, so a run whose model calls failed would otherwise be averaged in as if the
model had proposed its candidates.

    ~/wtrl/run.sh python scripts/dev/rosco_proposer_test.py [--tol 1.0] [--csv docs/tables/rosco_proposer_test.csv]
"""
from __future__ import annotations

import argparse
import csv as csvmod
import glob
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from eval.selection import PER_WIND_TOL, assess, per_wind_rank_key, rank_key, worst_per_wind  # noqa: E402
from scripts.rosco_native import fmt, key_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.expanduser("~/wtrl/exp/roscosearch_native"))
ap.add_argument("--tol", type=float, default=PER_WIND_TOL,
                help="the SHIPPING tolerance, applied off the design winds; the published 1.0")
ap.add_argument("--screen_tol", type=float, default=None,
                help="the SCREEN tolerance used to select on the design set (default: the same as --tol); "
                     "0.0 is the margin screen of roadmap s62.1")
ap.add_argument("--against", default="bayes", help="the arm the primary test is against (the strongest classical)")
ap.add_argument("--search_csv", default="docs/tables/rosco_search.csv",
                help="stage 3 of the campaign, for the off-design verification of the set-mean selections")
ap.add_argument("--csv", default=None,
                help="where to write the per-run table; nothing is written without it, so that a reading under "
                     "another screen never overwrites the committed docs/tables/rosco_proposer_test.csv")
a = ap.parse_args()

ARM_ORDER = ("random", "local", "bayes", "llm")


def perm_p(x: list[float], y: list[float]) -> float:
    """One-sided P(rank sum of x this low or lower) over all arm labellings. Exact, and correct under ties."""
    v = np.asarray(x + y, float)
    r = np.asarray([(np.sum(v < t) + (np.sum(v == t) + 1) / 2) for t in v])     # midranks
    n = len(x)
    obs = r[:n].sum()
    idx = range(len(v))
    tot = hit = 0
    for comb in itertools.combinations(idx, n):
        tot += 1
        if r[list(comb)].sum() <= obs + 1e-9:
            hit += 1
    return hit / tot


# ------------------------------------------------------------------------------------ one row per run
runs = {}
for h in sorted(glob.glob(f"{a.root}/*/history.jsonl")):
    name = os.path.basename(os.path.dirname(h))
    if "_r" not in name:
        continue                                        # _superseded and _verify are not runs
    arm, rep = name.rsplit("_r", 1)
    hist = [json.loads(line) for line in open(h)]
    if len(hist) > 1:
        runs[(arm, int(rep))] = hist
if not runs:
    sys.exit(f"no histories under {a.root}")

verified = {}
if os.path.exists(a.search_csv):
    for r in csvmod.DictReader(open(a.search_csv)):
        verified[(r["proposer"], int(r["repeat"]))] = r

def off_design(key: str, seeds: str) -> tuple[float, float]:
    """The verified regulation and worst single-wind load term of one tuning, from the verification cache.

    Bin mean first, then the worst bin, as everywhere else in the project. Returns nan if that tuning has not
    been evaluated on those seeds, which is the signal that it still needs 28 episodes."""
    hits = glob.glob(f"{a.root}/_verify/cache/eval_{key}_s{seeds}_*.json")
    if not hits:
        return float("nan"), float("nan")
    fit = json.load(open(hits[0]))
    by: dict[float, dict[str, list]] = {}
    for e in fit.get("per_episode", []):
        if e.get("mean_wind") is None:
            continue
        d = by.setdefault(e["mean_wind"], {})
        for nm, c in (("tower", "TwrBsMyt_DEL_red_pct"), ("blade", "RootMyc1_DEL_red_pct")):
            v = e.get(c)
            if v is not None and v == v:
                d.setdefault(nm, []).append(float(v))
    worst = min((float(np.mean(v)) for d in by.values() for v in d.values()), default=float("nan"))
    return float(assess(fit).get("objective", float("nan"))), worst


rows = []
for (arm, rep), hist in sorted(runs.items(), key=lambda kv: (ARM_ORDER.index(kv[0][0])
                                                             if kv[0][0] in ARM_ORDER else 9, kv[0][1])):
    cand = hist[1:]
    budget = len(cand)
    sm = max(hist, key=rank_key)                                    # the shipped point is always an option
    st = a.tol if a.screen_tol is None else a.screen_tol            # the screen; shipping stays at a.tol
    pw = max(hist, key=lambda r: per_wind_rank_key(r, st))
    first_sm = next((h["i"] for h in cand if h["assess"].get("feasible") and h["assess"]["objective"] > 0), None)
    first_pw = next((h["i"] for h in cand if per_wind_rank_key(h, st)[0] == 1
                     and h["assess"]["objective"] > 0), None)
    v = verified.get((arm, rep), {})
    k_pw = key_of(pw["params"]) if first_pw is not None else ""
    reg_ho, w_ho = off_design(k_pw, "3456") if k_pw else (float("nan"), float("nan"))
    reg_fr, w_fr = off_design(k_pw, "78910") if k_pw else (float("nan"), float("nan"))
    # what the run DELIVERS: the tuning its own rule selected on the design set, measured where it was not
    # designed. Both seed sets must hold, because s55 found every failure worse on fresh seeds than held out.
    delivers = bool(k_pw) and w_ho == w_ho and w_fr == w_fr and min(w_ho, w_fr) >= -a.tol
    rows.append({
        "proposer": arm, "repeat": rep, "budget": budget,
        "n_filled_randomly": sum(1 for h in cand if "filled randomly" in str(h.get("why", ""))),
        "n_diverged": sum(1 for h in cand if h["assess"].get("failure_kind") == "diverged"),
        "setmean_gain": first_sm is not None, "setmean_sims_to_first": first_sm,
        "setmean_regulation": round(sm["assess"]["objective"], 2),
        "setmean_worst_per_wind_design": round(worst_per_wind(sm)[0], 2),
        "setmean_heldout_worst_per_wind": v.get("heldout_worst_per_wind", ""),
        "setmean_fresh_worst_per_wind": v.get("fresh_worst_per_wind", ""),
        "setmean_fresh_regulation": v.get("fresh_regulation", ""),
        "perwind_gain": first_pw is not None, "perwind_sims_to_first": first_pw,
        "perwind_regulation": round(pw["assess"]["objective"], 2) if first_pw is not None else 0.0,
        "perwind_worst_per_wind_design": round(worst_per_wind(pw)[0], 2) if first_pw is not None else "",
        "perwind_candidate": f"{arm}_r{rep}#{pw['i']}" if first_pw is not None else "",
        "perwind_params": fmt(pw["params"]) if first_pw is not None else "",
        "perwind_key": k_pw,
        "perwind_travel": pw.get("travel_ratio", float("nan")) if first_pw is not None else "",
        "perwind_heldout_regulation": round(reg_ho, 2) if reg_ho == reg_ho else "",
        "perwind_heldout_worst": round(w_ho, 2) if w_ho == w_ho else "",
        "perwind_fresh_regulation": round(reg_fr, 2) if reg_fr == reg_fr else "",
        "perwind_fresh_worst": round(w_fr, 2) if w_fr == w_fr else "",
        "delivers_off_design": delivers,
        "delivered_sims_to_first": first_pw if delivers else None,
        "delivered_regulation": round(reg_fr, 2) if delivers else 0.0,
    })

print(f"{len(rows)} runs under {a.root}, budget {rows[0]['budget']} candidates each, per-wind tolerance "
      f"{a.tol:.1f} point.\n")
print("Each run twice: under the rule it searched under (set mean) and under the rule the project ships by")
print("(both loads within tolerance at every wind speed). 'sims' is the index of the first candidate that was")
print("both feasible under that rule and better than shipped; '-' means the run never found one.\n")
print(f"{'run':>11} {'div':>4} {'fill':>5} |{'set-mean rule: sims':>21} {'regulation':>11} "
      f"{'design w/w':>11} {'fresh w/w':>10} |{'per-wind rule: sims':>21} {'regulation':>11} {'design w/w':>11} "
      f"{'heldout w/w':>12} {'fresh w/w':>10} {'ships':>6}  selection")
for r in rows:
    print(f"{r['proposer'] + '_r' + str(r['repeat']):>11} {r['n_diverged']:>4} {r['n_filled_randomly']:>5} |"
          f"{str(r['setmean_sims_to_first'] or '-'):>21} {r['setmean_regulation']:>11.2f} "
          f"{r['setmean_worst_per_wind_design']:>11.2f} {str(r['setmean_fresh_worst_per_wind']):>10} |"
          f"{str(r['perwind_sims_to_first'] or '-'):>21} {r['perwind_regulation']:>11.2f} "
          f"{str(r['perwind_worst_per_wind_design']):>11} {str(r['perwind_heldout_worst']):>12} "
          f"{str(r['perwind_fresh_worst']):>10} "
          f"{('yes' if r['delivers_off_design'] else ('NO' if r['perwind_key'] else '-')):>6}  "
          f"{r['perwind_candidate']}")
print()
print("'design w/w' is the worst single-wind load term on the 14 design episodes, where the per-wind rule is a")
print("screen with two episodes per wind; 'heldout w/w' and 'fresh w/w' are the same quantity on 28 episodes")
print("each, which is the measurement. 'ships' is yes only when both of those hold within the tolerance.")

# ------------------------------------------------------------------------------------ the two tests, both rules
arms = [k for k in ARM_ORDER if any(r["proposer"] == k for r in rows)]
for rule, gain_k, sims_k, reg_k in (("the rule the search used (set mean)", "setmean_gain",
                                     "setmean_sims_to_first", "setmean_regulation"),
                                    ("the rule the project ships by, screened on the design set",
                                     "perwind_gain", "perwind_sims_to_first", "perwind_regulation"),
                                    ("the rule the project ships by, MEASURED off the design winds - what each "
                                     "run actually delivers", "delivers_off_design",
                                     "delivered_sims_to_first", "delivered_regulation")):
    print(f"\n=== {rule}")
    print(f"{'proposer':>9} {'runs':>5} {'with a gain':>12} {'sims to it: median':>19} {'values':>28} "
          f"{'regulation of the gains':>25}")
    per_arm = {}
    for arm in arms:
        rs = [r for r in rows if r["proposer"] == arm]
        sims = [r[sims_k] if r[sims_k] is not None else rs[0]["budget"] + 1 for r in rs]
        got = [r for r in rs if r[gain_k]]
        reg = [r[reg_k] for r in got]
        per_arm[arm] = {"n": len(rs), "k": len(got), "sims": sims, "reg": reg}
        print(f"{arm:>9} {len(rs):>5} {len(got):>7} of {len(rs):<2} {np.median(sims):>19.1f} "
              f"{str(sorted(sims)):>28} "
              f"{(f'{np.mean(reg):+.2f} ({np.std(reg, ddof=1):.2f})' if len(reg) > 1 else (f'{reg[0]:+.2f}' if reg else '-')):>25}")
    print(f"  censored runs enter the median and the test at {rows[0]['budget'] + 1} (budget + 1), tied last")

    if "llm" in per_arm:
        for other in [k for k in arms if k != "llm"]:
            A, B = per_arm["llm"], per_arm[other]
            p_sims = perm_p(A["sims"], B["sims"])
            t = [[A["k"], A["n"] - A["k"]], [B["k"], B["n"] - B["k"]]]
            print(f"  llm against {other:>6}: sims to the first gain one-sided exact p = {p_sims:.4f} "
                  f"({A['n']} against {B['n']}, smallest attainable "
                  f"{1 / len(list(itertools.combinations(range(A['n'] + B['n']), A['n']))):.4f})"
                  f"   |   runs with a gain {A['k']}/{A['n']} against {B['k']}/{B['n']}, Fisher two-sided "
                  f"p = {fisher_exact(t)[1]:.4f}, one-sided p = {fisher_exact(t, alternative='greater')[1]:.4f}")

# -------------------------------------------------------------- what the per-wind re-selections still need
print("\nThe per-wind rule re-selects a different tuning in a run than the set-mean rule did, so its winners")
print("have to be confirmed on winds they were not designed on. Verification cache of the selected tunings:")
need = []
for r in rows:
    if not r["perwind_key"]:
        continue
    have = {s: bool(glob.glob(f"{a.root}/_verify/cache/eval_{r['perwind_key']}_s{s}_*.json"))
            for s in ("3456", "78910")}
    tag = ("already verified" if all(have.values()) else
           "needs " + " and ".join(f"seeds {s}" for s, ok in have.items() if not ok))
    if not all(have.values()):
        need.append(r)
    print(f"  {r['proposer'] + '_r' + str(r['repeat']):>11}  {r['perwind_candidate']:>14}  {tag}")
print(f"{len(need)} of {sum(1 for r in rows if r['perwind_key'])} per-wind selections are not yet measured off "
      f"the design winds; each costs two evaluations of 28 episodes.")

if a.csv and rows:
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {a.csv}")
