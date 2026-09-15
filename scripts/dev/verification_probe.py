"""How reliable is the supervision loop's verification? Zero-simulation probe on the J-era run artefacts.

(1) Checkpoint selection vs held-out, run level. For every finished J-era run: the objective J and its four
    terms on the supervisor winds (TurbSim seeds 1-2) at the selected best checkpoint, against the same
    checkpoint on the two held-out sets (seeds 3-6, 7-10). Spearman rho per base controller (GSPI / MPC),
    overall and per term; the agreement between the two held-out sets is the ceiling a 4-seed validation set
    reaches. The gap (supervisor-wind J - held-out J) is the selection optimism.
(2) Fork verification. For every fork decision: the chosen candidate's J after its 30-episode fork, against
    the same policy's J at the next evaluation (~30 episodes later, before any rollback masking). Winner's
    curse = mean drop; persistence = Spearman over decisions; per term. Predicted gain (fork J - J at the
    decision) against realised gain (next J - J at the decision).
(3) Fork ranking. Margin between the chosen candidate and the runner-up, against the realised gain.

    ~/wtrl/run.sh python scripts/dev/verification_probe.py [--exp ~/wtrl/exp] [--csv docs/tables/verification_probe.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from statistics import mean, median

import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser(os.environ.get("WTRL_EXP", "~/wtrl/exp")))
ap.add_argument("--csv", default=None)
a = ap.parse_args()
E = os.path.expanduser(a.exp)
TERMS = {"J": "J", "power": "J_power_mse_red_pct", "speed": "J_gen_speed_mse_red_pct",
         "tower": "J_TwrBsMyt_DEL_red_pct", "blade": "J_RootMyc1_DEL_red_pct"}
HO = {"S3-6": "eval_heldout_s3456_ckpt_best.json", "S7-10": "eval_heldout2_s78910.json"}


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return float("nan"), float("nan"), int(ok.sum())
    r = stats.spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue), int(ok.sum())


def fmt(r):
    return f"rho {r[0]:+.2f} (p {r[1]:.3f}, n {r[2]})" if np.isfinite(r[0]) else f"n {r[2]}"


# ---------------------------------------------------------------- collect runs
runs = []
for d in sorted(glob.glob(f"{E}/*_s[0-9]*")):
    try:
        s = json.load(open(f"{d}/summary.json"))
    except (FileNotFoundError, json.JSONDecodeError):
        continue
    if s.get("objective") != "J" or not all(os.path.exists(f"{d}/{f}") for f in HO.values()):
        continue
    cfg = json.load(open(f"{d}/config.json"))
    arm = os.path.basename(d).rsplit("_s", 1)[0]
    rows = list(csv.DictReader(open(f"{d}/evals.csv")))
    be = int(s["best_episode"])
    cand = [r for r in rows if int(r["episode"]) == be and r["J"] not in ("", "nan")]
    if not cand:
        continue
    sup = min(cand, key=lambda r: abs(float(r["J"]) - float(s["best_F"])))
    ho = {k: json.load(open(f"{d}/{f}")) for k, f in HO.items()}
    runs.append({"run": os.path.basename(d), "arm": arm, "base": cfg.get("base", "gspi"),
                 "sup": {t: float(sup[c]) for t, c in TERMS.items()},
                 "ho": {k: {t: float(v[c]) for t, c in TERMS.items()} for k, v in ho.items()},
                 "decisions": [json.loads(l) for l in open(f"{d}/decisions.jsonl")] if os.path.exists(f"{d}/decisions.jsonl") else []})

out_rows = []
print(f"J-era runs with both held-out sets: {len(runs)}")
print("\n(1) checkpoint selection: supervisor winds (seeds 1-2) vs held-out, Spearman over runs")
for base in ("gspi", "mpc"):
    R = [r for r in runs if r["base"] == base]
    if len(R) < 5:
        continue
    print(f"  base {base}  ({len(R)} runs, {len({r['arm'] for r in R})} arms)")
    for t in TERMS:
        s36 = rho([r["sup"][t] for r in R], [r["ho"]["S3-6"][t] for r in R])
        s710 = rho([r["sup"][t] for r in R], [r["ho"]["S7-10"][t] for r in R])
        ceil = rho([r["ho"]["S3-6"][t] for r in R], [r["ho"]["S7-10"][t] for r in R])
        gap = mean(r["sup"][t] - (r["ho"]["S3-6"][t] + r["ho"]["S7-10"][t]) / 2 for r in R)
        print(f"    {t:>6}: sup->S3-6 {fmt(s36)} | sup->S7-10 {fmt(s710)} | S3-6<->S7-10 {fmt(ceil)} | optimism {gap:+.2f}")
        out_rows.append({"analysis": "selection", "base": base, "term": t, "rho_sup_S36": s36[0], "rho_sup_S710": s710[0],
                         "rho_S36_S710": ceil[0], "n": s36[2], "optimism_mean": gap})
    # within-arm: remove each arm's mean before ranking (arm differences inflate the pooled rho)
    def dm(key_fn):
        arms = {}
        for r in R:
            arms.setdefault(r["arm"], []).append(r)
        out = {}
        for arm, rs in arms.items():
            if len(rs) < 3:
                continue
            m = mean(key_fn(x) for x in rs)
            for x in rs:
                out[x["run"]] = key_fn(x) - m
        return out
    for t in ("J", "tower", "speed"):
        a1 = dm(lambda x: x["sup"][t]); a2 = dm(lambda x: (x["ho"]["S3-6"][t] + x["ho"]["S7-10"][t]) / 2)
        b1 = dm(lambda x: x["ho"]["S3-6"][t]); b2 = dm(lambda x: x["ho"]["S7-10"][t])
        ks = sorted(a1)
        w1 = rho([a1[k] for k in ks], [a2[k] for k in ks]); w2 = rho([b1[k] for k in ks], [b2[k] for k in ks])
        print(f"    within-arm {t:>5}: sup -> mean held-out {fmt(w1)} | S3-6 <-> S7-10 {fmt(w2)}")
        out_rows.append({"analysis": "selection_within_arm", "base": base, "term": t, "rho_sup_heldout": w1[0], "rho_S36_S710": w2[0], "n": w1[2]})

# ---------------------------------------------------------------- (2)(3) forks
print("\n(2) fork verification: chosen candidate after its 30-episode fork vs the same policy at the next evaluation")
F = []
for r in runs:
    D = r["decisions"]
    for i, dec in enumerate(D):
        fk = dec.get("fork")
        if not fk or not isinstance(fk.get("chosen"), int) or i + 1 >= len(D):
            continue
        cands = fk.get("candidates") or []
        ch = fk["chosen"]
        if ch >= len(cands) or not cands[ch].get("fit") or "J" not in cands[ch]["fit"]:
            continue
        nxt = D[i + 1]["fit"]
        pre = dec["fit"]
        cf = cands[ch]["fit"]
        others = sorted((c["fit"]["J"] for j, c in enumerate(cands) if j != ch and c.get("fit") and "J" in c["fit"]), reverse=True)
        F.append({"run": r["run"], "base": r["base"], "arm": r["arm"],
                  "sup_kind": "llm" if r["arm"].startswith(("jrw", "mrw", "jhp", "jcb", "jlf")) else "random",
                  "pre": {t: float(pre.get("J_measured", pre["J"]) if t == "J" else pre.get(c, float("nan"))) for t, c in TERMS.items()},
                  "fork": {t: float(cf.get(c, float("nan"))) for t, c in TERMS.items()},
                  "next": {t: float(nxt.get("J_measured", nxt["J"]) if t == "J" else nxt.get(c, float("nan"))) for t, c in TERMS.items()},
                  "margin": (cf["J"] - others[0]) if others else float("nan"),
                  "spread": (cf["J"] - others[-1]) if others else float("nan"),
                  "next_rolled_back": "rollback" in D[i + 1]})
print(f"  fork decisions with a following evaluation: {len(F)} (GSPI base {sum(f['base'] == 'gspi' for f in F)}, MPC base {sum(f['base'] == 'mpc' for f in F)})")
for base in ("gspi", "mpc", "all"):
    G = [f for f in F if base == "all" or f["base"] == base]
    if len(G) < 5:
        continue
    print(f"  base {base} ({len(G)} decisions; next evaluation triggered a rollback in {sum(f['next_rolled_back'] for f in G)})")
    for t in TERMS:
        drop = [f["fork"][t] - f["next"][t] for f in G]
        pers = rho([f["fork"][t] for f in G], [f["next"][t] for f in G])
        pred = [f["fork"][t] - f["pre"][t] for f in G]
        real = [f["next"][t] - f["pre"][t] for f in G]
        pr = rho(pred, real)
        sign = np.mean([(p > 0) == (q > 0) for p, q in zip(pred, real) if np.isfinite(p) and np.isfinite(q)])
        print(f"    {t:>6}: winner's curse mean {mean(drop):+.2f} median {median(drop):+.2f} | fork->next {fmt(pers)} | "
              f"predicted vs realised gain {fmt(pr)}, sign agreement {sign:.2f}")
        out_rows.append({"analysis": "fork", "base": base, "term": t, "curse_mean": mean(drop), "curse_median": median(drop),
                         "rho_fork_next": pers[0], "rho_pred_real": pr[0], "sign_agree": sign, "n": pers[2]})
    mg = rho([f["margin"] for f in G], [f["next"]["J"] - f["pre"]["J"] for f in G])
    sp = rho([f["spread"] for f in G], [f["next"]["J"] - f["pre"]["J"] for f in G])
    print(f"    (3) margin chosen - runner-up vs realised gain {fmt(mg)}; best - worst candidate {fmt(sp)}; "
          f"median margin {median(f['margin'] for f in G if np.isfinite(f['margin'])):.2f}")
    out_rows.append({"analysis": "fork_margin", "base": base, "term": "J", "rho_margin_real": mg[0], "rho_spread_real": sp[0], "n": mg[2]})

# control: runs without forks (fixed reward) -- how often does the next evaluation collapse anyway?
print()
print("  control: consecutive evaluations of fixed-reward runs (no fork)")
for base in ("gspi", "mpc"):
    ch, rb = [], 0
    for r in runs:
        if r["base"] != base or any("fork" in d for d in r["decisions"]):
            continue
        D = r["decisions"]
        for i in range(1, len(D) - 1):
            j0 = float(D[i]["fit"].get("J_measured", D[i]["fit"]["J"])); j1 = float(D[i + 1]["fit"].get("J_measured", D[i + 1]["fit"]["J"]))
            ch.append(j0 - j1); rb += "rollback" in D[i + 1]
    if ch:
        print(f"    base {base}: {len(ch)} consecutive pairs, drop mean {mean(ch):+.2f} median {median(ch):+.2f}, next evaluation rolled back in {rb} ({rb / len(ch):.0%})")
        out_rows.append({"analysis": "control_no_fork", "base": base, "term": "J", "curse_mean": mean(ch), "curse_median": median(ch), "rollback_frac": rb / len(ch), "n": len(ch)})
for base in ("gspi", "mpc"):
    G = [f for f in F if f["base"] == base]
    if G:
        print(f"    forks, base {base}: next evaluation rolled back in {sum(f['next_rolled_back'] for f in G)} of {len(G)} ({sum(f['next_rolled_back'] for f in G) / len(G):.0%})")

for kind in ("llm", "random"):
    G = [f for f in F if f["sup_kind"] == kind]
    if len(G) >= 5:
        pr = rho([f["fork"]["J"] - f["pre"]["J"] for f in G], [f["next"]["J"] - f["pre"]["J"] for f in G])
        print(f"  proposer {kind}: {len(G)} decisions, winner's curse {mean(f['fork']['J'] - f['next']['J'] for f in G):+.2f}, predicted vs realised gain {fmt(pr)}")

if a.csv:
    keys = []
    for r in out_rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.expanduser(a.csv), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out_rows)
    print("->", a.csv)
