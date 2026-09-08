"""B3: paired statistics for the arm comparisons, reported the way the manuscript must report them.

For each comparison it prints the per-seed values, the paired difference, a paired t-test, the
EXACT two-sided sign-flip permutation test (2^n sign assignments; at n = 5 its smallest attainable
p is 0.0625 — a resolution floor, not evidence of "not significant"), Cohen's d_z and the number of
seeds an 80 %-power replication would need at the observed effect.

    python scripts/dev/stats_table.py                                   # the standard set
    python scripts/dev/stats_table.py --a "~/wtrl/exp/tq_off_s*" --b "~/wtrl/exp/n1_mono_s*" \
                                      --label "guard vs mono"
Values are F_tol2 of eval_heldout_s3456_ckpt_best.json (= F_strict wherever the run is strict).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from itertools import product

import numpy as np

KEY = "F_tol2"
EVALS = ("eval_heldout_s3456_ckpt_best.json", "eval_paper_s3456.json")

DEFAULT_COMPARISONS = [
    ("spec+guard vs mono (F1: does region specialisation win?)", "~/wtrl/exp/tq_off_s*", "~/wtrl/exp/n1_mono_s*"),
    ("llm_fork vs random_fork (F5: does the proposer matter?)", "~/wtrl/exp/n1_llmfork_s*", "~/wtrl/exp/n1_randfork_s*"),
    ("llm_fork vs llm single-proposal (B1: does verification matter?)", "~/wtrl/exp/n1_llmfork_s*", "~/wtrl/exp/n1_llmsingle_s*"),
    ("schedule replay vs guard (does supervision help at all?)", "~/wtrl/exp/sched2_ep_s*", "~/wtrl/exp/tq_off_s*"),
    ("pitch only vs +torque residual (roadmap 15)", "~/wtrl/exp/tq_off_s*", "~/wtrl/exp/tq_on3_s*"),
]


def load_arm(pattern: str) -> dict[int, float]:
    out = {}
    for d in sorted(glob.glob(os.path.expanduser(pattern))):
        m = re.search(r"_s(\d+)$", os.path.basename(d.rstrip("/")))
        if not m:
            continue
        f = next((os.path.join(d, n) for n in EVALS if os.path.exists(os.path.join(d, n))), None)
        if f:
            out[int(m.group(1))] = float(json.load(open(f, encoding="utf-8"))[KEY])
    return out


def perm_p(d: list[float]) -> float:
    """Exact two-sided sign-flip permutation test on the paired differences."""
    obs = abs(sum(d))
    tot = hit = 0
    for signs in product((1, -1), repeat=len(d)):
        tot += 1
        hit += abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-12
    return hit / tot


def report(label: str, pa: str, pb: str):
    A, B = load_arm(pa), load_arm(pb)
    seeds = sorted(set(A) & set(B))
    print(f"\n### {label}")
    print(f"    A = {pa}   B = {pb}")
    if len(seeds) < 2:
        missing_a = sorted(set(B) - set(A))
        missing_b = sorted(set(A) - set(B))
        print(f"    NOT ENOUGH PAIRED SEEDS (A has {sorted(A)}, B has {sorted(B)})"
              + (f"; A missing {missing_a}" if missing_a else "")
              + (f"; B missing {missing_b}" if missing_b else ""))
        return
    a = np.array([A[s] for s in seeds], float)
    b = np.array([B[s] for s in seeds], float)
    d = a - b
    n = len(seeds)
    print("    seed:      " + "  ".join(f"s{s}" for s in seeds))
    print("    A:         " + "  ".join(f"{v:5.2f}" for v in a) + f"   mean {a.mean():6.2f} ± {a.std(ddof=1):.2f}")
    print("    B:         " + "  ".join(f"{v:5.2f}" for v in b) + f"   mean {b.mean():6.2f} ± {b.std(ddof=1):.2f}")
    print("    A - B:     " + "  ".join(f"{v:5.2f}" for v in d) + f"   mean {d.mean():+6.2f} ± {d.std(ddof=1):.2f}")
    sd = d.std(ddof=1)
    if sd == 0:
        print("    (zero variance in the differences)")
        return
    t = d.mean() / (sd / math.sqrt(n))
    try:
        from scipy import stats
        p_t = float(2 * stats.t.sf(abs(t), n - 1))
    except Exception:  # noqa: BLE001
        p_t = float("nan")
    dz = d.mean() / sd
    n80 = math.ceil((2.8 / dz) ** 2) if dz else float("inf")   # (z_{1-a/2}+z_{1-b})^2 / dz^2, two-sided 5 %/80 %
    pp = perm_p(list(d))
    floor = 2.0 / (2 ** n)
    print(f"    paired t({n-1}) = {t:+.2f}, p = {p_t:.4f} | exact sign-flip permutation p = {pp:.4f} "
          f"(floor at n={n}: {floor:.4f}) | d_z = {dz:+.2f} | n for 80 % power ≈ {n80}")
    if abs(pp - floor) < 1e-12:
        print("    NOTE: the permutation p equals its resolution floor — report it as '≤ this value at n = %d'." % n)


def main():
    global KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--a"), ap.add_argument("--b"), ap.add_argument("--label", default="A vs B")
    ap.add_argument("--key", default=KEY, help="field of the eval json (default F_tol2)")
    args = ap.parse_args()
    KEY = args.key
    print(f"paired statistics on {KEY} of {EVALS[0]} (held-out S3-S6, best checkpoint)")
    if args.a and args.b:
        report(args.label, args.a, args.b)
    else:
        for label, pa, pb in DEFAULT_COMPARISONS:
            report(label, pa, pb)
    print("\nReporting rule: at n = 5 the exact permutation test cannot go below p = 0.0625; quote both "
          "the parametric and the exact p, and state the resolution limit.")


if __name__ == "__main__":
    main()
