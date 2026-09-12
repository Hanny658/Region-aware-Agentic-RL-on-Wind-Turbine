"""Stage-3 summary under the metric-set objective: per run the supervisor-wind best J and the two
held-out J values with their four terms, per arm mean +- std, and seed-paired comparisons (paired
t and the exact sign-flip test, whose resolution floor is 2/2^n).

    python scripts/dev/j_table.py --arms jg2 jg1 jhp jrhp jrw --ref jg2
"""
import argparse
import glob
import json
import os
from itertools import product
from statistics import mean, pstdev

import numpy as np

EXP = os.path.expanduser(os.environ.get("WTRL_EXP", "~/wtrl/exp"))
TAGS = {"S3-S6": "eval_heldout_s3456_ckpt_best.json", "S7-S10": "eval_heldout2_s78910.json"}
LABEL = {"jg2": "guard-v2 (fixed hparams, reward v2)", "jg1": "guard-v1 (reward v1)", "jhp": "llm_hparam",
         "jrhp": "random_hparam", "jrw": "llm_reward", "jllm": "llm weights (v2)", "jmono": "mono (v2)",
         "jg3": "guard-v3 (reward v3, untuned)", "jwf3": "random_fork weights (v3)", "jwf3k4": "random_fork weights (v3, 4-knob bug)",
         "jg3t": "guard-v3 tuned default", "jhp3t": "llm_hparam (v3, tuned)", "jrhp3t": "random_hparam (v3, tuned)",
         "jrw3t": "llm_reward (v3, tuned)", "jg3L3": "guard-v3 tuned, lambda_tower 3", "jg3L10": "guard-v3 tuned, lambda_tower 10",
         "jg3L30": "guard-v3 tuned, lambda_tower 30"}


def perm_p(d):
    obs = abs(mean(d))
    return sum(1 for s in product((1, -1), repeat=len(d)) if abs(mean(a * b for a, b in zip(s, d))) >= obs - 1e-12) / 2 ** len(d)


def paired_t(d):
    from scipy import stats
    return float(stats.ttest_1samp(d, 0.0).pvalue) if len(d) > 1 and pstdev(d) > 0 else float("nan")


ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=["jg2", "jg1", "jhp", "jrhp", "jrw"])
ap.add_argument("--ref", default="jg2")
ap.add_argument("--seeds", nargs="+", type=int, default=None)
ap.add_argument("--csv", default=None, help="also write per-run rows + arm aggregates + paired tests to this CSV")
a = ap.parse_args()
csv_rows = []

runs = {}   # (arm, seed) -> dict
for arm in a.arms:
    for d in sorted(glob.glob(f"{EXP}/{arm}_s*")):
        seed = int(d.rsplit("_s", 1)[1])
        if a.seeds and seed not in a.seeds:
            continue
        if not os.path.exists(f"{d}/summary.json"):
            continue
        row = {"train": json.load(open(f"{d}/summary.json"))["best_F"]}
        for k, fn in TAGS.items():
            if os.path.exists(f"{d}/{fn}"):
                row[k] = json.load(open(f"{d}/{fn}", encoding="utf-8"))
        runs[(arm, seed)] = row

print(f"{'run':>9} {'J_S1S2':>7} | {'J_S3-6':>7} {'P':>6} {'w':>6} {'T':>6} {'B':>6} {'E%':>5} | {'J_S7-10':>7} {'P':>6} {'w':>6} {'T':>6} {'B':>6} {'E%':>5}")
for (arm, seed), r in sorted(runs.items(), key=lambda kv: (a.arms.index(kv[0][0]), kv[0][1])):
    cells = [f"{arm + '_s' + str(seed):>9} {r['train']:7.2f}"]
    for k in TAGS:
        j = r.get(k)
        cells.append(f"{j['J']:7.2f} {j['J_power_mse_red_pct']:6.1f} {j['J_gen_speed_mse_red_pct']:6.1f} "
                     f"{j['J_TwrBsMyt_DEL_red_pct']:6.1f} {j['J_RootMyc1_DEL_red_pct']:6.1f} {j['energy_loss_pct']:5.2f}" if j else f"{'-':>41}")
    print(" | ".join(cells))
    rec = {"kind": "run", "run": f"{arm}_s{seed}", "arm": arm, "label": LABEL.get(arm, arm), "seed": seed, "J_train_S1S2": round(r["train"], 3)}
    for k in TAGS:
        j = r.get(k)
        if j:
            rec |= {f"J_{k}": round(j["J"], 3), f"P_{k}": round(j["J_power_mse_red_pct"], 2), f"w_{k}": round(j["J_gen_speed_mse_red_pct"], 2),
                    f"T_{k}": round(j["J_TwrBsMyt_DEL_red_pct"], 2), f"B_{k}": round(j["J_RootMyc1_DEL_red_pct"], 2),
                    f"E_{k}": round(j["energy_loss_pct"], 3), f"F_{k}": round(j["F"], 3)}
    csv_rows.append(rec)

print(f"\n{'arm':>34} {'n':>2} {'J_S1S2':>12} {'J_S3-S6':>12} {'J_S7-S10':>12} {'P_S3-6':>7} {'w_S3-6':>7} {'T_S3-6':>7} {'B_S3-6':>7}")
for arm in a.arms:
    rs = [r for (ar, _), r in runs.items() if ar == arm]
    if not rs:
        continue
    def ms(vals):
        v = [x for x in vals if x is not None]
        return f"{mean(v):6.2f}±{pstdev(v):4.2f}" if v else f"{'-':>11}"
    def term(key):
        v = [r["S3-S6"][key] for r in rs if "S3-S6" in r]
        return f"{mean(v):7.1f}" if v else f"{'-':>7}"
    csv_rows.append({"kind": "arm", "arm": arm, "label": LABEL.get(arm, arm), "n": len(rs),
                     "J_train_S1S2": ms([r["train"] for r in rs]).strip(),
                     "J_S3-S6": ms([r["S3-S6"]["J"] for r in rs if "S3-S6" in r]).strip(),
                     "J_S7-S10": ms([r["S7-S10"]["J"] for r in rs if "S7-S10" in r]).strip(),
                     "P_S3-S6": term("J_power_mse_red_pct").strip(), "w_S3-S6": term("J_gen_speed_mse_red_pct").strip(),
                     "T_S3-S6": term("J_TwrBsMyt_DEL_red_pct").strip(), "B_S3-S6": term("J_RootMyc1_DEL_red_pct").strip()})
    print(f"{LABEL.get(arm, arm):>34} {len(rs):2d} {ms([r['train'] for r in rs]):>12} "
          f"{ms([r['S3-S6']['J'] for r in rs if 'S3-S6' in r]):>12} {ms([r['S7-S10']['J'] for r in rs if 'S7-S10' in r]):>12} "
          f"{term('J_power_mse_red_pct')} {term('J_gen_speed_mse_red_pct')} {term('J_TwrBsMyt_DEL_red_pct')} {term('J_RootMyc1_DEL_red_pct')}")

print(f"\nseed-paired comparisons (difference of J; exact floor = 2/2^n):")
pairs = [(x, a.ref) for x in a.arms if x != a.ref] + [("jhp", "jrhp")]
for x, y in pairs:
    for k in TAGS:
        d = [runs[(x, s)][k]["J"] - runs[(y, s)][k]["J"] for (ar, s) in runs if ar == x and (y, s) in runs
             and k in runs[(x, s)] and k in runs[(y, s)]]
        if len(d) < 2:
            continue
        wins = sum(1 for v in d if v > 0)
        print(f"  {x:>5} - {y:<5} {k:>6}: mean {mean(d):+6.2f}  wins {wins}/{len(d)}  exact p {perm_p(d):.3f}  paired-t p {paired_t(d):.3f}  diffs {[round(v, 1) for v in d]}")
        csv_rows.append({"kind": "paired", "comparison": f"{x} - {y}", "wind_set": k, "n": len(d), "mean_diff": round(mean(d), 3),
                         "wins": wins, "exact_p": round(perm_p(d), 4), "paired_t_p": round(paired_t(d), 4), "diffs": " ".join(f"{v:.2f}" for v in d)})
if a.csv:
    import csv as _csv
    keys = []
    for r in csv_rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.expanduser(a.csv), "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(csv_rows)
    print("->", a.csv)
