"""Re-score every existing held-out evaluation under the baseline paper's metric set.

The evaluation JSONs already carry the four paper metrics (power MSE, generator-speed MSE,
tower-base DEL, blade-root DEL — all as % reduction vs the paired GSPI) plus the energy loss, so a
candidate composite objective can be previewed on the existing policies at zero cost. This does NOT
replace a re-run: the policies were trained and checkpoint-selected under F (tower-DEL priority), so
the preview shows how the *ranking* moves, not what a J-trained policy would reach.

    python scripts/dev/rescore_metric_set.py [--w 1 1 1 1] [--k_energy 20] [--clip 100]

J = sum_i w_i * m_i / sum_i w_i  -  k_energy * max(0, energy_loss% - 1),  m_i clipped to +-clip.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

METRICS = ("power_mse_red_pct", "gen_speed_mse_red_pct", "TwrBsMyt_DEL_red_pct", "RootMyc1_DEL_red_pct")
ARMS = [  # (label, glob, eval file)
    ("guard / broken critic", "~/wtrl/exp/tq_off_s*", "eval_heldout_s3456_ckpt_best.json"),
    ("mono / broken", "~/wtrl/exp/n1_mono_s*", "eval_heldout_s3456_ckpt_best.json"),
    ("llm weights / broken", "~/wtrl/exp/n1_llmfork_s*", "eval_heldout_s3456_ckpt_best.json"),
    ("random_fork / broken", "~/wtrl/exp/n1_randfork_s*", "eval_heldout_s3456_ckpt_best.json"),
    ("schedule / broken", "~/wtrl/exp/sched2_ep_s*", "eval_heldout_s3456_ckpt_best.json"),
    ("guard / fixed critic", "~/wtrl/exp/vnorm_s*", "eval_heldout_s3456_ckpt_best.json"),
    ("llm weights / fixed", "~/wtrl/exp/vnorm_llm_s*", "eval_heldout_s3456_ckpt_best.json"),
    ("llm hparam / fixed", "~/wtrl/exp/vnorm_hp_s*", "eval_heldout_s3456_ckpt_best.json"),
    ("llm reward / fixed", "~/wtrl/exp/vnorm_rw_s*", "eval_heldout_s3456_ckpt_best.json"),
    ("LPV-MPC (F-tuned)", "~/wtrl/exp/mpc", "eval_heldoutW_N20q1r0.02qt0w0.35.json"),
    ("GSPI + tower damper (F-tuned)", "~/wtrl/exp/gspi_td", "eval_heldout_s3456.json"),
]


def J(j: dict, w, k_e: float, clip: float) -> float:
    vals = []
    for m in METRICS:
        v = j.get(m)
        v = 0.0 if v is None or v != v else max(-clip, min(clip, float(v)))
        vals.append(v)
    comp = sum(wi * v for wi, v in zip(w, vals)) / sum(w)
    return comp - k_e * max(0.0, float(j.get("energy_loss_pct", 0.0)) - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", nargs=4, type=float, default=[1, 1, 1, 1],
                    help="weights on (power MSE, speed MSE, tower DEL, blade DEL)")
    ap.add_argument("--k_energy", type=float, default=20.0)
    ap.add_argument("--clip", type=float, default=100.0)
    a = ap.parse_args()
    print(f"J = weighted mean of the four % reductions, w = {a.w}, energy penalty {a.k_energy}/% over 1 %, "
          f"terms clipped to +-{a.clip:g}\n")
    print(f"{'arm':>30} {'n':>2} {'F':>7} {'J':>7} | {'pwrMSE':>7} {'spdMSE':>7} {'twrDEL':>7} {'bldDEL':>7} {'Eloss':>6} | per-seed J")
    rows = []
    for label, pat, evf in ARMS:
        vals = defaultdict(list)
        per = {}
        for d in sorted(glob.glob(os.path.expanduser(pat))):
            f = os.path.join(d, evf)
            if not os.path.exists(f):
                continue
            j = json.load(open(f, encoding="utf-8"))
            s = re.search(r"_s(\d+)$", os.path.basename(d))
            per[s.group(1) if s else "-"] = J(j, a.w, a.k_energy, a.clip)
            vals["F"].append(j.get("F_tol2", j["F"]))
            for m in METRICS + ("energy_loss_pct",):
                v = j.get(m)
                vals[m].append(0.0 if v is None or v != v else float(v))
        if not per:
            continue
        n = len(per)
        mean = lambda k: st.mean(vals[k]) if vals[k] else float("nan")
        rows.append((label, n, mean("F"), st.mean(per.values()), *(mean(m) for m in METRICS), mean("energy_loss_pct"), per))
    for label, n, F, Jm, p, s, t, b, e, per in sorted(rows, key=lambda r: -r[3]):
        print(f"{label:>30} {n:2d} {F:7.2f} {Jm:7.2f} | {p:7.2f} {s:7.2f} {t:7.2f} {b:7.2f} {e:6.2f} | "
              + " ".join(f"s{k}:{v:.1f}" for k, v in sorted(per.items())))
    print("\nranking by F (old objective):")
    print("  " + "  >  ".join(f"{r[0]} ({r[2]:.1f})" for r in sorted(rows, key=lambda r: -r[2])[:6]))
    print("ranking by J (paper metric set):")
    print("  " + "  >  ".join(f"{r[0]} ({r[3]:.1f})" for r in sorted(rows, key=lambda r: -r[3])[:6]))


if __name__ == "__main__":
    main()
