"""Mechanism analysis of the fork-verified supervisors: what does the LLM propose that random
search does not? Reads decisions.jsonl of llm_fork / random_fork runs (every candidate's knobs,
its one-wave fork evaluation and which one was kept) and reports, per arm:

  * candidate-set quality  : mean / best fork-F of the K candidates per decision, and how often the
                             set contained at least one candidate better than the pre-decision F
  * decision yield         : F of the kept candidate minus the F measured just before the decision
  * exploration            : |log| knob change of the kept candidate, and the share of decisions
                             where nothing moved ("hold")
  * knob direction         : end-of-run knob values relative to the guard defaults

    python scripts/dev/fork_analysis.py "~/wtrl/exp/n1_llmfork_s*" "~/wtrl/exp/n1_randfork_s*"

Note on the built-in asymmetry: RandomCandidateSupervisor always puts "keep current knobs" first,
so one of its three candidates is a hold by construction; the LLM may or may not include one.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

DEFAULTS = {"lambda_load_R2": 1.0, "lambda_load_R3": 1.0, "w_power": 220.0, "w_speed": 20.0,
            "dbeta_max_R2": 0.05, "dbeta_max_R3": 0.05}


def log_move(a: dict, b: dict) -> float:
    """Mean |log ratio| over the knobs that moved — a scale-free 'how big was the step'."""
    v = [abs(math.log(max(b[k], 1e-9) / max(a[k], 1e-9))) for k in a if k in b and a[k] > 0]
    return float(np.mean(v)) if v else 0.0


def main():
    pats = sys.argv[1:] or ["~/wtrl/exp/n1_llmfork_s*", "~/wtrl/exp/n1_randfork_s*"]
    arms = defaultdict(lambda: defaultdict(list))
    finals = defaultdict(list)
    for pat in pats:
        for d in sorted(glob.glob(os.path.expanduser(pat))):
            f = os.path.join(d, "decisions.jsonl")
            if not os.path.exists(f):
                continue
            arm = re.sub(r"_s\d+$", "", os.path.basename(d.rstrip("/")))
            recs = [json.loads(l) for l in open(f, encoding="utf-8")]
            prev_F = None
            for r in recs:
                if r.get("type") == "eval" and "fit" in r:
                    F_before = r["fit"].get("F_measured", r["fit"]["F"])
                fk = r.get("fork")
                if not fk:
                    if r.get("type") == "eval":
                        prev_F = F_before
                    continue
                cands = fk["candidates"]
                Fs = [c["fit"]["F"] for c in cands]
                kept = cands[[c["i"] for c in cands].index(fk["chosen"])]
                arms[arm]["set_mean"].append(float(np.mean(Fs)))
                arms[arm]["set_best"].append(float(np.max(Fs)))
                arms[arm]["set_spread"].append(float(np.max(Fs) - np.min(Fs)))
                arms[arm]["n_strict_in_set"].append(sum(c["fit"].get("tier") == "strict" for c in cands))
                if prev_F is not None:
                    arms[arm]["yield"].append(kept["fit"]["F"] - prev_F)
                    arms[arm]["set_beats_prev"].append(float(np.max(Fs) > prev_F))
                arms[arm]["kept_move"].append(log_move(r["knobs"], kept["knobs"]))
                arms[arm]["kept_is_hold"].append(float(log_move(r["knobs"], kept["knobs"]) < 1e-9))
                arms[arm]["kept_tier_strict"].append(float(kept["fit"].get("tier") == "strict"))
                prev_F = kept["fit"]["F"]
            summ = os.path.join(d, "summary.json")
            if os.path.exists(summ):
                finals[arm].append(json.load(open(summ, encoding="utf-8"))["final_knobs"])

    def row(label, key, fmt="{:7.2f}"):
        cells = []
        for arm in sorted(arms):
            v = arms[arm][key]
            cells.append(fmt.format(float(np.mean(v))) if v else "    n/a")
        print(f"  {label:<44}" + "".join(f"{c:>16}" for c in cells))

    print("fork-decision mechanism analysis (per decision, averaged over runs)\n")
    print(f"  {'':<44}" + "".join(f"{a:>16}" for a in sorted(arms)))
    print(f"  {'decisions analysed':<44}" + "".join(f"{len(arms[a]['set_mean']):>16d}" for a in sorted(arms)))
    row("candidate-set mean fork F", "set_mean")
    row("candidate-set best fork F", "set_best")
    row("candidate-set spread (best - worst)", "set_spread")
    row("strict candidates per set (of 3)", "n_strict_in_set")
    row("set contained something better than before", "set_beats_prev", "{:7.2f}")
    row("decision yield  (kept F - F before)", "yield")
    row("kept candidate: mean |log| knob move", "kept_move", "{:7.3f}")
    row("kept candidate was a hold (fraction)", "kept_is_hold", "{:7.2f}")
    row("kept candidate was strict (fraction)", "kept_tier_strict", "{:7.2f}")

    print("\nfinal knobs vs the guard defaults (geometric mean of the ratio over runs):")
    print(f"  {'knob':<20}{'default':>10}" + "".join(f"{a:>16}" for a in sorted(finals)))
    for k, dv in DEFAULTS.items():
        cells = []
        for arm in sorted(finals):
            r = [max(fk[k], 1e-9) / dv for fk in finals[arm] if k in fk]
            cells.append(f"{math.exp(float(np.mean(np.log(r)))):>15.2f}x" if r else "            n/a")
        print(f"  {k:<20}{dv:>10g}" + "".join(cells))


if __name__ == "__main__":
    main()
