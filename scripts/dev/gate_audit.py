"""Audit the competence gate of `schedule_comp` runs: when was each curriculum entry applied, and
was that decision a guardrail rollback?

The original arm was contaminated: after a rollback the trainer replaces F with the historical best
(so the run's bookkeeping stays monotone), and the competence gate read that masked value, letting
aggressive entries fire straight into a crash. The fix (2026-09-02) makes the gate read `F_measured`.
This script shows the difference in the decision logs, at zero compute.

    python scripts/dev/gate_audit.py "~/wtrl/exp/sched2_comp_s*" "~/wtrl/exp/sched3_comp_s*"
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict


def audit(run_dir: str) -> dict:
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "decisions.jsonl"), encoding="utf-8")]
    applied, in_rollback, waits, gates, deadlines = 0, 0, 0, 0, 0
    for r in recs:
        if r.get("type") != "eval":
            continue
        why = str((r.get("proposal") or {}).get("rationale", ""))
        changed = bool(r.get("changed"))
        rolled = "rollback" in r
        if "schedule_comp entry" in why and changed:
            applied += 1
            in_rollback += rolled
            gates += "via gate" in why
            deadlines += "via max_wait" in why
        elif "waiting" in why:
            waits += 1
    return {"entries_applied": applied, "applied_inside_a_rollback": in_rollback,
            "applied_via_competence_gate": gates, "applied_via_max_wait_fallback": deadlines,
            "decisions_spent_waiting": waits, "rollbacks_total": sum("rollback" in r for r in recs)}


def main():
    pats = sys.argv[1:] or ["~/wtrl/exp/sched2_comp_s*", "~/wtrl/exp/sched3_comp_s*"]
    arms = defaultdict(list)
    for pat in pats:
        for d in sorted(glob.glob(os.path.expanduser(pat))):
            if os.path.exists(os.path.join(d, "decisions.jsonl")):
                arms[re.sub(r"_s\d+$", "", os.path.basename(d.rstrip("/")))].append((os.path.basename(d), audit(d)))
    cols = ["entries_applied", "applied_inside_a_rollback", "applied_via_competence_gate",
            "applied_via_max_wait_fallback", "decisions_spent_waiting", "rollbacks_total"]
    print(f"{'run':>18} " + " ".join(f"{c.replace('_', ' '):>28}" for c in cols))
    for arm, runs in sorted(arms.items()):
        for name, a in runs:
            print(f"{name:>18} " + " ".join(f"{a[c]:>28d}" for c in cols))
        tot = {c: sum(a[c] for _, a in runs) for c in cols}
        print(f"{arm + ' TOTAL':>18} " + " ".join(f"{tot[c]:>28d}" for c in cols))
        print()
    print("`applied_inside_a_rollback` > 0 means the gate fired on a masked F: those entries are the "
          "contamination. A clean run should show 0.")


if __name__ == "__main__":
    main()
