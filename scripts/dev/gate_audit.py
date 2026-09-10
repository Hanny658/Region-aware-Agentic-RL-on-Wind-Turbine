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
    seen_F = []
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
            # the rationale prints the F the SUPERVISOR used; before the 2026-09-02 fix that was
            # the rollback-masked historical best, afterwards the measured value. `rec["fit"]` is
            # snapshotted before masking, so the rationale is the only place the difference shows.
            m = re.search(r"F (-?[\d.]+) vs gate (-?[\d.]+)", why)
            if m and rolled:
                seen_F.append((float(m.group(1)), float(m.group(2)), "max_wait" in why))
        elif "waiting" in why:
            waits += 1
    # an entry firing during a rollback decision is only contamination if the gate was actually
    # satisfied by a masked (too high) F; a negative gate or the unconditional max_wait fallback
    # firing there is the specified behaviour
    suspicious = sum(1 for F, g, mw in seen_F if not mw and F >= g and g > 0)
    return {"entries_applied": applied, "applied_inside_a_rollback": in_rollback,
            "of_those_actually_suspicious": suspicious,
            "applied_via_competence_gate": gates, "applied_via_max_wait_fallback": deadlines,
            "decisions_spent_waiting": waits, "rollbacks_total": sum("rollback" in r for r in recs)}


def main():
    pats = sys.argv[1:] or ["~/wtrl/exp/sched2_comp_s*", "~/wtrl/exp/sched3_comp_s*"]
    arms = defaultdict(list)
    for pat in pats:
        for d in sorted(glob.glob(os.path.expanduser(pat))):
            if os.path.exists(os.path.join(d, "decisions.jsonl")):
                arms[re.sub(r"_s\d+$", "", os.path.basename(d.rstrip("/")))].append((os.path.basename(d), audit(d)))
    cols = ["entries_applied", "applied_inside_a_rollback", "of_those_actually_suspicious",
            "applied_via_competence_gate", "applied_via_max_wait_fallback",
            "decisions_spent_waiting", "rollbacks_total"]
    print(f"{'run':>18} " + " ".join(f"{c.replace('_', ' '):>28}" for c in cols))
    for arm, runs in sorted(arms.items()):
        for name, a in runs:
            print(f"{name:>18} " + " ".join(f"{a[c]:>28d}" for c in cols))
        tot = {c: sum(a[c] for _, a in runs) for c in cols}
        print(f"{arm + ' TOTAL':>18} " + " ".join(f"{tot[c]:>28d}" for c in cols))
        print()
    print("`of_those_actually_suspicious` counts entries whose POSITIVE competence gate was")
    print("satisfied during a rollback decision, i.e. the gate read a rollback-masked F. A clean")
    print("run shows 0. Entries firing there through the unconditional max_wait fallback, or")
    print("through a negative (vacuous) gate, are the specified behaviour and not contamination.")


if __name__ == "__main__":
    main()
