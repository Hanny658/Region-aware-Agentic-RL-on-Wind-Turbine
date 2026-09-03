"""Extract the accepted knob trajectory of a supervised run into a fixed schedule JSON
(for the `schedule` supervisor ablation):  python scripts/dev/extract_schedule.py <run> <out.json>"""
import json
import os
import sys

run = os.path.expanduser(sys.argv[1] if "/" in sys.argv[1] else f"~/wtrl/exp/{sys.argv[1]}")
out = os.path.expanduser(sys.argv[2])
sched = []
for ln in open(os.path.join(run, "decisions.jsonl"), encoding="utf-8"):
    d = json.loads(ln)
    gate = {}  # donor's pre-decision evaluation: the competence at which it made this change
    if isinstance(d.get("fit"), dict) and "F" in d["fit"]:
        gate = {"F_gate": d["fit"]["F"], "tier_gate": d["fit"].get("tier")}
    if d.get("accepted") and d.get("changed"):
        knobs = dict(d["knobs"])
        for kk, (a, b) in d["changed"].items():
            knobs[kk] = b
        sched.append({"episode": d["episode"], "knobs": knobs, **gate})
    if "fork" in d:
        c = d["fork"]["candidates"][d["fork"]["chosen"]]
        sched.append({"episode": d["episode"], "knobs": c["knobs"], **gate})
json.dump(sched, open(out, "w"), indent=1)
print(f"{len(sched)} schedule entries -> {out}")
for e in sched:
    print(e["episode"], {k: round(v, 4) for k, v in e["knobs"].items()})
