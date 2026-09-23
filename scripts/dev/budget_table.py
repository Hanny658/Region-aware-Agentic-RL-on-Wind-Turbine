"""Simulation budget actually consumed by each arm (2026-09-23), from the per-episode training logs.

A referee objected that "300 episodes per run", "a decision every 30 episodes" and "five decisions per run" cannot all
be true at once, and that the agent arms may therefore be getting a larger search budget than their controls. The
training log distinguishes main-trajectory episodes from fork episodes (the `fork` column of episodes.csv), and the
supervisor's own evaluations are in evals.csv, so the question is answerable exactly rather than by arithmetic on the
flags. Reports, per arm: main episodes, fork episodes, supervisor evaluation episodes, decisions, and the total
simulated seconds, which is the number every comparison of methods should be read against.

    ~/wtrl/run.sh python scripts/dev/budget_table.py [--runs trwCwG3t_s? mrwCwG3t_s? mgCwG3t_s?] [--csv ...]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser("~/wtrl/exp"))
ap.add_argument("--runs", nargs="+", default=["trwCwG3t_s?", "mrwCwG3t_s?", "mgCwG3t_s?", "mrrCwR3t_s?"])
ap.add_argument("--csv", default=None)
a = ap.parse_args()

rows = []
for pat in a.runs:
    for d in sorted(glob.glob(f"{a.exp}/{pat}")):
        ep = f"{d}/episodes.csv"
        if not os.path.exists(ep):
            continue
        main = fork = 0
        ep_s = 0.0
        cfg = json.load(open(f"{d}/config.json")) if os.path.exists(f"{d}/config.json") else {}
        L = float(cfg.get("episode_s", 150.0))
        for r in csv.DictReader(open(ep)):
            if (r.get("fork") or "").strip():
                fork += 1
            else:
                main += 1
            ep_s += L
        ev = 0
        if os.path.exists(f"{d}/evals.csv"):
            ev = sum(1 for _ in csv.DictReader(open(f"{d}/evals.csv")))
        dec = 0
        if os.path.exists(f"{d}/decisions.jsonl"):
            dec = sum(1 for line in open(f"{d}/decisions.jsonl") if json.loads(line).get("type") != "init")
        n_eval_eps = len(cfg.get("eval_seeds", []) or [1, 2]) * len(cfg.get("means", []) or [1])
        ev_s = ev * n_eval_eps * L
        rows.append({"run": os.path.basename(d), "supervisor": cfg.get("supervisor", "?"),
                     "episode_s": L, "main_episodes": main, "fork_episodes": fork, "decisions": dec,
                     "supervisor_evaluations": ev, "evaluation_episodes": ev * n_eval_eps,
                     "total_simulated_h": round((ep_s + ev_s) / 3600.0, 2)})

w = max((len(r["run"]) for r in rows), default=3)
print(f"{'run':>{w}} {'supervisor':>13} {'main':>6} {'fork':>6} {'dec':>4} {'evals':>6} {'eval eps':>9} {'total sim h':>12}")
for r in rows:
    print(f"{r['run']:>{w}} {r['supervisor']:>13} {r['main_episodes']:>6} {r['fork_episodes']:>6} {r['decisions']:>4}"
          f" {r['supervisor_evaluations']:>6} {r['evaluation_episodes']:>9} {r['total_simulated_h']:>12.2f}")

by_sup: dict[str, list] = {}
for r in rows:
    by_sup.setdefault(r["supervisor"], []).append(r)
print("\nper supervisor (mean over runs): main / fork / evaluation episodes and total simulated hours")
for sup, rs in by_sup.items():
    f = lambda k: sum(r[k] for r in rs) / len(rs)  # noqa: E731
    print(f"  {sup:>13}  n={len(rs)}  main {f('main_episodes'):6.0f}  fork {f('fork_episodes'):6.0f}"
          f"  eval {f('evaluation_episodes'):6.0f}  total {f('total_simulated_h'):6.2f} h")

if a.csv and rows:
    with open(a.csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print("\n->", a.csv)
