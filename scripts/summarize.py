"""Summarise experiment directories (evals.csv / episodes.csv / decisions.jsonl) into one table.

    python scripts/summarize.py ~/wtrl/exp/toy_s1_lam1 ~/wtrl/exp/toy_s3_*    (or a glob)
Columns: final / best F (ground-truth fitness), DEL reduction, energy loss, R3 speed-std ratio,
number of accepted supervisor decisions and rollbacks, wall time.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd


def summarize(d: Path) -> dict | None:
    ev = d / "evals.csv"
    if not ev.exists():
        return None
    e = pd.read_csv(ev)
    last, best = e.iloc[-1], e.loc[e["F"].idxmax()]
    row = {"run": d.name, "episodes": int(last["episode"]), "F_final": last["F"], "F_best": best["F"],
           "ep_best": int(best["episode"]), "DELred_final%": last["del_red_pct"],
           "Eloss_final%": last["energy_loss_pct"], "spd_ratio_final": last["speed_std_ratio"],
           "ok_final": bool(last["constraints_ok"])}
    dec = d / "decisions.jsonl"
    if dec.exists():
        recs = [json.loads(l) for l in open(dec, encoding="utf-8")]
        row["accepted"] = sum(1 for r in recs if r.get("accepted"))
        row["rejected"] = sum(1 for r in recs if r.get("accepted") is False and r.get("changed"))
        row["rollbacks"] = sum(1 for r in recs if "rollback" in r)
    s = d / "summary.json"
    if s.exists():
        js = json.load(open(s))
        row["wall_min"] = round(js.get("wall_min", float("nan")), 1)
        row["final_knobs"] = {k: round(v, 4) for k, v in js.get("final_knobs", {}).items()}
        row["llm_calls"] = js.get("llm_calls", 0)
    ep = d / "episodes.csv"
    if ep.exists():
        p = pd.read_csv(ep)
        tail = p[p["episode"] >= p["episode"].max() - 30]
        for u, g in tail.groupby("mean_wind"):
            row[f"train_r_U{u:g}"] = round(float(g["reward_mean"].mean()), 3)
        row["terminated"] = int(p["terminated"].sum())
    return row


def main():
    paths = []
    for a in sys.argv[1:]:
        paths += glob.glob(os.path.expanduser(a))
    rows = [r for r in (summarize(Path(p)) for p in sorted(paths)) if r]
    if not rows:
        print("no runs found")
        return
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    cols = [c for c in ("run", "episodes", "F_final", "F_best", "ep_best", "DELred_final%", "Eloss_final%",
                        "spd_ratio_final", "ok_final", "accepted", "rejected", "rollbacks", "terminated",
                        "wall_min", "llm_calls") if c in df]
    print(df[cols].round(3).to_string(index=False))
    for r in rows:
        if "final_knobs" in r:
            print(f"{r['run']:>18} knobs: {r['final_knobs']}")
    # F learning curves side by side
    curves = {}
    for p in sorted(paths):
        ev = Path(p) / "evals.csv"
        if ev.exists():
            e = pd.read_csv(ev)
            curves[Path(p).name] = e.set_index("episode")["F"].round(2)
    if curves:
        print("\nF at each evaluation point:")
        print(pd.DataFrame(curves).to_string())


if __name__ == "__main__":
    main()
