#!/usr/bin/env python3
"""Rebuild every derived table from the run artifacts.

Reads the run directories (default `~/wtrl/exp`, override with WTRL_EXP) and writes docs/tables/.
Standard library only, so it also runs on a machine that only has the repository:
    python3 scripts/dev/build_tables.py

Outputs (docs/tables/):
    all_evals.csv          one row per (run, eval tag) - the raw material for every other table
    table_heldout_main.csv main line, held-out S3-S6, best ckpt, grouped by arm
    table_paper_metrics.csv Wang et al. metric set for the same arms
    table_robustness.csv   stress classes (TI14 / TI22@15 / U18) x arm, incl. MPC
    table_torque.csv       tq_off vs tq_on3 seed-paired, with an exact sign-flip permutation test
    table_mpc.csv          MPC tuning grid + held-out + stress rows
    knob_trajectories.csv  supervisor knob path + training-eval F per decision (curriculum figure)

Significance: exact sign-flip permutation test over the 2^n sign assignments of the paired
differences (n <= 5 here), two-sided. No scipy dependency, and exact rather than asymptotic.
"""
from __future__ import annotations

import csv
import json
import os
from itertools import product
from pathlib import Path
from statistics import mean, pstdev

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "docs" / "tables"
RUNS = Path(os.path.expanduser(os.environ.get("WTRL_EXP", "~/wtrl/exp")))
MPC = RUNS / "mpc"
MPC_EXTRA = {"mpc600": "600 s window", "mpc600s": "150 s window on the 600 s bank"}

# run -> (arm label, objective, campaign, note)
ARMS = {
    **{f"n1_llmfork_s{i}": ("llm_fork", "tower", "n=7 one bank", "s0-2 re-run 2026-09-09, s5-6 added; s3-4 from the mac") for i in range(7)},
    **{f"n1_randfork_s{i}": ("random_fork", "tower", "n=7 one bank", "s0-2 re-run 2026-09-09, s5-6 added; s3-4 from the mac") for i in range(7)},
    **{f"n1_llmsingle_s{i}": ("llm_single", "tower", "B1", "single proposal, no fork, no dry run (--no_dry_run)") for i in range(5)},
    **{f"n1_mono_s{i}": ("mono", "tower", "A1", "monolithic control arm, seed-paired with tq_off") for i in range(5)},
    **{f"sched2_ep_s{i}": ("schedule_ep", "tower", "sched2", "episode-indexed replay") for i in range(5)},
    **{f"sched2_comp_s{i}": ("schedule_comp", "tower", "sched2", "competence-indexed replay (gate bug, see README)") for i in range(5)},
    **{f"tq_off_s{i}": ("guard", "tower", "torque", "pitch-only control arm = spec+guard main line") for i in range(5)},
    **{f"tq_on3_s{i}": ("guard+torque", "tower", "torque", "R2 torque residual, all guards + reshape fix") for i in range(5)},
    "tq_on_s0": ("guard+torque(debug)", "tower", "torque", "pre-fix, 2000 Nm"),
    "tq_on2k_s0": ("guard+torque(debug)", "tower", "torque", "pre-fix, +WSE patch"),
    "tq_on2kg_s0": ("guard+torque(debug)", "tower", "torque", "pre-fix, +speed gate/power cap"),
    "tq_on2kk_s0": ("guard+torque(debug)", "tower", "torque", "pre-fix, +KE-exact reward"),
    **{f"ipc_off_s{i}": ("guard (blade obj)", "blade", "robust", "RL-alone; the only stress-swept RL set") for i in range(5)},
    **{f"toy_dtau_{t}": ("toy torque sweep", "blade", "torque", "1-DOF twin") for t in ("0", "250", "500", "2000", "fix500")},
    "gspi_td": ("GSPI+tower damper", "tower", "A2", "ROSCO TD_Mode=1 reference controller, gain sweep + held-out"),
    "_migration_check": ("(migration checks)", "-", "infra", "GSPI identity checks; provenance for the 2026-09-09 baseline restore"),
}

FIELDS = [
    "F", "F_strict", "F_tol2", "tier", "target",
    "del_red_pct", "TwrBsMyt_DEL_red_pct", "RootMyc1_DEL_red_pct",
    "speed_std_ratio", "speed_mae_ratio_R3", "energy_loss_pct",
    "power_mse_red_pct", "gen_speed_mse_red_pct", "constraints_ok",
]


def num(x, nd=3):
    return round(x, nd) if isinstance(x, (int, float)) and not isinstance(x, bool) else x


def load_evals() -> list[dict]:
    rows = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        arm, obj, camp, note = ARMS.get(d.name, ("?", "?", "?", ""))
        for f in sorted(d.glob("eval_*.json")):
            j = json.load(open(f, encoding="utf-8"))
            tag = f.stem[len("eval_"):]
            rows.append({"run": d.name, "arm": arm, "objective": obj, "campaign": camp,
                         "eval_tag": tag, **{k: num(j.get(k)) for k in FIELDS}, "note": note})
    for d, note in [(MPC, "deterministic, n=1")] + [(RUNS / k, v) for k, v in MPC_EXTRA.items()]:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("eval_*.json")):
            j = json.load(open(f, encoding="utf-8"))
            rows.append({"run": d.name, "arm": "LPV-MPC", "objective": "tower", "campaign": d.name,
                         "eval_tag": f.stem[len("eval_"):], **{k: num(j.get(k)) for k in FIELDS},
                         "note": note})
    return rows


def write(name: str, rows: list[dict], header: list[str] | None = None):
    if not rows:
        return
    header = header or list(rows[0].keys())
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  {name:26s} {len(rows):3d} rows")


def perm_p(diffs: list[float]) -> float:
    """Exact two-sided sign-flip permutation test on the mean of paired differences."""
    obs = abs(mean(diffs))
    hits = sum(1 for s in product((1, -1), repeat=len(diffs))
               if abs(mean(si * di for si, di in zip(s, diffs))) >= obs - 1e-12)
    return hits / 2 ** len(diffs)


def agg(rows, arms, tag, key="F_tol2"):
    out = []
    for a in arms:
        sel = sorted([r for r in rows if r["arm"] == a and r["eval_tag"] == tag],
                     key=lambda r: r["run"])
        if not sel:
            continue
        v = [r[key] for r in sel]
        out.append({
            "arm": a, "objective": sel[0]["objective"], "n_seeds": len(v),
            "per_seed": " ".join(f"{x:.2f}" for x in v),
            "mean": round(mean(v), 2), "std": round(pstdev(v) * (len(v) / (len(v) - 1)) ** 0.5, 2) if len(v) > 1 else 0.0,
            "strict": f"{sum(1 for r in sel if r['tier'] == 'strict')}/{len(sel)}",
            "speed_std_ratio_mean": round(mean(r["speed_std_ratio"] for r in sel), 3),
            "energy_loss_pct_mean": round(mean(r["energy_loss_pct"] for r in sel), 3),
            "runs": " ".join(r["run"] for r in sel),
        })
    return out


def main():
    print("building tables from", RUNS)
    rows = load_evals()
    write("all_evals.csv", rows)

    HELD = "heldout_s3456_ckpt_best"
    HELD2 = "heldout2_s78910"          # second, disjoint held-out wind set (S7-S10), roadmap S22
    main_arms = ["guard", "llm_fork", "llm_single", "mono", "random_fork", "schedule_ep",
                 "schedule_comp", "guard+torque"]
    write("table_heldout_main.csv", agg(rows, main_arms + ["guard (blade obj)"], HELD))
    write("table_heldout2_s78910.csv", agg(rows, main_arms, HELD2))

    # the manuscript's central table: every arm on both held-out wind sets side by side
    both = []
    for a in main_arms:
        r1, r2 = agg(rows, [a], HELD), agg(rows, [a], HELD2)
        if not r1:
            continue
        row = {"arm": a, "n_seeds": r1[0]["n_seeds"],
               "S3-S6_mean": r1[0]["mean"], "S3-S6_std": r1[0]["std"], "S3-S6_strict": r1[0]["strict"],
               "S3-S6_spd": r1[0]["speed_std_ratio_mean"]}
        if r2:
            row |= {"S7-S10_mean": r2[0]["mean"], "S7-S10_std": r2[0]["std"],
                    "S7-S10_strict": r2[0]["strict"], "S7-S10_spd": r2[0]["speed_std_ratio_mean"],
                    "delta_mean": round(r2[0]["mean"] - r1[0]["mean"], 2)}
        both.append(row)
    write("table_wind_sets.csv", both)

    # paired comparisons, both wind sets, exact + parametric (the C6 reporting rule)
    def paired(a, b, tag):
        A = {r["run"][-1]: r["F_tol2"] for r in rows if r["arm"] == a and r["eval_tag"] == tag}
        B = {r["run"][-1]: r["F_tol2"] for r in rows if r["arm"] == b and r["eval_tag"] == tag}
        k = sorted(set(A) & set(B))
        if len(k) < 2:
            return None
        d = [A[i] - B[i] for i in k]
        sd = (sum((x - mean(d)) ** 2 for x in d) / (len(d) - 1)) ** 0.5
        return {"comparison": f"{a} - {b}", "wind_set": "S3-S6" if tag == HELD else "S7-S10",
                "n": len(d), "mean_diff": round(mean(d), 2), "std_diff": round(sd, 2),
                "n_positive": sum(x > 0 for x in d), "exact_perm_p": round(perm_p(d), 4),
                "exact_floor": round(2 / 2 ** len(d), 4),
                "d_z": round(mean(d) / sd, 2) if sd else ""}

    comps = [("llm_fork", "random_fork"), ("llm_fork", "guard"), ("llm_fork", "llm_single"),
             ("random_fork", "guard"), ("schedule_ep", "guard"), ("guard", "mono"),
             ("guard", "guard+torque")]
    write("table_paired_stats.csv",
          [r for a, b in comps for tag in (HELD, HELD2) if (r := paired(a, b, tag))])

    # For the MPC the held-out row is the WIND-RELABELED pairing ("heldoutW", roadmap S16 finding 4):
    # the oracle region label keys off ROSCO's native command, which the MPC override distorts, so
    # both sides are relabeled by wind (v_hub > rated). The plain "heldout_s3456" MPC file uses the
    # native-label pairing and is kept only as the pre-fix reference.
    MPC_HELD = "heldoutW_N20q1r0.02qt0w0.35"

    def held_rows(arm):
        tag = MPC_HELD if arm == "LPV-MPC" else HELD
        return [r for r in rows if r["arm"] == arm and r["eval_tag"] == tag]

    paper = []
    for a in main_arms + ["guard (blade obj)", "LPV-MPC"]:
        sel = held_rows(a)
        if not sel:
            continue
        paper.append({"arm": a, "n": len(sel), **{
            k: round(mean(r[k] for r in sel), 2) for k in
            ("power_mse_red_pct", "gen_speed_mse_red_pct", "TwrBsMyt_DEL_red_pct",
             "RootMyc1_DEL_red_pct", "energy_loss_pct", "speed_std_ratio")}})
    write("table_paper_metrics.csv", paper)

    rob = []
    for tag in (HELD, "robust_ti14", "robust_ti22", "robust_u18"):
        for a in ("guard (blade obj)", "LPV-MPC"):
            if a == "LPV-MPC":
                mtag = MPC_HELD if tag == HELD else f"{tag}_N20q1r0.02qt0w0.35"
                sel = [r for r in rows if r["arm"] == a and r["eval_tag"] == mtag]
            else:
                sel = [r for r in rows if r["arm"] == a and r["eval_tag"] == tag]
            sel = [r for r in sel if r["TwrBsMyt_DEL_red_pct"] is not None]
            if not sel:
                continue
            rob.append({"class": tag, "arm": a, "n": len(sel),
                        "blade_DEL_red_mean": round(mean(r["RootMyc1_DEL_red_pct"] for r in sel), 2),
                        "blade_DEL_red_std": round(pstdev(r["RootMyc1_DEL_red_pct"] for r in sel), 2),
                        "tower_DEL_red_mean": round(mean(r["TwrBsMyt_DEL_red_pct"] for r in sel), 2),
                        "tower_DEL_red_std": round(pstdev(r["TwrBsMyt_DEL_red_pct"] for r in sel), 2),
                        "speed_std_ratio": round(mean(r["speed_std_ratio"] for r in sel), 3),
                        "strict": f"{sum(1 for r in sel if r['tier'] == 'strict')}/{len(sel)}"})
    write("table_robustness.csv", rob)

    pairs = []
    for i in range(5):
        off = [r for r in rows if r["run"] == f"tq_off_s{i}" and r["eval_tag"] == HELD]
        on = [r for r in rows if r["run"] == f"tq_on3_s{i}" and r["eval_tag"] == HELD]
        if off and on:
            pairs.append({"seed": i, "tq_off_F": off[0]["F_tol2"], "tq_on3_F": on[0]["F_tol2"],
                          "diff": round(on[0]["F_tol2"] - off[0]["F_tol2"], 2),
                          "tq_off_eloss": off[0]["energy_loss_pct"], "tq_on3_eloss": on[0]["energy_loss_pct"]})
    if pairs:
        d = [p["diff"] for p in pairs]
        pairs.append({"seed": "mean+-std", "tq_off_F": round(mean(p["tq_off_F"] for p in pairs), 2),
                      "tq_on3_F": round(mean(p["tq_on3_F"] for p in pairs), 2),
                      "diff": f"{mean(d):.2f} +- {pstdev(d) * (len(d) / (len(d) - 1)) ** 0.5:.2f}",
                      "tq_off_eloss": f"perm p = {perm_p(d):.4f}", "tq_on3_eloss": ""})
    write("table_torque.csv", pairs)

    # llm_fork vs random_fork, paired by RL seed (only s3/s4 are local; s0-2 from roadmap S13)
    lf = {r["run"][-1]: r["F_tol2"] for r in rows if r["arm"] == "llm_fork" and r["eval_tag"] == HELD}
    rf = {r["run"][-1]: r["F_tol2"] for r in rows if r["arm"] == "random_fork" and r["eval_tag"] == HELD}
    common = sorted(set(lf) & set(rf))
    if common:
        d = [lf[k] - rf[k] for k in common]
        print(f"  llm_fork - random_fork on local seeds {common}: "
              f"{mean(d):+.2f} (perm p = {perm_p(d):.3f}) - see README for the 5-seed figure")

    write("table_mpc.csv", [r for r in rows if r["arm"] == "LPV-MPC"])

    knobs = []
    for d in sorted(RUNS.iterdir()):
        f = d / "decisions.jsonl"
        if not f.is_dir() and f.exists():
            for line in open(f, encoding="utf-8"):
                j = json.loads(line)
                fit = j.get("fit") or {}
                knobs.append({"run": d.name, "arm": ARMS.get(d.name, ("?",))[0], "type": j.get("type"),
                              "episode": j.get("episode"), "F": num(fit.get("F"), 2), "tier": fit.get("tier"),
                              **{k: num(v, 4) for k, v in (j.get("knobs") or {}).items()}})
    write("knob_trajectories.csv", knobs,
          header=["run", "arm", "type", "episode", "F", "tier", "lambda_load_R2", "lambda_load_R3",
                  "w_power", "w_speed", "dbeta_max_R2", "dbeta_max_R3"])


if __name__ == "__main__":
    main()
