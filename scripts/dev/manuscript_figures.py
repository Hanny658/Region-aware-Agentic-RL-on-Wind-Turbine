"""Manuscript figures (docs/manuscript/figures/*.png) from the run artefacts and the docs tables.

    ~/wtrl/run.sh python scripts/dev/manuscript_figures.py [--exp ~/wtrl/exp] [--out docs/manuscript/figures]

fig1_levers.png       lever x objective: seed-paired differences (F era and J era), both held-out sets
fig2_composition.png  the four terms of J per controller (S3-S6 means; arm-mean over seeds)
fig3_mechanism.png    (a) critic collapse vs value normalisation, (b) the rollback loop under reward v2
                      and its absence under v3, (c) fork myopia: lambda_tower candidates never kept
fig4_reference.png    J on both held-out sets for every controller, plus the MPC model-mismatch rows
fig_mpc_error.png     the MPC (as base controller) under model error; fig_mpcbase.png the residual on the MPC base
fig_trajectory.png    one held-out 15 m/s episode: base / fixed-reward residual / agent-reward residual, both bases
fig_training.png      training-time J of fixed-reward and agent-reward runs on both bases (fork decisions marked)
Every MPC-alone row is the MPC as the environment base controller with a zero residual (eval_base0_*).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser(os.environ.get("WTRL_EXP", "~/wtrl/exp")))
ap.add_argument("--out", default=os.path.join(REPO, "docs", "manuscript", "figures"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
EXP = os.path.expanduser(a.exp)
TAGS = {"S3-S6": "eval_heldout_s3456_ckpt_best.json", "S7-S10": "eval_heldout2_s78910.json"}
C = {"F": "#4C72B0", "J": "#DD8452", "S3-S6": "#4C72B0", "S7-S10": "#DD8452"}
WS = {"S3-S6": "wind seeds 3–6", "S7-S10": "wind seeds 7–10"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "figure.dpi": 150, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})


def load_J(arm: str, seeds=None) -> dict:
    """{seed: {'S3-S6': json, 'S7-S10': json, 'train': best J}} for one J-era arm prefix."""
    out = {}
    for d in sorted(glob.glob(f"{EXP}/{arm}_s*")):
        s = int(d.rsplit("_s", 1)[1])
        if seeds is not None and s not in seeds:
            continue
        if not os.path.exists(f"{d}/summary.json"):
            continue
        row = {"train": json.load(open(f"{d}/summary.json"))["best_F"]}
        for k, fn in TAGS.items():
            if os.path.exists(f"{d}/{fn}"):
                row[k] = json.load(open(f"{d}/{fn}", encoding="utf-8"))
        out[s] = row
    return out


def paired(x: dict, y: dict, k: str) -> list[float]:
    return [x[s][k]["J"] - y[s][k]["J"] for s in x if s in y and k in x[s] and k in y[s]]


# ------------------------------------------------------------------ fig 1: lever x objective
def fig_levers():
    """Seed-paired differences of J against the tuned fixed reward, both held-out sets."""
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    guard = load_J("jg3t")
    arms = [("jhp3t", "LLM\nhyper-params"), ("jrhp3t", "random\nhyper-params"), ("jcb3t", "LLM hyper-params\n+ reward"),
            ("jrw3t", "LLM reward\nexpression"), ("jrwF3t", "LLM reward, told\nload-priority objective"), ("jrr3t", "random reward\nstructure"),
            ("jrwO3t", "LLM reward,\nsingle-shot")]
    for i, (arm, label) in enumerate(arms):
        x = load_J(arm)
        for j, ws in enumerate(("S3-S6", "S7-S10")):
            d = paired(x, guard, ws)
            if not d:
                continue
            xx = i + (j - 0.5) * 0.22
            ax.errorbar(xx, mean(d), yerr=pstdev(d) / np.sqrt(len(d)) if len(d) > 1 else 0, fmt="o", color=C[ws], capsize=3,
                        label=WS[ws] if i == 0 else None)
            ax.scatter([xx] * len(d), d, s=9, color=C[ws], alpha=0.35, zorder=1)
            ax.annotate(f"{sum(v > 0 for v in d)}/{len(d)}", (xx, mean(d)), textcoords="offset points",
                        xytext=(-7 if j == 0 else 7, 0), ha="right" if j == 0 else "left", va="center", fontsize=6.5, color=C[ws])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([w[1] for w in arms], fontsize=8)
    ax.set_ylabel("paired difference of J vs. fixed reward")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig1_levers.png"))
    plt.close(fig)


# ------------------------------------------------------------------ fig 2: composition of J
TERM_KEYS = ("power_mse_red_pct", "gen_speed_mse_red_pct", "TwrBsMyt_DEL_red_pct", "RootMyc1_DEL_red_pct")


def mpc_json(tag: str):
    """An MPC evaluation json; files written before the J era get J and the clipped terms computed
    from their stored aggregates (eval/fitness.objective_J), so every row is on the same scale."""
    p = f"{EXP}/mpc/{tag}.json"
    if not os.path.exists(p):
        return None
    j = json.load(open(p, encoding="utf-8"))
    if "J" not in j:
        import sys
        sys.path.insert(0, REPO)
        from eval.fitness import objective_J
        J, parts = objective_J(j, float(j["energy_loss_pct"]))
        j["J"] = J
        j.update(parts)
    return j


MM_VARIANTS = [("", "exact model"), ("_cp0.85", "$C_P/C_T$ ×0.85"), ("_cp0.95", "$C_P/C_T$ ×0.95"),
               ("_cp1.05", "$C_P/C_T$ ×1.05"), ("_cp1.15", "$C_P/C_T$ ×1.15"),
               ("_ft0.9", "tower f ×0.9"), ("_ft1.1", "tower f ×1.1"), ("_m0.8", "modal mass ×0.8"), ("_m1.2", "modal mass ×1.2")]


def terms(j: dict) -> list[float]:
    return [j.get("J_" + k, j.get(k, float("nan"))) for k in TERM_KEYS]


def arm_terms(runs: dict, ws="S3-S6"):
    keys = ("J_power_mse_red_pct", "J_gen_speed_mse_red_pct", "J_TwrBsMyt_DEL_red_pct", "J_RootMyc1_DEL_red_pct")
    vals = [[r[ws][k] for k in keys] for r in runs.values() if ws in r]
    return (np.mean(vals, axis=0), len(vals)) if vals else (None, 0)


def fig_composition():
    items = []
    for tag, label in (("eval_base0_s3456", "LPV-MPC (exact model)"),
                       ("eval_base0_regonly_s3456", "LPV-MPC,\nregulation-only cost")):
        j = mpc_json(tag)
        if j:
            items.append((label, terms(j), 1, j["J"]))
    for arm, label in (("jrw3t", "LLM reward expression"), ("jrwF3t", "LLM reward, told load-priority obj."), ("jrr3t", "random reward structure"),
                       ("jrwO3t", "LLM reward, single-shot"), ("jcb3t", "LLM hyper-params + reward"), ("jhp3t", "LLM hyper-parameters"),
                       ("jrhp3t", "random hyper-parameters"), ("jg3t", "fixed reward (tuned)")):
        runs = load_J(arm)
        t, n = arm_terms(runs)
        if t is not None:
            items.append((f"{label} (n={n})", list(t), n, mean(r["S3-S6"]["J"] for r in runs.values() if "S3-S6" in r)))
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    w = 0.2
    names = ["power MSE", "speed MSE", "tower fatigue", "blade fatigue"]
    cols = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for k in range(4):
        ax.bar(np.arange(len(items)) + (k - 1.5) * w, [it[1][k] for it in items], w, color=cols[k], label=names[k])
    for i, it in enumerate(items):
        ax.annotate(f"J={it[3]:.1f}", (i, max(it[1]) + 1.5), ha="center", fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([it[0] for it in items], rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("% reduction vs GSPI\n(held-out wind seeds 3–6)")
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig2_composition.png"))
    plt.close(fig)


# ------------------------------------------------------------------ fig 3: mechanism
def measured_J(run: str):
    p = f"{EXP}/{run}/decisions.jsonl"
    if not os.path.exists(p):
        return [], [], []
    ep, J, rb = [], [], []
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        f = r["fit"]
        if "J" not in f:
            continue
        ep.append(r["episode"]); J.append(f.get("J_measured", f["J"])); rb.append("rollback" in r)
    return ep, J, rb


def vloss_curve(run: str):
    p = f"{EXP}/{run}/episodes.csv"
    if not os.path.exists(p):
        return None, None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    col = next((c for c in rows[0] if c.endswith("v_loss") and c.startswith("R3")), None) or \
          next((c for c in rows[0] if "v_loss" in c), None)
    if col is None:
        return None, None
    x, y = [], []
    for r in rows:
        try:
            v = float(r[col])
        except (TypeError, ValueError):
            continue
        if v == v:
            x.append(int(r["episode"])); y.append(v)
    return x, y


def critic_stats(run: str):
    """R3 critic diagnostics from scripts/dev/critic_health.py (V std, layer-2 saturation %, corr(V, -|dw|))."""
    import re
    import subprocess
    import sys
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "dev", "critic_health.py"), "--runs", f"{EXP}/{run}"],
                         capture_output=True, text=True, cwd=REPO).stdout
    for line in out.splitlines():
        if " R3 " in line and "corr(V" in line:
            m = re.search(r"V\s+(-?[0-9.]+)\s+±\s+([0-9.]+).*L2 sat\s+([0-9]+)%.*corr\(V,-\|dw\|\)\s+([+-][0-9.]+)", line)
            if m:
                return {"V_std": float(m.group(2)), "L2_sat": float(m.group(3)), "corr": float(m.group(4))}
    return None


def fig_mechanism():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    # (a) rollback loop: measured J per decision, untuned reward (speed term by router region) vs tuned v3
    ax = axes[0]
    for run, label, col in (("jg2_s0", "speed term gated by the router's region", "#C44E52"),
                            ("jg3t_s0", "speed term gated by the wind label (tuned)", "#4C72B0")):
        ep, J, rb = measured_J(run)
        if ep:
            ax.plot(ep, J, "-o", ms=3, lw=0.9, color=col, label=label)
            ax.scatter([e for e, r in zip(ep, rb) if r], [j for j, r in zip(J, rb) if r], marker="x", s=40, color=col, zorder=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("training episode"); ax.set_ylabel("measured J (supervisor winds)")
    ax.set_title("(a) the rollback loop (× = rolled back)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    # (b) fork myopia: lambda_tower of every candidate vs its fork J, chosen marked
    ax = axes[1]
    rng = np.random.default_rng(0)
    for d in sorted(glob.glob(f"{EXP}/jwf3_s*")):
        for l in open(f"{d}/decisions.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if "fork" not in r:
                continue
            for i, c in enumerate(r["fork"]["candidates"]):
                lt = c["knobs"].get("lambda_tower")
                if lt is None:
                    continue
                kept = i == r["fork"]["chosen"]
                ax.scatter(lt * (1 + 0.02 * rng.standard_normal()), c["fit"]["J"], s=30 if kept else 14,
                           color="#4C72B0" if kept else "0.55", marker="o" if kept else "x", zorder=3 if kept else 2)
    ax.set_xlabel("candidate tower weight λ_T"); ax.set_ylabel("J of the 30-episode fork")
    ax.set_title("(b) fork search: no λ_T > 1 is ever kept")
    ax.scatter([], [], s=30, color="#4C72B0", label="kept"); ax.scatter([], [], s=14, marker="x", color="0.55", label="rejected")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig3_mechanism.png"))
    plt.close(fig)


# ------------------------------------------------------------------ fig 4: reference + mismatch
def fig_reference():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    rows = []
    for tag_a, tag_b, label in (("eval_base0_s3456", "eval_base0_s78910", "LPV-MPC, exact model"),
                                ("eval_base0_cp0.95_s3456", "eval_base0_cp0.95_s78910", "LPV-MPC, $C_P/C_T$ ×0.95"),
                                ("eval_base0_regonly_s3456", "eval_base0_regonly_s78910", "LPV-MPC, regulation-only cost")):
        ja, jb = mpc_json(tag_a), mpc_json(tag_b)
        if ja and jb:
            rows.append((label, [ja["J"]], [jb["J"]]))
    for arm, label in (("jrw3t", "LLM reward expression"), ("jrwF3t", "LLM reward, told load-priority obj."), ("jrr3t", "random reward structure"),
                       ("jrwO3t", "LLM reward, single-shot"), ("jcb3t", "LLM hyper-params + reward"), ("jhp3t", "LLM hyper-parameters"),
                       ("jrhp3t", "random hyper-parameters"), ("jg3t", "fixed reward (tuned)")):
        runs = load_J(arm)
        A = [r["S3-S6"]["J"] for r in runs.values() if "S3-S6" in r]
        B = [r["S7-S10"]["J"] for r in runs.values() if "S7-S10" in r]
        if A:
            rows.append((f"{label} (n={len(A)})", A, B))
    rows.append(("GSPI (pairing reference)", [0.0], [0.0]))
    for i, (label, A, B) in enumerate(rows):
        for j, (vals, ws) in enumerate(((A, "S3-S6"), (B, "S7-S10"))):
            if not vals:
                continue
            y = i + (j - 0.5) * 0.25
            ax.errorbar(mean(vals), y, xerr=pstdev(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0, fmt="o",
                        color=C[ws], capsize=3, label=WS[ws] if i == 0 else None)
            ax.scatter(vals, [y] * len(vals), s=9, color=C[ws], alpha=0.35)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("J (held-out; mean ± s.e., dots = seeds)")
    ax.set_title("(a) every controller, both held-out wind-seed sets"); ax.legend(frameon=False, loc="lower right")
    # (b) MPC model mismatch and stress classes
    ax = axes[1]
    labels, vals = [], []
    for suf, lab in MM_VARIANTS:
        j = mpc_json(f"eval_base0{suf}_s3456")
        if j:
            labels.append(lab); vals.append(j["J"])
    for tag, lab in (("eval_base0_ti14", "turbulence intensity 14 %"), ("eval_base0_ti22", "turbulence intensity 22 %, 15 m/s"),
                     ("eval_base0_u18", "18 m/s")):
        j = mpc_json(tag)
        if j:
            labels.append(lab); vals.append(j["J"])
    cols = ["#4C72B0" if v > 0 else "#C44E52" for v in vals]
    ax.barh(range(len(vals)), vals, color=cols)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=7.5); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("J, LPV-MPC (base controller)")
    ax.set_title("(b) MPC: model error, off-design winds")
    for i, v in enumerate(vals):
        if v >= 0:
            ax.annotate(f"{v:.1f}", (v, i), textcoords="offset points", xytext=(4, 0), ha="left", va="center", fontsize=7)
        else:
            ax.annotate(f"{v:.1f}", (min(v + 0.5, -0.5), i), textcoords="offset points", xytext=(3, 0), ha="left",
                        va="center", fontsize=7, color="white" if v < -6 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig4_reference.png"))
    plt.close(fig)



# ------------------------------------------------------------------ fig: MPC model-error fragility (motivation)
def fig_mpc_error():
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    labels, vals = [], []
    for suf, lab in MM_VARIANTS:
        j = mpc_json(f"eval_base0{suf}_s3456")
        if j:
            labels.append(lab); vals.append(j["J"])
    cols = ["#4C72B0" if v > 0 else "#C44E52" for v in vals]
    ax.barh(range(len(vals)), vals, color=cols)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}", (max(v, 0) + 0.4 if v >= -6 else v + 0.5, i), va="center", fontsize=7,
                    ha="left", color="white" if v < -6 else "black")
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("J on held-out wind seeds 3–6, LPV-MPC as the base controller, zero residual")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_mpc_error.png"))
    plt.close(fig)


# ------------------------------------------------------------------ fig: residual on the MPC base (main result)
def fig_mpcbase():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    rows = []   # (label, [J S3-S6 per seed], [J S7-S10 per seed], terms S3-S6 mean)
    for tag_a, tag_b, label in (("eval_base0_s3456", "eval_base0_s78910", "MPC alone, exact model"),):
        ja, jb = mpc_json(tag_a), mpc_json(tag_b)
        if ja and jb:
            rows.append((label, [ja["J"]], [jb["J"]], terms(ja)))
    for arm, label in (("mg3t", "MPC + residual, fixed reward"), ("mrw3t", "MPC + residual, LLM reward")):
        runs = load_J(arm)
        A = [r["S3-S6"]["J"] for r in runs.values() if "S3-S6" in r]; B = [r["S7-S10"]["J"] for r in runs.values() if "S7-S10" in r]
        t, n = arm_terms(runs)
        if A:
            rows.append((f"{label} (n={len(A)})", A, B, list(t)))
    for tag_a, tag_b, label in (("eval_base0_cp0.95_s3456", "eval_base0_cp0.95_s78910", "MPC alone, $C_P/C_T$ ×0.95"),):
        ja, jb = mpc_json(tag_a), mpc_json(tag_b)
        if ja and jb:
            rows.append((label, [ja["J"]], [jb["J"]], terms(ja)))
    for arm, label in (("mgC3t", "×0.95 MPC + residual, fixed reward"), ("mrwC3t", "×0.95 MPC + residual, LLM reward")):
        runs = load_J(arm)
        A = [r["S3-S6"]["J"] for r in runs.values() if "S3-S6" in r]; B = [r["S7-S10"]["J"] for r in runs.values() if "S7-S10" in r]
        t, n = arm_terms(runs)
        if A:
            rows.append((f"{label} (n={len(A)})", A, B, list(t)))
    ax = axes[0]
    for i, (label, A, B, _) in enumerate(rows):
        for j, (vals, ws) in enumerate(((A, "S3-S6"), (B, "S7-S10"))):
            if not vals:
                continue
            y = i + (j - 0.5) * 0.25
            ax.errorbar(mean(vals), y, xerr=pstdev(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0, fmt="o",
                        color=C[ws], capsize=3, label=WS[ws] if i == 0 else None)
            ax.scatter(vals, [y] * len(vals), s=10, color=C[ws], alpha=0.4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("J (held-out; mean ± s.e., dots = seeds)")
    ax.set_title("(a) the residual on the MPC, exact and mismatched model"); ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    w = 0.2
    names = ["power MSE", "speed MSE", "tower fatigue", "blade fatigue"]
    cols = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for k in range(4):
        ax.bar(np.arange(len(rows)) + (k - 1.5) * w, [r[3][k] for r in rows], w, color=cols[k], label=names[k])
    ax.axhline(0, color="k", lw=0.6)
    short = ["MPC\nexact", "+ resid.\nfixed rew.", "+ resid.\nLLM rew.", "MPC\n$C_P$ ×0.95", "+ resid.\nfixed rew.", "+ resid.\nLLM rew."][:len(rows)]
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(short, fontsize=7.5)
    ax.set_ylabel("% reduction vs GSPI (wind seeds 3–6)"); ax.set_title("(b) what the J is made of", pad=22)
    ax.legend(ncol=4, frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_mpcbase.png"))
    plt.close(fig)


# ------------------------------------------------------------------ fig: trajectory overlays on one held-out episode
BASE_DIR = os.path.expanduser(os.environ.get("WTRL_BASE", "~/wtrl/baselines/openfast"))


def _ep_metrics(d):
    """Per-episode speed MSE on the wind-labelled above-rated steps and tower-base DEL (m = 4), as the objective computes them."""
    import sys
    sys.path.insert(0, REPO)
    import yaml
    from eval.metrics import del_rainflow
    tb = yaml.safe_load(open(os.path.join(REPO, "configs", "turbine", "nrel5mw.yaml")))
    act = d["warmup"] == 0
    r3 = act & (d["v_hub"] > float(tb["rated_wind_ms"]))
    wg = float(tb["rated_gen_speed_rads"])
    mse = float((((d["gen_speed"][r3] - wg) / wg) ** 2).mean()) if r3.sum() > 100 else float("nan")
    pmse = float((((d["P"][r3] - 5.0e6) / 5.0e6) ** 2).mean()) if r3.sum() > 100 else float("nan")
    k = d["outb_Time"] >= 20.0
    dto = float(d["outb_Time"][1] - d["outb_Time"][0])
    return mse, float(del_rainflow(d["outb_TwrBsMyt"][k], dto, 4) / 1e3), wg, pmse


def fig_trajectory():
    rows = [("GSPI base", [("GSPI", lambda k: f"{BASE_DIR}/U15_TI8_S{k}.npz", "0.45"),
                           ("+ residual, fixed reward", lambda k: f"{EXP}/jg3t_s0/logs_traj_s3456/U15_TI8_S{k}.npz", "#4C72B0"),
                           ("+ residual, agent-written reward", lambda k: f"{EXP}/jrw3t_s0/logs_traj_s3456/U15_TI8_S{k}.npz", "#DD8452")]),
            ("MPC base, $C_P/C_T$ ×0.95 in its model", [("MPC ×0.95 alone", lambda k: f"{EXP}/mpc/logs_base0_cp0.95_s3456/U15_TI8_S{k}.npz", "0.45"),
                           ("+ residual, fixed reward", lambda k: f"{EXP}/mgC3t_s2/logs_traj_s3456/U15_TI8_S{k}.npz", "#4C72B0"),
                           ("+ residual, agent-written reward", lambda k: f"{EXP}/mrwC3t_s1/logs_traj_s3456/U15_TI8_S{k}.npz", "#DD8452")])]
    # episode: the 15 m/s realisation on which the GSPI-base arms' tower-fatigue difference is closest to its mean over the four
    diffs = {}
    for k in (3, 4, 5, 6):
        pa, pb = rows[0][1][1][1](k), rows[0][1][2][1](k)
        if os.path.exists(pa) and os.path.exists(pb):
            diffs[k] = _ep_metrics(np.load(pa))[1] - _ep_metrics(np.load(pb))[1]
    if not diffs:
        raise FileNotFoundError("no trajectory logs (logs_traj_s3456)")
    m = mean(diffs.values())
    k = min(diffs, key=lambda kk: abs(diffs[kk] - m))
    print(f"   trajectory episode: U15_TI8_S{k} (tower-DEL differences fixed-agent: {diffs})")
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 7.6), gridspec_kw={"height_ratios": [1.25, 1, 1]})
    for j, (rname, curves) in enumerate(rows):
        for label, pf, col in curves:
            pth = pf(k)
            if not os.path.exists(pth):
                continue
            d = np.load(pth)
            mse, delT, wg, pmse = _ep_metrics(d)
            t = np.arange(len(d["gen_speed"])) * 0.01
            w = (t >= 20) & (t <= 150)
            lab = f"{label}  (power MSE {pmse * 1e5:.2f}e-5, tower {delT:.2f} MN m)"
            axes[0, j].plot(t[w], d["P"][w] / 1e6, color=col, lw=0.7, label=lab)
            z = (t >= 60) & (t <= 100)
            axes[1, j].plot(t[z], d["P"][z] / 1e6, color=col, lw=0.9)
            kk = (d["outb_Time"] >= 20) & (d["outb_Time"] <= 150)
            axes[2, j].plot(d["outb_Time"][kk], d["outb_TwrBsMyt"][kk] / 1e3, color=col, lw=0.7)
        axes[0, j].axhline(5.0, color="k", lw=0.5, ls="--"); axes[1, j].axhline(5.0, color="k", lw=0.5, ls="--")
        axes[0, j].set_title(rname, fontsize=9)
        axes[0, j].legend(frameon=False, fontsize=6.6, loc="upper left")
        lo, hi = axes[0, j].get_ylim(); axes[0, j].set_ylim(lo, hi + 0.55 * (hi - lo))
        axes[0, j].set_xlim(20, 150); axes[1, j].set_xlim(60, 100); axes[2, j].set_xlim(20, 150)
        axes[2, j].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("electrical power [MW]\n(rated dashed)"); axes[1, 0].set_ylabel("electrical power [MW],\n60–100 s")
    axes[2, 0].set_ylabel("tower-base fore-aft\nmoment [MN m]")
    fig.suptitle(f"Held-out episode: 15 m/s, turbulence intensity 8 %, TurbSim seed {k}", fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_trajectory.png"))
    plt.close(fig)


# ------------------------------------------------------------------ fig: training-time J of the supervised runs
def _evals(run: str):
    p = f"{EXP}/{run}/evals.csv"
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    return ([int(r["episode"]) for r in rows], [float(r["J"]) for r in rows], [r["tag"].startswith("fork") for r in rows])


def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=False)
    for ax, title, arms in ((axes[0], "(a) GSPI base", (("jg3t", "fixed reward", "#4C72B0"), ("jrw3t", "agent-written reward", "#DD8452"))),
                            (axes[1], "(b) MPC base, exact model", (("mg3t", "fixed reward", "#4C72B0"), ("mrw3t", "agent-written reward", "#DD8452")))):
        for arm, label, col in arms:
            first = True
            for d in sorted(glob.glob(f"{EXP}/{arm}_s*")):
                e = _evals(os.path.basename(d))
                if e is None:
                    continue
                ep, J, fk = e
                ax.plot(ep, J, color=col, lw=0.9, alpha=0.75, label=label if first else None)
                if any(fk):
                    ax.scatter([x for x, f in zip(ep, fk) if f], [y for y, f in zip(J, fk) if f], marker="^", s=14, color=col, zorder=3,
                               label="fork decision (best candidate kept)" if first and arm.startswith(("jrw", "mrw")) else None)
                first = False
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(title); ax.set_xlabel("training episode"); ax.set_ylabel("J on the supervisor winds (seeds 1–2)")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_training.png"))
    plt.close(fig)


for f in (fig_levers, fig_composition, fig_mechanism, fig_reference, fig_mpc_error, fig_mpcbase, fig_trajectory, fig_training):
    try:
        f()
        print("ok", f.__name__)
    except Exception as e:  # noqa: BLE001 - one missing artefact must not block the other figures
        print("FAILED", f.__name__, type(e).__name__, e)
