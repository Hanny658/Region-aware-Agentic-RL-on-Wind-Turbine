"""Verified DESIGN search for the compensated LPV-MPC on the above-rated range set (2026-09-21).

Question: is the main-claim controller of roadmap s33 (offset-free LPV-MPC with a wind-scheduled tower weight) something
the supervised design loop produces by itself, and does the proposer matter? Same loop as scripts/mpc_param_search.py
(a proposer suggests batches, every candidate is verified by a deterministic OpenFAST evaluation, the best verified
candidate is the incumbent), with three changes:

  space      the MPC parameters PLUS the tower-weight schedule (qt_ratio, v_lo, v_span: the weight is qt up to v_lo and
             qt * qt_ratio from v_lo + v_span, linear between; qt_ratio = 1 is the unscheduled controller)
  winds      the range bank (IEC class B, 600 s): selection on TurbSim seeds 1-2 at 12 / 16 / 20 / 24 m/s (8 episodes);
             the report uses seeds 3-6 at all seven wind speeds and is never seen by a proposer
  objective  S = J - 4 * V, V = mean over wind speeds of the summed shortfalls below -1 % of the four per-wind terms:
             J with the per-wind load rule of s28 as a soft constraint (2 episodes per wind are too noisy for a hard one)

Proposers, budget-matched: random (uniform / log-uniform over the box), es ((1+lambda) perturbation of the incumbent),
llm (reads the parameter semantics, the objective and its own history). Every arm starts from the same hand-set,
UNSCHEDULED reference; arms never see each other's history; candidates are cached by parameter hash across arms.

    WTRL_HOME=~/wtrl600 WTRL_WIND=~/wtrl/wind600 ~/wtrl/run.sh python scripts/mpc_design_search.py --arm llm --rep 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

BOX = {
    "horizon":   ("int_choice", [10, 15, 20, 30]),
    "r":         ("log", 0.03, 3.0),
    "qt":        ("log", 0.1, 30.0),
    "wc_v":      ("log", 0.1, 1.5),
    "tau_adapt": ("log", 1.0, 30.0),
    "adapt":     ("choice", ["offset", "rls"]),
    "qt_ratio":  ("log", 1.0, 8.0),
    "v_lo":      ("lin", 13.0, 22.0),
    "v_span":    ("lin", 1.0, 8.0),
}
INCUMBENT0 = {"horizon": 20, "r": 0.3, "qt": 3.0, "wc_v": 0.35, "tau_adapt": 5.0, "adapt": "offset",
              "qt_ratio": 1.0, "v_lo": 18.0, "v_span": 4.0}
TERMS = (("power", "power_mse_red_pct"), ("speed", "gen_speed_mse_red_pct"), ("tower", "TwrBsMyt_DEL_red_pct"), ("blade", "RootMyc1_DEL_red_pct"))
TOL, PEN_V = 1.0, 4.0
SEMANTICS = """Parameters of a 3-state LPV-MPC for collective pitch of the NREL 5 MW turbine (states: rotor speed, tower fore-aft
position and velocity; linearised every 0.1 s from the Cp/Ct tables; wind held at a low-passed estimate over the horizon;
torque is the baseline's). Cost per step: (speed error / (0.005*rated))^2 + qt_eff*(tower-top velocity / 0.2 m/s)^2 +
r*(pitch increment / 0.002 rad)^2. A disturbance estimator compensates aerodynamic model error.
- horizon: prediction steps of 0.1 s (10, 15, 20 or 30).
- r: pitch-increment weight [0.03, 3]. Lower = more aggressive pitching (tighter regulation, more pitch travel).
- qt: tower-velocity weight below v_lo [0.1, 30]. Higher damps the ~0.32 Hz tower mode at the cost of regulation.
- qt_ratio, v_lo, v_span: wind-scheduled tower weight. qt_eff = qt for estimated wind <= v_lo, qt*qt_ratio for wind >=
  v_lo + v_span, linear between. qt_ratio in [1, 8] (1 = no schedule), v_lo in [13, 22] m/s, v_span in [1, 8] m/s.
  At high wind the thrust is very sensitive to pitch, so tight speed regulation excites the tower more there.
- wc_v: corner of the wind-estimate low-pass [0.1, 1.5] rad/s. Low = smooth but lagging; high = responsive but noisy.
- adapt: "offset" = lumped aerodynamic-torque disturbance estimate; "rls" = gain on the model aerodynamics.
- tau_adapt: time constant of that estimate [1, 30] s."""
OBJECTIVE = """Objective S (maximise) = J - 4 * V.
J = mean of four percentage reductions relative to the paired gain-scheduled PI baseline on the same wind: power MSE and
generator-speed MSE (steps with hub wind above rated), damage-equivalent loads of tower-base fore-aft moment and blade-root
out-of-plane moment; each episode's reduction clipped to +-100 %; minus 20 points per % of energy loss beyond 1 %.
V = per-wind load rule: for every mean wind speed and every one of the four terms, the shortfall below -1 % (a term that is
worse than the baseline by more than 1 % AT THAT WIND SPEED), summed over terms and averaged over wind speeds. A controller
that is better on average but worse than the baseline at some wind speed is penalised.
Evaluated on 600 s episodes, IEC turbulence class B, mean wind 12 / 16 / 20 / 24 m/s, two turbulence realisations each.
per_wind in the history gives the four terms at each mean wind speed."""


def clip_params(p: dict) -> dict:
    out = {}
    for k, spec in BOX.items():
        v = p.get(k, INCUMBENT0[k])
        if spec[0] in ("log", "lin"):
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = INCUMBENT0[k]
            out[k] = float(f"{float(np.clip(v, spec[1], spec[2])):.3g}")
        elif spec[0] == "int_choice":
            try:
                v = int(round(float(v)))
            except (TypeError, ValueError):
                v = INCUMBENT0[k]
            out[k] = min(spec[1], key=lambda c: abs(c - v))
        else:
            out[k] = v if v in spec[1] else INCUMBENT0[k]
    if out["qt_ratio"] < 1.05:            # no schedule: the two schedule parameters are inert, fix them for the cache key
        out["qt_ratio"], out["v_lo"], out["v_span"] = 1.0, INCUMBENT0["v_lo"], INCUMBENT0["v_span"]
    return out


def key_of(p: dict) -> str:
    return hashlib.sha1(json.dumps(clip_params(p), sort_keys=True).encode()).hexdigest()[:12]


def mpc_kw(p: dict) -> dict:
    p = clip_params(p)
    kw = {k: p[k] for k in ("horizon", "r", "qt", "wc_v", "tau_adapt", "adapt")}
    if p["qt_ratio"] > 1.0:
        kw["qt_sched"] = [round(p["qt"] * p["qt_ratio"], 4), p["v_lo"], round(p["v_lo"] + p["v_span"], 3)]
    return kw


def summarise(fit: dict) -> dict:
    per = {}
    for pe in fit.get("per_episode", []):
        d = per.setdefault(f"{pe['mean_wind']:g}", {n: [] for n, _ in TERMS})
        for name, col in TERMS:
            v = pe.get(col)
            if v is not None and v == v:
                d[name].append(max(-100.0, min(100.0, float(v))))
    per = {u: {n: (round(float(np.mean(v)), 1) if v else None) for n, v in d.items()} for u, d in per.items()}
    V = float(np.mean([sum(max(0.0, -TOL - x) for x in d.values() if x is not None) for d in per.values()])) if per else 0.0
    trav = (float(np.mean([q["pitch_travel_deg"] for q in fit["per_episode"]]) / np.mean([q["pitch_travel_base_deg"] for q in fit["per_episode"]]))
            if fit.get("per_episode") else float("nan"))
    J = float(fit["J"])
    return {"S": round(J - PEN_V * V, 3), "J": round(J, 3), "V": round(V, 3),
            "power": round(float(fit["J_power_mse_red_pct"]), 2), "speed": round(float(fit["J_gen_speed_mse_red_pct"]), 2),
            "tower": round(float(fit["J_TwrBsMyt_DEL_red_pct"]), 2), "blade": round(float(fit["J_RootMyc1_DEL_red_pct"]), 2),
            "energy_loss_pct": round(float(fit["energy_loss_pct"]), 3), "travel_x_baseline": round(trav, 3), "per_wind": per}


class Evaluator:
    def __init__(self, a):
        self.a = a
        self.cache = Path(os.path.expanduser(a.root)) / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)

    def run(self, p: dict, slot: int) -> dict:
        tag = f"{key_of(p)}_m{'-'.join(f'{m:g}' for m in self.a.means)}_s{''.join(map(str, self.a.seeds))}_e{self.a.episode_s:g}"
        out = self.cache / f"eval_{tag}.json"
        if out.exists():
            return json.load(open(out))
        log = ""
        for attempt in range(2):
            port = self.a.port0 + 300 * slot + (150 if attempt else 0)
            cmd = [sys.executable, str(PROJ / "scripts" / "evaluate.py"), "--run", os.path.expanduser(self.a.config_run), "--gspi",
                   "--backend", "openfast", "--means", *[f"{m:g}" for m in self.a.means], "--seeds", *map(str, self.a.seeds), "--ti", self.a.ti,
                   "--episode_s", str(self.a.episode_s), "--workers", str(self.a.workers), "--port0", str(port), "--tag", tag,
                   "--out", str(self.cache), "--base_mm_cp", "1", "--base_mpc_json", json.dumps(mpc_kw(p))]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if out.exists():
                return json.load(open(out))
            log += f"--- attempt {attempt} (port {port})\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}\n"
            if "needs .outb metrics" in r.stderr:
                break
        fail = {"J": -100.0, "J_power_mse_red_pct": -100.0, "J_gen_speed_mse_red_pct": -100.0, "J_TwrBsMyt_DEL_red_pct": -100.0,
                "J_RootMyc1_DEL_red_pct": -100.0, "energy_loss_pct": float("nan"), "per_episode": [],
                "failed": "evaluation did not complete (episode terminated or simulator failure)", "log_tail": log[-3000:]}
        json.dump(fail, open(out, "w"), indent=1)
        return fail


def propose_random(rng, hist, n):
    cands = []
    for _ in range(n):
        p = {}
        for k, spec in BOX.items():
            if spec[0] == "log":
                p[k] = math.exp(rng.uniform(math.log(spec[1]), math.log(spec[2])))
            elif spec[0] == "lin":
                p[k] = rng.uniform(spec[1], spec[2])
            else:
                p[k] = spec[1][rng.integers(len(spec[1]))]
        cands.append({"params": clip_params(p), "rationale": "uniform random"})
    return cands


def propose_es(rng, hist, n, sigma=0.35):
    inc = max(hist, key=lambda h: h["result"]["S"])["params"]
    cands = []
    for _ in range(n):
        p = dict(inc)
        for k, spec in BOX.items():
            if spec[0] == "log":
                p[k] = inc[k] * math.exp(rng.normal(0.0, sigma))
            elif spec[0] == "lin":
                p[k] = inc[k] + rng.normal(0.0, 0.2 * (spec[2] - spec[1]))
            elif spec[0] == "int_choice" and rng.random() < 0.3:
                i = spec[1].index(inc[k]) + int(rng.choice([-1, 1]))
                p[k] = spec[1][int(np.clip(i, 0, len(spec[1]) - 1))]
            elif spec[0] == "choice" and rng.random() < 0.15:
                p[k] = [c for c in spec[1] if c != inc[k]][0]
        cands.append({"params": clip_params(p), "rationale": "perturbation of the incumbent"})
    return cands


def propose_llm(client, hist, n, round_i):
    rows = sorted(hist, key=lambda h: -h["result"]["S"])
    table = [{"params": h["params"], **{k: h["result"][k] for k in ("S", "J", "V", "power", "speed", "tower", "blade", "energy_loss_pct", "travel_x_baseline")},
              "per_wind": h["result"]["per_wind"], "round": h["round"]} for h in rows]
    system = ("You design a model predictive pitch controller for a wind turbine. Every candidate you propose is evaluated "
              "deterministically in OpenFAST on the objective below and the best verified candidate is kept. Propose candidates that "
              "are informative: exploit what the history shows, and explore where it is thin. Answer in JSON only.")
    keys = ", ".join(f'"{k}": {"int" if BOX[k][0] == "int_choice" else "str" if BOX[k][0] == "choice" else "float"}' for k in BOX)
    user = (f"{SEMANTICS}\n\n{OBJECTIVE}\n\nHistory of evaluated candidates, best first (terms are % reductions vs the baseline):\n"
            f"{json.dumps(table, indent=0)}\n\nRound {round_i}. Propose exactly {n} new parameter vectors, all different from each other "
            f"and from the history, inside the stated ranges. Return JSON: {{\"analysis\": \"<what the history says, 2-4 sentences>\", "
            f"\"candidates\": [{{\"params\": {{{keys}}}, \"rationale\": \"<one sentence>\"}}]}}")
    out = client.ask_json(system, user, tag=f"mpc_design_r{round_i}")
    cands = [{"params": clip_params(c.get("params", {})), "rationale": str(c.get("rationale", ""))[:300]} for c in (out.get("candidates") or [])][:n]
    return cands, str(out.get("analysis", ""))[:1500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["random", "es", "llm"])
    ap.add_argument("--rep", type=int, default=0, help="repeat index (its own history and random stream)")
    ap.add_argument("--budget", type=int, default=24, help="verified candidates (the starting reference excluded)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--means", nargs="+", type=float, default=[12, 16, 20, 24])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--ti", default="B")
    ap.add_argument("--episode_s", type=float, default=600.0)
    ap.add_argument("--root", default="~/wtrl/exp/mpcdesign")
    ap.add_argument("--config_run", default="~/wtrl/exp/mg3t_s0")
    ap.add_argument("--port0", type=int, default=7200)
    a = ap.parse_args()

    arm_dir = Path(os.path.expanduser(a.root)) / f"{a.arm}_r{a.rep}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    hist_path = arm_dir / "history.jsonl"
    ev = Evaluator(a)
    hist = [json.loads(l) for l in open(hist_path)] if hist_path.exists() else []
    rng = np.random.default_rng([a.rep, {"random": 11, "es": 22, "llm": 33}[a.arm], len(hist)])
    name = f"{a.arm}_r{a.rep}"
    if not hist:
        res = summarise(ev.run(INCUMBENT0, 0))
        hist.append({"round": 0, "params": clip_params(INCUMBENT0), "result": res, "rationale": "hand-set unscheduled reference"})
        open(hist_path, "a").write(json.dumps(hist[-1]) + "\n")
        print(f"[{name}] reference S {res['S']:.2f} (J {res['J']:.2f}, V {res['V']:.2f})", flush=True)
    client = None
    if a.arm == "llm":
        from llm.client import LLMClient
        client = LLMClient(transcript_path=arm_dir / "llm_transcript.jsonl", reasoning_effort="medium")
    seen = {key_of(h["params"]) for h in hist}
    n_done, round_i = len(hist) - 1, max(h["round"] for h in hist)
    while n_done < a.budget:
        round_i += 1
        n = min(a.batch, a.budget - n_done)
        analysis, cands = "", []
        for _ in range(4):
            if a.arm == "random":
                cands = propose_random(rng, hist, n)
            elif a.arm == "es":
                cands = propose_es(rng, hist, n)
            else:
                for _net in range(15):
                    try:
                        cands, analysis = propose_llm(client, hist, n, round_i)
                        break
                    except Exception as e:  # noqa: BLE001 - connection errors, timeouts, malformed JSON
                        print(f"[{name}] round {round_i}: proposal failed ({type(e).__name__}: {str(e)[:120]}); retry in 120 s", flush=True)
                        time.sleep(120)
                else:
                    raise RuntimeError("LLM proposals failed for 30 minutes")
            uniq, keys = [], set()
            for c in cands:
                k = key_of(c["params"])
                if k not in seen and k not in keys:
                    uniq.append(c); keys.add(k)
            if uniq:
                cands = uniq[:n]
                break
        else:
            print(f"[{name}] round {round_i}: no new candidates after 4 attempts, stopping", flush=True)
            break
        with ThreadPoolExecutor(max_workers=a.parallel) as pool:
            results = [f.result() for f in [pool.submit(ev.run, c["params"], i % a.parallel) for i, c in enumerate(cands)]]
        for c, fit in zip(cands, results):
            rec = {"round": round_i, "params": c["params"], "result": summarise(fit), "rationale": c["rationale"]}
            if analysis:
                rec["analysis"] = analysis
            hist.append(rec); seen.add(key_of(c["params"])); n_done += 1
            open(hist_path, "a").write(json.dumps(rec) + "\n")
        best = max(hist, key=lambda h: h["result"]["S"])
        print(f"[{name}] round {round_i}: {n_done}/{a.budget}; batch S {[round(h['result']['S'], 2) for h in hist[-len(cands):]]}; "
              f"incumbent S {best['result']['S']:.2f} (J {best['result']['J']:.2f}, V {best['result']['V']:.2f}) {best['params']}", flush=True)
    best = max(hist, key=lambda h: h["result"]["S"])
    json.dump({**best, "mpc_kw": mpc_kw(best["params"])}, open(arm_dir / "best.json", "w"), indent=1)
    print(f"[{name}] done: best S {best['result']['S']:.2f} (J {best['result']['J']:.2f}, V {best['result']['V']:.2f}) {best['params']}", flush=True)


if __name__ == "__main__":
    main()
