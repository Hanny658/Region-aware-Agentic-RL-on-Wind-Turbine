"""Verified search over the parameters of the compensated LPV-MPC (2026-09-15).

The controller is the MPC as the environment's base controller with a zero residual (the evaluation path of
roadmap v2 s18), with model-error compensation (s19). A proposer suggests parameter vectors in batches; every
candidate is verified by a deterministic evaluation on the supervisor winds (TurbSim seeds 1-2, 8 / 12.5 / 15
m/s) under the objective J; the incumbent is the best verified candidate. An MPC evaluation has no training
noise, which is what made candidate choice unreliable for the learned residual (roadmap v2 s20).

Proposers, budget-matched (same number of verified candidates, same batch size):
  random  uniform (log-uniform for scales) over the box -- the non-sequential control
  es      (1+lambda) local search: log-normal perturbations of the incumbent, occasional categorical flips
          -- the sequential non-LLM control
  llm     a reasoning model reads the parameter semantics, the objective's definition and the full history
          (parameters -> J, its four terms, per mean wind) and proposes the next batch

Every candidate is cached by its parameter hash under <root>/cache, shared by all arms (the incumbent is
evaluated once). Resumable: re-running continues from the cache and the arm's history.

    WTRL_HOME=~/wtrl600 WTRL_WIND=~/wtrl/wind600 ~/wtrl/run.sh python scripts/mpc_param_search.py \\
        --arm llm --budget 40 --root ~/wtrl/exp/mpcsearch600 --episode_s 600
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

# parameter box: name -> (kind, lo, hi) ; kinds: log, int_choice, choice
BOX = {
    "horizon":   ("int_choice", [10, 15, 20, 30]),
    "r":         ("log", 0.03, 3.0),
    "qt":        ("log", 0.1, 30.0),
    "wc_v":      ("log", 0.1, 1.5),
    "tau_adapt": ("log", 1.0, 30.0),
    "adapt":     ("choice", ["offset", "rls"]),
}
INCUMBENT0 = {"horizon": 20, "r": 0.3, "qt": 3.0, "wc_v": 0.35, "tau_adapt": 5.0, "adapt": "offset"}
SEMANTICS = """Parameters of a 3-state LPV-MPC for collective pitch of the NREL 5 MW turbine (states: rotor speed,
tower fore-aft position and velocity; linearised every 0.1 s from the Cp/Ct tables; wind held at a low-passed
estimate over the horizon; torque is ROSCO's). Cost per step: q*(speed error / (0.005*rated))^2 + qt*(tower-top
velocity / 0.2 m/s)^2 + r*(pitch increment / 0.002 rad)^2, with q = 1 fixed.
- horizon: prediction steps of 0.1 s (10, 15, 20 or 30). Longer sees further but the linearisation is held.
- r: pitch-increment weight [0.03, 3]. Lower = more aggressive pitching (tighter regulation, more actuation
  and possibly more blade/tower excitation); higher = smoother.
- qt: tower-velocity weight [0.1, 30]. Higher damps the ~0.32 Hz tower mode (tower fatigue) at the cost of
  regulation; too high sacrifices speed and power regulation.
- wc_v: corner of the wind-estimate low-pass [0.1, 1.5] rad/s. Low = smooth but lagging feed-forward;
  high = responsive but noisy (can make the loop hunt).
- adapt: model-error compensation. "offset" = offset-free MPC (a lumped aerodynamic-torque disturbance
  estimated from the rotor torque balance, held over the horizon); "rls" = adaptive MPC (a gain on the
  model aerodynamics estimated by forgetting least squares).
- tau_adapt: time constant of that estimate [1, 30] s. Short tracks faster but passes more noise."""
OBJECTIVE = """Objective J (maximise): the mean of four percentage reductions relative to the paired gain-scheduled
PI baseline on the same wind: power MSE and generator-speed MSE (on steps with hub wind above rated), and
damage-equivalent loads of tower-base fore-aft moment and blade-root out-of-plane moment; each episode's
reduction is clipped to +-100 %; minus 20 points per % of energy loss beyond 1 %. Evaluated on 600 s
episodes at 8, 12.5 and 15 m/s mean wind, two turbulence realisations each; 8 m/s has no regulation terms."""


def clip_params(p: dict) -> dict:
    out = {}
    for k, spec in BOX.items():
        v = p.get(k, INCUMBENT0[k])
        if spec[0] == "log":
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = INCUMBENT0[k]
            out[k] = float(np.clip(v, spec[1], spec[2]))
        elif spec[0] == "int_choice":
            try:
                v = int(round(float(v)))
            except (TypeError, ValueError):
                v = INCUMBENT0[k]
            out[k] = min(spec[1], key=lambda c: abs(c - v))
        else:
            out[k] = v if v in spec[1] else INCUMBENT0[k]
    # round the scales to 3 significant digits so the cache key is stable and proposals are readable
    for k, spec in BOX.items():
        if spec[0] == "log":
            out[k] = float(f"{out[k]:.3g}")
    return out


def key_of(p: dict) -> str:
    return hashlib.sha1(json.dumps(clip_params(p), sort_keys=True).encode()).hexdigest()[:12]


def summarise(fit: dict) -> dict:
    per = {}
    for pe in fit.get("per_episode", []):
        u = f"{pe['mean_wind']:g}"
        d = per.setdefault(u, {"power": [], "speed": [], "tower": [], "blade": []})
        for name, col in (("power", "power_mse_red_pct"), ("speed", "gen_speed_mse_red_pct"),
                          ("tower", "TwrBsMyt_DEL_red_pct"), ("blade", "RootMyc1_DEL_red_pct")):
            v = pe.get(col)
            if v is not None and v == v:
                d[name].append(float(v))
    per = {u: {n: (round(float(np.mean(v)), 1) if v else None) for n, v in d.items()} for u, d in per.items()}
    return {"J": round(float(fit["J"]), 3), "power": round(float(fit["J_power_mse_red_pct"]), 2),
            "speed": round(float(fit["J_gen_speed_mse_red_pct"]), 2), "tower": round(float(fit["J_TwrBsMyt_DEL_red_pct"]), 2),
            "blade": round(float(fit["J_RootMyc1_DEL_red_pct"]), 2), "energy_loss_pct": round(float(fit["energy_loss_pct"]), 3),
            "per_wind": per}


class Evaluator:
    def __init__(self, a):
        self.a = a
        self.cache = Path(os.path.expanduser(a.root)) / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)

    def path(self, p: dict, cp: float, seeds) -> Path:
        return self.cache / f"eval_{key_of(p)}_cp{cp:g}_s{''.join(map(str, seeds))}.json"

    def run(self, p: dict, slot: int, cp: float = 1.0, seeds=None) -> dict:
        seeds = seeds or self.a.seeds
        out = self.path(p, cp, seeds)
        if out.exists():
            return json.load(open(out))
        tag = out.stem[len("eval_"):]
        kw = dict(clip_params(p))
        cmd = [sys.executable, str(PROJ / "scripts" / "evaluate.py"), "--run", os.path.expanduser(self.a.config_run),
               "--gspi", "--backend", "openfast", "--seeds", *map(str, seeds), "--episode_s", str(self.a.episode_s),
               "--workers", str(self.a.workers), "--port0", str(self.a.port0 + 300 * slot), "--tag", tag,
               "--out", str(self.cache), "--base_mm_cp", str(cp), "--base_mpc_json", json.dumps(kw)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not out.exists():
            raise RuntimeError(f"evaluation failed for {kw}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        return json.load(open(out))


# ---------------------------------------------------------------- proposers
def propose_random(rng, hist, n):
    cands = []
    for _ in range(n):
        p = {}
        for k, spec in BOX.items():
            if spec[0] == "log":
                p[k] = math.exp(rng.uniform(math.log(spec[1]), math.log(spec[2])))
            else:
                p[k] = spec[1][rng.integers(len(spec[1]))]
        cands.append({"params": clip_params(p), "rationale": "uniform random"})
    return cands


def propose_es(rng, hist, n, sigma=0.35):
    inc = max(hist, key=lambda h: h["result"]["J"])["params"]
    cands = []
    for _ in range(n):
        p = dict(inc)
        for k, spec in BOX.items():
            if spec[0] == "log":
                p[k] = inc[k] * math.exp(rng.normal(0.0, sigma))
            elif spec[0] == "int_choice" and rng.random() < 0.3:
                i = spec[1].index(inc[k]) + int(rng.choice([-1, 1]))
                p[k] = spec[1][int(np.clip(i, 0, len(spec[1]) - 1))]
            elif spec[0] == "choice" and rng.random() < 0.15:
                p[k] = [c for c in spec[1] if c != inc[k]][0]
        cands.append({"params": clip_params(p), "rationale": "log-normal perturbation of the incumbent"})
    return cands


def propose_llm(client, hist, n, round_i):
    rows = sorted(hist, key=lambda h: -h["result"]["J"])
    table = [{"params": h["params"], **{k: h["result"][k] for k in ("J", "power", "speed", "tower", "blade", "energy_loss_pct")},
              "per_wind": h["result"]["per_wind"], "round": h["round"]} for h in rows]
    system = ("You tune a model predictive pitch controller for a wind turbine. Every candidate you propose is "
              "evaluated deterministically in OpenFAST on the objective below and the best verified candidate is kept. "
              "Propose candidates that are informative: exploit what the history shows, and explore where it is thin. "
              "Answer in JSON only.")
    user = (f"{SEMANTICS}\n\n{OBJECTIVE}\n\nHistory of evaluated candidates, best first (terms are % reductions vs the "
            f"baseline; per_wind gives the four terms per mean wind speed):\n{json.dumps(table, indent=0)}\n\n"
            f"Round {round_i}. Propose exactly {n} new parameter vectors, all different from each other and from the "
            f"history, inside the stated ranges. Return JSON: {{\"analysis\": \"<what the history says, 2-4 sentences>\", "
            f"\"candidates\": [{{\"params\": {{\"horizon\": int, \"r\": float, \"qt\": float, \"wc_v\": float, "
            f"\"tau_adapt\": float, \"adapt\": \"offset\"|\"rls\"}}, \"rationale\": \"<one sentence>\"}}]}}")
    out = client.ask_json(system, user, tag=f"mpc_search_r{round_i}")
    cands = [{"params": clip_params(c.get("params", {})), "rationale": str(c.get("rationale", ""))[:300]}
             for c in (out.get("candidates") or [])][:n]
    return cands, str(out.get("analysis", ""))[:1500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["random", "es", "llm", "reference"])
    ap.add_argument("--budget", type=int, default=40, help="verified candidates per arm (the incumbent excluded)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=2, help="candidates evaluated at the same time")
    ap.add_argument("--workers", type=int, default=6, help="OpenFAST workers per candidate evaluation")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--episode_s", type=float, default=600.0)
    ap.add_argument("--root", default="~/wtrl/exp/mpcsearch600")
    ap.add_argument("--config_run", default="~/wtrl/exp/mg3t_s0", help="any MPC-base run (its config.json)")
    ap.add_argument("--port0", type=int, default=7200)
    ap.add_argument("--rng", type=int, default=0)
    a = ap.parse_args()

    root = Path(os.path.expanduser(a.root))
    arm_dir = root / a.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    hist_path = arm_dir / "history.jsonl"
    ev = Evaluator(a)
    rng = np.random.default_rng(a.rng + {"random": 11, "es": 22, "llm": 33, "reference": 0}[a.arm])

    hist = [json.loads(l) for l in open(hist_path)] if hist_path.exists() else []
    if not hist:
        res = summarise(ev.run(INCUMBENT0, 0))
        hist.append({"round": 0, "params": clip_params(INCUMBENT0), "result": res, "rationale": "incumbent (s19 selection)"})
        with open(hist_path, "a") as f:
            f.write(json.dumps(hist[-1]) + "\n")
        print(f"[{a.arm}] incumbent J {res['J']:.2f}  ({res['power']} / {res['speed']} / {res['tower']} / {res['blade']})", flush=True)
    if a.arm == "reference":
        return

    client = None
    if a.arm == "llm":
        from llm.client import LLMClient
        client = LLMClient(transcript_path=arm_dir / "llm_transcript.jsonl", reasoning_effort="medium")
    seen = {key_of(h["params"]) for h in hist}
    n_done = len(hist) - 1
    round_i = max(h["round"] for h in hist)
    while n_done < a.budget:
        round_i += 1
        n = min(a.batch, a.budget - n_done)
        analysis = ""
        for attempt in range(4):
            if a.arm == "random":
                cands = propose_random(rng, hist, n)
            elif a.arm == "es":
                cands = propose_es(rng, hist, n)
            else:
                cands, analysis = propose_llm(client, hist, n, round_i)
            uniq, keys = [], set()
            for c in cands:
                k = key_of(c["params"])
                if k not in seen and k not in keys:
                    uniq.append(c); keys.add(k)
            if len(uniq) >= 1:
                cands = uniq[:n]
                break
        else:
            print(f"[{a.arm}] round {round_i}: no new candidates after 4 attempts, stopping", flush=True)
            break
        with ThreadPoolExecutor(max_workers=a.parallel) as pool:
            futs = [pool.submit(ev.run, c["params"], i % a.parallel) for i, c in enumerate(cands)]
            results = [f.result() for f in futs]
        for c, fit in zip(cands, results):
            rec = {"round": round_i, "params": c["params"], "result": summarise(fit), "rationale": c["rationale"]}
            if analysis:
                rec["analysis"] = analysis
            hist.append(rec); seen.add(key_of(c["params"]))
            with open(hist_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n_done += 1
        best = max(hist, key=lambda h: h["result"]["J"])
        print(f"[{a.arm}] round {round_i}: {n_done}/{a.budget} verified; batch J "
              f"{[round(h['result']['J'], 2) for h in hist[-len(cands):]]}; incumbent J {best['result']['J']:.2f} {best['params']}", flush=True)
    best = max(hist, key=lambda h: h["result"]["J"])
    json.dump(best, open(arm_dir / "best.json", "w"), indent=1)
    print(f"[{a.arm}] done: best J {best['result']['J']:.2f} {best['params']}", flush=True)


if __name__ == "__main__":
    main()
