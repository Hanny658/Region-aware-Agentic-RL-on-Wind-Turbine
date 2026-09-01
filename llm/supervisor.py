"""Slow-timescale supervisors that adjust the six knobs between training waves.

    KNOBS = lambda_load_R2, lambda_load_R3, w_power, w_speed, dbeta_max_R2, dbeta_max_R3

Every `supervise_every` episodes the trainer (1) runs deterministic evaluation episodes,
(2) computes the ground-truth fitness F (eval/fitness.py), (3) asks the supervisor for a new
knob vector given a JSON summary + decision history, (4) validates it (bounds, max x3 change,
optional dry run in the toy digital twin), (5) applies it, (6) at the next decision point rolls
back knobs *and* policies to the best-F state so far if F fell by more than `rollback_drop`.

Supervisors:
    none    keep knobs fixed, no guardrail (control)
    guard   keep knobs fixed, guardrail only (control)
    random  log-uniform perturbation within the same bounds / step limits (search control)
    llm     LLM proposes knobs from the same summary (AgentHPO / L2R-style parameter tuning)
"""
from __future__ import annotations

import json
import math

import numpy as np

KNOBS = ("lambda_load_R2", "lambda_load_R3", "w_power", "w_speed", "dbeta_max_R2", "dbeta_max_R3")
BOUNDS = {
    "lambda_load_R2": (0.0, 50.0), "lambda_load_R3": (0.0, 50.0),
    "w_power": (1.0, 1000.0), "w_speed": (1.0, 1000.0),
    "dbeta_max_R2": (0.005, 0.10), "dbeta_max_R3": (0.005, 0.10),
}
MAX_RATIO = 3.0          # a single decision may change a knob by at most x3 / /3
ROLLBACK_DROP = 5.0      # points of F (== percentage points of DEL reduction)

SYSTEM_PROMPT = """You are a senior wind-turbine control engineer supervising a reinforcement-learning experiment.

Plant: NREL 5 MW onshore turbine in OpenFAST (or its 1-DOF digital twin), ROSCO gain-scheduled PI baseline.
Two residual PPO agents add a collective pitch offset dbeta on top of ROSCO: one acts in Region 2
(below rated: torque does MPPT, pitch sits on the fine-pitch / peak-shaving floor; a positive dbeta
sheds thrust and blade-root load at the cost of power) and one in Region 3 (above rated, constant
torque: pitch regulates generator speed; dbeta refines speed regulation and can trade regulation
against blade-root load). Region labels come from a fixed oracle rule; you cannot change it.

Per-step reward (fixed structure, you tune only its weights):
  R2: w_power*(P/P_gspi - 1) - lambda_load_R2*load_proxy_t - 0.1*(dbeta/0.1)^2
  R3: w_speed*exp(-|dw_gen|/0.02) - lambda_load_R3*load_proxy_t - 0.1*(dbeta/0.1)^2
where load_proxy_t is a per-step fatigue proxy of the blade-root out-of-plane moment (the growth of
its peak-to-peak range over a trailing 10 s window, normalised so it averages ~1 per step under ROSCO),
so the load term averages about -lambda_load under the baseline.
Action bounds: |dbeta| <= dbeta_max_R2 (R2, non-negative only) / dbeta_max_R3 (R3), radians.

Ground-truth fitness F (you cannot change it, it is measured on deterministic evaluation episodes
against the paired ROSCO baseline on identical wind):
  maximise blade-root DEL reduction [%], subject to energy loss <= 1 % and Region-3
  generator-speed std not worse than ROSCO. F = DEL_red% - 20*max(0, energy_loss% - 1) - 20*max(0, 100*(speed_std_ratio - 1)).

Diagnostics in training_last_window: pitch_rate_power_tower_band_frac = share of pitch-rate power in
the tower fore-aft band (0.25-0.40 Hz); a rising share means the residual is exciting the tower mode
(the classic failure: fast pitching on speed error, cured by stronger load weight / smaller residual
authority / higher speed weight). fa_acc_rms = tower-top fore-aft acceleration RMS.
The summary also contains, for the last training window, the per-episode blade-root DEL and
generator-speed std of the *exploratory* training episodes grouped by mean wind speed, with a
linear-trend slope: a rising DEL trend while speed std keeps falling is the signature of
load-feedback pitching that later collapses F; act before the evaluation shows it.
Guidelines: reason about which region/term is limiting F, change few knobs at a time, never move a
knob by more than a factor of 3 per decision, respect the bounds, prefer conservative steps when the
policy is still improving, and use the decision history to avoid repeating failed moves.
Respond with ONE JSON object only:
{"knobs": {"lambda_load_R2": x, "lambda_load_R3": x, "w_power": x, "w_speed": x, "dbeta_max_R2": x, "dbeta_max_R3": x},
 "rationale": "<= 3 sentences", "expected_effect": "<= 2 sentences", "confidence": 0-1}
"""


def clamp_proposal(proposal: dict, current: dict) -> tuple[dict, list[str]]:
    """Bounds + max-ratio validation. Returns (accepted knobs, notes)."""
    out, notes = {}, []
    for k in KNOBS:
        lo, hi = BOUNDS[k]
        cur = float(current[k])
        v = proposal.get(k, cur)
        try:
            v = float(v)
            if not math.isfinite(v):
                raise ValueError
        except (TypeError, ValueError):
            notes.append(f"{k}: non-numeric -> kept {cur:g}")
            v = cur
        if cur > 0:
            r_lo, r_hi = cur / MAX_RATIO, cur * MAX_RATIO
            if v > r_hi or v < r_lo:
                notes.append(f"{k}: {v:g} exceeds x{MAX_RATIO:g} step from {cur:g} -> clipped")
                v = min(max(v, r_lo), r_hi)
        if v < lo or v > hi:
            notes.append(f"{k}: {v:g} outside [{lo:g},{hi:g}] -> clipped")
            v = min(max(v, lo), hi)
        out[k] = v
    return out, notes


class NoneSupervisor:
    name = "none"

    def propose(self, summary: dict) -> dict:
        return {"knobs": dict(summary["current_knobs"]), "rationale": "fixed knobs (control)"}


class RandomSupervisor:
    """Same search space and step limits as the LLM; log-uniform perturbation of 1-3 knobs."""
    name = "random"

    def __init__(self, seed: int = 0, n_change=(1, 3)):
        self.rng = np.random.default_rng(seed)
        self.n_change = n_change

    def propose(self, summary: dict) -> dict:
        cur = summary["current_knobs"]
        k = int(self.rng.integers(self.n_change[0], self.n_change[1] + 1))
        names = self.rng.choice(KNOBS, size=k, replace=False)
        new = dict(cur)
        for n in names:
            f = float(np.exp(self.rng.uniform(-np.log(MAX_RATIO), np.log(MAX_RATIO))))
            new[n] = float(cur[n]) * f if cur[n] > 0 else float(self.rng.uniform(*BOUNDS[n]))
        return {"knobs": new, "rationale": f"random log-uniform perturbation of {[str(n) for n in names]}"}


def system_prompt(load_signal: str = "M_oop", fitness_target: str = "blade") -> str:
    s = SYSTEM_PROMPT
    if load_signal == "fa_acc":
        s = s.replace("blade-root out-of-plane moment (the growth of\nits peak-to-peak range",
                      "tower-top fore-aft acceleration (the growth of\nits peak-to-peak range")
    if fitness_target == "tower":
        s = s.replace("maximise blade-root DEL reduction [%]", "maximise tower-base fore-aft DEL reduction [%]")
    return s


class LLMSupervisor:
    name = "llm"

    def __init__(self, client, load_signal: str = "M_oop", fitness_target: str = "blade"):
        self.client = client
        self.system = system_prompt(load_signal, fitness_target)

    def propose(self, summary: dict) -> dict:
        user = ("Current training summary and history (JSON). Propose the next knob vector.\n\n"
                + json.dumps(summary, indent=1, ensure_ascii=False))
        out = self.client.ask_json(self.system, user, tag=f"decision_{summary.get('decision_index', 0)}")
        if "knobs" not in out or not isinstance(out["knobs"], dict):
            out = {"knobs": dict(summary["current_knobs"]), "rationale": f"malformed reply kept knobs: {str(out)[:200]}"}
        return out


def build_summary(decision_index: int, episode: int, total_episodes: int, current_knobs: dict,
                  fit: dict, train_window: dict, history: list[dict], backend: str, method: str,
                  trends: dict | None = None) -> dict:
    return {
        "decision_index": decision_index,
        "episodes_done": episode, "episodes_total": total_episodes,
        "backend": backend, "method": method,
        "knob_bounds": BOUNDS, "max_change_ratio_per_decision": MAX_RATIO,
        "current_knobs": current_knobs,
        "evaluation_now": {k: v for k, v in fit.items() if k != "per_episode"},
        "evaluation_per_episode": fit["per_episode"],
        "training_last_window": train_window,
        "training_trends_by_wind": trends,
        "history": history[-8:],
    }


# ====================================================================== v2: candidate sets, forks, schedules
CANDIDATES_FORMAT = """
You will propose K candidate knob vectors instead of one. Each candidate is trained for a short fork
from the same checkpoint and evaluated on the ground-truth fitness on several wind seeds; the best
fork is kept (RHyVE-style short-horizon fork verification / population-based training guided by you).
Make the candidates deliberately different: e.g. one conservative (small changes), one aggressive,
one structural (different knob family). Use the fork outcomes in the history to learn which kind of
move pays off. Respond with ONE JSON object only:
{"analysis": "<= 4 sentences on what limits F now and why",
 "candidates": [{"style": "conservative|aggressive|structural|hold", "knobs": {...all six knobs...}, "rationale": "<= 2 sentences"}, ...]}
"""


class LLMCandidateSupervisor(LLMSupervisor):
    name = "llm_fork"

    def __init__(self, client, n_candidates: int = 3, **kw):
        super().__init__(client, **kw)
        self.K = n_candidates
        self.system = self.system.split("Respond with ONE JSON object only:")[0] + CANDIDATES_FORMAT.replace("K ", f"{self.K} ")

    def propose_candidates(self, summary: dict) -> list[dict]:
        user = (f"Current training summary, decision history with fork outcomes (JSON). Propose {self.K} candidates.\n\n"
                + json.dumps(summary, indent=1, ensure_ascii=False))
        out = self.client.ask_json(self.system, user, tag=f"decision_{summary.get('decision_index', 0)}")
        cands = out.get("candidates") if isinstance(out, dict) else None
        if not isinstance(cands, list) or not cands:
            cands = [{"style": "hold", "knobs": dict(summary["current_knobs"]), "rationale": f"malformed reply: {str(out)[:120]}"}]
        cands = [c for c in cands if isinstance(c, dict) and isinstance(c.get("knobs"), dict)][:self.K]
        for c in cands:
            c["analysis"] = str(out.get("analysis", ""))[:400] if isinstance(out, dict) else ""
        return cands or [{"style": "hold", "knobs": dict(summary["current_knobs"]), "rationale": "no valid candidate"}]


class RandomCandidateSupervisor(RandomSupervisor):
    """Same fork verification as llm_fork but random candidates (the control that isolates the LLM)."""
    name = "random_fork"

    def __init__(self, seed: int = 0, n_candidates: int = 3):
        super().__init__(seed)
        self.K = n_candidates

    def propose_candidates(self, summary: dict) -> list[dict]:
        out = [{"style": "hold", "knobs": dict(summary["current_knobs"]), "rationale": "keep current knobs"}]
        for _ in range(self.K - 1):
            pr = self.propose(summary)
            out.append({"style": "random", "knobs": pr["knobs"], "rationale": pr["rationale"]})
        return out


class ScheduleSupervisor:
    """Replay a fixed knob schedule [{"episode": e, "knobs": {...}}, ...] (ablation: is the LLM's
    contribution just the curriculum it produced?)."""
    name = "schedule"

    def __init__(self, path: str):
        import os
        self.items = sorted(json.load(open(os.path.expanduser(path))), key=lambda x: x["episode"])
        self.i = 0

    def propose(self, summary: dict) -> dict:
        ep = summary["episodes_done"]
        chosen = None
        while self.i < len(self.items) and self.items[self.i]["episode"] <= ep:
            chosen = self.items[self.i]; self.i += 1
        if chosen is None:
            return {"knobs": dict(summary["current_knobs"]), "rationale": "schedule: no change"}
        return {"knobs": dict(chosen["knobs"]), "rationale": f"schedule entry @ep {chosen['episode']}"}
