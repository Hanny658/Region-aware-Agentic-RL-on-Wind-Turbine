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

from envs.reward import compile_reward_code   # validation of random / LLM reward expressions
import math

import numpy as np

KNOBS = ("lambda_load_R2", "lambda_load_R3", "w_power", "w_speed", "dbeta_max_R2", "dbeta_max_R3")
BOUNDS = {
    "lambda_load_R2": (0.0, 50.0), "lambda_load_R3": (0.0, 50.0),
    "w_power": (1.0, 1000.0), "w_speed": (1.0, 1000.0),
    "dbeta_max_R2": (0.005, 0.10), "dbeta_max_R3": (0.005, 0.10),
    # reward v2/v3 load weights (tower and blade proxies); absent until 2026-09-11 evening, which
    # silently dropped them from every proposal (clamp_proposal keeps only keys in BOUNDS)
    "lambda_tower": (0.0, 50.0), "lambda_blade": (0.0, 50.0),
    # optional 7th knob (present only in IPC experiments): dq cyclic-pitch authority [rad/axis]
    "ipc_max": (0.002, 0.035),
}
MAX_RATIO = 3.0          # a single decision may change a knob by at most x3 / /3
ROLLBACK_DROP = 5.0      # points of F or J (percentage points)

F_OBJECTIVE_TEXT = """Ground-truth fitness F (you cannot change it, it is measured on deterministic evaluation episodes
against the paired ROSCO baseline on identical wind):
  maximise blade-root DEL reduction [%], subject to energy loss <= 1 % and Region-3
  generator-speed std not worse than ROSCO. F = DEL_red% - 20*max(0, energy_loss% - 1) - 20*max(0, 100*(speed_std_ratio - 1))."""

J_OBJECTIVE_TEXT = """Ground-truth objective J (you cannot change it; it is measured on deterministic evaluation episodes
against the paired ROSCO baseline on identical wind, and none of your choices enter its computation):
  J = mean of four percentage reductions vs ROSCO  -  20 * max(0, energy_loss% - 1)
      the four: power-output MSE, generator-speed MSE (both on above-rated steps), tower-base
      fore-aft fatigue DEL, blade-root out-of-plane fatigue DEL. Each is clipped to +-100 %.
  Speed/power regulation is therefore an OBJECTIVE here (half of J), not a constraint; the only
  constraint is energy (a 20-point penalty per % of loss beyond 1 %). There are no tiers: J and
  its four terms (J_power_mse_red_pct, J_gen_speed_mse_red_pct, J_TwrBsMyt_DEL_red_pct,
  J_RootMyc1_DEL_red_pct) are reported as continuous quantities in evaluation_now."""

C_OBJECTIVE_TEXT = """Ground-truth objective C (you cannot change it; it is measured on deterministic evaluation episodes
against the paired baseline controller on identical wind, and none of your choices enter its computation):
  C = mean(power-output MSE reduction %, generator-speed MSE reduction %)   [both on above-rated steps]
      - 20 * [ max(0, -1 - towerDEL_red%) + max(0, -1 - bladeDEL_red%) + max(0, energy_loss% - 1) ]
  i.e. regulation is the TARGET and the two fatigue loads are CONSTRAINTS: tower-base fore-aft DEL and
  blade-root out-of-plane DEL may not get more than 1 % worse than the baseline, energy loss at most 1 %.
  A violated constraint costs 20 points per %, so C is only large when the loads are held. The terms are
  reported as continuous quantities in evaluation_now (C, C_goal, C_violation_pct and the four
  J_*_red_pct terms)."""

CT_OBJECTIVE_TEXT = """Ground-truth objective CT (you cannot change it; it is measured on deterministic evaluation episodes
against the paired baseline controller on identical wind, and none of your choices enter its computation):
  CT = tower-base fore-aft fatigue DEL reduction %
       - 20 * [ max(0, -1 - powerMSE_red%) + max(0, -1 - speedMSE_red%) + max(0, -1 - bladeDEL_red%) + max(0, energy_loss% - 1) ]
  i.e. tower fatigue is the TARGET and regulation and blade fatigue are CONSTRAINTS: power MSE, speed MSE
  (above-rated steps) and blade-root DEL may not get more than 1 % worse than the baseline, energy loss at
  most 1 %. A violated constraint costs 20 points per %. The terms are reported as continuous quantities in
  evaluation_now (CT, CT_goal, CT_violation_pct and the four J_*_red_pct terms)."""


def objective_text(objective: str) -> str:
    return {"J": J_OBJECTIVE_TEXT, "C": C_OBJECTIVE_TEXT, "CT": CT_OBJECTIVE_TEXT}.get(objective, F_OBJECTIVE_TEXT)


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

If `ipc_max` appears in current_knobs, the experiment also has an individual-pitch (IPC) channel:
a separate slow agent sets a dq-frame (Coleman) cyclic-pitch amplitude pair once per rotor
rotation, bounded by +-ipc_max [rad] per axis (Region 3 only). Cyclic pitch cancels the
once-per-revolution blade-load variation from wind shear; a well-used channel cuts blade-root DEL
substantially at the cost of pitch travel. Diagnostic `ipc_amp_deg` in training_last_window is the
mean dq amplitude the policy actually uses, in degrees: if it stays far below ipc_max (channel
unused) consider whether the load weight or the authority should change; if pitch travel grows
with little DEL gain, shrink ipc_max. ipc_max is a knob like the others (bounds, x3 step rule).

{OBJECTIVE}

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
Respond with ONE JSON object only (include every knob present in current_knobs):
{"knobs": {"lambda_load_R2": x, "lambda_load_R3": x, "w_power": x, "w_speed": x, "dbeta_max_R2": x, "dbeta_max_R3": x, ...},
 "rationale": "<= 3 sentences", "expected_effect": "<= 2 sentences", "confidence": 0-1}
"""


def clamp_proposal(proposal: dict, current: dict) -> tuple[dict, list[str]]:
    """Bounds + max-ratio validation. Returns (accepted knobs, notes). The knob set is whatever
    `current` carries (the base six, plus e.g. ipc_max in IPC experiments)."""
    out, notes = {}, []
    for k in (kk for kk in current if kk in BOUNDS):
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

    # resume support (train.py --resume): the generator state is the only thing that matters
    def get_state(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def set_state(self, st: dict):
        if st.get("rng"):
            self.rng.bit_generator.state = st["rng"]

    def propose(self, summary: dict) -> dict:
        cur = summary["current_knobs"]
        k = int(self.rng.integers(self.n_change[0], self.n_change[1] + 1))
        names = self.rng.choice([kk for kk in cur if kk in BOUNDS], size=k, replace=False)
        new = dict(cur)
        for n in names:
            f = float(np.exp(self.rng.uniform(-np.log(MAX_RATIO), np.log(MAX_RATIO))))
            new[n] = float(cur[n]) * f if cur[n] > 0 else float(self.rng.uniform(*BOUNDS[n]))
        return {"knobs": new, "rationale": f"random log-uniform perturbation of {[str(n) for n in names]}"}


def system_prompt(load_signal: str = "M_oop", fitness_target: str = "blade", objective: str = "F",
                  reward_version: str = "v1") -> str:
    s = SYSTEM_PROMPT.replace("{OBJECTIVE}", objective_text(objective))
    if reward_version in ("v2", "v3"):
        s = s.replace(
            "  R2: w_power*(P/P_gspi - 1) - lambda_load_R2*load_proxy_t - 0.1*(dbeta/0.1)^2\n"
            "  R3: w_speed*exp(-|dw_gen|/0.02) - lambda_load_R3*load_proxy_t - 0.1*(dbeta/0.1)^2",
            "  R2: w_power*(P/P_gspi - 1) - lambda_tower*tower_proxy_t - lambda_blade*blade_proxy_t - 0.1*(dbeta/0.1)^2\n"
            "  R3: -w_speed*(dw_gen/0.005)^2 - lambda_tower*tower_proxy_t - lambda_blade*blade_proxy_t - 0.1*(dbeta/0.1)^2\n"
            "(reward v2: the speed term is the instantaneous speed-MSE contribution; the two load proxies "
            "are the 10 s peak-to-peak growth of tower-top fore-aft acceleration and of blade-root "
            "out-of-plane moment, each ~1 per step under ROSCO)")
    if load_signal == "fa_acc":
        s = s.replace("blade-root out-of-plane moment (the growth of\nits peak-to-peak range",
                      "tower-top fore-aft acceleration (the growth of\nits peak-to-peak range")
    if fitness_target == "tower":
        s = s.replace("maximise blade-root DEL reduction [%]", "maximise tower-base fore-aft DEL reduction [%]")
    return s


class LLMSupervisor:
    name = "llm"

    def __init__(self, client, load_signal: str = "M_oop", fitness_target: str = "blade",
                 objective: str = "F", reward_version: str = "v1"):
        self.client = client
        self.system = system_prompt(load_signal, fitness_target, objective, reward_version)

    def get_state(self) -> dict:   # resume support: keep the call/usage counters continuous
        return {"n_calls": self.client.n_calls, "usage": dict(self.client.usage)}

    def set_state(self, st: dict):
        self.client.n_calls = int(st.get("n_calls", 0))
        if st.get("usage"):
            self.client.usage.update(st["usage"])

    def propose(self, summary: dict) -> dict:
        user = ("Current training summary and history (JSON). Propose the next knob vector.\n\n"
                + json.dumps(summary, indent=1, ensure_ascii=False))
        try:
            out = self.client.ask_json(self.system, user, tag=f"decision_{summary.get('decision_index', 0)}")
        except Exception as e:  # noqa: BLE001 - an API failure must not end the training run
            return {"knobs": dict(summary["current_knobs"]),
                    "rationale": f"LLM call failed ({type(e).__name__}); kept the current knobs"}
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
        "evaluation_now": ({k: v for k, v in fit.items()
                            if k != "per_episode" and k not in ("tier", "F_strict", "F_tol2", "constraints_ok")}
                           if "J" in fit and fit.get("_objective") in ("J", "C", "CT") else
                           {k: v for k, v in fit.items() if k != "per_episode"}),
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
        super().__init__(client, **{k: v for k, v in kw.items()
                                    if k in ("load_signal", "fitness_target", "objective", "reward_version")})
        self.K = n_candidates
        self.system = self.system.split("Respond with ONE JSON object only:")[0] + CANDIDATES_FORMAT.replace("K ", f"{self.K} ")

    def propose_candidates(self, summary: dict) -> list[dict]:
        user = (f"Current training summary, decision history with fork outcomes (JSON). Propose {self.K} candidates.\n\n"
                + json.dumps(summary, indent=1, ensure_ascii=False))
        try:
            out = self.client.ask_json(self.system, user, tag=f"decision_{summary.get('decision_index', 0)}")
        except Exception as e:  # noqa: BLE001 - an API failure must not end the training run
            return [{"style": "hold", "knobs": dict(summary["current_knobs"]),
                     "rationale": f"LLM call failed ({type(e).__name__}); kept the current knobs"}]
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

    def get_state(self) -> dict:
        return {"i": self.i}

    def set_state(self, st: dict):
        self.i = int(st.get("i", 0))

    def propose(self, summary: dict) -> dict:
        ep = summary["episodes_done"]
        chosen = None
        while self.i < len(self.items) and self.items[self.i]["episode"] <= ep:
            chosen = self.items[self.i]; self.i += 1
        if chosen is None:
            return {"knobs": dict(summary["current_knobs"]), "rationale": "schedule: no change"}
        return {"knobs": dict(chosen["knobs"]), "rationale": f"schedule entry @ep {chosen['episode']}"}


class CompetenceScheduleSupervisor:
    """Replay the same distilled schedule, but indexed on competence instead of episode count
    (litreview 2026-09-01: episode-indexed replay is gain scheduling on a variable that can
    decouple from the policy's state; P5 showed the timing matters). Entry i is applied at the
    first decision point where the training-eval F has reached the donor's F at the decision
    that produced entry i (its `F_gate`), advancing at most one entry per decision. As an
    open-loop fallback an entry is applied unconditionally `max_wait` episodes after the
    donor's own episode index, so the schedule cannot stall forever."""
    name = "schedule_comp"

    def __init__(self, path: str, max_wait: float = 60.0):
        import os
        self.items = sorted(json.load(open(os.path.expanduser(path))), key=lambda x: x["episode"])
        missing = [it["episode"] for it in self.items if "F_gate" not in it]
        if missing:
            raise ValueError(f"schedule entries without F_gate (re-run scripts/dev/extract_schedule.py "
                             f"on the donor run): episodes {missing}")
        self.max_wait = float(max_wait)
        self.i = 0

    def get_state(self) -> dict:
        return {"i": self.i}

    def set_state(self, st: dict):
        self.i = int(st.get("i", 0))

    def propose(self, summary: dict) -> dict:
        n = len(self.items)
        if self.i >= n:
            return {"knobs": dict(summary["current_knobs"]), "rationale": "schedule_comp: exhausted"}
        ep = summary["episodes_done"]
        # after a rollback the guardrail masks F with the historical best (train.py); the gate
        # must see the *measured* competence or it re-applies aggressive entries into the crash
        # (observed sched2_comp_s2, 2026-09-02: 4 entries applied inside rollback decisions)
        ev = summary["evaluation_now"]
        F = float(ev.get("F_measured", ev["F"]))
        it = self.items[self.i]
        gate, deadline = float(it["F_gate"]), it["episode"] + self.max_wait
        if F >= gate or ep >= deadline:
            self.i += 1
            why = "gate" if F >= gate else "max_wait"
            return {"knobs": dict(it["knobs"]),
                    "rationale": f"schedule_comp entry {self.i}/{n} via {why} "
                                 f"(F {F:.1f} vs gate {gate:.1f}, ep {ep} vs donor {it['episode']})"}
        return {"knobs": dict(summary["current_knobs"]),
                "rationale": f"schedule_comp: waiting for entry {self.i + 1}/{n} "
                             f"(F {F:.1f} < gate {gate:.1f}, fallback @ep {deadline:g})"}


# ====================================================================== v3: other agentic levers
# The knob supervisor above tunes the *reward weights*. Once the value-target scale was fixed
# (2026-09-10) that lever stopped paying, which raises the question whether other levers do. Two
# more are implemented here; both reuse the fork verification of LLMCandidateSupervisor unchanged.
#
#   llm_hparam : the LLM tunes the PPO hyper-parameters instead of the reward. Never tested before,
#                and it could not have worked before: with an unreachable value target the critic
#                learning rate is irrelevant. `gae_lambda` in particular sets the credit-assignment
#                window 1/(1-gamma*lambda), which the 2026-09-10 diagnosis identified as the
#                dominant factor in this task.
#   llm_reward : the LLM rewrites the per-step reward as a short expression (Eureka-style), a
#                strictly larger action space than re-weighting a fixed formula. The ground-truth
#                fitness stays unmodifiable, so a reward that games itself is caught by evaluation.

HPARAM_BOUNDS = {
    "critic_lr": (1.0e-4, 1.0e-2),
    "actor_lr": (1.0e-5, 1.0e-3),
    "gae_lambda": (0.80, 0.995),
    "clip_range": (0.05, 0.30),
    "entropy_coef": (0.0, 0.01),
    "policy_std": (0.05, 1.00),          # exp(log_std); exploration scale of the residual
}
BOUNDS.update(HPARAM_BOUNDS)             # clamp_proposal validates them with the same rules

HPARAM_PROMPT = """You are a senior reinforcement-learning engineer supervising a PPO experiment on a
wind-turbine pitch controller. Two residual PPO agents (one per operating region) add a collective
pitch offset on top of a gain-scheduled PI baseline. You do NOT change the reward or the action
bounds - those are fixed. You tune the PPO hyper-parameters of both learners.

Fixed context you must reason with:
  * control step 10 ms, discount gamma = 0.998, episodes of 130 scored seconds (13000 steps);
  * the advantage is GAE, so the effective credit-assignment window is 1/(1 - gamma*lambda) steps
    = 10 ms / (1 - 0.998*lambda). At lambda = 0.98 that is 0.46 s; the tower fore-aft mode has a
    period near 3 s and one rotor revolution at rated speed is 5 s. Raising lambda lengthens the
    window but raises the variance of the advantage;
  * the critic regresses on standardised targets, so v_loss near 1.0 means it explains nothing and
    v_loss well below 1.0 means it fits;
  * networks are 64x64 tanh MLPs, 10 epochs per update, batch 1024, grad-norm clip 0.5.

{OBJECTIVE}

Diagnostics in training_last_window: `<agent>/approx_kl` (policy step size; above ~0.02 the updates
are too aggressive for the clip range), `<agent>/clipfrac`, `<agent>/v_loss`, `<agent>/std` (current
exploration scale), pitch_rate_power_tower_band_frac (share of pitch-rate power in the 0.25-0.40 Hz
tower band; a rising share means the residual is exciting the tower mode).
Guidelines: change few knobs at a time, never by more than a factor of 3 per decision, respect the
bounds, and use the decision history to avoid repeating moves that did not pay.
"""

REWARD_PROMPT = """You are a senior reinforcement-learning engineer designing the per-step reward for a
wind-turbine pitch controller. Two residual PPO agents (Region 2 below rated, Region 3 above rated)
add a collective pitch offset on top of a gain-scheduled PI baseline. You write the reward as ONE
Python expression. You do NOT change anything else.

{VARIABLES}

{OBJECTIVE}

Hard-won facts about this reward - do not rediscover them:
  * penalising the load LEVEL instead of its growth makes the policy pitch on the load signal and
    excites the flap mode: fatigue got 17-740 % worse;
  * the coefficient on `load` has a cliff - starting above about 1 collapses training, but raising
    it after the policy is competent works;
  * the tower-base fatigue reduction comes almost entirely from Region 2 peak shaving (thrust down
    at an energy cost), not from Region 3 speed regulation;
  * per-step rewards much larger than about 100 make the value target hard to fit.
Keep the expression under 200 characters, finite for every combination of the inputs above, and
bounded in magnitude by a few hundred.
"""

REWARD_VARS_V1 = """Your expression is evaluated every control step (10 ms) with exactly these variables in scope:
  region      0 in Region 2 (below rated: torque does MPPT, pitch sits on its floor),
              1 in Region 3 (above rated: constant torque, pitch regulates generator speed)
  d_wg        (generator speed - rated) / rated. Typical |d_wg| is about 0.005 in R3; positive is overspeed
  p_ratio     electrical power divided by the baseline's power on the same wind at the same instant (about 1.0)
  load        per-step fatigue proxy >= 0 of the target load signal: the growth of its peak-to-peak
              range over a trailing 10 s window, normalised so it averages about 1.0 per step under
              the baseline. This is the only load information available per step
  act         squared normalised actuation cost, (dbeta / 0.1 rad)**2, >= 0
and these functions: exp, log, sqrt, tanh, abs, min, max, and the constant pi.

The reward currently in use, which you are asked to improve on, is:
    (220*(p_ratio - 1) if region == 0 else 20*exp(-abs(d_wg)/0.02)) - 1.0*load - 0.1*act
"""

REWARD_VARS_V2 = """Your expression is evaluated every control step (10 ms) with exactly these variables in scope:
  region      0 in Region 2 (below rated: torque does MPPT, pitch sits on its floor),
              1 in Region 3 (above rated: constant torque, pitch regulates generator speed)
  d_wg        (generator speed - rated) / rated. Typical |d_wg| is about 0.005 in R3; positive is overspeed
  p_ratio     electrical power divided by the baseline's power on the same wind at the same instant (about 1.0)
  load_t      per-step TOWER fatigue proxy >= 0: growth of the peak-to-peak range of tower-top
              fore-aft acceleration over a trailing 10 s window, ~1.0 per step under the baseline
  load_b      per-step BLADE fatigue proxy >= 0: the same statistic of the blade-root out-of-plane
              moment, ~1.0 per step under the baseline
  act         squared normalised actuation cost, (dbeta / 0.1 rad)**2, >= 0
and these functions: exp, log, sqrt, tanh, abs, min, max, and the constant pi.

The reward currently in use, which you are asked to improve on, is:
    (220*(p_ratio - 1) if region == 0 else -20*(d_wg/0.005)**2) - 1.0*load_t - 1.0*load_b - 0.1*act
"""

REWARD_VARS_V3 = REWARD_VARS_V2.replace(
    "  d_wg        (generator speed - rated) / rated.",
    "  region_w    1 when the hub wind is above rated, else 0 — the label the objective's speed/power MSE\n"
    "              terms use (they are scored on region_w == 1 steps, whatever the controller does)\n"
    "  d_wg        (generator speed - rated) / rated.").replace(
    "    (220*(p_ratio - 1) if region == 0 else -20*(d_wg/0.005)**2) - 1.0*load_t - 1.0*load_b - 0.1*act",
    "    (220*(p_ratio - 1) if region == 0 else 0) - (20*(d_wg/0.005)**2 if region_w == 1 else 0) - 1.0*load_t - 1.0*load_b - 0.1*act")

COMBO_PROMPT = """You are a senior reinforcement-learning engineer supervising a PPO experiment on a
wind-turbine pitch controller. Two residual PPO agents (one per operating region) add a collective
pitch offset on top of a gain-scheduled PI baseline. You control BOTH the learner and its reward:
  (a) the PPO hyper-parameters of both learners (actor_lr, critic_lr, gae_lambda, clip_range,
      entropy_coef, policy_std), and
  (b) the per-step reward: its weights (w_power, w_speed, lambda_tower, lambda_blade), the residual
      bounds (dbeta_max_R2, dbeta_max_R3) and, if you provide "reward_code", the whole expression.
The two levers do different things: the hyper-parameters set the credit-assignment window and the
step size (they decide whether the learner can find a solution), the reward decides which solution
it finds. Treat them as such: a candidate may change one lever, or both if you have a reason.

Fixed context you must reason with:
  * control step 10 ms, discount gamma = 0.998, episodes of 130 scored seconds (13000 steps);
  * the advantage is GAE, so the effective credit-assignment window is 1/(1 - gamma*lambda) steps
    = 10 ms / (1 - 0.998*lambda). At lambda = 0.98 that is 0.46 s; the tower fore-aft mode has a
    period near 3 s and one rotor revolution at rated speed is 5 s;
  * the critic regresses on standardised targets (v_loss near 1.0 = explains nothing);
  * networks are 64x64 tanh MLPs, 10 epochs per update, batch 1024, grad-norm clip 0.5.

{OBJECTIVE}

{VARIABLES}
If you give "reward_code", it replaces the parametric reward entirely (the weight knobs are then
unused); a candidate without "reward_code" keeps the expression currently in use.

Diagnostics in training_last_window: `<agent>/approx_kl`, `<agent>/clipfrac`, `<agent>/v_loss`,
`<agent>/std`, pitch_rate_power_tower_band_frac (share of pitch-rate power in the 0.25-0.40 Hz
tower band). Guidelines: never move a numeric knob by more than a factor of 3 per decision,
respect the bounds, and use the decision history to avoid repeating moves that did not pay.
"""

CANDIDATES_TAIL = """
You will propose K candidates instead of one. Each is trained for a short fork from the same
checkpoint and evaluated on the ground-truth fitness on several wind seeds; the best fork is kept.
Make them deliberately different (conservative / aggressive / structural) and use the fork outcomes
in the history to learn which kind of move pays off. Respond with ONE JSON object only:
{"analysis": "<= 4 sentences on what limits F now and why",
 "candidates": [{"style": "conservative|aggressive|structural|hold", FIELD, "rationale": "<= 2 sentences"}, ...]}
"""


class LLMHparamSupervisor(LLMCandidateSupervisor):
    """Fork-verified LLM proposals over the PPO hyper-parameters instead of the reward weights."""
    name = "llm_hparam"

    def __init__(self, client, n_candidates: int = 3, objective: str = "F", **_kw):
        LLMSupervisor.__init__(self, client)
        self.K = n_candidates
        tail = CANDIDATES_TAIL.replace("FIELD", '"knobs": {...all the hyper-parameters...}')
        head = HPARAM_PROMPT.replace("{OBJECTIVE}", objective_text(objective))
        self.system = head + tail.replace("K candidates", f"{n_candidates} candidates")


class LLMComboSupervisor(LLMCandidateSupervisor):
    """One agent over both namespaces: PPO hyper-parameters + reward (weights, bounds, expression).
    Candidates carry "knobs" (any subset of the numeric knobs) and optionally "reward_code"."""
    name = "llm_combo"

    def __init__(self, client, n_candidates: int = 3, objective: str = "F", reward_version: str = "v1", **_kw):
        LLMSupervisor.__init__(self, client)
        self.K = n_candidates
        tail = CANDIDATES_TAIL.replace("FIELD", '"knobs": {...the knobs you change...}, "reward_code": "<optional expression>"')
        head = (COMBO_PROMPT
                .replace("{OBJECTIVE}", objective_text(objective))
                .replace("{VARIABLES}", {"v2": REWARD_VARS_V2, "v3": REWARD_VARS_V3}.get(reward_version, REWARD_VARS_V1)))
        self.system = head + tail.replace("K candidates", f"{n_candidates} candidates")

    def propose_candidates(self, summary: dict) -> list[dict]:
        user = (f"Current training summary and decision history with fork outcomes (JSON). Propose "
                f"{self.K} candidates; each may change hyper-parameters, reward knobs, the reward "
                f"expression, or several of these.\n\n" + json.dumps(summary, indent=1, ensure_ascii=False))
        try:
            out = self.client.ask_json(self.system, user, tag=f"decision_{summary.get('decision_index', 0)}")
        except Exception as e:  # noqa: BLE001 - an API failure must not end the training run
            return [{"style": "hold", "knobs": {}, "rationale": f"LLM call failed ({type(e).__name__}); kept everything"}]
        cands = out.get("candidates") if isinstance(out, dict) else None
        res = []
        for c in (cands or []):
            if not isinstance(c, dict):
                continue
            kn = dict(c.get("knobs") or {})
            src = c.get("reward_code") or kn.get("reward_code")
            kn.pop("reward_code", None)
            if isinstance(src, str) and src.strip():
                kn["reward_code"] = src.strip()
            if not kn and c.get("style", "") != "hold":
                continue
            res.append({"style": str(c.get("style", "?")), "knobs": kn,
                        "rationale": str(c.get("rationale", ""))[:200],
                        "analysis": str(out.get("analysis", ""))[:400]})
            if len(res) >= self.K:
                break
        if not res:
            return [{"style": "hold", "knobs": {}, "rationale": f"no usable candidate in the reply: {str(out)[:160]}"}]
        return res


class RandomRewardSupervisor:
    """Random STRUCTURAL control for llm_reward (2026-09-14): candidates are reward expressions
    drawn from a fixed grammar — the term shapes the LLM agent has used (quadratic / absolute /
    saturated speed terms; linear / capped / centred-bounded load terms; optional power-deviation
    term) with log-uniform weights — and verified by the same fork loop. If this arm reaches the
    balanced solution as often as the LLM, the agent's contribution is efficiency, not the prior."""
    name = "random_reward"

    SPEED = ["(d_wg/0.005)**2", "abs(d_wg)/0.005", "tanh(abs(d_wg)/0.005)", "tanh((d_wg/0.005)**2)", "sqrt(abs(d_wg)/0.005)"]
    LOAD = ["{L}", "min({L},3)", "tanh({L}-1)", "tanh({L})", "max(0,{L}-1)", "log(1+{L})"]
    POWER_R2 = ["(p_ratio-1)", "tanh(3*(p_ratio-1))", "tanh(6*(p_ratio-1))"]
    POWER_R3 = [None, "tanh(6*abs(p_ratio-1))", "abs(p_ratio-1)", "tanh(30*abs(p_ratio-1))"]
    ACT = ["act", "tanh(act)"]

    def __init__(self, seed: int = 0, n_candidates: int = 3, reward_version: str = "v3", **_kw):
        self.rng = np.random.default_rng(seed)
        self.K = n_candidates
        self.gate = "region_w==1" if reward_version == "v3" else "region==1"

    def get_state(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def set_state(self, st: dict):
        if st.get("rng"):
            self.rng.bit_generator.state = st["rng"]

    def _lu(self, lo, hi):
        return float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))

    def _sample(self) -> str:
        r = self.rng
        p2 = f"{self._lu(30, 300):.3g}*{r.choice(self.POWER_R2)}*(region==0)"
        sp = f"-{self._lu(5, 150):.3g}*{r.choice(self.SPEED)}*({self.gate})"
        p3 = r.choice(self.POWER_R3)
        p3 = f"-{self._lu(5, 100):.3g}*{p3}*({self.gate})" if p3 else ""
        lt = f"-{self._lu(0.2, 5):.3g}*" + r.choice(self.LOAD).format(L="load_t")
        lb = f"-{self._lu(0.2, 5):.3g}*" + r.choice(self.LOAD).format(L="load_b")
        act = f"-{self._lu(0.02, 0.5):.3g}*{r.choice(self.ACT)}"
        return p2 + sp + p3 + lt + lb + act

    def propose_candidates(self, summary: dict) -> list[dict]:
        out = [{"style": "hold", "knobs": {}, "rationale": "keep the current reward"}]
        while len(out) < self.K:
            src = self._sample()
            try:
                compile_reward_code(src, version="v3" if "region_w" in self.gate else "v2")
            except ValueError:
                continue
            out.append({"style": "random", "knobs": {"reward_code": src}, "rationale": "random grammar sample"})
        return out


class LLMRewardSupervisor(LLMCandidateSupervisor):
    """Fork-verified LLM proposals that rewrite the per-step reward expression (Eureka-style)."""
    name = "llm_reward"

    def __init__(self, client, n_candidates: int = 3, objective: str = "F", reward_version: str = "v1", **_kw):
        LLMSupervisor.__init__(self, client)
        self.K = n_candidates
        tail = CANDIDATES_TAIL.replace("FIELD", '"reward_code": "<one Python expression>"')
        head = (REWARD_PROMPT
                .replace("{OBJECTIVE}", objective_text(objective))
                .replace("{VARIABLES}", {"v2": REWARD_VARS_V2, "v3": REWARD_VARS_V3}.get(reward_version, REWARD_VARS_V1)))
        self.system = head + tail.replace("K candidates", f"{n_candidates} candidates")

    def propose_candidates(self, summary: dict) -> list[dict]:
        # NOT via LLMCandidateSupervisor.propose_candidates: that one drops every candidate without
        # a "knobs" dict, and these candidates carry a reward expression instead.
        user = (f"Current training summary and decision history with fork outcomes (JSON). Propose "
                f"{self.K} candidate reward expressions.\n\n" + json.dumps(summary, indent=1, ensure_ascii=False))
        try:
            out = self.client.ask_json(self.system, user, tag=f"decision_{summary.get('decision_index', 0)}")
        except Exception as e:  # noqa: BLE001 - an API failure must not end the training run
            return [{"style": "hold", "knobs": {},
                     "rationale": f"LLM call failed ({type(e).__name__}); kept the current reward"}]
        cands = out.get("candidates") if isinstance(out, dict) else None
        res = []
        for c in (cands or []):
            if not isinstance(c, dict):
                continue
            src = c.get("reward_code") or (c.get("knobs") or {}).get("reward_code")
            if not isinstance(src, str) or not src.strip():
                continue
            res.append({"style": str(c.get("style", "?")), "knobs": {"reward_code": src.strip()},
                        "rationale": str(c.get("rationale", ""))[:200],
                        "analysis": str(out.get("analysis", ""))[:400]})
            if len(res) >= self.K:
                break
        if not res:
            return [{"style": "hold", "knobs": {},
                     "rationale": f"no usable expression in the reply: {str(out)[:160]}"}]
        return res
