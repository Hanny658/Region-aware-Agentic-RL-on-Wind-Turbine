"""Train one method with an equal-episode budget, optionally under a slow-timescale supervisor.

    python scripts/train.py --backend toy --method spec --episodes 300 --workers 9 --out ~/wtrl/exp/toy_spec
    python scripts/train.py --backend toy --method spec --supervisor llm --supervise_every 30 --out ~/wtrl/exp/toy_spec_llm

Methods (all share env, reward structure, safety layer, episode schedule and PPO hyper-parameters):
    mono       one PPO on every step (no region information in the observation)
    mono_flag  one PPO, region one-hot appended to the observation
    spec       two PPOs; each step goes to the buffer of the agent that acted (oracle router)
    r3only     one PPO acting/learning on R3 steps only; zero residual in R2 (use --means 15)
Episode k of the schedule is episodes[k % len(episodes)] (cycling 8 / 12.5 / 15 m/s ...), so every
method sees the identical sequence of wind files. Updates happen after every wave of `--workers`
episodes (paper: after every single episode).

Supervisor loop (llm/supervisor.py): every --supervise_every episodes -> deterministic evaluation
on the eval wind files -> ground-truth fitness F (eval/fitness.py) -> rollback check -> proposal
(none / random / llm) -> bounds + step validation -> dry run in the toy digital twin -> apply.
Everything is logged to decisions.jsonl / evals.csv / llm_transcript.jsonl.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from agents.ppo import PPOConfig, PPOLearner, SharedCriticPPO
from agents.rollout import WorkerPool
from controllers.router import R2, R3
from envs.base_env import PROJ, default_config
from envs.factory import baseline_dir, episode_list
from eval.fitness import baseline_metrics, fitness
from llm.supervisor import (KNOBS, ROLLBACK_DROP, CompetenceScheduleSupervisor, LLMCandidateSupervisor,
                            LLMSupervisor, NoneSupervisor, RandomCandidateSupervisor, RandomSupervisor,
                            ScheduleSupervisor, build_summary, clamp_proposal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="toy", choices=["toy", "openfast"])
    ap.add_argument("--method", required=True, choices=["mono", "mono_flag", "spec", "spec_sc", "r3only"])
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--means", nargs="+", type=float, default=[8, 12.5, 15])
    ap.add_argument("--eval_means", nargs="+", type=float, default=None, help="default: same as --means")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--eval_seeds", nargs="+", type=int, default=None,
                    help="TurbSim seeds for the supervisor's evaluation F (default: same as --seeds). P2: 1 2")
    ap.add_argument("--episode_s", type=float, default=150.0)
    ap.add_argument("--warmup_s", type=float, default=20.0)
    ap.add_argument("--lambda_load", type=float, default=None)
    ap.add_argument("--w_power", type=float, default=None)
    ap.add_argument("--w_speed", type=float, default=None)
    ap.add_argument("--tau_speed_err", type=float, default=None, help="speed-term scale (normalised gen-speed error); default 0.02")
    ap.add_argument("--gamma", type=float, default=None, help="PPO discount (default configs/ppo.yaml: 0.99 = 1 s horizon at 10 ms)")
    ap.add_argument("--gae_lambda", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt_every", type=int, default=30, help="episodes")
    ap.add_argument("--port0", type=int, default=5800)
    ap.add_argument("--min_batch", type=int, default=2048, help="skip an update with fewer samples")
    # supervisor
    ap.add_argument("--supervisor", default="none",
                    choices=["none", "guard", "random", "llm", "llm_fork", "random_fork", "schedule", "schedule_comp"],
                    help="none: fixed knobs, no rollback | guard: fixed knobs + rollback-to-best guardrail | "
                         "random / llm: proposals + guardrail")
    ap.add_argument("--supervise_every", type=int, default=30, help="episodes between evaluations/decisions")
    ap.add_argument("--rollback_drop", type=float, default=ROLLBACK_DROP)
    ap.add_argument("--rollback_after", type=int, default=60, help="no rollback before this episode (grace period)")
    ap.add_argument("--rollback_on", default="drop", choices=["drop", "violation"],
                    help="drop: F fell > rollback_drop below best | violation: only when the tol2 tier is broken "
                         "(constraint violation), P4")
    ap.add_argument("--no_dry_run", action="store_true", help="skip the digital-twin dry run of proposals")
    ap.add_argument("--reasoning_effort", default="medium")
    ap.add_argument("--n_candidates", type=int, default=3, help="llm_fork / random_fork: candidates per decision")
    ap.add_argument("--fork_waves", type=int, default=1, help="training waves per fork before its evaluation")
    ap.add_argument("--knob_schedule", default=None, help="schedule supervisor: JSON [{episode, knobs}]")
    ap.add_argument("--sched_max_wait", type=float, default=60.0,
                    help="schedule_comp: apply an entry unconditionally this many episodes after its donor episode")
    ap.add_argument("--log_std_init", type=float, default=None, help="initial policy log-std (default from PPOConfig, -1.0)")
    ap.add_argument("--dbeta_max", type=float, default=None, help="override residual bound [rad] for both regions")
    ap.add_argument("--dtau_max", type=float, default=0.0,
                    help="R2 torque residual bound [Nm]; > 0 enables the 2nd action dim (needs the Controllers.f90 patch)")
    ap.add_argument("--ipc_max", type=float, default=0.0,
                    help="R3 dq-frame cyclic-pitch bound [rad]; > 0 adds 2 action dims + (Md,Mq) obs (OpenFAST only)")
    ap.add_argument("--dbeta_max_R3", type=float, default=None, help="override residual bound [rad] for R3 only")
    ap.add_argument("--load_signal", default=None, choices=["M_oop", "fa_acc"], help="reward load signal (default from reward.yaml)")
    ap.add_argument("--fitness_target", default=None, choices=["blade", "tower"], help="fitness objective (default from reward.yaml)")
    ap.add_argument("--obs_fa_acc", action="store_true", help="append tower-top fore-aft acceleration to the observation")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)

    ppo_yaml = yaml.safe_load(open(PROJ / "configs" / "ppo.yaml"))
    pcfg = PPOConfig(actor_lr=ppo_yaml["actor_lr"], critic_lr=ppo_yaml["critic_lr"],
                     gamma=(args.gamma if args.gamma is not None else ppo_yaml["gamma"]),
                     gae_lambda=(args.gae_lambda if args.gae_lambda is not None else ppo_yaml["gae_lambda"]),
                     clip_range=ppo_yaml["clip_range"],
                     batch_size=ppo_yaml["batch_size"], n_epochs=ppo_yaml["n_epochs"],
                     hidden=tuple(ppo_yaml["hidden"]),
                     log_std_init=(args.log_std_init if args.log_std_init is not None else -1.0))

    cfg_over = {"baseline_dir": baseline_dir(args.backend), "region_flag_in_obs": args.method == "mono_flag"}
    cfg = default_config(**cfg_over)
    if args.lambda_load is not None:
        cfg.reward["lambda_load"] = args.lambda_load
    if args.w_power is not None:
        cfg.reward["w_power"] = args.w_power
    if args.w_speed is not None:
        cfg.reward["w_speed"] = args.w_speed
    if args.tau_speed_err is not None:
        cfg.reward["tau_speed_err"] = args.tau_speed_err
    if args.dbeta_max is not None:
        cfg.dbeta_max = args.dbeta_max
        cfg_over["dbeta_max"] = args.dbeta_max
    if args.load_signal is not None:
        cfg.reward["load_signal"] = args.load_signal
    if args.fitness_target is not None:
        cfg.reward["fitness_target"] = args.fitness_target
    fitness_target = cfg.reward.get("fitness_target", "blade")
    if args.backend == "toy" and fitness_target == "tower":
        print("[note] toy backend has no tower channel: fitness target falls back to blade", flush=True)
        fitness_target = "blade"
        cfg.reward["fitness_target"] = "blade"
    if args.obs_fa_acc:
        cfg.obs_fa_acc = True
        cfg_over["obs_fa_acc"] = True
    if args.dtau_max > 0.0:
        cfg.dtau_max_nm = args.dtau_max
        cfg_over["dtau_max_nm"] = args.dtau_max
        cfg.reward.setdefault("kappa_tau_nm", 2.0 * args.dtau_max)
    if args.ipc_max > 0.0:
        if args.backend == "toy":
            raise SystemExit("--ipc_max requires the openfast backend (1-DOF toy has no per-blade pitch)")
        if args.supervisor in ("random", "llm") and not args.no_dry_run:
            raise SystemExit("--ipc_max with a proposing supervisor needs --no_dry_run (toy twin has no IPC)")
        cfg.ipc_max_rad = args.ipc_max
        cfg_over["ipc_max_rad"] = args.ipc_max
        cfg.reward.setdefault("kappa_ipc_rad", 2.0 * args.ipc_max)
    cfg_over["reward"] = cfg.reward
    obs_dim = (5 + (2 if cfg.region_flag_in_obs else 0) + (1 if cfg.obs_fa_acc else 0)
               + (2 if args.ipc_max > 0.0 else 0))
    act_dim = 1 + (1 if args.dtau_max > 0.0 else 0) + (2 if args.ipc_max > 0.0 else 0)
    dt, wg_rated = float(cfg.turbine["dt_ctrl_s"]), float(cfg.turbine["rated_gen_speed_rads"])

    episodes = episode_list(args.means, args.seeds, episode_s=args.episode_s, warmup_s=args.warmup_s)
    eval_episodes = episode_list(args.eval_means or args.means, args.eval_seeds or args.seeds,
                                 episode_s=args.episode_s, warmup_s=args.warmup_s)
    lam = cfg.reward["lambda_load"]
    knobs = {"lambda_load_R2": float(lam["R2"] if isinstance(lam, dict) else lam),
             "lambda_load_R3": float(lam["R3"] if isinstance(lam, dict) else lam),
             "w_power": float(cfg.reward["w_power"]), "w_speed": float(cfg.reward["w_speed"]),
             "dbeta_max_R2": float(cfg.dbeta_max),
             "dbeta_max_R3": float(args.dbeta_max_R3 if args.dbeta_max_R3 is not None else cfg.dbeta_max)}
    json.dump({**vars(args), "reward": cfg.reward, "ppo": ppo_yaml, "obs_dim": obs_dim, "knobs0": knobs,
               "episodes": [e.__dict__ for e in episodes], "eval_episodes": [e.__dict__ for e in eval_episodes]},
              open(out / "config.json", "w"), indent=1)

    # ---------------------------------------------------------------- learners per method
    if args.method in ("mono", "mono_flag"):
        shared = PPOLearner(obs_dim, pcfg, "mono", act_dim=act_dim)
        learners = {R2: shared, R3: shared}
    elif args.method == "spec":
        learners = {R2: PPOLearner(obs_dim, pcfg, "R2", act_dim=act_dim),
                    R3: PPOLearner(obs_dim, pcfg, "R3", act_dim=act_dim)}
    elif args.method == "spec_sc":
        sc = SharedCriticPPO(obs_dim, pcfg, regions=(R2, R3), names=("R2", "R3"), act_dim=act_dim)
        learners = {R2: sc.view(R2), R3: sc.view(R3)}
    else:  # r3only
        learners = {R2: None, R3: PPOLearner(obs_dim, pcfg, "R3", act_dim=act_dim)}
    sc = sc if args.method == "spec_sc" else None
    unique_learners = list({id(l): l for l in learners.values() if l is not None}.values())

    def policy_set():
        return {r: (None if l is None else l.snapshot()) for r, l in learners.items()}

    def learners_state():
        if sc is not None:
            return {"shared": copy.deepcopy(sc.state_dict())}
        return {r: (None if l is None else copy.deepcopy(l.state_dict())) for r, l in learners.items()}

    def learners_load(state):
        if sc is not None:
            sc.load_state_dict(state["shared"])
            return
        for r, l in learners.items():
            if l is not None:
                l.load_state_dict(state[r])

    def split_segments(traj: dict) -> dict:
        """Cut a trajectory into contiguous same-region segments and route them to learners.
        A region switch ends the segment for that learner (no bootstrap across the switch); the
        episode end bootstraps through the critic unless the env terminated."""
        reg = traj["region"]
        T = len(reg)
        bounds = [0] + [i for i in range(1, T) if reg[i] != reg[i - 1]] + [T]
        routed = {id(l): [] for l in unique_learners}
        for a, b in zip(bounds[:-1], bounds[1:]):
            l = learners[int(reg[a])]
            if l is None:
                continue
            end_of_episode = b == T
            routed[id(l)].append({
                "obs": traj["obs"][a:b], "act": traj["act"][a:b], "logp": traj["logp"][a:b],
                "rew": traj["rew"][a:b],
                "terminal": (traj["terminated"] if end_of_episode else True),
                "last_obs": traj["last_obs"] if end_of_episode else traj["obs"][b],
            })
        return routed

    # ---------------------------------------------------------------- pools, baselines, supervisor
    # worker envs carry the training episodes followed by the evaluation episodes (indices offset by len(episodes))
    run_tag = re.sub(r"[^A-Za-z0-9_.-]", "_", out.name)[:40]
    pool = WorkerPool(args.workers, args.backend, episodes + eval_episodes, cfg_over, hidden=pcfg.hidden,
                      port0=args.port0, tag=f"wk_{run_tag}")
    base = baseline_metrics(baseline_dir(args.backend), eval_episodes, dt, wg_rated)
    proposes = args.supervisor in ("random", "llm", "schedule", "schedule_comp")
    forks = args.supervisor in ("llm_fork", "random_fork")
    use_rollback = args.supervisor != "none"
    use_twin = proposes and args.supervisor not in ("schedule", "schedule_comp") and not args.no_dry_run
    if use_twin and args.backend != "toy":
        # the toy twin has no tower signal: dry runs use the blade-moment reward (F is reward-independent)
        twin_pool = WorkerPool(min(3, len(eval_episodes)), "toy", eval_episodes,
                               {**cfg_over, "baseline_dir": baseline_dir("toy"),
                                "reward": {**cfg.reward, "load_signal": "M_oop"}}, hidden=pcfg.hidden,
                               tag=f"tw_{run_tag}")
        twin_base = baseline_metrics(baseline_dir("toy"), eval_episodes, dt, wg_rated)
    else:
        twin_pool, twin_base = pool, base

    if args.supervisor == "llm":
        from llm.client import LLMClient
        sup = LLMSupervisor(LLMClient(out / "llm_transcript.jsonl", reasoning_effort=args.reasoning_effort),
                            load_signal=cfg.reward.get("load_signal", "M_oop"), fitness_target=fitness_target)
    elif args.supervisor == "llm_fork":
        from llm.client import LLMClient
        sup = LLMCandidateSupervisor(LLMClient(out / "llm_transcript.jsonl", reasoning_effort=args.reasoning_effort),
                                     n_candidates=args.n_candidates,
                                     load_signal=cfg.reward.get("load_signal", "M_oop"), fitness_target=fitness_target)
    elif args.supervisor == "random":
        sup = RandomSupervisor(seed=args.seed)
    elif args.supervisor == "random_fork":
        sup = RandomCandidateSupervisor(seed=args.seed, n_candidates=args.n_candidates)
    elif args.supervisor == "schedule":
        sup = ScheduleSupervisor(args.knob_schedule)
    elif args.supervisor == "schedule_comp":
        sup = CompetenceScheduleSupervisor(args.knob_schedule, max_wait=args.sched_max_wait)
    else:
        sup = NoneSupervisor()

    def evaluate(pool_, base_, knobs_, ps=None) -> dict:
        ps = ps or policy_set()
        off = len(episodes) if pool_ is pool else 0
        jobs = [{"policy_set": ps, "episode_index": off + i, "deterministic": True, "seed": 12345, "knobs": knobs_}
                for i in range(len(eval_episodes))]
        return fitness(pool_.run(jobs), base_, target=(fitness_target if pool_ is pool else "blade"))

    # ---------------------------------------------------------------- bookkeeping
    csv_path, evals_path, dec_path = out / "episodes.csv", out / "evals.csv", out / "decisions.jsonl"
    rows, eval_rows, history = [], [], []

    def dump_csv(path, rs):
        keys = []
        for r in rs:
            keys += [kk for kk in r if kk not in keys]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rs)

    def log_decision(rec: dict):
        with open(dec_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")

    def record_eval(k, fit, tag):
        eval_rows.append({"episode": k, "tag": tag, **{kk: v for kk, v in fit.items() if kk != "per_episode"},
                          **knobs})
        dump_csv(evals_path, eval_rows)

    def train_window_stats(since: int) -> dict:
        w = [r for r in rows if r["episode"] >= since]
        if not w:
            return {}

        def m(key):
            v = [r[key] for r in w if key in r and r[key] == r[key]]
            return float(np.mean(v)) if v else None

        st = {"episodes": len(w), "terminated": int(sum(r["terminated"] for r in w)),
              "reward_mean_by_wind": {}, "r_task_R2": m("r_task_R2"), "r_load_R2": m("r_load_R2"),
              "r_task_R3": m("r_task_R3"), "r_load_R3": m("r_load_R3"),
              "dbeta_abs_mean_deg": m("dbeta_abs_mean_deg"), "pitch_travel_deg": m("pitch_travel_deg"),
              "pitch_rate_power_tower_band_frac": m("pitch_rate_power_tower_band_frac"),
              "pitch_rate_power_3P_band_frac": m("pitch_rate_power_3P_band_frac"), "fa_acc_rms": m("fa_acc_rms"),
              "RootMoop_DEL_MNm": m("RootMoop_DEL_MNm"), "energy_MWh": m("energy_MWh")}
        for u in sorted({r["mean_wind"] for r in w}):
            st["reward_mean_by_wind"][f"U{u:g}"] = float(np.mean([r["reward_mean"] for r in w if r["mean_wind"] == u]))
        for l in unique_learners:
            for kk in ("approx_kl", "clipfrac", "std", "v_loss"):
                v = m(f"{l.name}/{kk}")
                if v is not None:
                    st[f"{l.name}/{kk}"] = v
        return st

    def train_trends(since: int) -> dict:
        """Per-wind per-episode DEL / speed std of the exploratory episodes since `since`, with slopes."""
        w = [r for r in rows if r["episode"] >= since]
        out = {}
        for u in sorted({r["mean_wind"] for r in w}):
            g = [r for r in w if r["mean_wind"] == u]
            ep = np.array([r["episode"] for r in g], float)
            for key, name in (("RootMoop_DEL_MNm", "DEL_MNm"), ("gen_speed_std", "gen_speed_std")):
                y = np.array([r[key] for r in g], float)
                d = out.setdefault(f"U{u:g}", {})
                d[name] = [round(float(v), 4) for v in y]
                if len(y) >= 3 and np.ptp(ep) > 0:
                    slope = float(np.polyfit(ep, y, 1)[0])
                    d[f"{name}_slope_pct_per_10ep"] = round(100.0 * slope * 10.0 / max(abs(float(y.mean())), 1e-9), 2)
        return out

    # ---------------------------------------------------------------- initial evaluation
    t_start = time.time()
    fit = evaluate(pool, base, knobs)
    record_eval(0, fit, "init")
    best = {"episode": 0, "F": fit["F"], "knobs": dict(knobs), "state": learners_state()}
    log_decision({"index": 0, "episode": 0, "type": "init", "knobs": knobs,
                  "fit": {kk: v for kk, v in fit.items() if kk != "per_episode"}, "per_episode": fit["per_episode"]})
    print(f"[init] F={fit['F']:.2f} DELred={fit['del_red_pct']:.2f}% Eloss={fit['energy_loss_pct']:.2f}% "
          f"spd={fit['speed_std_ratio']:.3f}", flush=True)
    decision_index, next_decision_at, window_start = 0, args.supervise_every, 0

    k = 0

    def train_wave(knobs_: dict, fork: str = "") -> tuple[list, dict]:
        """One wave of training episodes with the given knobs (rollout + PPO update + logging)."""
        nonlocal k
        wave = min(args.workers, args.episodes - k)
        jobs = [{"policy_set": policy_set(), "episode_index": (k + i) % len(episodes), "deterministic": False,
                 "seed": args.seed * 100003 + k + i, "knobs": knobs_} for i in range(wave)]
        t0 = time.time()
        results = pool.run(jobs)
        t_roll = time.time() - t0
        t0 = time.time()
        upd = {}
        if sc is not None:
            upd = sc.update_from_trajectories(results, args.min_batch)
        else:
            pending = {id(l): [] for l in unique_learners}
            for r in results:
                for lid, segs in split_segments(r).items():
                    pending[lid].extend(segs)
            for l in unique_learners:
                segs = pending[id(l)]
                n = sum(len(s["rew"]) for s in segs)
                if n >= args.min_batch:
                    upd.update(l.update(l.prepare(segs)))
                else:
                    upd[f"{l.name}/skipped_n"] = n
        t_upd = time.time() - t0
        for i, r in enumerate(results):
            rows.append({"episode": k + i, "fork": fork, "wall_rollout_s": t_roll, "wall_update_s": t_upd,
                         "mean_wind": r["mean_wind"], "terminated": int(r["terminated"]), **r["metrics"],
                         **{f"knob_{kk}": v for kk, v in knobs_.items()}, **upd})
        dump_csv(csv_path, rows)
        k += wave
        print(f"[{k:4d}/{args.episodes}]{(' ' + fork) if fork else ''} roll {t_roll:5.1f}s upd {t_upd:4.1f}s | "
              + " ".join(f"U{r['mean_wind']:g}:{r['metrics']['reward_mean']:6.2f}" for r in results)
              + " | " + " ".join(f"{kk.split('/')[0]}.kl={v:.4f}" for kk, v in upd.items() if kk.endswith("approx_kl")),
              flush=True)
        return results, upd

    TIER_RANK = {"strict": 2, "tol2": 1, "degraded": 0}

    try:
        while k < args.episodes:
            wave = min(args.workers, args.episodes - k)
            results, upd = train_wave(knobs)
            if k % args.ckpt_every < wave or k >= args.episodes:
                torch.save(learners_state(), out / f"ckpt_{k:05d}.pt")
                torch.save(learners_state(), out / "ckpt_last.pt")

            # ---------------------------------------------------- evaluation + supervision
            if k >= next_decision_at or k >= args.episodes:
                next_decision_at += args.supervise_every
                decision_index += 1
                t0 = time.time()
                fit = evaluate(pool, base, knobs)
                record_eval(k, fit, "eval")
                rec = {"index": decision_index, "episode": k, "type": "eval", "wall_eval_s": time.time() - t0,
                       "knobs": dict(knobs), "fit": {kk: v for kk, v in fit.items() if kk != "per_episode"},
                       "per_episode": fit["per_episode"]}
                # outcome of the previous decision
                if history:
                    history[-1]["F_after"] = fit["F"]
                    history[-1]["outcome"] = {kk: fit[kk] for kk in ("del_red_pct", "energy_loss_pct", "speed_std_ratio")}
                # guardrail (Lakhani-style supervisor): if F fell by > rollback_drop below the best evaluation
                # so far, restore the best state (knobs + policies) and continue from there
                if args.rollback_on == "violation":
                    do_rollback = use_rollback and k >= args.rollback_after and fit.get("tier") == "degraded"
                else:
                    do_rollback = use_rollback and k >= args.rollback_after and fit["F"] < best["F"] - args.rollback_drop
                if do_rollback:
                    knobs = dict(best["knobs"])
                    learners_load(best["state"])
                    rec["rollback"] = {"to_episode": best["episode"], "F_now": fit["F"], "F_best": best["F"]}
                    if history:
                        history[-1]["rolled_back"] = True
                    print(f"[sup ] ROLLBACK to best state @ep {best['episode']} (F {fit['F']:.2f} < "
                          f"{best['F']:.2f} - {args.rollback_drop})", flush=True)
                    fit = dict(fit, F=best["F"], F_measured=fit["F"])
                elif fit["F"] > best["F"]:
                    best = {"episode": k, "F": fit["F"], "knobs": dict(knobs), "state": learners_state()}
                    torch.save({"episode": k, "F": fit["F"], "knobs": dict(knobs), "state": best["state"]},
                               out / "ckpt_best.pt")
                print(f"[eval] ep {k}: F={fit['F']:.2f} DELred={fit['del_red_pct']:.2f}% "
                      f"Eloss={fit['energy_loss_pct']:.2f}% spd={fit['speed_std_ratio']:.3f} "
                      f"tier={fit.get('tier', '?')} F_tol2={fit.get('F_tol2', float('nan')):.2f} ({rec['wall_eval_s']:.0f}s)", flush=True)
                if forks and k < args.episodes:
                    summary = build_summary(decision_index, k, args.episodes, knobs, fit,
                                            train_window_stats(window_start), history, args.backend, args.method,
                                            trends=train_trends(window_start))
                    t0 = time.time()
                    cands = sup.propose_candidates(summary)
                    S0, k0 = learners_state(), k
                    outcomes = []
                    for ci, c in enumerate(cands):
                        kc, notes = clamp_proposal(c.get("knobs", {}), knobs)
                        learners_load(S0)
                        k = k0
                        for _ in range(args.fork_waves):
                            if k < args.episodes:
                                train_wave(kc, fork=f"fork{ci}")
                        fit_c = evaluate(pool, base, kc)
                        outcomes.append({"i": ci, "style": c.get("style", "?"), "knobs": kc, "notes": notes,
                                         "rationale": str(c.get("rationale", ""))[:200], "fit": fit_c,
                                         "state": learners_state(), "k_after": k})
                    best_c = max(outcomes, key=lambda o_: (TIER_RANK.get(o_["fit"].get("tier"), 0), o_["fit"]["F"], o_["fit"].get("F_tol2", 0.0)))
                    learners_load(best_c["state"])
                    knobs = dict(best_c["knobs"])
                    k = min(args.episodes, k0 + sum(o_["k_after"] - k0 for o_ in outcomes))   # every fork's episodes count toward the budget
                    fit = best_c["fit"]
                    record_eval(k, fit, f"fork_select{best_c['i']}")
                    if fit["F"] > best["F"]:
                        best = {"episode": k, "F": fit["F"], "knobs": dict(knobs), "state": learners_state()}
                        torch.save({"episode": k, "F": fit["F"], "knobs": dict(knobs), "state": best["state"]}, out / "ckpt_best.pt")
                    rec["fork"] = {"chosen": best_c["i"], "candidates": [
                        {kk: v for kk, v in o_.items() if kk != "state"} | {"fit": {q: w for q, w in o_["fit"].items() if q != "per_episode"}}
                        for o_ in outcomes]}
                    rec["analysis"] = cands[0].get("analysis", "") if cands else ""
                    rec["wall_supervise_s"] = time.time() - t0
                    history.append({"decision": decision_index, "episode": k, "knobs": dict(knobs), "accepted": True,
                                    "chosen_style": best_c["style"], "F_before": fit["F"],
                                    "candidates": [{"style": o_["style"], "knobs": o_["knobs"], "F": o_["fit"]["F"],
                                                    "tier": o_["fit"].get("tier"), "del_red_pct": o_["fit"]["del_red_pct"],
                                                    "speed_std_ratio": o_["fit"]["speed_std_ratio"],
                                                    "energy_loss_pct": o_["fit"]["energy_loss_pct"],
                                                    "rationale": o_["rationale"]} for o_ in outcomes],
                                    "rationale": best_c["rationale"]})
                    print(f"[sup ] {sup.name}: " + " | ".join(f"{o_['i']}:{o_['style'][:4]} F={o_['fit']['F']:.1f}/{o_['fit'].get('tier','?')[:4]}" for o_ in outcomes)
                          + f" -> keep {best_c['i']} ({best_c['style']}) knobs " + ", ".join(f"{kk}={v:g}" for kk, v in knobs.items()), flush=True)
                    window_start = k
                    next_decision_at = k + args.supervise_every
                elif proposes and k < args.episodes:
                    summary = build_summary(decision_index, k, args.episodes, knobs, fit,
                                            train_window_stats(window_start), history, args.backend, args.method,
                                            trends=train_trends(window_start))
                    t0 = time.time()
                    proposal = sup.propose(summary)
                    new_knobs, notes = clamp_proposal(proposal.get("knobs", {}), knobs)
                    rec["proposal"] = {kk: v for kk, v in proposal.items() if kk != "knobs"}
                    rec["proposed_knobs_raw"] = proposal.get("knobs")
                    rec["validation_notes"] = notes
                    changed = {kk: (knobs[kk], new_knobs[kk]) for kk in KNOBS if abs(new_knobs[kk] - knobs[kk]) > 1e-12}
                    accepted = bool(changed)
                    if changed and use_twin:
                        ps = policy_set()
                        fit_cur = fit if twin_pool is pool else evaluate(twin_pool, twin_base, knobs, ps)
                        fit_new = evaluate(twin_pool, twin_base, new_knobs, ps)
                        rec["dry_run"] = {"F_current": fit_cur["F"], "F_new": fit_new["F"],
                                          "terminated": fit_new["terminated_any"]}
                        if fit_new["terminated_any"] or fit_new["F"] < fit_cur["F"] - args.rollback_drop:
                            accepted = False
                            rec["dry_run"]["rejected"] = True
                    rec["changed"] = changed
                    rec["accepted"] = accepted
                    rec["wall_supervise_s"] = time.time() - t0
                    if accepted:
                        knobs = new_knobs
                    history.append({"decision": decision_index, "episode": k, "knobs": dict(knobs),
                                    "accepted": accepted, "changed": {kk: v[1] for kk, v in changed.items()},
                                    "F_before": fit["F"], "rationale": str(proposal.get("rationale", ""))[:300]})
                    print(f"[sup ] {sup.name}: " + (", ".join(f"{kk}: {a:g}->{b:g}" for kk, (a, b) in changed.items())
                                                  if changed else "no change")
                          + ("" if accepted else "  [REJECTED]") + f" | {str(proposal.get('rationale', ''))[:160]}",
                          flush=True)
                log_decision(rec)
                window_start = k
    finally:
        pool.close()
        if twin_pool is not pool:
            twin_pool.close()
    json.dump({"final_knobs": knobs, "best_F": best["F"], "best_episode": best["episode"], "best_knobs": best["knobs"],
               "history": history, "wall_min": (time.time() - t_start) / 60,
               "llm_calls": getattr(getattr(sup, "client", None), "n_calls", 0),
               "llm_usage": getattr(getattr(sup, "client", None), "usage", None)},
              open(out / "summary.json", "w"), indent=1, default=float)
    print(f"done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
