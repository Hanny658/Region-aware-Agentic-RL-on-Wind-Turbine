# Region-aware Agentic RL on Wind Turbine

Residual-RL pitch control for the NREL 5 MW turbine: ROSCO GSPI baseline + operating-region-specialised
PPO residual agents (R2/R3 split by an oracle rule), with an LLM supervisor tuning six reward/action
knobs at a slow timescale. Baseline paper: Wang/Dong/Zhao, IEEE TSTE 2026 (`RelatedWorks/`, not in git).

## Read these first
- `docs/REPORT_2026-09-01.md` — consolidated, verified findings F1–F7 + final disposition (**start here**).
- `docs/roadmap_2026-08-30.md` — day-by-day experiment log, all intermediate tables (sections 1–21).
- Design decisions history: the user's explicit answers are recorded in the report; do not re-ask them.

## Scope (narrowed 2026-09-08)
This repository is the **collective-pitch half**: region-aware residual RL + agentic (LLM)
supervision + the GSPI, LPV-MPC and ROSCO-tower-damper baselines (roadmap §1–21). The follow-up research — IPC,
trajectory auditor, LLM-evolved symbolic laws, law+RL composition, IEA 15 MW — moved to its own
repository; the removed roadmap sections are kept at
`wtrl-migration/_second_paper_moved/roadmap_sections_17-25.md`. The IPC/torque code paths stay in
the tree (inert at `--ipc_max 0` / `--dtau_max 0`) because the migrated run configs reference them.

## Final state (campaigns complete; manuscript in preparation)
- Headline (revised 2026-09-09 after redoing everything on ONE wind bank, roadmap §21): the
  **proposer** is what matters — llm_fork 17.89 ± 2.02 (7 seeds) beats random_fork 12.36 ± 5.30 in
  7/7 seeds (paired t p = 0.0196, exact permutation p = 0.0156) and guard 13.93 ± 2.29 (p = 0.041).
  **Fork verification adds nothing on top of a good proposer** (single-proposal LLM 17.92 ± 1.57,
  p = 0.76 vs llm_fork), and "unverified LLM supervision is harmful" does NOT replicate. The old
  §13 "all supervised variants are tied" came from pairing seeds across two wind banks (caveat C1).
  mono is 0/5 strict vs spec's 5/5 while matching it on F_tol2 (p = 0.96): F1 is a constraint claim,
  not a mean claim. Mechanism in §21b (`scripts/dev/fork_analysis.py`).
- Baselines: LPV-MPC (§16) wins speed regulation at 150 s and loses tower DEL (−17.6 %), but its
  advantage is window-dependent — at 600 s its speed ratio crosses 1.0 (§19) — and it is negative in
  every stress class (§17). ROSCO's own tower damper reaches no strict configuration at any gain
  (§18). Every supervised spec variant Pareto-dominates both. R2 torque residual: clear seed-paired
  negative (§15). Robustness: the residual's tower gain is invariant TI8→TI22 and 41/42 stress
  evaluations stay strict, but it vanishes at U18 (no R2 to peak-shave) (§17).
- Data on this machine: `wtrl-migration/cpc-result/` (38 runs, MPC grid, aggregate rebuild script,
  caveats C1–C10) + `wtrl-migration/{wind,openfast}` (canonical wind bank + paired GSPI baselines).
  These are copied into `~/wtrl/{wind,baselines/openfast,exp}` inside WSL by the migration step —
  **never regenerate the wind bank**, or historical F values stop being seed-paired (caveat C1).

## Environment / how to run
- Linux/WSL: `WTRL_SKIP_WIND=1 bash scripts/wsl/bootstrap.sh` from the repo root inside WSL
  Ubuntu-24.04 (the flag skips wind/baseline generation when a canonical bank is being migrated) — builds the
  micromamba env `wtrl` (openfast 4.2 conda-forge, torch-cpu, sb3, fatpack), compiles the patched
  ROSCO (`controllers/rosco_patch/` → 22-channel ZMQ), makes the toy DISCON + OpenFAST case template,
  generates wind S1–S6 + GSPI baselines (skipped by `WTRL_SKIP_WIND=1`), and writes `~/wtrl/run.sh`
  for the current repo path.
  Everything then runs as `~/wtrl/run.sh python scripts/...`.
- macOS: works with the same conda-forge stack — see `scripts/mac/finish_bootstrap.sh` (gfortran
  via conda, `.dylib`→`.so` symlink for the DISCON, BSD sed differences).
- LLM credentials: `.env` at repo root with `LLM_BASE_URL / LLM_API_KEY / LLM_MODEL` — **never commit
  it, never print the key**. Model gpt-5.6-luna: use `max_completion_tokens`, `reasoning_effort`,
  JSON mode; `max_tokens` and `temperature≠1` are rejected.

## Key commands
- Train: `~/wtrl/run.sh python scripts/train.py --backend {toy,openfast} --method {spec,mono,mono_flag,spec_sc,r3only} --supervisor {none,guard,random,llm,llm_fork,random_fork,schedule} --episodes 300 --workers 8 --seeds 1 --eval_seeds 1 2 --lambda_load 1 --rollback_on violation --load_signal fa_acc --fitness_target tower --obs_fa_acc --port0 5800 --out ~/wtrl/exp/<name>`
- Evaluate held-out: `scripts/evaluate.py --run <dir> --ckpt ckpt_best.pt --backend openfast --seeds 3 4 5 6 --tag <tag>`
- Tables: `scripts/summarize.py`, `scripts/dev/{heldout_table,paper_table,per_wind_table,eval_detail,power_mae_table}.py`
- Campaign patterns (resumable, run-level skip on existing summary.json): `scripts/wsl/campaign_*.sh`.

## Hard-won implementation facts (do not rediscover)
- PPO γ must be 0.998 (10 ms steps; 0.99 is myopic w.r.t. the ~3 s tower mode and every method fails).
- Load reward proxy must be the 10 s trailing peak-to-peak increment (`range_inc`); |M| and |ΔM| both
  destabilise training. λ_L has a cliff: start ≤1 and increase only after the policy has learned.
- Fitness tiers: strict (speed std ≤ GSPI, energy ≤1 %) / tol2 (speed ≤1.02×) / degraded; always use
  best-checkpoint (ckpt_last is frequently degraded). LLM proposals require fork verification.
- Parallel runs need distinct `--port0` (ZMQ) per run; work dirs are isolated per run automatically.
- On this Windows setup pgrep -c was unreliable (use `ps aux | grep`), and long jobs must be launched
  as detached processes or the WSL VM dies with the session (irrelevant on native Linux/macOS).
