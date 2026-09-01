# Region-aware Agentic RL on Wind Turbine

Residual-RL pitch control for the NREL 5 MW turbine: ROSCO GSPI baseline + operating-region-specialised
PPO residual agents (R2/R3 split by an oracle rule), with an LLM supervisor tuning six reward/action
knobs at a slow timescale. Baseline paper: Wang/Dong/Zhao, IEEE TSTE 2026 (`RelatedWorks/`, not in git).

## Read these first
- `docs/REPORT_2026-09-01.md` — consolidated, verified findings F1–F7 + open items (**start here**).
- `docs/roadmap_2026-08-30.md` — day-by-day experiment log, all intermediate tables (sections 1–12).
- Design decisions history: the user's explicit answers are recorded in the report; do not re-ask them.

## State at last session (2026-09-01)
- All local campaigns finished. Headline: 9/9 supervised spec runs beat GSPI on all four paper metrics
  on held-out wind (strict tier); mono 0/9; the three supervised variants (llm_fork / random_fork /
  schedule) are statistically tied at 3 RL seeds. No MPC baseline exists.
- Raw results live on the Windows laptop in WSL `~/wtrl/exp/*` (not in git; ~GBs). Wind bank
  `~/wtrl/wind` is the canonical one — copy it when migrating, or regenerate (then old comparisons
  are not seed-paired).
- Next steps (priority): ≥5–8 seeds to separate the supervised variants (Slurm/cloud); MPC baseline;
  R2 torque residual (1-line ROSCO patch); Eureka-style reward-code route; learned router.

## Environment / how to run
- Linux/WSL: `bash scripts/wsl/bootstrap.sh` from the repo root inside WSL Ubuntu-24.04 — builds the
  micromamba env `wtrl` (openfast 4.2 conda-forge, torch-cpu, sb3, fatpack), compiles the patched
  ROSCO (`controllers/rosco_patch/` → 22-channel ZMQ), makes the toy DISCON + OpenFAST case template,
  generates wind S1–S6 + GSPI baselines, and writes `~/wtrl/run.sh` for the current repo path.
  Everything then runs as `~/wtrl/run.sh python scripts/...`.
- macOS: untested; the same conda-forge packages exist — follow bootstrap.sh step by step, expect to
  adjust the ROSCO build (gfortran via conda) and skip WSL-specific notes.
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
