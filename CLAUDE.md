# Region-aware Agentic RL on Wind Turbine

Residual-RL pitch control for the NREL 5 MW turbine: ROSCO GSPI baseline + operating-region-specialised
PPO residual agents (R2/R3 split by an oracle rule), with an LLM supervisor tuning six reward/action
knobs at a slow timescale. Baseline paper: Wang/Dong/Zhao, IEEE TSTE 2026 (`RelatedWorks/`, not in git).

## Read these first
- `docs/REPORT_2026-09-01.md` — consolidated, verified findings F1–F7 + final disposition (**start here**).
- `docs/roadmap_2026-08-30.md` — day-by-day experiment log, all intermediate tables (sections 1–22).
- Design decisions history: the user's explicit answers are recorded in the report; do not re-ask them.

## Scope
This repository covers region-aware residual RL for **collective pitch**, agentic (LLM)
supervision, the GSPI / LPV-MPC / ROSCO-tower-damper baselines (roadmap §1–23), and the
**individual-pitch negative result** (`docs/RESULTS_2026-09-05_ipc-negative.md`), which is kept
because it bounds the supervision claim. **Do not describe research directions beyond that scope**
in this repo's docs, commits or manuscripts. The IPC/torque code paths stay in the tree (inert at
`--ipc_max 0` / `--dtau_max 0`) because run configs reference them.

## Final state (2026-09-15; manuscript v4, MPC + agent-supervised residual main line)
Objective: **J** = mean of the baseline paper's four % reductions vs paired GSPI (power MSE,
gen-speed MSE, tower DEL, blade DEL; per-episode clipped ±100), −20 per % energy loss over 1 %, no
tiers, wind-labelled R3 subsets (`docs/roadmap_2026-09-11_metric-set-objective.md`, decisions D1–D6).
F (tower-DEL priority with constraint tiers) stays computable for the historical tables.
- **The model-based reference is the strongest controller on J, and it must be evaluated as the
  base controller** (roadmap v2 s18): the 3-state LPV-MPC with its cost re-scaled so the tower term
  works (`--scale v2`, qt = 3) scores **16.3 / 17.2** on the two held-out sets as the environment's
  base with a zero residual (`evaluate.py --gspi --run <mpc-base run>`), every term positive; with
  Cp/Ct x0.95 in its model 8.5 / 3.6, +-15 % -> -24. Pushing the MPC through the RL's action channel
  (`mpc_baseline.py`, the old 12.4 / 14.1 and 12.0 / 11.6 rows) handicaps it: the channel's R2
  non-negativity rule chops its negative offsets at region transitions. The F-era "MPC loses tower
  DEL -17.6 %" was a cost-scaling artefact (speed term O(1e-5) vs tower term O(1)).
- **Offset-free / adaptive MPC is now the strongest controller on J** (roadmap v2 s19, `adapt` in
  `controllers/mpc.py`, tau 5 s): 20.5 / 21.7 exact, 21.7 / 21.9 (offset) and 20.4 / 21.6 (RLS) with Cp/Ct x0.95,
  RLS flat 20.4-20.7 over +-15 %. It beats every residual run and refutes manuscript v4's main line
  ("residual adds to the MPC and repairs its model"); x0.95-trained residuals do not transfer to x0.9 / x1.05.
  TCST campaign gate STOPPED before extra seeds; the next direction is the user's decision.
- **Control direction (2026-09-16/17, roadmap s22-s26):** every MPC row buys its J with ~7.6x the GSPI pitch
  travel (s22); the travel is broadband, not 3P (a 3P notch removes the line, cuts travel 2-3 %, costs 1.5-2.4 J,
  s25); the J-vs-actuator-duty front from the 600 s search cache has its knee at ~3x GSPI travel, and that knee
  holds held-out: same J as the J-selected point at 38-41 % of its actuation, blade DEL 3 -> 5-6 % (s26). On a
  compensated base the residual has nothing of size left to learn (5 of 6 mismatched runs never beat episode 0,
  s24) and residuals do not transfer across model error while the estimator holds +-15 % (s23). ROSCO's tower
  damper is worth ~0 J at 600 s (s23). Literature scan: `docs/literature_2026-09-16.md`.
- **Wind-range protocol (2026-09-18, roadmap s27-s28; the main evaluation set from here on):** 12-24 m/s in 2 m/s
  steps, IEC turbulence class B, 600 s, TurbSim seeds 1-6 (42 held-out episodes; `campaign_must1_range.sh`,
  `scripts/dev/range_table.py`, `docs/tables/range_*.csv`). Several s19-s26 statements are specific to the
  low-turbulence 8 / 12.5 / 15 m/s set: over the range the MPC rows travel 1.7-2.2x the GSPI (not 7.6x), Cp/Ct x0.95
  costs the nominal MPC 0.7 J (not half), offset-free compensation is +1.0 J but a trade (+5.6 power, +5.9 speed,
  -6.6 tower, all intervals exclude 0) and every compensated MPC has a negative tower term from 20 m/s up while the
  nominal MPC never does. Report per-wind-speed terms; differences between deterministic controllers are paired
  bootstraps over episodes (`scripts/dev/paired_bootstrap.py`).
- **Tuned ROSCO baseline (roadmap s29, s31; `scripts/rosco_tune.py`, template `~/wtrl/runs/template_5mw_rosco_tuned`):**
  the shipped speed-filter corner (1.57 rad/s) is the untuned part of the GSPI; 3.42 rad/s with kp x1.13 scores
  16.2 / 17.1 on the low-turbulence held-out sets and 15.4 on the range set at 1.1x pitch travel. Against it the
  MPC rows keep 4.5-5.4 J at low turbulence (tower fatigue near rated only) and ~14 J at class B over the range
  (23-24 points of regulation plus 9 of tower). **Every comparison table must carry this row.**
- **Residual line closed from the learning side (s30):** the residual trained over Cp/Ct in [0.85, 1.15] repairs
  only the underestimate side, nothing at +15 %, and costs 3-6 J at the exact model; the estimator is 10-40 J above
  every residual at every error. The tower weight is not the lever for the compensated MPC's negative high-wind
  tower term (s31): qt 6 makes it non-negative at -2.2 J and +20 % travel, larger weights destabilise 12 m/s;
  under a per-wind load constraint the nominal MPC (28.5) beats the constrained compensated MPC (27.3).
- **Residual on the MPC base** (`--base mpc [--base_mm_cp 0.95]`, n = 3, s17-s18): adds +0.3 (fixed
  reward) / +1.0 (LLM reward) to the exact-model MPC, within noise, half the runs never beat episode 0;
  repairs the x0.95 model (8.5 / 3.6 -> 14.2 / 15.4 fixed, 12.7 / 18.3 LLM). LLM reward vs fixed on
  the MPC base: +0.6 / +0.8 (2/3), -1.5 / +2.9 mismatched -- no level effect on the strong base either.
- **RL under J, fair default** (reward v3 = speed term gated by the wind label; weights J-tuned;
  n = 3 each): guard 5.8 / 6.8, llm_hparam 8.2 / 9.9, random_hparam 7.4 / 8.3, llm_reward 8.3 / 9.2.
  Agentic gains are +1.5…+3 J (not significant at n = 3); the LLM proposer equals random search in
  the hyper-parameter namespace; **llm_reward is the only RL arm positive on all four terms**
  (saturated speed term, centred bounded load terms — a change of reward shape the weight search
  cannot reach). Final n = 5: fixed 4.86 / 5.50, llm_reward 5.30 / 6.04 (+0.4 / +0.5, 3/5 balanced),
  told-F 7.00 / 8.33, combo 6.37 / 8.65 (0/5 balanced), random reward structure (n = 3) 6.13 / 7.06.
  **Manuscript (docs/manuscript/main.tex, v4, user instruction 2026-09-15)**: main line = the agent-supervised
  residual improves the MPC (+1.0 / +1.6, all terms positive) and repairs its x0.95 model (8.5 / 3.6 ->
  12.7 / 18.3); the agent's reward shapes (saturated speed + power terms, bounded/centred fatigue terms).
  Negative results only in the Ablations section (fixed reward on the MPC base ~ same level; GSPI base:
  single-shot, random vocabulary) or in appendices never cited from the main text (hyper-parameter /
  combined / told-F arms, gating defect). arXiv style `docs/manuscript/arxiv.sty`. No development shorthand.
- **Objective decides the lever** (roadmap v2 §9 vs 08-30 §21): on F the reward-weight *proposer*
  mattered (llm_fork > random_fork 7/7); on J the untuned default collapsed into a rollback loop
  (12.5 m/s speed-MSE), hyper-parameter search repaired it (+6…+8, proposer-agnostic), and once the
  default is fixed the effect shrinks. Mechanism chain: critic scale (`--value_norm`), credit window
  (γλ), reward gating by the objective's own subset.
- F-era headline, kept as history (08-30 roadmap §21–§23, REPORT_2026-09-01): llm_fork 17.89 ± 2.02
  (7 seeds) > random_fork 12.36 ± 5.30 in 7/7 seeds (exact p = 0.0156); fork verification adds nothing
  on top of the LLM proposer; every paired comparison replicates on S7–S10; F1 = mono runs at +0.05
  speed-std ratio in 10/10 seeds. All of it is one wind bank (caveat C1); never regenerate it.
- Data: WSL `~/wtrl/{exp,wind,baselines/openfast}`, copy at `../wtrl-data/`; J tables via
  `scripts/dev/j_table.py [--csv]`, F-era tables in `docs/tables/` (`scripts/dev/build_tables.py`);
  provenance and caveats C1–C12 in `docs/RESULTS_2026-09-10_data-and-caveats.md`.

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
- Metric-set objective (roadmap v2, 2026-09-11): add `--objective J --reward v2 --value_norm` (guard-v1
  control: `--reward v1`); supervisors `llm_hparam | random_hparam | llm_reward` need `--n_candidates 3`.
  Runs are pausable: `--resume_every_s 300 --resume_keep 5` write `resume_*.pt`, `--resume` continues;
  `scripts/wsl/campaign_ctl.sh {pause|resume|stop|status}` drives a whole campaign session.
- Evaluate held-out: `scripts/evaluate.py --run <dir> --ckpt ckpt_best.pt --backend openfast --seeds 3 4 5 6 --tag <tag>`
  (J runs are scored with wind-labelled R3 subsets automatically; prints J and the four terms)
- Tables: `scripts/summarize.py`, `scripts/dev/{heldout_table,paper_table,per_wind_table,eval_detail,power_mae_table,pick_by_J}.py`
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
