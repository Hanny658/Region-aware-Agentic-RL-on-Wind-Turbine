# Region-aware Agentic RL on Wind Turbine

Residual reinforcement learning for collective-pitch control of the NREL 5 MW turbine in OpenFAST,
built on top of ROSCO's gain-scheduled PI (GSPI): operating-region-specialised PPO residual agents
(R2 below rated / R3 above rated, split by an oracle rule) with an LLM supervisor tuning six
reward/action knobs at a slow timescale. Baseline paper: Wang/Dong/Zhao, IEEE TSTE 2026
(Region-III-only residual RL on the IEA 15 MW); our extensions are the region split, the
constraint-tiered evaluation, and the agentic supervision layer.

**Scope**: region-aware residual RL for collective pitch, agentic (LLM) supervision, the
GSPI / LPV-MPC / ROSCO-tower-damper baselines, and the individual-pitch negative result that bounds
the supervision claim.

Detailed, dated records: `docs/README.md` (index), `docs/REPORT_2026-09-01.md` (verified
findings F1–F7), `docs/roadmap_2026-08-30.md` (day-by-day experiment log, §1–23),
`docs/litreview_schedule_paradigm_2026-09-01.md` (supervision-paradigm literature review).

## System

- **Plant**: NREL 5 MW onshore, OpenFAST 4.2.1, ROSCO v2.10.5 patched (22-channel ZMQ measurements,
  torque-offset application, WSE sees applied torque). Constant torque above rated (paper-consistent),
  PS_Mode 1, 10 ms control step. A 1-DOF toy twin drives the same `libdiscon` for cheap screening.
- **Architecture**: GSPI + region-gated residual Δβ (oracle rule: native pitch cmd > current pitch
  floor + 0.5° held 1 s ⇒ R3, with hysteresis), second-order critically damped smoothing, safety
  clamps. One PPO per region (`spec`); `mono`/`mono_flag`/`r3only` as controls.
- **Unified region-conditional reward**: R2 power term, R3 speed term, load term
  `−λ_L · range_inc` (10 s trailing peak-to-peak increment of the load signal), action penalty.
- **Fitness F (unmodifiable by any learner/supervisor)**: see *Evaluation metrics* below.
- **Supervisors** (every 30 episodes, evaluated on train-side wind seeds): `guard` (fixed knobs),
  `llm_fork` (LLM proposes 3 candidates, each fork-trained one wave and verified, best kept),
  `random_fork` (same forks, random candidates), `schedule` / `schedule_comp` (replay of a
  distilled knob curriculum, episode- or competence-indexed).

## Evaluation metrics

**Ground-truth fitness F** (`eval/fitness.py`; constraint form, fixed 2026-08-29; no learner or
supervisor can modify it). Everything is measured on deterministic evaluation episodes against the
seed-paired GSPI baseline (same wind file, same TurbSim seed):

```
F = DEL_red_pct − 20 · max(0, energy_loss_pct − 1.0) − 20 · max(0, 100 · (speed_std_ratio − 1))
```

- *objective* `DEL_red_pct`: percentage reduction of the target damage-equivalent load —
  tower-base fore-aft `TwrBsMyt` (m = 4) for the tower objective, blade-root out-of-plane
  `RootMyc1`/`RootMoop` (m = 10) for the blade objective; rainflow-counted (fatpack).
- *constraint 1*: episode energy loss vs GSPI ≤ 1 % (all episodes);
- *constraint 2*: generator-speed std not worse than GSPI on R3-dominated episodes (≥ 50 % R3 steps);
- every percentage point of constraint violation costs 20 points of DEL reduction.

**Tiers**: **strict** = speed std ratio ≤ 1.0 and energy loss ≤ 1 % (non-inferior to ROSCO);
**tol2** = speed std ratio ≤ 1.02 (engineering-equivalent); **degraded** otherwise. `F_strict`
applies the penalty at ratio 1.0, `F_tol2` at 1.02; both are always reported, and any
"beats GSPI" claim in this repo is bound to its tier.

**Baseline-paper metric set** (Wang/Dong/Zhao): power-output MSE, generator-speed MSE (with MAE as
companion), tower-base fore-aft DEL, blade-root out-of-plane DEL — all reported as percentage
reduction vs GSPI (`scripts/dev/paper_table.py`); energy loss is our addition (the Region-III-only
paper holds torque constant and does not track it).

## Results under the baseline paper's metric set (J, 2026-09-11/12)

From 2026-09-11 every run is scored, selected and rolled back by **J** = mean of the four paper
metrics (% reduction vs paired GSPI of power MSE, generator-speed MSE, tower-base DEL, blade-root
DEL; per-episode clipped ±100), −20 per % of energy loss over 1 %, no tiers, R3 subsets labelled by
wind speed (`docs/roadmap_2026-09-11_metric-set-objective.md`). Held-out on two disjoint wind sets
(S3–S6 / S7–S10), best-by-J checkpoints, three seeds per RL arm:

| controller | J S3–S6 | J S7–S10 | four terms (S3–S6) P / ω / T / B |
|---|---|---|---|
| **LPV-MPC, tower term active, through the RL's ±2.9° residual channel** | **11.97** | **11.61** | 8.5 / 20.5 / 13.2 / 5.6 |
| LPV-MPC, same cost, wide-open channel | 12.40 | 14.10 | 8.9 / 19.8 / 15.8 / 5.1 |
| llm_reward (LLM-written reward, reward v3, J-tuned default) | 8.30 ± 1.93 | 9.17 ± 1.13 | 9.3 / 15.7 / **4.3** / **3.9** |
| llm_hparam (LLM tunes PPO hyper-parameters) | 8.22 ± 2.20 | 9.89 ± 1.90 | 18.8 / 23.5 / −10.9 / 1.5 |
| random_hparam (same fork search, random candidates) | 7.37 ± 1.02 | 8.32 ± 0.80 | 16.5 / 20.5 / −9.9 / 2.3 |
| guard (fixed knobs, reward v3, J-tuned default) | 5.76 ± 1.76 | 6.81 ± 1.29 | 18.2 / 21.1 / −17.3 / 1.1 |
| guard (fixed knobs, reward v2, untuned) | 1.59 ± 0.87 | 0.97 ± 1.37 | −6.9 / −0.6 / 10.4 / 3.4 |
| ROSCO tower damper (best gain under J) | −0.06 | −0.26 | ≈ 0 |
| LPV-MPC regulation-only (the F-era tuning) | −1.64 | −8.08 | 7.5 / 4.2 / −17.6 / −0.7 |

What changed relative to the F-era results below:
- **The MPC's tower-DEL loss was a cost-scaling artefact.** With the speed, pitch-rate and
  tower-velocity terms scaled to O(1) at their typical values, the tower weight can act
  (qt = 3) and the same 3-state LPV-MPC is positive on all four metrics, F-strict 15.8 / 18.3,
  and ahead of every RL arm — also when driven through the RL's own bounded, damped residual
  channel (authority explains only 0.4–2.5 J of its lead).
- **Residual RL needs three things to be competitive on J**: a normalised value target
  (`--value_norm`; the critic was a constant before), a speed term gated by the wind label the
  objective uses (`--reward v3`; otherwise training collapses into a rollback loop at 12.5 m/s),
  and one agentic lever on top. On that fair default the agentic gains are +1.5…+3 J (n = 3,
  not significant); the LLM proposer equals random search for hyper-parameters; the LLM-written
  reward is the only RL variant positive on all four terms (a saturated speed term and centred,
  bounded load terms — a change the weight search cannot reach, and a heavier tower weight
  λ_T ∈ {3, 10, 30} does not reach either).
- Tables: `docs/tables/table_J_stage3.csv`, `table_J_fairness.csv` (`scripts/dev/j_table.py --csv`).

## Headline results under F (CPC + region-aware, held-out wind S3–S6, best checkpoint) — historical

**1. Region specialisation buys speed regulation, not load (F1, restated 2026-09-10).** `spec + guard`
and `mono` are indistinguishable on load: 13.93 ± 2.29 vs 13.80 ± 2.71 (paired p = 0.96) and
12.54 ± 1.44 vs 12.25 ± 3.10 on a second, disjoint wind set (p = 0.89). What separates them is the
constraint they spend: **mono runs at +0.05 generator-speed std relative to spec, in 10/10 seeds
across both wind sets** (paired t p = 0.0202 and p = 0.0066). The tier count that earlier stated
this (`mono` 0/5 strict vs 5/5) is *not* a robust statistic — on the second wind set mono is 4/5
strict, because those realisations lower every controller's speed ratio by ~0.05 and mono sits
within 2 % of the boundary. Report the paired ratio; let the tier follow from it.

**2. The proposer is what matters — not the verification (F5, revised 2026-09-09).** The earlier
"the three supervised variants are tied" result paired seeds across two different wind banks
(caveat C1). Redone on one bank, with 7 seeds for the two fork arms:

| supervisor | n | F per seed | mean ± std | strict |
|---|---|---|---|---|
| **llm_fork** | 7 | 18.3, 17.6, 19.4, 18.0, 13.6, 19.7, 18.5 | **17.89 ± 2.02** | 6/7 |
| llm, single proposal (no fork, no dry run) | 5 | 16.1, 17.0, 17.9, 18.4, 20.3 | **17.92 ± 1.57** | 4/5 |
| schedule replay | 5 | 14.1, 20.1, 13.8, 18.8, 11.3 | 15.61 ± 3.68 | 5/5 |
| guard (fixed λ) | 5 | 11.8, 16.0, 13.4, 16.7, 11.9 | 13.93 ± 2.29 | 5/5 |
| random_fork | 7 | 13.2, 3.1, 18.8, 16.5, 8.1, 12.0, 14.7 | 12.36 ± 5.30 | 7/7 |

`llm_fork − random_fork` = **+5.53 ± 4.64, positive in 7/7 seeds, paired t p = 0.0196, exact
sign-flip permutation p = 0.0156**; `llm_fork − guard` = +3.48, p = 0.041. But `llm_fork −
llm single-proposal` = −0.52, **p = 0.76**: fork verification adds nothing to a good proposer, and
the earlier claim that unverified LLM supervision is *harmful* does not replicate. Nor does
supervision buy compliance — `random_fork` is the most compliant arm (7/7 strict) and the worst on
load; the guardrail plus best-checkpoint layer, which every arm has, is what delivers tiers.

**2c. Where this stops working (roadmap §23).** The supervisor's advantage is not a general
property of LLM supervision. On the individual-pitch (dq cyclic) channel the same fork-verified
supervisor — same prompt, same knobs, same guardrail — lost to fixed weights in three consecutive
companion campaigns (7.9 ± 2.9 vs 9.2 ± 0.9; 8.7 ± 3.4 vs 11.3 ± 1.3; 7.6 ± 1.2 vs 11.0 ± 1.0), with
a reproducible failure attractor: λ_load_R3 driven to 18–20 and w_speed to 360–400 while the
channel's own authority is strangled. Fork verification cannot see it, because every individual step
stays locally acceptable while the trajectory drifts. The reason is that on that axis the bottleneck
is discovery, not reward weighting — the channel is physically worth −14…−22 % blade DEL (a static
1° tilt and a hand-tuned classical controller both show it) yet trained policies used ≤ 11 % of its
authority across six mechanism variants. **Supervision helps where its knobs are the binding
constraint, and not elsewhere**; unlocking the cyclic channel is future work.

**2b. Why (mechanism, `scripts/dev/fork_analysis.py`, 35 fork decisions per arm).** The LLM's
candidate *sets* are good before any verification (mean fork F **+4.93** vs **−26.2** for random;
2.2 of 3 candidates strict vs 1.2), and its accepted moves are 2.3× larger and never a hold
(0 % vs 29 % for random). Over a run it executes a coherent curriculum — λ_load_R3 **6.5×**,
w_speed **7.1×**, Δβ_max_R3 **0.20×** relative to the defaults — while random search ends within
0.94–1.20× of where it started on every knob. Random's verifier spends 29 % of its decisions
retreating to "change nothing", which is why that arm is simultaneously the most compliant and the
least effective: **verification substitutes for proposal quality rather than compounding with it.**

**3. Schedule replay is robust — because of the protection layer (roadmap §14).** Replaying a
distilled knob curriculum on fresh seeds: episode-indexed 15.6 ± 3.7, competence-indexed
15.1 ± 4.0, both 10/10 strict, matching llm_fork at zero API cost — contradicting IPBT's
(arXiv 2511.09190) negative RL replay result, but only thanks to the violation-rollback guardrail
plus best-checkpoint selection (1–6 rollbacks per run). Competence re-indexing adds nothing when
the guardrail is active. Variance is a property of the curriculum content, not the paradigm
(an earlier σ = 0.4 claim was retracted).

**4. R2 torque residual: clear negative (roadmap §15).** Seed-paired 5 + 5:

| arm | F mean ± std | strict | energy loss |
|---|---|---|---|
| pitch only | 13.9 ± 2.3 | 5/5 | 0.26 % |
| + torque (±2000 Nm, gated, KE-exact reward) | 4.1 ± 2.9 | 5/5 | 0.05 % |

Paired p = 0.007. The channel nearly eliminates an energy cost that was never binding, while the
extra action dimension dilutes pitch learning within the 300-episode budget. Two mathematically
real reward loopholes were identified and guarded on the way (overspeed farming through the
region label; kinetic-energy draining), plus one critical infrastructure bug
(action-buffer misalignment for multi-dim actions) — see roadmap §15 for the honest chronicle.

**5. LPV-MPC baseline (roadmap §16).** Comparison baselines are **GSPI + MPC**, joined by
ROSCO's own tower damper in item 7 (decision
2026-09-03; the Wang et al. paper numbers stay a method reference, not a numeric baseline). The MPC is a
3-state LPV controller (rotor + first tower fore-aft mode, Cp/Ct-table linearisation each 0.1 s,
OSQP, live peak-shaving floor, no preview), tuned on the supervisor winds by F — the same model
selection every method gets. Held-out S3–S6:

| method | F | Pwr MSE ↓% | Spd MSE ↓% | TwrBsMyt DEL ↓% | RootMyc1 DEL ↓% | Eloss % | spd std ratio |
|---|---|---|---|---|---|---|---|
| GSPI (ref) | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 |
| LPV-MPC (tuned) | **−17.6** | +7.5 | +4.2 | **−17.6** | −0.7 | −0.30 | **0.912** |
| spec + guard (5 seeds) | +13.9 ± 2.3 | +6.2 | +6.2 | +13.9 | +4.4 | +0.26 | 0.968 |
| spec + schedule (5 seeds) | +15.6 ± 3.7 | −2.4 | +6.0 | +15.6 | +5.1 | +0.36 | 0.968 |
| spec + llm_fork (2 local seeds) | +15.8 | +1.7 | +1.6 | +15.8 | +5.4 | +0.33 | 0.991 |

The MPC reproduces (amplified) the classic trade-off: best regulation of all methods, paid for
with tower fatigue — a regulation-optimal pitch loop pumps the ~0.32 Hz tower mode. The
region-aware RL rows Pareto-dominate it on the combined objective. Ours is deliberately a simple
MPC (no preview, no load terms — fatigue is not expressible as a QP cost, and a naive
tower-velocity term drives the optimum to feathering); on the 1-DOF toy twin, where its model is
exact, the same MPC beats everything (speed std 0.03 vs GSPI 0.48) — its shortfall on OpenFAST is
model mismatch plus the non-quadratic objective, which is precisely the gap the residual-RL fills.

### R3 trajectories (held-out episode U15 TI8 S5)

Single above-rated episode, all three controllers on identical wind; in each figure the
best-scoring method (per-figure metric in the legend) is drawn on top. **Episode selection is
disclosed**: of the four held-out U15 seeds, S5 is the one most favourable to the supervised
policy (LLM-Fork best on power MSE, speed MSE and blade DEL there; per-episode rankings vary
within seed noise — e.g. on S3/S6 the MPC leads the MSE metrics). Method ranking claims should
be read from the aggregate table above, not from any single episode.

![R3 power](docs/figures/r3_power.png)
![R3 generator speed](docs/figures/r3_speed.png)
![R3 tower-base fore-aft moment](docs/figures/r3_tower.png)
![R3 blade-root out-of-plane moment](docs/figures/r3_blade.png)

Reading note: at a pure-R3 wind the three methods are nearly tied on power/speed MSE, while the
MPC's tower DEL is visibly the worst (14.6 vs 8.9/9.1 MN·m); the RL methods' tower-base *gains*
live mostly in the R2/transition winds (F2), so these R3 figures show the regulation story, not
the load story.

**5b. Everything above replicates on a second, disjoint held-out wind set (roadmap §22).** Four
fresh TurbSim realisations (S7–S10) with their own paired baselines, training untouched, 34 runs
re-scored: all six paired comparisons keep their sign, significance and effect size
(llm_fork − random_fork = +4.97, 7/7 seeds, t p = 0.0106, exact p = 0.0156). Absolute F falls
1.4–2.2 for *every* arm — the level belongs to the wind, the differences belong to the methods.

**6. Robustness under stress classes (roadmap §17, evaluation only).** The trained policies
re-evaluated on wind never seen in training — TI14 @ {8, 12.5, 15}, TI22 @ 15 and U18 @ TI8
(out-of-distribution deep Region 3), seeds 3–4, paired GSPI baselines per class, best checkpoint.
Tower-base DEL reduction, mean ± std over RL seeds:

| class | spec + guard (5 seeds) | schedule replay (5 seeds) | LPV-MPC |
|---|---|---|---|
| TI8 (held-out reference) | 13.93 ± 2.29 | 15.61 ± 3.68 | −17.6 |
| TI14 | 14.12 ± 2.42 | 15.23 ± 4.03 | −8.9 |
| TI22 @ 15 | 14.87 ± 4.42 | 17.81 ± 6.39 | −27.2 |
| U18 @ TI8 (OOD) | 3.86 ± 1.73 | 3.12 ± 2.42 | −56.1 |

The residual's gain is **invariant** under stress (differences inside the seed spread) and **41 of
42 stress evaluations stay strict** — speed std remains below GSPI in every class. The MPC is
negative in all four classes and worst out of distribution. At U18 the residual keeps only 3–4 %:
the tower gain is R2/transition peak-shaving (F2) and deep-R3 wind has no R2 to shave — a boundary
the manuscript states rather than hides.

**7. ROSCO's own tower damper is not a stronger baseline (roadmap §18).** Sweeping `TD_Mode=1`
over `FA_KI` ∈ [0.001, 0.3] with the same model-selection rule as the MPC: **no configuration
reaches the strict tier**. +0.6 % tower DEL already costs 0.2 % speed std, +4.9 % costs 3.1 %, and
beyond `FA_KI` = 0.1 the loop destabilises (speed std 2.2× GSPI). ROSCO's damper sits on the same
Pareto trade-off as the MPC — buy tower fatigue with speed regulation — while the region-aware
residual moves both. Held-out, all three reference controllers side by side:

| controller (held-out S3–S6) | F | tower DEL ↓% | speed-std ratio | tier |
|---|---|---|---|---|
| GSPI (reference) | 0 | 0 | 1.000 | — |
| GSPI + ROSCO tower damper (F-optimal) | −4.5 | +0.7 | 1.003 | tol2 |
| LPV-MPC (tuned) | −17.6 | −17.6 | 0.912 | strict |
| **spec + guard residual (5 seeds)** | **+13.9 ± 2.3** | **+13.9** | **0.968** | **5/5 strict** |

**8. Hard-won implementation facts (F3, F4, F7).** PPO γ must be 0.998 at 10 ms steps (0.99 is
myopic w.r.t. the ~3 s tower mode and every method fails); the load reward must be the trailing
peak-to-peak increment (`range_inc`) — |M| and |ΔM| both destabilise; λ_L has a cliff (start ≤ 1,
raise only after competence — the curriculum effect, first found by the LLM); `ckpt_last` is
frequently degraded, best-checkpoint selection is essential; the toy twin screens mechanisms and
rewards but its policies do not transfer zero-shot.

**Boundaries.** Numbers are not directly comparable to the Wang et al. 15 MW results (different
plant and objective weights: we prioritise loads); their paper serves as method reference, not a
numeric baseline. Wind bank was regenerated on migration (2026-09-01);
cross-machine F values are not seed-paired. MPC rows use wind-relabeled pairing (both sides),
since the oracle region label keys off ROSCO's native command.

## Status

Experiment campaigns are complete; the repository is in **manuscript preparation**. Verdicts:

- **CPC + agentic supervision**: concluded at the 5-seed statistical budget — all supervised spec
  variants beat GSPI on held-out wind (strict tier), and the variants (llm_fork / random_fork /
  schedule) are statistically tied (roadmap §13–§14). Supervision helps; the supervisor's identity
  does not, and the verification machinery (not the proposer) is what guarantees compliance.
- **R2 torque residual**: clear seed-paired negative (roadmap §15).
- **MPC baseline**: the F-era verdict ("wins speed regulation, loses tower loads") was a
  cost-scaling artefact; with the tower term active the LPV-MPC is the strongest controller on the
  paper's metric set and F-strict (roadmap v2 §8b, §10). The RL arms match it only in their best
  seeds.
- **Objective J (2026-09-11)**: the manuscript is scored on the paper's four metrics; the agentic
  supervision claims are being re-established under it (roadmap v2 §9–§11, n = 3 so far).
- **Data**: every derived table is in `docs/tables/` and rebuilt by
  `python3 scripts/dev/build_tables.py` (standard library only) from the run artifacts in
  `~/wtrl/exp`. The run inventory, the environment and the reporting caveats C1–C12 are in
  `docs/RESULTS_2026-09-10_data-and-caveats.md`. The wind bank and paired GSPI baselines
  (64 realisations, 5.5 GB) live in WSL with a copy at `../wtrl-data/` — copy them, never
  regenerate them.
- **Open before submission**: seeds 3–4 for the J arms (n = 5 lifts the exact-test floor to
  0.0625) and the caveats in `docs/RESULTS_2026-09-10_data-and-caveats.md` §4.

## How to run

Linux/WSL: `bash scripts/wsl/bootstrap.sh` (micromamba env, patched ROSCO build, case template,
wind bank, GSPI baselines). macOS: same conda-forge stack, see `scripts/mac/finish_bootstrap.sh`
and the notes in `CLAUDE.md`. Training/evaluation entry points: `scripts/train.py`,
`scripts/evaluate.py`; campaign runners under `scripts/wsl/` and `scripts/mac/`; tables via
`scripts/summarize.py` and `scripts/dev/*_table.py`. LLM supervision needs an `.env` with
`LLM_BASE_URL / LLM_API_KEY / LLM_MODEL` (never committed).
