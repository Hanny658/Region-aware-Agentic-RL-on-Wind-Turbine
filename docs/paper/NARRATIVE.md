# The paper's narrative (2026-10-15)

**What this file is.** The paper's claim, its contributions, its section order, and which number and which
figure carry each section. It is the only place the *paper's* argument lives.

**Its relation to `../MAIN_LINE.md`.** `MAIN_LINE.md` stays the project's complete internal record - every
leg, every control, every null, and *what cannot be claimed*. This file is narrower by the user's decision of
2026-10-15: **the main text carries the positive results; the nulls and the negative results go to the
appendix** (`../appendix/APPENDIX_PLAN.md`). Nothing is deleted and nothing is restated more favourably than
it was measured. Where a positive sentence here has a measured limit, the limit is named in the same row of
the plan so it cannot be lost on the way into LaTeX.

**Status, 2026-10-15 evening.** `../manuscript/main.tex` **v7 is written from this file**, from scratch rather
than by editing v6, and compiles with MiKTeX (`latexmk -pdf main.tex`, 16 pages, 7 figures, 9 tables, no
undefined references). Its ten sections are §3 below in order; its appendices are `../appendix/APPENDIX_PLAN.md`
A-J (lettered A-K in the PDF because the IPC chapter is its own appendix). The three disclosure sentences of
§6 are in the paper's §9. What remains is in §8.

---

## 1. The claim

> **An LLM agent in the simulation loop tunes a shipped wind turbine pitch controller, repairs an untuned
> model-predictive one, and writes the reward structure of a learned layer - the same loop in all three,
> with no per-space machinery - and every candidate it proposes is a controller the plant can run.**

What the loop delivers, in the four published reductions, on wind seeds no part of the design saw:

- **On the shipped ROSCO GSPI:** a tuning file the controller can take that reduces generator-speed
  regulation error by **+23.0 %** and tower-base fatigue by **+8.7 %** at **exactly the shipped pitch duty**
  (1.000x travel, ADC 0.048 against 0.0476), found in 8 simulations; and a second row at **+28.4 % / +16.3 %**
  for 1.65x travel. Both hold every load term at every wind speed on 28 held-out *and* 28 fresh episodes.
  Random search finds nothing in 72 candidates.
- **On an untuned offset-free LPV-MPC:** the agent finds the wind schedule in 24 simulations and turns a
  controller that loses **8.8 points** of tower-base fatigue at 24 m/s into one inside the tolerance
  (**-8.76 -> -0.33**), for 1.7 points of regulation. Random search: 0 of 48.
- **On a learned residual layer:** the agent-written reward structure adds **+5.8 to +6.4** power MSE and
  **+5.9 to +6.8** speed MSE over the tuned baseline, three seeds, every paired interval clear of zero, with
  tower fatigue never negative.
- **Across 144 candidates on the real plant, none destabilised it** - against 13 of 144 for a local search,
  9 of 144 for Bayesian optimisation and 16 of 72 for random draws. Zero malformed proposals in six runs.

### Title candidates

1. *An LLM agent in the tuning loop of a wind turbine pitch controller: shipped-duty load and regulation
   reductions from the controller's own parameters*
2. *Agentic controller tuning for wind turbines: one simulation-verified loop on a shipped PI controller, an
   untuned MPC, and a learned residual layer*
3. *What an LLM agent is worth in wind turbine control design: tuning files the controller can take*

**Guardrail on the title (pre-registered, roadmap s59.1):** the title may name the agent as the *object of
study* and may claim the *application* as new. It may **not** claim or imply that the agent beats another
search method - that endpoint was fixed before the runs and came out null (appendix A1). Candidates 1-3 are
written to satisfy this; anything of the form "LLM-*guided* / -*optimal* tuning outperforms ..." does not.

---

## 2. Contributions, each with its novelty status

| # | contribution | evidence | novelty status |
|---|---|---|---|
| **C1** | **First LLM agent in the loop for wind turbine controller tuning.** The agent proposes complete ROSCO tuning files; OpenFAST and the constrained selection rule decide. | roadmap s52-s53, s59; `../tables/rosco_search.csv` | **Domain-first, defensible with "to our knowledge".** Four targeted searches (2026-10-15) and the 2026-09-23 audit found no LLM in any wind turbine control-design loop; the one conference record with a relevant title could not be verified (repository entry only) and is, by the user's decision, neither cited nor mentioned (§8). Mechanism prior art, in other domains, must be cited beside the claim: ControlAgent (arXiv:2410.19811), Shou et al. (arXiv:2607.26594, an LLM agent tuning PID for chemical processes), Chen et al. (Processes 14:2954, 2026, LLM gain-ranking benchmark), Wu et al. (Building Simulation 2026, LLM reward weights on a real energy system). |
| **C2** | **One loop, three design spaces, no rebuild** - ROSCO's own gains and filters, an MPC's cost structure, an RL reward expression - with no acquisition function, no population and no per-space surrogate. It beats random search in all three. | s53, s38, s49 | **New as a demonstration.** It is a portability claim, not a performance claim, which is why n = 6 does not threaten it. |
| **C3** | **What the shipped controller leaves on the table, in tuning files:** one line of the shipped file (`F_LPFCornerFreq`, a generic rule of thumb rather than an optimisation) is worth +23.0 % of regulation and +8.7 % of tower fatigue at shipped duty; the verified band is a **duty ladder** from 0.97x to 2.02x travel. | s52.3, s55.2-55.3; `../tables/rosco_band_verified.csv`, `rosco_band_rows_{fresh,heldout}.csv` | **Survives (audit A(iv)):** no CPC-only method beating a *tuned* ROSCO on regulation and both DELs was found. Needs the Abbas et al. WES 7:53 (2022) §2.1 citation for why the shipped corner is the untuned part. |
| **C4** | **Proposal admissibility: 0 of 144 candidates destabilised the plant**, 0 malformed replies in six runs, against 13/144, 9/144 and 16/72 for the classical and random arms (Fisher p = 0.00018, 0.0034, < 1e-5). The agent spends no simulation budget on controllers that cannot run. | s59.4; `../tables/rosco_proposer_test.csv`, column `n_diverged` | **New and large.** Reported as **post hoc** - s59.1 did not pre-register it. See the wording guardrail in §7. |
| **C5** | **The measurement protocol:** load constraints held per wind speed and *measured off the design winds*, paired stratified bootstraps over episodes, actuator duty beside every row, one aggregation rule. | s28, s52.3, s54, s55; `scripts/dev/paired_bootstrap.py` | **Above the field's bar** (audit §3.7-3.8). Stated as the discipline the results are measured under, in the setup and the discussion. |

---

## 3. Section order, and what carries each section

| § | what it argues | evidence | floats |
|---|---|---|---|
| **1 Introduction** | A shipped controller is the realistic starting point, its tuning is a rule of thumb, and tuning it is a search problem in a space a language model can read and write. Contributions C1-C5. | - | - |
| **2 Related work** | Four paragraphs: RL for wind pitch control; MPC for wind turbines; residual RL and gating; **LLM agents for control design and tuning**. The last one carries C1's attributions and says plainly what is prior art. | `literature_2026-09-23*.md` §1 A(i)-A(iv), D1-D3 | - |
| **3 Setup** | Plant, wind protocol, the four published reductions, the seed split, the paired statistics, the constrained selection rule, and the agent loop itself (slow timescale, JSON proposal, deterministic verification). | s28, `design_2026-10-11_rosco-native-search.md`, `RESULTS_2026-09-10_data-and-caveats.md` | T1, **F1** |
| **4 Leg 1 - the shipped pitch controller** | **The paper's centre.** The search space (five ROSCO parameters, every candidate a tuning file, every box limit read off the plant); what the agent finds and how fast; the verified duty ladder; the one-line result. | s51-s53, s55; `rosco_search.csv`, `rosco_band_verified.csv`, `rosco_band_rows_fresh.csv` | T2, T3, **F2**, **F3** |
| **5 Leg 2 - the untuned MPC** | The agent repairs a per-wind feasibility violation it did not design for, in 24 simulations, where random search does not. The set mean hides the entire repair - which is also C5 in action. | s33, s38, s57.2; `mpc_leg_fresh.csv`, `design_search.csv` | T4, F4 |
| **6 Leg 3 - the learned layer** | A searched reward *structure* buys the level on top of the tuned baseline; the wind gate is where the layer may act (prior art, attributed). | s41, s56; `gated_native.csv`, `gated_residual.csv` | T5, F4 |
| **7 The ladder, and what each rung costs** | The four reductions from shipped to the full stack, per term, with duty. The second rung is a trade, not an addition: no single decomposition exists. | s54; `ladder_published_fresh.csv` | T6, **F5** |
| **8 How the numbers were read** (discipline) | Aggregation convention and wind weighting; the per-wind constraint; actuator duty. v6's claim, demoted to method. | s42, s48, s52.3; `lifetime_del.csv`, `wohler.csv` | T7, F6 |
| **9 Discussion** | What transfers (the space and the protocol), what the loop costs, and - in one short paragraph - **what this does not show**, pointing at the appendix (see §6). | - | - |
| **10 Conclusion** | C1-C5, nothing new. | - | - |

v6's `main.tex` is the substrate: its §3 setup, §5.2 aggregation, §5.3 Wohler, §5.6 actuation and appendices
A-G are reusable almost as they stand. What changes is that §5's organising principle becomes the three legs
instead of the six reporting choices, and the composite leaves every reported number (reasons (A)-(D) of
`../archive/plan_2026-10-12_manuscript-rewrite.md`, still the correct site-by-site list).

---

## 4. Figures

| id | content | source | status |
|---|---|---|---|
| **F1** | **The loop**, one panel: proposer -> tuning file -> OpenFAST (7 winds x 2 seeds) -> four reductions + three constraints -> history -> next proposal. The deterministic half is boxed, so what the language model does and does not decide is visible. | - | **done**: TikZ inside `main.tex` (Fig. 1) |
| **F2** | **The front against duty**, two panels. (a) the design screen: all 485 distinct proposed tunings, regulation against pitch-travel ratio, split into the ones that lose a constraint, hold it on the set mean, and also hold both loads at every wind speed; log duty and symmetric-log regulation, so the long bad tail is shown rather than cropped. (b) the same x axis with fresh-seed regulation for the ten verified band members, filled where they hold per wind on both seed sets, the one-line row and stage 1 labelled. Replaces v6's `f5_duty.png`, which plotted the composite. | the campaign's `history.jsonl`, `rosco_band_verified.csv`, `rosco_tuned_baseline.csv` | **done**: `f5_front.png`, `scripts/dev/figures_leg1.py` |
| **F3** | **Search traces**, one panel per proposer: cumulative best feasible regulation against the simulation index, one line per run, the first feasible gain marked, censoring named in the panel title. This is the honest form of "fewer simulations" - the random arm never arrives. | `history.jsonl` under `~/wtrl/exp/roscosearch_native` | **done**: `f5_search.png`, same script |
| **F4** | **Per-wind terms**, the four reductions against mean wind speed for the four rungs, gate ramp shaded. Carries the MPC's tower-fatigue loss above 18 m/s and the learned layer's "nothing below the gate". | the rungs' own per-episode evaluations, joined on the wind file | **done**: `f5_perwind.png`, `scripts/dev/ladder_published.py --figdir` |
| **F5** | **The ladder**, four published reductions per rung with paired bootstrap bars. **No composite panel, and the hand-scaled baseline is not a rung.** | the same arrays as T6 | **done**: `f5_stack.png`, same script |
| **F6** | **Weighting**: Rayleigh weights for classes I-III, and the same controllers equally weighted and site weighted, **on the native baseline** (the v6 panel was keyed on the hand-scaled row). | `lifetime_del.csv` as `lifetime_del.py --csv` now writes it | **done**: `f5_weighting.png`, `scripts/dev/figures_weighting.py` |
| **F7** | **Before and after, as levels** (the user's request of 2026-10-15): the per-episode quantities the reductions are computed from, bin means per wind speed, the shipped controller overlaid with the agent's two tuning files (top row) and the untuned MPC with the agent-scheduled one (bottom row), MSE panels logarithmic. The shipped level is derived from each evaluation's own reduction and must agree across evaluations file by file; the script checks it (6e-16). | the four evaluations' own CSV/JSON pairs | **done**: `f5_overlay.png`, `scripts/dev/figures_overlay.py` |

F2-F5 are drawn from the arrays their tables are made from, not from a parallel path: `figures_leg1.py`
re-derives the first feasible gain of all 21 runs and **exits non-zero if any of them, or any arm's count and
median, disagrees with `rosco_proposer_test.csv` and s59.4**; `ladder_published.py --figdir` draws F4 and F5
inside the script that writes T6, so the figure cannot drift from the table.

    ~/wtrl/run.sh python scripts/dev/figures_leg1.py
    ~/wtrl/run.sh python scripts/dev/ladder_published.py --seeds s78910 \
        --csv docs/tables/ladder_published_fresh.csv --figdir docs/manuscript/figures

Retired from the figure set: `fig1_levers`, `fig2_composition`, `fig3_mechanism`, `fig4_reference`,
`fig_mpc_error`, `fig_mpcbase`, `fig_training`, `fig_trajectory` (v4 / F-era, superseded objective and
baseline). `../figures/{mpc600_search,weight_sensitivity}.png` are F-era. `../figures/pareto_duty.png` (the
duty knee, s26) and `../figures/r3_*.png` (R3 trajectories) are appendix candidates only.

## 5. Tables

| id | content | source CSV | generator |
|---|---|---|---|
| **T1** | Wind protocol and seed split: 12-24 m/s in 2 m/s steps, class B, 600 s simulated / 20 s discarded / 580 s scored; seeds 1-2 design, 3-6 held out, 7-10 fresh; episodes per set. | - | hand, from s28 + `RESULTS_2026-09-10_data-and-caveats.md` |
| **T2** | **The verified duty ladder.** Four band members and the shipped controller: four reductions with intervals, energy, travel ratio, ADC, worst single-wind term on both seed sets, and which parameters differ from shipped. | `rosco_band_rows_fresh.csv`, `rosco_band_rows_heldout.csv`, `rosco_band_verified.csv` | `scripts/dev/rosco_band_table.py` |
| **T3** | **The search, per arm.** Candidates proposed, distinct, unrunnable, runs reaching a feasible gain, simulations to the first one, regulation of the selection. | `rosco_search.csv`, `rosco_proposer_test.csv` | `scripts/dev/rosco_search_table.py`, `rosco_proposer_test.py` |
| **T4** | **Leg 2.** Unscheduled and agent-scheduled MPC in four reductions with the paired difference, plus tower DEL per wind speed at 16-24 m/s. | `mpc_leg_fresh.csv`, `mpc_leg_heldout.csv` | `scripts/dev/mpc_leg_table.py` |
| **T5** | **Leg 3.** The gated layer on the native baseline, three seeds, paired against its own base, both seed sets. | `gated_native.csv` | `scripts/dev/gated_native_table.py` |
| **T6** | **The ladder** in four published reductions, fresh seeds, with duty. | `ladder_published_fresh.csv` | `scripts/dev/ladder_published.py` |
| **T7** | **Three aggregations** of five controllers (native baseline, the agent's one-line row, the scheduled MPC, the two gated rungs): per-episode mean, level ratio equally weighted, Rayleigh at V_ave 10 / 8.5 / 7.5, for tower DEL and speed MSE. | `lifetime_del.csv` | `scripts/dev/lifetime_del.py --csv` (rows moved to the native baseline on 2026-10-15; the hand-scaled row is out) |

Appendix tables: `gate_proposer_control.csv` (A3), `range_residual.csv` (A3), `rosco_proposer_test.csv`
(A1-A2, the same file read under the shipping rule), `wohler.csv` (A5), `verification_probe.csv`,
`windobs.csv`, `budget.csv`, `pareto_duty.csv`, `pitch_spectrum.csv`, `actuation.csv`.

**Superseded; cite nothing from them:** `table_J_*.csv`, `table_heldout_main.csv`,
`table_heldout2_s78910.csv`, `table_mpc.csv`, `table_paper_metrics.csv`, `table_paired_stats.csv`,
`table_robustness.csv`, `table_torque.csv`, `table_wind_sets.csv`, `table_night1_SUPERSEDED_by_S21.csv`,
`tuned_rosco_residual.csv`, `all_evals.csv`, `knob_trajectories.csv`, `range_{controllers,paired,per_wind}.csv`
(the 42-episode range protocol over the superseded baseline), `mpc600_search.csv`, `weight_sensitivity.csv`.
They are F-era, or were produced under the composite and the hand-scaled baseline that s52 voided.

---

## 6. What the main text does not carry, and the one paragraph that stays

By the user's decision the nulls and the negative results are appendix material
(`../appendix/APPENDIX_PLAN.md`, items A1-A6). Two of them - A1 and A2 - were **pre-registered in roadmap
s59.1 before the runs existed**, which makes them different in kind from a result that simply did not make
the cut: a reader who learns later that a pre-registered endpoint was measured and not reported has grounds
to distrust everything else in the paper.

**Recommendation, costing three sentences in §9:** state that the proposer comparison against two dedicated
searches was pre-registered and came out null at n = 6, that under the shipping rule the classical arms
delivered more often than the agent arm, and that both are reported in full in appendix A1-A2. The positive
claims C1-C5 do not depend on either, so the disclosure costs the paper nothing and buys it the one thing a
null result can buy: the referee believes the rest.

## 7. Wording guardrails that survive into LaTeX

1. **Never** "the agent designs a better controller", "LLM-optimal", or any comparative against another
   search method on the level or on the speed of the search. A fixed recipe beats it on leg 1 and a hand
   design on leg 2, and both are measured (s53.2, s38).
2. The four published reductions only, each on its own, each against the **paired** shipped controller on the
   same wind file. The composite is the campaigns' *selection and training* objective, named as such, never a
   reported result. The symbol *J* is the MPC cost function.
3. The gate is **prior art** - Kim (arXiv:2609.21307), Abbas (IET CTA 2026) - attributed where it is
   introduced; the MPC weight schedule is Wintermeyer-Kallen et al. (2021); offset-free MPC in wind is Liu
   et al. (IEEE TII 2024). Our claim on each is the measurement, never the formulation.
4. C4 is **post hoc**, and it says that none of the agent's candidates destabilised the plant, i.e. it is
   about the admissibility of the proposals. It is **not** evidence that the delivered tunings are more
   robust - no gain or phase margin and no plant-parameter sweep was run on them.
5. Name the band member used in every comparison; the baseline is a band from 0.97x to 2.02x travel, not a
   row.
6. One aggregation rule everywhere: bin mean, then weighted power mean.
7. Site weighting is reported as the convention's effect, never as a result, and the weighted aggregates
   carry no interval. Against the **native** baseline (fresh seeds, `lifetime_del.csv`): the MPC's regulation
   advantage is +25.6 per episode and +13.7 / +6.6 / +1.9 under Rayleigh weights at V_ave 10 / 8.5 / 7.5 - it
   *shrinks*; its tower-fatigue disadvantage, -7.9 per episode and -4.5 as an equally weighted level ratio, is
   +2.8 / +5.2 / +6.9 under the same weights, because the weights favour the bins near rated. The v6 numbers
   "+24.6 per episode, -0.3 Rayleigh-weighted" were measured over the hand-scaled baseline and are void.
8. Report 600 s simulated / 20 s discarded / 580 s scored, and ADC against the 10 deg/s rate limit beside
   every travel ratio.

## 8. What remains before submission (2026-10-15)

1. ~~Open Teke, ICENSOS 2025~~ **Closed 2026-10-15:** only a repository record exists (no abstract, DOI or
   text; no 2025 proceedings volume online), so its content cannot be verified, and by the user's decision it
   is **removed from the manuscript entirely** - neither cited nor mentioned. C1 is printed as "to our
   knowledge", and the `\pending` is gone.
2. ~~Complete four author lists~~ **Done 2026-10-15:** all four arXiv pages are open access and were fetched
   directly; `references.bib` carries the full names (note the first author of arXiv:2609.20752 is
   ArjomandBigdeli; the bib key keeps the short form).
3. `main.tex` carries no `\pending`. Two overfull boxes remain (the band table at 34 pt inside its
   `resizebox`, the reward verbatim at 43 pt), cosmetic.
4. The Wohler appendix shows the MPC + layer row for seed 1 only: seeds 0 and 2 were evaluated before the extra
   exponents were recorded and would need `campaign_mexp.sh` re-run to appear (no result depends on it).
5. Authors and affiliations come from `authors.tex` (git-ignored); the anonymous fallback builds without it.
6. **Leg 1 under the margin screen (2026-10-16).** The s62.1 re-read is in app. E of `main.tex` and in
   `../appendix/APPENDIX_PLAN.md` D3: with one point of margin in the design-set screen the agent's
   selections ship in 2 of 6 runs (0 before) and the classical arms still at least as often, so no proposer
   claim. The campaign in which every arm selects under that screen and the agent is also told the rule
   (s62.2, s62.3, finished 2026-10-17) came out in the pre-registered branch that moves nothing into the
   main text: the agent drives to the screen's edge in all six runs and ships in 1 of 6; app. E carries it
   in full, including that the task text named the filter corner the earlier campaign had found, so the
   one-simulation time to the first screen-holding candidate measures the prompt and not the search.
   The rule was then completed (s62.4, finished 2026-10-19): the set-mean tower margin of 6.0 as a constraint
   for every arm, the leaked findings removed from the task text. On the existing runs 3 of its 6 new
   selections ship (all failures the 24 m/s tower cell); in the third campaign every one of its six selections
   ships (Bayesian optimisation 4 of 6, local 2 of 6) and the agent, told only the rule, proposes one
   regulation gain in 144 candidates and delivers 0 of 6 - the pre-registered branch again: one sentence in
   §9, app. E in full with two disclosures (the task text's history list carried no shipped row to anchor
   zero; the arms receive the rule asymmetrically). The precision condition for a §8 statement (0.9 with at
   least ten selections) was not met on count, so §8 keeps its four instances.
