# Documentation index

**Start with [`MAIN_LINE.md`](MAIN_LINE.md).** It is one page: the claim, the three legs of evidence with their
current numbers, what cannot be claimed, and the measurement discipline. Everything in this directory is
evidence for a leg, discipline, or history - this index says which, so nothing has to be read to find out.

**Rule for anything written from here on:** a new experiment document opens with the leg it serves and one
sentence on which number in `MAIN_LINE.md` it changes. If it changes none, it is a note, not a document, and it
goes in the roadmap only. When a roadmap section does change a main-line number, update `MAIN_LINE.md` in the
same commit - that file is the only place the project's claim lives, and a stale claim is worse than a long log.

## By leg

| leg | what it asks | where the evidence is |
|---|---|---|
| **1. The shipped GSPI, tuned by the agent** | what agentic search is worth on a classical pitch loop, in tuning files the controller can take | roadmap v2 §51-§55; `design_2026-10-11_rosco-native-search.md`; `tables/rosco_{band_verified,tuned_baseline,perwind}.csv`, `tables/rosco_band_rows_*.csv` |
| **2. The untuned MPC, designed by the agent** | whether the agent can repair a model-based controller it did not design | roadmap v2 §33, §36, §38, §57; `tables/design_search.csv`, `tables/range_*.csv` |
| **3. The learned layer, and who writes its reward** | whether the level or the constraint is what the agent buys | roadmap v2 §32, §34, §35, §37, §39-§41, §47, **§49**, §56, §58; `tables/{gated_native,gated_residual,range_residual,mpc_residual}.csv` |
| **discipline** | why the three legs are believable | roadmap v2 §22, §25-§28, §42, §52.3, §54, §55; `RESULTS_2026-09-10_data-and-caveats.md`; `tables/ladder_published_*.csv`, `tables/{lifetime_del,wohler,verification_probe}.csv` |

## By file

| file | what it is | read it for |
|---|---|---|
| `MAIN_LINE.md` | the claim and the three legs, one page | **first, always** |
| `roadmap_2026-09-11_metric-set-objective.md` | the current experiment log, §1-§57, 2800 lines | the detail behind any single number; use the section map below, do not read it through |
| `design_2026-10-11_rosco-native-search.md` | why the first parameter search was asking the wrong question, and the redesign | leg 1's parameterisation, box limits and constrained selection |
| `plan_2026-10-12_manuscript-rewrite.md` | what has to change in `main.tex`, site by site, with four change reasons | before touching the manuscript |
| `RESULTS_2026-09-10_data-and-caveats.md` | data provenance, run inventory, caveats C1-C12 | before quoting any number anywhere |
| `RESULTS_2026-09-05_ipc-negative.md` | the individual-pitch chapter, a documented negative | the boundary of the supervision claim: where the same supervisor stops winning |
| `literature_2026-09-23.md` + `_wind-rl-mpc` / `_residual-llm` / `_conventions` | the novelty and prior-art audit | what must be attributed and never claimed (the gate, the weight schedule, offset-free MPC in wind) |
| `literature_2026-09-16.md`, `litreview_schedule_paradigm_2026-09-01.md` | earlier scans | why the supervisor is shaped the way it is |
| `REPORT_2026-09-01.md` | the F-era consolidated findings F1-F7 | **history.** Its objective, baseline and aggregation are all superseded; read it for the mechanism chain, not for numbers |
| `roadmap_2026-08-30.md` | the F-era day-by-day log, §1-§23 | **history.** What was tried first, including what failed and why |
| `RESTORE.md` | how to rebuild the environment and data | after a VM loss or a full disk |
| `tables/` | every derived table as CSV | the numbers themselves |
| `figures/`, `manuscript/figures/` | R3 trajectories and the v5/v6 figure set | the qualitative comparison; `scripts/dev/figures_v5.py` regenerates |
| `logs/` | campaign logs, including failed attempts | provenance of specific runs |
| `plant/DISCON.IN` | the ROSCO configuration as actually run | which load-mitigation features were on or off (caveat C4) |

## Section map of the current roadmap

Numbers are sections of `roadmap_2026-09-11_metric-set-objective.md`.

- **Leg 1:** §51 the search was asking the wrong question of the wrong box - §52 the baseline re-derived as a
  real tuning (§52.2 the front, §52.3 one line of the shipped file) - §53 the 12-run campaign, 321 tunings
  (§53.2 the set-mean constraint is gamed on a classical controller too) - §55 the band verified off the design
  winds, the rule picks stage 1, the band is a duty ladder.
- **Leg 2:** §33 the wind-scheduled tower weight - §36 model error over the range - §38 the verified design
  loop finds the schedule by itself and does not beat the hand design - §57 the unscheduled reference on fresh
  seeds.
- **Leg 3:** §32 constrained objectives and two defects of the formulation - §34 the agent-written reward holds
  the loads, the fixed one does not - §35 where the layer does *not* pay - §37 n = 5 and two controls - §39-§40
  the layer on the MPC base - §41 the wind gate - §47 the gate-threshold sweep - **§49 the proposer question
  settled at n = 8** - §56 retrained on the baseline that is a tuning file - §58 the matched random-structure
  control for the gated layer, decision rule pre-registered before the result.
- **Discipline:** §22 actuator cost - §25-§26 where the actuation goes, the duty knee - §27 paired statistics -
  §28 the wind-range protocol - §42 site weighting - §54 the ladder in the four published reductions - §55.2
  the duty ladder.
- **Referee and audit passes:** §43-§46 novelty, protocol, the three blocked papers, a referee reading of v5,
  the pre-registration of three experiments - §48 the Wohler exponent as an inert control - §50 a WES-style
  referee reading of v6.
- **Earlier, now mostly superseded:** §1-§21 the objective switch, the critic/MPC-scaling findings and the
  first J campaigns; §29-§31 the first (hand-scaled) tuned baseline, which §52 replaced.

## A note on retracted claims

Statements that later evidence reversed are kept visible rather than edited away, because the reason each one
changed is itself a result. `MAIN_LINE.md` lists the live ones under *what cannot be claimed*; the F-era three
are marked where they appear: "the supervised variants are statistically tied" (08-30 §13, reversed in §21 once
both arms shared one wind bank), "schedule replay is the lowest-variance method" (§11, retracted in §14), and
"mono never satisfies the speed constraint" (restated in §22 as a paired ratio, because the tier count flips
between wind sets).
