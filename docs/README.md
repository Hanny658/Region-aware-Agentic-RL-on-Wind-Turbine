# Documentation index

Three groups, and every file is in exactly one of them:

- **`paper/`** - what the main text draws on. Start at [`paper/NARRATIVE.md`](paper/NARRATIVE.md) (the claim,
  the contributions, the section order, the figure and table plan) and [`paper/EVIDENCE.md`](paper/EVIDENCE.md)
  (every main-text number with its source). If a number is not in `EVIDENCE.md`, it is not a main-text number.
- **`appendix/`** - what is measured and real but not in the main narrative, including the nulls and the
  negative results. [`appendix/APPENDIX_PLAN.md`](appendix/APPENDIX_PLAN.md) lists them letter by letter.
- **`archive/`** - superseded or wrong. [`archive/ARCHIVE.md`](archive/ARCHIVE.md) says why each file is out
  and what replaced it. **No number from `archive/` may enter the manuscript.**

At the top level, three files that are none of the three: [`MAIN_LINE.md`](MAIN_LINE.md), the project's
complete internal record (every leg, every control, every null, and *what cannot be claimed*) - read it before
writing anything, because `paper/NARRATIVE.md` is deliberately narrower than it;
[`roadmap_2026-09-11_metric-set-objective.md`](roadmap_2026-09-11_metric-set-objective.md), the live
experiment log, §0-§60, 3400 lines, which is not meant to be read through (use the section map below); and
[`RESTORE.md`](RESTORE.md), how to rebuild the environment and the data.

**Rule for anything written from here on:** a new experiment document opens with the leg it serves and one
sentence on which number in `MAIN_LINE.md` it changes. If it changes none, it is a note, not a document, and
it goes in the roadmap only. When a roadmap section does change a main-line number, update `MAIN_LINE.md` in
the same commit, and if it changes a *paper* number, `paper/EVIDENCE.md` too.

## `paper/` - the main text

| file | what it is |
|---|---|
| `NARRATIVE.md` | **the paper's claim, contributions C1-C5 with their novelty status, section order, figures F1-F6, tables T1-T7, and the wording guardrails** |
| `EVIDENCE.md` | the number book: every main-text number, by section, with its table and roadmap section |
| `design_2026-10-11_rosco-native-search.md` | the leg-1 search: why the first parameter search was asking the wrong question, the native parameterisation, the box limits, the constrained selection |
| `RESULTS_2026-09-10_data-and-caveats.md` | data provenance, run inventory, caveats C1-C12 - **before quoting any number anywhere** |
| `literature_2026-09-23.md` | the novelty and prior-art audit: claim by claim, with a verification status per entry, the related-works entries grouped by theme, the field's reporting conventions, and verified BibTeX |
| `literature_2026-09-23_wind-rl-mpc.md`, `_residual-llm.md`, `_conventions.md` | the three delegated deep scans behind it |

## `appendix/` - measured, not in the main narrative

| file | what it is |
|---|---|
| `APPENDIX_PLAN.md` | appendices A-J, including **D1** the pre-registered proposer null and **D2** the delivery count under the shipping rule |
| `RESULTS_2026-09-05_ipc-negative.md` | the individual-pitch chapter, a documented negative: the boundary of the supervision claim |

## `archive/` - superseded or wrong

`ARCHIVE.md`, `roadmap_2026-08-30.md`, `REPORT_2026-09-01.md`, `plan_2026-10-12_manuscript-rewrite.md`,
`literature_2026-09-16.md`, `litreview_schedule_paradigm_2026-09-01.md`.

## Other directories

| directory | what it is |
|---|---|
| `tables/` | every derived table as CSV. `paper/NARRATIVE.md` §5 classifies them into main text, appendix and **superseded, cite nothing from them** |
| `manuscript/` | `main.tex` (**v7**, rewritten 2026-10-15 from `paper/NARRATIVE.md`; `latexmk -pdf main.tex` under MiKTeX), `references.bib`, `references.md` (verification status per entry), `arxiv.sty`, `figures/` (F2-F7 as `scripts/dev/figures_{leg1,overlay,weighting}.py` and `ladder_published.py --figdir` write them) |
| `figures/` | R3 trajectories and F-era figures; appendix candidates only, see `paper/NARRATIVE.md` §4 |
| `logs/` | campaign logs, including failed attempts - provenance of specific runs |
| `plant/DISCON.IN` | the ROSCO configuration as actually run (which load-mitigation features were on or off, caveat C4) |

## By leg

| leg | what it asks | where the evidence is |
|---|---|---|
| **1. The shipped GSPI, tuned by the agent** | what agentic search is worth on a classical pitch loop, in tuning files the controller can take | roadmap §51-§55, **§59**; `paper/design_2026-10-11_rosco-native-search.md`; `tables/rosco_{band_verified,tuned_baseline,perwind,proposer_test,search}.csv`, `tables/rosco_band_rows_*.csv` |
| **2. The untuned MPC, designed by the agent** | whether the agent can repair a model-based controller it did not design | roadmap §33, §36, §38, §57; `tables/design_search.csv`, `tables/mpc_leg_*.csv` |
| **3. The learned layer, and who writes its reward** | whether the level or the constraint is what the agent buys | roadmap §32, §34, §35, §37, §39-§41, §47, **§49**, §56, §58; `tables/{gated_native,gated_residual,range_residual,mpc_residual,gate_proposer_control}.csv` |
| **discipline** | why the three legs are believable | roadmap §22, §25-§28, §42, §52.3, §54, §55; `paper/RESULTS_2026-09-10_data-and-caveats.md`; `tables/ladder_published_*.csv`, `tables/{lifetime_del,wohler,verification_probe}.csv` |

## Section map of the current roadmap

Numbers are sections of `roadmap_2026-09-11_metric-set-objective.md`.

- **Leg 1:** §51 the search was asking the wrong question of the wrong box - §52 the baseline re-derived as a
  real tuning (§52.2 the front, §52.3 one line of the shipped file) - §53 the 12-run campaign, 321 tunings
  (§53.2 the set-mean constraint is gamed on a classical controller too) - §55 the band verified off the design
  winds, the rule picks stage 1, the band is a duty ladder - **§59 the proposer question at n = 6: the primary
  endpoint is null and the delivery count is reversed** (§59.1 the pre-registration, §59.2 the restatement
  under the rule that ships, §59.4 the null primary and the 0-of-144 divergence count, §59.6 the closing
  count) - **§60 the documents regrouped and the paper's narrative fixed**.
- **Leg 2:** §33 the wind-scheduled tower weight - §36 model error over the range - §38 the verified design
  loop finds the schedule by itself and does not beat the hand design - §57 the unscheduled reference on fresh
  seeds.
- **Leg 3:** §32 constrained objectives and two defects of the formulation - §34 the agent-written reward holds
  the loads, the fixed one does not - §35 where the layer does *not* pay - §37 n = 5 and two controls - §39-§40
  the layer on the MPC base - §41 the wind gate - §47 the gate-threshold sweep - **§49 the proposer question
  settled at n = 8** - §56 retrained on the baseline that is a tuning file - §58 the matched control: under the
  gate the proposer is not distinguishable, and the gate is why.
- **Discipline:** §22 actuator cost - §25-§26 where the actuation goes, the duty knee - §27 paired statistics -
  §28 the wind-range protocol - §42 site weighting - §54 the ladder in the four published reductions - §55.2
  the duty ladder.
- **Referee and audit passes:** §43-§46 novelty, protocol, the three blocked papers, a referee reading of v5,
  the pre-registration of three experiments - §48 the Wohler exponent as an inert control - §50 a WES-style
  referee reading of v6.
- **Earlier, now mostly superseded:** §1-§21 the objective switch, the critic/MPC-scaling findings and the
  first J campaigns; §29-§31 the first (hand-scaled) tuned baseline, which §52 replaced.
