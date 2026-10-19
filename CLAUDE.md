# Region-aware Agentic RL on Wind Turbine

Residual-RL pitch control for the NREL 5 MW turbine: ROSCO GSPI baseline + operating-region-specialised
PPO residual agents (R2/R3 split by an oracle rule), with an LLM supervisor tuning six reward/action
knobs at a slow timescale. Baseline paper: Wang/Dong/Zhao, IEEE TSTE 2026 (`RelatedWorks/`, not in git).

## Read these first
- `docs/MAIN_LINE.md` — the claim, the three legs of evidence, what cannot be claimed (**start here**, one page).
- `docs/paper/NARRATIVE.md` — the PAPER's narrative, which is NARROWER than MAIN_LINE by the user's decision of
  2026-10-15: the main text carries the positive results, the nulls and negative results are appendix
  material. It holds the claim, contributions C1–C5 with their novelty status, the section order, figures
  F1–F6, tables T1–T7 and the wording guardrails. `docs/paper/EVIDENCE.md` is the number book behind it
  (every main-text number with its source); `docs/appendix/APPENDIX_PLAN.md` holds what is measured but not
  in the main narrative, D1 and D2 being the pre-registered proposer null and the delivery count.
- `docs/README.md` — the three groups (`paper/`, `appendix/`, `archive/`), which document is evidence for which
  leg, and a section map of the 3400-line roadmap.
- `docs/roadmap_2026-09-11_metric-set-objective.md` — the current experiment log, §0–§62. Use the section map;
  it is not meant to be read through.
- `docs/paper/RESULTS_2026-09-10_data-and-caveats.md` — provenance and caveats C1–C12, before quoting any number.
- `docs/archive/` — superseded or wrong; `ARCHIVE.md` says why each file is out and what replaced it. The F-era
  records (`REPORT_2026-09-01.md`, `roadmap_2026-08-30.md`) are history: read them for mechanism, never for
  numbers, and no number from that directory may enter the manuscript.
- Design decisions history: the user's explicit answers are recorded in the report; do not re-ask them.

## Scope
This repository covers region-aware residual RL for **collective pitch**, agentic (LLM)
supervision, the GSPI / LPV-MPC / ROSCO-tower-damper baselines (roadmap §1–23), and the
**individual-pitch negative result** (`docs/appendix/RESULTS_2026-09-05_ipc-negative.md`), which is kept
because it bounds the supervision claim. **Do not describe research directions beyond that scope**
in this repo's docs, commits or manuscripts. The IPC/torque code paths stay in the tree (inert at
`--ipc_max 0` / `--dtau_max 0`) because run configs reference them.

## The main line (2026-10-13) - read `docs/MAIN_LINE.md` first, it is one page

**Agentic search buys feasibility and sample efficiency on real controllers; it does not buy peak level.** The
same agentic loop is put on three objects - the shipped ROSCO pitch loop (leg 1), an untuned LPV-MPC (leg 2),
and a residual layer's reward structure (leg 3) - and all three come out that shape: it turns a design that is
not deliverable into one that is, in few simulations, where random search finds nothing, and in none of the
three does it beat a careful human or a fixed classical recipe on the level. `docs/MAIN_LINE.md` holds the
claim, the three legs with their current numbers, **what cannot be claimed**, and the open items;
`docs/README.md` says which file is evidence for which leg. Keep `MAIN_LINE.md` current in the same commit as
any result that changes one of its numbers - the bullets below are the detail, not the claim.

The composite the campaigns selected and trained on (it is method, never a reported result):

Objective: **J** = mean of the baseline paper's four % reductions vs paired GSPI (power MSE,
gen-speed MSE, tower DEL, blade DEL; per-episode clipped ±100), −20 per % energy loss over 1 %, no
tiers, wind-labelled R3 subsets (`docs/roadmap_2026-09-11_metric-set-objective.md`, decisions D1–D6).
F (tower-DEL priority with constraint tiers) stays computable for the historical tables.

### Leg 1 - the shipped GSPI, tuned by the agent (`docs/MAIN_LINE.md` leg 1)

- **REVIEW ROUND 1 APPLIED (2026-10-19, roadmap s63; `docs/manuscript/Review.md` is the user's reviewer notes).**
  Two counts in the manuscript were wrong and are corrected everywhere: the 21 runs made **504 candidate
  evaluations on 485 distinct tunings** (the "561" of s59.6 is withdrawn), and the plant could not run **38**
  of them while **one evaluation did not complete for a non-plant reason** (bayes r4 #20, `failure_kind`
  infrastructure, counted as a failed candidate; App. C no longer says "zero infrastructure failures"). Two
  additions, pre-registered and done: **the one-knob sweep** (`rosco_search.py --proposer sweep`, grid 2.04 to
  11 rad/s, `docs/tables/rosco_sweep.csv`) finds the one-line row at its third point and the agent's 6.5 row at
  its sixth, selects 6.5 under every reading and that row ships; above 6.5 the ladder turns down (+11.1 at 8.5
  for 3.7x travel, -15.5 at 11). It is a control row of Table 3, not a search; no proposer claim. **The spectra**
  (`scripts/dev/rosco_spectra.py`, `f5_spectra.png`, `rosco_spectra.csv`): the shipped corner's 21 deg of lag at
  omega_pc lets the loop ring just below 1 rad/s in speed, tower moment and pitch rate; corners 3.0 and 6.5
  remove it (power below 1.57 rad/s: speed -17/-23 %, tower moment -34/-53 %, pitch rate -33/-52 %) and pass 3P
  content into the pitch (3P-band pitch-rate power x2.4/x5.6); at 3.0 the two cancel (total pitch-rate power
  0.470 vs 0.485 = the 1.000x travel), at 6.5 the total doubles (the 1.65x). Manuscript: new title (three
  objects), abstract with the pre-registered null, three contributions, 4.1 states how Table 2 was fixed and
  prints the one-line row's 0.02 points of room (per-wind interval table in App. E, 83 % of resamples inside
  the tolerance), 4.2 the spectra, 4.4 candidates against delivery (Table 4), 6 the gated null in the text, App.
  E the band of ten. `band_perwind_intervals.py`, `rosco_sweep_report.py` are the new tables' scripts. Nothing
  of s63 is pending.
- **FINISHED 2026-10-19 03:15 (roadmap s62.4, pre-registered; the user's instruction: remove the prompt leak, then
  test the set-mean margin): THE RULE COMPLETED, AND THE AGENT TOLD ONLY THE RULE FINDS NOTHING.** The leak is out of `SHIP_TEXT` (no finding of an
  earlier campaign is told any more, only the rule and the design set's resolution). The census of the 36
  tunings verified on both seed sets: at a design-set set-mean tower-base reduction >= 6.0 points **12 of 12
  ship**, below it 2 of 24, so the rule is now feasible AND worst single wind >= 0 AND set-mean tower >= 6.0 on
  the design set (`per_wind_rank_key(rec, tol, tower_margin)`, `rosco_search.py --screen_tol 0.0
  --tower_margin 6.0`; the shipped tuning stays admissible so a holding row with negative regulation never
  wins). 62.4a (`campaign_rule6.sh`, about 2 h) re-reads the 39 existing runs under it: on the s59 runs it
  selects seven rows that all already ship (llm 2, local 4, bayes 1 of 6), on the s62.3 runs it selects six
  unmeasured rows (five of them the agent's, walked past by the per-wind screen) - endpoint: how many of the
  six ship. 62.4b (`campaign_rosco_rule6.sh`, waits for 62.4a, about 32 h, repeats 6-11 so nothing replays,
  `~/wtrl/exp/rosco_rule6.log`, `campaign_ctl.sh ... campaign_rosco_rule6`): every arm selects under the
  completed rule, the agent is told it. Endpoints fixed in s62.4: deliveries per arm, the PRECISION of the
  rule (ships / selections), and what each outcome means; no superiority claim unless 6 of 6 against at most
  1 of 6. `rosco_perwind_select.py --roots A B` now prefixes run names with the root (two campaigns share
  `llm_r0`). **62.4a RESULT (2026-10-17 14:44; `docs/tables/rosco_proposer_test_rule6_{s59,s62}.csv`): 3 of the
  six ship** (bayes r1 #15 +21.22 fresh, llm r4 #4 +27.89 with every load cell positive on both seed sets, llm r5
  #1 +24.92 = the one-knob corner 3.4); the three failures (llm r1 #7, r2 #15, r3 #11 at margins 7.28 / 6.13 /
  6.09) all lose the tower-base load at 24 m/s on fresh seeds (-1.34 / -1.95 / -1.83) - the pre-registered
  branch 'the margin is not a sufficient proxy at its boundary': the margin sits at 12-16 m/s and none of it at
  24, the s32 aggregation defect one level up. Re-read on 42 tunings: 7.5 and above 14 of 14 ship, 6.0-7.5 1 of
  4, below 6.0 2 of 24. Counts under the completed rule: s59 runs llm 2 / local 4 / bayes 1 of 6 (precision 7 of
  7), s62.3 runs llm 3 / local 3 / bayes 2 (8 of 11), pooled 15 of 18; no Fisher below 0.57. **62.4b INTERIM
  (2026-10-18 09:20): THE AGENT ARM IS FINISHED AT 0 OF 6** - told the complete rule and nothing else, 1 of its
  144 candidates improves regulation at all (infeasible), 80 of 144 hold the rule with regulation -2.7 to -463,
  every selection is the shipped tuning; it walks omega down to 0.25-0.49 and never puts the corner above 2.4,
  its `why` claims a regulation improvement 62 times while the term falls, 0 malformed replies, 0 divergences
  of its own (7 among 15 random fills after repeated proposals). DISCLOSE: the task text lists holding rows at
  -113 as 'best first' without the shipped row (my defect; not relaunched unprompted). Pre-registered branch:
  nothing into the main text, D3 in full, s59.6 wording stands. **62.4b RESULT (`docs/tables/
  rosco_proposer_test_rule6_campaign.csv`, 18 runs, 432 tunings): llm 0 of 6, local 2 of 6, bayes 4 of 6, and
  EVERY ONE OF THE SIX SELECTIONS SHIPS** (precision 6 of 6; 21 of 24 over everything measured under the completed
  rule, the three failures all the agent's 62.4a rows at margins 6.1-7.3 on the 24 m/s tower cell). Fisher llm
  against bayes p = 0.061, against local 0.45; on the search's own set-mean rule the agent found a gain in 0 of 6
  against 5 of 6 and 5 of 6 (p = 0.015, not the pre-registered primary). Delivered rows: bayes r6 #17 +32.13 fresh
  at 1.259x (damper off), local r11 #21 +27.85 at 1.276x, bayes r8 #21 +21.29 at 1.086x, **local r8 #12 +12.80 at
  0.997x - the first search delivery on LESS pitch travel than shipped**, two heavy-damper bayes rows +10.3 / +7.2
  at 3.3x / 3.5x. Bayes with the margin in its feasibility model has its highest count of any campaign. Margin
  re-read on 48 tunings: 7.5 and above 18 of 18 ship, 6.0-7.5 3 of 6, below 6.0 2 of 24; the binding cell in 7 of
  the 8 boundary rows is the tower-base load at 24 m/s on fresh seeds - the remaining lever is design episodes at
  24 m/s, not a larger margin and not the prompt. The precision condition for a §8 statement (0.9 with at least
  ten selections) is not met on count (six), so the completed rule is D3 and one sentence of §9 only. Across
  leg 1 the agent arm's regulation came from the race-framed exploration of s59 or the leak of s62.3, never from
  being told the rule. Nothing of s62.4 is pending; do not launch any of its scripts again.
- **THE SCREEN WAS THE FAILURE, MEASURED (roadmap s62.1, 2026-10-16, pre-registered; `campaign_rule_margin.sh`,
  `docs/tables/rosco_proposer_test_margin.csv`): with one point of margin in the design-set screen (worst
  single-wind load term >= 0; shipping rule unchanged) the SAME 21 runs deliver local 4 of 6, bayes 2 of 6,
  llm 2 of 6 (was 0 of 6), random 0 of 3.** llm r1 #11 (+28.35 at 1.57x) and r5 #14 (+35.71 at 2.18x, the
  most-regulating delivered row of the campaign) ship, llm r2 #12 misses by 0.50 at 24 m/s on fresh seeds;
  Fisher p = 0.57 against local - the classical arms still deliver at least as often, so no proposer claim.
  The two remaining failures have design margins inside the range of the eight that ship: the 14-episode
  screen's resolution is now the limit, not the margin. **THE ARM THAT IS TOLD THE RULE IS FINISHED (s62.2 pre-registered, s62.3 result,
  2026-10-17; `campaign_rosco_margin.sh`, `rosco_search.py --screen_tol 0.0`,
  `docs/tables/rosco_proposer_test_margin_campaign.csv`; 18 runs, 416 tunings, zero failures): told the screen
  and shown the per-wind terms, the agent drives to the screen's edge in all six runs (selections within omega
  0.60-0.64, zeta 0.70-0.81, corner 3.4-3.75, 1.12-1.20x travel, design worst -0.0 to +0.4, 19-21 of 24
  candidates holding the screen, 0 of 144 destabilising again) and SHIPS IN 1 OF 6 - five of six lose 1.6-2.2
  points of tower DEL at 24 m/s on fresh seeds while holding held out; local 3 of 6 (four runs are exact
  replays of s59, the screen never changed their incumbent), bayes 2 of 6. The pre-registered branch: nothing
  moves into the main text, D3 reports it in full. A per-wind screen at two episodes per wind is gamed by an
  optimiser that is told it (fifth screen-gaming instance), and the set-mean tower margin predicts holding for
  the sixth time (failing +3.5 to +5.0, shipping +7.4 and above, one exception at +3.9): THAT margin, as a
  constraint, or episodes at the binding winds, is the next lever - not the prompt. DISCLOSE WHEREVER THIS
  CAMPAIGN IS CITED: `SHIP_TEXT` told the agent that the filter corner is nearly free to 3.4 rad/s and that
  the damper off loses the tower load at high wind (the s52.3/s55 findings in words), so the agent's
  one-simulation time to the first screen-holding candidate (p = 0.0011) measures the prompt, not the search,
  and no between-arm reading of this campaign in the agent's favour exists. Do not launch it again.** `rosco_proposer_test.py` no longer writes
  `rosco_proposer_test.csv` unless `--csv` names it (a reading without the flag overwrote the committed
  table once; restored from git).
- **THE PROPOSER CLAIM IS GONE FROM LEG 1 TOO: at n = 6 the model is not distinguishable from classical
  search (roadmap s59.4, 2026-10-14; 21 runs, 504 candidate evaluations, `scripts/dev/rosco_proposer_test.py`,
  `docs/tables/rosco_proposer_test.csv`).** The primary endpoint of s59.1, fixed before the runs existed:
  simulations to the first feasible gain, one-sided exact permutation - llm against bayes **p = 0.31**,
  against local p = 0.19, against random p = 0.048; runs with a gain llm 5/6, bayes 5/6, local 4/6,
  random 0/3; medians 10 / 18 / 14. The s53 reading (3/3, median 8 against 14 and 18) did not survive
  doubling: llm r3 found nothing at all, bayes r4 found its gain in 5 simulations. The model still walks
  furthest up the front (+38.40 (8.47) against local +29.23, bayes +23.79) and that is exactly the part
  of the front where s55 found everything fails per wind: measured off the design winds, all 11 per-wind
  selections verified, **the model arm delivers 0 of 6 against 4 of 6 for the local search and 2 of 6 for
  bayes** (s59.6, final; Fisher two-sided p = 0.061 against local, 0.030 one-sided in the direction the
  data went, which is the opposite of the pre-registered one). Everything that ships sits between +11.6
  and +28.7 points of regulation and everything that fails between +26.7 and +37.7, the model's three
  selections being the highest rows of all - position on the front, not the proposer. The one row a user
  would take out of 21 runs is local r3 #8, one filter corner and a small damper: +28.66 of regulation at
  1.05x shipped pitch travel. **Post hoc, the one large difference in the model's favour: 0 of 144 of its candidates
  destabilised the plant**, against 13/144 local (p = 0.00018), 9/144 bayes (p = 0.0034) and 16/72
  random (p < 1e-5), with the malformed-reply audit at 0 in all six runs; and its runs repeat each other
  (2 distinct per-wind selections of 3, against 4 of 4 in both classical arms). **No leg now has a
  proposer claim against a dedicated search** - leg 3 lost it in s58, leg 2 never had it (llm = es) -
  so what survives is the search SPACE, the measurement discipline, and one loop that runs in three
  spaces at parity. s59.1 fixed this reading in advance, including that the title loses the proposer.
- **FINISHED 2026-10-15 11:18, nothing of s59 is pending** (stage 3 at 10:23 after the overnight WSL
  restart took the pause down with it - relaunching `campaign_rosco_native_n6.sh` skipped all nine runs on
  their best.json and `integrity_check.py` was clean; then `scripts/wsl/campaign_perwind_verify.sh`, which
  runs the new `rosco_perwind_select.py --verify` pass for the per-wind selections stage 3 does not cover,
  because stage 3 only re-scores each run's SET-MEAN selection). Stage 3 by-products: 21 of 21 selections
  feasible held out and fresh, design against held-out correlation +1.00 (rank +1.00), optimism -0.43 mean
  and +0.88 worst. One infrastructure failure, reported and repaired: the new script did not export
  `WTRL_WIND` / `WTRL_HOME`, so four evaluations died instantly on the 150 s wind bank; those cache entries
  were deleted and the four re-run. **Any later verification pass must export the 600 s bank the way
  `campaign_rosco_native.sh` does.**
  s58 took the proposer claim off leg 3, so leg 1 carries it alone, and leg 1 was n = 3 per arm where 3 of 3
  against 2 of 3 is Fisher p = 1.0. Endpoints and power pre-registered in s59.1 BEFORE the runs exist:
  **primary** the simulations to the first feasible gain (Mann-Whitney one-sided llm against bayes, about
  p = 0.03 if s53 replicates, 0.0011 at best; a run that never finds one is censored and enters at 25);
  **secondary** the count of runs delivering a shippable gain (underpowered and said so in advance: 6 of 6
  against 4 of 6 is p = 0.45) and the per-wind restatement of both. s59.1 fixes what each outcome means,
  including that a null primary leaves **no proposer claim in any leg**. All of it is measured; do not launch
  the campaign again.
- **THE RESTATEMENT IS ALREADY IN, AND IT IS NOT FLATTERING (roadmap s59.2, no simulations;
  `scripts/dev/rosco_proposer_test.py`, `docs/tables/rosco_proposer_test.csv`).** The twelve runs of s53
  read three ways. Under the rule the search used: llm 3/3 runs with a shippable gain, median 8 sims,
  +42.18 regulation; bayes 2/3, 21; local 1/3; random 0/3 - but against bayes that is one-sided exact
  p = 0.30 on the simulations and Fisher p = 1.0 on the counts, so **at n = 3 nothing separates the model
  from Bayesian optimisation**. Under the same rule with the loads required at every wind speed and
  MEASURED off the design winds - what a user gets - **llm 0/3, bayes 1/3 (+16.57), local 1/3 (+11.62),
  random 0/3**: the llm selections miss the tolerance by 0.17 points (llm r1 #12: design -0.60, held out
  +1.06, fresh -1.17) and by 2.40 (llm r2 #24), on the tower load, worse fresh than held out every time.
  **The mechanism is the screen, not the search:** llm r1 visited two tunings that DO hold off the design
  winds (#9 and #11, the corner ladder of s52.3, one of them the +22.6-at-1.004x row in the verified band)
  and its own rule took #12 instead, because on 14 design episodes #12 reads +36.96 of regulation with a
  worst single wind of -0.60 and two episodes per wind cannot tell -0.60 from -1.20. Separate cheap
  question, not this campaign: screen with a margin (s55: the set-mean margin predicts holding) or spend
  episodes at the winds where the constraint binds.
- **Parameter-search audit and redesign (2026-10-11, `docs/paper/design_2026-10-11_rosco-native-search.md`,
  `scripts/rosco_native.py`, `eval/selection.py`, `campaign_rosco_native.sh`):** the s29/s31 tuned baseline was
  **not a ROSCO tuning**. It scaled `PC_GS_KP` / `PC_GS_KI` by two scalars, and because ROSCO builds the schedule as
  `Kp = (2 zeta omega + A(beta))/B(beta)`, `Ki = omega^2/B(beta)`, a scalar on Kp implies a damping ratio that varies
  6.8-12.8 % along the schedule: those gain tables are unreachable by any tuning file. A, B are recovered exactly from
  the shipped tables (round trip 8.7e-19, shipped tuning zeta 0.7 / omega 0.6). The box limits were ours and two of
  them produced results: the filter corner was capped at 4.0 rad/s and both llm winners sat on that cap (the measured
  ceiling is 19.2 rad/s, where the filter passes 99.99 % of the generator-speed power), and `FA_KI` was capped at
  0.004, worth **0.25 deg** of damper pitch authority, so **the s23 claim that ROSCO's tower damper is worth ~0 is
  withdrawn**. The composite score is gone from selection and reporting: the search now maximises the generator-speed
  regulation reduction subject to tower DEL, blade DEL and energy being no worse than shipped, a constraint counting
  as violated only where the loss exceeds its own 95 % paired-bootstrap interval (measured median half-widths 1.30 /
  1.00 / 0.031 points), ranked by constraint domination. Searched quantities: omega_pc, zeta_pc, F_LPFCornerFreq,
  FA_KI, FA_HPFCornerFreq, every limit read off the plant (`scripts/dev/search_box_bounds.py`).
- **The tuned baseline, re-derived as a real ROSCO tuning (roadmap s52, 2026-10-11; template
  `~/wtrl/runs/template_5mw_rosco_tuned_native`, `docs/tables/rosco_tuned_baseline.csv`). This is the baseline row
  from here on.** omega_pc 0.6048, zeta_pc 0.7378, F_LPFCornerFreq 6.529 rad/s, FA_KI 0.0331, FA_HPFCornerFreq 0.692,
  TD_Mode 1. Against the shipped controller on the same wind fields: gen-speed MSE **+28.30** [25.35, 31.23], power MSE
  +24.50, tower DEL **+22.77** [22.08, 23.45], blade DEL +3.46, energy +0.145 on the 14 design episodes; +28.32 /
  +23.93 held out (seeds 3-6) and +29.75 / +25.32 on fresh seeds 7-10, feasible on both, worst single-wind load term
  -0.24 (blade, 16 m/s), design optimism -0.02. Pitch travel **2.01x** shipped, ADC 0.0948 vs 0.0471 - the scheduled
  MPC spends 2.14x, so duty no longer separates them. The damper asks ~2.0 deg of pitch (limiter 10, not reached):
  **the s23 "tower damper worth ~0" claim is replaced by its opposite.** A search that finds nothing selects the
  shipped tuning itself and reports 0.00 (s52.1); the first random run did exactly that in 24 candidates.
- **CAMPAIGN FINISHED (roadmap s53, 2026-10-12 16:33): 12 runs, 321 tunings, all 12 identity checks exactly
  0.000, 37 divergences, ZERO infrastructure failures.** Proposers (runs that bought a shippable gain; regulation
  of the selection, mean (sd); median sims to the first gain): random **0/3**; local 1/3, +14.30, 18; bayes 2/3,
  +31.06 (22.20), 14; **llm 3/3, +42.18 (5.22), 8** - the model arm sweeps one quantity at a time with the plant
  frequencies as landmarks (its transcript says so), which is what found the one-knob ladder. **Do not write that
  it designs a better controller:** s53.2 shows the objective as stated was insufficient. Design vs held-out
  correlation **+1.00**, rank +1.00, optimism -0.13, 12/12 feasible held out and fresh - the s38 "a small design
  set cannot rank designs" does NOT transfer to this space. **DECISIVE (s53.2): the set-mean load constraint is
  gamed on a classical controller too** - off the design winds the three most-regulating winners lose 2.7 to 7.6
  points of tower DEL at a single wind speed (llm r0 -4.64 at 12 m/s, llm r1 -7.61 at 12, llm r2 -3.73 at 24)
  while holding the set mean; the size of the set-mean margin predicts whether a tuning holds per wind. s32 now
  has a second instance with no learning involved. **The duty argument against the MPC is gone: a PI tuning
  reaches +33 regulation at 1.12x travel, the scheduled MPC sits at 2.14x.** Open, ~1 h: the per-wind measurement
  of the three unverified band members; stage 1 holds the stated rule and stands until something beats it.
- **The legitimate tuning space is a FRONT, not a point (roadmap s52.2, s53.3; band: +22.6 to +37.0 regulation,
  +2.5 to +22.8 tower DEL, 1.00x to 2.72x travel).** Same box, same protocol, every row a
  tuning file the controller can take: llm r0 regulation **+47.79** with tower DEL -0.61 (set mean inside the
  measurement resolution, but **-5.9 at 18 m/s**) at 1.76x travel; bayes r1 +46.76 / +4.24 at 1.53x; stage 1
  +28.30 / +22.77 at 2.01x; local r0 +14.30 / +17.78 at 2.72x. The regulation objective walks up the front and
  switches the damper off. Rule recommended BEFORE stage 3 is read, following s32 (set-mean load constraints are
  gamed across wind speeds): **the tuned baseline is the largest regulation reduction whose PER-WIND load terms
  hold off the design winds**. Proposers so far: a shippable gain in 0/3 random, 1/2 local, 1/2 bayes, 2/2 llm
  runs, median 8 simulations.
- **ONE LINE of the shipped tuning file is worth a fifth of its regulation error at the SAME pitch travel
  (roadmap s52.3; `scripts/dev/rosco_perwind_select.py`, `docs/tables/rosco_perwind.csv`).** Only
  `F_LPFCornerFreq`, shipped gains untouched, every load term positive at every wind speed: 3.0 rad/s gives
  gen-speed MSE **+22.62**, power MSE +16.90, tower DEL +7.83, blade DEL +4.26, energy +0.084 at travel
  **1.004x** (ADC 0.0473 vs 0.0471); then 3.8 (3P) +25.91, 5.0 +27.39, 6.5 +28.60 at 1.06 / 1.18 / 1.57x. Nothing
  is traded - it is a default left unexamined, and every comparison that uses the shipped ROSCO as its baseline
  starts this far below the same controller with one number changed.
- **The verified band is a DUTY LADDER of the same controller (s55.2-55.3), four rows, every one a tuning file,
  each holding both loads at every wind speed on 28 held-out AND 28 fresh episodes** (fresh seeds; power / speed
  / tower / blade reduction vs shipped, travel, ADC): stage 1 (corner 6.53 + damper 0.0331)
  +25.3 / +29.7 / +25.3 / +1.7 at 2.016x, ADC 0.0968; corner 6.500 damper off +24.2 / +28.3 / +16.3 / +2.5 at
  1.647x; **one knob (corner 3.000) +17.7 / +23.0 / +8.7 / +2.0 at 1.000x travel, ADC 0.0480 vs 0.0476 shipped**;
  a slower loop with the damper on (omega 0.562 zeta 0.663 corner 4.146 damper 0.0018) +8.0 / +11.4 / +17.0 /
  +2.7 at **0.974x - all four terms positive on LESS pitch travel than the shipped controller**. The damper buys
  the last 9 points of tower fatigue for 0.37x of travel. A fifth row is excluded and is a warning: omega 0.582
  zeta 0.590 corner 2.595 reads **power MSE -1.9 [-2.6, -1.2] with speed MSE +0.7 [-0.4, +1.9]** - the
  power/speed redundancy argument behind using speed alone as the objective holds on the model-based rows
  (0.94-0.95) and is weak on PI rows (0.19-0.83), and this is the weak case; say so where that convention is
  stated.
- **The baseline is a BAND, not a row (s52.3):** 245 distinct tunings evaluated, 52 hold both loads at every wind
  speed; the band runs +22.6 to +36.9 of regulation and +4.4 to +22.8 of tower DEL at 1.00x to 2.72x travel. The
  damper buys tower fatigue and costs regulation and duty; the filter corner buys regulation nearly free to about
  3.4 rad/s. **The per-wind screen is a SCREEN on the design set, not the decision:** with two episodes per wind
  it rejects the stage-1 tuning at -1.1 (blade, 22 m/s) although the same tuning reads -0.24 / -0.34 off the
  design winds where there are four. Screen on design, measure per wind off the design winds (stage 3 does this).
  **BAND VERIFICATION FINISHED (s55, 2026-10-12 22:10; `campaign_verify_band.sh`, 10 members, 560 episodes,
  `docs/tables/rosco_band_verified.csv`, `rosco_band_rows_{fresh,heldout}.csv`): 6 of 10 fail the per-wind rule,
  and EVERY member that buys more regulation than the standing baseline is one of the six.** All six fail on the
  tower-base load, five of six at 24 m/s, each worse on fresh seeds than held out while its regulation
  replicates (+36.96 design -> +36.98 held out -> +37.74 fresh, worst per wind -1.17; +34.59 -> -2.04;
  +33.60 -> -3.40; +33.34 -> -2.98; +32.14 -> -3.04; +30.69 -> -3.00). Set-mean tower margin of the six: +1.0 to
  +4.4; of the four that hold: +7.8 to +15.9 - the margin is the predictor, third instance after s32 and s53.2.
  **THE RULE SELECTS STAGE 1 AND IT IS NOT CLOSE:** paired on the same episodes, the closest verified member
  (corner 6.500, damper off, 1.647x travel) is -1.40 [-1.74, -1.08] in speed MSE and -9.07 [-9.55, -8.61] in
  tower DEL on fresh seeds. The baseline row is verified three ways (design, 28 held out, 28 fresh).
- **Tuned ROSCO baseline (roadmap s29, s31; `scripts/rosco_tune.py`, template `~/wtrl/runs/template_5mw_rosco_tuned`):**
  the shipped speed-filter corner (1.57 rad/s) is the untuned part of the GSPI; 3.42 rad/s with kp x1.13 scores
  16.2 / 17.1 on the low-turbulence held-out sets and 15.4 on the range set at 1.1x pitch travel. Against it the
  MPC rows keep 4.5-5.4 J at low turbulence (tower fatigue near rated only) and ~14 J at class B over the range
  (23-24 points of regulation plus 9 of tower). **SUPERSEDED on 2026-10-11** by the re-derived baseline below: this
  row is a hand-scaled gain table no tuning file can produce, selected and reported under the composite that is now
  gone, so its number is not comparable and every table carrying it must be re-measured.

### Leg 2 - the untuned MPC, designed by the agent (leg 2)

- **WHAT THE AGENTIC DESIGN SEARCH IS WORTH ON AN UNTUNED MPC, in the four published reductions (roadmap s57.2,
  2026-10-13; `scripts/dev/mpc_leg_table.py`, `docs/tables/mpc_leg_{fresh,heldout}.csv`,
  `campaign_mpc_unsched_fresh.sh`).** s38 reported the discovery of the wind schedule in composite points; this
  is the same thing in published terms. Fresh seeds 7-10, 28 episodes, vs the paired shipped controller:
  unscheduled offset-free LPV-MPC +50.7 / +57.1 / **+15.6** / +0.5 at 1.912x travel; agent-found schedule
  +49.1 / +55.4 / **+17.4** / -0.0 at 2.076x. Paired: -1.65 power, -1.74 speed, **+1.83 tower** [+1.29, +2.72],
  -0.53 blade. **The set mean hides the whole repair:** tower DEL at 24 m/s goes **-8.76 -> -0.33** (held out
  -6.27 -> +0.08), so against the s28 rule (no term below -1 % at any wind speed) **the unscheduled controller
  fails by 8.8 points on winds nothing has seen and the schedule brings it inside the tolerance for 1.7 points
  of regulation** - a feasibility repair, which is leg 2 of `docs/MAIN_LINE.md`. Two qualifications: at -0.33
  it is inside the tolerance, not positive; and the tuned PI loop beats both MPC rows on tower fatigue at every
  wind speed from 16 m/s up (+26.9 / +20.5 / +18.7 / +16.9 / +15.3 against +17.7 / +6.5 / +5.4 / +3.4 / -0.3)
  at comparable duty, the per-wind form of s54 and s56.
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
- **Main-claim controller (roadmap s33): offset-free LPV-MPC with a wind-scheduled tower weight** (`qt_sched=[6, 18, 22]`
  via `--base_mpc_json`, qt 3 below 18 m/s): selected on range seeds 1-2, on held-out seeds 3-6 (28 episodes) J 28.95 at
  2.14x travel with NO term below -1 % at any wind speed (tower +1.6 / +0.2 / +0.1 % at 20 / 22 / 24 m/s vs +0.4 / -2.5 /
  -6.3 % unscheduled), -0.56 J for the schedule, +13.7 J [13.0, 14.4] over the tuned ROSCO. Model error on the same
  episodes (s36): scheduled 28.81 / 29.00 / 28.95 / 28.44 at Cp/Ct x0.85 / 0.95 / 1 / 1.15 (worst per-wind term -2.1 % at
  x0.85), nominal MPC 21.38 / 28.17 / 28.49 / 23.07 (-7.1 and -5.4 J at +-15 %).
- **Verified design loop for the MPC (roadmap s38; `scripts/mpc_design_search.py`, `docs/tables/design_search.csv`):** from the
  hand-set UNSCHEDULED reference, llm and es proposers both discover the wind schedule in 24 simulations and repair the
  per-wind violation on held-out winds (S = J - 4 V: 25.66 -> 26.8-28.2); random search 0/48. They do NOT beat the hand design
  of s33 (28.95) and llm = es in the result; llm proposals beat the reference more often (25/48 vs 15/48, p = 0.03). Selection on
  8 episodes cannot rank schedules (rank correlation selection vs held-out -0.9 over the scheduled designs). Do not claim
  "the agent designs a better controller" or "the LLM beats other search".
- **Residual line closed from the learning side (s30):** the residual trained over Cp/Ct in [0.85, 1.15] repairs
  only the underestimate side, nothing at +15 %, and costs 3-6 J at the exact model; the estimator is 10-40 J above
  every residual at every error. The tower weight is not the lever for the compensated MPC's negative high-wind
  tower term (s31): qt 6 makes it non-negative at -2.2 J and +20 % travel, larger weights destabilise 12 m/s;
  under a per-wind load constraint the nominal MPC (28.5) beats the constrained compensated MPC (27.3).

### Leg 3 - the learned layer, and who writes its reward (leg 3)

- **WHO WRITES THE REWARD DOES NOT MATTER UNDER THE GATE (roadmap s58, 2026-10-13;
  `campaign_gate_native_rand.sh` arm `tnrrCwG3t`, `scripts/dev/gate_proposer_control.py`,
  `docs/tables/gate_proposer_control.csv`; three control seeds, 4 h 10 min, decision rule and its power
  pre-registered in s58.1 before any result).** The matched random-structure control for the gated layer on
  the native tuned baseline, three seeds paired to the agent runs seed by seed. Under the rule that DECIDES
  this question - the worst single-wind load term against its own base, tolerance -1 point, as in s49, and
  NOT the per-wind Cw against the shipped controller that s56 quoted - **agent 3/3, random reward structure
  3/3 on both seed sets, Fisher two-sided p = 1.0**, and the control holds with MORE margin: no control seed
  is negative at any wind speed held out, against the agent arm worst cell of -0.98 (tower, 18 m/s, fresh).
  It is not behind in level either (power MSE added over the base +6.13 to +7.89 against the agent +5.80 to
  +6.44 on fresh seeds, the same ordering held out; n = 3 per arm so a direction and not a result, and it
  spends 1.23-1.49x the base pitch travel against 1.22-1.31x). **Mechanism, and it is in the per-wind
  block:** both arms are exactly +0.00 at 12 m/s by construction and no worse than -0.09 at 14 m/s, which is
  exactly where the ungated failures of s34 and s49 were (3 to 7 points of tower fatigue lost at 12-16 m/s),
  so **the gate removes the bins where the constraint binds and supplies the feasibility the proposer used
  to supply** - the same reading covers s40, where the arms tied on a base that already holds its own loads.
  So s49's "the constraint is the proposer" is scoped **in the text, not in a limitation** to the ungated
  PI base; leg 3 credits the searched reward STRUCTURE, not the proposer; and the gate itself is prior art
  (Kim arXiv:2609.21307, Abbas IET CTA 2026) and must be attributed where it is introduced. Open, ~3 h and
  not started: the fixed-reward arm under the gate on this base, which says whether SEARCHING the structure
  buys anything there at all.
- **THE LEARNED LAYER RE-TRAINED ON THE BASELINE THAT IS A TUNING FILE (roadmap s56, 2026-10-13;
  `campaign_gate_native.sh` arm `tnrwCwG3t`, three agent seeds, 4 h 32 min, identity check of the new paired
  tree exactly 0.000; `scripts/dev/gated_native_table.py`, `docs/tables/gated_native.csv`).** The manuscript's
  PI-base rung (s34/s37/s41) was trained against the superseded baseline's own rollouts, so it could not be
  re-read onto the baseline s55 fixed. Fresh seeds 7-10, against the paired shipped controller: native tuned
  baseline +25.3 / +29.7 / +25.3 / +1.7 at 2.016x travel -> + gated layer +31.1 / +35.9 / +25.3 / +3.2 at
  2.460x (seed 1 of 3; the three seeds span 2.46-2.61x, ADC 0.118-0.131). Paired against its own base the layer
  adds power MSE **+5.80 to +6.44** and speed MSE **+5.91 to +6.78** on fresh seeds (+5.40 to +6.24 and +5.54 to
  +6.58 held out), every interval clear of zero, tower DEL +0.01 to +0.64 (**never negative**, one seed
  unresolved), blade DEL +0.47 to +1.48, and under the rule that DECIDES the proposer question (the worst single-wind load term
  against its own base, tolerance -1 point, as in s49 - not the weaker per-wind Cw against the shipped
  controller that s56 quoted) the arm holds 3/3 on both seed sets, seed 1 by only 0.02 points at 18 m/s
  (s58.2, `scripts/dev/gate_proposer_control.py`). 150 s training side:
  3/3 strict, all four terms positive in every seed, within half a point of each other (the old runs spanned six
  points). **The layer is about three quarters of what it was on the hand-scaled row and cleaner** - two of the
  three superseded-base seeds lost tower DEL (-0.31 [-0.62, -0.05], -0.46 [-0.77, -0.15]), none of the native
  ones does. **The headline decomposition must be written per term:** of 58.0 points of speed MSE, tuning +29.7
  (51 %), model-based +25.7 (44 %), learning +2.6 (5 %); of 51.5 power MSE, 49 / 46 / 5 %; but of 18.9 tower
  DEL, tuning +25.3, model-based **-7.9**, learning +1.4 - so the old "49 / 45 / 6 %" was an average over terms
  that disagree in sign, and no single set of shares describes the ladder. **Duty does not explain the gap
  between the two routes:** the learned PI route sits at 2.46x shipped travel and the MPC alone at 2.08x, yet
  the MPC has +19.5 [18.3, 20.7] more speed MSE at less actuation while giving up -7.9 of tower DEL.
- **WIND-GATED residual (roadmap s41; `EnvConfig.residual_gate`, `--residual_gate 15 17`, `docs/tables/gated_residual.csv`):
  the current best configuration.** The layer acts only above 15-17 m/s, where the base leaves room. All 16 rows (two
  bases x seeds x two seed sets) improve their base with the paired interval above zero, ON FRESH TurbSim seeds 7-10 that
  no training, selection or earlier report has seen. Scheduled MPC 30.46 -> 32.54 (+1.85 mean, was +0.81 ungated);
  tuned ROSCO 15.93 -> 20.68 (+4.41 mean at 1.40-1.45x travel, vs +4.02 at 1.29-1.61x ungated). Headline vs the ORIGINAL
  GSPI on those fresh winds: power MSE -51.5 %, speed MSE -58.0 %, tower DEL -18.9 % at 2.9x travel; vs the TUNED ROSCO
  -36 % / -39 % / -13 %. The fixed reward also learns with the gate on the MPC base but loses 5.9 points of tower
  fatigue at 20 m/s where the agent seeds stay within 2.0.
- **Agent layer STACKED ON the scheduled MPC (roadmap s39; `campaign_probe_mpc.sh`, GO / NO-GO rule committed before the
  results):** paired baselines = the MPC's own rollouts (`~/wtrl_mpc_range`), per-wind Cw. Agent-written reward: held-out Cw
  positive 3/3, 600 s J 29.72 / 29.25 / 30.62 against the MPC alone 28.95 (+0.76 / +0.30 / +1.67, every paired interval above
  zero; tower +0.9 to +2.1 points, power MSE worse at 12-14 m/s); fixed reward learns nothing. With n = 5 and controls
  (s40, `docs/tables/mpc_residual.csv`): agent +0.81 mean (5/5 intervals > 0), random reward structure +1.10 (3/3), fixed
  +0.1 - the searched-structure loop adds ~+1 J on the strongest controller, the LLM proposer is NOT distinguishable from
  a random structure on this base (it is on the PI base, s37). Headline stack on the same 28 held-out episodes: tuned
  ROSCO 15.24 -> scheduled MPC 28.95 -> MPC + residual 29.6-30.7. Next: the wind-gated residual (`--residual_gate 15 17`,
  `campaign_gate.sh`, report on fresh wind seeds 7-10).
- **Agent layer on the tuned ROSCO (roadmap s34; first campaign after the three defects of s32 were fixed):** residual
  trained on the above-rated range winds under per-wind Cw. Held-out Cw (150 s, 28 episodes, vs the tuned ROSCO): agent
  reward +1.1 / +6.2 / +7.1 (3/3 positive, per-wind violation <= 0.16 %), fixed reward with tower weight 10 -27.9 / -7.9 /
  -3.7 (0/3; tower -3 to -7 % at 12-16 m/s). 600 s range set, best seed: tuned ROSCO 15.24 -> **19.62** at 1.38x travel,
  +4.37 J [4.12, 4.61] with power +6.3, speed +8.0, tower +2.3, blade +1.0 all excluding zero; fixed residual +1.17.
  An improvement layer for the PI loop, not a substitute for the MPC (28.95 at 2.14x).
  n = 5 and two controls (s37, `scripts/dev/range_residual_table.py`): 600 s J agent 19.27 (+4.02 over the tuned ROSCO),
  random reward structure 18.74, fixed rewards 16.4-17.2 - a structural reward search buys the LEVEL; both loads within
  -1 % at every wind speed at 600 s: agent 4/5, random 0/3, fixed 0/8 (p = 0.003) - the agent is what HOLDS THE LOADS.
  Clean re-run on the low-turbulence 150 s bank (s35): no arm finds a load-holding gain there - the agent layer pays
  where the baseline leaves room (class B from 18 m/s up), not where the tuned ROSCO already matches the MPC.
- **Residual on the tuned ROSCO, constrained objectives (roadmap s32; `eval/fitness.py` C / CT / Cw / CTw, `--objective`):**
  18 runs, no checkpoint holds the constraints per wind speed on held-out winds; what the agent-written rewards find
  are trades (regulation for tower fatigue near rated, tower fatigue for regulation above rated). Two defects found:
  a set-mean constraint is gamed across wind speeds (use the per-wind Cw / CTw), and the wind-labelled above-rated
  subset was enabled for objective J only (fixed in a969ce9; those 18 runs were trained under the oracle subset, the
  clean re-run is `campaign_next2.sh`). Third base on which the residual only re-allocates.
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
- **Objective decides the lever** (roadmap v2 §9 vs 08-30 §21): on F the reward-weight *proposer*
  mattered (llm_fork > random_fork 7/7); on J the untuned default collapsed into a rollback loop
  (12.5 m/s speed-MSE), hyper-parameter search repaired it (+6…+8, proposer-agnostic), and once the
  default is fixed the effect shrinks. Mechanism chain: critic scale (`--value_norm`), credit window
  (γλ), reward gating by the objective's own subset.

### Where the legs meet, and the measurement discipline

- **THE LADDER IN THE FOUR PUBLISHED REDUCTIONS (roadmap s54, 2026-10-12; `scripts/dev/ladder_published.py`,
  `docs/tables/ladder_published_{fresh,heldout}.csv`, no simulations - the rungs are joined on the wind file so
  all rows are the same 28 episodes).** Fresh seeds 7-10, each term against the paired shipped controller:
  superseded tuned baseline +24.5 / +30.7 / **+7.1** / +1.4 at 1.120x travel; native tuned baseline +25.3 /
  +29.7 / **+25.3** / +1.7 at 2.016x; scheduled offset-free LPV-MPC +49.1 / +55.4 / **+17.4** / -0.0 at 2.076x;
  MPC + gated layer +51.5 / +58.0 / +18.9 / +1.8 at 2.899x. **The model-based controller's tower-fatigue
  advantage was an artefact of the baseline not being a tuning file:** +10.3 against the superseded row,
  -7.91 [8.85, 6.97] fresh and -7.43 [8.37, 6.44] held out against the native one, paired intervals clear of
  zero; the learned layer recovers +1.4 to +2.0 of it, not enough. The two baselines are indistinguishable in
  regulation (speed MSE -0.91 [-1.94, +0.10]), so the native tuning buys +18.2 of tower DEL for no regulation -
  it buys it with pitch travel, 1.12x -> 2.02x, which is also where the MPC sits (2.08x), so **at equal duty the
  second rung is a TRADE: +23.7 power and +25.6 speed for -7.9 tower.** The composite hid it by averaging the
  four terms. The PI-base learned rung was retrained on the native baseline (s56), which is the row below.
- **Control direction (2026-09-16/17, roadmap s22-s26):** every MPC row buys its J with ~7.6x the GSPI pitch
  travel (s22); the travel is broadband, not 3P (a 3P notch removes the line, cuts travel 2-3 %, costs 1.5-2.4 J,
  s25); the J-vs-actuator-duty front from the 600 s search cache has its knee at ~3x GSPI travel, and that knee
  holds held-out: same J as the J-selected point at 38-41 % of its actuation, blade DEL 3 -> 5-6 % (s26). On a
  compensated base the residual has nothing of size left to learn (5 of 6 mismatched runs never beat episode 0,
  s24) and residuals do not transfer across model error while the estimator holds +-15 % (s23). ROSCO's tower
  damper is worth ~0 J at 600 s (s23). Literature scan: `docs/archive/literature_2026-09-16.md`.
- **Wind-range protocol (2026-09-18, roadmap s27-s28; the main evaluation set from here on):** 12-24 m/s in 2 m/s
  steps, IEC turbulence class B, 600 s, TurbSim seeds 1-6 (42 held-out episodes; `campaign_must1_range.sh`,
  `scripts/dev/range_table.py`, `docs/tables/range_*.csv`). Several s19-s26 statements are specific to the
  low-turbulence 8 / 12.5 / 15 m/s set: over the range the MPC rows travel 1.7-2.2x the GSPI (not 7.6x), Cp/Ct x0.95
  costs the nominal MPC 0.7 J (not half), offset-free compensation is +1.0 J but a trade (+5.6 power, +5.9 speed,
  -6.6 tower, all intervals exclude 0) and every compensated MPC has a negative tower term from 20 m/s up while the
  nominal MPC never does. Report per-wind-speed terms; differences between deterministic controllers are paired
  bootstraps over episodes (`scripts/dev/paired_bootstrap.py`).

### Manuscript

- **THE NARRATIVE IS FIXED AND THE DOCUMENTS ARE REGROUPED (2026-10-15, user's decision, roadmap s60):
  the main text carries the POSITIVE results and the nulls go to the appendix, and the headline
  contribution is the first LLM agent in the loop for wind turbine controller tuning.**
  `docs/paper/NARRATIVE.md` is now the only place the paper's argument lives: the claim, contributions
  C1–C5 with a novelty status each, the ten sections with the evidence for every one, figures F1–F6 and
  tables T1–T7 with their generators, and eight wording guardrails. `docs/paper/EVIDENCE.md` is the
  number book; `docs/appendix/APPENDIX_PLAN.md` is appendices A–J, D1 and D2 being the pre-registered
  proposer null (p = 0.31 against bayes) and the delivery count (0 of 6 against 4 of 6). **C1's novelty was
  checked on 2026-10-15:** four targeted searches plus the 2026-09-23 audit found no LLM in any wind
  turbine control-design loop, so the claim stands as a domain-first with "to our knowledge", with the
  mechanism attributed to ControlAgent (arXiv:2410.19811), Shou (arXiv:2607.26594), Chen (Processes 2026)
  and Wu (Building Simulation 2026). One conference record with a relevant title surfaced in that search
  and could not be verified (repository entry only, no abstract, DOI or text); by the user's decision it is
  REMOVED from the manuscript entirely - neither cited nor mentioned (s61.4). C1 stays "to our knowledge".
  **F2-F5 exist as of 2026-10-15:** `scripts/dev/figures_leg1.py`
  draws the front against pitch duty (`f5_front.png`, all 485 proposed tunings on log/symlog axes) and the
  per-run search traces (`f5_search.png`), and it EXITS NON-ZERO if its first-feasible-gain index for any of
  the 21 runs, or any arm's count and median, disagrees with `docs/tables/rosco_proposer_test.csv` and s59.4;
  `scripts/dev/ladder_published.py --figdir` draws the ladder (`f5_stack.png`, no composite panel, the hand-
  scaled baseline not a rung) and the per-wind panel (`f5_perwind.png`) inside the script that writes the
  table, so figure and table cannot drift. F1 is drawn in TikZ inside `main.tex`; F7 (`f5_overlay.png`,
  `scripts/dev/figures_overlay.py`) overlays the untuned and the agent-tuned controllers as LEVELS per wind
  speed and checks that the shipped level derived from every evaluation is the same rollout (6e-16). The title
  may name the agent and may claim the application as new; by s59.1 it may NOT claim or imply that the agent
  beats another search.
- **main.tex v7 EXISTS (2026-10-15, roadmap s61): rewritten from scratch under `docs/paper/NARRATIVE.md`,
  compiles with MiKTeX (`latexmk -pdf main.tex` in `docs/manuscript`, 16 pages, 7 figures, 9 tables, no
  undefined references).** Ten sections (intro, related work, setup with the loop as Fig. 1, leg 1 with the
  verified duty ladder, leg 2, leg 3, the ladder, how the numbers were read, discussion with the three
  disclosure sentences, conclusion) and appendices A-K (D = the pre-registered proposer null, E = the delivery
  count, both in full). Do NOT edit v6's text back in: the v6 bullet below is history. The v6 weighting numbers
  ("+24.6 per episode, -0.3 Rayleigh-weighted") were measured over the hand-scaled baseline and are VOID;
  `lifetime_del.py` and `wohler_sensitivity.py` now run on the native baseline (s61): the MPC's regulation
  advantage over the native baseline SHRINKS from +25.6 per episode to +13.7 / +6.6 / +1.9 under Rayleigh
  weights at V_ave 10 / 8.5 / 7.5, and its tower disadvantage (-7.9 per episode) is +2.8 / +5.2 / +6.9 under
  the same weights; weighted aggregates carry no interval. The four bib entries added for s2 (Shou,
  Li, ArjomandBigdeli, Gao) were fetched from their arXiv pages on 2026-10-15 - all open access - and carry
  full author lists; `references.md` records each check. Before submission: fill `authors.tex`.
- **THE CLAIM CHANGED ON 2026-10-13 (user's decision): the paper is about what the agentic loop delivers on the
  shipped GSPI and on an untuned MPC, i.e. the three legs of `docs/MAIN_LINE.md`.** v6's single claim - *baseline
  tuning, aggregation convention and wind weighting decide the apparent benefit of advanced pitch control* -
  becomes the paper's **measurement discipline**: it keeps its evidence and its sections, but it moves to the
  setup and the discussion and stops being the headline. `docs/archive/plan_2026-10-12_manuscript-rewrite.md` is
  archived but its site-by-site table is still the correct checklist for reasons (A)-(D); the
  structure below is v6's and is the substrate, not the target.
- **Manuscript (docs/manuscript/main.tex, v6, restructured 2026-09-24; target venue Wind Energy Science).**
  v6's claim, now demoted as above: *baseline tuning, aggregation convention and wind weighting decide the
  apparent benefit of advanced pitch control*. The controllers were INSTRUMENTS under it. Structure: 1 intro, 2 related work, 3 setup, 4 the ladder,
  5 what the reporting choices are worth (5.1 baseline, 5.2 aggregation + weighting, 5.3 the Wohler exponent as the
  INERT control, 5.4 where the layer may act, 5.5 who proposes the reward, 5.6 actuation), 6 discussion, 7 conclusion,
  references, appendices A-G. Headline numbers: of 32.54 points over the shipped controller (fresh seeds), tuning
  15.93 (49 %), model-based +14.53 (45 %), learning +2.08 (6 %); the MPC's power-MSE advantage over the tuned baseline
  is +24.6 [23.4, 25.7] per-episode and -0.3 [-1.7, +1.6] Rayleigh-weighted (an advantage that VANISHES, not a
  reversal -- never write "flip"), while its tower advantage goes +10.3 -> +18.1 -> +29.0. `scripts/dev/convention_flip.py`.
  ONE aggregation rule everywhere: bin mean, then weighted power mean (an earlier table mixed arithmetic and power
  means; the equally weighted tower reduction is 27.8 %, not 22.1 %). v5's and v4's main lines are DISCARDED. The v5
  ladder text is still the substrate:
  shipped GSPI -> tuned GSPI (+15.4) -> scheduled offset-free LPV-MPC (30.5 fresh) -> wind-gated learned layer
  (+1.5-2.1 on the MPC, +4.2-4.8 on the tuned baseline at 1.40x travel), 16/16 paired intervals above zero on
  held-out AND fresh seeds; then the site-weighted reading (s42) and ablations (shape vs level, proposer vs
  random structure, gate vs no gate, estimator vs residual). Figures `figures/f5_{stack,perwind,weighting,duty}.png`
  from `scripts/dev/figures_v5.py`. Novelty audit `docs/paper/literature_2026-09-23.md`: the GATE (Kim arXiv:2609.21307,
  Abbas IET CTA 2026), the WEIGHT SCHEDULE (Wintermeyer-Kallen 2021) and OFFSET-FREE MPC in wind (Liu IEEE TII 2024)
  are prior art and must be attributed, never claimed; what survives is CPC-only beating a TUNED ROSCO on regulation
  and both DELs, and the shape-vs-weights discriminator (cite Wu Building Simulation 2026, Chen Processes 2026).
  Engage Anand & Bottasso WES 2026 (same turbine/simulator, offline model repair) directly. Report 600 s simulated /
  20 s discarded / 580 s scored, and ADC against the 10 deg/s rate limit alongside the travel ratio.
  arXiv style `docs/manuscript/arxiv.sty`. No development shorthand.

### Infrastructure, data, and F-era history

- **After any interruption (VM restart, full disk) run `scripts/dev/integrity_check.py --since <time> [--delete]` FIRST:**
  a full disk left 7 zero-byte TurbSim fields (the generator skips existing names) and 9 unreadable baseline .npz.
- Data: WSL `~/wtrl/{exp,wind,baselines/openfast}`, copy at `../wtrl-data/`; J tables via
  `scripts/dev/j_table.py [--csv]`, F-era tables in `docs/tables/` (`scripts/dev/build_tables.py`);
  provenance and caveats C1–C12 in `docs/paper/RESULTS_2026-09-10_data-and-caveats.md`.

- F-era headline, kept as history (08-30 roadmap §21–§23, REPORT_2026-09-01): llm_fork 17.89 ± 2.02
  (7 seeds) > random_fork 12.36 ± 5.30 in 7/7 seeds (exact p = 0.0156); fork verification adds nothing
  on top of the LLM proposer; every paired comparison replicates on S7–S10; F1 = mono runs at +0.05
  speed-std ratio in 10/10 seeds. All of it is one wind bank (caveat C1); never regenerate it.
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
