# The main line (2026-10-13)

**Read this before any other document in `docs/`.** Everything else is evidence for one of the three legs
below, measurement discipline that makes the legs believable, or history. If a new experiment does not change a
number on this page, it belongs under a leg heading in `README.md` and nowhere else.

**This page and the paper are not the same scope (user's decision, 2026-10-15).** This is the project's
complete record, nulls included. The *paper's* narrative is `docs/paper/NARRATIVE.md`: the main text carries
the positive results, and the nulls and negative results - including the pre-registered proposer endpoint and
the delivery count below - are appendix material (`docs/appendix/APPENDIX_PLAN.md`, items D1-D2). Nothing on
this page is softened to make that split work; the three sentences of disclosure that keep the paper honest
about it are specified in `NARRATIVE.md` §6.

## The claim

**One agentic loop, three design spaces, no rebuild: it beats random search in all three, reaches the level of
the best dedicated search in each without exceeding it, and never buys peak level.**

The project puts the same loop - the same prompt shape, the same history, no acquisition function and no
population - on the shipped ROSCO pitch loop, an untuned LPV-MPC and a residual layer's reward structure. Three
scopes, all measured rather than assumed:

- **Against random search it separates in all three legs:** leg 1 llm 5 of 6 runs with a gain against random
  0 of 3 (p = 0.048, s59.4), leg 2 0 of 48 for random search (s38), leg 3 4 of 5 against 1 of 8 (p = 0.032,
  ungated, s49).
- **Against a dedicated search it is at parity in all three:** leg 1 against constrained Bayesian optimisation
  p = 0.31 and against a local search p = 0.19 at n = 6 on the pre-registered endpoint (s59.4); leg 2 llm = es
  (s38); leg 3 under the gate 3 of 3 against a random draw from its own reward grammar, which also holds with
  more margin (s58). **No leg of this study has a proposer claim against a dedicated search.**
- **It never buys the level:** a fixed recipe wins leg 1, a hand design wins leg 2, and the searched reward
  *structure* rather than its proposer carries leg 3.

**Where feasibility is concerned the legs disagree, and leg 1 now cuts the other way.** On the MPC the loop
repairs an 8.8-point per-wind violation for 1.7 points of regulation (s57.2), and on the ungated PI base the
agent arm is what holds the loads (s49). On the shipped PI loop it does the opposite: it optimises the stated
set-mean objective *better* than the classical searches, walks further up the regulation front than the rule
can support, and therefore delivers nothing at all under the rule that ships - **0 of 6 runs against 4 of 6
for a plain local search** and 2 of 6 for Bayesian optimisation, all 21 runs measured (s59.6, final; Fisher
two-sided p = 0.061 against the local search). The fourth instance of a set-mean constraint being gamed
(s32, s53.2, s55), with the agent on the losing side of it. Read again with one point of margin in the
screen (s62.1, pre-registered, no search re-run), the same candidates deliver in 2 of the agent's 6 runs (0
before), 2 of 6 for Bayesian optimisation (a different pair) and the same 4 of 6 for the local search: the
screen was what failed in two of the three agent runs that had a candidate, and the agent's delivered rows
carry the most regulation of any arm (+32.0 (5.2) against +19.9 and +26.0). The arm that is told the rule
ran next (s62.2, s62.3): it drives to the screen's edge in all six runs - six selections within omega 0.60-0.64,
zeta 0.70-0.81, corner 3.4-3.75 at 1.12-1.20x travel, design worst single wind -0.0 to +0.4 - and ships in
**1 of 6**, five of six losing 1.6-2.2 points of tower DEL at 24 m/s on fresh seeds; local 3 of 6 (replays of
s59 in four runs), bayes 2 of 6. The per-wind screen at two episodes per wind is gamed too, and the set-mean
tower margin predicts it for the sixth time (failing rows +3.5 to +5.0, shipping rows +7.4 and above with one
exception at +3.9).

What the loop does win on leg 1, post hoc and reported as such: **it never proposed a candidate the plant
could not run** (0 of 144, against 13 of 144 for the local search, 9 of 144 for Bayesian optimisation and 16 of
72 for random draws), and its runs repeat each other where a GP with a random start does not. See *what cannot
be claimed* for the rest.

## Leg 1 - the shipped GSPI, tuned by the agent

Object: ROSCO's own parameters (`omega_pc`, `zeta_pc`, `F_LPFCornerFreq`, `FA_KI`, `FA_HPFCornerFreq`), every
candidate a tuning file the controller can take. Evidence: roadmap s52-s55,
`docs/paper/design_2026-10-11_rosco-native-search.md`, `docs/tables/rosco_band_verified.csv`,
`rosco_tuned_baseline.csv`, `rosco_perwind.csv`.

Agent-found rows that hold both load terms at **every** wind speed on 28 held-out *and* 28 fresh episodes
(four published reductions vs the paired shipped controller, fresh seeds):

| row | power | speed | tower | blade | travel | ADC |
|---|---|---|---|---|---|---|
| `llm_r1#9` - **one line changed** (`F_LPFCornerFreq` 1.57 -> 3.0) | +17.7 | +23.0 | +8.7 | +2.0 | **1.000x** | 0.0480 vs 0.0476 shipped |
| `llm_r1#11` - the same line at 6.5 rad/s | +24.2 | +28.3 | +16.3 | +2.5 | 1.647x | - |
| stage 1, **the baseline row** - filter + damper (fixed recipe, *not* an agent arm) | +25.3 | +29.7 | +25.3 | +1.7 | 2.016x | 0.0968 |

- **Search efficiency, at n = 6 per arm and on a pre-registered endpoint (s59.4, 21 runs, 561 tunings; the
  s53 numbers below it were n = 3):** simulations to the first feasible gain, one-sided exact permutation -
  llm against bayes **p = 0.31**, against local p = 0.19, against random p = 0.048. Runs with a gain: llm 5/6,
  bayes 5/6, local 4/6, random 0/3. Medians 10 / 18 / 14 / censored. The s53 reading (llm 3/3, median 8
  against 14 and 18) did not survive doubling - llm r3 found nothing, bayes r4 found its gain in 5 simulations.
  Regulation of the selections is still highest in the llm arm, +38.40 (8.47) against local +29.23 (12.14) and
  bayes +23.79 (14.03), and that is the part of the front where s55 found everything fails per wind.
- **What each arm delivers under the rule that ships, all 11 per-wind selections now measured on 28 held-out
  and 28 fresh episodes (s59.6, final): local 4 of 6, bayes 2 of 6, llm 0 of 6, random 0 of 3.** Fisher
  two-sided p = 0.061 for llm against local (0.030 one-sided in the direction the data went, which is the
  opposite of the one s59.1 set up), p = 0.45 against bayes. The deliveries carry +19.87 (7.43) and +19.94
  (4.76) points of regulation; the model's two distinct selections miss the tolerance by 0.17 and 2.40.
  **Position on the front is the mechanism:** everything that ships sits between +11.6 and +28.7 points of
  regulation and everything that fails between +26.7 and +37.7, the model's three selections being the highest
  rows of all. The one row a user would take out of 21 runs is **local r3 #8 - one filter corner and a small
  damper, +28.66 of regulation at 1.05x shipped pitch travel**, and the two rows with the largest margins are
  the two the campaign's own stage 3 never looked at (local r4 #10, r5 #18). **Under the margin screen of
  s62.1 (worst single-wind load term >= 0 on the design set, shipping rule unchanged) the same runs read
  local 4 of 6, bayes 2 of 6, llm 2 of 6:** llm r1 #11 (+28.35 at 1.57x) and r5 #14 (+35.71 at 2.18x, the
  most-regulating delivered row of the 21 runs) ship, llm r2 #12 misses by 0.50 at 24 m/s on fresh seeds;
  Fisher p = 0.57 against local. The agent's candidates deliver when the screen lets them, in 2 of the 3
  runs where it can; the classical arms still deliver at least as often.
- **Post hoc, and the one difference that is large: 0 of 144 llm candidates destabilised the plant**, against
  13 of 144 local (Fisher p = 0.00018), 9 of 144 bayes (p = 0.0034) and 16 of 72 random (p < 1e-5); every one
  of those 144 was really proposed by the model (the malformed-reply audit reads 0 in all six runs). It also
  repeats itself across seeds: 2 distinct per-wind selections out of 3, against 4 of 4 in both classical arms.
- **Per-arm census under the per-wind rule.** The design-set screen leaves exactly 10 tunings with positive
  regulation (an eleventh is the shipped tuning itself, at 0.00 by construction) and s55 measured all 10 off
  the design winds, so this census is complete: **llm proposed 7 and 2 hold** (`llm_r1#9` and `llm_r1#11`),
  **local proposed 2 and both hold** (both at *sub*-shipped duty, 0.976x and 0.880x, buying little regulation),
  bayes proposed 1 and 0 hold, random proposed none. The s53 count (llm 3/3 runs, random 0/3) was taken under
  the objective as stated; this is the count under the rule that now decides. Both belong in the paper.
- **Against the agent: it does not win the level.** The selected baseline came from `--proposer tune`, the fixed
  s29 recipe, which the source marks "not as an arm of the comparison". And the agent's most-regulating
  proposals (+47.79, +41.27 and four more) all lose 2.2 to 6.6 points of tower-base fatigue at a *single* wind
  speed off the design winds (s53.2, s55). The agent walks up the regulation front and switches the damper off.

## Leg 2 - the untuned MPC, designed by the agent

Object: the cost structure of the offset-free LPV-MPC - the tower weight and its wind schedule. Evidence:
roadmap s33, s38, s57, `scripts/mpc_design_search.py`, `scripts/dev/mpc_leg_table.py`,
`docs/tables/design_search.csv`, `docs/tables/mpc_leg_{heldout,fresh}.csv`.

- **s38:** from the hand-set **unscheduled** reference, the llm and es proposers both discover the wind schedule
  in 24 simulations and repair the per-wind violation on held-out winds; **random search 0/48**. llm proposals
  beat the reference more often than es (25/48 vs 15/48, p = 0.03) but the two are equal in the result, and
  neither beats the hand design of s33.
- **What it is worth in published terms** (fresh seeds 7-10, 28 episodes, paired on the same wind file):
  unscheduled +50.7 / +57.1 / +15.6 / +0.5 at 1.912x travel; agent-found schedule +49.1 / +55.4 / +17.4 / -0.0
  at 2.076x. Paired difference **-1.65 power, -1.74 speed, +1.83 tower [+1.29, +2.72], -0.53 blade**, every
  interval resolved. Held-out seeds 3-6 give the same thing: -1.59 / -1.63 / +1.49 / -0.50.
- **Per wind speed is where the repair is, and the set mean hides all of it.** Tower-base DEL reduction at
  24 m/s, fresh seeds: **-8.76 unscheduled -> -0.33 scheduled**; at 22 m/s +0.55 -> +3.44 (held out: -6.27 ->
  +0.08 and -2.46 -> +0.24). The per-wind load rule of s28 is *no term below -1 % at any wind speed*: **the
  unscheduled controller fails it by 8.8 points on winds nothing has seen, and the agent-found schedule is
  inside the tolerance** - though at -0.33 it is inside it, not positive. That is the whole of leg 2: the agent
  buys 8.4 points of the worst single-wind load term for 1.7 points of regulation. A feasibility repair, not a
  level gain.
- **The repair does not make the model-based route the load-optimal one.** On the same fresh episodes the tuned
  PI loop beats *both* MPC rows on tower-base fatigue at every wind speed from 16 m/s up: +26.9 / +20.5 / +18.7
  / +16.9 / +15.3 against the scheduled MPC's +17.7 / +6.5 / +5.4 / +3.4 / -0.3, and it does so at comparable
  duty (2.016x against 2.076x). This is the per-wind form of the s54 and s56 finding - the model-based rung
  buys regulation and spends tower fatigue - and it is why no single decomposition of the headline exists.

## Leg 3 - the learned layer, and who writes its reward

Object: a wind-gated residual pitch layer on top of either base, trained under a per-wind constrained objective,
with the reward *structure* proposed by the agent. Evidence: roadmap s34, s37, s39, s40, s41, **s49**, s56,
**s58**; `docs/tables/gate_proposer_control.csv`, `gated_native.csv`, `gated_residual.csv`, `mpc_residual.csv`,
`range_residual.csv`.

- **On the baseline that is a tuning file (s56, the current row):** paired against its own base on the same 28
  fresh episodes, three agent seeds - power MSE **+5.80 to +6.44**, speed MSE **+5.91 to +6.78**, tower DEL
  +0.01 to +0.64 (never negative), blade DEL +0.47 to +1.48; 1.22-1.29x the base pitch travel. **Under the rule
  that decides the proposer question** - the worst single-wind load term against its own base, tolerance -1
  point, as in s49 and not the weaker per-wind Cw against the shipped controller that s56 quoted - the arm is
  **3/3 on both seed sets** (s58.2, `docs/tables/gate_proposer_control.csv`), but seed 1 holds by **0.02
  points** (tower DEL at 18 m/s, fresh seeds), so a count a re-run could flip; say so wherever 3/3 appears.
- **On the scheduled MPC (s41, s54):** +2.41 power, +2.59 speed, +1.44 tower, +1.87 blade on fresh seeds, at
  2.076x -> 2.899x travel.
- **Who writes the reward: it depends on the configuration, and the honest statement is narrower than it was.**
  On the **ungated** PI base, settled at n = 8 (s49): runs whose loads hold under the per-wind rule at 600 s are
  agent-written reward **4/5**, random reward structure with the same fork verification **1/8**, fixed reward
  0/8; Fisher two-sided **p = 0.032** against the random structures, p = 0.007 against the fixed rewards. The
  *level* of the two searched arms is identical (+4.02 against +4.01). **More search found the level; it did not
  find the constraint.**
- **Under the GATE, on the baseline the paper headlines, that difference is gone (s58, 2026-10-13, the matched
  control, rule pre-registered).** Agent **3/3**, random reward structure **3/3**, both seed sets, Fisher
  two-sided **p = 1.0** - and the control holds with *more* margin: no control seed is negative at any wind
  speed on the held-out set, against the agent arm worst cell of -0.98. The control is also not behind in level
  (power MSE added +6.13 to +7.89 against the agent +5.80 to +6.44 on fresh seeds; consistent on both seed sets,
  n = 3 per arm, so a direction and not a result, and the control spends more actuation).
- **Why, and it is the reading that matters:** both arms are +0.00 at 12 m/s by construction and no worse than
  -0.09 at 14 m/s, which is exactly where the ungated failures of s34 and s49 were (3 to 7 points of tower
  fatigue lost at 12-16 m/s). **The gate removes the bins where the constraint binds, so the configuration
  supplies the feasibility the proposer used to supply.** Same reading covers s40, where the arms were
  indistinguishable on a scheduled MPC base that already holds its own loads. So leg 3 credits the **search
  space** - a searched reward *structure* buys the level, and the constraint where it binds - and credits the
  *proposer* in exactly one configuration of this study, which is not the headline one. The gate itself is prior
  art (Kim arXiv:2609.21307, Abbas IET CTA 2026) and must be attributed where it is introduced.

## What cannot be claimed

1. **"The agent designs a better controller."** It loses the level to a fixed recipe on leg 1 and to a hand
   design on leg 2 (s53.2 and s38 state this explicitly).
2. **"The LLM beats other search."** Tested at n = 6 per arm on leg 1 with the endpoint fixed in advance and
   it does not: p = 0.31 against constrained Bayesian optimisation, p = 0.19 against a local search, 5/6
   against 5/6 and 4/6 on the counts (s59.4). es equals it on leg 2; under the gate a random draw from its own
   reward grammar equals it and holds with more margin (3/3 against 3/3, s58). It separates from **random
   search** in all three legs and from a dedicated search in none. The one remaining proposer separation in
   the whole study is the **ungated** PI base of leg 3 (4/5 against 1/8, p = 0.032, s49).
3. **"The agent finds the shippable design."** The reverse is measured: off the design winds the classical
   arms deliver a tuning holding both loads at every wind speed and the model arm never does (local 4/6,
   bayes 2/6, model 0/6, s59.6). The model finds the largest *set-mean* number fastest, and that number does
   not ship. With one point of margin in the screen the same runs read 2/6 against 4/6 and 2/6 (s62.1): the
   agent's candidates ship where the screen lets them, and the classical arms still ship at least as often.
   Told that screen, the agent drives to its edge in all six runs and ships in 1/6 (s62.3).
4. **A composite score.** Report the four published reductions, each on its own. The symbol *J* is the MPC cost
   function. The composite stays as the *selection and training* objective of the campaigns, under its own name.
5. **"ROSCO's tower damper is worth about nothing"** (withdrawn in s52: its gain had been capped at 0.004, worth
   0.25 deg of authority; searched, it asks about 2 deg and buys most of the +25.3 tower DEL).
6. **The duty argument against the model-based controller** (gone in both directions, s53 and s56: a PI tuning
   reaches +33 regulation at 1.12x travel, and the learned PI route reaches 2.46x without catching the MPC).
7. **One decomposition of the headline.** Of 58.0 points of speed MSE the full stack buys over the shipped
   controller: tuning 51 %, model-based 44 %, learning 5 %; of 51.5 power MSE: 49 / 46 / 5 %; but of 18.9 tower
   DEL: tuning +25.3, model-based **-7.9**, learning +1.4. No single set of shares describes the ladder.

## Measurement discipline (why the legs are believable)

This is the part that used to be mistaken for the claim. It is method, and it belongs in the paper's setup and
discussion, not in its title.

- **The four published reductions only**, each against the *paired* shipped controller on the same wind file;
  differences are stratified paired bootstraps over episodes (`scripts/dev/paired_bootstrap.py`). Rungs are
  joined on the wind file so every comparison is the same episodes (`scripts/dev/ladder_published.py`).
- **Load constraints hold per wind speed, measured off the design winds.** A set-mean constraint is gamed, with
  learning (s32) and without it (s53.2, s55) - three instances. Screen on the design set, *measure* off it.
- **Protocol:** 12-24 m/s in 2 m/s steps, IEC class B, 600 s simulated / 20 s discarded / 580 s scored; seeds
  1-2 design, 3-6 held out, 7-10 fresh (`docs/paper/RESULTS_2026-09-10_data-and-caveats.md`).
- **Actuator duty beside every row:** pitch travel ratio and ADC against the 10 deg/s rate limit.
- **One aggregation rule:** bin mean, then weighted power mean. Site weighting is not cosmetic, and it is
  reported as the convention's effect, never as a result (s61, native baseline, `docs/tables/lifetime_del.csv`):
  the MPC's regulation advantage over the tuned baseline is +25.6 per episode and +13.7 / +6.6 / +1.9 under
  Rayleigh weights at V_ave 10 / 8.5 / 7.5 - it *shrinks* - while its tower-fatigue disadvantage, -7.9 per
  episode, is +2.8 / +5.2 / +6.9 under the same weights. The weighted aggregates carry no interval. (The
  "+24.6 per episode, -0.3 Rayleigh-weighted" of v6 was measured over the hand-scaled baseline and is void.)
- **The baseline is a band, not a row** (s52.3, s55.2): 1.00x to 2.72x travel, +22.6 to +37.0 regulation. Name
  the member used in every comparison.

## Open, in order

1. **The selection rule, which is now the cheapest open question in the study (not started).** s59.6 closed
   the leg-1 campaign with the screen, not the search, as the thing that fails: a 14-episode design set with
   two episodes per wind cannot tell -0.60 from -1.20, so the rule hands the user the highest row on the front
   and that row loses the tower-base load off the design winds. Two candidate repairs, both cheap because the
   candidate cache is on disk: screen with a margin (s55 and s59.6 both found the margin predicts holding), or
   spend design episodes at the winds where the constraint binds. This decides whether the search space is
   usable, independently of who proposes.
2. **The fixed-reward arm under the gate on the native base** - three runs, about 3 h, not started. s58 showed
   that *who* proposes the reward structure does not matter under the gate; this says whether **searching** the
   structure matters there at all. If a fixed reward also holds the loads, leg 3 collapses to the gate; if it
   does not, the structure search is load-bearing and only the proposer is not. s41 has the fixed reward under
   the gate on the MPC base losing 5.9 points of tower fatigue at 20 m/s, which points to the second reading,
   but the native base has never had that arm.
3. **The manuscript: v7 is written (2026-10-15) and compiles** - `docs/manuscript/main.tex`, rewritten from
   scratch under `docs/paper/NARRATIVE.md` (positive results in the main text, nulls in appendices D-E with the
   three disclosure sentences in §9), 16 pages, 7 figures (F1 in TikZ, F2-F7 from `figures_leg1.py`,
   `ladder_published.py --figdir`, `figures_weighting.py`, `figures_overlay.py`, each refusing to run or
   cross-checking against its table) and 9 tables. Before submission: fill `authors.tex`. (The four new bib
   entries were fetched from arXiv on 2026-10-15, open access, full author lists; an unverifiable conference
   record found the same day was removed from the manuscript entirely by the user's decision, and C1 stays
   "to our knowledge".) The site-by-site table of
   `docs/archive/plan_2026-10-12_manuscript-rewrite.md` is satisfied: the composite is out of every reported
   number, the baseline is the native band, the two false statements are gone, the PI rung is the retrained one.
