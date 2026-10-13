# The main line (2026-10-13)

**Read this before any other document in `docs/`.** Everything else is evidence for one of the three legs
below, measurement discipline that makes the legs believable, or history. If a new experiment does not change a
number on this page, it belongs under a leg heading in `README.md` and nowhere else.

## The claim

**Agentic search buys feasibility and sample efficiency on real controllers; it does not buy peak level.**

The project puts the same agentic loop on three different objects - the shipped ROSCO pitch loop, an untuned
LPV-MPC, and a residual layer's reward structure - and all three come out the same shape: the agent turns a
design that is not deliverable into one that is, in few simulations, where random search finds nothing; and in
all three it fails to beat a careful human or a fixed classical recipe on the level of the result.

One scope in that sentence is now measured instead of assumed. The contrast against *random search in the
design space* is what legs 1 and 2 carry (0 of 3 runs, 0 of 48 simulations). In leg 3 the contrast is against a
random draw from a reward grammar, and it holds only **without** the wind gate (s49 against s58): with the gate,
the configuration supplies the feasibility and the proposer is not distinguishable.

That shape is the result. It is also why the alternative claim is not available: see *what cannot be claimed*.

## Leg 1 - the shipped GSPI, tuned by the agent

Object: ROSCO's own parameters (`omega_pc`, `zeta_pc`, `F_LPFCornerFreq`, `FA_KI`, `FA_HPFCornerFreq`), every
candidate a tuning file the controller can take. Evidence: roadmap s52-s55,
`docs/design_2026-10-11_rosco-native-search.md`, `docs/tables/rosco_band_verified.csv`,
`rosco_tuned_baseline.csv`, `rosco_perwind.csv`.

Agent-found rows that hold both load terms at **every** wind speed on 28 held-out *and* 28 fresh episodes
(four published reductions vs the paired shipped controller, fresh seeds):

| row | power | speed | tower | blade | travel | ADC |
|---|---|---|---|---|---|---|
| `llm_r1#9` - **one line changed** (`F_LPFCornerFreq` 1.57 -> 3.0) | +17.7 | +23.0 | +8.7 | +2.0 | **1.000x** | 0.0480 vs 0.0476 shipped |
| `llm_r1#11` - the same line at 6.5 rad/s | +24.2 | +28.3 | +16.3 | +2.5 | 1.647x | - |
| stage 1, **the baseline row** - filter + damper (fixed recipe, *not* an agent arm) | +25.3 | +29.7 | +25.3 | +1.7 | 2.016x | 0.0968 |

- **Search efficiency (s53, 12 runs, 321 tunings):** runs that bought a shippable gain - llm **3/3**, bayes 2/3,
  local 1/3, **random 0/3**; median 8 simulations to the first gain for the llm arm.
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
2. **"The LLM beats other search."** es equals it on leg 2; local search found two verified rows on leg 1; and
   under the gate on the native baseline a random draw from its own reward grammar equals it on the load rule
   and is not behind on the level (3/3 against 3/3, s58). It separates in exactly one configuration of this
   study, the **ungated** PI base (s49).
3. **A composite score.** Report the four published reductions, each on its own. The symbol *J* is the MPC cost
   function. The composite stays as the *selection and training* objective of the campaigns, under its own name.
4. **"ROSCO's tower damper is worth about nothing"** (withdrawn in s52: its gain had been capped at 0.004, worth
   0.25 deg of authority; searched, it asks about 2 deg and buys most of the +25.3 tower DEL).
5. **The duty argument against the model-based controller** (gone in both directions, s53 and s56: a PI tuning
   reaches +33 regulation at 1.12x travel, and the learned PI route reaches 2.46x without catching the MPC).
6. **One decomposition of the headline.** Of 58.0 points of speed MSE the full stack buys over the shipped
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
  1-2 design, 3-6 held out, 7-10 fresh (`docs/RESULTS_2026-09-10_data-and-caveats.md`).
- **Actuator duty beside every row:** pitch travel ratio and ADC against the 10 deg/s rate limit.
- **One aggregation rule:** bin mean, then weighted power mean. Site weighting is not cosmetic - the MPC's
  power-MSE advantage over the tuned baseline is +24.6 per episode and -0.3 Rayleigh-weighted
  (`scripts/dev/convention_flip.py`); an advantage that *vanishes*, never a reversal.
- **The baseline is a band, not a row** (s52.3, s55.2): 1.00x to 2.72x travel, +22.6 to +37.0 regulation. Name
  the member used in every comparison.

## Open, in order

1. **The fixed-reward arm under the gate on the native base** - three runs, about 3 h, not started. s58 showed
   that *who* proposes the reward structure does not matter under the gate; this says whether **searching** the
   structure matters there at all. If a fixed reward also holds the loads, leg 3 collapses to the gate; if it
   does not, the structure search is load-bearing and only the proposer is not. s41 has the fixed reward under
   the gate on the MPC base losing 5.9 points of tower fatigue at 20 m/s, which points to the second reading,
   but the native base has never had that arm. It is the last question that can still move this page.
2. Leg 1's agent-vs-random comparison is n = 3 per arm and was counted under the as-stated objective; restated
   under the per-wind rule it is 2 verified rows of 7 proposals against none proposed.
3. The manuscript rewrite, `docs/plan_2026-10-12_manuscript-rewrite.md` - all of its gates are now open.
