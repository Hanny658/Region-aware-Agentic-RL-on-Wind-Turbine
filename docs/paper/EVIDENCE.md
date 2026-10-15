# The number book (2026-10-15)

Every number the main text needs, in the order the sections need them, each with the file it comes from and
the roadmap section that produced it. This replaces reading the 3357-line roadmap to write the paper: if a
number is not here, it is not a main-text number.

Convention for every row below: the four **published reductions** are percentage reductions against the
**paired shipped ROSCO GSPI** on the same wind file - power MSE / generator-speed MSE / tower-base DEL /
blade-root DEL - followed by energy change, pitch-travel ratio against shipped and ADC against the 10 deg/s
rate limit. Intervals are 95 % stratified paired bootstraps over episodes. Seed sets: **design** 1-2
(14 episodes), **held out** 3-6 (28), **fresh** 7-10 (28).

---

## §3 Setup

| quantity | value | source |
|---|---|---|
| Plant | NREL 5 MW, OpenFAST 4.2, patched ROSCO over 22-channel ZMQ | s7 |
| Wind | 12, 14, 16, 18, 20, 22, 24 m/s, IEC turbulence class B, TurbSim | s28 |
| Episode | 600 s simulated, 20 s discarded, 580 s scored | s28, audit §3.4 |
| Seed split | 1-2 design (14 ep), 3-6 held out (28 ep), 7-10 fresh (28 ep) | s28, `RESULTS_2026-09-10_data-and-caveats.md` |
| Leg-1 search space | `omega_pc`, `zeta_pc`, `F_LPFCornerFreq`, `FA_KI`, `FA_HPFCornerFreq`; every box limit measured on the plant | s51, `design_2026-10-11_rosco-native-search.md`, `scripts/dev/search_box_bounds.py` |
| Why these are the right five | ROSCO builds `Kp = (2 zeta omega + A(beta))/B(beta)`, `Ki = omega^2/B(beta)`; A, B recovered from the shipped tables with a round trip of 8.7e-19; shipped tuning is zeta 0.7 / omega 0.6 | s51, s52 |
| Selection rule | maximise generator-speed regulation reduction subject to tower DEL, blade DEL and energy no worse than shipped; a constraint counts as violated only where the loss exceeds its own 95 % paired-bootstrap interval (measured median half-widths 1.30 / 1.00 / 0.031 points); ranked by constraint domination | s51, `eval/selection.py` |
| Shipped speed-filter corner | 1.57 rad/s, from ROSCO's generic "one quarter of the blade first edgewise frequency" rule - a rule of thumb, not an optimisation | Abbas et al. WES 7:53 (2022) §2.1 |
| Measured filter ceiling | 19.2 rad/s, where the filter passes 99.99 % of the generator-speed power | s51 |
| Budget per run | 24 candidates (leg 1); identity check of the paired baseline tree exactly 0.000 in every run | s53, s59.4 |

## §4 Leg 1 - the shipped pitch controller

### 4.1 The verified duty ladder (`../tables/rosco_band_rows_fresh.csv`, `rosco_band_verified.csv`; s55.2-55.3)

Fresh seeds, 28 episodes. Every row is a tuning file; every row holds **both** load terms at **every** wind
speed on 28 held-out *and* 28 fresh episodes.

| row | what differs from shipped | power | speed | tower | blade | energy | travel | ADC |
|---|---|---|---|---|---|---|---|---|
| **`llm_r1#9`** | **one line**: corner 1.57 -> 3.0 | **+17.73** [16.45, 19.08] | **+22.98** [22.14, 23.83] | **+8.66** [7.63, 9.77] | +2.02 [1.02, 3.04] | +0.103 | **1.000x** | 0.048 (shipped 0.0476) |
| **`llm_r1#11`** | the same line at 6.5 | +24.20 [22.50, 25.93] | +28.35 [27.07, 29.72] | +16.25 [14.34, 18.27] | +2.48 [1.50, 3.48] | +0.175 | 1.647x | 0.0791 |
| stage 1 (s52), *fixed recipe, not an agent arm* | omega 0.605, zeta 0.738, corner 6.529, damper 0.0331, hpf 0.692 | +25.34 [23.39, 27.34] | +29.75 [28.42, 31.13] | +25.32 [23.48, 27.24] | +1.72 [0.66, 2.85] | +0.167 | 2.016x | 0.0968 |
| `local_r0#22` | omega 0.562, zeta 0.663, corner 4.146, damper 0.0018 | +8.05 | +11.42 | +17.00 | +2.73 | +0.156 | **0.974x** | 0.0468 |

- Worst single-wind load term of `llm_r1#9`: **-0.98** (tower, 24 m/s) fresh and **+0.26** held out, inside
  the -1.0 point tolerance on both sets.
- `local_r0#18` (corner 2.595, 0.884x travel) is the **warning row**: power MSE **-1.89** [-2.63, -1.17] with
  speed MSE +0.73 [-0.36, +1.89]. The power/speed redundancy that justifies using speed alone holds on the
  model-based rows (0.94-0.95) and is weak on PI rows (0.19-0.83); say so where that convention is stated.
- The damper buys the last ~9 points of tower fatigue for ~0.37x of travel (`llm_r1#11` -> stage 1).
- Searched, ROSCO's tower damper asks about **2.0 deg** of pitch (limiter 10 deg, not reached) and buys most
  of stage 1's +25.3 tower DEL. The s23 claim that it is worth about nothing is **withdrawn** - its gain had
  been capped at 0.004, worth 0.25 deg of authority (s52).

### 4.2 The one-line ladder on the design screen (`rosco_perwind.csv`; s52.3)

`F_LPFCornerFreq` alone, shipped gains untouched, every load term positive at every wind speed:
3.0 rad/s **+22.62** regulation at **1.004x** travel; 3.8 (3P) +25.91 at 1.06x; 5.0 +27.39 at 1.18x;
6.5 +28.60 at 1.57x. Nothing is traded - it is a default left unexamined, and every comparison that uses the
shipped ROSCO as its baseline starts this far below the same controller with one number changed.

### 4.3 The search, per arm (`rosco_search.csv`, `rosco_proposer_test.csv`; s53, s59.4)

21 runs, **561 tunings proposed**, 486 distinct, 39 the plant could not run, 211 holding the set-mean
constraints, 114 also holding per wind on the design screen. Zero infrastructure failures in the campaign
itself.

| arm | runs | candidates | **unrunnable** | runs reaching a feasible gain | median sims to it | regulation of the selection |
|---|---|---|---|---|---|---|
| llm | 6 | 144 | **0** | 5 | 10 | +38.40 (8.47) |
| local | 6 | 144 | 13 | 4 | 14 | +29.23 (12.14) |
| bayes | 6 | 144 | 9 | 5 | 18 | +23.79 (14.03) |
| random | 3 | 72 | 16 | **0** | censored | - |

- Unrunnable counts, Fisher two-sided against the llm arm: local **p = 0.00018**, bayes **p = 0.0034**,
  random **p < 1e-5**. Malformed-reply audit: **0 in all six llm runs**.
- Random search finds no feasible gain in 72 candidates; a search that finds nothing selects the shipped
  tuning itself and reports 0.00 (s52.1).
- Mechanism of the llm arm, from its own transcript: it sweeps one quantity at a time with the plant
  frequencies as landmarks, which is what found the one-knob ladder of §4.2.
- Selection quality of the campaign as a whole: **21 of 21** selections feasible held out and fresh, design
  against held-out correlation **+1.00** (rank +1.00), optimism -0.43 mean / +0.88 worst (s59.6). The s38
  finding that a small design set cannot rank designs does **not** transfer to this space.

### 4.4 Numbers that belong to the appendix, not here

The pre-registered proposer endpoint, the delivery count under the shipping rule, and the front-position
mechanism: appendix A1-A2 (`../appendix/APPENDIX_PLAN.md`).

## §5 Leg 2 - the untuned MPC (`../tables/mpc_leg_fresh.csv`, `mpc_leg_heldout.csv`, `design_search.csv`; s33, s38, s57.2)

Fresh seeds, 28 episodes, against the paired shipped controller:

| row | power | speed | tower | blade | travel | ADC |
|---|---|---|---|---|---|---|
| offset-free LPV-MPC, **unscheduled** (the hand-set reference the agent starts from) | +50.73 | +57.13 | +15.59 | +0.50 | 1.912x | 0.0918 |
| offset-free LPV-MPC, **agent-found schedule** | +49.08 | +55.39 | +17.42 | -0.04 | 2.076x | 0.0997 |

- Paired difference, schedule minus unscheduled: -1.65 power, -1.74 speed, **+1.83 tower [+1.29, +2.72]**,
  -0.53 blade. Held out: -1.59 / -1.63 / +1.49 / -0.50.
- **The repair is per wind speed and the set mean hides all of it.** Tower DEL at 24 m/s, fresh:
  **-8.76 -> -0.33**; at 22 m/s +0.55 -> +3.44. Held out: -6.27 -> +0.08 and -2.46 -> +0.24. Against the s28
  rule (no term below -1 point at any wind speed) the unscheduled controller **fails by 8.8 points on winds
  nothing has seen** and the schedule brings it inside the tolerance for 1.7 points of regulation.
- Found in **24 simulations** by both the llm and the es proposer; **random search 0 of 48**. llm proposals
  beat the reference more often than es (25/48 against 15/48, p = 0.03) and the two are equal in the result.
  Neither beats the hand design of s33 (say so).
- At -0.33 it is *inside* the tolerance, not positive. Say that too.
- The hand-designed schedule of s33: `qt_sched = [6, 18, 22]` with qt 3 below 18 m/s; held out, J 28.95 at
  2.14x travel, no term below -1 point at any wind speed.

## §6 Leg 3 - the learned layer (`../tables/gated_native.csv`, `gated_residual.csv`; s41, s56)

Wind-gated residual (`--residual_gate 15 17`) on the native tuned baseline, three agent seeds, paired
against **its own base** on the same 28 episodes:

| seed set | d power MSE | d speed MSE | d tower DEL | d blade DEL | travel vs base |
|---|---|---|---|---|---|
| fresh 7-10 | +5.80 / +5.80 / +6.44 | +5.91 / +6.16 / +6.78 | +0.47 / +0.01 / +0.64 | +1.22 / +1.48 / +0.47 | 1.22-1.29x |
| held out 3-6 | +5.40 / +5.72 / +6.24 | +5.54 / +6.04 / +6.58 | +0.46 / +0.06 / +0.48 | +1.15 / +1.00 / +0.42 | 1.23-1.31x |

- Every power and speed interval is clear of zero; **tower DEL is never negative** (one seed unresolved at
  +0.01 [-0.24, +0.25]). On the superseded baseline two of three seeds *lost* tower DEL - that is the row the
  manuscript used to carry and it has been replaced (s56).
- Absolute, against the shipped controller on fresh seeds: native baseline + gated layer +31.15 / +35.91 /
  +25.33 / +3.20 at 2.46x travel.
- On the scheduled MPC base: +2.41 power, +2.59 speed, +1.44 tower, +1.87 blade, 2.076x -> 2.899x travel.
- The gate is prior art (Kim arXiv:2609.21307, Abbas IET CTA 2026) and is attributed where introduced; our
  claim is the gating *variable* (an exogenous, physically interpretable regime variable) and the measurement.
- Below the gate both arms are **+0.00 at 12 m/s by construction** and no worse than -0.09 at 14 m/s.

## §7 The ladder (`../tables/ladder_published_fresh.csv`; s54)

Fresh seeds, 28 episodes, every rung joined on the wind file so all rows are the same episodes:

| rung | power | speed | tower | blade | energy | travel |
|---|---|---|---|---|---|---|
| native tuned baseline (s52) | +25.34 | +29.75 | +25.32 | +1.72 | +0.167 | 2.016x |
| scheduled offset-free LPV-MPC | +49.08 | +55.39 | +17.42 | -0.04 | +0.257 | 2.076x |
| MPC + wind-gated learned layer | +51.49 | +57.98 | +18.86 | +1.84 | +0.088 | 2.899x |
| native baseline + gated layer | +31.15 | +35.91 | +25.33 | +3.20 | +0.172 | 2.46x |

- **The second rung is a trade, not an addition.** Against the native baseline the MPC is **-7.91
  [-8.85, -6.97]** in tower DEL on fresh seeds (-7.43 [-8.37, -6.44] held out) while buying +23.7 power and
  +25.6 speed, at comparable duty (2.016x against 2.076x). The learned layer recovers +1.4 to +2.0 of it.
- **No single decomposition of the headline exists.** Of 58.0 points of speed MSE the full stack buys:
  tuning 51 %, model-based 44 %, learning 5 %; of 51.5 power MSE: 49 / 46 / 5 %; but of 18.9 tower DEL:
  tuning +25.3, model-based **-7.9**, learning +1.4. The old "49 / 45 / 6 %" was an average over terms that
  disagree in sign. Write the three rows and make that the point of the paragraph.
- The tuned PI loop beats **both** MPC rows on tower fatigue at every wind speed from 16 m/s up: +26.9 /
  +20.5 / +18.7 / +16.9 / +15.3 against the scheduled MPC's +17.7 / +6.5 / +5.4 / +3.4 / -0.3.
- The superseded (hand-scaled) baseline row, for the record only: +24.49 / +30.66 / +7.14 / +1.42 at 1.12x.
  Its tower term is why the model-based controller looked load-optimal; it is **not a tuning file** and no
  number of its is comparable (s52).

## §8 Discipline

| statement | number | source |
|---|---|---|
| Site weighting is not cosmetic (native baseline, fresh seeds, `lifetime_del.csv`, s61) | tower DEL reduction, per-episode mean / level ratio equal / Rayleigh V_ave 10 / 8.5 / 7.5: native baseline 25.3 / 32.3 / 37.6 / 38.6 / 39.2; agent one-line 8.7 / 13.4 / 16.6 / 17.1 / 17.3; scheduled MPC 17.4 / 27.8 / 40.4 / 43.8 / 46.1; baseline + layer (s1) 25.3 / 32.4 / 37.6 / 38.6 / 39.2; MPC + layer (s1) 18.9 / 29.4 / 40.6 / 43.7 / 46.0. Speed MSE: native 29.7 / 28.8 / 28.6 / 28.2 / 27.6; agent one-line 23.0 / 20.4 / 23.7 / 24.5 / 24.8; MPC 55.4 / 60.9 / 42.3 / 34.8 / 29.5; baseline + layer 35.9 / 37.3 / 32.4 / 30.4 / 28.9; MPC + layer 58.0 / 64.9 / 43.7 / 35.4 / 29.6. So the MPC's regulation advantage over the native baseline *shrinks* from +25.6 per episode to +1.9 at V_ave 7.5, and its tower disadvantage (-4.5 as an equal-weight level ratio) becomes +2.8 / +5.2 / +6.9 under the weights. No interval on a weighted aggregate. | `scripts/dev/lifetime_del.py --csv`, `figures_weighting.py` |
| One aggregation rule | bin mean, then weighted power mean; mixing arithmetic and power means moved the equally weighted tower reduction from 22.1 to 27.8 % | s42 |
| The fatigue conclusion does not depend on the Wohler exponent (native rows, fresh seeds) | tower m = 3 / 4 / 5, blade m = 8 / 10 / 12: native baseline 23.3 / 25.3 / 26.2, 1.6 / 1.7 / 1.7; agent one-line 7.9 / 8.7 / 8.9, 2.0 / 2.0 / 2.0; agent corner 6.5 14.3 / 16.2 / 17.2, 2.5 / 2.5 / 2.4; scheduled MPC 14.5 / 17.4 / 19.2, -0.2 / -0.0 / 0.0; baseline + layer s0/s1/s2 23.4, 23.1, 23.6 / 25.8, 25.3, 26.0 / 27.1, 26.4, 27.2; MPC + layer s1 16.2 / 18.9 / 20.3, 1.6 / 1.8 / 1.9. The ordering is the same in every column. | s48, s61; `../tables/wohler.csv`, `scripts/dev/wohler_sensitivity.py --csv` |
| Every curve is paired to the same rollout | the shipped level derived from each evaluation's own reduction (level / (1 - red/100)) agrees across the leg-1 and leg-2 evaluations to a relative 6e-16 on every wind file and term | `scripts/dev/figures_overlay.py` (F7) |
| A set-mean load constraint is gamed | four instances: with learning (s32) and without it (s53.2, s55, s59.6) | s32, s53.2, s55 |
| Screen on design, measure off it | with two episodes per wind the per-wind screen rejects the stage-1 tuning at -1.1 (blade, 22 m/s) although the same tuning reads -0.24 / -0.34 off the design winds, where there are four | s52.3 |
| Actuator duty | pitch-travel ratio and ADC against the 10 deg/s rate limit beside every row; the duty knee sits at about 3x shipped travel and holds held out (same level at 38-41 % of the actuation) | s22, s25-s26 |
| Paired statistics | stratified paired bootstrap over episodes for every difference between deterministic controllers | s27, `scripts/dev/paired_bootstrap.py` |
| Caveats | C1-C12 before quoting any number | `RESULTS_2026-09-10_data-and-caveats.md` |

## Provenance notes that must survive

- The wind bank is canonical and is never regenerated (caveat C1).
- The 600 s protocol lives in its own bank: any evaluation must export `WTRL_HOME=$HOME/wtrl600` and
  `WTRL_WIND=$HOME/wtrl/wind600` the way `scripts/wsl/campaign_rosco_native.sh` does, or it dies on the
  150 s fields before running an episode (s59.6).
- After any interruption run `scripts/dev/integrity_check.py --since <time>` before trusting an artefact.
