# Redesign of the proposer comparison: ROSCO-native quantities, measured bounds, constrained selection

2026-10-11. Supersedes the design of `scripts/wsl/campaign_rosco_search.sh` as first run on 2026-10-09.
Nothing here has been run yet. The campaign of 2026-10-09 is paused with 8 of its 12 runs finished; §7 says
what becomes of them.

The trigger was a question: are some of the searched inputs not ROSCO tuning quantities at all, and are the
box limits ours rather than the machine's? Both are true. This document audits every number the experiment
depends on, separates the ones a reader can check against the literature or the plant from the ones we
invented, and replaces the invented ones.

---

## 1 Two of the four searched inputs are not ROSCO tuning quantities

The search edited `PC_GS_KP` and `PC_GS_KI` — the gain-schedule tables in `DISCON.IN` — by multiplying each
table by a scalar (`kp_scale`, `ki_scale`). A ROSCO user does not set those tables. They are *generated* by
the toolbox from two quantities in the tuning file, `omega_pc` (the pitch loop's target natural frequency,
rad/s) and `zeta_pc` (its target damping ratio), through `rosco/toolbox/controller.py::second_order_PI`:

```
Kp(beta) = (2*zeta*omega + A(beta)) / B(beta)
Ki(beta) =  omega**2 / B(beta)
```

`A` and `B` are the linearised plant coefficients at that operating point. The shipped NREL 5 MW tuning is
`zeta_pc = 0.7`, `omega_pc = 0.6` rad/s (`ROSCO/Examples/Tune_Cases/NREL5MW.yaml`), so `A` and `B` can be
recovered exactly from the shipped tables — and they are smooth and physical:

| recovered from the 30 shipped schedule points | value |
|---|---|
| `B` = d(speed rate)/d(pitch) | −187.8 … −42.8, monotone in pitch angle |
| `A` = d(speed rate)/d(speed) | −0.4353 … +0.0475 |
| round trip: does (0.7, 0.6) reproduce the shipped tables? | max error 8.7e−19 |

Now the defect. Scaling `Ki` by *b* is exactly a change of `omega_pc` to `omega*sqrt(b)`. But scaling `Kp` by
*a* demands `2*zeta'*omega' = a*2*zeta*omega + (a-1)*A(beta)`, and `A` varies over the schedule, so the
implied damping ratio is **not constant**. A ROSCO tuning has one `zeta_pc`. The four winners of the paused
campaign imply these:

| searched multipliers | equivalent `omega_pc` | implied `zeta_pc` across the schedule | spread |
|---|---|---|---|
| kp 1.130, ki 1.020 (the tuned baseline in the manuscript) | 0.606 rad/s | 0.7365 … 0.7883 | 6.8 % |
| kp 1.150, ki 1.400 (`llm r1`) | 0.710 rad/s | 0.6344 … 0.6854 | 7.7 % |
| kp 1.270, ki 1.640 (`bayes r1`) | 0.768 rad/s | 0.6177 … 0.7025 | 12.8 % |
| kp 0.887, ki 1.210 (`local r1`) | 0.660 rad/s | 0.5604 … 0.6017 | 7.1 % |

So every design the search produced sits **outside the set of gain schedules ROSCO can generate**. It is not a
controller anyone could ship by editing a tuning file, and "we tuned ROSCO's own pitch loop" overstates what
was done. The two remaining inputs are genuine `DISCON.IN` fields: `F_LPFCornerFreq` and `FA_KI`.

**Fix.** Search `omega_pc` [rad/s] and `zeta_pc` [–] and build the tables with ROSCO's own formula, using the
`A`, `B` recovered above. The recovery makes this exact and needs no new dependency; it is self-checking,
because (0.7, 0.6) must return the shipped tables bit for bit. ROSCO also allows both quantities to be lists
over wind speed (`U_pc`); we keep them scalar, as the shipped tuning does, and note the schedule as a
follow-up rather than quietly using it.

---

## 2 The box limits were ours. Where they bind, they decided the answer

Every bound below is now read off the plant or the controller, measured on the 70 above-rated 600 s baselines
of the shipped controller (`scripts` in §6 reproduce them).

Measured plant and signal facts:

| quantity | value | how |
|---|---|---|
| first tower fore-aft mode | 0.3241 Hz = **2.037 rad/s** | peak of the tower-base fore-aft moment spectrum |
| rotor speed above rated | 12.09 rpm → 3P = 0.604 Hz = 3.797 rad/s | mean over the baselines |
| generator-speed signal content | 99.99 % of its power below 3.05 Hz = **19.2 rad/s** | cumulative spectrum |
| nacelle fore-aft acceleration | s.d. 0.216 m/s², worst episode 0.326 | `NcIMUTAxs` |
| max abs integral of its high-passed value | **1.076 m/s** median episode, 2.885 worst | ROSCO's own first-order high pass at 0.172 rad/s |
| controller step `DLL_DT` | 0.01 s → Nyquist 314 rad/s | ServoDyn |
| shipped pitch travel | 0.464 °/s mean abs rate = 4.6 % of the 10 °/s limit | baselines |

### The old box against the new one

| searched quantity | old box | new box | reason for the new limit |
|---|---|---|---|
| `omega_pc` [rad/s] | — (implied 0.379 … 1.039 via `ki_scale`) | **0.10 … 2.04** | the upper limit is the measured first tower fore-aft frequency: a collective-pitch loop whose bandwidth reaches that mode drives thrust straight into it. The lower limit is a 10 s loop time constant, below which the loop is not regulating. |
| `zeta_pc` [–] | — (not representable, §1) | **0.10 … 2.00** | from nearly undamped to heavily overdamped; shipped 0.7. |
| `F_LPFCornerFreq` [rad/s] | 0.6 … 4.0 | **0.20 … 19.2** | above 19.2 rad/s a first-order filter passes 99.99 % of the measured generator-speed power and is indistinguishable from no filter, so 19.2 rad/s is the physical ceiling, not an opinion. The old ceiling of 4.0 rad/s sat just above 3P — **and both `llm` repeats put their winner exactly on it**, so that bound, not the plant, set the result. |
| `FA_KI` [rad·s/m] | 0 … 0.004 | **0 … 0.20** | at the old ceiling the damper's unsaturated pitch demand reaches 0.004 × 1.076 = 0.0043 rad = **0.25°**, so the damper was searched over a range in which it can do essentially nothing, and its saturation limit was never reachable. At 0.20 the demand reaches 12.3° on a median above-rated episode, comparable to the whole above-rated pitch range, beyond which it is no longer a trim. **This invalidates the s23 conclusion that ROSCO's tower damper is worth about nothing** — that was measured inside a box 50× too narrow. |
| `FA_HPFCornerFreq` [rad/s] | frozen at 0.172 | **searched, 0.05 … 1.20** | the damper must reject quasi-static thrust and pass the tower mode. At 1.20 rad/s the mode still passes within 31° of phase (`arctan(w_c/w_twr)`); below 0.05 rad/s the high pass no longer blocks the slow turbulent thrust. The frozen 0.172 came from our own `template_5mw_td`, not from ROSCO, whose own default for all three damper fields is 0. |
| `FA_IntSat` [rad] | frozen at 0.0873 (5°) | **fixed at 0.175 (10°), stated, and reported whenever it binds** | a safety limiter, not a tuning knob: `FA_KI` is the authority knob. Fixing it generously and reporting how often it acts is honest; freezing it at a value that silently caps the damper is not. |

At the far corner of the new box (`zeta_pc` 2.0, `omega_pc` 2.04) the gains reach 19× the shipped `Kp` and
11× the shipped `Ki`, against 3× in the old box. Candidates there will be unstable. That is the point: an
infeasible region inside the box is evidence the box is wide enough.

Sampling: log-uniform for the three quantities that span decades and are positive-definite (`omega_pc`,
`F_LPFCornerFreq` and `FA_KI`, the last with the usual small shift so that 0 is reachable), uniform for the two
of order one (`zeta_pc`, `FA_HPFCornerFreq`). Stated once, identical for every arm.

---

## 3 Audit of every other number the experiment reports

| quantity | ours or recognised? | verdict |
|---|---|---|
| power-regulation error `mean(((P - 5 MW)/5 MW)^2)` on above-rated samples | recognised (normalised regulation error against rated) | keep |
| generator-speed error `mean(((w - w_rated)/w_rated)^2)`, same samples | recognised | keep |
| tower-base and blade-root damage-equivalent load, Wöhler m = 4 and m = 10 | recognised (steel / composite) | keep |
| reference cycle count in the DEL, `neq_per_s = 1` | ours | **immaterial, and we will say so**: both controllers run equal episode lengths, so the ratio of two DELs is `(sum r1^m / sum r2^m)^(1/m)` and the reference count cancels exactly |
| above-rated sample subset labelled by true hub wind on **both** sides | ours | keep and state: the label is identical for the two controllers on a given wind file, so they are compared on the same samples. **Implementation hazard**: `evaluate.py` only enables it automatically when the borrowed config run carries `objective: J`. Once we stop using that objective, `--relabel_wind` must be passed explicitly, or the subset silently reverts to a per-controller pitch rule and the comparison stops being paired. |
| episode counted only if at least half its samples are above rated | ours | keep, state, and report how many episodes it removes at 12 m/s |
| composite score = mean of the four reductions, minus 20 points per % of energy loss above 1 % | **ours** | **removed from everything reported** (§4) |
| per-episode clip of a reduction at ±100 points | ours | kept only as a numerical guard, and it is now verifiable that it never touches a reported number: of 141 evaluated candidates, 8 reached the clip, and every one of them lost *every* term, so all 8 are infeasible under §4 |
| 20 points per point of constraint violation | ours | **removed** — replaced by a feasibility rule with no weights (§4) |
| energy tolerance of 1 % | ours | **removed** — energy becomes a constraint at "no worse than shipped" |
| per-wind tolerance of 1 point | ours | **removed** from selection; the constraint is applied where the resolution supports it (§4) and the per-wind values are reported |
| "simulations until a candidate above 10" in the result table | ours, arbitrary | **removed** — replaced by the whole best-so-far trace and by "simulations to the first feasible candidate that improves on shipped", whose threshold is zero |
| parameters rounded to 3 significant figures before being written | ours, and it does damage | **removed**: it is why the identity check reads −0.15 instead of 0 (the shipped corner 1.5708 was written as 1.57). ROSCO's own writer uses 5 decimals; so will we, and the identity check then has to return exactly 0, which makes every run self-validating |
| pitch travel ratio against the shipped controller; actuator duty cycle against the 10 °/s rate limit | plain ratio / recognised | keep |
| wind protocol: 12–24 m/s in 2 m/s steps, IEC class B, Kaimal, power law, 600 s with the first 20 s discarded | IEC 61400-1 ed. 3 plus a stated protocol | keep |
| design seeds 1–2, held-out 3–6, fresh 7–10 | protocol | keep, see §4 on resolution |

---

## 4 Selection: one objective, physical constraints, no weights

The paused campaign showed why this matters. Its top-scoring winners bought regulation with tower fatigue —
`bayes r1` at −11.4 points of tower-base DEL, `random r1` at −20.3 — because a plain mean of four reductions
will happily trade one for the others. Only 4 of the 9 winners kept all four terms within a point of the
shipped controller. The aggregation chose the controller.

So the search gets the standard engineering statement instead:

**Objective** — the generator-speed regulation error reduction against the paired shipped controller. One
quantity; the one the pitch loop's PI acts on, and one of the baseline paper's four. Power regulation is
reported but not optimised: above rated at constant torque the two carry much the same information (we
measured a correlation of 0.94–0.95 on the model-based rows), so optimising both double-counts it.

**Constraints**, all against the paired shipped controller, all at "no worse than shipped", i.e. a reduction
of at least zero:

1. tower-base damage-equivalent load,
2. blade-root damage-equivalent load,
3. energy capture.

Applied on the 14-episode design set, whose resolution is about a point; the per-wind-speed values are
reported for every candidate and the per-wind check is applied in the verification stage, on the 28 held-out
episodes, where the resolution supports it. This is deliberately the honest order: we do not pretend that two
episodes per wind speed can enforce a per-wind constraint.

**Ranking rule** — constraint domination (Deb 2000, the rule NSGA-II uses), which needs no weights:

* a feasible candidate beats any infeasible candidate;
* between two feasible candidates, the larger speed-regulation reduction wins;
* between two infeasible candidates, the smaller total shortfall (summed over the violated constraints, in
  points) wins.

With this rule the ±100 clip, the 20-point penalty, the 1 % energy tolerance and the composite all leave the
experiment, and every reported number is one of the four published reductions, the energy, the travel ratio or
the duty cycle.

**What gets reported per arm and repeat**: the four reductions, energy, travel ratio and duty cycle of the
selected design; whether it is feasible; the number of simulations to the first feasible improvement; and the
whole best-so-far trace against simulation count, so no threshold is needed to read how fast an arm got
there. Then the verification pass on held-out seeds 3–6 and fresh seeds 7–10, with the design-to-held-out gap
and the rank correlation between design and held-out ordering — the check that failed for the MPC design
search at s38.

---

## 5 Arms and budget

Four budget-matched proposers over the same box from the same shipped start, three repeats each, because one
run cannot separate a proposer from a lucky draw:

| arm | method |
|---|---|
| `random` | independent draws from the box under the §2 sampling rule |
| `local` | one random point, then a (1 + λ) local search around the incumbent |
| `bayes` | Gaussian-process surrogate with expected improvement |
| `llm` | a reasoning model shown the box, the five quantities with their units and physical meaning, the objective and constraints, and the whole history |

The `llm` arm's prompt is rewritten for the new space: it now receives what a human tuner receives — a loop
bandwidth, a damping ratio, a filter corner, a damper gain and its high-pass corner — instead of opaque table
multipliers. That makes the arm a more meaningful test, not an easier one.

**Budget: 24 candidates per run**, up from 16, because the space is five-dimensional and 16 draws in five
dimensions is too sparse for random search to be a fair control. Each run also evaluates the shipped settings
free of budget as the identity check of §3.

Cost, from the measured 4.6 min per candidate at 7 workers (14 episodes of 600 s, two waves):

| stage | cost |
|---|---|
| 4 arms × 3 repeats × 24 candidates | 288 evaluations ≈ 110 min per run ≈ **22 h** |
| verification of 12 winners on held-out and fresh (28 + 28 episodes each) | ≈ **5 h** |
| total | **≈ 27 h**, pausable at the candidate as before |

A 16-candidate variant costs ≈ 20 h in total if the longer run is unwelcome; the trace reporting means a
smaller budget shortens the comparison rather than invalidating it.

---

## 6 Implementation changes

1. `scripts/rosco_native.py` (new): recover `A`, `B` from the shipped tables, write `DISCON.IN` from
   (`omega_pc`, `zeta_pc`, `F_LPFCornerFreq`, `FA_KI`, `FA_HPFCornerFreq`) with ROSCO's formula and ROSCO's
   own 5-decimal format, `TD_Mode = 1` and `FA_IntSat = 0.175` whenever `FA_KI > 0`. Self-test: (0.7, 0.6)
   reproduces the shipped tables exactly, and the identity evaluation must return exactly 0.
2. `eval/` : a selection rule implementing §4 — one objective, three constraints at zero, constraint
   domination. No new weights, no new tolerances.
3. `scripts/rosco_search.py` : new box and sampling, no parameter rounding, the rewritten `llm` prompt, and
   `--relabel_wind` passed explicitly to `evaluate.py`.
4. `scripts/dev/rosco_search_table.py` : report the four reductions, energy, travel and duty; feasibility;
   simulations to first feasible improvement; the best-so-far trace; and the verification block. Remove the
   arbitrary threshold column.
5. A measurement script for the §2 bounds, so the box is reproducible rather than asserted.
6. `scripts/wsl/campaign_rosco_search.sh` : budget 24, new root, same pause/resume contract.

---

## 7 What happens to the paused campaign

Its 8 finished runs stay on disk and keep one finding, which does not depend on the box or the parameterisation
and which motivates §4: **an unconstrained mean of four reductions makes a parameter search buy regulation with
tower fatigue** — 5 of 9 winners lost a load term, two of them by more than 11 points. That is reported as the
motivation for the constrained formulation.

Everything else from those runs is withdrawn. The designs are not ROSCO-reachable (§1), the filter corner
winners sat on an invented ceiling (§2), and the damper was searched inside a range 50× too narrow (§2). No
proposer ranking from that campaign is carried forward.

Two manuscript consequences follow and must be made before the next version:

* the tuned baseline (kp ×1.13, ki ×1.02, corner 3.42 rad/s, damper 0.00102) is **not a ROSCO tuning**; it is
  a hand-scaled gain table whose implied damping ratio varies by 6.8 % across the schedule. Either it is
  re-derived in the native parameterisation and re-measured, or it is described for what it is. The honest
  move is to re-derive it: it is the baseline the entire paper measures against.
* the claim that ROSCO's tower damper is worth about nothing on this protocol is unsupported, because the
  gain never exceeded 0.25° of pitch authority.
