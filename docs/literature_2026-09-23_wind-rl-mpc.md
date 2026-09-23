# Literature recall, 2026-09-23 — RL pitch control, MPC, and estimators

Scope of this scan: recent (2024–2026) published work in three themes relevant to the TCST
submission — (1) RL pitch control, (2) MPC for wind turbines with attention to scheduled cost
*weights*, (3) disturbance observers / offset-free / adaptive MPC and effective-wind-speed
estimation. Companion to `docs/literature_2026-09-16.md`; that earlier file is not superseded.

## Verification discipline

Every entry carries a URL and one of:

- **VERIFIED** — the page was fetched and read; bibliographic fields and any quoted numbers come
  from that page, with the section named.
- **VERIFIED (bib only)** — the record was fetched from a repository or deposit rather than the
  publisher; bibliographic fields are as that record states and may carry repository-side errors.
  Numbers are quoted only where the record itself carried them.
- **UNVERIFIED** — seen in search results only, could not be opened. **No numbers are quoted from
  these.** Do not cite without opening them first.

Access note: ScienceDirect, Wiley Online Library, IEEE Xplore, MDPI and PubMed returned HTTP 403
throughout this scan, and several PDFs (the WES PDF mirror, the IOP PDF endpoint, elib.dlr.de)
returned unparseable binary. The verified set is therefore skewed toward Copernicus HTML,
IOPscience HTML, Zenodo, arXiv, institutional repositories and Springer reached via its cookie
redirect. Absence from the verified list below is an artefact of access, not of relevance.

Already-known works excluded by the brief (not re-reported here): Espinoza et al. ISA Trans 169:428
2026; Xie/Dong/Zhao IEEE TASE 2024; Moldenhauer & Schmid WES 10:1907 2025; Guo & Schlipf WES 8:1299
2023; de Lataillade TORQUE 2026; Hummel/Kober/Mulders WES 10:2005 2025; Abbas et al. WES 7:53 2022;
Soler ESWA 2024; Tang Front. Energy Res. 2023; Abbas/Jasa/Zalkind Applied Energy 2023.

---

## Headline finding — scheduled MPC cost weights

The construct this project calls a wind-scheduled tower weight (roadmap s33, `qt_sched`) **has
direct prior art, but from 2021, and it appears not to have been revisited since.**

### Wintermeyer-Kallen et al. 2021 — weight-scheduled LTV-MPC

**Wintermeyer-Kallen, T., Dickler, S., Zierath, J., Konrad, T., Abel, D. "Weight-scheduling for
linear time-variant model predictive wind turbine control toward field testing."
*Forschung im Ingenieurwesen* 85, 385–394 (2021). DOI 10.1007/s10010-021-00475-w**

<https://link.springer.com/article/10.1007/s10010-021-00475-w> — **VERIFIED**
(reachable only through the `?error=cookies_not_supported` redirect chain; the direct link 303s to
the Springer IdP)

The **Q and R weighting matrices themselves are scheduled**, not merely the prediction model or the
feedback gains. Q and R weight the control objectives (power, rotor speed, tower acceleration)
against actuator effort (pitch rate, generator torque), and are interpolated quadratically,

    q_i(y(k)) = q_i,0 + ((y_j − y_j,0)/(y_j,1 − y_j,0))² (q_i,1 − q_i,0)

with the page stating the schedule depends on **wind speed and rotor speed**. The stated purpose is
a smooth partial-load ↔ full-load transition, i.e. the same motivation as our region split.

- Plant: W2E-120/3.0fc, 3 MW, collective pitch, rated 12.5 m/s, Kankel, Germany.
- Baseline: the operator's production SISO PID controller.
- Numbers as rendered in Sect. 4.2: generator-torque DEL reduced "up to 50 %" above rated;
  **pitch-activity DEL increased approximately two-fold**; thrust-force DEL differences "less than
  5 %"; rotor-speed standard deviation lower for the MPC than the baseline controller.

**Consequences for the manuscript.**

1. The scheduled-weight MPC of s33 must cite this paper and state the delta explicitly. Ours
   schedules a *tower* weight on *wind speed alone* across 12–24 m/s and is evaluated under a
   per-wind non-negativity requirement; theirs schedules the full Q/R pair on wind and rotor speed
   for region blending. That is a real difference, but it is a difference, not a first.
2. Any novelty sentence must be phrased as "to our knowledge, scheduled cost weights have not been
   revisited since Wintermeyer-Kallen et al. (2021)" rather than as an absolute claim. Four
   separate query formulations for 2024–2026 scheduled / online-adapted MPC weights returned no
   newer instance; the 2025–2026 neighbours are *adaptive cost* work (Anand & Bottasso below), not
   scheduled weights.
3. Their "torque DEL down up to 50 %, pitch activity roughly doubled" is an external, published
   precedent for the J-versus-actuator-duty trade measured in s22/s26 (~7.6× GSPI travel at low
   turbulence; 1.7–2.2× over the class-B range; 2.14× for the s33 main-claim controller). It is
   useful when defending the knee-point argument against a reviewer who treats increased pitch
   travel as disqualifying.

### Same group, follow-ups

**Klein, A., Wintermeyer-Kallen, T., Zierath, J., Kluge, K., Abel, D., Vallery, H., Basler, M.
"Design and Practical Evaluation of Robust Model Predictive Wind Turbine Control." *Wind Energy*
28(5), e70011 (2025). DOI 10.1002/we.70011**

<https://repository.tudelft.nl/record/uuid:7745a21d-5ccd-490c-939d-0df20cf48e43> —
**VERIFIED (bib only)** — the publisher page returned 403; volume, issue and article number are as
the TU Delft repository record states and should be re-checked against the publisher before use.

Robust linear time-varying MPC combined with an **extended Kalman filter** for nonlinear state
estimation, field-tested on a 3 MW turbine in Northern Germany. The record states "3-h continuous
full access", wind speeds 4.76–13.06 m/s, across partial load, transition and full load. Baseline: a
reference feedback controller. Simulated power curves comparable to the reference controller; the
authors concede that "the tuning effort still remains complex".

This is a Theme 2 *and* Theme 3 paper — an estimator inside a field-validated MPC — and it is the
best available citation for "MPC is deployable but tuning-heavy", which is the opening this
project's design-search work (s38) addresses.

**UNVERIFIED**, same group: "Challenges of applying model-based predictive wind turbine control in
the field", *Forschung im Ingenieurwesen* (2023), DOI 10.1007/s10010-023-00634-1 —
<https://link.springer.com/article/10.1007/s10010-023-00634-1> — blocked at the Springer IdP after
two redirects. No numbers quoted.

---

## Theme 2 — MPC for wind turbines, 2025–2026

### Anand & Bottasso 2026 — adaptive economic NMPC (the closest competitor)

**Anand, A. and Bottasso, C. L. "Adaptive economic wind turbine control." *Wind Energy Science* 11,
1989–2008 (2026). DOI 10.5194/wes-11-1989-2026**

<https://wes.copernicus.org/articles/11/1989/2026/> — **VERIFIED**

- Formulation: economic **nonlinear** MPC (Sect. 3) maximising profit = power revenue − fatigue
  cost, with a "Parametric Online Rain Flow Counting" (PORFC) continuous surrogate standing in for
  the discontinuous fatigue cost.
- Mismatch repair: **offline neural-network model augmentation, not recursive estimation**
  (Sect. 2.2). The corrected dynamics are `ẋ(t) = F_ROM(x,u,d) + ΔF_ROM(x,u,d,p)` with `p` the
  network weights and biases; a feed-forward network, 20 hidden neurons, radial-basis activations,
  trained by Levenberg–Marquardt on 216,071 samples.
- Plant: **NREL 5 MW, 33 states, 15 DOF, in OpenFAST** (Sect. 4.1). ROM: 8 states, 3 DOF —
  drivetrain speed plus tower fore-aft and side-side — i.e. structurally very close to our 3-state
  LPV-MPC.
- Baseline: the identical ENMPC **without** the correction term (Sect. 4.3).
- Numbers: **+9 % profit** for the augmented controller (Sect. 4.3.1); **~20 % lower mean
  rotor-speed error** (Sect. 4.2, Table 2); pitch travel and torque travel both reduced by the
  adaptation (Sect. 4.3.1, Fig. 6); a LiDAR-preview scenario gives **+30 % profit** at 10 SQP
  iterations versus **+22 %** at 5 (Sect. 4.3.2, Fig. 8), the 5-iteration variant costing only 15 %
  more CPU (Sect. 4.3.3, Fig. 9); performance plateaus at 85 % of the training set (Sect. 4.2,
  Fig. 5).

**This is the paper the s19 / s36 offset-free and adaptive-MPC results must be positioned against.**
It makes the same claim — repair the reduced-order model's mismatch and the MPC wins — on the same
turbine and the same simulator, but reaches it with a *learned offline* correction rather than an
online offset/RLS estimator. Two defensible distinctions: their correction is trained once and does
not track drift, whereas ours is online and stays flat over Cp/Ct ±15 % (s19, s36); and their
objective is economic profit, not the four-term J with a per-wind load requirement. It also
*corroborates* s24 — once the model error is compensated, a residual has little of size left to
learn — which is worth saying plainly rather than hiding.

### Ribnitzky, Petrović & Kühn 2026 — experimental, and an external actuator-duty datapoint

**Ribnitzky, D., Petrović, V., Kühn, M. "Experimental investigation of wind turbine controllers for
the Hybrid-Lambda Rotor." *Wind Energy Science* 11, 469–491 (2026). DOI 10.5194/wes-11-469-2026**

<https://wes.copernicus.org/articles/11/469/2026/> — **VERIFIED**

Not MPC. Wind-tunnel experiments on MoWiTO 1.8 (1.8 m rotor, 3×3 m tunnel with an active grid)
comparing a feed-forward controller (upstream hot-wire preview), a load-feedback controller (strain
gauges) and their combination against a **model-based wind-speed-estimator plus pitch look-up-table
baseline**.

Numbers read from the page: the baseline **exceeded its load constraint by 9 %** because of model
uncertainty (Fig. 9, Sect. 3.1); the feed-forward controller cut gust peak-load overshoot to **4 %**
above steady state versus **25 %** for the baseline (Fig. 10, Sect. 3.2); the feed-forward
controller runs at **~3× the baseline pitch duty cycle** during gusts (Fig. 15, Sect. 3.5); the
load-feedback controller gives **5 % lower average power** under load-constrained operation
(Sect. 3.3); the feed-forward controller achieved the lowest DEL during gusts (Fig. 15, Sect. 3.5).

Two uses. First, an independent and *experimental* instance of "roughly 3× pitch travel buys the
load benefit", which supports the knee-point reading in s26 (same J at 38–41 % of the J-selected
point's actuation, ~3× GSPI travel). Second, the 9 % constraint violation caused by model error is a
clean external motivation for offset-free / adaptive correction — better than arguing it from our
own Cp/Ct sweeps alone.

### Bai et al. 2025 — IPC against the ROSCO IPC module

**Bai, D., Wang, B., Li, Y., Wang, W. "Study on load reduction and vibration control strategies for
semi-submersible offshore wind turbines." *Scientific Reports* 15, 1148 (2025).
DOI 10.1038/s41598-025-85476-3**

<https://www.nature.com/articles/s41598-025-85476-3> — **VERIFIED**

Equivalent-wind-speed-based IPC (EWIPC) versus NREL's reference open-source IPC (the ROSCO IPC
module), on the IEA 15 MW semi-submersible in OpenFAST, with collective pitch control as the
baseline. Metrics: 1P blade-root bending moment, blade flapwise displacement, tower-top fore-aft
deflection, platform surge and pitch, damage equivalent loads.

Results on the page are reported qualitatively: at 18 m/s steady wind both IPC methods "effectively
attenuate the 1P frequency"; EWIPC shows a "notable decrease in amplitude" for blade flapping and
tower deflection relative to ROSIPC. Under turbulent wind, **IPC effectiveness was "less pronounced"
than under steady conditions**, and EWIPC only "performs slightly better".

Relevant to `docs/RESULTS_2026-09-05_ipc-negative.md`: an independent statement that the IPC benefit
shrinks under realistic turbulence, which is the same direction as our negative result.

**Caution.** The introduction reviews MPC, H∞ and PI control from the literature, but the actual
simulation compares only EWIPC against ROSIPC. Several secondary sources misreport this as an
MPC-versus-H∞ benchmark. Do not cite it as one.

### Unverified Theme 2 leads

Search-result text only; no numbers quoted; open before citing.

- Abbasi et al., "Multiple model predictive control for offshore wind turbines operating in the
  full-load range", *Asian Journal of Control* (2025), DOI 10.1002/asjc.3549 —
  <https://onlinelibrary.wiley.com/doi/full/10.1002/asjc.3549> — Wiley 403.
- "An optimal model predictive control based on Hammerstein model considering fatigue load reduction
  for wind turbines", *Int. J. Electr. Power Energy Syst.* (2025) —
  <https://www.sciencedirect.com/science/article/pii/S0142061525006805> — ScienceDirect 403.
  Reported to use the NREL 5 MW and to compare Hammerstein MPC, conventional MPC and multiple MPC.
- Wang et al., "Wind Turbine Load Reduction Predictive Control Strategy Based on Dynamic Model and
  Data-Driven Approach", *IET Renewable Power Generation* (2026), DOI 10.1049/rpg2.70337 —
  <https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rpg2.70337> — Wiley 403.
- Dittmer et al. (DLR), WESC 2025, on optimising wind turbine load reduction —
  <https://elib.dlr.de/218870/> — PDF returned unparseable binary.

---

## Theme 3 — estimators, offset-free and adaptive MPC

### Pamososuryo, Spagnolo & Mulders 2025 — REWS estimator design and gain calibration

**Pamososuryo, A. K., Spagnolo, F., Mulders, S. P. "Analysis and calibration of optimal power
balance rotor-effective wind speed estimation schemes for large-scale wind turbines."
*Wind Energy Science* 10, 987–1006 (2025). DOI 10.5194/wes-10-987-2025**

<https://wes.copernicus.org/articles/10/987/2025/> — **VERIFIED**

Factorises rotor-effective-wind-speed estimation into two subcomponents and analyses all four
combinations:

- aerodynamic power estimator — filtered numerical derivative, versus a **Luenberger state
  estimator**;
- wind-speed solver — continuous (forward Euler, backward difference, Tustin discretisations),
  versus an iterative single-step Newton–Raphson variant.

Turbines: **NREL 5 MW** and IEA 22 MW, with an inertia-based framework spanning 5–25 MW. Metrics:
absolute mean and standard deviation of estimation error, phase lag, noise resilience, stability
regions, REWS tracking accuracy.

The mechanism worth importing is the **gain-tailoring rule** (Sect. 4.2.1): scale the estimator gain
with turbine inertia as `L₂ = h·J·ω₀²`, which holds the estimator's natural frequency and damping
constant across turbine size. A constant-gain strategy degrades across the 5–25 MW range while
tailored gains keep absolute error means stable (Sect. 4.2.2). The state-estimation power estimator
shows "less noise propagation and phase lag" than the numerical derivative (Sect. 4.3), and the
recommended pairing is the state estimator with the iterative solver (Sects. 5–6).

**Direct use:** this supplies a principled basis for the estimator time constant in
`controllers/mpc.py`, replacing the hand-picked τ = 5 s of s19 with a gain chosen from a stated
frequency/damping target. Same author group as the ROSCO and qLPV-MPC lines already cited.

### Cross-listed

- **Anand & Bottasso 2026** (above) is also the Theme 3 headline: it is the 2026 answer to "how do
  you remove plant–model mismatch in wind-turbine MPC", and it chose offline learning over online
  estimation. Frame the RLS / offset-free result against that choice.
- **Klein et al. 2025** (above) is the field-validated EKF-inside-MPC instance.

### Unverified Theme 3 leads

- **The most important gap.** "Offset-Free Stochastic MPC for Uncertain Wind Energy Conversion
  System", IEEE — <https://ieeexplore.ieee.org/document/10505032/> — IEEE 403. This is the single
  paper most likely to pre-empt the s19 offset-free result. **Open it before any novelty claim about
  offset-free MPC in wind energy goes into the manuscript.** No numbers quoted, and I could not
  confirm authors, year or venue.
- Lio, W. H. et al., "Real-time rotor effective wind speed estimation based on actuator disc theory:
  Design and full-scale experimental validation", *Wind Energy* (2023), DOI 10.1002/we.2858 —
  <https://onlinelibrary.wiley.com/doi/full/10.1002/we.2858> — Wiley 403. Appears to be the best
  "does it work on a real turbine" REWS citation available.
- arXiv:2104.07696, "The Immersion and Invariance Wind Speed Estimator Revisited and New Results"
  (2021) — <https://arxiv.org/abs/2104.07696> — background only if a convergence-proof reference is
  needed; not fetched in this scan.

---

## Theme 1 — RL pitch control, 2024–2026

Honest assessment: **the 2025–2026 RL-pitch literature I could actually open is thinner than the MPC
literature, and none of it is a collective-pitch residual on a ROSCO/GSPI base.** The strongest
CPC-versus-ROSCO item remains Espinoza et al. ISA Trans 2026, already on the known list — and note
that I could not open either its PubMed record or its ScienceDirect page in this scan, so its
headline figure is not independently confirmed here. The positioning of this project's contribution
is not threatened by anything found below.

### de Frutos et al. 2024 — DDQN torque-pitch under a multi-objective front

**de Frutos, M., Marino, O. A., Huergo, D., Ferrer, E. "Deep Reinforcement Learning for
Multi-Objective Optimization: Enhancing Wind Turbine Energy Generation while Mitigating Noise
Emissions." arXiv:2407.13320v1 (2024).**

<https://arxiv.org/html/2407.13320v1> — **VERIFIED**

- Algorithm: double DQN with an experience replay buffer and soft target-network updates.
- Turbine and simulator: Siemens SWT2.3-93 (2.3 MW, 93 m rotor), OpenFAST with a blade-element
  momentum solver plus the Brooks–Pope–Marcolini aeroacoustic model. Torque **and** pitch.
- Baseline: a standard variable-speed controller — torque below rated, pitch above rated — described
  in their Appendix B. **A plain baseline, not ROSCO.**
- Numbers: under a 45 dB SPL constraint at 100 m downwind, yearly extracted energy falls **22 %**
  (Sect. 3.3, Table 4: 2,722 MWh versus 3,458 MWh); power coefficients held in the range
  **0.26–0.30** across the Pareto front (Sect. 3.1).

Relevance: the cleanest recent example of RL pitch control evaluated as a *trade-off front* rather
than a single scalar — methodologically the same move as the J-versus-actuator-duty front of s26.

A journal version appears to exist as *Wind Energy* 28, e70041 (2025) — **UNVERIFIED**, seen only as
a PDF filename at oa.upm.es; do not cite that version without confirming it.

**Overlap warning.** arXiv:2402.11384 (Soler, Mariño, Huergo, de Frutos, Ferrer, "Reinforcement
learning to maximise wind turbine energy generation", <https://arxiv.org/abs/2402.11384>,
**VERIFIED** — double deep Q-learning with prioritised replay, a blade-element-momentum model, and a
**PID baseline**) is the preprint of the Soler ESWA 2024 paper already on the known list. Same group
as the entry above. Cite one or the other, not both as independent evidence.

### Nilsen et al. 2026 — RL needs a warm start from a model-based expert

**Nilsen, M. B., Quick, J., Göçmen, T., Dimitrov, N., Réthoré, P.-E. "Accelerating Reinforcement
Learning for Wind Farm Control via Expert Demonstrations." *J. Phys.: Conf. Ser.* 3224, 032016
(2026). DOI 10.1088/1742-6596/3224/3/032016** (TORQUE 2026)

<https://iopscience.iop.org/article/10.1088/1742-6596/3224/3/032016> — **VERIFIED**

Wind-farm yaw rather than turbine pitch, so out of this repository's control scope — cited only for
its *learning* premise, which is the one this project's residual architecture rests on. A soft
actor–critic agent is initialised by behaviour cloning on PyWake expert demonstrations, in the
WindGym environment.

Numbers as rendered: the untrained agent **underperforms the greedy zero-yaw baseline by ~12 %**;
pretraining on expert demonstrations "raises initial performance to near-baseline levels"; all
configurations converge within **250,000** environment steps, ultimately exceeding a look-up-table
controller that reaches **~7 %** power gain after 500,000 steps.

Use: a 2026 independent confirmation that a from-scratch RL wind controller *starts below* its
baseline, and that warm-starting from a model-based expert removes that phase. That is the argument
for residual-on-a-strong-base. It is also a useful contrast with s24 — their expert is weak enough
that RL retains headroom above it, whereas our compensated MPC base is not.

### Hummel, Kober & Mulders 2026 — newer companion to the known WES paper

**Hummel, J. I. S., Kober, J., Mulders, S. P. "Increasing the blade tip-to-tower clearance using
individual pitch control." *J. Phys.: Conf. Ser.* 3224, 052022 (2026).
DOI 10.1088/1742-6596/3224/5/052022** (TORQUE 2026)

<https://iopscience.iop.org/article/10.1088/1742-6596/3224/5/052022> — **VERIFIED**

This is the newer companion to Hummel/Kober/Mulders WES 10:2005 (2025) on the known list, and is
reported here as the update the brief asked for. Two IPC variants — "zero yaw" and "free yaw" —
adjust rotor tilt only when blade deflection exceeds threshold limits. IEA Wind 15 MW, design load
cases 1.1 and 1.5, simulated in WEIS.

Number from the abstract: both methods "improve the tower clearance by more than 5 m, even with a
slight positive effect on annual energy production". The zero-yaw variant reduces the extra blade
oscillation that clearance control normally induces; the free-yaw variant minimises control effort
by not regulating yaw deflection.

Relevant to the IPC negative result: here is an IPC objective — clearance, an extreme-load and
constraint quantity — where IPC clearly does pay. That sharpens rather than weakens our claim, which
is specifically that IPC does not pay **for the fatigue-DEL objective J**.

### Lara Ortiz et al. 2026 — a third external fatigue-per-actuation datapoint

**Lara Ortiz, M., Ruz, M. L., Vazquez, F., Garrido, J. "Feedforward individual pitch control of wind
turbines in the nominal region." *J. Phys.: Conf. Ser.* 3224, 052002 (2026).
DOI 10.1088/1742-6596/3224/5/052002** (TORQUE 2026)

<https://zenodo.org/records/20433042> — **VERIFIED (bib only)** — fetched as the Zenodo deposit of
the paper rather than the IOP page; the page range "1–12" is as the deposit states.

Adaptive feed-forward IPC running alongside collective pitch control, on a 15 MW monopile, in the
nominal region. Baseline: CPC alone. The deposit reports **15–20 % fatigue reduction at a 15–20 %
increase in control effort**, framed explicitly as the trade-off between reducing blade fatigue and
minimising pitch-actuator damage.

A near one-to-one fatigue-gain-per-actuation-cost ratio — a third external datapoint for the
duty-cycle front alongside Wintermeyer-Kallen 2021 and Ribnitzky 2026.

### Corredera et al. 2026 — ROSCO as the industrial reference

**Corredera, G., López, I., Vázquez, F., Garrido, J., Ruz, M. L. "ROSCO Control on PLC via
Software-in-the-Loop Architecture for Wind Turbines." *Jornadas de Automática* 47 (2026).
DOI 10.17979/ja-cea.2026.47.13854**

<https://zenodo.org/records/22339260> — **VERIFIED**

ROSCO translated into Structured Control Language and run on a virtualised Siemens industrial PLC,
co-simulated through PLCSim Advanced, Simulink and OpenFAST, under deterministic and turbulent wind,
with the IPC module's effect on structural fatigue quantified by damage equivalent load. Reported:
**a Variance Accounted For index of over 90 % in most cases** against the native implementation.

Minor but useful: a 2026 citation that ROSCO is the *industrial* reference and is being ported to
production PLCs. This is the answer to a reviewer who calls the ROSCO GSPI baseline an academic
strawman — and it strengthens the case for carrying the tuned-ROSCO row (s29, s31) in every table.

### Unverified Theme 1 leads

- The TORQUE 2026 RL-IPC paper appears in search results as *J. Phys.: Conf. Ser.* 3224, **072017**
  (2026). This is very likely the de Lataillade entry already on the known list, now with an article
  number. I could not open the IOP page, so treat 072017 as unconfirmed.
- A "Didier et al. (2026)" TRPO-based floating-offshore pitch controller was described in search
  result prose, but I could not find a fetchable URL and could not confirm that the paper exists.
  **Do not cite it.**

---

## Gaps and actions

1. **Scheduled cost weights**: no 2024–2026 instance found; closest prior art is 2021
   (Wintermeyer-Kallen et al.). Good for novelty. Phrase the claim as "not revisited since", cite
   the 2021 paper, and state the delta (tower weight on wind speed alone, under a per-wind
   non-negativity requirement).
2. **Residual RL in wind**: nothing verified. The only residual-RL-on-a-classical-controller paper I
   opened is from another domain — Abbas, Chasparis & Kelleher, arXiv:2310.14788,
   <https://arxiv.org/abs/2310.14788>, **VERIFIED to be the Tennessee Eastman chemical process, not
   wind**. The residual-on-MPC framing appears genuinely unoccupied in wind energy.
3. **Offset-free MPC in wind**: nothing verified for 2025–2026. IEEE document 10505032 (above) is the
   likeliest pre-emption of s19 and **must be opened before the offset-free novelty claim is made**.
4. **Access**: a re-run of this scan from a network with publisher access would most likely convert
   the four unverified Theme 2 leads and the two unverified Theme 3 leads. Nothing in the unverified
   set currently contradicts any finding in `CLAUDE.md`, but nothing in it has been checked either.
