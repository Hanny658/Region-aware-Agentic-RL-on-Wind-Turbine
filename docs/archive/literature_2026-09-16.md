# Literature scan, 2026-09-16 — where our numbers sit and what is still open

Four parallel scans (wind-turbine control; RL+MPC integration; LLM agents for control/optimisation; safe/constrained
tuning and noisy search). Every entry below was seen in a search result with the URL given. Items marked
**UNVERIFIED** could not be opened; no number from them may be quoted. Numbers in quotes were read from a fetched
page. This file records the state of the art we will be judged against, not a plan.

## 1. Calibration: our numbers against the published ones

Ours, for reference: the compensated LPV-MPC (offset-free or RLS-adaptive, as the base controller with a zero
residual) reaches, against the paired ROSCO GSPI on 600 s held-out episodes at 8 / 12.5 / 15 m/s and 8 % TI:
power MSE −22…−27 %, generator-speed MSE −33…−37 %, tower-base fore-aft DEL −21…−23 %, blade-root out-of-plane
DEL −3…−4 %.

| paper | baseline | fatigue | regulation / power | conditions |
|---|---|---|---|---|
| Espinoza et al., ISA Trans. 169:428, 2026 (DDQN pitch), https://pubmed.ncbi.nlm.nih.gov/41539907/ | **ROSCO** | tower FA scaled-STD −13.3 %, blade −1.1 % (**not DEL**) | text claims −15.7 % power regulation; **the printed table contradicts the text** | 1 turbulent + 1 sawtooth profile, 15 m/s, 5 % TI, **no seeds**, no DEL, no energy |
| Xie, Dong, Zhao, IEEE TASE 2024 (IDHP-PI), https://wrap.warwick.ac.uk/177502/ | plain PI | tower FA / platform roll "up to −11 %" vs uncontrolled | **power MSE −56.9 %** | FAST, OC3 spar, 600 s, 18 m/s |
| Moldenhauer & Schmid, WES 10:1907, 2025 (lidar), https://wes.copernicus.org/articles/10/1907/2025/ | **ROSCO** | **blade flapwise DEL −6.7 %** | speed tracking matched, **pitch rate −36 %** | OpenFAST, IEA 15 MW, 5–20 m/s |
| Guo & Schlipf, WES 8:1299, 2023, https://wes.copernicus.org/articles/8/1299/2023/ | ROSCO-tuned MVFB | tower FA **+19.7 yr** lifetime but blade OoP **−3.3 yr (worse)**; with lidar +24.3 / +3.8 yr | AEP +0.28 % | DLC 1.2/1.3, 10–24 m/s, **6 seeds**, 700 s (100 s discarded) |
| de Lataillade et al., TORQUE 2026 (RL-IPC), https://iopscience.iop.org/article/10.1088/1742-6596/3224/7/072017/pdf | CPC | blade DEL **−92.8 % uniform inflow, −11.1 % at 10 % TI** (PI-IPC: −78.2 % / −0.6 %) | — | 1 seed, 12 m/s |
| Hummel, Kober, Mulders, WES 10:2005, 2025, https://wes.copernicus.org/articles/10/2005/2025/ | full IPC | **50 % blade DEL at 16.4 % of full-IPC actuator effort** | — | IEA 15 MW, 15 m/s, **8 % TI** |
| Abbas et al., WES 7:53, 2022 (ROSCO), https://wes.copernicus.org/articles/7/53/2022/ | prior NREL controller | peak shaving: max thrust −10 % at **−1.7 % AEP**; FOWT feedback: max platform pitch −30 % | — | the baseline's own load-mitigation features |
| Soler et al., ESWA 249:123502, 2024, https://arxiv.org/pdf/2402.11384 | self-described limited PID | none (no structural model) | **+67.7 % yearly energy** | BEM only |
| Tang et al., Front. Energy Res. 2023 (BO-tuned robust MPC), https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2023.1306167/full | un-tuned robust MPC | **none reported** | **+1.3…+2.6 % power** | Simulink + FAST |
| Abbas, Jasa, Zalkind et al., Applied Energy 2023 (WEIS control co-design), https://www.osti.gov/servlets/purl/2222415 | un-optimised ROSCO | objective **is** tower-base DEL, constraint rotor overspeed ≤ 30 % | LCOE −1 % (tower) / −4.2 % (platform) | DLC 1.1 (2 seeds), 1.3 (3), 1.6 (3), 5–25 m/s |
| Wang, Dong, Zhao, IEEE TSTE 2026 (our baseline paper), DOI 10.1109/TSTE.2026.3712960 | — | **UNVERIFIED (contents not obtained; bibliographic record confirmed)** | — | — |

Reading: the large published regulation gains are all against weak baselines (a plain PI, a self-described limited
PID). The only RL-vs-ROSCO pitch paper reports a dispersion metric rather than MSE, on one wind profile with no
seed statistics, and is internally inconsistent. Generator-speed MSE against ROSCO is not reported by anyone.
Our tower-base DEL sits above the published band for a collective-pitch controller without lidar; our blade-root
DEL is modest but comes from the collective channel, so it is not comparable with IPC results.

## 2. The open gap (what no paper in the scan does)

- **No published method improves power and speed regulation against a tuned ROSCO while reducing both tower-base
  and blade-root DEL, using only collective pitch and standard turbine sensing.** Published fatigue reductions
  buy it with another actuator (IPC), with lidar, or with a regulation / energy penalty the authors report as a
  trade; Guo & Schlipf document explicitly that retuning for tower load *worsens* blade load.
- **Nobody quantifies what a disturbance observer recovers versus what a learned residual recovers on one plant.**
  The closest (Kalaria et al., https://arxiv.org/html/2410.06570v1) reports only an ordering: observer alone ≈
  learned residual alone, and only their combination is best. Our Cp/Ct ×0.95 study is that missing measurement.
- **No control paper ablates an LLM proposer against random search over the LLM's own generated vocabulary.**
  Eureka ablates the loop ("w/o Evolution"), not the vocabulary; the budget-matched HPO study
  (https://arxiv.org/abs/2606.21641) traces an LLM advisor's entire advantage to a default configuration.
- **No LLM has been placed in a wind-turbine control loop at any timescale**; every LLM-in-wind paper found is
  operations, condition monitoring or forecasting.

## 3. The regime boundary we must concede

- Lin, McPhee, Azad (https://arxiv.org/abs/1910.12047): with no model error a learned controller is *equivalent*
  to a long-horizon MPC; learning wins only when the modelling error is **large**. Our 5 % parametric error is
  the regime where learning should not be expected to win.
- Fu et al., Energy 273:127073, 2023 (https://www.osti.gov/pages/biblio/2331292): a 6-point optimality lead of
  DRL over MPC collapses to parity once realistic variability is admitted.
- Zhang et al. 2026 (https://arxiv.org/html/2605.26418): a *calibrated* rule-based controller beats six deep RL
  algorithms on cost across every workload; RL's only win is constraint compliance, at +24 % cost.
- Silver et al. (https://arxiv.org/abs/1812.06298) and Johannink et al. (https://arxiv.org/abs/1812.03201) scope
  residual RL to *imperfect* bases and to phenomena the analytic model structurally lacks (friction, contact);
  Policy Decorator (https://arxiv.org/abs/2412.13630) explains why a bounded residual on a strong base can only
  make small corrections.
- The baselines that beat our residual are established, guaranteed methods, not ad-hoc: offset-free MPC
  (Maeder & Morari, Automatica 2010; stability under plant-model mismatch proved by Kuntz & Rawlings,
  https://arxiv.org/html/2412.08104) and RLS-based adaptive tube MPC (Zhang & Shi, https://arxiv.org/abs/1901.03930).
  Son et al. (https://arxiv.org/abs/2012.02753) combine the estimator with learning rather than replacing it.

## 4. Methods the scan recommends for "raise regulation without degrading loads"

- **Constraints, not weights.** Safe BO with separate constraint GPs: SafeOpt-MC
  (https://arxiv.org/abs/1602.04450), GoOSE (https://arxiv.org/abs/1910.13726) as a wrapper that makes an
  existing proposer safe, RaGoOSE (https://arxiv.org/abs/2306.13479) for heteroscedastic noise. Domain precedent:
  safe BO of a floating-turbine IPC, "zero safety violations", tower fatigue −12.8 % with no power loss
  (https://www.sciopen.com/article/10.26599/OCEAN.2025.9470012).
- **Why our weight search plateaued**: linear scalarisation provably cannot reach non-convex parts of the Pareto
  front (https://link.springer.com/chapter/10.1007/978-3-540-89378-3_37, UNVERIFIED author list); the MORL
  practical guide (https://arxiv.org/abs/2103.09568) gives the utility-based framing.
- **If the learned part stays**: PID-Lagrangian constraint handling (https://arxiv.org/pdf/2007.03964) or a
  closed-form safety layer on the residual (https://arxiv.org/abs/1801.08757) instead of another reward weight;
  report the Safety-Gym triple (performance, constraint satisfaction, training-time violation).
- **For our noisy candidate ranking** (roadmap s20): race candidates over paired wind seeds with early
  statistical elimination (irace, https://www.sciencedirect.com/science/article/pii/S2214716015300270) under a
  Hyperband budget ladder (https://arxiv.org/abs/1603.06560); report IQM with stratified-bootstrap intervals
  (https://arxiv.org/abs/2108.13264); Cawley & Talbot (https://www.jmlr.org/papers/v11/cawley10a.html) is the
  citation for why the selector's own scores must never be the reported ones.

## 5. Evaluation protocol a control-journal reviewer will expect

From the best papers in the scan (Guo & Schlipf; WEIS control co-design; Hummel et al.; Moldenhauer & Schmid):

1. IEC 61400-1 power-production load cases (DLC 1.2 for fatigue, 1.3/1.6 for extremes), above-rated range in
   2 m/s steps, **600 s of usable data after a discarded transient** — our 600 s protocol now matches.
2. **At least 3, preferably 6 turbulent seeds per wind speed** (Guo & Schlipf use 6; WEIS uses 2–3).
3. DEL by rainflow counting with conventional Wöhler exponents (m ≈ 10 blade, 4–5 tower) — as we do.
4. **Report pitch rate / actuator duty cycle and energy change alongside every load claim.** Two of the most
   relevant papers do; we currently do not, and load gains bought with actuation are treated as suspect.
5. **The baseline must be ROSCO with its load-mitigation features enabled**, not a stripped GSPI. We already
   measured the tower-damper baseline (roadmap v2 §8, ≈ 0 on J); it belongs in the main table, not an appendix.
6. Seed-paired comparison with a stated test — our design satisfies the pairing; n = 3–5 will be challenged
   (Henderson et al., https://arxiv.org/abs/1709.06560; Colas et al., https://arxiv.org/abs/1806.08295).
