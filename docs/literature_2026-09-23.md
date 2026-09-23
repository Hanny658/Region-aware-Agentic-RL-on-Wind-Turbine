# Literature recall, 2026-09-23 — novelty audit and related-works foundation (IEEE TCST)

Extends `docs/literature_2026-09-16.md`; entries already in that file are not repeated unless a newer or
better version was found. Target venue: IEEE TCST special issue on data-driven and learning-based control.

**Verification legend.** Every entry carries a URL and one of:

- `VERIFIED` — I fetched the URL in this session and read the content quoted.
- `VERIFIED (bib only)` — bibliographic record fetched from a metadata API (Semantic Scholar / OpenAlex);
  publisher page blocked. Title, authors, journal, volume, pages, DOI are trustworthy; **no page content**.
- `VERIFIED (delegated)` — a delegated scan fetched the page and reported section-level detail; I did not
  reproduce the fetch. Numbers are attributed but are second-hand — **re-check before they enter the paper**.
- `UNVERIFIED` — seen in a search result only. **No number from it may be quoted anywhere.**

Publisher blocks encountered throughout: ScienceDirect, Wiley Online Library, IEEE Xplore, MDPI and PubMed
return HTTP 403; Springer requires the `?error=cookies_not_supported` redirect and is intermittent.
Copernicus (Wind Energy Science), arXiv, IOPscience, Zenodo and the OpenAlex/Semantic Scholar APIs fetch cleanly.

---

## 1. Novelty assessment, claim by claim

### A(i) — Wind-speed-scheduled / gain-scheduled MPC **cost weights** in wind turbine MPC

**Verdict: WEAKENED.** The mechanism is published prior art; our specific instantiation is not.

The undercutting paper:

> **Wintermeyer-Kallen, T.; Dickler, S.; Zierath, J.; Konrad, T.; Abel, D.**
> *Weight-scheduling for linear time-variant model predictive wind turbine control toward field testing.*
> Forschung im Ingenieurwesen **85**, 385–394 (2021). DOI 10.1007/s10010-021-00475-w
> <https://link.springer.com/article/10.1007/s10010-021-00475-w> — **VERIFIED** (reachable only through the
> `?error=cookies_not_supported` redirect chain).

What I read on the fetched page: weight-scheduling here means exactly that the **MPC cost-function weighting
matrices Q and R are functions of the plant's actual output y, varying with wind speed and rotor speed**.
Plant: 3 MW W2E-120/3.0fc prototype, rated 12.5 m/s; software-in-the-loop against a FLEX5 multibody model;
reported outcome is drive-train fatigue reduced in the full-load region with increased pitch activity, and
improved power/speed tracking. `VERIFIED (delegated)` for the finer detail: the schedule is a quadratic
interpolation `q_i(y) = q_i,0 + ((y_j − y_j,0)/(y_j,1 − y_j,0))^2 (q_i,1 − q_i,0)`, the scheduled terms weight
power, rotor speed and tower acceleration against pitch-rate and torque effort, and Sect. 4.2 reports
generator-torque DEL down "up to 50 %" with pitch-activity DEL roughly doubled and thrust-force DEL
differences under 5 %. **Do not put those numbers in the manuscript until someone re-reads Sect. 4.2 directly.**

What still survives for us, and how to phrase it:

- Their schedule exists to smooth the **partial-load ↔ full-load transition**; ours schedules a **tower-damping
  weight inside the full-load region** (low below 18 m/s, higher above 22 m/s) to trade regulation against
  tower fatigue where the trade actually bites. Roadmap s31 is the evidence that a *constant* tower weight
  cannot do this (qt = 6 makes the high-wind tower term non-negative at −2.2 J and +20 % travel, and larger
  weights destabilise 12 m/s).
- No 2024–2026 instance of scheduled MPC cost weights in wind turbines was found across four query
  formulations. Two 2025–2026 near-neighbours are *adaptive cost*, not *scheduled weights*: Anand & Bottasso
  (WES 2026, §2) keep the revenue-rate weight `w_p` fixed — I verified this on the article page — and Klein
  et al. (Wind Energy 2025) report only that "the tuning effort still remains complex".
- **Required manuscript wording:** "to our knowledge, scheduled MPC cost weights for wind turbines have not
  been revisited since Wintermeyer-Kallen et al. (2021), and have not been applied to the tower-fatigue
  weight within the full-load region" — never "no prior work schedules MPC weights".

A secondary point that *helps* us: ROSCO's own load-mitigation gain is **not** scheduled.
From Abbas et al., WES 7:53 (2022), <https://wes.copernicus.org/articles/7/53/2022/> — **VERIFIED**:
the floating-feedback gain is "defined analytically as k_β_float = Γ_v / Γ_β" (Eq. 62, Sect. 5.5), computed
once from aerodynamic derivatives at nominal conditions, not varied across operating regimes. The pitch PI
gains *are* scheduled, but on low-pass-filtered **pitch angle**, not wind speed (Sect. 4).

---

### A(ii) — Residual RL with an action gate / regime-switched activation, in any domain

**Verdict: DEAD as a mechanism claim. SURVIVES only as a domain-and-gating-variable claim.**

Four papers occupy this space; two of them are close enough that a reviewer who knows the area will name them.

1. **Abbas, A. N.; Chasparis, G. C.; Kelleher, J. D.** *Specialized Deep Residual Policy Reinforcement
   Learning Framework for Safe and Adaptive Continuous Control.* IET Control Theory & Applications **20**(1)
   (2026). DOI 10.1049/cth2.70099. <https://api.openalex.org/works/doi:10.1049/cth2.70099> —
   **VERIFIED (bib only)**; preprint **VERIFIED** at <https://arxiv.org/abs/2310.14788> (arXiv:2310.14788,
   15 Oct 2023). From the preprint abstract, quoted verbatim: "Residual policy learning allows learning a
   hybrid control architecture where the reinforcement learning agent acts in synchronous collaboration with
   the conventional controller", and "The specialization through the input-output hidden Markov model helps
   to optimize policy that lies within the region of interest (such as abnormality) … where the reinforcement
   learning agent is required and is activated." Validated on Tennessee Eastman process control.
   **This is residual RL + a learned regime gate, published.**

2. **Kim, J.; Kwon, J.; Kim, H. S.; Seo, T.; Seo, H.-T.** *Stability-aware Residual Reinforcement Learning
   Framework for Robotic Manipulator Disturbance Compensation.* arXiv:2609.21307 (18 Sep 2026).
   <https://arxiv.org/abs/2609.21307> — **VERIFIED**. Abstract, verbatim: "This study proposes a residual
   reinforcement learning DOB framework that pairs an analytical observer with an RL policy. **The
   deterministic baseline operates within a reliable region, whereas the RL policy explicitly targets the
   residuals that the model cannot capture.** … we derived and enforced a **state-dependent action bound** on
   the RL policy from an input-to-state stability (ISS) analysis such that the closed loop provably confines
   the tracking error to a certified envelope for arbitrary policy outputs." Domain: 6-DOF robotic manipulator
   with hardware validation. `VERIFIED (delegated)` for the body: the base is NMPC + disturbance observer
   (τ_cmd = τ_MPC − d̂_filt, §III-C), and the bound ρ_k = (c_x/γ_0)(‖x_k‖² − r²)₊ vanishes inside a deadband
   radius r, i.e. the residual is suppressed there (§V-B).
   **This is the single most dangerous paper for us**: residual RL, on a model-based base *with a disturbance
   observer*, with a state-dependent gate, and the same "base handles the reliable region, residual handles
   what the model misses" framing we use. Published one week before this scan.

3. **Luque-Cerpa, A.; Wang, …; Carlsson, …; Seshia, S. A.; Dubhashi, D.; Torfah, H.** *Learning Contextual
   Runtime Monitors for Safe AI-Based Autonomy.* arXiv:2601.20666 (28 Jan 2026, rev. 2 Apr 2026).
   <https://arxiv.org/abs/2601.20666> — **VERIFIED (delegated)**. A contextual-bandit monitor observes the
   system's context and **selects the controller best suited to current conditions**, with safety guarantees
   during selection. Autonomous driving. Author list beyond the first two names is **uncertain** — check it.

4. **Saxena, …; Scheinker, A.** *Mahalanobis-Guided Latent OOD Detection for Hybrid ES-DRL Control in
   Time-Varying Systems.* arXiv:2606.11474 (9 Jun 2026). <https://arxiv.org/abs/2606.11474> —
   **VERIFIED (delegated)**. Binary switch between an RL controller and an extremum-seeking fallback, driven
   by a latent-space OOD detector. Particle-accelerator control.

Also in the neighbourhood, related but different:

- **Cao, H.; Zhao, L.; Gu, Y.; Hovakimyan, N.; Sha, L.; Caccamo, M.** *Safe Online Learning via Smooth
  Safety-Structured Policy Composition* (AutoSafe). arXiv:2606.31320 (30 Jun 2026).
  <https://arxiv.org/abs/2606.31320> — **VERIFIED**. Abstract: "smooth, **risk-dependent** transitions between
  performance-driven and safety-preserving behaviors". A *safety* gate, not an operating-region gate.
- **Jeon, …; Lee, H. J.; Hong, …; Kim, S.** *Residual MPC: Blending Reinforcement Learning with
  GPU-Parallelized Model Predictive Control.* arXiv:2510.12717 (14 Oct 2025).
  <https://arxiv.org/abs/2510.12717> — **VERIFIED (delegated)**. Residual RL on an MPC base, legged
  locomotion, **no activation logic** — the cleanest "residual-on-MPC without a gate" citation.

**What survives.** Nothing about "gate the residual" is new as a mechanism. Our defensible deltas, in order
of strength: (a) the gating variable is an **exogenous, physically interpretable regime variable** (wind
speed) rather than a learned abnormality model (Abbas), a tracking-error deadband (Kim) or a risk estimate
(Cao); (b) the objective is a four-term regulation/fatigue metric rather than tracking error; (c) first
instance in wind-turbine control — the delegated scan found no verifiable residual RL on a model-based
controller anywhere in the wind literature, so that sub-claim holds.
**Recommended framing: "a design choice validated in a new domain", never "a new mechanism".**

---

### A(iii) — LLM-written reward functions for control, and any LLM in a physical control loop

**Verdict: WEAKENED.** The generic claim is gone; the *shape-not-weights* claim is still standing and is now
the only thing carrying the contribution.

The nearest analogue to our supervisor:

> **Wu, Z.; Fu, X.; Li, S.; Peng, J.-Q.; Li, H.-Q.; Li, H.-P.** *A hierarchical control framework for
> photovoltaic-driven air conditioning systems: **LLM-based reward weight adaptation** and TD3 real-time
> control.* Building Simulation (2026). DOI 10.1007/s12273-026-1467-3 —
> <https://api.semanticscholar.org/graph/v1/paper/DOI:10.1007/s12273-026-1467-3> — **VERIFIED (bib only)**;
> the publisher elides the abstract and I could not reproduce a content fetch.

The **title alone** establishes the damaging fact: an LLM adapting RL reward weights, in a slow supervisory
loop, over a real energy control system, published 2026. A delegated fetch reported comfort/cost/carbon
numbers and 18 % / 14.2 % / 35.88 % improvements over fixed-weight baselines; **I could not reproduce that
fetch and those numbers must not be used until someone reads the paper.**

Our discriminator against it is precisely the axis the repo already measured: they tune **weights**, we have
the LLM write the **reward expression**. Roadmap evidence — llm_reward is the only RL arm positive on all
four terms, via a saturated speed term and centred bounded load terms, "a change of reward shape the weight
search cannot reach" (CLAUDE.md, final state). **This sentence must become load-bearing in the manuscript,
with Wu et al. cited immediately next to it, or a reviewer will make the comparison for us.**

The rest of the now-existing literature, all of it outside wind:

- **Cardenoso, F.; Caarls, W.** *Leveraging LLMs for reward function design in reinforcement learning
  control tasks* (LEARN-Opt). arXiv:2511.19355 (24 Nov 2025). <https://arxiv.org/abs/2511.19355> —
  **VERIFIED**. LLM generates, executes and evaluates reward-function candidates, deriving its own
  performance metrics from a textual system description. Abstract, verbatim and directly useful to us:
  "We find that **automated reward design is a high-variance problem, where the average-case candidate fails,
  requiring a multi-run approach to find the best candidates.**" That is the same phenomenon as our
  "3/5 balanced, +0.4 / +0.5 J" result, and it is the citation that makes our n-run reporting the correct
  practice rather than a weakness.
- **Casas, W. P.; da Fonseca, N. L. S.; Astudillo, C. A.** *LLM-Driven Automated Reward Design for
  Reinforcement Learning-Based Routing in LEO Satellite Networks* (LARGE). arXiv:2608.01649 (3 Aug 2026).
  <https://arxiv.org/abs/2608.01649> — **VERIFIED**. LLM-generated reward refined by **iterative
  simulator-in-the-loop evaluation** — structurally our loop, in an engineering domain. Their honest
  outcome is parity: best configuration within ~3 % of the expert-designed baseline on goodput.
- **Gao, …; Zhang, X.; Jiang, …; You, …; Zhang, M.; Deng, …** *RF-Agent: Automated Reward Function Design
  via Language Agent Tree Search.* arXiv:2602.23876 (27 Feb 2026). <https://arxiv.org/abs/2602.23876> —
  **VERIFIED (delegated)**. MCTS over LLM-generated reward code, evaluated on 17 low-level control tasks.
  This supersedes Eureka/Text2Reward as the "current state of the art" citation.
- **Rasheed, A.; Ravik, O.; San, O.** *Large Language Models for Control.* arXiv:2511.00337 (1 Nov 2025).
  <https://arxiv.org/abs/2511.00337> — **VERIFIED**. LLM generating **control actions directly**, in three
  variants (prompt-only, tool-assisted with history, prediction-assisted). This is an LLM *inside* the fast
  loop — the opposite design choice to ours, and the right paper to cite when we justify keeping the LLM at
  a slow supervisory timescale.
- **Chen, J.; Shu, Y.; Li, H.-N.** *Benchmarking Performance Judgment in Open-Weight LLM Controller Tuning:
  Control Knowledge Does Not Ensure Reliable Gain Ranking.* Processes **14**(18), 2954 (2026).
  DOI 10.3390/pr14182954 — <https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/pr14182954> —
  **VERIFIED** (full abstract read from the API). Declarative control-theory accuracy 98–100 % pooled by
  plant, but accuracy at ranking PI gain sets is **46.8 %, 31.8 % and 53.2 %**; adding equations and
  14-point response trajectories raises pooled accuracy only from 46.8 % to 51.2 % / 51.0 %; **"All 12
  prespecified trajectory-versus-qualitative intervals include zero, and none survives Holm correction"**;
  a numerical scaffold raises evidence-consistent accuracy from 51.1 % to 88.1 %. Closing sentence:
  **"Simulator-based validation remains necessary before LLM judgments are used for controller tuning."**
  This is the strongest available justification for our deterministic-simulation verification loop, and it
  should be cited the first time the verification loop is introduced.
- **Shou, …; Hong, …; Ren, …; Wang, J.; Yang, …; Liao, …** *A Physics-Informed Framework for PID Tuning of
  Chemical Processes Using Large Language Model Agents.* arXiv:2607.26594 (29 Jul 2026).
  <https://arxiv.org/abs/2607.26594> — **VERIFIED (delegated)**. LLM agent proposing PID gains against
  simulation-verified IMC targets, 100 FOPDT + 100 SOPDT plants.
- **Guo, X.; Keivan, D.; Syed, U.; Qin, L.; Zhang, H.; Dullerud, G.; Seiler, P.; Hu, B.** *ControlAgent:
  Automating Control System Design via Novel Integration of LLM Agents and Domain Expertise.*
  arXiv:2410.19811 (17 Oct 2024). <https://arxiv.org/abs/2410.19811> — **VERIFIED**. Multi-agent LLM design
  of controllers with a Python computation agent doing evaluation and a history/feedback module; benchmark
  ControlEval, 500 tasks. LLM designs **controllers**, not rewards.
- **Zhang, Z.; Shen, …; Wan, …; Song, …; Sun, M.** *LLM-Guided Safe Reinforcement Learning for Energy System
  Topology Reconfiguration.* arXiv:2603.14018 (14 Mar 2026). <https://arxiv.org/html/2603.14018> —
  **VERIFIED (delegated)**. LLM as a transition-refinement operator whose every proposal is **validated in
  the Grid2Op simulator before storage** (§3.4). Best "LLM + RL in a power system with a simulator gate".
- **Nosrati, …; Tepljakov, A.; Belikov, J.; Petlenkov, E.** *When control meets large language models: From
  words to dynamics.* arXiv:2602.03433 (3 Feb 2026). <https://arxiv.org/abs/2602.03433> —
  **VERIFIED (delegated)**. Survey/position paper; the single framing citation for a TCST introduction.

**What survives.** No LLM has yet been placed in a wind-turbine control loop at any timescale — the
2026-09-16 scan's finding is unchanged, and a targeted re-search today returned only operations, forecasting
and condition-monitoring work in wind. Our *reward-shape* discriminator against Wu et al. survives.
The claim "first LLM-written reward for a control problem" is **dead**; do not make it.

> **Priority / disclosure flag (not a literature finding).** The public GitHub repository
> `Hanny658/Region-aware-Agentic-RL-on-Wind-Turbine` is indexed and was returned as a **top hit** on three
> independent searches I ran today (region-gated residual RL, LLM reward design for wind, RL-vs-ROSCO pitch),
> with the search engine paraphrasing the architecture, the oracle region rule and the gating thresholds
> back to me. This is prior public disclosure of the method. It does not affect novelty for a journal, but
> it does mean a reviewer or a competitor can find the design, and it means the manuscript cannot claim
> the architecture is undisclosed.

---

### A(iv) — Any collective-pitch-only method beating a **tuned** ROSCO on regulation and both DELs at once

**Verdict: SURVIVES.** No counterexample found, in this scan or the previous one.

What was checked and why each candidate fails to undercut us:

- **Lara, M.; Mulders, S. P.; van Wingerden, J.-W.; Vázquez, F.; Garrido, J.** *Simultaneous tuning of
  collective and individual pitch controllers for blade load reduction and power regulation in offshore wind
  turbines.* Ocean Engineering (2026). DOI 10.1016/j.oceaneng.2026.127761.
  <https://zenodo.org/records/22685882> — **VERIFIED**. This is the closest thing to a counterexample and it
  is not one: the baseline is **ROSCO–IPC**, the improvement **requires IPC as well as CPC**, and the abstract
  reports **blade** fatigue loads only — no tower DEL. Numbers read from the Zenodo abstract: conventional
  CPC–IPC "approximately 7–8 % in blade fatigue loads and 23–24 % in power regulation"; azimuth-offset and
  static-inverted-decoupling configurations "approximately 17–19 % … and 29–30 %"; pitch actuator activity
  up "approximately 3–3.5 %". 15 MW turbine, OpenFAST. Seeds and wind speeds not stated in the abstract.
  Bibliography confirmed via <https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.oceaneng.2026.127761>;
  **volume and pages are uncertain** (one source renders "367(2), 127761", the API returns none — do not
  print a volume).
- **Espinoza et al., ISA Transactions 169:428 (2026)** — already in the 2026-09-16 scan. Reports a scaled
  standard-deviation dispersion metric, **not DEL**, on one wind profile with no seed statistics, and its
  text contradicts its own table. Still the only published RL-pitch-vs-ROSCO comparison. Not a counterexample.
- **Guo & Schlipf, WES 8:1299 (2023)** — documents the opposite: retuning for tower load *worsens* blade load
  (tower FA +19.7 yr lifetime, blade OoP −3.3 yr without lidar). This paper is **evidence for** our claim.
- **Anand & Bottasso, WES 11:1989 (2026)** — compares ENMPC-with-augmentation against ENMPC-without; there is
  no ROSCO in the comparison at all (verified on the article page, §4.3).

**How to phrase it, and the one thing that must be defended.** The claim is safe, but it rests entirely on
the tuned-ROSCO baseline being *credible*, so the manuscript has to justify that the shipped corner frequency
is genuinely the untuned part. There is a clean citation for this. From Abbas et al., WES 7:53 (2022),
<https://wes.copernicus.org/articles/7/53/2022/> — **VERIFIED** — Sect. 2.1: "the ROSCO toolbox tuning
process **generically tunes the cutoff frequency for the filter to be one-quarter of the blade's first
edgewise natural frequency**". That is a turbine-agnostic rule of thumb, not an optimisation, which is
exactly the argument for retuning it to 3.42 rad/s and reporting both. Same page, Sect. 4: the pitch PI
gains are interpolated on low-pass-filtered pitch angle, which is the part of the GSPI that *is* tuned.

Supporting fact for the claim that ROSCO is the right baseline rather than an academic strawman:
**Corredera, G.; López, I.; Vázquez, F.; Garrido, J.; Ruz, M. L.** *ROSCO Control on PLC via
Software-in-the-Loop Architecture for Wind Turbines.* Jornadas de Automática **47** (2026).
DOI 10.17979/ja-cea.2026.47.13854. <https://zenodo.org/records/22339260> — **VERIFIED (delegated)**.
ROSCO ported to Structured Control Language on a virtualised Siemens PLC; VAF > 90 % vs the native
implementation in most cases.

---

### Three additional dangers not in the original A list

**D1. Offset-free MPC for wind turbines already exists, with a Luenberger observer, validated in FAST.**

> **Liu, X.; Guo, S.; Kong, X.; Ma, L.; Lee, K. Y.** *Offset-Free Stochastic MPC for Uncertain Wind Energy
> Conversion System.* IEEE Transactions on Industrial Informatics **20**(7), 9487–9496 (2024).
> DOI 10.1109/TII.2024.3384525. <https://api.openalex.org/works?search=Offset-Free+Stochastic+MPC+for+Uncertain+Wind+Energy+Conversion+System>
> — **VERIFIED (bib only)**; the OpenAlex record's abstract states a Luenberger observer estimates modelling
> errors from linearization, tube-based techniques handle disturbances, and validation uses simulation and
> FAST. IEEE Xplore returns 403; **read the paper before the related-works paragraph is finalised.**

Consequence: we must not claim offset-free MPC for wind turbines as new. Our claim has to be the *measurement*
— what the estimator recovers versus what a learned residual recovers on the same plant across ±15 % Cp/Ct —
which is still, as far as this and the previous scan can tell, unquantified by anyone.

**D2. The same-turbine, same-simulator competitor for "repair the model and the MPC wins".**

> **Anand, A.; Bottasso, C. L.** *Adaptive economic wind turbine control.* Wind Energy Science **11**,
> 1989–2008 (2026). DOI 10.5194/wes-11-1989-2026. <https://wes.copernicus.org/articles/11/1989/2026/> —
> **VERIFIED**.

NREL 5 MW in OpenFAST (33 states) with an 8-state reduced-order model — structurally very close to our
3-state LPV-MPC. Economic NMPC maximising power revenue minus fatigue cost via a "Parametric Online Rain Flow
Counting" continuous surrogate. Mismatch is repaired by **offline** neural-network augmentation of the ROM:
"Adaptivity is obtained by a controller-internal gray-box model … implemented via a neural network that is
trained offline." Results I read on the article page: §4.3.1 "ENMPCaug results in 9 % higher economic profit
than ENMPC", with "significantly smaller pitch and torque travel"; §4.3.2 reports 7 % (anemometer) and 30 %
(lidar) profit increases. Cost weights are **not** wind-scheduled — the revenue rate w_p is fixed (verified).

Two honest differentiators to state: their correction is **offline** (certifiable, but does not track drift),
ours is **online** and flat over ±15 % Cp/Ct; and their objective is economic profit, not the four-term J.
Their result also *supports* roadmap s24 — once the model error is compensated, a residual has little left
to learn.

**D3. A 2026 physics-plus-neural-residual MPC for wind turbine load reduction.**

> **Wang, B.; Zhao, Z.; Wang, Y.; Lin, J.; Chen, T.; Xiong, Z.; He, W.** *Wind Turbine Load Reduction
> Predictive Control Strategy Based on Dynamic Model and Data-Driven Approach.* IET Renewable Power
> Generation (2026). DOI 10.1049/rpg2.70337.
> <https://api.semanticscholar.org/graph/v1/paper/DOI:10.1049/rpg2.70337> — **VERIFIED (bib only)**.
> The API abstract describes a hybrid physics + neural-network-correction MPC (DDMPC) over 3–25 m/s and
> mentions a 13.8 % tower-base moment reduction; the Wiley page is 403 and **that figure is second-hand —
> do not quote it.** The distinction that matters for us: their network is a residual on the **model**, ours
> is a residual on the **action**. Worth stating explicitly.

---

## 2. Related-works entries, grouped by theme

### 2.1 Wind turbine pitch control with RL (2024–2026)

| Entry | URL | Status | Relevance |
|---|---|---|---|
| de Frutos, M.; Marino, O. A.; Huergo, D.; Ferrer, E. *DRL for Multi-Objective Optimization: Enhancing Wind Turbine Energy Generation while Mitigating Noise Emissions*, arXiv:2407.13320 (2024) | <https://arxiv.org/html/2407.13320v1> | VERIFIED (delegated) | Double DQN, torque + pitch, Siemens SWT2.3-93 in OpenFAST with BPM aeroacoustics. Baseline is a **plain** variable-speed controller, not ROSCO. §3.3 Table 4: under a 45 dB constraint yearly energy falls 22 % (2 722 vs 3 458 MWh). The cleanest recent RL-pitch paper evaluated as a **Pareto front** — methodologically the same move as our J-vs-actuator-duty front. A Wind Energy 28, e70041 (2025) journal version may exist — **UNVERIFIED**. |
| Nilsen, M. B.; Quick, J.; Göçmen, T.; Dimitrov, N.; Réthoré, P.-E. *Accelerating Reinforcement Learning for Wind Farm Control via Expert Demonstrations*, J. Phys. Conf. Ser. **3224**, 032016 (2026), TORQUE 2026 | <https://iopscience.iop.org/article/10.1088/1742-6596/3224/3/032016> | VERIFIED (delegated) | Wind-*farm* yaw, not pitch, but it is the 2026 statement of the premise our architecture rests on: the untrained agent underperforms the greedy baseline by ~12 %, behaviour-cloning pretraining raises it to near-baseline, all configurations converge within 250 000 steps. Cite as independent confirmation that from-scratch RL starts *below* its baseline. |
| Soler et al., ESWA 249:123502 (2024) — preprint arXiv:2402.11384 | <https://arxiv.org/pdf/2402.11384> | VERIFIED (delegated) | **Same group as de Frutos et al.** Cite one or the other, not both as independent evidence. |
| Espinoza et al., ISA Trans. 169:428 (2026) | <https://pubmed.ncbi.nlm.nih.gov/41539907/> | UNVERIFIED (PubMed cookie wall, both scans) | Only published RL-pitch-vs-ROSCO comparison; already characterised in the 2026-09-16 file. **No number from it may be quoted.** |

### 2.2 MPC for wind turbines

| Entry | URL | Status | Relevance |
|---|---|---|---|
| Wintermeyer-Kallen et al., Forsch. Ingenieurwes. **85**, 385–394 (2021) | <https://link.springer.com/article/10.1007/s10010-021-00475-w> | VERIFIED | **The scheduled-weight prior art** — see §1 A(i). Must be cited. |
| Wintermeyer-Kallen, T.; Basler, M.; Konrad, T.; Zierath, J.; Abel, D. *Challenges of applying model-based predictive wind turbine control in the field*, Forsch. Ingenieurwes. **87**, 119–128 (2023) | <https://api.semanticscholar.org/graph/v1/paper/DOI:10.1007/s10010-023-00634-1> | VERIFIED (bib only; full abstract read) | Field-test lessons: "the highly varying sensitivity to the pitch angle and the dynamic responses of the rotor speed and mechanical loads to the actuator commands over the partial and full load ranges", limited real-time computation, safety. The motivation citation for scheduling. |
| Klein, A.; Wintermeyer-Kallen, T.; Zierath, J.; Kluge, K.; Abel, D.; Vallery, H.; Basler, M. *Design and Practical Evaluation of Robust Model Predictive Wind Turbine Control*, Wind Energy **28**(5), e70011 (2025) | <https://api.semanticscholar.org/graph/v1/paper/DOI:10.1002/we.70011> | VERIFIED (bib only; full abstract read) | EKF for nonlinear state estimation + robust LTV-MPC, **field-validated on a 3 MW turbine**, "3-h continuous full access", measured wind 4.76–13.06 m/s, partial/transition/lower-full load. Authors concede "the tuning effort still remains complex". The honest counterpoint to our simulation-only study, and the best "MPC is deployable but tuning-heavy" citation. Gold OA. |
| Anand, A.; Bottasso, C. L., WES **11**, 1989–2008 (2026) | <https://wes.copernicus.org/articles/11/1989/2026/> | VERIFIED | See D2. Same turbine, same simulator, offline NN model repair, +9 % profit. |
| Ribnitzky, D.; Petrović, V.; Kühn, M. *Experimental investigation of wind turbine controllers for the Hybrid-Lambda Rotor*, WES **11**, 469–491 (2026) | <https://wes.copernicus.org/articles/11/469/2026/> | VERIFIED (delegated) | Wind-tunnel experiment on MoWiTO 1.8. §3.1: model uncertainty makes the baseline exceed its load constraint by 9 %. §3.2: feed-forward cuts gust overshoot to 4 % vs 25 %. **§3.5: feed-forward costs ~3× the baseline pitch duty cycle.** An independent experimental instance of the "≈3× pitch travel buys the load benefit" knee we found in s26. |
| Bai, D.; Wang, B.; Li, Y.; Wang, W. *Study on load reduction and vibration control strategies for semi-submersible offshore wind turbines*, Sci. Rep. **15**, 1148 (2025) | <https://www.nature.com/articles/s41598-025-85476-3> | VERIFIED (delegated) | Equivalent-wind-speed IPC vs ROSCO's IPC module, IEA 15 MW, OpenFAST. Reports that **IPC effectiveness is "less pronounced" under turbulent than steady wind** — independent support for our IPC negative result. Do **not** cite it as an MPC-vs-H∞ benchmark; secondary sources misreport this. |
| Wang, B. et al., IET Renew. Power Gener. (2026), DOI 10.1049/rpg2.70337 | <https://api.semanticscholar.org/graph/v1/paper/DOI:10.1049/rpg2.70337> | VERIFIED (bib only) | See D3. Residual on the **model**, not the action. |
| Abbasi et al., *Multiple model predictive control for offshore wind turbines operating in the full-load range*, Asian J. Control (2025), DOI 10.1002/asjc.3549 | — | UNVERIFIED | Wiley 403. Open before citing. |
| *An optimal model predictive control based on Hammerstein model considering fatigue load reduction for wind turbines*, IJEPES (2025), PII S0142061525006805 | — | UNVERIFIED | NREL 5 MW, HMPC vs MPC vs MMPC. ScienceDirect 403. |

### 2.3 Disturbance observers / offset-free & adaptive MPC in wind energy

| Entry | URL | Status | Relevance |
|---|---|---|---|
| Liu, X.; Guo, S.; Kong, X.; Ma, L.; Lee, K. Y., IEEE TII **20**(7), 9487–9496 (2024) | OpenAlex record (see D1) | VERIFIED (bib only) | **The offset-free-MPC-in-wind prior art.** Read before finalising. |
| Pamososuryo, A. K.; Spagnolo, F.; Mulders, S. P. *Analysis and calibration of optimal power balance rotor-effective wind speed estimation schemes for large-scale wind turbines*, WES **10**, 987–1006 (2025) | <https://wes.copernicus.org/articles/10/987/2025/> | VERIFIED (delegated) | The right citation for the estimator half of our compensated MPC, and a principled replacement for our hand-picked τ = 5 s: §4.2.1 gives a gain-tailoring rule L₂ = h·J·ω₀² that holds estimator natural frequency and damping constant as inertia scales; §4.3 finds the state-estimator power estimator has "less noise propagation and phase lag" than numerical derivatives. NREL 5 MW and IEA 22 MW. |
| Klein et al. (2025) | see §2.2 | VERIFIED (bib only) | EKF inside a field-validated MPC. |
| Lio, W. H. et al., *Real-time rotor effective wind speed estimation based on actuator disc theory: Design and full-scale experimental validation*, Wind Energy (2023), DOI 10.1002/we.2858 | — | UNVERIFIED | Wiley 403. The best "does REWS estimation work on a real turbine" citation if it can be opened. |

### 2.4 Residual RL

See §1 A(ii) for the gated entries (Abbas 2026 / arXiv:2310.14788; Kim arXiv:2609.21307; Luque-Cerpa
arXiv:2601.20666; Saxena & Scheinker arXiv:2606.11474; Cao arXiv:2606.31320; Jeon arXiv:2510.12717).
Additional non-gated entries worth citing:

| Entry | URL | Status | Relevance |
|---|---|---|---|
| Kwon, K.-b.; Park, S.; Kim, D. *Degradation-Aware Pumping Control of Variable-Speed Pumped Storage via Residual Reinforcement Learning*, arXiv:2607.06911 (8 Jul 2026) | <https://arxiv.org/abs/2607.06911> | VERIFIED | The closest **energy-domain** analogue: residual RL over a deterministic feedforward-PI base, residual "adjusts only the rotor speed within a fixed bound the gate loop can always absorb, so the worst-case command is bounded by construction", trained against a degradation index combining hydraulic loss with power and actuation variation. Abstract numbers: BEP tracking error down "roughly 96 %" vs a fixed-speed baseline, degradation down "up to about 56 %". A fatigue-style objective with an actuation term — our J's nearest cousin outside wind. |
| Liu, Q.; Guo, …; Deng, …; Liu, H.; Li, D.; Sun, … *Residual Deep Reinforcement Learning for Inverter-based Volt-Var Control*, arXiv:2408.06790 | <https://arxiv.org/abs/2408.06790> | VERIFIED (delegated) | Residual DRL on a model-based optimisation base in power distribution, learning in a reduced residual action space. arXiv admin note flags text overlap with arXiv:2210.07360 — cite with care. |
| Sheng, …; Huang, …; Chen, … *Knowledge-informed model-based residual RL for CAV trajectory control*, Commun. Transp. Res. **4**, 100142 (2024); arXiv:2408.17380 | <https://arxiv.org/abs/2408.17380> | VERIFIED (delegated) | Base = Intelligent Driver Model, NN learns the residual dynamics. Model-prior-plus-residual in a physical control domain. |
| *Accelerating Residual RL with Uncertainty Estimation*, arXiv:2506.17564 | <https://arxiv.org/abs/2506.17564> | VERIFIED (delegated, abstract only) | Uses base-policy uncertainty to focus **exploration** where the base is not confident — a soft analogue of a gate that steers exploration, not residual authority. |

**Deliberately excluded:** arXiv:2512.06250 *Learning When to Switch: Adaptive Policy Selection via
Reinforcement Learning* — the title matches our need but the content is grid-world maze navigation,
single-author. Recorded here only so it is not "discovered" later as a missed hit. **Do not cite in TCST.**

### 2.5 LLM agents for reward design and for optimisation / tuning

See §1 A(iii) for the full list (Wu et al. 2026; Cardenoso & Caarls arXiv:2511.19355; Casas et al.
arXiv:2608.01649; Gao et al. arXiv:2602.23876; Rasheed et al. arXiv:2511.00337; Chen et al. Processes 2026;
Shou et al. arXiv:2607.26594; Guo et al. arXiv:2410.19811; Zhang et al. arXiv:2603.14018; Nosrati et al.
arXiv:2602.03433). The LLM-proposer-versus-random-search strand, which our ablations need:

| Entry | URL | Status | Relevance |
|---|---|---|---|
| Li, Y. *Greedy Is a Strong Default: Agents as Iterative Optimizers*, arXiv:2603.27415 (28 Mar 2026) | <https://arxiv.org/abs/2603.27415> | VERIFIED (delegated) | The companion to arXiv:2606.21641 already in our files, cutting the other way: simulated annealing, parallel investigators and a second LLM give **no benefit over greedy hill climbing at 2–3× the evaluations**. Reported across four tasks, including MobileNetV3-Small/STL-10 84.5 → 85.8 % with **zero catastrophic failures vs 60 % for random search**, and "tying CMA-ES with 3× fewer evaluations" on XGBoost/Adult. Together with 2606.21641 this is the citation pair for **"the namespace, not the proposer, is where the LLM earns its keep"** — our exact positioning. |
| Bigdeli, …; Zhou, J.; Bak, S. *Large Language Models as Falsifiers for Cyber-Physical Systems*, arXiv:2609.20752 (17 Sep 2026) | <https://arxiv.org/html/2609.20752> | VERIFIED (delegated) | LLM proposes candidate inputs, **the CPS simulator and an STL monitor decide** — LLM-proposes / simulation-verifies, in a control setting. Ranked highest on 14 of 21 specifications against eight ARCH-COMP 2025 tools; for AT1, 1.0 mean simulations vs FReaK 4.8, with uniform random failing to succeed. Double duty: it is also a rare budget-aware LLM-vs-random comparison in control. |
| Zhang, B.; Wang, C.; Wu, K. *EVOM: Agentic Meta-Evolution of Actor-Critic Architectures for RL*, arXiv:2606.26327 | <https://arxiv.org/abs/2606.26327> | VERIFIED (delegated, abstract only) | Compares against **an LLM-guided random search** — structurally our `random_hparam` / random-reward-structure control. The abstract gives **no numbers and does not state whether the random control is budget-matched**; treat as unquantified. |
| Guo, …; Li, C.; Feng, …; Zhang, C. *Code Evolution for Control: Synthesizing Policies via LLM-Driven Evolutionary Search*, arXiv:2601.06845 | <https://arxiv.org/html/2601.06845v1> | VERIFIED (delegated) | LLM evolves **policy code**, not rewards. Benchmark is LunarLander-v3 only — adjacency citation, not evidence. |

### 2.6 IPC context (for the negative-result section)

| Entry | URL | Status | Relevance |
|---|---|---|---|
| Hummel, J. I. S.; Kober, J.; Mulders, S. P. *Increasing the blade tip-to-tower clearance using individual pitch control*, J. Phys. Conf. Ser. **3224**, 052022 (2026), TORQUE 2026 | <https://iopscience.iop.org/article/10.1088/1742-6596/3224/5/052022> | VERIFIED (delegated) | The 2026 companion to the WES 10:2005 (2025) entry already in our files. IEA 15 MW, DLC 1.1 and 1.5 in WEIS; both variants "improve the tower clearance by more than 5 m, even with a slight positive effect on annual energy production". An IPC objective where IPC *does* pay — which sharpens our claim that it does not pay for the fatigue-DEL objective J. |
| Lara Ortiz, M.; Ruz, M. L.; Vazquez, F.; Garrido, J. *Feedforward individual pitch control of wind turbines in the nominal region*, J. Phys. Conf. Ser. **3224**, 052002 (2026) | <https://zenodo.org/records/20433042> | VERIFIED (delegated) | 15–20 % fatigue reduction at 15–20 % increased control effort, framed explicitly as blade fatigue vs pitch-actuator damage. A third external data point on the actuation-cost front. |

---

## 3. Reporting conventions — what the field actually does

Every number in this section was read on a fetched page. Sources are Wind Energy Science article pages
(which fetch reliably) plus arXiv HTML. **IEC 61400-1 itself could not be fetched in any edition** — every
statement about what the standard "requires" below is a *paper's attribution*, not the standard's text.

### 3.1 Short-term DEL, as printed

Three independent WES papers print the same form:

- `https://wes.copernicus.org/articles/9/1885/2024/` — `DEL_ST := (Σ n_i S_i^m / n_ref)^(1/m)` with
  **n_ref = 600** for 1 Hz DELs over 10 minutes.
- `https://wes.copernicus.org/articles/10/2903/2025/` §3.3.1 Eq. (5) — `DEL = [Σ_i n_i L_i^m / n_eq]^(1/m)`.
- `https://wes.copernicus.org/articles/10/2005/2025/` §3.1 Eq. (9) — same form.

Palmgren–Miner as printed: `https://arxiv.org/html/1411.3925` §3.1 gives S-N `s^k N = K`, damage
`D(T) = Σ 1/N_i = Σ s_i^k / K`, with **k = 4, K = 6.25e37** "adequate for steel structures".
`https://wes.copernicus.org/articles/7/1171/2022/` §1 states the design principle: "The Palmgren–Miner rule
is the standard approach followed in the design of wind turbines by which it is ensured that the linear
damage sum over an intended lifetime is lower than unity after considering required safety margins",
with a **25-year** design lifetime.

### 3.2 Aggregation over the wind distribution — Rayleigh vs Weibull

Both conventions are live, and papers state which they use.

- **Rayleigh (IEC class), the cleanest printed lifetime aggregation**:
  `https://wes.copernicus.org/articles/9/799/2024/` prints
  `DEL_lifetime^m = Σ_Vbin Σ_Tbin [(DEL_bin)^m · P(T_bin|V_bin) · P(V_bin)]`, with a **Rayleigh** distribution
  for the mean wind speed, IEC **class 1A (V_ref = 50 m/s, V_ave = 0.2·V_ref = 10 m/s)**, **20-year** design
  lifetime, load cases taken as **IEC DLC 1.2**. Note it also integrates over a turbulence bin conditional on
  wind speed, which the plain Rayleigh-only scheme does not.
- **Site-specific Weibull**: Guo & Schlipf, `https://wes.copernicus.org/articles/8/1299/2023/` §5.2 Eq. (10) —
  `DEL = (Σ_i Σ_j A_eq,ij^m · N_10min/N_ref · P_Uhub,i · P_sta,j)^(1/m)`, Weibull **shape 2.02, scale 9.41 m/s
  at 10 m**, three atmospheric stability classes with FINO1 probabilities. Their Eq. (12) converts a DEL ratio
  into an **extended lifetime**: `EL = 20 (DEL_i/DEL_j)^(−m−1)` over a 20-year design life — a convenient way
  to express a % DEL change in years. (I fetched this page independently and confirm §4.1, §5.2 and Eq. 12.)
- **Neither** — measured conditions instead of a fitted distribution:
  `https://wes.copernicus.org/articles/10/2865/2025/` uses ERA5 hourly conditions over a 25-year lifetime,
  Eq. (11), with **n_eq = 1e6** reference cycles.

**Recommendation for our results section.** Our 12–24 m/s span covers only part of the Rayleigh mass, so
either report a **truncated-and-renormalised** Rayleigh weighting (stating the truncation explicitly) or
report **per-wind-speed DELs only**. The literature is split and several of the papers above do the latter.
Since we already report per-wind-speed terms, the cheapest credible addition is one lifetime-DEL row computed
as `LDEL = [Σ_j p_j DEL_j^m]^(1/m)` with p_j the class-B Rayleigh bin probability, plus the Guo–Schlipf
extended-lifetime conversion so the tower gain reads in years. State design lifetime (20 or 25 yr) and n_eq.

### 3.3 Seeds per wind speed

**Six is the de-facto floor**, repeatedly attributed to IEC 61400-1 in peer-reviewed work, and never verified
from the standard here:

- `https://wes.copernicus.org/articles/9/799/2024/`: "we use a sample size of six (recommended number of
  samples by the IEC 61400-1)", DLC 1.2.
- `https://wes.copernicus.org/articles/4/397/2019/`: "six seeds were used to limit the computational cost of
  the MC analysis, following accepted international standards (IEC61400-1, 2005)".
- `https://wes.copernicus.org/articles/6/1401/2021/` §3, a fully specified DLC 1.2 matrix: "18 10 min load
  simulations (three yaw directions: 0° ± 10°, and **six turbulent wind seeds**) for each mean wind speed
  ranging from **4 to 26 m/s in the interval of 2 m/s**, which results in a total of **216 simulations**",
  sampled at 50 Hz.
- `https://wes.copernicus.org/articles/9/1791/2024/` §3.2 and `https://arxiv.org/html/2601.01657`: six seeds.
- `https://ar5iv.labs.arxiv.org/html/2110.14169`: "6 random turbulence seeds at each whole-numbered wind
  speed", 12–24 m/s, "78 simulation cases for each controller".
- Guo & Schlipf §4.1, confirmed on my own fetch: "six independent simulations are performed that have
  different random seed numbers", 10–24 m/s.

Papers using more: `https://wes.copernicus.org/articles/8/149/2023/` §5.1.3 uses **12 seeds** per stability
class at "12 to 24 m/s with a step of 2 m/s" — essentially our protocol with twice the seeds;
`https://wes.copernicus.org/articles/10/2005/2025/` uses 10 seeds at one wind speed; surrogate-modelling
papers use 44 and 300.

**Where that leaves us.** 6 seeds × 7 wind speeds = 42 episodes sits **at the floor** and **below** the
closest-matching control study (WES 8:149, 12 seeds, same wind grid). The honest options are (a) state that
six is the conventional number with the citations above, and lean on the paired bootstrap to carry the
statistics, or (b) add 6 more seeds on the final comparison table only. Do **not** write "IEC requires six
seeds" without reading ed. 4 §7.4 offline.

### 3.4 Episode length and discarded transient

The convention is unambiguous: **simulate transient + 600 s, analyse the last 600 s.** Discarded transients
in the fetched set range from 60 s to 400 s.

| Source | Total | Discarded | Analysed |
|---|---|---|---|
| `wes.copernicus.org/articles/9/799/2024/` | 700 s | first **100 s** ("recognized as transient time and is omitted") | 600 s |
| `wes.copernicus.org/articles/8/1299/2023/` (Guo & Schlipf, §4.1) | 700 s | "the initial **100 s** results are ignored" | 600 s |
| `wes.copernicus.org/articles/9/1885/2024/` | 900 s | "the first **300 s** is discarded to exclude the initial transient" | 600 s |
| `wes.copernicus.org/articles/10/2865/2025/` | 300 s ramped init + 600 s | **300 s** | 600 s |
| `arxiv.org/html/2601.01657` (FLOAT) | 1000 s | first **400 s** | 600 s |
| `ar5iv.labs.arxiv.org/html/2110.14169` | 800 s | "the first **200 seconds** of transient settling discarded" | 600 s |
| `wes.copernicus.org/articles/8/149/2023/` §5.1.3 | 31 min | "the initial **60 s** … which contains the initialization" | 30 min |
| `wes.copernicus.org/articles/10/2005/2025/` §3.3 | 2100 s | first **300 s** | 1800 s |
| `wes.copernicus.org/articles/10/2903/2025/` | 4600 s | first **1000 s** | 3600 s |
| `wes.copernicus.org/articles/9/1791/2024/` §3.3 | 10-min bins | "the first **minute** of each 10 min bin" | ~9 min |

On the 600-s norm itself: `https://wes.copernicus.org/articles/8/575/2023/` — "Following the IEC 61400-1
recommendation for fatigue load evaluation, a 600 s stochastic wind profile generated using TurbSim software
is used."

**Action for us.** Our 600 s episodes must be checked: if they *include* the OpenFAST/ROSCO startup
transient, we are out of step with every paper above and a reviewer will say so. Either simulate 700 s and
score the last 600 s, or state in the methods exactly how many seconds are discarded and why.

### 3.5 Wöhler exponents

Our m = 4 (tower FA) and m = 10 (blade root OoP) are squarely conventional, but m = 3 and m = 3.5 are live
competitors for the tower and the choice should be justified in one sentence.

- **Tower m = 4**: `wes.copernicus.org/articles/10/2903/2025/` (justified as "a compromise based on an
  investigation of tower stress cycles … distributed on both sides of the transition point of the bi-linear
  S–N curve"); Guo & Schlipf §4.1 ("m = 4 is considered for the tower and shaft loads, m = 10 is considered
  for the blade loads" — confirmed on my own fetch); `arxiv.org/html/2601.01657` §3.7.1 (DNV-RP-C203 Type E,
  "m set by default to 4, the average slope of the two S-N curve regimes", slopes 3 and 5, transition at 1e7).
- **Tower m = 3**: `wes.copernicus.org/articles/9/799/2024/` ("equal to 3 in the case of steel components");
  `wes.copernicus.org/articles/8/575/2023/` ("typically 3 for steel materials like the tower and 10 for
  composites like the blade").
- **Tower m = 3.5, blade flap 10, blade edge 8**: `wes.copernicus.org/articles/9/1885/2024/` and
  `wes.copernicus.org/articles/10/2865/2025/`.
- **Sensitivity ranges actually studied**: `wes.copernicus.org/articles/9/799/2024/` examines
  **m = 8, 10, 12 for blades and m = 3, 4, 5 for the tower base**. This is the citation that makes our
  choice a mid-range convention, and it makes an m-sensitivity row cheap to defend.

### 3.6 Actuator duty — reported routinely, and there is a standard definition

- **ADC**, `https://wes.copernicus.org/articles/10/2005/2025/` §3.1 Eq. (10):
  `ADC = (1/T) ∫₀ᵀ |u̇(t)| / u̇_max dt`, with **u̇_max = 2° s⁻¹** for the IEA 15 MW.
- Same definition, `https://wes.copernicus.org/articles/7/523/2022/` Eqs. (38)–(39):
  `ADC = (1/T) ∫₀ᵀ β̇(t)/β_max dt` with **β_max = ±8° s⁻¹** for a 10 MW turbine.
  ADC is therefore **time-averaged pitch rate normalised by the actuator rate limit** — dimensionless, and
  **not** the same as raw pitch travel in degrees. Because the normaliser is turbine-specific (2 vs 8 °/s),
  ADC values are not comparable across papers unless the limit is stated.
- **Pitch travel as a % change vs baseline**, which is our convention:
  `https://wes.copernicus.org/articles/8/575/2023/` reports "the average total pitch travel marginally
  increases by 0.13 %" alongside its fatigue claims.
- Guo & Schlipf report blade pitch **rate** statistics and flag that their optimally tuned feedback "gives
  higher blade pitch rates, which are even doubled for very high mean wind speed ranges" (§6) — an
  independent precedent for the roughly-2× travel our range-set MPC rows show.

**Recommendation.** Keep the pitch-travel ratio vs the tuned ROSCO, and **add ADC with the rate limit
stated** so the actuator cost is directly comparable with WES 7:523 and WES 10:2005.

### 3.7 Statistical practice — the field's bar is low, and this is an opportunity

- `https://wes.copernicus.org/articles/4/397/2019/`: "the use of only six seeds does not guarantee the full
  convergence of all quantities, especially in terms of standard deviations"; the differences in AEP and DELs
  are small, "this is not true for the ultimate loads". **The citation for "six seeds is adequate for DELs,
  marginal for higher moments."**
- `https://wes.copernicus.org/articles/10/2903/2025/`: individual seeds as transparent markers, seed means
  opaque, **standard deviation across seeds as error bars, no significance testing or confidence intervals**.
- `https://wes.copernicus.org/articles/10/2005/2025/` §3.3: means over 10 wind fields, **1σ bands**.
- `https://wes.copernicus.org/articles/9/1791/2024/` §3.3: **median and interquartile range**.
- Guo & Schlipf: plain seed means, no intervals.
- `https://ar5iv.labs.arxiv.org/html/2110.14169`: the two controllers share an identical 78-case matrix
  (a paired design by construction) but **no paired test is applied**.

**Nothing in the fetched set performs a paired bootstrap over episodes.** Our per-wind-speed terms with
paired bootstrap intervals over a common seed set put us **above** the field's norm. Say so explicitly in
the methods — it is a defensible protocol contribution, and it is the reason we can resolve ~1-J differences
that seed-mean-only papers cannot.

### 3.8 The per-wind-speed non-degradation constraint (our claim 5)

Reporting **per-bin** load metrics across wind speed is standard (the LPV and lidar-assisted comparison
papers above all do it). A hard acceptance criterion of the form "no metric may be more than 1 % worse than
the baseline at *any* wind speed" did not appear in anything I fetched. Treat it as a **protocol
contribution**, state it as such, and do not claim it is an established practice.

### 3.9 Gaps that need an offline check before the paper is finalised

1. **IEC 61400-1 ed. 4, DLC 1.2 clause text** — no fetchable copy parsed in either scan. The "six seeds"
   statement must be attributed to papers, not to the standard, until someone reads §7.4 / the DLC table.
2. **NREL MLife theory manual** (`nrel.gov` unreachable from this environment) — the canonical printed
   lifetime-DEL formula following IEC 61400-1 ed. 3 Annex G. Quote nothing from it yet.
3. **IEC turbulence class B, I_ref = 0.14** — could not be verified. Only the neighbouring value was seen in
   print: `https://wes.copernicus.org/articles/9/2001/2024/` states "a reference value of turbulence
   intensity was taken to be 0.12, for the least turbulent wind turbine class C". The A/B/C = 0.16/0.14/0.12
   and V_ref = 50/42.5/37.5 table appeared **only in search summaries** — do not cite it from here.

---

## 4. Suggested BibTeX — verified fields only

Fields below were read from a fetched page or a metadata API. **Missing fields are missing because they
could not be verified** — do not fill them in from memory. Entries marked with a comment are incomplete
by design.

```bibtex
@article{Wintermeyer2021weightsched,
  title   = {Weight-scheduling for linear time-variant model predictive wind turbine control toward field testing},
  author  = {Wintermeyer-Kallen, Thorben and Dickler, Sebastian and Zierath, J{\'a}nos and Konrad, Thomas and Abel, Dirk},
  journal = {Forschung im Ingenieurwesen},
  volume  = {85},
  pages   = {385--394},
  year    = {2021},
  doi     = {10.1007/s10010-021-00475-w}
}

@article{Wintermeyer2023challenges,
  title   = {Challenges of applying model-based predictive wind turbine control in the field},
  author  = {Wintermeyer-Kallen, Thorben and Basler, Maximilian and Konrad, Thomas and Zierath, J{\'a}nos and Abel, Dirk},
  journal = {Forschung im Ingenieurwesen},
  volume  = {87},
  pages   = {119--128},
  year    = {2023},
  doi     = {10.1007/s10010-023-00634-1}
}

@article{Klein2025robustmpc,
  title   = {Design and Practical Evaluation of Robust Model Predictive Wind Turbine Control},
  author  = {Klein, Andreas and Wintermeyer-Kallen, Thorben and Zierath, J{\'a}nos and Kluge, Kevin and Abel, Dirk and Vallery, Heike and Basler, Maximilian},
  journal = {Wind Energy},
  volume  = {28},
  number  = {5},
  year    = {2025},
  doi     = {10.1002/we.70011}
  % article number e70011 per the TU Delft repository record; page range not verified
}

@article{Anand2026adaptive,
  title   = {Adaptive economic wind turbine control},
  author  = {Anand, Abhinav and Bottasso, Carlo L.},
  journal = {Wind Energy Science},
  volume  = {11},
  pages   = {1989--2008},
  year    = {2026},
  doi     = {10.5194/wes-11-1989-2026}
}

@article{Liu2024offsetfree,
  title   = {Offset-Free Stochastic {MPC} for Uncertain Wind Energy Conversion System},
  author  = {Liu, Xiangjie and Guo, Shifan and Kong, Xiaobing and Ma, Lele and Lee, Kwang Y.},
  journal = {IEEE Transactions on Industrial Informatics},
  volume  = {20},
  number  = {7},
  pages   = {9487--9496},
  year    = {2024},
  doi     = {10.1109/TII.2024.3384525}
}

@article{Pamososuryo2025rews,
  title   = {Analysis and calibration of optimal power balance rotor-effective wind speed estimation schemes for large-scale wind turbines},
  author  = {Pamososuryo, Atindriyo Kusumo and Spagnolo, Fabio and Mulders, Sebastiaan P.},
  journal = {Wind Energy Science},
  volume  = {10},
  pages   = {987--1006},
  year    = {2025},
  doi     = {10.5194/wes-10-987-2025}
}

@article{Ribnitzky2026hybridlambda,
  title   = {Experimental investigation of wind turbine controllers for the Hybrid-Lambda Rotor},
  author  = {Ribnitzky, Daniel and Petrovi{\'c}, Vlaho and K{\"u}hn, Martin},
  journal = {Wind Energy Science},
  volume  = {11},
  pages   = {469--491},
  year    = {2026},
  doi     = {10.5194/wes-11-469-2026}
}

@article{Lara2026simultaneous,
  title   = {Simultaneous tuning of collective and individual pitch controllers for blade load reduction and power regulation in offshore wind turbines},
  author  = {Lara, Manuel and Mulders, Sebastiaan Paul and van Wingerden, Jan-Willem and V{\'a}zquez, Francisco and Garrido, Juan},
  journal = {Ocean Engineering},
  year    = {2026},
  doi     = {10.1016/j.oceaneng.2026.127761}
  % volume/pages not verified; article number reported as 127761 by one source only
}

@article{Abbas2026specialized,
  title   = {Specialized Deep Residual Policy Reinforcement Learning Framework for Safe and Adaptive Continuous Control},
  author  = {Abbas, Ammar N. and Chasparis, Georgios C. and Kelleher, John D.},
  journal = {IET Control Theory \& Applications},
  volume  = {20},
  number  = {1},
  year    = {2026},
  doi     = {10.1049/cth2.70099}
  % pages not verified
}

@misc{Abbas2023specializedarxiv,
  title         = {Specialized Deep Residual Policy Safe Reinforcement Learning-Based Controller for Complex and Continuous State-Action Spaces},
  author        = {Abbas, Ammar N. and Chasparis, Georgios C. and Kelleher, John D.},
  year          = {2023},
  eprint        = {2310.14788},
  archivePrefix = {arXiv}
}

@misc{Kim2026stabilityaware,
  title         = {Stability-aware Residual Reinforcement Learning Framework for Robotic Manipulator Disturbance Compensation},
  author        = {Kim, Jihong and Kwon, Joonhyuk and Kim, Hwa Soo and Seo, TaeWon and Seo, Hyung-Tae},
  year          = {2026},
  eprint        = {2609.21307},
  archivePrefix = {arXiv}
}

@misc{Kwon2026pumpedstorage,
  title         = {Degradation-Aware Pumping Control of Variable-Speed Pumped Storage via Residual Reinforcement Learning},
  author        = {Kwon, Kyung-bin and Park, SangWoo and Kim, Dam},
  year          = {2026},
  eprint        = {2607.06911},
  archivePrefix = {arXiv}
}

@misc{Cao2026autosafe,
  title         = {Safe Online Learning via Smooth Safety-Structured Policy Composition},
  author        = {Cao, Hongpeng and Zhao, Liqun and Gu, Yuliang and Hovakimyan, Naira and Sha, Lui and Caccamo, Marco},
  year          = {2026},
  eprint        = {2606.31320},
  archivePrefix = {arXiv}
}

@misc{Cardenoso2025learnopt,
  title         = {Leveraging {LLMs} for reward function design in reinforcement learning control tasks},
  author        = {Cardenoso, Franklin and Caarls, Wouter},
  year          = {2025},
  eprint        = {2511.19355},
  archivePrefix = {arXiv}
}

@misc{Casas2026large,
  title         = {{LLM}-Driven Automated Reward Design for Reinforcement Learning-Based Routing in {LEO} Satellite Networks},
  author        = {Casas, Walter P. and da Fonseca, Nelson L. S. and Astudillo, Carlos A.},
  year          = {2026},
  eprint        = {2608.01649},
  archivePrefix = {arXiv}
}

@misc{Rasheed2025llmcontrol,
  title         = {Large Language Models for Control},
  author        = {Rasheed, Adil and Ravik, Oscar and San, Omer},
  year          = {2025},
  eprint        = {2511.00337},
  archivePrefix = {arXiv}
}

@misc{Guo2024controlagent,
  title         = {{ControlAgent}: Automating Control System Design via Novel Integration of {LLM} Agents and Domain Expertise},
  author        = {Guo, Xingang and Keivan, Darioush and Syed, Usman and Qin, Lianhui and Zhang, Huan and Dullerud, Geir and Seiler, Peter and Hu, Bin},
  year          = {2024},
  eprint        = {2410.19811},
  archivePrefix = {arXiv}
}

@article{Chen2026llmranking,
  title   = {Benchmarking Performance Judgment in Open-Weight {LLM} Controller Tuning: Control Knowledge Does Not Ensure Reliable Gain Ranking},
  author  = {Chen, Jiaxuan and Shu, Yang and Li, Hao-Nan},
  journal = {Processes},
  volume  = {14},
  number  = {18},
  pages   = {2954},
  year    = {2026},
  doi     = {10.3390/pr14182954}
}

@article{Wu2026llmrewardweight,
  title   = {A hierarchical control framework for photovoltaic-driven air conditioning systems: {LLM}-based reward weight adaptation and {TD3} real-time control},
  author  = {Wu, Zhenxing and Fu, Xiaoqun and Li, Sihui and Peng, Jinqing and Li, Hongqiang and Li, Houpei},
  journal = {Building Simulation},
  year    = {2026},
  doi     = {10.1007/s12273-026-1467-3}
  % volume/pages not verified; abstract elided by the publisher
}

@article{Wang2026ddmpc,
  title   = {Wind Turbine Load Reduction Predictive Control Strategy Based on Dynamic Model and Data-Driven Approach},
  author  = {Wang, Bin and Zhao, Zhiguo and Wang, Yuyi and Lin, Jun and Chen, Tao and Xiong, Zhe and He, Wei},
  journal = {IET Renewable Power Generation},
  year    = {2026},
  doi     = {10.1049/rpg2.70337}
  % volume/pages not verified; author given names beyond the first are as reported by Semantic Scholar
}
```

Entries deliberately **not** given BibTeX because a required field is unverified: Luque-Cerpa et al.
(arXiv:2601.20666, author list incomplete), Saxena & Scheinker (arXiv:2606.11474, first author's given name
unknown), Gao et al. RF-Agent (arXiv:2602.23876, author list incomplete), Zhang et al. (arXiv:2603.14018),
Shou et al. (arXiv:2607.26594), Bigdeli et al. (arXiv:2609.20752), Li (arXiv:2603.27415), Nosrati et al.
(arXiv:2602.03433), Jeon et al. (arXiv:2510.12717), Liu et al. (arXiv:2408.06790), the two TORQUE 2026 IOP
papers and the Corredera PLC paper. Resolve the author lists from the arXiv or IOP pages before citing.

---

## 5. What changes in the manuscript

1. **Cite Wintermeyer-Kallen et al. (2021) where the scheduled tower weight is introduced**, and narrow the
   claim to "the tower-fatigue weight within the full-load region".
2. **Cite Kim et al. (arXiv:2609.21307) and Abbas et al. (IET CTA 2026) where the gate is introduced**, and
   claim the gate as a domain-and-variable choice, not a mechanism.
3. **Put Wu et al. (Building Simulation 2026) next to the LLM-reward contribution** and make the
   shape-versus-weights sentence load-bearing; the repo's weight-search-plateau evidence is what defends it.
4. **Cite Chen et al. (Processes 2026) the first time the deterministic verification loop appears** — it is
   the statistically careful reason the loop is necessary.
5. **Cite Liu et al. (IEEE TII 2024) before claiming anything about offset-free MPC in wind**; reposition our
   contribution as the *measurement* of estimator-versus-residual, not the formulation.
6. **Engage Anand & Bottasso (WES 2026) directly** — same turbine, same simulator, offline NN model repair,
   +9 % profit — and differentiate on online-versus-offline and on the objective.
7. **Read Abbas et al. WES 7:53 (2022) §2.1 into the tuned-ROSCO justification**: the shipped speed-filter
   corner comes from a generic one-quarter-of-blade-first-edgewise rule, which is exactly why retuning it is
   fair rather than strawman-building.
8. **Check whether our 600 s episodes include the startup transient** and either extend to 700 s or state the
   discard explicitly (§3.4).
9. **Add ADC with the pitch-rate limit stated** alongside the pitch-travel ratio (§3.6).
10. **Say in the methods that per-wind-speed paired bootstrap intervals exceed the field's norm** (§3.7), and
    present the per-wind-speed non-degradation constraint as a protocol contribution, not a convention (§3.8).
