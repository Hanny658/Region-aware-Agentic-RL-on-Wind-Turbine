# Literature scan, 2026-09-23 — gated residual RL, LLM-written rewards, LLM vs random proposers

Scope of the scan: prior art for (a) a residual PPO pitch offset on a model-based base that is
**gated on wind speed** (off below 15 m/s, on above 17 m/s, roadmap s41), (b) the **reward
expression written by the agent** in a loop with deterministic simulation verification, (c) the
**LLM proposer vs random search** control that our ablations report.

Verification discipline used here:

- **VERIFIED** — the page at the given URL was fetched and read; every number below it is quoted
  from the named section/abstract of that page.
- **VERIFIED (bib only)** — the publisher page refused automated fetch (MDPI and Wiley return 403,
  PubMed serves a cookie wall); title/authors/venue/DOI/abstract were read from the Semantic Scholar
  API record for that DOI, or from a publisher fallback URL, which is named. Numbers come from that
  abstract only.
- **UNVERIFIED** — seen in search results only. **No numbers are quoted from these.**

Already known before this scan and not re-reported: Silver et al. arXiv:1812.06298; Johannink et al.
arXiv:1812.03201; Policy Decorator arXiv:2412.13630; Kalaria et al. arXiv:2410.06570; Eureka
arXiv:2310.12931; Text2Reward; Language to Rewards; the budget-matched HPO study arXiv:2606.21641.
Earlier scan: `docs/literature_2026-09-16.md`.

---

## 1. Residual RL on a model-based base, and gating / when to apply the residual

### 1.1 Close to our claim — nearest prior art on a gated residual

**[R1] Abbas, Chasparis, Kelleher — "Specialized Deep Residual Policy Reinforcement Learning
Framework for Safe and Adaptive Continuous Control", IET Control Theory & Applications, 2026,
DOI 10.1049/cth2.70099.**
VERIFIED (bib only) — Wiley 403s; record read from
`https://api.semanticscholar.org/graph/v1/paper/DOI:10.1049/cth2.70099`.
Preprint VERIFIED at `https://arxiv.org/abs/2310.14788` (same three authors, arXiv:2310.14788,
Oct 2023).

The closest existing "gated residual" paper found. Combines residual policy learning on a
conventional controller, a cycle-of-learning warm start from expert trajectories, and a
**specialised RL agent activated only in critical states, identified by an input–output hidden
Markov model**. The published abstract states the framework "optimizes policy learning in critical
states using an input–output hidden Markov model" and is validated on the Tennessee Eastman process
"through experiments that analyse synchronization, **activation mechanisms** and an ablation study".
No quantitative results in either abstract.

Difference from ours: their gate is a *learned latent-regime / abnormality detector* on a PID base
in chemical process control; ours is an *exogenous, physically interpretable wind-speed gate* on an
MPC / tuned-ROSCO base. Pull the full IET text before any novelty sentence about gating is written.

**[R2] Kim, Kwon, H. S. Kim, Seo, Seo — "Stability-aware Residual Reinforcement Learning Framework
for Robotic Manipulator Disturbance Compensation", arXiv:2609.21307v1, 18 Sep 2026.**
VERIFIED — `https://arxiv.org/html/2609.21307`.

Residual RL on a base of exactly our shape: **NMPC + disturbance observer**
(tau_cmd = tau_MPC - d_hat_filt, Eq. 9, §III-C). The residual is **state-gated**: a state-dependent
action bound rho_k = (c_x/gamma_0)(||x_k||^2 - r^2)_+ capped at rho_max, so the bound vanishes
inside a deadband radius r and the policy output is suppressed there (§V-B, Eqs. 25/27). Constants
read in §V-B: c_x = 9.65, gamma_0 = 2.02, r = 0.05 m, kappa = 3.5, rho_max = 3.0 N·m. Results (§VI):
disturbance-estimation error -79.1 %, tracking RMSE -49.2 % (circular) / -16.1 % (figure-eight) in
simulation; zero-shot sim-to-real -27.8 % tracking error and -38.0 % under an unseen base-vibration
disturbance (§VI-C).

Difference from ours: the gate is a *tracking-error deadband* with an ISS certificate ("residual on
when error is large"); ours is "residual on in a named operating region". Cite for "gating the
residual is an established device"; distinguish on the gating *variable* and on the objective
(regulation/fatigue trade rather than tracking).

**[R3] Luque-Cerpa, M. Wang, Carlsson, Seshia, Dubhashi, Torfah — "Learning Contextual Runtime
Monitors for Safe AI-Based Autonomy", arXiv:2601.20666, 28 Jan 2026 (rev. 2 Apr 2026).**
VERIFIED — `https://arxiv.org/abs/2601.20666`.

Learns a **contextual-multi-armed-bandit monitor that observes the system's context and selects the
controller best suited to the current conditions**, instead of averaging an ensemble; claims
theoretical safety guarantees during controller selection and "significant improvements in both
safety and performance compared to non-contextual baselines" (abstract). Autonomous driving, two
simulated scenarios; no numbers in the abstract. This is the "learn *when* to hand authority to
which controller" formulation — ours is a fixed rule, which is a simplicity argument, not a novelty
claim.

**[R4] Saxena, Scheinker — "Mahalanobis-Guided Latent OOD Detection for Hybrid ES-DRL Control in
Time-Varying Systems", arXiv:2606.11474, 9 Jun 2026.**
VERIFIED — `https://arxiv.org/abs/2606.11474`.

**Binary switch between an RL controller and a model-independent fallback (extremum seeking)**,
driven by Mahalanobis distance in a VAE latent space detecting out-of-distribution operating
conditions at test time (abstract). Particle-accelerator control; no performance numbers in the
abstract. Same architecture (regime detector gates the learned controller), no residual structure.

### 1.2 Related but different — residual on a model-based base, no gate

**[R5] Jeon, H. J. Lee, Hong, S. Kim — "Residual MPC: Blending Reinforcement Learning with
GPU-Parallelized Model Predictive Control", arXiv:2510.12717, 14 Oct 2025.**
VERIFIED — `https://arxiv.org/abs/2510.12717`.
Residual RL blended with MPC at the torque-control level; RL trained at 100 Hz across thousands of
parallel agents. Abstract claims higher sample efficiency, greater asymptotic reward, wider
trackable velocity commands, zero-shot adaptation to unseen gaits and uneven terrain. **No gate or
activation schedule in the abstract.** Legged locomotion. The strongest recent "residual on an MPC
base" citation, and its lack of activation logic is our differentiator.

**[R6] Kwon, Park, D. Kim — "Degradation-Aware Pumping Control of Variable-Speed Pumped Storage via
Residual Reinforcement Learning", arXiv:2607.06911, 8 Jul 2026.**
VERIFIED — `https://arxiv.org/abs/2607.06911`.
Energy-system residual RL: base is a **deterministic feedforward-PI gate controller**; the RL
residual adjusts rotor speed within bounds the base loop can absorb; conditions on demand-dependent
best-efficiency-point references over five-minute dispatch blocks. Abstract: BEP tracking error
reduced "roughly 96 %" vs a fixed-speed baseline; total degradation cut "up to about 56 %" in the
most demanding dispatch scenarios; efficiency matches or slightly exceeds model-based optimisation
with tighter block tracking. Best 2026 precedent for "residual RL on a classical controller in a
power plant with a fatigue/degradation objective" — closest domain analogue outside wind.

**[R7] Q. Liu, Guo, Deng, H. Liu, D. Li, Sun — "Residual Deep Reinforcement Learning for
Inverter-based Volt-Var Control", arXiv:2408.06790, 13 Aug 2024.**
VERIFIED — `https://arxiv.org/abs/2408.06790`.
Residual DRL on a **model-based optimisation base** (approximate power-flow model), learning in a
*reduced residual action space*; distribution-network volt-var control. No gating; no numbers in the
abstract. An arXiv admin note flags text overlap with arXiv:2210.07360 — cite with care.

**[R8] Sheng, Huang, Chen — "Traffic expertise meets residual RL: Knowledge-informed model-based
residual reinforcement learning for CAV trajectory control", arXiv:2408.17380 (v2 Feb 2025);
the arXiv page states publication in Communications in Transportation Research 4 (2024) 100142.**
VERIFIED — `https://arxiv.org/abs/2408.17380`.
Base = Intelligent Driver Model, NN learns the residual dynamics. No gating, no numbers in the
abstract. Use only as "model-based prior + residual in a physical control domain".

**[R9] Accelerating Residual RL with Uncertainty Estimation, arXiv:2506.17564.**
VERIFIED (abstract only) — `https://arxiv.org/abs/2506.17564`.
Uses **base-policy uncertainty to focus exploration on regions where the base is not confident** — a
soft analogue of a gate, but it steers *exploration*, not residual *authority*. No numbers in the
abstract (claims outperforming baselines in simulation and zero-shot sim-to-real). Author list was
not read; do not cite without pulling the bib record.

**[R10] Tava — "Learning When to Switch: Adaptive Policy Selection via Reinforcement Learning",
arXiv:2512.06250v1, 6 Dec 2025.**
VERIFIED — `https://arxiv.org/html/2512.06250v1`.
Q-learning picks the **coverage threshold at which to switch** from spiral exploration to A* in maze
navigation; state = (coverage bucket, distance bucket), 50 states, thresholds 20–60 %. Results: 23 %
faster on 16x16, 55 % faster with 83 % variance reduction on 64x64, 46 % worst-case reduction on
128x128, over 240 test configurations. **The title matches our need but the content is grid-world
navigation, single author — do not cite in a TCST paper.** Recorded so it is not re-discovered as a
missed hit.

---

## 2. LLM-written / LLM-designed reward functions, control and energy first

### 2.1 Close to our claim

**[L1] Wu, Fu, S. Li, Peng, H. Li, H. Li — "A hierarchical control framework for photovoltaic-driven
air conditioning systems: LLM-based reward weight adaptation and TD3 real-time control", Building
Simulation, 2026, DOI 10.1007/s12273-026-1467-3.**
VERIFIED — fetched via the Springer cookie-fallback URL
`https://link.springer.com/article/10.1007/s12273-026-1467-3?error=cookies_not_supported`.

**The nearest published analogue of our LLM supervisor.** An LLM acts as a **high-level module
performing adaptive reward-*weight* tuning between control cycles** from historical operational
performance, while TD3 does real-time compressor-speed regulation — a genuine slow-timescale LLM
supervisory loop over a learning controller in an energy system. Abstract numbers: DeepSeek-R1
92.75 % comfort-time ratio; DeepSeek-V3 3.94 CNY/day cost saving and 2.8 kgCO2e carbon reduction;
18 %, 14.2 % and 35.88 % improvements over fixed-weight baselines.

Difference from ours, on the axis our own results already defend: **they tune weights, we have the
agent write the reward expression.** Our finding that a weight search cannot reach the shape change
(saturated speed/power terms, bounded centred load terms) is the discriminator. Cite it and draw the
distinction explicitly, or a reviewer will.

**[L2] Cardenoso, Caarls — "Leveraging LLMs for reward function design in reinforcement learning
control tasks" (LEARN-Opt), arXiv:2511.19355v1, 24 Nov 2025.**
VERIFIED — `https://arxiv.org/html/2511.19355v1`.
LLM writes Python reward functions; generation / execution (A2C training) / evaluation modules, with
**internal code-validation unit tests (§3.3.1)** and the LLM **deriving its own performance metrics
from the system description** (abstract). Protocol: 10 full pipeline runs per environment, each
discovered reward re-tested over 5 seeds (§4.5), on Cartpole, Quadcopter, Ant, Humanoid,
Franka-Cabinet (§4.3). Results (§5.1, Table 6): Cartpole best GP 3.40±0.57e-3 vs baseline
4.80±0.97e-3 (lower better); Franka-Cabinet 0.52±0.35 vs 0.36±0.35. Both LEARN-Opt and EUREKA show
**negative average performance across all 10 runs**, which the authors use to argue reward design is
a high-variance problem needing multi-run treatment.

Bearing on us: their variance finding is the same phenomenon as our "3/5 balanced, +0.4/+0.5 J" row
— a supporting citation for reporting run distributions rather than a best run.

**[L3] Gao, X. Zhang, Jiang, You, M. Zhang, Deng — "RF-Agent: Automated Reward Function Design via
Language Agent Tree Search", arXiv:2602.23876, 27 Feb 2026.**
VERIFIED — `https://arxiv.org/abs/2602.23876`.
Post-Eureka state of the art: MCTS over LLM-generated reward code, motivated by "poor utilization of
historical feedback and inefficient search" in greedy/evolutionary LLM reward design; **17 low-level
control tasks** (abstract). No random-search control stated in the abstract. This supersedes
Eureka/Text2Reward as the "current best" citation for LLM reward design.

**[L4] Casas, da Fonseca, Astudillo — "LLM-Driven Automated Reward Design for Reinforcement
Learning-Based Routing in LEO Satellite Networks" (LARGE), arXiv:2608.01649, 3 Aug 2026.**
VERIFIED — `https://arxiv.org/abs/2608.01649`.
LLM generates an initial reward from prior knowledge, then **iteratively refines it with
simulator-in-the-loop evaluation** — structurally our loop, in an engineering rather than
manipulation domain. Abstract: best configuration reaches goodput within ~3 % of the expert-designed
baseline with slightly lower end-to-end delay; reward quality improves "within a few iterations".
Their honest result is *parity with a human expert reward*, the same posture as our +0.4/+0.5 J.

**[L5] J. Chen, Shu, H.-N. Li — "Benchmarking Performance Judgment in Open-Weight LLM Controller
Tuning: Control Knowledge Does Not Ensure Reliable Gain Ranking", Processes 14(18):2954, 2026,
DOI 10.3390/pr14182954.**
VERIFIED (bib only) — MDPI 403s; full abstract read from
`https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/pr14182954`.

**The strongest available justification for our deterministic-simulation verification loop.**
Simulator-grounded benchmark, four open-weight checkpoints, quadruple-tank + recycle reactor +
synthetic 3x3 cyclic plant. From the abstract: declarative control-theory accuracy 98–100 %, but
**accuracy at ranking PI gain sets is 46.8 %, 31.8 % and 53.2 %**; adding equations and 14-point
response trajectories raises pooled accuracy only from 46.8 % to 51.2 % / 51.0 %; **all 12
prespecified trajectory-vs-qualitative intervals include zero and none survives Holm correction**;
swapping displayed trajectory values changes 9.5 % of parsed choices; a numerical scaffold raises
evidence-consistent accuracy from 51.1 % to 88.1 % (11/12 contrasts survive Holm). Closing sentence:
"Simulator-based validation remains necessary before LLM judgments are used for controller tuning."
Sanity checks in the same abstract: IAE and ISE rankings agree on 95.2 % of 271 replay-stable pairs;
88.7 % of seed–item labels preserved under a combined simulator perturbation.

**[L6] Shou, Hong, Ren, J. Wang, Yang, Liao — "A Physics-Informed Framework for PID Tuning of
Chemical Processes Using Large Language Model Agents", arXiv:2607.26594, 29 Jul 2026.**
VERIFIED — `https://arxiv.org/abs/2607.26594`.
LLM agent iteratively proposes PID gains from closed-loop response features and IMC demonstrations,
with **simulation-verified IMC targets** for SFT and a physics-informed RL stage (PI-GRPO). Plants:
100 FOPDT + 100 SOPDT cases. Abstract: hosted LLMs 75–89 % and 77–79 % final success; fine-tuned
Qwen3-0.6B 86.5 % first-recommendation success with SFT alone; 94.0 % first-attempt with PI-GRPO.
"LLM in the control-design loop with simulation verification" in an industrial plant setting; also
evidence that raw hosted LLMs are the weak point, echoing [L5].

### 2.2 Related but different

**[L7] Z. Zhang, Shen, Wan, Song, M. Sun — "LLM-Guided Safe Reinforcement Learning for Energy System
Topology Reconfiguration", arXiv:2603.14018v1, 14 Mar 2026 (the page states Applied Energy).**
VERIFIED — `https://arxiv.org/html/2603.14018`.
The LLM does **not** design rewards: it is a *transition-refinement* operator, triggered on
replay-buffer transitions with r_t < r_thr = 0.3, proposing alternative topology actions that are
then **validated in the Grid2Op simulator before storage** (§3.4); Safety-SAC dual-critic
architecture (§3.3). Results: IEEE 36-bus reward 3.088e4 vs SAC 1.643e4, survival steps 1028 vs
518.3, safety cost 0.880 vs 2.578; IEEE 118-bus reward 4.2898e5 vs SAC 1.1018e5, survival 357.4 vs
91.9, overload rate 0.418 % vs 1.033 %. The leading "LLM + RL in a power system with a simulator
gate on every LLM proposal" citation.

**[L8] Nosrati, Tepljakov, Belikov, Petlenkov — "When control meets large language models: From
words to dynamics", arXiv:2602.03433 [eess.SY], 3 Feb 2026.**
VERIFIED — `https://arxiv.org/abs/2602.03433`.
Survey / position paper; taxonomy of LLMs (i) assisting controller design and synthesis,
(ii) augmenting research workflows, (iii) treated as dynamical systems in state-space frameworks.
The single "LLMs and control theory" framing citation for a TCST introduction. No experiments.

**[L9] Bigdeli, J. Zhou, Bak — "Large Language Models as Falsifiers for Cyber-Physical Systems",
arXiv:2609.20752v1, 17 Sep 2026.**
VERIFIED — `https://arxiv.org/html/2609.20752`.
LLM proposes candidate input vectors; **each candidate is verified by running the CPS simulator and
an STL monitor**, with the robustness value fed back into the prompt history — an
LLM-proposes / simulation-decides loop in a safety-critical control setting. Results: highest rank on
14 of 21 specifications against eight ARCH-COMP 2025 tools; on six specifications a counterexample
found in a single simulation in every run; for AT1, 1.0 mean simulations vs FReaK 4.8, with uniform
random failing to succeed. Doubles as a theme-3 entry (LLM proposer beating numerical optimisers and
uniform random under a simulation budget, in control).

**[L10] Guo, C. Li, Feng, C. Zhang — "Code Evolution for Control: Synthesizing Policies via
LLM-Driven Evolutionary Search", arXiv:2601.06845v1, 11 Jan 2026.**
VERIFIED — `https://arxiv.org/html/2601.06845v1`.
LLM evolves **policy code**, not rewards; fitness = average reward over episodes (§III-C). Benchmark
is LunarLander-v3 only (§IV-A). Random baseline -200±50 / 0 % success; PPO at 1M steps 214±27 / 60 %;
best LLM method (EvoEngineer+) 143.6 / 70 % (§V-A); EvoEngineer reported as 7x over FunSearch and
EoH. Toy benchmark — cite only as adjacency.

---

## 3. LLM proposers vs random search / classical optimisers

**[H1] Y. Li — "Greedy Is a Strong Default: Agents as Iterative Optimizers", arXiv:2603.27415,
28 Mar 2026.**
VERIFIED — `https://arxiv.org/abs/2603.27415`.
The companion to arXiv:2606.21641, cutting the other way: it asks whether classical optimisation
machinery still helps once the proposer is an LLM. Abstract: simulated annealing, parallel
investigators and a second LLM (OpenAI Codex) give **no benefit over greedy hill climbing while
requiring 2–3x more evaluations**. Four tasks: Breast Cancer rule-based classification 86.0 to
96.5 % test accuracy; MobileNetV3-Small on STL-10 84.5 to 85.8 % with **zero catastrophic failures
vs 60 % for random search**; LoRA on Qwen2.5-0.5B/SST-2 89.5 to 92.7 %, "matching Optuna TPE with 2x
efficiency"; XGBoost/Adult AUC 0.9297 to 0.9317, "tying CMA-ES with 3x fewer evaluations".

Bearing on us: our result — LLM proposer ≈ random search in the hyper-parameter namespace, while the
LLM *reward structure* is the arm that holds all four terms, and is *not* distinguishable from a
random structure on the strongest base (s40) — sits between arXiv:2606.21641 (LLM adds nothing in
HPO) and this (the LLM prior is strong enough that the search wrapper adds nothing). Both support
the positioning that **the namespace, not the proposer, is where the agent earns its keep**.

**[H2] B. Zhang, C. Wang, K. Wu — "EVOM: Agentic Meta-Evolution of Actor-Critic Architectures for
Reinforcement Learning", arXiv:2606.26327, 24 Jun 2026.**
VERIFIED (abstract only) — `https://arxiv.org/abs/2606.26327`.
The abstract states EVOM outperforms a manually designed baseline, **an LLM-guided random search**,
and the LLM-guided programmatic policy search method MLES, on Ant-v4 and HalfCheetah-v4. **The
abstract gives no numbers and does not state whether the random-search control is budget-matched**;
the body was not read. Structurally the right shape of control (same LLM generator, mutation and
selection removed) — our `random_hparam` and random-reward-structure arms.

**[H3] S. Wu, Ye, Xiang, P. Chen, Xiong, T. Chen — "LLMSYS-HPOBench: Hyperparameter Optimization
Benchmark Suite for Real-World LLM Systems", arXiv:2605.08305, 2026.**
VERIFIED — `https://arxiv.org/abs/2605.08305`.
364,450 configurations, dimensionality 12–23, 3–5 fidelity dimensions (932 settings), 3–9 inference
objectives, 2–10 cost metrics; calls for "a revalidation of the existing HPO algorithms". **It
reports no LLM-vs-random-search comparison in the abstract** — infrastructure, not the control study
we wanted. Low priority.

---

## 4. UNVERIFIED leads — no numbers quoted, do not cite as they stand

- **RL-based pitch control for wind turbines using double deep Q-networks**, PubMed 41539907
  (appears to be early 2026). `https://pubmed.ncbi.nlm.nih.gov/41539907/` served a cookie wall and
  was not read. Search snippets claim a direct comparison against **ROSCO** with a reduction in
  power fluctuations at comparable structural loads, plus a reward function improving on a recent
  formulation. **If real this is a direct competitor to our GSPI/ROSCO comparison and must be
  retrieved (publisher DOI or OSTI) before the related-work section is finalised.** No figure from
  it is quoted here.
- **GUIDE: A Conversational LLM Framework for Dynamic DRL-Based Building Energy Management**,
  Springer chapter DOI 10.1007/978-3-032-22500-9_2 — LLM translating natural-language commands into
  a DRL reward function, building energy. Not fetched.
- **EvoNav: Evolutionary Reward Function Design for Robot Navigation with LLMs**, arXiv:2605.11859 —
  LLM mutation/crossover over reward functions. Not fetched; likely lower priority than [L3].
- **ThermoLLM**, arXiv:2606.22911, and **LLM-Enhanced MARL for P2P Energy Trading**,
  arXiv:2507.14995 — LLM + RL in energy, roles unverified.
- **"Discovering Control Scheduler Policies Through Reinforcement Learning and Evolutionary
  Strategies"**, Actuators 14(12):604, 2025 — MDPI 403'd. Search text suggests it learns a scheduler
  that selects a controller from system state, which would be a §1.1 gating hit. Worth one
  retrieval attempt via the DOI or Semantic Scholar.

---

## 5. What this means for the manuscript

1. **Gated / regime-switched residual RL is not novel as a mechanism.** [R1] already pairs residual
   policy learning with an activation mechanism driven by a learned regime model, and [R2] already
   gates a residual on an NMPC + observer base with a state-dependent bound and an ISS certificate.
   The gate must be claimed as a *design choice validated in a new domain with a physically
   interpretable exogenous gating variable* — not as a new mechanism. Defensible deltas: the gating
   variable is the exogenous input that defines the operating region (wind speed); the base is a
   strong model-based controller evaluated *as the base*; the objective is the four-term
   regulation/fatigue metric rather than tracking error.
2. **LLM-written rewards in energy and control now exist.** [L1] tunes reward *weights*, [L4]
   reaches only parity with an expert reward, [L2] and [L3] stay on simulation benchmarks. The
   differentiator — the agent changing reward *shape* in a way weight search demonstrably cannot
   reach — still stands, and is the sentence the manuscript must make load-bearing.
3. **The "LLM proposer ≈ random search" result is now a small literature** ([H1], arXiv:2606.21641,
   and the control-side caution in [L5]). Reporting it as a negative in the ablations is current and
   defensible, not a weakness; [L5] additionally gives a statistically careful, citable reason why
   deterministic simulation verification must sit inside the loop — which is exactly what our
   verification step does.
