# Roadmap v2 — switching the objective from F to the baseline paper's metric set (2026-09-11)

## 0. Why (the evidence that forced this)

Every number in the project so far is scored by `F = tower-DEL reduction − constraint penalties`.
Two findings of 2026-09-10 make that objective the wrong lens for the question the paper asks:

1. With the critic fixed (`--value_norm`), every agentic lever is ≤ 0 on F (weights −1.33, PPO
   hyper-parameters −0.15, reward code −2.65 vs the fixed-weight baseline; n = 1–2).
2. Re-scoring the same policies under the baseline paper's four metrics (power MSE, generator-speed
   MSE, tower DEL, blade DEL — % reduction vs paired GSPI) **inverts the ranking**
   (`scripts/dev/rescore_metric_set.py`, held-out S3–S6, best-by-F checkpoint):

| arm (fixed critic) | F | J (equal weights) | power MSE | speed MSE | tower DEL | blade DEL |
|---|---|---|---|---|---|---|
| LLM-written reward (n=1) | 15.35 | **18.90** | 28.3 | 28.3 | 15.4 | 3.6 |
| LLM hyper-parameters (n=2) | 18.06 | **13.61** | 15.1 | 15.1 | 18.1 | 6.3 |
| fixed weights (n=2) | **18.20** | 8.35 | 4.3 | 4.3 | 18.2 | 6.6 |
| LLM reward weights (n=2) | 16.87 | 8.14 | 5.2 | 5.2 | 16.9 | 5.2 |
| LPV-MPC (F-tuned) | −17.6 | −1.6 | 7.5 | 4.2 | −17.6 | −0.7 |

The two arms that F called noise both act on the Region-3 regulation loop (a steeper speed reward;
a longer credit window and a faster critic). F cannot see regulation once the speed constraint is
met; the paper's metric set is 50 % regulation. The same ranking holds with regulation counted once
(3-way weights: 15.8 > 13.1 > 9.7). **This is a preview on F-trained, F-selected policies** — it
shows the ranking moves, not what J-trained policies reach.

This is a post-hoc objective change and must be disclosed as such in the manuscript (§6).

## 1. The new objective J — definition and the decisions it needs

```
J = ( w_P·ΔMSE_power% + w_ω·ΔMSE_speed% + w_T·ΔDEL_tower% + w_B·ΔDEL_blade% ) / Σw
    − k_E · max(0, energy_loss% − 1)
```
computed at the **aggregate** level exactly as `fitness()` already does: MSE terms over the
R3-dominated evaluation episodes (frac_R3 ≥ 0.5), DEL terms over all episodes, energy over all.

| # | decision | recommendation | why |
|---|---|---|---|
| D1 | weights | **equal (1,1,1,1)** as primary, (½,½,1,1) as sensitivity | equal = the paper's metric set with no editorialising; but power ≡ speed MSE under constant torque, so equal weights are effectively 50 / 25 / 25 — state that. Ranking is identical under both. |
| D2 | energy | keep as a **penalty**, k_E = 20 per % over 1 % | the paper has no energy concept (constant torque, no R2); dropping it lets R2 shed power for load. Same convention as F. Report energy alongside. |
| D3 | R3 label for the MSE terms | **wind-based (v_hub > rated) for every row** | controller-independent; removes the C3 asymmetry (RL rows oracle-labelled, MPC/TD wind-labelled). The oracle label stays for *routing*, only the metric subset changes. |
| D4 | per-term clipping | clip each % reduction to ±100 before averaging | a single blown MSE episode (the −409 % artefacts of roadmap §16) must not own the average |
| D5 | tiers | **none**. Report J, the 4-vector and energy as continuous quantities; a boolean "energy ≤ 1 %" only | speed is now an objective, not a constraint; tier counts were shown non-robust anyway (C11) |

Guardrail without tiers: roll back when `J < best_J − 5` (the pre-existing `--rollback_on drop`
mode) **or** energy loss > 1 % (hard, physical). Best-checkpoint = best J on the supervisor's winds.

## 2. Reward v2 — aligning the per-step signal with J

The current reward's Region-3 term `w_ω·exp(−|δω|/0.02)` is nearly flat at the typical error
(|δω| ≈ 0.005) — E1 diagnosed this in August and the LLM re-discovered it on 2026-09-10 (it moved
the scale to 0.004). Under a regulation-weighted objective the fixed-weight baseline must not be
handicapped by a flat term, or every agentic "win" is again the repair of a bad default:

```
r_t = 1[R2]·w_P·(P/P_base − 1)                       energy honesty in R2 (unchanged)
    − 1[R3]·w_ω·(δω / 0.005)²                        quadratic = the instantaneous speed-MSE contribution
    − λ_T·range_inc(fa_acc) − λ_B·range_inc(M_oop)   BOTH load proxies (tower and blade DEL are both in J)
    − λ_A·(Δβ/κ)²                                     unchanged
```

Knob namespace for the weight supervisor becomes `{w_power, w_speed, lambda_tower, lambda_blade,
dbeta_max_R2, dbeta_max_R3}` (still six). `--value_norm` becomes the default. The old reward stays
selectable (`--reward v1`) for a continuity control (§4, stage 3).

Decision D6: run the fixed-weight baseline with **both** rewards (v1 old, v2 aligned). The gap
between them is the answer to "how much of the agentic gain was a better hand-designed default" —
the crux of the whole project — and costs 5 runs.

## 3. What changes in the code (by file)

| file | change |
|---|---|
| `eval/fitness.py` | add `J`, `J_terms`, `energy_ok` to the output (keep F/F_tol2/tier for old runs); `--metric_weights`; wind-label option for the R3 subset |
| `scripts/train.py` | `--objective {F,J}` → selection key for forks and best-ckpt, rollback rule (J-drop or energy), `record_eval`, the `[eval]`/`[sup]` prints; `--supervisor random_hparam` (random candidates in the hyper-parameter namespace = the control for `llm_hparam`); `--reward {v1,v2}` |
| `llm/supervisor.py` | the objective paragraph in all three prompts (weights / hparam / reward-code) rewritten for J; `evaluation_now` headlines `J` and the four terms, drops `tier`; diagnostics paragraph: "speed MSE is now an objective" |
| `envs/reward.py` + `envs/base_env.py` | second load-proxy buffer (tower + blade simultaneously); quadratic speed term; `lambda_tower/lambda_blade` knobs; expose both to the reward-code namespace (`load_t`, `load_b`) |
| `configs/reward.yaml` | v2 defaults; `lambda_tower`, `lambda_blade`, `speed_err_ref: 0.005` |
| `scripts/evaluate.py` | print J + the 4-vector; `--relabel_wind` default on under J |
| `scripts/mpc_baseline.py`, `campaign_a2_towerdamper.sh` | model selection by J |
| `scripts/dev/{stats_table,heldout_table,paper_table,build_tables}.py` | J column; `stats_table --key J` |
| `scripts/dev/rescore_metric_set.py` | done (this preview) |

Everything above is additive: old runs stay readable and F stays computable.

## 4. What is reused and what must be re-run

**Reused as-is**: wind banks (S1–S10, stress classes), paired GSPI baselines, the identity checks,
all existing evaluation JSONs (they already carry the 4 metrics → F-tables and the preview),
the MPC/TD sweep machinery, the critic diagnostics.

**Must be re-run** (trained *and* checkpoint-selected under J):

| stage | content | runs | wall |
|---|---|---|---|
| 1 | implement §2–§3, toy smoke of every arm, one 300-ep OpenFAST smoke | — | ½ day |
| 2 | baselines under J: MPC re-tune (r × wc_v × qt on S1+S2 by J), tower damper re-sweep by J, GSPI identity | eval only | 1 h |
| 3 | **core**: guard-v1, guard-v2, llm_hparam, random_hparam, llm_reward — **3 seeds each as proof of concept** (decision 2026-09-11; extend to 5 if the ranking holds) | 15 | ~10 h |
| 4 | secondary, 5 seeds: llm weights (v2), mono (v2) | 10 | ~6.5 h |
| 5 | second held-out set S7–S10 + stress classes on every stage-3/4 arm | eval only | ~1.5 h |
| 6 | statistics (paired, both wind sets, exact floor), tables, docs | — | ½ day |

Stage 3 order (two lanes, best information first): guard-v2 → llm_hparam → random_hparam →
guard-v1 → llm_reward. After guard-v2 + llm_hparam (10 runs, ~6.5 h) the central question already
has an n = 5 answer.

## 5. Risks and how each is handled

| risk | handling |
|---|---|
| post-hoc objective change | disclose; keep the F results as a secondary table; report both weightings (D1) |
| regulation double-counted | state the effective 50/25/25 split; show the 3-way sensitivity |
| MSE terms are noisy (±5–10 % seed spread) and R3-only | n = 5 minimum, both wind sets, per-term clipping; never quote a single-seed J |
| MPC gets stronger under a regulation-weighted J | re-tune it under J with the same rule; if it wins on J, that is the result |
| supervisor over-fits its own winds (llm_reward: 23.5 on S1+S2 → 15.4 held-out under F) | keep 2 supervisor seeds; consider 3 for the reward-code arm; report S1+S2 vs held-out gap per arm |
| energy creeps toward the 1 % edge (agentic arms sit at 0.5–0.8 %) | k_E penalty + hard rollback at 1 %; report energy in every table |
| a reward-code win that is really a scale fix | `critic_health.py` on every arm; the guard-v2 control already has the aligned scale |
| LLM API failure kills a run | fixed 2026-09-10: supervisors degrade to "hold" |

## 6. What the manuscript becomes if the preview holds at n = 5

Primary claim: under the baseline paper's own metric set, agentic supervision of the *learner*
(hyper-parameters) and of the *reward* improves regulation by 15–28 % at no load cost, where
supervision of the reward *weights* does not — and the mechanism is the credit-assignment scale
(§21–§23 of the 08-30 roadmap; critic finding; window vs rotor period). Secondary: the F results,
showing that the same levers are invisible to a load-priority objective — the objective decides
which lever "works". Controls that must be in the table: random_hparam (proposer vs search),
guard-v2 (agentic vs a good hand-designed default), MPC and the tower damper re-tuned under J.

If the preview does **not** hold at n = 5, the honest paper is the scaling story (roadmap §21–§23
plus the critic finding), with the objective sensitivity as a cautionary section.

## 7. Stage 1 as built (2026-09-11)

Everything in §3 is implemented and smoke-tested (toy: every arm; OpenFAST: guard-v2 with a
kill-free resume and a held-out evaluation). Details that differ from or refine §1–§3:

- **D4 is per-episode.** `fitness()` clips each episode's % reduction to ±100 *before* averaging
  for the four `J_*` terms (`J_power_mse_red_pct`, …); the unclipped means keep their historical
  names (`power_mse_red_pct`, …) so the F-era tables are unchanged. The per-episode reductions are
  stored in `per_episode` of every new eval json.
- **Model selection** everywhere is the tuple `(energy_ok, J)`: fork candidates, best checkpoint,
  the MPC grid and the tower-damper sweep (`scripts/dev/pick_by_J.py`). Rollback under J: J drops
  more than 5 below the best, or energy loss > 1 %. `ckpt_best.pt` and `summary.json` record
  `objective`; `evaluate.py` switches to wind-labelled R3 subsets automatically for J runs.
- **`random_hparam`** = `RandomCandidateSupervisor` in the hyper-parameter namespace (same fork
  verification as `llm_hparam`).
- **Pausable / resumable training.** `train.py` writes `resume_<episode>.pt` at wave boundaries
  every `--resume_every_s` (300 s; `--resume_keep 5`) with learners (incl. Adam moments and the
  return normaliser), knobs, best state, history, bookkeeping rows, supervisor state (RNG /
  schedule index / LLM call counters) and the reward source; `--resume` continues from the newest
  one and truncates `decisions.jsonl` to that episode. `scripts/wsl/campaign_ctl.sh
  {pause|resume|stop|status}` acts on a whole campaign session (SIGSTOP/SIGCONT/SIGTERM); a
  stopped campaign re-launched with the same script continues every unfinished run.
- **Campaign scripts:** `campaign_j_baselines.sh` (stage 2: identity, MPC grid r×qt×wc_v on S1+S2
  under the current wind-labelled pairing, tower-damper re-pick from the existing sweep, both
  held-out sets for each winner) and `campaign_j_core.sh` (stage 3, two lanes, information-ordered,
  held-out S3–S6 + S7–S10, critic health, paper table).
- Historical MPC tuning jsons (`eval_tune2/3_*`) predate the wind-labelled baseline pairing and
  read −100 % on power MSE under J; they are not reused — the grid is re-run (`tuneJ_s12`). The
  existing tower-damper sweep was already wind-labelled on both sides and is re-picked as is
  (J-best: FA_KI = 0.001, HPF = 0.172, J = 0.58 on S1+S2 — i.e. the damper has nothing to offer
  under J either; the held-out rows confirm or refute that in stage 2).

## 8. Stage 2 results — reference controllers under J (2026-09-11, `campaign_j_baselines.sh`)

| controller | selection on S1+S2 (by `(energy_ok, J)`) | S3–S6 J | S7–S10 J | notes |
|---|---|---|---|---|
| GSPI identity | — | **0.00** | — | zero-residual controller through the evaluation path: every term 0.0 |
| LPV-MPC | N=20, q=1, **r=0.02, qt=0, wc_v=0.35** (J = 0.96 on S1+S2) — the same point F selected | **−1.64** (P +7.5, ω +4.2, tower −17.6, blade −0.7) | **−8.08** (P +3.2, ω −10.3, tower −22.1, blade −3.1) | 24-point grid r × qt × wc_v; every qt > 0 point loses ~71 % energy (rotor stall, as in §16), r ≤ 0.01 loses on all four terms |
| ROSCO tower damper | **FA_KI = 0.001, HPF = 0.172** (J = 0.58) — the smallest gain in the sweep | **−0.06** (P −0.3, ω −0.5, tower +0.7, blade −0.0) | **−0.26** (tower +1.7, blade −1.8) | any larger gain trades regulation for tower DEL at a loss under J (KI 0.03: J −7.9; KI ≥ 0.1: −60 and worse) |

Reading: under the paper's metric set neither reference controller improves on GSPI — the MPC's
regulation gain on the first wind set (+7.5 / +4.2 %) does not survive the second set (ω −10.3 %)
and is paid for with −18 to −22 % tower DEL in both; the damper at its J-best gain is GSPI. The
bar for the RL arms in stage 3 is therefore J > 0 on both wind sets with energy ≤ 1 %, not "beat
the MPC". (MPC held-out rows are the wind-labelled `heldoutW` / `heldout2W` evaluations; the
oracle-labelled `heldout_s3456` files of the MPC are not comparable under J and are not used.)

### 8b. Stage 2b — the MPC's tower term was a scaling artefact (queued after stage 3)

Under J the regulation-only MPC is a single-objective controller scored on a multi-objective
metric, and the reason its tower term never worked (§16 of the 08-30 roadmap: any qt > 0
feathers the rotor, energy −71 %) is the same class of bug as the flat speed reward and the
unreachable value target: with `q·((ω−ω_r)/ω_r)²` the speed term is O(10⁻⁵) at the typical
error (|δω|/ω_r ≈ 0.005) while `(ẋ/0.2)²` is O(1), so qt = 0.3 out-weighed regulation by four
orders of magnitude. `controllers/mpc.py` now takes `err_ref`, `dbeta_ref`, `xd_ref`;
`mpc_baseline.py --scale v2` sets err_ref = 0.005, dbeta_ref = 0.002 rad/step so all three terms
are O(1) at typical values (the v1 optimum r = 0.02 maps to r ≈ 0.3 in v2; verified to reproduce
the same pitch trajectory on a synthetic operating point). `campaign_j_mpc2.sh` sweeps
N ∈ {20, 40} × r ∈ {0.1, 0.3, 1, 3} × qt ∈ {0, 0.1, 0.3, 1, 3} × wc_v ∈ {0.25, 0.35} on S1+S2 by
`(energy_ok, J)` and evaluates the winner on both held-out sets (tags `heldoutJ2_*`). No wind
preview is added (the RL has none either).

**Result (11:32).** 82 grid points; J-best on S1+S2 is N=20, r=0.3, **qt=3**, wc_v=0.35 (J = 17.24;
P +18.4, ω +28.0, tower +22.0, blade +0.5). N=40 is worse everywhere (≤ 9.1); qt=1 regulates
harder but leaves the tower at +1…+12 %; qt=3 is the point that has both. Held-out:

| MPC (cost scale v2, qt=3) | J | power MSE | speed MSE | tower DEL | blade DEL | energy | F |
|---|---|---|---|---|---|---|---|
| S3–S6 | **12.40** | +8.9 | +19.8 | +15.8 | +5.1 | −0.02 % | 15.79 strict |
| S7–S10 | **14.10** | +12.7 | +26.3 | +18.3 | −0.9 | −0.07 % | 18.30 strict |
| regulation-only MPC (v1 scale), S3–S6 / S7–S10 | −1.64 / −8.08 | +7.5 / +3.2 | +4.2 / −10.3 | −17.6 / −22.1 | −0.7 / −3.1 | −0.3 % | −17.6 / −22.1 |

So the §16 baseline statement "the MPC wins speed regulation and loses tower DEL (−17.6 %)" was
an artefact of the cost scaling, not a property of model-predictive pitch control: with its tower
term active the same 3-state LPV-MPC is positive on every term of J on both wind sets, beats every
stage-3 RL arm by ≈ 5 J (hparam arms 7.3 / 8.4–9.0), and under the old objective it is strict with
F 15.8 / 18.3 — on par with the F-era llm_fork (17.9 / ~16). It also improves energy slightly. The
RL arms' remaining edge is nil on J; the manuscript's comparison table must carry this row as the
model-based reference, and the residual-RL claim has to be made against it, not against GSPI.

## 9. Stage 3 results — core arms under J, 3-seed proof of concept (2026-09-11, final at 13:57)

All runs: `--objective J --reward v2 --value_norm` (guard-v1: `--reward v1`), 300 episodes, best
checkpoint by J on S1+S2, held-out on both disjoint wind sets (`scripts/dev/j_table.py`). The
`llm_reward` rows are the clean re-run after `908da1c` (its first three runs had every candidate
rejected by a v1-namespace compile and most replies truncated at 2000 tokens).

| arm | n | J on S1+S2 (train-time best) | **J on S3–S6** | **J on S7–S10** | S3–S6 terms P / ω / T / B |
|---|---|---|---|---|---|
| guard-v2 (fixed hparams, reward v2) | 3 | 3.07 ± 1.94 | 1.59 ± 0.87 | 0.97 ± 1.37 | −6.9 / −0.6 / 10.4 / 3.4 |
| guard-v1 (reward v1) | 3 | 2.64 ± 1.59 | 1.24 ± 1.02 | 0.42 ± 1.33 | −7.1 / −2.0 / 10.9 / 3.2 |
| **llm_hparam** | 3 | 10.69 ± 1.60 | **7.27 ± 1.70** | **8.98 ± 2.23** | 7.0 / 12.4 / 6.6 / 3.1 |
| **random_hparam** | 3 | 10.98 ± 3.51 | **7.40 ± 2.65** | **8.42 ± 3.65** | 6.3 / 14.1 / 5.1 / 4.1 |
| llm_reward (Eureka-style expression) | 3 | 7.19 ± 2.99 | 5.05 ± 2.02 | 6.00 ± 1.90 | 1.6 / 7.2 / 8.1 / 3.3 |
| **LPV-MPC, tower term active (§8b)** | — | 17.24 | **12.40** | **14.10** | 8.9 / 19.8 / 15.8 / 5.1 |
| LPV-MPC regulation-only (§8) | — | 0.96 | −1.64 | −8.08 | 7.5 / 4.2 / −17.6 / −0.7 |
| ROSCO tower damper (§8) | — | 0.58 | −0.06 | −0.26 | −0.3 / −0.5 / 0.7 / 0.0 |

Seed-paired differences of J (exact sign-flip floor 2/2³ = 0.25 at n = 3):

| comparison | S3–S6 | S7–S10 |
|---|---|---|
| llm_hparam − guard-v2 | +5.68, 3/3 seeds, t-test p = 0.055 | +8.01, 3/3, p = 0.062 |
| random_hparam − guard-v2 | +5.81, 3/3, p = 0.048 | +7.45, 3/3, p = 0.076 |
| llm_reward − guard-v2 | +3.46, 3/3, p = 0.107 | +5.03, 3/3, p = 0.059 |
| llm_hparam − random_hparam | −0.13, p = 0.95 | +0.56, p = 0.81 |
| guard-v1 − guard-v2 | −0.35, p = 0.72 | −0.55, p = 0.60 |

What the reward-code arm did: all three best checkpoints run an LLM-written expression (kept in
`ckpt_best.pt["knobs"]["reward_code"]`), e.g. seed 0 (S1+S2 J 10.51 at ep 280):
`180·tanh(15(p_ratio−1))` in R2, `−40·sqrt(tanh(|δω|/0.005)² + tanh(60|p_ratio−1|)²)` in R3,
`−0.9·tanh(log((2+load_t+load_b)/4)) − 0.1·tanh(act)` — a saturated, power-tracking-aware R3 term
plus a compressed joint load term. Its trajectories still show the guard's 12.5 m/s regulation
collapse and rollbacks; the rewrite raises the best point (+3.5 / +5.0 J over guard) without
removing the dynamics, and it stays ≈ 2–3 J below the hyper-parameter arms.

Findings:
1. **Under the paper's metric set the fixed-hyper-parameter residual is worth ≈ 1 J** — a
   +10 % tower-DEL gain paid with −7 % power MSE — and the reward version (v1 / v2) does not change
   that. Every guard seed shows the same training dynamics: an early best (ep 32–96), then a
   rollback at every later decision because the policy keeps re-entering a state that regulates
   worse in the 12.5 m/s episodes (speed-MSE −70…−110 % there, tower +20 %).
2. **Supervising the learner's hyper-parameters is worth +6…+8 J in every seed on both wind
   sets** (energy 0.04–0.19 %). The accepted candidates lengthen the credit window (gae_lambda
   0.98 → 0.992–0.995) and slow the actor (lr, clip, policy std); the best-J policies trade part of
   the tower gain (6 vs 10 %) for regulation (+7 / +12…+14 % power / speed MSE), which J prices 2:1.
3. **The proposer does not matter for this lever**: random candidates verified by the same fork
   search are indistinguishable from the LLM's (−0.1 / +0.6 J, p ≈ 0.9). This is the mirror image of
   the F-era reward-weight result (roadmap 08-30 §21: LLM > random 7/7). Read together: the LLM's
   value showed where the search space was *semantic* (reward weights: which term to move) and
   vanishes where a 6-D log-uniform perturbation plus verification already finds the answer.
4. **Rewriting the reward is the weakest of the three agentic levers under J** (+3.5 / +5.0), and
   it does not touch the learning dynamics that cap every RL arm.
5. **The model-based reference, once its tower term works, is the best controller in the table**
   (12.4 / 14.1, every term positive, energy ≤ 0, F strict 15.8 / 18.3). The regulation-only MPC
   and the tower damper are below GSPI. Ranking under J: MPC-v2 ≫ hparam arms > llm_reward >
   guard ≈ GSPI > damper > regulation-only MPC.

Consequence for the manuscript (replaces §6): the residual-RL claim cannot be "beats the model
baseline". What the data supports: (i) the objective decides which lever works (F: reward weights
and the proposer; J: the learner's hyper-parameters, proposer-agnostic) — the scaling / credit-window
story of the 08-30 roadmap §21–23 and the critic finding; (ii) a 3-state LPV-MPC with a correctly
scaled multi-objective cost is a strong, cheap (3.6 ms/solve) controller on the paper's own metrics;
(iii) the gap RL-to-MPC (≈ 5 J) is the open problem, and the guards' rollback loop names its
mechanism. Whether to close (iii) before writing — same hyper-parameter search plus the MPC's
knowledge (tower velocity in the reward or the observation), longer training, n = 5 — is the next
decision.

## 10. Fairness of the RL-vs-MPC comparison (2026-09-11, user decision: make it fair before writing)

The stage-3 table is not a like-for-like comparison. Asymmetries, verified in code:

| dimension | RL residual arms | MPC-v2 | favours |
|---|---|---|---|
| objective weights | reward weights hand-set (w_speed 20, λ_T = λ_B = 1), never tuned under J; the hparam arms tune PPO only | 82-point cost grid selected by J on S1+S2 | MPC |
| actuation authority | residual ±0.05 rad (2.9°) on the GSPI command, through a 2nd-order damper | replaces ROSCO's pitch PI, ±0.35 rad (20°), no damper | MPC |
| model knowledge | none | the simulation's own Cp/Ct tables + hand-set tower mode (0.324 Hz, 437 t) | MPC |
| 12.5 m/s transition | R2 residual pitches for tower load; wind-labelled J counts it as regulation loss (the guards' collapse) | rides the pitch floor below rated (= GSPI there) | MPC |
| wind information | observes simulator-truth v_hub | ROSCO's wind estimate, low-passed | RL |
| seeds | 3, best checkpoint | deterministic | neutral (held-out on both) |

Plan (in order; each step is a queued campaign):
- **Step 0** `campaign_j_mpc_res.sh`: the MPC through the RL's residual channel (`mpc_baseline.py
  --residual`: ±0.05 rad, damper on), 24-point grid under J, both held-out sets → how much of
  12.4 / 14.1 is authority.
- **Step 1** reward **v3** = v2 with the speed term gated by the wind label (`region_w`, the subset
  J's MSE terms use) instead of the router's region — under v2 the transition steps J scores as R3
  carried no speed penalty at all. Arms `jg3` (guard, v3) and `jwf3` (random_fork in the weight
  namespace, v3, 3 candidates) × 3 seeds: does the gating alone remove the rollback loop, and what
  does a J-tuned weight vector look like.
- **Step 2** the agentic arms re-run on the tuned default (`jhp3`, `jrhp3`, `jrw3`; `llm_reward`
  now sees `region_w` in its namespace) — the agentic effect sizes reported in the paper are the
  ones measured here, not §9's.
- Stated, not equalised: the MPC's perfect model; the RL's true-wind observation.

**Step 0 result (14:41).** Through the RL's channel (±0.05 rad, damper on) the MPC re-selects the
same weights (r = 0.3, qt = 3, wc_v = 0.35; S1+S2 J = 18.15, even higher than wide-open) and holds:

| MPC-v2 | J S3–S6 | J S7–S10 | S3–S6 terms P / ω / T / B |
|---|---|---|---|
| wide-open channel (±20°) | 12.40 | 14.10 | 8.9 / 19.8 / 15.8 / 5.1 |
| **RL residual channel (±2.9°, damped)** | **11.97** | **11.61** | 8.5 / 20.5 / 13.2 / 5.6 |

Authority is worth 0.4–2.5 J, mostly tower DEL on the second wind set; with identical actuation
the MPC still leads the best RL arm by ≈ 3–4.5 J. So the gap is not authority — it is what the
controller knows and optimises (a model with the tower state, weights selected on J), which is
what steps 1–2 give the RL.

**Step 1 result (23:21).** Reward v3 (speed term gated by the wind label) and the 6-knob weight
search, 3 seeds, held-out:

| arm | J S1+S2 | J S3–S6 | J S7–S10 | S3–S6 terms P / ω / T / B |
|---|---|---|---|---|
| guard-v2 (§9, for reference) | 3.07 ± 1.94 | 1.59 ± 0.87 | 0.97 ± 1.37 | −6.9 / −0.6 / **10.4** / 3.4 |
| **guard-v3** (`jg3`) | 7.40 ± 1.90 | 4.60 ± 2.38 | 5.96 ± 2.02 | 17.3 / 18.6 / **−18.0** / 0.5 |
| **random_fork weights, v3** (`jwf3`) | 8.12 ± 1.11 | 6.70 ± 2.12 | 7.89 ± 2.21 | 18.0 / 20.1 / **−13.2** / 1.9 |
| llm_hparam (v2, §9) | 10.69 ± 1.60 | 7.27 ± 1.70 | 8.98 ± 2.23 | 7.0 / 12.4 / 6.6 / 3.1 |

1. The gating **removes the rollback loop**: every v3 seed climbs through the run (best points at
   ep 184–300 instead of 32–96), and J rises by +3 / +5 over guard-v2 (0/3 and 1/3 seeds lose).
2. But it does so by **inverting the residual's trade**: regulation +17 / +19 %, tower DEL −13 to
   −22 %. Under J's 2 : 1 pricing that is net positive; as a load-reduction controller the residual
   is gone. The MPC gets both (+20 ω, +15 T) because its tower-velocity term is explicit and,
   after re-scaling, heavy (qt = 3 ≈ 170 % of the speed term at typical values); the RL's
   `range_inc` proxy at λ_T = 1 is nowhere near that.
3. The **weight search cannot find that region**: candidates with λ_T = 1.4–2.9 were proposed in
   every seed and never won a 30-episode fork — tower DEL responds over tens of episodes, the
   regulation gain within one fork. A myopic verification horizon is a structural limit of the
   fork search, and the reason the MPC's *grid* (full-run evaluation per point) is not the same
   tuning budget as the RL's *fork search*. The tuned default therefore keeps λ_T = 1
   (`configs/knobs_j_v3_tuned.json`: w_speed 25.2, λ_B 1.31, dbeta_R2 0.074, dbeta_R3 0.05).
4. Step 2 (agentic arms on that default) runs as planned; **added**: the RL analogue of the MPC's
   qt grid — guard-v3 from the tuned default with λ_T ∈ {3, 10, 30} (`jg3L*`, seed 0, selection by
   J on S1+S2, both held-out sets) — queued behind step 2. If a higher λ_T restores the tower gain
   at a small regulation cost, that is the RL's fair default; if not, the trade is intrinsic to the
   residual and the paper says so.

**Step 2 result (2026-09-12 07:41).** The agentic arms re-run from the J-tuned default (reward v3,
`configs/knobs_j_v3_tuned.json`), 3 seeds, held-out. **Correction (12:40):** the two hyper-parameter
arms of this table did *not* receive the tuned default — `--knobs_json` was applied to the reward
namespace and then discarded when the hyper-parameter namespace replaced it, so `jhp3t` / `jrhp3t`
ran on the untuned v3 weights (w_speed 20, λ 1, dbeta 0.05); their runs are renamed `jhp3u` /
`jrhp3u` and their fair comparison is against `jg3` (4.60 / 5.96): +3.6 / +3.9 (llm) and +2.8 / +2.4
(random). Fixed in train.py (`worker_fixed`); the tuned versions are re-run together with the
combined agent (§12). `jg3t` and `jrw3t` were tuned as stated.

| arm (reward v3, tuned default) | J S1+S2 | **J S3–S6** | **J S7–S10** | S3–S6 terms P / ω / T / B |
|---|---|---|---|---|
| guard (`jg3t`) | 8.22 ± 2.40 | 5.76 ± 1.76 | 6.81 ± 1.29 | 18.2 / 21.1 / **−17.3** / 1.1 |
| llm_hparam (`jhp3t`) | 10.76 ± 2.67 | 8.22 ± 2.20 | 9.89 ± 1.90 | 18.8 / 23.5 / −10.9 / 1.5 |
| random_hparam (`jrhp3t`) | 9.45 ± 0.78 | 7.37 ± 1.02 | 8.32 ± 0.80 | 16.5 / 20.5 / −9.9 / 2.3 |
| **llm_reward (`jrw3t`)** | 13.13 ± 1.44 | 8.30 ± 1.93 | 9.17 ± 1.13 | 9.3 / 15.7 / **+4.3** / **+3.9** |
| MPC-v2, residual channel (§10 step 0) | 18.15 | 11.97 | 11.61 | 8.5 / 20.5 / 13.2 / 5.6 |

Paired vs `jg3t` (n = 3, exact floor 0.25): llm_hparam +2.5 / +3.1 (2/3, 3/3 seeds; t p = 0.43 /
0.24), random_hparam +1.6 / +1.5 (2/3), llm_reward +2.5 / +2.4 (3/3; p = 0.30 / 0.20). Tuned
default vs untuned (`jg3t` − `jg3`): +1.2 / +0.9.

Reading:
1. **On a fair default the agentic effect sizes shrink** from +6…+8 (§9, against a default whose
   training collapsed) to +1.5…+3 J, below what n = 3 can establish. Most of §9's hyper-parameter
   gain was the repair of the rollback loop that the reward gating fixes by itself.
2. **The reward-code arm is the only RL arm with all four terms positive** (tower +4.3, blade
   +3.9, power +9.3, speed +15.7 on S3–S6; the same on S7–S10) — it equals llm_hparam on J with a
   qualitatively different solution. All three seeds' best expressions share one structure: a
   *saturated* speed term `tanh(|δω|/0.005)` on the wind-labelled subset (steep near zero where
   the quadratic is flat, bounded where the quadratic explodes), an explicit power-deviation term
   there, and **centred, bounded load terms `tanh(load − 1)`** that penalise only above-baseline
   load and cannot be swamped by the regulation term. That is the balanced trade the 6-knob weight
   search could not reach (§10 step 1, point 3): it is a change of *shape*, not of weights.
3. The hyper-parameter arms (both proposers) still buy regulation with tower DEL (−10 to −11 %);
   under J that is net positive, which is the objective's known 2 : 1 pricing.
4. Against the residual-channel MPC (12.0 / 11.6, every term positive, deterministic) the best RL
   arms are ≈ 2.5–3.5 J behind on the mean and match it in the best seed (jhp3t_s1 11.3 / 12.5,
   jrw3t_s1 10.3 / 10.6). The λ_T grid (`jg3L*`, running) tells whether a heavier tower weight
   alone gives the guard the MPC's balance.

**λ_T grid result (2026-09-12 11:43).** guard-v3 from the tuned default, seed 0, λ_T ∈ {1 (= `jg3t_s0`),
3, 10, 30}, held-out:

| λ_T | J S1+S2 | J S3–S6 | J S7–S10 | S3–S6 P / ω / T / B |
|---|---|---|---|---|
| 1 | 5.71 | 4.99 | 5.93 | 18.7 / 20.4 / −20.6 / 1.5 |
| 3 | 5.83 | 1.41 | 3.74 | 16.0 / 8.9 / −18.2 / −1.1 |
| 10 | 2.54 | −2.88 | −3.27 | 10.6 / −0.3 / −16.9 / −4.9 |
| 30 | 0.04 (never left the initial policy) | — | — | tower −2…−11 % during training, regulation negative |

A heavier tower weight **does not buy the tower back**: tower DEL stays at −17…−21 % up to
λ_T = 10 while regulation and J fall, and at λ_T = 30 the load-noise term dominates the reward and
learning stops. The `range_inc` proxy scaled up is not the MPC's tower-velocity cost — so the
balanced solution is unreachable on the *weight* axis of this reward family, and reachable on the
*shape* axis (the reward-code arm's saturated speed term and centred `tanh(load − 1)` load terms,
step 2 point 2). That closes the fairness campaign.

### Closing table of the fairness campaign (held-out S3–S6 / S7–S10, best-by-J checkpoints)

| controller | n | J S3–S6 | J S7–S10 | all four terms > 0? |
|---|---|---|---|---|
| **LPV-MPC-v2, residual channel (±2.9°)** | det. | **11.97** | **11.61** | yes |
| LPV-MPC-v2, wide-open (±20°) | det. | 12.40 | 14.10 | yes |
| llm_reward, v3, tuned default | 3 | 8.30 ± 1.93 | 9.17 ± 1.13 | **yes** (T +4.3, B +3.9) |
| llm_hparam, v3, tuned default | 3 | 8.22 ± 2.20 | 9.89 ± 1.90 | no (T −10.9) |
| random_hparam, v3, tuned default | 3 | 7.37 ± 1.02 | 8.32 ± 0.80 | no (T −9.9) |
| guard, v3, tuned default | 3 | 5.76 ± 1.76 | 6.81 ± 1.29 | no (T −17.3) |
| guard, v2 (§9) | 3 | 1.59 ± 0.87 | 0.97 ± 1.37 | no (P −6.9) |
| ROSCO tower damper | det. | −0.06 | −0.26 | ≈ GSPI |
| LPV-MPC regulation-only (v1 scale) | det. | −1.64 | −8.08 | no (T −17.6) |

`docs/tables/table_J_fairness.csv`, `table_J_stage3.csv`; rebuild with `scripts/dev/j_table.py --csv`.

## 11. What the manuscript can claim (2026-09-12)

1. **Primary**: on the baseline paper's own metric set, a 3-state LPV-MPC with a correctly scaled
   multi-objective cost is the strongest collective-pitch controller in this study (J ≈ 12 on two
   disjoint held-out wind sets, all four terms positive, 3.6 ms per solve), and it keeps that lead
   through the same ±2.9° damped residual channel the RL uses. The tower-DEL loss reported for MPC
   in the F-era tables was a cost-scaling artefact; the paper states it and shows the fix.
2. **Residual RL under J** reaches 8–10 (best seeds 11–12) only with (a) a critic whose value
   target is normalised, (b) a reward whose speed term is gated by the objective's own subset, and
   (c) one of the agentic levers; without (b) it collapses into a rollback loop, without (a) the
   critic is a constant. These are the mechanism results (08-30 roadmap §21–§23, critic finding,
   roadmap v2 §9–§10) and they are seed-paired and replicated on both wind sets.
3. **Agentic supervision — what survives a fair default** (§12–§13): the hyper-parameter lever
   is worth nothing once the default is tuned (LLM −0.5 / +0.3, random −0.8 / −0.9); the combined
   agent is not additive (6.3 / 8.1); the reward-code agent has **no mean effect at n = 5**
   (+0.4 / +0.5, 3/5 seeds, p ≈ 0.7) but **changes what the same J is made of**: 3/5 seeds reach the
   balanced solution (all four terms positive; arm-mean tower DEL +1.5 % vs −17.6 % for the guard at
   equal J) that no weight search (fork-myopic), weight sweep (λ_T grid), hyper-parameter search or
   combined agent ever reached. The honest claim is reachability with 3/5 reliability, not a
   level gain. The F-era proposer result (LLM > random 7/7 on reward weights) stands as the
   complementary case: the LLM matters where the search space is semantic.
4. **Honest framing**: the objective decides which lever "works" (F: proposer; J: none on the
   level of J, reward shape on its composition). The RL does not beat the fixed model-based
   reference; its best seeds match it. Every agentic difference under J is inside seed noise at
   n = 5; what is reportable is the solution class the agent reaches and how often.

## 12. Combined agent — hyper-parameters and reward in one supervisor (2026-09-12, user: paper centred on the agent)

Decision: keep the agent as a *supervisor* of the RL learner and make the reward-shape lever
rigorous. First step: `llm_combo` — one agent whose candidates may change the PPO hyper-parameters,
the reward weights / bounds, and the reward expression (`LLMComboSupervisor`; a candidate without
`reward_code` inherits the expression in use — before this fix a hold / hparams-only candidate
silently reverted to the parametric reward). Rationale: in isolation the hyper-parameter lever
repaired the learner (credit window) and the reward-code lever changed the solution's shape (all
four terms positive); if the two are complementary the combined agent should approach the
residual-channel MPC (12.0 / 11.6). Arms: `jcb3t` × 3 seeds, plus the corrected `jhp3t` / `jrhp3t`
× 3 (tuned default this time). Then: n = 5 for `jrw3t` / `jg3t`, the agent-loop ablations
(single-shot, objective-conditioning), see the plan agreed on 2026-09-12.

**Result (2026-09-12 20:11; held-out S3–S6 / S7–S10, 3 seeds, all on the tuned v3 default):**

| arm | J S1+S2 | J S3–S6 | J S7–S10 | S3–S6 P / ω / T / B | vs guard (S3–S6 / S7–S10) |
|---|---|---|---|---|---|
| guard (`jg3t`) | 8.22 ± 2.40 | 5.76 ± 1.76 | 6.81 ± 1.29 | 18.2 / 21.1 / −17.3 / 1.1 | — |
| llm_hparam, tuned (`jhp3t`, corrected) | 8.35 ± 1.62 | 5.26 ± 2.39 | 7.13 ± 3.05 | 17.1 / 20.7 / −18.1 / 1.2 | −0.5 / +0.3 (p ≈ 0.9) |
| random_hparam, tuned (`jrhp3t`, corrected) | 8.14 ± 0.25 | 4.93 ± 2.02 | 5.96 ± 2.17 | 15.1 / 17.0 / −13.2 / 0.8 | −0.8 / −0.9 (p ≈ 0.8) |
| **llm_reward** (`jrw3t`) | 13.13 ± 1.44 | **8.30 ± 1.93** | **9.17 ± 1.13** | 9.3 / 15.7 / **+4.3** / **+3.9** | +2.5 / +2.4 (3/3) |
| llm_combo (`jcb3t`) | 8.26 ± 0.93 | 6.28 ± 1.95 | 8.07 ± 1.12 | 15.9 / 16.4 / −8.1 / 0.9 | +0.5 / +1.3 (2/3, p ≈ 0.5–0.9) |
| MPC-v2, residual channel | 18.15 | 11.97 | 11.61 | 8.5 / 20.5 / 13.2 / 5.6 | |

1. **On a tuned default the hyper-parameter lever is worth nothing** (−0.5 / +0.3 for the LLM,
   −0.8 / −0.9 for random; every difference within noise). §9's +6…+8 and the "+2.5" of the first
   step-2 table were both measured against defaults whose training was broken (rollback loop) or
   untuned; with the reward gated and the weights tuned, the learner no longer needs the repair.
   The lever's value was entirely the repair of a bad default — the D6 question answered.
2. **The combined agent is not additive**: 6.3 / 8.1 sits between guard and the reward-only agent,
   below the latter by ≈ 2 / 1 J. Its best checkpoints show why: the agent raised the learning
   rates ×2–5 and γλ to 0.99–0.995 *and* wrote *quadratic* regulation terms with linear or
   `max(0, load−1)` load terms — a weaker reward shape than the reward-only agent's saturated
   `tanh` terms — while tower DEL only recovered to −7…−11 %. Two levers in one prompt split the
   agent's attention; the reward-only agent, asked one question, answered it better.
3. Therefore the paper's agentic result is **one lever, the reward's shape**: +2.5 / +2.4 over a
   tuned fixed reward in 3/3 seeds, the only RL variant positive on all four terms, unreachable by
   weight search, weight sweep, hyper-parameter search or the combined agent. Everything else the
   agent was given to tune is either noise on a fair default (hyper-parameters) or a repair of a
   default that should not have been broken (reward weights under F, hyper-parameters under J).

Next (plan of 2026-09-12): n = 5 for `jrw3t` / `jg3t` (seeds 3–4), then the agent-loop ablations —
single-shot without fork verification / history, and the objective-conditioning test (the same
agent given the F text under J).

## 13. n = 5 for the reward-code agent (2026-09-13 07:00) — the mean effect does not survive

Seeds 3–4 added to `jrw3t` and `jg3t` (same tuned v3 default), held-out S3–S6 / S7–S10:

| arm | n | J S1+S2 | J S3–S6 | J S7–S10 | S3–S6 P / ω / T / B | seeds with all four terms > 0 |
|---|---|---|---|---|---|---|
| guard (`jg3t`) | 5 | 7.05 ± 2.71 | 4.86 ± 2.76 | 5.50 ± 2.09 | 17.4 / 19.0 / −17.6 / 0.7 | 0 / 5 |
| llm_reward (`jrw3t`) | 5 | 9.36 ± 4.75 | 5.30 ± 4.17 | 6.04 ± 4.28 | 6.8 / 10.1 / **+1.5** / 2.7 | **3 / 5** (s0–s2); s3 tower −12 %, s4 the 12.5 m/s collapse |

Paired: +0.44 (S3–S6) / +0.53 (S7–S10), 3/5 seeds, exact p = 0.75 / 0.69, t-test p = 0.8 / 0.7.
The +2.5 of n = 3 was two good seeds; at n = 5 the reward-code agent and the tuned guard have
**the same J**. What the agent does change is the *composition* of J: three of five seeds land on
the balanced solution (every term positive) that no guard seed, no weight search, no λ_T sweep and
no hyper-parameter arm ever reached, and the arm-mean tower DEL flips from −17.6 % to +1.5 % at
equal J. Seed 4 wrote a reward that let the 12.5 m/s regulation collapse through (the guard's own
failure mode) — the agent is a capability, not a guarantee, and its reliability at n = 5 is 3/5.

`docs/tables/table_J_n5.csv`.

## 14. Agent-loop ablations of the reward-code agent (2026-09-13 11:04; n = 3, tuned v3 default)

| arm | what differs | J S1+S2 | J S3–S6 | J S7–S10 | S3–S6 P / ω / T / B | all-positive seeds |
|---|---|---|---|---|---|---|
| guard (`jg3t`, s0–2) | no agent | 8.22 ± 2.40 | 5.76 ± 1.76 | 6.81 ± 1.29 | 18.2 / 21.1 / −17.3 / 1.1 | 0/3 |
| llm_reward (`jrw3t`, s0–2) | full loop, told J | 13.13 ± 1.44 | 8.30 ± 1.93 | 9.17 ± 1.13 | 9.3 / 15.7 / 4.3 / 3.9 | 3/3 |
| **told F, J selects** (`jrwF3t`) | prompt, evaluation_now and fork outcomes show **F**; checkpoint / fork / rollback use J | 11.39 ± 1.43 | **8.58 ± 0.59** | **9.89 ± 0.31** | 10.0 / 16.6 / 3.5 / 4.3 | 2/3 |
| **single blind proposal** (`jrwO3t`) | one call at decision 1, applied without a fork, no history afterwards | 8.79 ± 1.98 | 4.70 ± 2.34 | 6.14 ± 2.33 | 14.4 / 14.0 / −8.6 / −1.0 | 0/3 |

Paired vs guard (S3–S6 / S7–S10): told-F +2.8 / +3.1 (3/3, t p = 0.16 / 0.11); single-shot
−1.1 / −0.7 (1/3). Paired vs the J-told agent: told-F +0.3 / +0.7 (inside noise).

1. **The objective text the agent reads does not matter.** Told F — a tower-priority objective
   with a speed *constraint* — the agent wrote the same saturated-speed / bounded-load shapes
   (`tanh(d_wg/.005)**2`, `tanh((1−load)/(1+load))`, `min(load, 3)`) and, with J doing the
   selection, landed on the balanced solution as often and with a third of the variance. The
   fork picks show it choosing between candidates whose F and J disagreed (e.g. F 20.7 vs J −0.1)
   — J won every time because J does the selecting. The agent's contribution is a *structural
   prior* on reward shape; the objective is enforced by the verification loop, not by the prompt.
2. **The loop is what makes the prior pay.** The same agent asked once, applied blind, is the
   guard (4.7 / 6.1, tower −8.6 %): its first expressions are of the same family, but without
   verification and iteration two of three seeds keep the guard's tower trade. Fork verification
   against the objective, not the proposal, turns the prior into the balanced solution.
3. Read with §13 (n = 5: no mean effect): the reward-code agent is a *shape prior plus a
   verification loop*; the loop is necessary (single-shot fails), the prompt's objective is not
   (told-F works), and the level gain over a tuned fixed reward is inside seed noise while the
   composition change (all four terms positive) is what it reliably-enough produces (3/5, 2/3, 3/3
   across the three arms that had the loop; 0/3 without it, 0/5 for the guard).

`docs/tables/table_J_ablations.csv`. The two arms live in `campaign_j_core.sh` (`jrwF3t`, `jrwO3t`).

## 15. Robustness of the MPC-v2 reference (2026-09-13 13:03; residual channel, J-selected weights)

The controller's *model* is perturbed, the plant never (`mpc_baseline.py --mm_cp/--mm_ft/--mm_m`);
held-out S3-S6. Stress classes as in the 08-30 roadmap s17 (RL rows there are F-era runs).

| variant | J | P / w / T / B | reading |
|---|---|---|---|
| exact model | 11.97 | 8.5 / 20.5 / 13.2 / 5.6 | reference |
| Cp/Ct x0.85 | -0.37 | -3.8 / -2.8 / 2.6 / 2.5 | regulation gone, tier degraded (spd 1.030) |
| Cp/Ct x1.15 | -27.53 | -58.2 / -74.2 / 16.5 / 5.7 | over-estimated aero torque: regulation collapses, tower gain stays |
| tower frequency x0.9 | 6.91 | 2.4 / 11.7 / 10.1 / 3.5 | halved |
| tower frequency x1.1 | 10.63 | 6.1 / 17.8 / 13.7 / 4.9 | minor |
| modal mass x0.8 | 11.00 | 3.7 / 16.8 / 19.2 / 4.3 | minor |
| modal mass x1.2 | 12.22 | 12.1 / 22.2 / 9.5 / 5.0 | none |
| TI 14 % (S3-S4) | 10.43 | -8.4 / 21.7 / 18.0 / 10.3 | holds |
| TI 22 %, U15 (S3-S4) | 12.31 | -1.9 / 20.7 / 31.0 / -0.5 | holds |
| U18 (S3-S4) | 22.45 | 44.9 / 44.9 / -0.6 / 0.6 | regulation only (no R2, no tower gain) |

The MPC is robust to structural-model errors of 10-20 % and to off-design turbulence, and **fragile
to aerodynamic-model error**: +-15 % on Cp/Ct takes it from 12 to GSPI level or far below. Its lead
over the RL arms is therefore a *perfect-aero-model* ceiling: the manuscript's reference row is
"LPV-MPC with an exact aerodynamic model", and the model-error rows go in the same table.
`docs/manuscript/figures/fig4_reference.png` (b).

## 16. Overnight supplements (2026-09-14): A ablations to n = 5, C aero-model error at +-5 %

**A (03:00).** Seeds 3-4 added to the told-F and single-shot arms (tuned v3 default), held-out:

| arm | n | J S3-S6 | J S7-S10 | S3-S6 P / w / T / B | all-positive seeds | vs guard (n = 5) |
|---|---|---|---|---|---|---|
| guard (`jg3t`) | 5 | 4.86 +- 2.76 | 5.50 +- 2.09 | 17.4 / 19.0 / -17.6 / 0.7 | 0/5 | - |
| llm_reward, told J (`jrw3t`, s13) | 5 | 5.30 +- 4.17 | 6.04 +- 4.28 | 6.8 / 10.1 / +1.5 / 2.7 | 3/5 | +0.4 / +0.5 (3/5, p 0.75 / 0.69) |
| llm_reward, told F, J selects (`jrwF3t`) | 5 | 7.00 +- 2.12 | 8.33 +- 2.38 | 9.5 / 14.6 / +0.9 / 3.0 | 3/5 | +2.1 / +2.8 (4/5, exact p 0.31 / 0.19, t p 0.26 / 0.10) |
| llm_reward, single blind proposal (`jrwO3t`) | 5 | 4.97 +- 1.90 | 6.66 +- 2.75 | 11.6 / 13.3 / -5.5 / 0.5 | 1/5 | +0.1 / +1.2 (2/5, 3/5; p ~ 1) |

The n = 3 picture survives in kind, not in size: the prompt's objective still does not matter
(told F is, if anything, the better arm: 4/5 wins, the only RL arm with a positive tower mean),
and the loop still does (single-shot 1/5 all-positive, level = guard). Pooled over the two
looped arms (told J + told F, n = 10, same agent, same loop): all-positive 6/10, paired
+1.3 / +1.7 J vs guard; without the loop 1/5; guard 0/5.

**C (02:57).** MPC-v2 (residual channel) with the aerodynamic model off by +-5 %: Cp/Ct x0.95 ->
J 5.42 (P +1.4, w +7.8, T +8.3), x1.05 -> J 3.10 (P -9.1, w -0.1, T +17.0). Even a 5 % aero
error halves to quarters the MPC's J; with +-15 % it is at or below GSPI (s15). The reference
row is a perfect-aero-model ceiling in the strict sense.

**B (09:45).** Random structural control `jrr3t` (`RandomRewardSupervisor`: expressions from a fixed
grammar of the term families the agent uses, log-uniform coefficients, same fork loop; 3 seeds,
tuned v3 default), held-out:

| arm | n | J S3-S6 | J S7-S10 | S3-S6 P / w / T / B | all-positive | vs guard |
|---|---|---|---|---|---|---|
| random reward structure + loop (`jrr3t`) | 3 | 6.13 +- 0.52 | 7.06 +- 1.35 | 9.5 / 10.5 / +2.1 / 2.5 | 1/3 | +0.4 / +0.3 (2/3, 1/3; p ~ 1) |
| llm_reward told J (`jrw3t`) | 5 | 5.30 +- 4.17 | 6.04 +- 4.28 | 6.8 / 10.1 / +1.5 / 2.7 | 3/5 | +0.4 / +0.5 |
| llm_reward told F (`jrwF3t`) | 5 | 7.00 +- 2.12 | 8.33 +- 2.38 | 9.5 / 14.6 / +0.9 / 3.0 | 3/5 | +2.1 / +2.8 |

The random draw from the agent's own vocabulary, verified by the same loop, lands where the agent
lands: same level, positive tower mean, the balanced solution in 1/3 seeds, and its kept
expressions are the saturated forms (tanh speed terms). The agent's contribution is therefore the
**vocabulary of shapes**; given it, verification finds the balanced solution without an LLM.
Reliability (6/10 vs 1/3) is not separable at these n. Manuscript 3.2 and the design rules say so.

**D (11:13).** Combined agent `jcb3t` at n = 5: 6.37 +- 2.94 / 8.65 +- 2.60; vs guard +1.5 / +3.2
(4/5, exact p 0.31 / 0.13); composition 15.9 / 16.3 / -7.6 / 0.9, all-positive 0/5. It raises the
level a little (not significant) and never the composition: two levers in one prompt buy
regulation the way the hyper-parameter arms do. Final tables: `docs/tables/table_J_final.csv`;
manuscript Table 1 updated. All supplements A-D done; machine idle.

## 17. Residual on the LPV-MPC base (2026-09-15 07:00; `--base mpc`, `campaign_j_mpcbase.sh`)

The residual sits on the J-selected LPV-MPC (wide-open) instead of the GSPI: the MPC runs in the
worker, the residual (+-0.05 rad, damped) is added to its target, the MPC target is in the
observation. Zero residual == the wide-open MPC, so episode 0 is the MPC (plus the untrained
actor's ~0.3 deg constant offset, which happens to help: init J 20.5 on S1+S2 vs 17.2 for the MPC).
Tuned v3 default, 3 seeds, held-out S3-S6 / S7-S10 (two exact-base runs were killed by the nightly /tmp purge at ep 224/272 and finished after a resume;
final n = 3 everywhere):

| controller | n | J S3-S6 | J S7-S10 | S3-S6 P / w / T / B |
|---|---|---|---|---|
| LPV-MPC alone, exact model, wide-open | - | 12.40 | 14.10 | 8.9 / 19.8 / 15.8 / 5.1 |
| **MPC + fixed-reward residual** (`mg3t`) | 3 | 16.60 +- 0.78 | 18.06 +- 1.10 | 12.1 / 23.6 / 24.2 / 6.4 |
| **MPC + LLM-reward residual** (`mrw3t`) | 3 | 17.23 +- 1.30 | 18.84 +- 1.04 | 13.4 / 24.7 / 23.6 / 7.2 |
| LPV-MPC alone, Cp/Ct x0.95, wide-open | - | 4.48 | -1.0 | (regulation lost) |
| **mismatched MPC + fixed-reward residual** (`mgC3t`) | 3 | 14.17 +- 0.82 | 15.35 +- 0.68 | 18.6 / 27.6 / 6.7 / 3.8 |
| **mismatched MPC + LLM-reward residual** (`mrwC3t`) | 3 | 12.66 +- 5.13 | 18.25 +- 2.52 | 13.0 / 22.7 / 9.3 / 5.7 |

Reading: (1) the supervised residual adds ~+5 J to the strongest controller with every term
positive and the tower gain rising from 16 to 23-25 %; (2) on the MPC whose aero model is 5 % off
(alone: 4.5 / -1) the residual restores 13-18 J, i.e. above the exact-model MPC alone - the residual
learns what the model gets wrong; (3) between the fixed and the LLM-written reward the difference
is again small (+0.6/+0.8 exact, 2/3; -1.5/+2.9 mismatched, one bad seed). This is the manuscript's new
main result; the GSPI-base arms become the ablations of the supervision design.
`docs/tables/table_J_mpcbase.csv`.

**Superseded by s18 (2026-09-15 09:30): the two "MPC alone" rows above are not the MPC the residual
sits on** (they were the MPC pushed through the RL's action channel); with the MPC as the base the
residual adds +0.3 / +1.0, not +5, and the mismatched base is 8.5 / 3.6, not 4.5 / -1.


## 18. The MPC reference was evaluated through the wrong channel; corrected (2026-09-15 08:40-09:30; `campaign_mpcbase0.sh`, `evaluate.py --gspi` on an MPC-base config)

**Finding.** Three of the six exact-base residual runs (`mg3t_s1`, `mg3t_s2`, `mrw3t_s0`) have their
best checkpoint at episode 0 (|dbeta| ~ 0.001 deg, i.e. the MPC itself) and still score 15.7-16.5 on
S3-S6 against the "MPC alone, wide-open" row of 12.40. The standalone rows (`mpc_baseline.py`) send
the MPC's target through the RL's residual channel (`safety.apply`) with the bound opened; that
channel has the R2 non-negativity rule written for the learned residual (`d = max(d, 0)` while the
router says R2), which zeroed the MPC's negative offsets during the router's transition hold at
12.5 m/s and produced pitch discontinuities. One-episode diagnostic (U12.5 TI8 S3, same MPC, same
wind): tower DEL 11.58 MN m through the channel vs 7.32 MN m as the base controller (GSPI 19.40);
the difference is in [40, 150) s (11.9 vs 6.6), not a start-up transient ([20, 40) s: 8.8 vs 8.9);
the base offset is negative 62 % of the scored steps. So the MPC-in-env is the faithful MPC and
every "MPC alone" row must be the MPC as the base with a zero residual (`evaluate.py --gspi --run
<mpc-base run>`; new `--base_mm_cp/ft/m`, `--base_mpc_json`, `--out`, `--dump_log`).

**Corrected reference rows** (MPC as base, zero residual; `~/wtrl/exp/mpc/eval_base0_*.json`):

| variant | J S3-S6 | J S7-S10 | S3-S6 P / w / T / B |
|---|---|---|---|
| exact model | **16.28** | **17.22** | 11.0 / 22.3 / 24.1 / 7.8 |
| Cp/Ct x0.95 | **8.53** | **3.57** | 1.5 / 4.7 / 23.4 / 4.5 |
| Cp/Ct x0.85 / x1.05 / x1.15 | -24.01 / 5.71 / -24.58 | | regulation collapses at +-15 % |
| tower f x0.9 / x1.1 | 12.03 / 13.38 | | |
| modal mass x0.8 / x1.2 | 13.33 / 17.64 | | |
| TI 14 % (S3-S4) / TI 22 % U15 (S3-S4) / U18 (S3-S4) | 12.56 / 14.22 / 23.06 | | off-design winds |
| regulation-only cost (qt = 0, unscaled refs) | -2.19 / -6.20 (10.5 / 1.2 / -22.0 / 1.6) | | the published verdict |

**Consequences for s17** (differences vs the corrected reference, seed-wise, exact floor 0.25):
- exact base: fixed-reward residual +0.32 / +0.84 (2/3, 2/3); LLM-reward residual +0.95 / +1.62
  (2/3, 3/3). The residual adds ~nothing to the exact-model MPC; half the runs never beat episode 0.
- Cp x0.95 base: fixed +5.6 / +11.8 (3/3, 3/3); LLM +4.1 / +14.7 (2/3, 3/3). The repair holds
  (8.5 / 3.6 -> 14.2 / 15.4 and 12.7 / 18.3) but from a higher floor than 4.5 / -1.
- LLM reward vs fixed reward on the MPC base: unchanged (+0.6 / +0.8 exact, 2/3; -1.5 / +2.9
  mismatched). The agent does not move the level on the strong base either.
- The old standalone model-error rows (s15-s16, residual channel) are superseded by the base rows
  above; as the base (full authority) the MPC is *more* sensitive to its aero model (+-15 %: -24).

**Manuscript** rewritten around the agent (user instruction 09:05: "MPC+residual is not new; avoid
two papers fighting"): the MPC base is the strong-base test of the same supervision study, one
subsection; new figures `fig_trajectory.png` (one held-out 15 m/s episode, both bases; episode by
a stated rule) and `fig_training.png` (J on S1-S2 vs episode, fixed vs agent, both bases; fork
decisions and rollback dips visible); `fig_mpc_error`, `fig_mpcbase`, `fig2_composition`,
`fig4_reference` re-sourced from the base0 rows. `j_table.py --ref_json` writes the reference rows
and seed-wise differences into `table_J_mpcbase.csv`. Per-step logs of the trajectory runs:
`<run>/logs_traj_s3456/`, `mpc/logs_base0_*`.


## 19. Offset-free and adaptive MPC close the model error without learning; the residual story does not survive (2026-09-15 14:17-15:10; `campaign_tcst.sh`, gate STOP)

Two textbook compensators added to `controllers/mpc.py` (`adapt`): **offset** = offset-free MPC, a lumped
aerodynamic-torque disturbance from the rotor balance J dw/dt = T_aero,model + d - T_gen (measured
generator torque, filtered acceleration), held over the horizon; **rls** = adaptive MPC, a scalar gain on
the model aerodynamics (torque and thrust) by forgetting least squares. Synthetic check with a x0.95 model:
d/T -> +0.0500, theta -> 1.0526 (exact). Time constant selected on seeds 1-2 by mean J over exact/x0.95
(tau 2 / 5 / 15 s): offset 11.9 / **22.7** / 21.4, rls 16.2 / **22.3** / 20.3 -> both tau = 5 s. All rows: MPC
as the env base, zero residual (`eval_base0_{offset,rls}_*`).

| MPC variant | model | J S3-S6 | J S7-S10 | S3-S6 P / w / T / B |
|---|---|---|---|---|
| nominal | exact | 16.28 | 17.22 | 11.0 / 22.3 / 24.1 / 7.8 |
| nominal | Cp x0.95 | 8.53 | 3.57 | 1.5 / 4.7 / 23.4 / 4.5 |
| **offset-free** | exact | **20.47** | **21.69** | 22.0 / 33.7 / 21.2 / 5.0 |
| **offset-free** | Cp x0.95 | **21.66** | **21.87** | 23.9 / 35.5 / 19.6 / 7.7 |
| **adaptive (RLS)** | exact | **20.44** | **21.66** | 21.7 / 33.5 / 21.4 / 5.2 |
| **adaptive (RLS)** | Cp x0.95 | **20.38** | **21.58** | 21.7 / 33.5 / 21.4 / 4.9 |
| best residual run on any MPC base (mrw3t_s2) | exact | 18.91 | 19.99 | 17.4 / 28.7 / 22.0 / 7.6 |

Model-error sweep, S3-S6 (nominal / offset-free / adaptive): x0.85 -24.0 / 21.8 / 20.6; x0.9 -16.1 / 21.6 / 20.7;
x0.95 8.5 / 21.7 / 20.4; x1.05 5.7 / 20.0 / 20.6; x1.15 -24.6 / 15.8 / **20.6**. The adaptive MPC is flat
(20.4-20.7) over +-15 %; offset-free degrades only at +15 % (it corrects the torque bias, not the gain).
Even with the exact model both add ~4 J: they also absorb what the model misses beyond Cp (drivetrain
losses, dynamic inflow, the LSS torque balance), mostly on regulation (power MSE 11 -> 22 %, speed 22 -> 34 %)
at a small tower-fatigue cost (24 -> 21 %).

Residual generalisation (S3-S6): residuals trained on the x0.95 MPC fall apart off their training error
(fixed reward: x0.9 5.8 / 8.8 / 7.0, x1.05 5.5 / 4.9 / 5.7; LLM reward: x0.9 10.8 / 11.5 / 4.6, x1.05 -21.7 / 4.0 / 1.8);
residuals trained on the exact MPC, run on the x0.95 model, are the nominal x0.95 MPC again (8.8-12.1).

**Verdict (gate STOP, seeds 3-7 not launched).** A compensator with no training data, no reward and no
agent reaches 20.4-21.9 on both held-out sets with a 5 % (and up to 15 %) aerodynamic error; the best
supervised residual reaches 18.9 / 20.0 with an exact model and 12.7 / 18.3 with the error, and does not
transfer to a different error. The manuscript v4 main line ("the residual adds to the strongest controller
and repairs its model") is refuted by the textbook baseline a TCST reviewer would ask for; the offset-free /
adaptive MPC is the new strongest controller on J. Open for the user's decision: residual/supervision on the
compensated MPC, supervision of the MPC's interpretable parameters, or verification-centred supervision.

## 20. How reliable is the loop's verification? Zero-simulation probe (2026-09-15; `scripts/dev/verification_probe.py`, `docs/tables/verification_probe.csv`)

75 J-era runs with both held-out sets (63 GSPI base, 12 MPC base); 240 fork decisions with a following evaluation.

**Checkpoint selection transfers.** Spearman between the supervisor-wind (seeds 1-2) score of the selected
checkpoint and its held-out score, after removing each arm's mean (the pooled rho is inflated by arm differences):

| base | term | supervisor winds -> held-out | held-out S3-6 <-> S7-10 (ceiling) |
|---|---|---|---|
| GSPI (61 runs) | J | 0.87 | 0.90 |
| GSPI | tower fatigue | 0.91 | 0.94 |
| GSPI | speed MSE | 0.92 | 0.97 |
| MPC (12 runs) | J | 0.24 | 0.01 |
| MPC | tower fatigue | 0.97 | 0.34 |

On the GSPI base two supervisor wind seeds rank runs almost as well as another four-seed held-out set; the
selection optimism (+2.2 J on average) is a shift, not a reordering. On the MPC base the runs differ by less
than the wind noise: the two held-out sets do not agree with each other (0.01), so nothing can be ranked there.
Tower fatigue is the most reliable term on both bases, not the least.

**Fork choice is below the training noise.** Chosen candidate after its 30-episode fork vs the same policy at
the next evaluation: persistence rho 0.57 for J (0.86 tower, 0.62 speed, 0.63 power) on the GSPI base, 0.14 on
the MPC base; winner's curse median +0.3 J (GSPI) / +10 J (MPC). The margin between the chosen candidate and the
runner-up is small (median 0.3 J) and does not predict the realised gain (rho +0.02, n 240); best-minus-worst
candidate rho -0.03. The fork does predict whether training on from this checkpoint helps (predicted vs realised
gain rho 0.46, sign agreement 0.71), but not which candidate is better. LLM proposals: predicted vs realised 0.48
(165 decisions); random proposals 0.30 (75).

**Collapses come from training, not from forks.** Share of evaluations that trigger the guardrail's rollback at
the next decision: fixed-reward runs (no forks) 39 % GSPI / 57 % MPC; after a fork 25 % GSPI / 73 % MPC.

**Consequences.**
- The hypothesis "the validation winds are too few" is not supported on the GSPI base.
- The hypothesis "a 30-episode fork is myopic for tower fatigue" (manuscript v4, Appendix A) is not supported:
  tower fatigue is the most persistent and best-transferring term; the lambda_T > 1 candidates lost on J for
  another reason (the full 300-episode lambda_T sweep is worse too). That sentence must be revised.
- What is supported: (i) the difference between candidates is smaller than the noise of 30 further PPO
  episodes, which is why the proposer (LLM vs random) barely matters; (ii) training instability is pervasive
  (a rollback-sized drop at 25-73 % of evaluations), and the guardrail carries much of the loop's value.
  A verification-centred method would have to replicate forks over training seeds (the noise is the
  learner's, not the wind's) and/or stabilise the update, not enlarge the validation wind set.

## 21. 600 s protocol and a verified search over the compensated MPC's parameters (2026-09-15 23:50 - 2026-09-16 07:30; `campaign_mpc600_search.sh`, `scripts/mpc_param_search.py`, `scripts/dev/mpc_search_report.py`)

**600 s protocol.** The separate 650 s bank `~/wtrl/wind600` was extended with TurbSim seeds 1-2 (supervisor
winds) and 7-10 (second held-out set) at 8 / 12.5 / 15 m/s; paired GSPI baselines at 600 s live in the separate
home `~/wtrl600` (30 files). Identity check: GSPI against its own 600 s baselines gives J = 0.00. The canonical
150 s bank and its baselines were verified untouched (the 2026-09-09 incident does not recur: `run.sh` honours a
preset `WTRL_HOME`). Every row below is 600 s, 6 episodes on the supervisor winds and 12 per held-out set.

**References at 600 s** (MPC as the base controller, zero residual; J on supervisor winds / seeds 3-6 / seeds 7-10):

| MPC | exact model | Cp/Ct x0.95 |
|---|---|---|
| nominal | 18.01 / 16.84 / 18.33 | 7.88 / 10.29 (held-out) |
| offset-free | 22.53 / 20.89 / 21.71 | 21.26 / 22.26 (held-out) |
| adaptive (RLS) | 22.84 / 21.09 / 22.47 | 21.03 / 22.22 (held-out) |

The s19 verdict replicates at 600 s: compensation is worth ~4 J with an exact model and ~11 J with a 5 % aero
error, and both compensators are insensitive to that error (0.2-0.6 J).

**The search.** Box: horizon {10,15,20,30}, r [0.03,3], qt [0.1,30], wc_v [0.1,1.5], tau_adapt [1,30],
adapt {offset,rls}; start = the offset-free MPC of s19. Three proposers, 40 verified candidates each, batches of
4, deterministic evaluation on the supervisor winds (no training noise, unlike the residual of s20).

| proposer | best J after 8 / 16 / 24 / 32 / 40 | J <= 0 candidates | best held-out (seeds 3-6 / 7-10) |
|---|---|---|---|
| LLM | 22.84 / 23.17 / 23.30 / 23.40 / 23.40 | 1/40 | **21.37 / 22.75** |
| local search (1+lambda around the incumbent) | 22.53 / 22.53 / 22.53 / 22.53 / 22.68 | 1/40 | 21.08 / 21.96 |
| uniform random over the box | never beat the start | 34/40 | (start) 20.89 / 21.71 |

Reading: (1) the tuned corner of the box is small - uniform random destabilises the MPC in 34 of 40 draws, so it
is a weak control; the local search is the meaningful non-LLM comparison. (2) The LLM's gain over the start is
+0.87 on the supervisor winds and +0.48 / +1.04 held-out; over the *adaptive* reference (same weights, rls) it is
+0.56 supervisor and +0.28 held-out on seeds 3-6. Both proposers converge to the same direction: slightly lower r
(0.25-0.26 vs 0.3), slightly higher qt (3.2-3.25 vs 3.0), faster wind filter (0.38-0.40 vs 0.35). (3) The
supervisor-wind ranking transfers: Spearman +0.93 (p 0.001, 8 selected candidates) with a mean optimism of only
+0.36 J - the deterministic MPC evaluation is a far better verifier than the residual's fork (s20). (4) The gains
are ~1 J on a 21-point controller; n = 1 search per proposer, so no significance claim.

Artefacts: `docs/tables/mpc600_search.csv`, `docs/figures/mpc600_search.png`, histories under
`~/wtrl/exp/mpcsearch600/{llm,es,random}/history.jsonl` (LLM transcript included), cache of 143 evaluations.

Paused at 07:30 and resumed at 08:39 for the remaining reference rows; the campaign finished at 08:59.

**Weight sensitivity (2026-09-15, no simulations, `scripts/dev/weight_sensitivity.py`).** Re-weighting the four
metrics of the 150 s held-out results: wherever regulation carries >= 25 % of the weight the compensated MPC is
the best controller; under fatigue-dominated weights (regulation share <= 12.5 %, or tower weight 0.7) the
nominal MPC and the MPC-base residuals lead, because compensation buys regulation at a small fatigue cost.
`docs/tables/weight_sensitivity.csv`, `docs/figures/weight_sensitivity.png`.

## 22. Actuator cost: the MPC's gains are bought with ~7.6x the pitch travel (2026-09-16, `scripts/dev/actuation_table.py`, no simulations)

The literature scan (`docs/literature_2026-09-16.md`, items 12 and 16) reports fatigue reductions jointly with
pitch travel / actuator duty cycle and treats load gains bought with actuation as suspect. Our evaluation files
already carry pitch travel per episode, so the column was free. Held-out wind seeds 3-6, mean over episodes and
seeds; travel is the summed |d beta| of the scored window, duty is travel per scored second.

| controller | pitch travel [deg/ep] | vs GSPI | duty [deg/s] | J |
|---|---|---|---|---|
| GSPI (pairing reference) | 27.9 | 1.00 | 0.215 | 0 |
| LPV-MPC, nominal | 213.8 | 7.66 | 1.644 | 16.28 |
| LPV-MPC, offset-free | 213.1 | 7.64 | 1.639 | 20.47 |
| LPV-MPC, adaptive (RLS) | 204.6 | 7.33 | 1.574 | 20.44 |
| LPV-MPC, nominal, Cp x0.95 | 197.1 | 7.06 | 1.516 | 8.53 |
| LPV-MPC, offset-free, Cp x0.95 | 196.9 | 7.06 | 1.515 | 21.66 |
| GSPI + fixed-reward residual (n=5) | 62.4 | 2.24 | 0.480 | 4.86 |
| **GSPI + agent-reward residual (n=5)** | **41.0** | **1.47** | **0.315** | 5.30 |
| MPC + fixed-reward residual (n=3) | 152.0 | 5.45 | 1.169 | 16.60 |
| MPC + agent-reward residual (n=3) | 215.6 | 7.73 | 1.658 | 17.23 |
| x0.95 MPC + fixed-reward residual (n=3) | 218.0 | 7.81 | 1.677 | 14.17 |
| x0.95 MPC + agent-reward residual (n=3) | 161.9 | 5.80 | 1.245 | 12.66 |

**Consequences.**
1. Every MPC row in this repository buys its regulation and load reductions with roughly 7x the pitch activity of
   the GSPI baseline. No table before today reported this, and the two most comparable papers in the scan would
   treat the omission as a defect. The actuator column now belongs in every main table.
2. The ranking changes under an actuation budget: per unit of pitch travel the agent-reward residual on the GSPI
   base is the most efficient controller we have (5.30 J at 1.47x travel), while the compensated MPC is the least
   (20.5 J at 7.6x). The objective J does not see this at all.
3. The control-direction formulation therefore needs three constraints, not two: tower-base DEL, blade-root DEL
   **and** actuator duty cycle, with regulation as the objective. That is the safe-BO formulation of s21's
   follow-up, and it is the formulation under which the two controller families are actually comparable.
4. Caveat to check before publishing: part of the MPC's travel may be chatter from re-solving every 0.1 s rather
   than useful actuation. `pitch_rate_power_tower_band_frac` and `pitch_rate_power_3P_band_frac` are already in
   the per-episode metrics; a spectral decomposition of pitch rate will say how much of the 7x is broadband
   chatter and how much is load-driven motion.

## 23. Control direction, stage 1 (2026-09-16, `campaign_control1.sh`; paused 18:07 mid-training)

Three questions from the literature scan. Two are answered, the third is answered in part.

**Q2 — how far does a residual transfer across model error?** Residuals trained on the Cp/Ct x0.95 MPC,
evaluated at other errors (wind seeds 3-6, mean over 3 seeds):

| controller | x0.85 | x0.9 | x0.95 | x1.05 | x1.15 |
|---|---|---|---|---|---|
| fixed-reward residual | -12.2 | 7.2 | 14.2 | 5.4 | -24.7 |
| agent-reward residual | -13.1 | 9.0 | 12.7 | -5.3 | -29.5 |
| nominal MPC alone | -24.0 | -16.1 | 8.5 | 5.7 | -24.6 |
| **offset-free MPC alone** | **21.8** | **21.6** | **21.7** | **20.0** | **15.8** |

The residual only works at the error it was trained on and collapses to (or below) the nominal MPC elsewhere;
the estimator holds 15.8-21.8 across the whole +-15 % range. This is the quantified decomposition the literature
lacks (the closest prior work reports only an ordering, observer ~ residual).

**Q3 — ROSCO with its tower damper enabled, 600 s** (the baseline a reviewer will demand): J = -0.11 / -0.17 /
-0.08 on supervisor winds / seeds 3-6 / seeds 7-10, every term within +-1 %. The tower damper contributes
nothing under this objective, so our tower-DEL gains are not an artefact of a stripped baseline. The row belongs
in the main table (`~/wtrl/exp/mpcsearch600/refs/eval_towerdamper_*.json`).

**Q1 — does the residual still add anything once the estimator has removed the model error?** Training on the
offset-free base with Cp/Ct x0.95 (150 s bank, tuned default, 3 seeds per arm): **5 of the 6 finished runs have
their best checkpoint at episode 0**, i.e. 300 episodes of PPO never beat the zero-residual compensated MPC on
the supervisor winds (moC3t 22.74 / 22.73 / 22.93, morC3t 22.74 / 22.93 at episode 0; only morC3t_s1 improved,
23.46 at episode 224). Held-out evaluation of these arms had not started when the campaign was paused.

**State at the pause.** `campaign_ctl.sh pause campaign_control1` stopped 43 processes; `mo3t_s0` and `mo3t_s1`
(residual on the offset-free base with the exact model) were mid-training with rolling resume points, `mo3t_s2`
not started. `resume` continues; the core campaign then runs the 150 s held-out evaluations of all three arms,
after which the script evaluates them at 600 s. Everything else is cached.

## 24. Stage 1 complete: on a compensated base the residual has (almost) nothing left to learn (2026-09-17 02:20)

Residual on the offset-free MPC base, 150 s training with the tuned default, 3 seeds per arm, evaluated on
both held-out sets at 600 s (`eval_heldout600_*`). Reference rows are the same base with a zero residual.

| configuration | best checkpoint (episode) | 600 s seeds 3-6 | 600 s seeds 7-10 | P / w / T / B (seeds 3-6) |
|---|---|---|---|---|
| offset-free base, exact, alone | - | 20.47 | 21.69 | 22.0 / 33.7 / 21.2 / 5.0 |
| + fixed-reward residual (`mo3t`) | 64 / 96 / 96 | 20.87 | 21.71 | 24.6 / 33.6 / 21.8 / 3.5 |
| offset-free base, Cp x0.95, alone | - | 21.66 | 21.87 | 23.9 / 35.5 / 19.6 / 7.7 |
| + fixed-reward residual (`moC3t`) | **0 / 0 / 0** | 21.51 | 22.06 | 26.1 / 35.1 / 21.6 / 3.2 |
| + agent-reward residual (`morC3t`) | 0 / **224** / 0 | 21.93 | 22.41 | 26.5 / 35.5 / 22.0 / 3.6 |

Reading. (1) With the model error removed by the estimator, 5 of 6 mismatched-base runs never beat their
initial policy on the supervisor winds; the one that did (`morC3t_s1`, 22.80 / 23.19 held-out) is +1.1 over the
base. On the exact base the residual adds +0.4 / +0.0. (2) The rows whose best checkpoint is episode 0 still
differ from the base on the held-out sets (e.g. 21.51 vs 21.66) because the untrained actor's output is not
exactly zero; the difference is within the wind noise of s20. (3) What the residual does change is the
composition: power MSE up (22.0 -> 24.6), blade-root DEL down (5.0 -> 3.5 %; 7.7 -> 3.2 %) - a re-allocation
between terms, not a net gain. Together with s23 (the residual does not transfer across model error while the
estimator holds +-15 %) this completes the decomposition the literature lacks: parametric model error belongs to
the estimator; after it, a bounded residual has nothing of size to learn.

Infrastructure note: stage 2 was queued behind stage 1 with `until ! pgrep -f campaign_control1.sh`, which
matches its own command line and never fires; stage 2 was started by hand at 08:46 after a six-hour gap.

## 25. Where the MPC's actuation goes, and the J-vs-duty front (2026-09-17; `campaign_control2.sh`, `pitch_spectrum_table.py`, `pareto_duty.py`)

**The 3P hypothesis is refuted.** The spectrum table showed 29-34 % of the MPC's pitch-rate power in the 3P band
(0.5-0.75 Hz) against 4 % for GSPI, so a speed-scheduled 3P notch was added to the MPC's speed measurement
(`notch_3p_q` in `controllers/mpc.py`; on a synthetic signal Q = 2 removes the 3P line, share 0.297 -> 0.001,
and improves slow tracking). On the plant it does not pay: on the supervisor winds Q = 1 / 2 / 5 give
-22.9 / 14.2 / 20.6 (mean over exact and x0.95) against 22.7 without the notch, and at Q = 5 held-out
(600 s, exact / x0.95; seeds 3-6 / 7-10):

| offset-free MPC | J 3-6 | J 7-10 | pitch travel [deg/ep] | 3P share of pitch-rate power |
|---|---|---|---|---|
| no notch, exact | 20.89 | 21.71 | 968 / 951 | 0.290 |
| + notch Q5, exact | 19.21 | 20.22 | 945 / 933 | 0.100 |
| no notch, x0.95 | 21.26 | 22.26 | 903 / 892 | - |
| + notch Q5, x0.95 | 19.24 | 20.51 | 889 / 877 | - |

The notch removes the 3P line (share 0.29 -> 0.10) but cuts travel by only 2-3 % and costs 1.5-2.4 J, mostly
tower fatigue (22 -> 19 %) and regulation. The 3P share was a share of pitch-rate *power*; the travel itself is
broadband (the "other" band holds 63-68 % for every MPC row), i.e. fast actuation across the band, governed by
the pitch-increment weight r, not by one spectral line. Kept in the code as an option, off by default.

**The J-vs-actuator-duty front.** The 600 s search cache holds 123 valid MPC settings on the supervisor winds
with J and per-episode pitch travel; the trade-off is therefore already measured (`docs/figures/pareto_duty.png`,
`docs/tables/pareto_duty.csv`). Pareto-efficient settings (duty = pitch travel per scored second):

| duty [deg/s] | x GSPI | J | setting |
|---|---|---|---|
| 0.31 | 1.5 | 11.4 | N10, r 0.46, qt 2.3, rls |
| 0.52 | 2.4 | 18.9 | N15, r 0.53, qt 3.4, offset |
| **0.63** | **2.9** | **21.0** | N15, r 0.41, qt 3.7, offset (the knee) |
| 0.86 | 4.0 | 21.5 | N15, r 0.29, qt 4.3, offset |
| 1.28 | 5.9 | 22.5 | N20, r 0.35, qt 3.25, rls |
| 1.70 | 7.9 | 23.4 | the J-selected point (s21) |

Reading. (1) Selecting on J alone puts the operating point at 7.9x the GSPI travel for 2.4 J more than the knee
at 2.9x. An actuator-duty constraint of ~3x GSPI keeps 90 % of the J. (2) s22's statement that the agent-reward
residual is "the most actuation-efficient controller" was wrong: at the same 1.5x duty the MPC front reaches 11.4
against the residual's 5.3; the MPC family dominates the learned residual at every duty level. (3) The knee
points were scored on the supervisor winds only (one deterministic evaluation each); stage 3 (`campaign_control3.sh`)
evaluates three of them on both 600 s held-out sets, the knee also with Cp/Ct x0.95.

## 26. Stage 3: the duty knee holds on held-out winds - same J at 38 % of the actuation (2026-09-17 10:07; `campaign_control3.sh`)

Three Pareto settings of s25 on both 600 s held-out sets (duty = pitch travel per scored second; xGSPI = travel
relative to the paired GSPI run; terms P / w / T / B on the same set):

| setting | model | seeds 3-6: J, duty, xGSPI | seeds 7-10: J, duty, xGSPI | P / w / T / B (3-6) |
|---|---|---|---|---|
| J-selected point (s21 LLM best, N20 r .255 qt 3.25 rls) | exact | 21.37, 1.73, 7.7x | 22.75, 1.71, 8.0x | 25.0 / 34.8 / 22.9 / 2.8 |
| offset-free reference (N20 r .3 qt 3) | exact | 20.89, 1.67, 7.4x | 21.71, 1.64, 7.6x | 24.6 / 33.6 / 22.3 / 3.1 |
| knee3 (N15 r .29 qt 4.3 offset) | exact | 21.56, 0.86, 3.8x | 22.03, 0.87, 4.0x | 21.9 / 35.9 / 23.0 / 5.3 |
| **knee2 (N15 r .41 qt 3.7 offset)** | exact | **20.86, 0.63, 2.8x** | **21.46, 0.64, 3.0x** | 21.4 / 34.7 / 21.6 / 5.8 |
| knee1 (N15 r .53 qt 3.4 offset) | exact | 18.48, 0.52, 2.3x | 18.82, 0.52, 2.4x | 19.5 / 31.8 / 17.5 / 5.0 |
| offset-free reference | Cp x0.95 | 21.26, 1.56, 6.9x | 22.26, 1.54, 7.1x | 26.1 / 35.1 / 21.0 / 2.7 |
| **knee2** | Cp x0.95 | **21.52, 0.64, 2.8x** | **21.87, 0.63, 2.9x** | 23.6 / 36.5 / 20.6 / 5.4 |

Reading. (1) The knee transfers: knee2 matches the offset-free reference's J on both held-out sets and both
models (-0.03 / -0.25 exact, +0.26 / -0.39 x0.95) at **38-41 % of its pitch travel**; knee3 is at or above the
reference at 52 %. (2) The composition shifts the right way for a load-aware reader: the cheaper settings give
up 2-3 points of power MSE and keep tower fatigue, and their blade-root DEL reduction roughly doubles
(3 -> 5-6 %), presumably because the smoother pitch stops exciting the blades. (3) Below the knee the cost is
real: knee1 (2.3x) loses 2.4-2.9 J, mostly tower fatigue. (4) Together with s22 and s25 this is the
control-direction result: selecting an MPC on J alone lands at 7-8x the baseline's actuation; a duty
constraint at ~3x recovers the same objective with better blade fatigue. The learned residual arms sit
below the MPC front at every duty level (s25).

Artefacts: `~/wtrl/exp/mpcsearch600/refs/eval_knee{1,2,3}_*.json`, `docs/tables/pareto_duty.csv`,
`docs/figures/pareto_duty.png`. Machine idle after 10:07.

## 27. Paired statistics for the deterministic controllers (2026-09-17; `scripts/dev/paired_bootstrap.py`, no simulations)

An MPC evaluation is deterministic given the wind file, so the uncertainty of "controller A vs controller B" is
the wind realisation, not a training seed. All reference controllers ran on the same 24 held-out 600 s episodes
(8 / 12.5 / 15 m/s x TurbSim seeds 3-10), so differences are paired by episode: 20 000 bootstrap resamples,
A and B drawn with the same indices, stratified by mean wind, J recomputed on every resample exactly as
`eval/fitness.py` does. `docs/tables/paired_bootstrap.csv`.

| A - B | model | diff in J | 95 % interval |
|---|---|---|---|
| offset-free - nominal MPC | exact | +3.72 | [+3.02, +4.48] |
| offset-free - nominal MPC | Cp/Ct x0.95 | +12.67 | [+11.33, +14.05] |
| adaptive (RLS) - offset-free | exact | +0.48 | [+0.24, +0.76] |
| J-selected point - offset-free reference | exact | +0.76 | [+0.41, +1.14] |
| knee2 - offset-free reference | exact | -0.14 | [-0.68, +0.43] |
| knee2 - offset-free reference | Cp/Ct x0.95 | -0.07 | [-0.72, +0.61] |
| knee3 - offset-free reference | exact | +0.49 | [-0.32, +1.29] |
| knee1 - offset-free reference | exact | -2.65 | [-3.21, -2.07] |
| knee2 - J-selected point | exact | -0.90 | [-1.43, -0.37] |
| offset-free + 3P notch - offset-free | exact | -1.58 | [-1.84, -1.34] |

Reading. The statements of s19-s26 that were made on set means survive pairing: compensation beats the nominal
MPC, the J-selected point of the verified search beats the hand-set reference, the 3P notch costs J, knee1 is
below the knee. "knee2 equals the reference at 38-41 % of the travel" is an equivalence within +-0.7 J, and knee2
is 0.9 J below the J-selected point: the duty constraint is not free against the best J, it is free against the
hand-set reference. Known gap: the peak generator-speed column of the script reads NaN (the quantity is in the
evaluation CSV, not in the per-episode JSON).

## 28. Wind coverage: 12-24 m/s, IEC turbulence class B, 600 s, six realisations (2026-09-17 12:20 - 09-18 01:32; `campaign_must1_range.sh`, `scripts/dev/range_table.py`)

Protocol. 42 new wind fields `U{12,14,...,24}_TIB_S{1..6}` (650 s, TurbSim normal turbulence model class B:
turbulence intensity ~17 % at 12 m/s to ~14 % at 24 m/s) added to the separate 600 s bank `~/wtrl/wind600`;
paired GSPI baselines in `~/wtrl600`; the canonical bank is untouched. None of these winds selected anything,
so all 42 episodes are held-out for every controller. Identity check J = 0.00. Every controller is the base of
the environment with a zero residual. Tables: `docs/tables/range_{controllers,per_wind,paired}.csv`.

| controller | J | duty deg/s | x GSPI | energy | power / speed / tower / blade | J at 12 / 14 / 16 / 18 / 20 / 22 / 24 m/s |
|---|---|---|---|---|---|---|
| ROSCO + tower damper | -0.05 | 0.478 | 1.00 | -0.00 % | -0.3 / -0.6 / 0.5 / 0.2 | -0.4 0.1 -0.1 -0.1 -0.0 0.0 0.1 |
| nominal MPC | 28.48 | 0.955 | 2.01 | +0.23 % | 42.0 / 46.9 / 20.9 / 4.1 | 19.9 22.5 20.0 29.1 34.1 36.5 37.3 |
| offset-free MPC | 29.50 | 0.940 | 1.97 | +0.23 % | 47.6 / 52.7 / 14.3 / 3.3 | 20.8 21.9 24.4 31.8 34.7 36.1 36.7 |
| J-selected MPC (s21) | 29.76 | 1.065 | 2.24 | +0.35 % | 47.8 / 53.4 / 14.8 / 3.0 | 22.5 23.1 25.2 32.1 34.5 35.5 35.5 |
| duty knee 3 (s26) | 30.28 | 0.979 | 2.05 | +0.24 % | 47.2 / 54.8 / 15.4 / 3.7 | 22.6 24.2 25.8 32.6 34.4 36.0 36.4 |
| duty knee 2 (s26) | 29.66 | 0.799 | 1.68 | +0.23 % | 46.2 / 53.2 / 14.6 / 4.7 | 22.4 23.3 24.8 31.8 33.9 35.3 36.1 |
| nominal MPC, Cp/Ct x0.95 | 27.78 | 0.979 | 2.06 | +0.47 % | 43.5 / 45.0 / 19.0 / 3.7 | 19.2 20.7 19.8 29.0 33.4 35.7 36.6 |
| offset-free MPC, Cp/Ct x0.95 | 29.50 | 0.958 | 2.01 | +0.21 % | 48.1 / 53.5 / 13.3 / 3.1 | 20.4 21.4 24.8 32.1 34.9 36.2 36.8 |
| duty knee 2, Cp/Ct x0.95 | 29.63 | 0.809 | 1.70 | +0.21 % | 46.8 / 53.9 / 13.2 / 4.5 | 21.7 22.8 24.6 32.2 34.3 35.6 36.2 |

Paired bootstrap over the 42 episodes (A - B, 95 % interval; J and the four terms):

| A - B | J | power | speed | tower | blade |
|---|---|---|---|---|---|
| offset-free - nominal | +1.02 [+0.59, +1.45] | +5.56 [+4.77, +6.34] | +5.85 [+5.10, +6.60] | **-6.59 [-7.06, -6.13]** | -0.76 [-1.37, -0.17] |
| offset-free - nominal, both x0.95 | +1.72 [+1.12, +2.31] | +4.60 | +8.52 | -5.61 [-6.30, -4.91] | -0.64 |
| nominal x0.95 - nominal exact | -0.70 [-1.12, -0.27] | +1.48 | -1.92 | -1.99 | -0.39 |
| offset-free x0.95 - offset-free exact | -0.00 [-0.14, +0.14] | +0.52 | +0.75 | -1.01 | -0.26 |
| knee2 - offset-free | +0.16 [-0.09, +0.39] | -1.43 [-1.74, -1.13] | +0.45 | +0.25 | +1.35 [+1.06, +1.64] |
| knee3 - offset-free | +0.78 [+0.54, +1.03] | -0.39 | +2.10 | +1.09 | +0.34 |
| J-selected - offset-free | +0.26 [+0.05, +0.46] | +0.21 | +0.70 | +0.46 | -0.34 |
| knee2 - J-selected | -0.10 [-0.39, +0.19] | -1.64 | -0.25 | -0.21 | +1.68 [+1.26, +2.14] |
| knee2 x0.95 - knee2 exact | -0.03 [-0.14, +0.08] | +0.67 | +0.75 | -1.35 | -0.18 |
| ROSCO tower damper - GSPI | -0.05 [-0.34, +0.21] | -0.28 | -0.60 | +0.47 [-0.00, +0.84] | +0.19 |

Reading - several earlier statements are specific to the low-turbulence 8 / 12.5 / 15 m/s set and must not be
carried into a paper as general:

1. **Actuation.** "The MPC buys its J with ~7.6x the GSPI pitch travel" (s22) holds at turbulence intensity 8 %,
   where the GSPI barely moves (0.22 deg/s). At class B the GSPI itself travels 0.48 deg/s and every MPC row is at
   **1.7-2.2x**; the ratio grows with wind speed (1.2x at 14 m/s, 3.0x at 24 m/s). The knee still orders the same
   way: knee2 has the J of the reference (+0.16 [-0.09, +0.39]) at 85 % of its travel and 75 % of the travel of the
   J-selected point, with +1.4 / +1.7 points of blade fatigue. The margin is 15-25 % here, not 60 %.
2. **Model-error fragility.** Cp/Ct x0.95 costs the nominal MPC 0.70 J [0.27, 1.12] over the range, not half of
   its J (16.3 -> 8.5 on the low-turbulence set, where the 8 m/s episodes carry the loss). The compensated rows do
   not move at all (-0.00 and -0.03, intervals +-0.14). Compensation is worth +1.0 J exact and +1.7 J at x0.95.
3. **Composition of the compensation.** Offset-free compensation is a trade, not a gain on every term: +5.6 power,
   +5.9 speed, **-6.6 tower** (all intervals exclude 0). Per wind speed (`range_per_wind.csv`) the tower term of
   every compensated row - reference, J-selected, knee2, knee3 - is **negative from 20 m/s up** (-1...-9 % at
   20-24 m/s, i.e. worse tower fatigue than the GSPI), while the nominal MPC stays non-negative at every wind speed
   (58.8 % at 12 m/s down to 0.9 % at 24 m/s). With "better regulation without degrading loads" as the claim, the
   aggregate J hides a per-wind-speed violation. All MPC settings were selected on 8 / 12.5 / 15 m/s winds; the
   tower weight that is right at 12-15 m/s is too small at 20+ m/s.
4. **Where the J comes from.** J rises with wind speed (20 at 12 m/s, 37 at 24 m/s) because the regulation terms
   saturate at 70-77 % while the tower term falls from 55-60 % to ~0; the blade term is 16-20 % at 12 m/s and ~0
   above 16 m/s.
5. The ROSCO tower damper is worth nothing here either (-0.05 [-0.34, +0.21]; tower +0.5, speed -0.6).
6. **Overspeed margin.** Largest generator speed over the 42 episodes: GSPI 1.106 x rated, nominal MPC 1.068,
   compensated rows 1.056-1.063 (`peak_gen_speed_rel` in `range_controllers.csv`). Every MPC row roughly halves the
   peak excursion above rated; none comes near a trip level.

Consequence for the evaluation protocol: report the range set as the main table (it is what the literature scan
of `docs/literature_2026-09-16.md` found to be expected), report per-wind-speed terms next to the aggregate, and
treat "no term negative at any wind speed" as the load constraint instead of the set mean.

## 29. Tuned ROSCO baseline: three quarters of the MPC margin belongs to an untuned baseline (2026-09-18 01:32-03:15; `scripts/rosco_tune.py`, stage A of `campaign_must2.sh`)

Question. Every percentage in this repository is a reduction relative to the GSPI baseline as ROSCO ships it for
the NREL 5 MW. What is left of the margins when that baseline gets the budget every other controller got?

Search. Four parameters of the baseline's own pitch loop: factors on the gain-scheduled proportional and integral
gain tables (0.4-3), corner of the generator-speed low-pass filter (0.6-4 rad/s; the upper bound keeps the corner
near the 3P frequency at rated speed, 3.8 rad/s), gain of ROSCO's fore-aft tower damper (0-0.004). 40 candidates
(20 random, 20 local search around the incumbent), each a copy of the OpenFAST case with an edited `DISCON.IN`,
scored under J against the ORIGINAL GSPI baselines on the 600 s supervisor winds (8 / 12.5 / 15 m/s, TurbSim
seeds 1-2). Identity check of the template-editing path: the untuned start scores J = -0.04 (the filter corner is
rounded to three digits).

Result. Best candidate: proportional x1.13, integral x1.02, **filter corner 1.57 -> 3.42 rad/s**, tower damper
0.00102. Supervisor J 16.90; held-out **16.19** (wind seeds 3-6: 22.6 / 35.1 / 4.9 / 2.1) and **17.07** (wind
seeds 7-10: 23.4 / 35.9 / 7.4 / 1.6), all four terms positive on both sets, energy +0.07 %, **pitch travel 1.08x
the untuned GSPI**. No overfit to the supervisor winds. The random half of the box is mostly destructive
(18 of 20 below -10 J; its one hit is the 16.90 point, which the local search never beat), the neighbourhood of
that point is a broad plateau: 17 of 20 local candidates positive, 11 of them at 9.8-16.5 J with filter corners
2.3-3.8 rad/s and travel 0.75-1.3x; one of them (x1.02 / x0.86 / 3.43 rad/s) scores 15.6 with LESS travel than the untuned GSPI (0.94x)
and 11 % tower fatigue. The lever is the speed filter: the shipped corner costs the loop phase, and a faster
filter buys regulation (40 % power and speed MSE at 15 m/s, the same as the MPC rows) at no actuation cost.

Paired bootstrap over the 24 held-out 600 s episodes (`docs/tables/paired_bootstrap.csv`):

| A - B | diff in J | 95 % interval | travel of A / B (x untuned GSPI) |
|---|---|---|---|
| nominal MPC - tuned ROSCO | +0.95 | [+0.22, +1.74] | 7.5 / 1.08 |
| offset-free reference - tuned ROSCO | +4.67 | [+3.81, +5.57] | 7.5 / 1.08 |
| J-selected point - tuned ROSCO | +5.43 | [+4.47, +6.44] | 7.8 / 1.08 |
| knee2 - tuned ROSCO | +4.53 | [+3.53, +5.55] | 2.9 / 1.08 |

Per mean wind (power / speed / tower / blade): tuned ROSCO 12.5 m/s 6 / 31 / 12 / 5 and 15 m/s 40 / 40 / 7 / 1;
offset-free reference 8 / 27 / 56 / 10 and 43 / 43 / 10 / -1.

Reading. (1) On this set the MPC rows keep **4.5-5.4 J of their 21-22 J** against a tuned baseline; the nominal
MPC keeps 1 J while spending 7x the actuation. Roughly three quarters of every margin reported in s18-s26 was the
untuned speed filter. (2) What the MPC keeps is specific: tower fatigue at 12.5 m/s (56-62 % vs 12 %), i.e. the
term that needs a tower model. Regulation at 15 m/s is a tie. (3) Every comparison in a paper has to carry the
tuned ROSCO as a baseline row, and the MPC claim becomes "tower fatigue near rated at equal regulation", not
"better regulation". (4) The tuned ROSCO on the wind-range set (s28) is queued in `campaign_must3.sh`.

Artefacts: `~/wtrl/exp/roscotune600/{history.jsonl,best.json,cache/}`.
