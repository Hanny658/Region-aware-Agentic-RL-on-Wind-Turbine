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
