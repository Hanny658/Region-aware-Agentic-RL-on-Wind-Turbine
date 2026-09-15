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
