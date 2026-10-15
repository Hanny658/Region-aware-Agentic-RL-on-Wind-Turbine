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

The literature scan (`docs/archive/literature_2026-09-16.md`, items 12 and 16) reports fatigue reductions jointly with
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
of `docs/archive/literature_2026-09-16.md` found to be expected), report per-wind-speed terms next to the aggregate, and
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

## 30. Domain-randomised residual: training over the whole model-error range does not rescue the residual (2026-09-18 03:15-09:31; stage B of `campaign_must2.sh`, arms `mgR3t` / `mrwR3t`)

Question. s23 trained the residual at ONE model error (Cp/Ct x0.95) and found it does not transfer. The fair best
effort for the learning side is to train it over the whole range: every training episode draws the nominal
MPC's Cp/Ct scale uniformly from [0.85, 1.15] (`--base_mm_cp_range`, `cp_scale_range` in the MPC base of
`envs/base_env.py`); deterministic evaluations during training use a fixed grid over the range assigned by
episode index, so checkpoint selection scores the range and is repeatable. Fixed reward (`mgR3t`) and agent-written
reward (`mrwR3t`), 3 seeds each, 300 episodes, 150 s bank as every residual arm. Held-out sets are evaluated with the
exact model (as in s17-s18); the transfer row re-evaluates the selected checkpoint at fixed errors on wind seeds
3-6 (`eval_gen_cp<cp>_s3456.json`).

| run | best episode | held-out exact (seeds 3-6 / 7-10) | x0.85 | x0.9 | x0.95 | x1.05 | x1.15 |
|---|---|---|---|---|---|---|---|
| fixed reward, seed 0 | 184 | 15.02 / 13.93 | -3.6 | 12.4 | 15.1 | 4.0 | -24.8 |
| fixed reward, seed 1 | 300 | 13.94 / 13.11 | -23.8 | 5.6 | 14.2 | 5.4 | -24.7 |
| fixed reward, seed 2 | 96 | 10.06 / 7.27 | -6.7 | 8.3 | 7.2 | 2.6 | -24.6 |
| **fixed reward, mean** | | **13.0 / 11.4** | **-11.4** | **8.8** | **12.2** | **4.0** | **-24.7** |
| agent reward, seed 0 | 32 | 15.30 / 18.87 | -20.9 | 6.4 | 17.5 | 4.1 | -26.3 |
| agent reward, seed 1 | 0 | 16.47 / 16.99 | -23.7 | -15.9 | 8.9 | 5.4 | -24.5 |
| agent reward, seed 2 | 32 | 4.28 / 4.21 | -8.9 | 12.6 | 14.1 | -14.3 | -42.6 |
| **agent reward, mean** | | **12.0 / 13.4** | **-17.8** | **1.0** | **13.5** | **-1.6** | **-31.1** |
| nominal MPC alone (s23) | | 16.3 / 17.2 | -24.0 | -16.1 | 8.5 | 5.7 | -24.6 |
| x0.95-trained residual, fixed (s23) | | | -12.2 | 7.2 | 14.2 | 5.4 | -24.7 |
| **offset-free MPC alone (s23)** | | **20.5 / 21.7** | **21.8** | **21.6** | **21.7** | **20.0** | **15.8** |

Reading. (1) The domain-randomised residual lands where the x0.95-trained one did: it repairs the underestimate
side partially (x0.85: -24 -> -11 on average, -3.6 for the best seed; x0.9: -16 -> +9) and does **nothing at
+15 %** (-24.7 vs -24.6 for the bare MPC, in every seed). The in-training objective over the grid never rose
above -5.6 (from -12 at the zero residual): the overestimate side dominates the mean and the bounded, region-gated
residual cannot reach it in 300 episodes. (2) The price is paid at the exact model: 13.0 / 11.4 against 16.3 / 17.2
for the bare MPC, 3-6 J. (3) The agent-written reward does not help here: one seed never left episode 0 (its
held-out rows are the bare MPC), one collapsed at x1.05 / x1.15 (-14 / -43), the mean transfer is worse than the
fixed reward. (4) The estimator without learning is 10-40 J above every residual at every error and costs nothing
at the exact model. Together with s23 (single-error training) and s24 (nothing left to learn on a compensated base)
this closes the residual-repairs-the-model line from the learning side as well: the fair best effort for the
learner (range training, agent reward, 3 seeds) does not change the ordering.

Infrastructure note. The post-training "critic health" diagnostic of `campaign_j_core.sh` fails on these runs
(`scripts/dev/critic_health.py` builds the critic for the 6-dimensional GSPI-base observation while MPC-base runs
observe the base command as a 7th input); it is a diagnostic only, the campaign continued and every evaluation
completed.

Artefacts: `~/wtrl/exp/{mgR3t,mrwR3t}_s{0,1,2}/eval_*.json`.

## 31. Range set: the tuned ROSCO row, and the tower weight is not the lever for the high-wind tower term (2026-09-18 09:31-10:54; `campaign_must3.sh`)

Both stages on the wind-range set of s28 (12-24 m/s, IEC class B, 600 s, 42 held-out episodes).
Tables regenerated: `docs/tables/range_{controllers,per_wind,paired}.csv`.

**A. Tuned ROSCO (s29) on the range.** J **15.39** (23.2 / 29.2 / 5.4 / 3.8), pitch travel 1.14x the untuned
GSPI, peak generator speed 1.104 x rated (the untuned GSPI: 1.106), energy +0.12 %. Per mean wind 12 -> 24 m/s:
10.4 15.5 16.9 19.1 16.7 15.1 14.2, i.e. flat, while every MPC row climbs from ~20 to ~37.

| A - B (42 episodes) | J | power | speed | tower | blade | travel A / B |
|---|---|---|---|---|---|---|
| tuned ROSCO - GSPI | +15.39 [+14.87, +15.90] | +23.2 | +29.2 | +5.4 | +3.8 | 1.14 / 1.00 |
| offset-free MPC - tuned ROSCO | +14.11 [+13.42, +14.80] | +24.4 | +23.5 | +9.0 [+7.6, +10.4] | -0.5 [-1.2, +0.2] | 1.97 / 1.14 |
| duty knee 2 - tuned ROSCO | +14.26 [+13.52, +15.02] | +23.0 | +24.0 | +9.2 | +0.9 [+0.1, +1.6] | 1.68 / 1.14 |

Reading. On the low-turbulence set (s29) the tuned ROSCO tied the MPC on regulation and the MPC kept 4.5-5.4 J
of tower fatigue near rated. At class B over the full above-rated range the MPC keeps **~14 J**: 23-24 points of
power and speed MSE (the faster speed filter does not reach the MPC's 70-77 % regulation at 18-24 m/s) plus 9
points of tower fatigue, at 1.5-1.7x the tuned ROSCO's travel. The margin over a tuned baseline is
condition-dependent in the opposite direction from the actuation story: small at low turbulence, large at
class B and high wind.

**B. Tower-weight sweep for the compensated MPC** (diagnosis of s28 finding 3, the negative tower term from
20 m/s up; selects nothing, these winds stay held-out). Offset-free reference (N20 r .3 wc_v .35 tau 5) with qt
3 (reference) / 6 / 12 / 24, and the duty knee with qt 11 (3x its 3.68):

| setting | J | travel x GSPI | power / speed / tower / blade | tower term at 12 / 16 / 20 / 24 m/s | J at 12 m/s |
|---|---|---|---|---|---|
| qt 3 (reference) | 29.50 | 1.97 | 47.6 / 52.7 / 14.3 / 3.3 | 55 / 13 / -0.9 / -6.2 | 20.8 |
| qt 6 | 27.27 | 2.39 | 41.0 / 46.3 / 19.8 / 2.0 | 65 / 16 / 1.7 / 0.6 | 20.5 |
| qt 12 | 18.03 | 3.24 | 26.9 / 27.2 / 18.9 / -0.9 | 66 / 13 / 0.5 / 0.3 | -9.2 |
| qt 24 | -3.32 | 4.65 | -5.1 / -12.7 / 11.7 / -7.2 | 65 / 5 / -10.7 / -8.8 | -73.5 |
| knee2 qt 11 (vs knee2 29.66) | 21.27 | 2.26 | 30.7 / 29.5 / 21.9 / 3.0 | 68 / 19 / 2.1 / 1.6 | -29.4 |
| nominal MPC (s28) | 28.48 | 2.01 | 42.0 / 46.9 / 20.9 / 4.1 | 59 / 20 / 5.5 / 0.9 | 19.9 |

Paired: qt 6 - qt 3 = -2.23 [-2.52, -1.92] (power -6.6, speed -6.4, tower +5.5, blade -1.4);
qt 6 - nominal MPC = -1.21 [-1.75, -0.64], every term below the nominal.

Reading. (1) Doubling the tower weight does make the tower term non-negative at every wind speed (+0.5 to +1.7 %
at 20-24 m/s), so the per-wind load constraint of s28 is satisfiable, at a price of 2.2 J, 6.5 points of each
regulation term and **20 % more pitch travel** (the MPC damps the tower actively, which costs pitch action).
(2) Above that the tower term saturates: qt 12 gains nothing on the tower (0.3-0.5 % at 20-24 m/s) and loses
20 points of regulation; qt 24 and knee2 at qt 11 destabilise the 12 m/s episodes (speed MSE -41 to -100 %) and
the tower term goes negative again at high wind. The achievable tower-fatigue reduction of a collective-pitch
controller at 20-24 m/s is ~0-2 % under this objective; the nominal MPC's 0.9-5.5 % there is the ceiling of the
tested settings. (3) With the per-wind load constraint imposed, the nominal MPC (28.48, 2.0x, every term
non-negative everywhere) beats the constrained compensated MPC (27.27, 2.4x): compensation buys regulation by
giving up tower fatigue, and a global weight cannot undo that trade without giving the regulation back. A
wind-scheduled weight (qt 3 up to 18 m/s, 6 from 20 m/s) would, from the per-wind rows, score ~28.7 on this set
with the constraint met, i.e. the nominal MPC's J with the estimator's robustness to model error; it has to be
selected on TurbSim seeds 1-2 and reported on 3-6 if it is ever used.

Artefacts: `~/wtrl/exp/mpcsearch600/range/eval_{rosco_tuned,offset_qt6_cp1,offset_qt12_cp1,offset_qt24_cp1,knee2_qt11_cp1}.json`,
tuned ROSCO case template `~/wtrl/runs/template_5mw_rosco_tuned`. Machine idle after 10:54.

## 32. Residual on the tuned ROSCO under constrained objectives: no one-sided gain, and two defects of the formulation (2026-09-18 12:36 - 09-19 23:30; `campaign_agent_rt.sh`, `campaign_rt_fix.sh`, `scripts/dev/rt_table.py`)

Question (user, 2026-09-18). The control-only claim of s27-s31 is not novel. Can the agent-supervised residual give
a ONE-SIDED gain on top of a tuned baseline - regulation up with both fatigue loads not worse, or tower fatigue down
with regulation and blade not worse? Every earlier residual arm was selected under J, which lets one term pay for
another, and none was tried on the tuned ROSCO.

Setup. Base = tuned ROSCO of s29 (`WTRL_TEMPLATE`), reference = paired baselines of the SAME tuned ROSCO on the
canonical 150 s winds in the separate home `~/wtrl_rt` (identity J = C = CT = 0.00), so every reduction is "on top of
the tuned ROSCO". New objectives in `eval/fitness.py`, used for checkpoint selection, rollback and the supervisors'
fitness (the reward is unchanged):

    C  = mean(power MSE red, speed MSE red) - 20 * [violations of tower >= -1 %, blade >= -1 %, energy loss <= 1 %]
    CT = tower DEL red                      - 20 * [violations of power, speed, blade >= -1 %, energy loss <= 1 %]
    Cw / CTw = the same with every constraint taken PER MEAN WIND SPEED (violation = mean over wind speeds of
               max(0, -1 - term at that wind)); the target stays the set mean, energy stays a set-level constraint

18 runs, 300 episodes, 3 seeds per arm: fixed reward with the J-tuned weights under C; agent-written reward under C;
agent-written reward and fixed reward with tower weight 10 under Cw; the same two under CTw.

**Defect 1 - a set-mean constraint is gamed across wind speeds.** The first agent-reward run under C reported
"all four terms positive" on held-out seeds 3-6 (9.3 / 9.3 / 7.4 / 3.1). Per wind: tower +13.5 % at 8 m/s (below-rated
peak shaving, bought with 0.9 % energy), +26.3 % at 12.5 m/s, **-17.5 % at 15 m/s in every episode**. Hence Cw / CTw.

**Defect 2 - the constrained objectives were trained and scored on the wrong above-rated subset.** `train.py` and
`evaluate.py` enabled the wind-labelled subset (roadmap 16, finding 4) only for objective "J"; the C family used the
oracle rule, which leaves the 12.5 m/s episodes out of the regulation terms. Fixed in a969ce9; the six runs whose
selected checkpoint is not episode 0 were re-evaluated (`campaign_rt_fix.sh`, the oracle files are kept as
`*_oracle.json`). The re-evaluation is what the table reports. **All 18 runs were still TRAINED and selected under
the oracle subset**, so the table is evidence about that formulation; the clean re-run is `campaign_next2.sh`.

Held-out, wind-labelled subset, against the tuned ROSCO (objective / power / speed / tower / blade), and against the
original GSPI (J; the tuned ROSCO alone is 14.2 on these winds):

| arm | seed | selected episode | seeds 3-6: objective, P / w / T / B | seeds 7-10: objective | J vs GSPI |
|---|---|---|---|---|---|
| fixed reward (tower weight 1), C | 0, 1, 2 | 0, 0, 0 | 0 (tuned ROSCO alone) | | 14.2 |
| fixed reward (tower weight 10), Cw | 0, 1, 2 | 0, 0, 0 | 0 | | 14.2 |
| fixed reward (tower weight 10), CTw | 0, 1, 2 | 0, 0, 0 | 0 | | 14.2 |
| agent reward, Cw | 0, 2 | 0, 0 | 0 | | 14.2 |
| agent reward, Cw | 1 | 280 | -10.96: -5.4 / -7.6 / +4.2 / +2.2 | -11.44 | 13.2 |
| agent reward, CTw | 0, 2 | 0, 0 | 0 | | 14.2 |
| agent reward, CTw | 1 | 280 | -215.9: -4.9 / -6.8 / +4.1 / +2.1 | -217.0 | 13.4 |
| agent reward, C (set mean) | 0 | 280 | -36.3: -27.2 / -45.4 / +7.4 / +3.1 | -62.4 | -6.1 |
| agent reward, C (set mean) | 1 | 300 | -28.9: -15.5 / -42.2 / +2.4 / +3.7 | -120.8 | 1.4 |
| agent reward, C (set mean) | 2 | 300 | -31.7: -19.9 / -43.5 / +2.8 / +4.0 | -36.3 | -2.7 |

Per wind speed (seeds 3-6; `rt_perwind.py`): the set-mean agent runs gain 9-14 % of regulation at 15 m/s and lose
15-19 % of tower fatigue there, gain 17-26 % of tower fatigue at 12.5 m/s and lose **45-64 % of power MSE and
139-234 % of speed MSE** there (the part the oracle subset hid), with 1.1-1.6 % energy loss at 12.5 m/s. The two seed-1
per-wind runs learned a small near-rated offset (|dbeta| 0.03 deg on average): tower +14 % and blade +7 % at 12.5 m/s
for -11 % power and -15 % speed MSE there, and -1.7 % tower at 15 m/s. On the range set (12-24 m/s class B, 600 s, vs
the original GSPI) the episode-0 runs reproduce the tuned ROSCO (15.23-15.25 at 1.14x travel); the best set-mean
agent run scores 16.17 (23.3 / 31.2 / 5.2 / 4.9) at **4.06x** the GSPI pitch travel: +0.8 J for 3.6x the actuation.

Reading. (1) In-training, the fixed reward under C never beat episode 0 in any seed: the residual buys up to 5 points of
regulation with 2-13 % of tower fatigue and is rolled back at every evaluation; raising the tower weight of the
reward from 1 to 10 does not change that (seed 0: tower -2.4 to -13 % in every evaluation). (2) No arm produced a
checkpoint that holds the constraints per wind speed on held-out winds; what the agent-written rewards find are
trades - regulation for tower fatigue near rated, tower fatigue for regulation above rated - and a set-mean score
hides them. (3) This is the third base on which the residual only re-allocates (nominal MPC s17-s18, compensated
MPC s24, tuned ROSCO here). (4) The conclusion is conditional on defect 2: selection never saw the 12.5 m/s
regulation damage, and the agent was told the truth only about the subset it was scored on. `campaign_next2.sh`
re-runs the three informative arms with the fixed labelling; `campaign_next1.sh` stage B trains on the above-rated
range winds, where the two labellings nearly coincide.

**Defect 3 (found 2026-09-20 01:30) - the agent arms chose among their forked candidates by F.** `train.py` ranked
the forked reward candidates of a decision by `(energy_ok, J)` only for objective "J" and by the historical tier / F
ranking (tower-DEL priority) for every other objective, so the agent-reward arms of this section verified their
candidates against the wrong objective; checkpoint selection and rollback did use C / Cw / CTw. The same J-only test
guarded the RUN side of the wind labelling, which made a969ce9 (baseline side only) inconsistent for training: the
first two range-wind runs of `campaign_next1.sh` read P +13.4 / w +5.4 at the zero residual and were discarded
(`*_badlabel`). Both fixed in 4901e94 (`METRIC_SET`); identity after the fix: Cw = -0.02 / -0.00 at episode 0. The
agent rows of the table above therefore carry defects 2 AND 3; `campaign_next2.sh` is the clean re-run.

Reproducibility note. On wind seeds 7-10 the zero-residual re-run of one 8 m/s episode (TurbSim seed 8) differs from
its own baseline by tower -18.5 % / blade +13.8 % with identical energy; the set carries a -1.2 % tower / +1.3 %
blade offset at identity (a per-wind objective of -18 for a zero residual). Seeds 3-6 are the primary held-out set.

Artefacts: `~/wtrl/exp/{tgC3t,trwC3t,trwCw3t,tgCw3L10,trwTw3t,tgTw3L10}_s{0,1,2}/`, `docs/tables/tuned_rosco_residual.csv`,
`~/wtrl/exp/agent_rt_identity/` (identity and tuned-ROSCO-alone rows: 17.14 / 14.20 / 17.09 on seeds 1-2 / 3-6 / 7-10).

## 33. A wind-scheduled tower weight closes the high-wind load gap of the compensated MPC (2026-09-19 22:55 - 09-20 00:30; stage A of `campaign_next1.sh`, `scripts/dev/qt_sched_select.py`)

Problem (s28, s31). Every compensated MPC has a negative tower term from 20 m/s up, and a global tower weight cannot
repair it without giving the regulation back. Remedy tried here: `qt_sched = [qt_hi, v_lo, v_hi]` in
`controllers/mpc.py` - the tower weight stays at qt up to v_lo and ramps linearly to qt_hi at v_hi, on the filtered
wind-speed estimate, re-evaluated at every solve (the QP is rebuilt every solve anyway).

Protocol. Range bank (12-24 m/s, IEC class B, 600 s). SELECTION on TurbSim seeds 1-2 (14 episodes): offset-free
reference (qt 3) with qt_hi 6 / 9 over 16-20 and 18-22 m/s and 12 over 18-22 m/s, the duty knee with 11 over
18-22 m/s; rule = per-wind load rule of s28 (no term below -1 % at any mean wind speed), then J. REPORT on seeds 3-6
(28 episodes), which no selection touched; the unscheduled rows are the same 28 episodes of the s28 / s31 evaluations.

Selection table (seeds 1-2): every schedule lifts the 20-24 m/s tower term from -3.6 / -7.4 / -6.0 to between -2.7 and
+2.0; higher qt_hi buys nothing more (the saturation of s31) and costs J and travel (qt_hi 12: 26.9 at 2.43x). With two
episodes per wind speed no candidate - not even the nominal MPC (worst term -2.5) - meets the -1 % rule, so the rule fell
back to "least infeasible": offset-free 3 -> 6 over 18-22 m/s (28.97, worst -5.2) and knee2 3.68 -> 11 (28.65).
Two episodes per wind are too few for a 1 % rule; the selection should use all the supervisor episodes it can get.

Held-out (seeds 3-6, 28 episodes):

| controller | J | travel x GSPI | power / speed / tower / blade | tower term at 12 / 14 / 16 / 18 / 20 / 22 / 24 m/s | worst per-wind term |
|---|---|---|---|---|---|
| **offset-free, qt 3 -> 6 over 18-22 m/s** | **28.95** | 2.14 | 45.7 / 50.7 / 16.5 / 2.9 | 55.6 40.7 13.5 3.9 **1.6 0.2 0.1** | -0.8 (blade, 16 m/s) |
| knee2, qt 3.68 -> 11 over 18-22 m/s | 28.90 | 1.95 | 43.6 / 50.5 / 17.3 / 4.2 | 59.3 43.4 13.3 3.8 -0.1 0.4 1.0 | -2.9 (power, 12 m/s) |
| offset-free, unscheduled | 29.51 | 1.98 | 47.3 / 52.4 / 15.0 / 3.4 | 55.6 40.7 13.5 3.7 0.4 -2.5 -6.3 | -6.3 (tower, 24 m/s) |
| knee2, unscheduled | 29.74 | 1.69 | 45.9 / 53.0 / 15.1 / 5.0 | 59.3 43.4 13.3 3.3 -1.5 -4.4 -7.8 | -7.8 (tower, 24 m/s) |
| nominal MPC | 28.49 | 2.01 | 41.5 / 46.7 / 21.5 / 4.2 | 58.4 48.4 20.9 11.4 6.8 3.7 0.9 | none negative |
| tuned ROSCO | 15.24 | 1.15 | 23.1 / 28.9 / 5.3 / 3.6 | 7.6 16.6 3.1 4.5 2.1 2.0 1.3 | none negative |

Paired bootstrap over the 28 episodes (A - B, 95 % interval):

| A - B | J | power | speed | tower | blade |
|---|---|---|---|---|---|
| scheduled - unscheduled offset-free | -0.56 [-0.67, -0.43] | -1.59 | -1.63 | +1.49 [+1.22, +1.76] | -0.50 |
| scheduled offset-free - nominal MPC | +0.46 [-0.06, +1.02] | +4.18 [+3.21, +5.15] | +4.02 [+3.16, +4.88] | -5.00 [-5.62, -4.37] | -1.36 |
| scheduled - unscheduled knee2 | -0.83 [-1.06, -0.65] | -2.37 | -2.44 | +2.24 | -0.76 |
| scheduled offset-free - tuned ROSCO | +13.71 [+13.00, +14.39] | +22.56 | +21.80 | +11.18 [+9.61, +12.69] | -0.72 [-1.59, +0.16] |

Reading. (1) The schedule does what the global weight could not: on winds no selection touched, the compensated MPC
with qt 3 -> 6 over 18-22 m/s has **no term below -1 % at any wind speed** (tower +1.6 / +0.2 / +0.1 % at 20 / 22 /
24 m/s instead of +0.4 / -2.5 / -6.3 %), for 0.56 J and 8 % more pitch travel. (2) Against the nominal MPC it is
the same J within the interval with a different composition (+4 regulation, -5 tower); what it adds is the
estimator, i.e. the robustness to model error - measured so far on the low-turbulence set (s19, s23) and at x0.95 for
the unscheduled rows (s28); `campaign_next3.sh` measures it for the scheduled controller at x0.85 / x0.95 / x1.15 on
these 28 episodes. (3) Against the tuned ROSCO the margin is +13.7 J with every term except blade fatigue clearly
positive (blade -0.7 [-1.6, +0.2]). (4) The knee with a schedule keeps its low travel (1.95x) but carries the knee's
own -2.9 % power term at 12 m/s, which the schedule does not touch. (5) The controller of the paper's main claim is
therefore: offset-free LPV-MPC, tower weight scheduled on wind speed, selected by a per-wind load rule.

Artefacts: `~/wtrl/exp/mpcsearch600/qtsched/eval_*_s12.json` (selection), `eval_{off_q6_18_22,knee2_q11_18_22}_s3456.json`.

## 34. With the formulation fixed, the agent-written reward gives a load-holding gain on top of the tuned ROSCO; the fixed reward does not (2026-09-20 01:35 - 06:59; stage B of `campaign_next1.sh`)

Setup. The first training campaign after the three defects of s32 were fixed (per-wind constraints, wind-labelled
subset on both sides, fork candidates ranked by the run's own objective; identity at episode 0: Cw = -0.02 / -0.00).
Base = tuned ROSCO; training winds = the ABOVE-RATED range bank (12-24 m/s, IEC class B, first 150 s of the 600 s
fields, TurbSim seed 1 for training, seeds 1-2 for the supervisor's evaluations, 14 episodes); paired tuned-ROSCO
baselines at 150 s in `~/wtrl_rt_range`; objective Cw (regulation target, tower / blade constrained per wind speed,
energy <= 1 %); 300 episodes; arms: agent-written reward (`trwCwR3t`) and fixed reward v3 with tower weight 10
(`tgCwR3L10`), 3 seeds each.

Held-out at 150 s (range winds, TurbSim seeds 3-6, 28 episodes, against the tuned ROSCO):

| arm | seed | selected episode | Cw | per-wind violation | power / speed / tower / blade | pitch travel x tuned ROSCO |
|---|---|---|---|---|---|---|
| agent reward | 0 | 224 | **+1.12** | 0.00 % | +0.7 / +1.5 / +2.6 / +1.3 | 1.13 |
| agent reward | 1 | 280 | **+6.22** | 0.12 % | +6.9 / +10.3 / +3.6 / +2.3 | 1.34 |
| agent reward | 2 | 256 | **+7.07** | 0.16 % | +8.8 / +11.9 / +1.3 / +1.3 | 1.21 |
| fixed reward | 0 | 64 | -27.87 | 1.54 % | +3.0 / +2.7 / -1.2 / +0.7 | 1.08 |
| fixed reward | 1 | 64 | -7.94 | 0.67 % | +4.7 / +6.2 / -0.4 / +0.6 | 1.06 |
| fixed reward | 2 | 152 | -3.67 | 0.38 % | +3.2 / +4.8 / +0.6 / +0.2 | 1.04 |

Per wind (12 -> 24 m/s): the agent runs hold the tower term at or above -1.9 % everywhere (seed 1: +10.4 +8.6 +4.0 -1.8
-0.1 +2.5 +1.6) and gain their regulation from 18 m/s up (power MSE +1 to +4 % for seed 0, +11 to +21 % for seeds 1
and 2); all three lose power MSE at 12 m/s
(-3.9 / -15.2 / -4.1 %), which the objective allows because regulation is its set-mean target. The fixed reward finds
the same kind of regulation gain at a smaller size (+3 to +10 % from 18 m/s up) but pays with the tower at 12-16 m/s
(-3.3 to -7.2 % at 12 m/s, down to -4.8 % at 16 m/s): it does not hold the per-wind constraint in any seed.

600 s range set (seeds 3-6, 28 episodes, against the ORIGINAL GSPI; the best seed of each arm by held-out Cw):

| controller | J | pitch travel x GSPI | power / speed / tower / blade | tower term 12 -> 24 m/s |
|---|---|---|---|---|
| tuned ROSCO alone | 15.24 | 1.15 | 23.1 / 28.9 / 5.3 / 3.6 | 7.6 16.6 3.1 4.5 2.1 2.0 1.3 |
| + fixed-reward residual (seed 2) | 16.41 | 1.19 | 25.2 / 31.9 / 5.3 / 3.3 | 3.4 14.4 0.9 5.6 4.4 4.5 3.8 |
| **+ agent-reward residual (seed 2)** | **19.62** | 1.38 | 29.4 / 36.9 / 7.6 / 4.5 | 14.7 23.2 6.0 3.6 1.7 1.9 1.9 |

Paired bootstrap over the 28 episodes against the tuned ROSCO alone: agent residual **J +4.37 [+4.12, +4.61]**, power
+6.27 [+5.87, +6.68], speed +8.01 [+7.72, +8.35], tower +2.25 [+1.74, +2.74], blade +0.96 [+0.36, +1.41] - every term's
interval excludes zero; fixed residual J +1.17 [+0.88, +1.41], tower -0.04 [-0.53, +0.42], blade -0.33 [-0.97, +0.15].
The agent row's worst per-wind differences against the tuned ROSCO are tower -0.9 % at 18 m/s and power -1.1 % at
12 m/s; the episodes are four times longer than any the policy trained on.

Reading. (1) This is the answer to the question of s32 under the corrected formulation: **yes, on a PI base** - the
agent-supervised residual adds regulation (+6 to +8 points at 600 s) AND both fatigue loads (+2.3 / +1.0) on top of a
tuned industrial baseline, holds the loads per wind speed, and costs 20 % more pitch travel (1.38x vs 1.15x GSPI).
(2) The gain is agentic in the sense that matters for the paper: with the same base, winds, budget, objective and
selection rule, the fixed reward with a tower weight of 10 is negative on held-out Cw in 3 of 3 seeds and the
agent-written reward positive in 3 of 3 (mean +4.8 vs -13.2). What the agent changes is the SHAPE of the reward
(tanh-saturated regulation terms gated by the wind label, bounded load terms centred on the baseline level - e.g.
`-38*region_w*tanh(|d_wg|/.005) - .85*tanh(max(0, load_t-1)) - .95*tanh(max(0, load_b-1)) - .1*tanh(act)` for seed 2),
a change of vocabulary that a weight search over the fixed expression cannot reach. (3) Scale: the scheduled MPC of s33 is at 28.95 with 2.14x
travel on the same episodes; the agent layer is an improvement of the industrial PI loop, not a substitute for the
model-based controller. (4) Open: the 600 s rows of the other two agent seeds and of the fixed seeds (queued in
`campaign_next3.sh`); the same comparison on the canonical 150 s bank (`campaign_next2.sh`, running); n = 3 per arm.

Artefacts: `~/wtrl/exp/{trwCwR3t,tgCwR3L10}_s{0,1,2}/` (`eval_heldout_s3456_ckpt_best.json` = 150 s range winds vs the
tuned ROSCO, `eval_range_TIB_s3456.json` = 600 s vs the original GSPI), discarded first attempts `*_badlabel`.

## 35. Clean re-run on the low-turbulence bank: the negative result survives the fixes; the agent layer pays where the baseline leaves room (2026-09-20 07:01 - 13:41; `campaign_next2.sh`)

Why. All 18 runs of s32 carried defects 2 and 3 (oracle-rule subset in training, agent candidates ranked by F). The
three informative arms were re-run with the corrected code on the same base (tuned ROSCO), the same paired baselines
(`~/wtrl_rt`), the same canonical 150 s bank (8 / 12.5 / 15 m/s, turbulence intensity 8 %) and the same budget:
agent-written reward under Cw (`trwCwF3t`), fixed reward with tower weight 10 under Cw (`tgCwF3L10`), agent-written
reward under CTw (`trwTwF3t`), 3 seeds each. Identity at episode 0: Cw -0.02 / -0.01 / -0.01.

Held-out (wind seeds 3-6, 12 episodes, against the tuned ROSCO; J against the original GSPI, tuned ROSCO alone = 14.2):

| arm | seed | selected episode | objective | per-wind violation | power / speed / tower / blade | J vs GSPI |
|---|---|---|---|---|---|---|
| agent reward, Cw | 0 | 0 | -0.01 | 0 | tuned ROSCO alone | 14.21 |
| agent reward, Cw | 1 | 168 | -2.46 | 0.14 % | +0.5 / +0.2 / -0.4 / -0.4 | 14.12 |
| agent reward, Cw | 2 | 168 | -1.97 | 0.09 % | -0.1 / -0.1 / -0.5 / -0.2 | 13.99 |
| fixed reward (tower weight 10), Cw | 0, 1, 2 | 0, 0, 0 | -0.01 | 0 | tuned ROSCO alone | 14.2 |
| agent reward, CTw | 0 | 280 | -1.18 | 0 | +1.6 / +1.7 / -1.2 / +2.8 | 15.15 |
| agent reward, CTw | 1, 2 | 0, 0 | -0.1 | 0 | tuned ROSCO alone | 14.2 |

In training the fixed reward never beats episode 0 (tower -2.4 to -13 % and speed MSE -1 to -11 % in every
evaluation of seed 0); the two agent runs that leave episode 0 find checkpoints worth +0.3 / +1.3 on the supervisor
winds that do not hold on held-out winds (tower -1.3 / -1.4 % at 15 m/s). The tower-target run gains regulation
(+3.6 % at 15 m/s) and loses its own target (tower -3.8 % at 15 m/s).

Reading. (1) With the formulation fixed, **no arm finds a load-holding gain on this bank** - the negative result of
s32 was not an artefact of the defects. (2) Together with s34 the picture is consistent: on the low-turbulence
8 / 12.5 / 15 m/s set the tuned ROSCO already regulates as well as the MPC (s29: 40 % vs 43 % at 15 m/s) and there is
nothing for a residual to take without touching the tower; on the class-B range from 18 m/s up the tuned ROSCO trails
the MPC by 20+ points of regulation, and there the agent-written reward takes 6-8 of them with the loads held. The
agent layer pays where the baseline leaves room, and the paper has to say so. (3) The seeds 7-10 column is not shown:
it carries the identity offset of s32 (one chaotic 8 m/s episode; -18 at the zero residual).

Artefacts: `~/wtrl/exp/{trwCwF3t,tgCwF3L10,trwTwF3t}_s{0,1,2}/`.

## 36. Model error on the range set: the scheduled compensated MPC is flat over +-15 %, the nominal MPC is not (2026-09-20 13:41 - 22:59; `campaign_next3.sh`)

The estimator's robustness had been measured on the low-turbulence set (s19, s23) and at x0.95 on the range set
(s28). Here: the Cp/Ct table of the MPC's model scaled by 0.85 / 0.95 / 1 / 1.15, held-out range seeds 3-6 (28
episodes, 600 s, class B); the x0.95 rows of the nominal and unscheduled controllers are the same 28 episodes of s28.

| controller | Cp/Ct x | J | travel x GSPI | power / speed / tower / blade | tower term 12 -> 24 m/s | worst per-wind term |
|---|---|---|---|---|---|---|
| nominal MPC | 0.85 | 21.38 | 2.18 | 40.2 / 27.3 / 15.9 / 2.2 | 49 40 15 7.1 2.9 0.4 -3.3 | speed -31.0 % at 12 m/s |
| nominal MPC | 0.95 | 28.17 | 2.06 | 43.3 / 45.5 / 19.7 / 4.2 | 58 48 18 9.5 4.5 2.1 -0.7 | -0.7 |
| nominal MPC | 1 | 28.49 | 2.01 | 41.5 / 46.7 / 21.5 / 4.2 | 58 48 21 11.4 6.8 3.7 0.9 | none |
| nominal MPC | 1.15 | 23.07 | 1.86 | 28.2 / 32.8 / 24.4 / 6.9 | 65 53 25 14.7 8.4 3.8 1.4 | power -17.4 % at 16 m/s |
| offset-free, unscheduled | 0.85 | 29.33 | 2.09 | 48.5 / 55.3 / 11.1 / 2.4 | 49 35 9.5 0.7 -2.2 -5.0 -8.9 | tower -8.9 % at 24 m/s |
| offset-free, unscheduled | 0.95 | 29.59 | 2.02 | 47.9 / 53.3 / 14.1 / 3.1 | 54 39 12.5 2.9 0.0 -3.3 -6.6 | tower -6.6 % |
| offset-free, unscheduled | 1 | 29.51 | 1.98 | 47.3 / 52.4 / 15.0 / 3.4 | 56 41 13.5 3.7 0.4 -2.5 -6.3 | tower -6.3 % |
| offset-free, unscheduled | 1.15 | 29.07 | 1.90 | 45.4 / 48.9 / 17.7 / 4.3 | 61 45 17.4 5.8 1.5 -1.5 -4.5 | tower -4.5 % |
| **offset-free, scheduled (s33)** | 0.85 | **28.81** | 2.28 | 47.1 / 53.8 / 12.7 / 1.6 | 49 35 9.5 0.9 -1.1 -1.4 -2.1 | -2.1 (blade, 20 m/s) |
| **offset-free, scheduled** | 0.95 | **29.00** | 2.18 | 46.4 / 51.7 / 15.5 / 2.4 | 54 39 12.5 3.1 0.8 0.1 -1.1 | -1.1 (tower, 24 m/s) |
| **offset-free, scheduled** | 1 | **28.95** | 2.14 | 45.7 / 50.7 / 16.5 / 2.9 | 56 41 13.5 3.9 1.6 0.2 0.1 | -0.8 (blade, 16 m/s) |
| **offset-free, scheduled** | 1.15 | **28.44** | 2.04 | 43.4 / 46.9 / 19.5 / 3.9 | 61 45 17.4 5.9 3.1 3.1 1.8 | none |

Paired bootstrap over the 28 episodes (difference in J to the same controller with the exact model):
nominal -7.11 [-8.54, -5.83] / -0.32 [-0.85, +0.19] / -5.43 [-6.21, -4.69] at x0.85 / x0.95 / x1.15;
unscheduled offset-free -0.18 [-0.48, +0.15] / +0.08 / -0.44 [-0.62, -0.26];
scheduled -0.14 [-0.46, +0.19] / +0.05 [-0.10, +0.22] / -0.51 [-0.75, -0.26].
Scheduled - nominal at the same error: **+7.42 [+5.94, +9.02]** at x0.85, **+5.38 [+4.38, +6.35]** at x1.15.

Reading. (1) On the range set +-15 % of aerodynamic model error costs the nominal MPC 5.4-7.1 J - concentrated near
rated (speed MSE -31 % at 12 m/s, power MSE -17 % at 16 m/s) - and the compensated controllers 0.1-0.5 J. The
"x0.95 costs 0.7 J" of s28 was the benign end of the range; the fragility of the nominal MPC is real on the class-B
set as well. (2) The schedule and the estimator compose: the scheduled controller stays at 28.4-29.0 over the whole
range, and its high-wind tower term stays within -2.1 % at x0.85 and is non-negative at x1.15, where the unscheduled
controller is at -8.9 % / -4.5 %. With the model 15 % too weak the -1 % per-wind rule is missed by one point (tower
-1.1 / -1.4 / -2.1 % at 20 / 22 / 24 m/s, blade -2.1 % at 20 m/s): to be stated as such. (3) This completes the evidence
for the main-claim controller of s33: better regulation and tower fatigue than the tuned ROSCO at every wind speed
(+13.7 J), and no dependence on the accuracy of the aerodynamic model that the nominal MPC needs.

Infrastructure. The second stage of `campaign_next3.sh` (600 s rows of the other range-residual seeds) failed on a
broken line continuation written through a shell heredoc (fixed in 4f7c91e); those rows are produced by the last
stage of `campaign_next4.sh`, which evaluates every range-wind run that lacks its 600 s file.

Artefacts: `~/wtrl/exp/mpcsearch600/qtsched/eval_{sched,offset,nominal}_cp*_s3456.json`.

## 37. The agent layer with five seeds and two more controls: a structural reward search buys the level, the agent is what holds the loads (2026-09-20 22:59 - 09-21 10:38; `campaign_next4.sh`, `scripts/dev/range_residual_table.py`, `docs/tables/range_residual.csv`)

Setup as in s34 (tuned ROSCO base, above-rated range winds at 150 s for training, per-wind objective Cw, 300
episodes). Added: seeds 3-4 of the two s34 arms, and two controls with 3 seeds each - the fixed reward with the
J-tuned weights (the default the agent starts from) and a RANDOM reward structure drawn from the fixed grammar with
the same fork verification as the agent (the random-vocabulary control of v2 s9). Every run was then evaluated on the
600 s range set (TurbSim seeds 3-6, 28 episodes) against the original GSPI; tuned ROSCO alone = 15.24 at 1.15x travel.

| arm | held-out Cw at 150 s, per seed | positive | 600 s J, per seed | mean | J - tuned ROSCO | travel x GSPI | worst per-wind tower / blade difference to the tuned ROSCO at 600 s, per seed |
|---|---|---|---|---|---|---|---|
| **agent-written reward** | +1.12 +6.22 +7.07 +5.79 -2.50 | **4/5** | 16.80 20.28 19.62 20.12 19.51 | **19.27** | **+4.02** | 1.29-1.61 | tower -0.7 -0.4 -0.9 0.0 -4.0; blade never worse |
| random reward structure | -9.74 -12.31 +6.27 | 1/3 | 19.16 19.11 17.96 | 18.74 | +3.50 | 1.23-1.64 | tower -3.9 -2.9 0.0; blade 0 0 -2.0 |
| fixed reward, J-tuned weights | -35.66 +0.78 -4.54 | 1/3 | 16.17 18.94 16.35 | 17.15 | +1.91 | 1.21-1.32 | tower -5.3 -2.3 -4.1; blade -2.7 -1.2 -4.8 |
| fixed reward, tower weight 10 | -27.87 -7.94 -3.67 -17.16 -37.29 | 0/5 | 16.12 16.86 16.41 15.22 17.29 | 16.38 | +1.14 | 1.19-1.29 | tower -4.6 -3.0 -4.2 -3.8 -4.3; blade -1.9 -4.2 -4.8 -4.7 -2.0 |

Every 600 s difference to the tuned ROSCO has a paired 95 % interval that excludes zero except the fourth
tower-weight-10 seed (-0.03 [-0.15, +0.08]); the agent seeds are +1.55 [1.41, 1.69], +5.04 [4.47, 5.57], +4.37 [4.12, 4.61],
+4.88 [4.55, 5.24], +4.26 [3.88, 4.68]. Held-out Cw positive: agent 4/5 against 2/11 for the pooled controls, one-sided
Fisher exact p = 0.036. Both fatigue loads within -1 % of the tuned ROSCO at EVERY wind speed at 600 s (the
objective's own rule, applied to episodes four times longer than the training ones): agent 4/5, random structure 0/3,
fixed rewards 0/8 (p = 0.003 by the same test).

Reading.
1. **Level.** What lifts J by 3.5-4 points over the tuned ROSCO is a reward whose STRUCTURE was searched with fork
   verification: the random grammar reaches 18.74 against the agent's 19.27, the two fixed rewards 16.4-17.2. This
   repeats v2 s9 on a new base, objective and wind set: the proposer does not matter much for the level.
2. **Constraint.** What the agent adds is that the gain respects the per-wind load rule on winds it never saw: 4 of 5
   seeds hold tower and blade fatigue within 1 % at every wind speed at 600 s (the fifth loses 4.0 % of tower fatigue
   at one wind speed), no control run does. The random structures find the regulation (+5 to +9 % power / speed MSE at
   150 s) and lose 3-5 % of tower fatigue at 16-18 m/s; the fixed rewards lose tower fatigue at 12-16 m/s and, at
   600 s, blade fatigue as well.
   The agent reads the per-wind terms of the objective and writes load terms that penalise growth above the baseline
   level in the regions where it appears (s34); a grammar sampled blindly cannot do that.
3. **Size and cost.** +4 J on 15.2 (regulation +1 to +8 points, tower +2 to +8, blade +1 to +3) for 12-40 % more
   pitch travel than the tuned ROSCO (1.29-1.61x vs 1.15x GSPI). The scheduled MPC of s33 is at 28.95 with 2.14x.
4. **Claim this supports.** An LLM-supervised residual is an improvement layer for a tuned industrial PI loop in
   the operating range where that loop leaves regulation on the table (class B, from 18 m/s up): better regulation
   and both fatigue loads, per wind speed, at 600 s, in 4 of 5 seeds - and nothing on the low-turbulence set where
   the tuned loop already matches the MPC (s35). n = 5 / 3 / 3 / 5; the level comparison agent-vs-random is not
   significant at this n, the load-holding comparison is.

Artefacts: `~/wtrl/exp/{trwCwR3t,tgCwR3L10}_s{0..4}/`, `~/wtrl/exp/{trrCwR3t,tgCwR3t}_s{0,1,2}/`
(`eval_heldout_s3456_ckpt_best.json` at 150 s vs the tuned ROSCO, `eval_range_TIB_s3456.json` at 600 s vs the GSPI).
Machine idle after 10:38 on 2026-09-21.

## 38. The verified design loop finds the wind schedule by itself, does not beat the hand design, and the proposer does not decide the outcome (2026-09-21 11:15 - 21:42; `scripts/mpc_design_search.py`, `campaign_design.sh`, `scripts/dev/design_search_table.py`, `docs/tables/design_search.csv`)

Question (narrative decision of 2026-09-21: present the agent as the whole supervised design system, whose output
with a model is the MPC). Is the main-claim controller of s33 something the verified design loop produces by itself,
and does the proposer matter?

Setup. Space = the six MPC parameters of s21 plus the tower-weight schedule (`qt_ratio` in [1, 8], `v_lo` in [13, 22]
m/s, `v_span` in [1, 8] m/s; ratio 1 = unscheduled). Selection objective S = J - 4 V on range seeds 1-2 at 12 / 16 /
20 / 24 m/s (8 episodes, 600 s, class B), V = mean over wind speeds of the shortfalls of the four per-wind terms
below -1 % (the per-wind load rule as a soft constraint). Every arm starts from the hand-set UNSCHEDULED offset-free
reference (S 20.03 = J 28.93 - 4 x 2.23). Budget 24 verified candidates; proposers llm / es / random, two repeats each,
arms blind to each other. Report: the best candidate of every arm-run on range seeds 3-6 at all seven wind speeds
(28 episodes, never seen by a proposer).

Selection side:

| arm-run | incumbent S after 8 / 16 / 24 | proposals that beat the reference | failed (unstable) | best candidate |
|---|---|---|---|---|
| llm, repeat 0 | 24.52 / 25.43 / 27.73 | 13 / 24 | 0 | N15 r .5 qt 4 -> x1.25 over 18-23 m/s, travel 1.60x |
| llm, repeat 1 | 23.92 / 25.14 / 25.79 | 12 / 24 | 0 | N20 r .38 qt 4.4 -> x1.3 over 20-24 m/s, 2.01x |
| es, repeat 0 | 22.48 / 22.48 / 25.96 | 6 / 24 | 0 | N20 r .38 qt 4 -> x1.52 over 19.8-22.1 m/s, 1.99x |
| es, repeat 1 | 23.49 / 25.97 / 26.67 | 9 / 24 | 0 | N20 r .58 qt 3 -> x1.1 over 17.8-23 m/s, 1.49x |
| random, repeat 0 / 1 | 20.03 throughout | 0 / 24, 0 / 24 | 3, 4 | the reference |

Held-out side (28 episodes; S and V recomputed on these episodes; paired 95 % intervals on J):

| controller | J | S | V | travel x GSPI | power / speed / tower / blade | worst per-wind term | J - hand-set reference | J - hand design (s33) |
|---|---|---|---|---|---|---|---|---|
| hand-set unscheduled reference (= both random runs) | 29.51 | 25.66 | 0.96 | 1.98 | 47.3 / 52.4 / 15.0 / 3.4 | tower -6.3 % at 24 m/s | | +0.56 |
| **hand-designed schedule (s33)** | 28.95 | **28.95** | 0 | 2.14 | 45.7 / 50.7 / 16.5 / 2.9 | -0.8 | -0.56 [-0.67, -0.43] | |
| nominal MPC | 28.49 | 28.49 | 0 | 2.01 | 41.5 / 46.7 / 21.5 / 4.2 | none | -1.02 | -0.46 [-1.03, +0.07] |
| es, repeat 0 | 28.16 | 28.16 | 0 | 1.94 | 42.3 / 46.7 / 20.3 / 3.4 | -0.3 | -1.35 [-1.53, -1.17] | -0.79 [-0.95, -0.64] |
| llm, repeat 1 | 28.34 | 27.95 | 0.10 | 1.95 | 42.3 / 47.7 / 19.9 / 3.4 | power -1.7 % at 12 m/s | -1.17 [-1.46, -0.86] | -0.62 [-0.89, -0.33] |
| es, repeat 1 | 27.09 | 27.09 | 0 | **1.45** | 39.9 / 43.7 / 20.1 / 4.6 | -0.5 | -2.42 [-2.72, -2.13] | -1.86 [-2.18, -1.55] |
| llm, repeat 0 | 28.40 | 26.77 | 0.41 | **1.56** | 42.0 / 47.7 / 18.5 / 5.3 | power -3.2 % at 12 m/s, tower -1.7 % at 24 m/s | -1.11 [-1.38, -0.83] | -0.55 [-0.82, -0.27] |
| tuned ROSCO | 15.24 | 15.24 | 0 | 1.15 | 23.1 / 28.9 / 5.3 / 3.6 | none | -14.27 | -13.71 |

The hand-designed schedule on the SELECTION episodes (evaluated afterwards): S 23.28 (J 28.48, V 1.30: tower -1.8 % and
blade -1.6 % at 20 m/s, blade -4.8 % at 24 m/s), i.e. below all four searched designs on the set they were selected on.

Reading.
1. **The loop discovers the structure.** All four sequential runs end on a wind-scheduled tower weight starting at
   18-20 m/s, from an unscheduled start, within 24 simulations; all four repair the per-wind violation of the reference
   on winds they never saw (held-out V 0.96 -> 0 / 0 / 0.10 / 0.41) and raise the held-out design objective by 1.1-2.5 S.
   Random search over the same box: 0 of 48 proposals beat the reference, 7 are unstable. Two searched designs cut the
   pitch travel to 1.45-1.56x GSPI (the hand design: 2.14x) at a cost of 0.6-1.9 J.
2. **It does not beat the hand design.** On held-out S the hand-designed schedule (28.95) is above every searched design
   (26.8-28.2), and the nominal MPC (28.49) too. On the selection episodes the order is reversed. Over the five
   scheduled designs the rank correlation between selection S and held-out S is **-0.9**: eight episodes with two
   realisations per wind speed separate "scheduled vs unscheduled" (20.0 vs 23.3-27.7) reliably and cannot rank
   schedules - the search fits the realisation noise of the per-wind terms (blade -4.8 % at 24 m/s on seeds 1-2 is
   within -0.8 % on seeds 3-6). Same lesson as s33: a per-wind rule needs more realisations per wind speed than a set-mean J.
3. **The proposer.** Final quality: llm 27.73 / 25.79 vs es 25.96 / 26.67 on selection, 26.77 / 27.95 vs 28.16 / 27.09
   held-out - no difference. What differs is the quality of the individual proposal: 25 of 48 llm proposals beat the
   reference against 15 of 48 es proposals (one-sided Fisher exact p = 0.031), and the llm runs are ahead at a budget
   of 8 (24.5 / 23.9 vs 22.5 / 23.5). This matches s21 and the older proposer comparisons: the LLM is a better prior,
   not a better optimiser.
4. **What can be claimed.** (a) The headline numbers are those of the MPC family against the tuned ROSCO (+11.9 to
   +13.7 J held-out for every per-wind-feasible design in the table) and do not depend on who designed the controller. (b) "A verified
   sequential design loop recovers the wind schedule and the per-wind load rule from an unscheduled start in 24
   simulations; random search does not; the LLM proposer makes better individual proposals and is equal to a (1+lambda)
   search in the result." (c) NOT: "the agent designs a better controller than the engineer", and not "the LLM beats
   other search".

Artefacts: `~/wtrl/exp/mpcdesign/{llm,es,random}_r{0,1}/{history.jsonl,best.json}`, `~/wtrl/exp/mpcdesign/heldout/`,
`~/wtrl/exp/mpcdesign/cache/` (the s33 controller on the selection episodes included). Machine idle after 21:50.

## 39. Probe: the agent-supervised residual stacked on the scheduled MPC adds to it (2026-09-21 22:12 - 09-22 04:52; `campaign_probe_mpc.sh`, `scripts/dev/probe_decision.py`)

Why. Every earlier residual-on-MPC arm (s17-s24, s30) was run on the low-turbulence 150 s bank, under J, before the
defects of s32 were fixed, and found nothing. s34-s35 showed the layer pays where the base leaves room. The probe
repeats the s34 setup with the main-claim controller of s33 as the base.

Setup. Base = offset-free LPV-MPC with the wind-scheduled tower weight (`configs/mpc_sched_s33.json`, new
`--base_mpc_json` in `train.py` / `evaluate.py` / `make_baselines.py`). Paired baselines = THE MPC'S OWN zero-residual
rollouts on the range winds at 150 s (`~/wtrl_mpc_range`; identity J = Cw = 0.00 at all seven wind speeds), so Cw > 0
reads "regulation gained on top of the MPC with both loads held per wind speed". Training winds, budget, objective and
selection as in s34. Arms: agent-written reward (3 seeds), fixed reward with the J-tuned weights (2 seeds).

Decision rule, committed (2676b80) before any result existed (user instruction: more seeds if the effect is good,
otherwise examine the agent design): GO iff the held-out Cw is positive in >= 2 of 3 agent seeds AND the mean 600 s
difference in J to the MPC alone exceeds +0.5 with >= 2 of 3 paired 95 % intervals above zero.

Result (150 s held-out = range seeds 3-6 against the MPC; 600 s = the same seeds against the original GSPI, the MPC
alone is 28.95 with 45.7 / 50.7 / 16.5 / 2.9 at 2.14x pitch travel):

| run | selected episode | held-out Cw (violation) | 600 s J | J - MPC alone [95 %] | power / speed / tower / blade - MPC alone | travel x GSPI |
|---|---|---|---|---|---|---|
| agent reward, seed 0 | 280 | +0.56 (0.10 %) | 29.72 | +0.76 [+0.56, +0.97] | +0.2 / +1.4 / +0.9 / +0.5 | 2.22 |
| agent reward, seed 1 | 280 | +1.13 (0) | 29.25 | +0.30 [+0.01, +0.58] | -1.7 / +0.7 / +1.3 / +0.8 | 2.29 |
| agent reward, seed 2 | 256 | +4.96 (0.12 %) | **30.62** | **+1.67 [+1.30, +2.02]** | +0.7 / +3.4 / +2.1 / +0.4 | 2.62 |
| fixed reward, seed 0 | 0 | +0.62 (0) | 29.04 | +0.09 [-0.01, +0.19] | -0.2 / +0.7 / 0.0 / -0.2 | 2.14 |
| fixed reward, seed 1 | 0 | +0.59 (0) | 29.02 | +0.06 [-0.06, +0.19] | -0.2 / +0.7 / 0.0 / -0.3 | 2.13 |

Held-out Cw positive 3 / 3, mean 600 s difference +0.91, intervals above zero 3 / 3: **GO**; the seeds-and-controls
campaign (`campaign_probe_mpc_seeds.sh`) started at 04:55. (The fixed-reward rows are the untrained policy: the fixed
reward never produces a checkpoint that beats episode 0; their +0.6 Cw is the size of the initial network's noise.)

Reading. (1) My stated prior (+0 to +1.5 J, probably nothing) was too pessimistic about significance and right about
size: the agent-reward residual adds **+0.3 to +1.7 J on top of the strongest controller of this repository, in 3 of 3
seeds, on held-out 600 s episodes**, and the tower-fatigue term carries it (+0.9 / +1.3 / +2.1 points, every interval
above zero), with speed MSE +0.7 to +3.4 and blade +0.4 to +0.8. (2) It is agentic in the sense of s34 / s37: the fixed
reward learns nothing on this base. (3) The cost: power MSE at 12-14 m/s gets worse in all three seeds (-2 to -8 points
at those winds; set mean -1.7 to +0.7) and the pitch travel rises 4-22 %. Tower fatigue is never worse than -0.8
points at any wind speed, blade fatigue never worse than -1.1.

Diagnostic (NOT a clean result - checkpoint and gate were chosen after looking at held-out seeds 3-6). The FINAL
checkpoint of seed 0, held-out at 150 s against the MPC, per wind (power / speed / tower / blade): 12: -4/8/-5/1,
14: -2/0/-5/-2, 16: 3/7/-1/0, 18: 12/14/1/2, 20: 15/15/0/2, 22: 16/16/3/3, 24: 16/16/3/1 - the damage is near rated, the
gain (+12 to +16 % regulation on top of the MPC, loads held) from 18 m/s up; the same split is visible on the
supervisor winds in every late evaluation of both seeds. With the residual gated off below 15 m/s (1 above 17 m/s,
10 s low-pass of the wind estimate; `EnvConfig.residual_gate`, `--residual_gate`) the same checkpoint reads Cw
**+8.65 with zero violation** (+8.4 / +8.9 / +1.2 / +1.4); seed 1's final checkpoint stays negative with the gate
(-39: its high-wind policy is poor), so the gate has to be in the training loop. Clean test: `campaign_gate.sh`
(gate from scratch, selection on supervisor winds, report on fresh TurbSim seeds 7-10), queued after the seeds campaign.

Zero-simulation headroom (design-search cache, 137 stable candidates, selection episodes): per-wind-optimal MPC
parameters would give at most ~+1.5 J over the best single setting (28.0 -> 29.6 with the per-wind rule), with the
pitch-increment weight rising with wind (0.26-0.29 at 12-16 m/s, 0.5-0.6 at 20-24 m/s) - the alternative "agent through
the MPC" has about the same ceiling as what the parallel residual already delivers.

Artefacts: `~/wtrl/exp/{mrwCwR3t,mgCwR3t}_s*/`, `~/wtrl/exp/probe_decision.{txt,json}`, `~/wtrl_mpc_range/baselines/`.

## 40. The layer on the scheduled MPC with five seeds and two controls: a verified structural reward search adds ~+1 J to the strongest controller; the proposer is not what makes the difference here (2026-09-22 04:55 - 16:45; `campaign_probe_mpc_seeds.sh`, `scripts/dev/range_residual_table.py --base mpc`, `docs/tables/mpc_residual.csv`)

Setup as in s39 (scheduled MPC base, its own paired baselines, range winds at 150 s, per-wind Cw). Added: seeds 3-4 of
the agent arm, three seeds of the random reward structure with the same fork verification, seeds 2-4 of the fixed
reward. Every run on the 600 s range set (seeds 3-6, 28 episodes) against the original GSPI; the MPC alone is 28.95
(45.7 / 50.7 / 16.5 / 2.9) at 2.14x pitch travel.

| arm | 150 s held-out Cw vs the MPC, per seed | 600 s J - MPC alone, per seed [interval above zero] | mean | worst per-wind tower / blade difference to the MPC (600 s) | travel x GSPI |
|---|---|---|---|---|---|
| agent-written reward | +0.56 +1.13 +4.96 +0.93 +1.24 (5/5) | +0.76 +0.30 +1.67 +0.68 +0.64 [5/5] | **+0.81** | -0.8/-0.3, 0/-0.1, 0/-1.1, -1.7/0, 0/-0.7 | 2.15-2.62 |
| random reward structure | -0.30 +3.85 +5.28 (2/3) | +0.63 +0.95 +1.72 [3/3] | **+1.10** | -0.9/-0.8, -1.0/-1.1, -1.3/-1.1 | 2.34-2.47 |
| fixed reward, J-tuned weights | untrained policy in 5/5 (episode 0) | +0.09 +0.06 +0.12 +0.13 +0.10 [2/5] | +0.10 | within 0.2 / 1.8 | 2.13-2.14 |

Per-seed intervals of the agent arm: +0.76 [0.55, 0.97], +0.30 [0.00, 0.58], +1.67 [1.30, 2.02], +0.68 [0.33, 1.02],
+0.64 [0.43, 0.84]; of the random structure: +0.63 [0.43, 0.83], +0.95 [0.79, 1.11], +1.72 [1.53, 1.92]. The fixed
reward's +0.1 is the untrained network's noise (its selected checkpoint is episode 0 in every seed).

Reading.
1. **Additive on the strongest controller.** Eight of eight runs with a searched reward structure improve the scheduled
   MPC on held-out 600 s episodes, by +0.3 to +1.7 J (mean +0.8 to +1.1), with tower and blade fatigue never worse than
   1.7 points at any wind speed. The gain is mostly tower fatigue (+0.8 to +2.1 points) and speed MSE (+0.7 to +3.5);
   the cost is 0 to 22 % more pitch travel and 2-8 points of power MSE at 12-14 m/s.
2. **The proposer is not what makes the difference on this base.** The random grammar matches or beats the LLM on
   level (+1.10 vs +0.81) and holds the loads equally well (worst tower -1.3 vs -1.7). This differs from the tuned-PI
   base (s37), where the random structures found the level but lost 3-5 % of tower fatigue at 16-18 m/s: there the
   base left the loads to the residual, here the MPC already holds them and the residual's job is easier. A sign test
   is meaningless here (the untrained network reads +0.6 Cw on this base); the paired 600 s intervals are the evidence.
3. **What the paper can say.** The agent, understood as the supervised loop (structural reward search + simulation
   verification), improves even the model-based main controller: 28.95 -> 29.6-30.7 at 600 s, loads held per wind. The
   LLM's specific contribution is established on the PI base (s37: loads held 4/5 vs 0/11), not on the MPC base (n = 5
   vs 3, no difference). Headline stack on the same 28 held-out episodes: tuned ROSCO 15.24 -> scheduled MPC 28.95 ->
   MPC + searched residual 29.6-30.7, i.e. up to +15.4 J over the tuned industrial baseline.
4. The wind gate of s39 is the next design step (near-rated power MSE is the only term the layer costs); `campaign_gate.sh`
   started at 16:47 and reports on fresh wind seeds.

Artefacts: `~/wtrl/exp/{mrwCwR3t_s0..4,mrrCwR3t_s0..2,mgCwR3t_s0..4}/`, `docs/tables/mpc_residual.csv`.

## 41. The wind-gated residual: the layer improves both bases on winds nothing has seen (2026-09-22 16:47 - 09-23 12:20; `campaign_gate.sh`, `scripts/dev/gate_table.py`, `docs/tables/gated_residual.csv`)

Design change (from the diagnostic of s39). The residual's pitch increment is multiplied by a gate g(v): 0 below
15 m/s, 1 above 17 m/s, linear between, on a 10 s low-pass of the controller's wind-speed estimate
(`EnvConfig.residual_gate`, `--residual_gate`). The layer then acts only where the base leaves room: s39 showed the
ungated residual gains 12-16 points of regulation on top of the MPC from 18 m/s up and damages 12-14 m/s. The gate is
in the training loop, so selection sees it; everything else is as in s34 / s39.

Arms: agent-written reward on the scheduled MPC (3 seeds), fixed reward on the same (2), agent-written reward on the
tuned ROSCO (3). Report: 600 s range set against the ORIGINAL GSPI on the held-out seeds 3-6 AND on fresh TurbSim
seeds 7-10, generated for this section and seen by no training, no selection and no earlier report. Paired 95 %
intervals against the run's own base on the same 28 episodes.

| controller | seeds 3-6: J, difference to its base | seeds 7-10 (fresh): J, difference | travel x GSPI | worst per-wind difference to the base (fresh) |
|---|---|---|---|---|
| tuned ROSCO alone | 15.24 | 15.93 | 1.12-1.15 | |
| + agent reward, gated, seed 0 | 19.35 **+4.10** [3.96, 4.26] | 20.14 **+4.21** [3.95, 4.45] | 1.45 | tower -1.0 at 18 m/s |
| + agent reward, gated, seed 1 | 19.52 +4.27 [4.07, 4.49] | 20.19 +4.26 [3.98, 4.55] | 1.42 | tower -2.7 at 18 m/s |
| + agent reward, gated, seed 2 | 20.04 +4.79 [4.58, 5.02] | **20.68** +4.76 [4.45, 5.06] | 1.40 | tower -1.4 at 18 m/s |
| scheduled MPC alone (s33) | 28.95 | 30.46 | 2.08-2.14 | |
| + agent reward, gated, seed 0 | 30.60 +1.65 [1.41, 1.90] | 31.98 +1.52 [1.30, 1.75] | 2.69 | tower -1.7 at 24 m/s |
| + agent reward, gated, seed 1 | 31.22 +2.27 [1.99, 2.55] | **32.54** +2.08 [1.53, 2.63] | 2.90 | tower -1.6 at 18 m/s |
| + agent reward, gated, seed 2 | 31.10 +2.15 [1.92, 2.38] | 32.42 +1.96 [1.65, 2.26] | 3.10 | tower -2.0 at 24 m/s |
| + fixed reward, gated, seed 0 | 31.20 +2.24 [1.96, 2.53] | 32.24 +1.78 [1.54, 2.03] | 3.06 | **tower -5.9 at 20 m/s** |
| + fixed reward, gated, seed 1 | 30.05 +1.10 [0.84, 1.35] | 31.62 +1.16 [0.69, 1.65] | 2.38 | power -1.9 at 16 m/s |

All sixteen rows have a paired interval above zero, on the held-out and on the fresh seeds alike.

Reading.
1. **The gate roughly doubles the stacking gain and it replicates on unseen winds.** On the scheduled MPC the ungated
   layer was +0.81 mean (s40); gated it is +2.02 (seeds 3-6) and +1.85 (fresh). On the tuned ROSCO the ungated layer
   was +4.02 at 1.29-1.61x travel (s37); gated it is +4.39 and +4.41 at **1.40-1.45x**, i.e. the same gain for a third
   less actuation, and now consistent across seeds instead of 4 of 5.
2. **Why it works.** Below 15 m/s the layer is switched off and the four terms are exactly the base's; the whole
   learning budget goes to 18-24 m/s, where the base leaves 12-22 points of regulation on the table. What remains is
   1-3 points of tower fatigue at one wind speed - the per-wind rule (-1 %) is met at every wind speed by 1 of 3
   agent seeds on the MPC and missed by 1-3 points by the others.
3. **The agent still holds the loads better than the fixed reward.** With the gate, the fixed reward finally learns
   on the MPC base (it was stuck at episode 0 without it, s39-s40) and reaches a similar J, but its best seed loses
   **5.9 points of tower fatigue at 20 m/s** on the fresh winds, where the agent seeds stay within 2.0. This is the
   same distinction as s37: a structural reward search buys the level, the agent's reward shape keeps the loads.
4. **Headline, on winds nothing in the pipeline has seen** (fresh seeds 7-10, 28 episodes of 600 s, IEC class B,
   12-24 m/s), against the ORIGINAL GSPI: the full stack reaches power MSE **-51.5 %**, generator-speed MSE **-58.0 %**,
   tower-base DEL **-18.9 %**, blade-root DEL -1.8 %, at 2.9x the baseline pitch travel. Against the TUNED ROSCO
   (the honest industrial reference of s29) the same row is power MSE -36 %, speed MSE -39 %, tower DEL -13 %. The
   cheap configuration - tuned ROSCO plus the gated agent layer, no model - reaches 20.68 at **1.40x** travel.

Artefacts: `~/wtrl/exp/{mrwCwG3t_s0..2,mgCwG3t_s0..1,trwCwG3t_s0..2}/eval_range_TIB_s{3456,78910}.json`,
references `~/wtrl/exp/mpcsearch600/fresh/eval_{sched_mpc,rosco_tuned}_s78910.json`, `docs/tables/gated_residual.csv`.
The fresh wind fields `U{12..24}_TIB_S{7..10}` and their GSPI baselines were added to the separate 600 s bank
(`~/wtrl/wind600`, `~/wtrl600`); the canonical bank was not touched.

Infrastructure note. The campaign survived two interruptions: a WSL VM restart and a full C: drive. Both left
truncated artefacts that the pipelines would have silently skipped or read as data - 7 zero-byte TurbSim fields (the
generator skips existing names) and 9 unreadable baseline `.npz` (the denominators of every percentage). They are
found and removed by `scripts/dev/integrity_check.py --since <time> [--delete]`, which is now the first thing to run
after any interruption.


## 42. Reporting the same results over a site wind distribution: the model-based part doubles, the learned layer thins out (2026-09-23; `scripts/dev/lifetime_del.py`, `docs/tables/lifetime_del.csv`)

Every table so far reports the mean of the per-wind reductions over an equally weighted set of mean wind speeds, which
is the objective's definition and is what the set was designed for. Fatigue work in wind energy aggregates over the
site wind distribution instead: IEC 61400-1 uses a Rayleigh distribution (Weibull shape 2, annual mean 0.2 x the class
reference speed, i.e. 10 / 8.5 / 7.5 m/s for classes I / II / III). Damage adds linearly (Palmgren-Miner) and a
damage-equivalent load of Wohler exponent m aggregates as `DEL_eq = (sum_i w_i DEL_i^m / sum_i w_i)^(1/m)`; the two
regulation metrics are mean squares and aggregate with m = 1. No new simulations: the script re-weights the per-episode
values of the fresh-seed evaluations (seeds 7-10, 12-24 m/s, 600 s) and reports the reduction against the same paired
GSPI baseline. The weights are renormalised over the evaluated bins, so these are above-rated equivalent loads and the
ratio, not the level, is the meaningful quantity.

Reduction vs the original GSPI, fresh seeds 7-10, IEC class I (Vave = 10 m/s); `equal` repeats the set mean used everywhere else.

| controller | power MSE w / eq | speed MSE w / eq | tower DEL w / eq | blade DEL w / eq |
|---|---|---|---|---|
| tuned ROSCO | 7.7 / 10.4 | 29.4 / 28.4 | 11.3 / 7.9 | 1.4 / 1.5 |
| scheduled MPC | 7.4 / 15.3 | 42.3 / 60.9 | **40.4** / 22.1 | -1.8 / -0.5 |
| tuned ROSCO + gated residual (s0/s1/s2) | 8.0-8.1 / 12.3-12.7 | 34.3-35.1 / 39.0-40.9 | 11.2-11.4 / 7.6-8.1 | 2.3-3.4 / 2.3-3.4 |
| scheduled MPC + gated residual (s0/s1/s2) | 7.7-7.8 / 16.1-16.3 | 43.7-44.9 / 64.9-65.6 | 40.2-40.6 / 21.6-23.4 | -1.9 to +0.8 / -0.6 to +1.7 |

Class sensitivity (Vave 10 -> 8.5 -> 7.5): the scheduled MPC's tower reduction rises 40.4 -> 43.8 -> 46.1 % while its
speed-MSE reduction falls 42.3 -> 34.8 -> 29.5 %; the gated layer's speed-MSE contribution on the tuned ROSCO falls
from +7-8 to +4-5 to +2.4-2.7 points. Full table in `docs/tables/lifetime_del.csv`.

Reading.
1. **The weighting nearly doubles the model-based controller's fatigue claim.** Tower-base DEL 22.1 % equally weighted
   -> 40.4 % (class I) -> 46.1 % (class III). Tower fore-aft fatigue is dominated by the near-rated bins, which carry
   both the largest loads and the largest probability, and that is exactly where the wind-scheduled tower weight acts.
   Against the tuned ROSCO the scheduled MPC is +32.8 % (class I) to +39.0 % (class III) of equivalent tower load.
2. **The same weighting thins out the learned layer**, because the gate switches it on only above 15 m/s, which is the
   tail of the distribution. On the tuned ROSCO base its weighted tower contribution is 0.0 +- 0.1 points (it was +4.4 J
   on the set mean, s41); what survives the weighting is +0.4 power MSE, +7-8 speed MSE and +0.8-2.0 blade DEL at class I,
   and about a third of that at class III. On the MPC base the class-I contribution is +0.3 power, +2.4 speed, +0.3 tower
   and +2.6 blade (seed 1, ratios of the weighted aggregates).
3. **Both statements must appear in the manuscript.** The set-mean numbers answer "over the above-rated operating range,
   what does each layer do", which is the question the objective was built for and the only fair way to compare
   controllers per wind speed; the Rayleigh numbers answer "what would a class-I site see over a year", which is the
   convention a wind-energy reviewer expects for a fatigue claim. Reporting only the first understates the MPC's tower
   result by half; reporting only the second understates the layer, whose gains sit in the rare high winds - and those
   are the winds in which the regulation error and the pitch duty are largest, not a niche.
4. Consequence for the narrative: the fatigue headline belongs to the model-based scheduling and should be reported
   Rayleigh-weighted, the regulation headline belongs to the gated agent layer and should be reported per wind speed
   over the range; neither is a substitute for the other, and the set mean J stays the single-number objective.


## 43. Novelty audit and protocol audit before the rewrite (2026-09-23; `docs/paper/literature_2026-09-23*.md`, `scripts/dev/transient_check.py`)

Four literature scans and two checks against our own artefacts, run before the manuscript was rewritten. The scans
are in four files, every entry carrying a URL and one of VERIFIED / VERIFIED (bib only) / UNVERIFIED; no number from
an UNVERIFIED source may be quoted anywhere. Publisher blocks (ScienceDirect, Wiley, IEEE Xplore, MDPI, PubMed all
403) skew the verified set towards Copernicus, IOPscience, arXiv and Zenodo, so absence from it reflects access, not
relevance.

### What the audit takes away
| claim we could have made | verdict | what the paper says instead |
|---|---|---|
| wind-scheduled MPC cost weights | prior art: Wintermeyer-Kallen et al., Forsch. Ingenieurwes. 85:385 (2021) schedule Q and R on wind and rotor speed | cite it; our variant is the tower weight *inside* full load, motivated per wind speed |
| gated / regime-switched residual | prior art in two forms: Abbas et al. (IET CTA 2026) activate a specialised residual in critical states via an HMM; Kim et al. (arXiv:2609.21307) gate a residual on an NMPC + disturbance-observer base with a deadband bound and an ISS certificate | claim the gating *variable* and the evidence for where it must close, never the mechanism |
| offset-free MPC in wind | prior art: Liu et al., IEEE TII 20(7):9487 (2024) | claim the *measurement* estimator-vs-residual, not the formulation |
| first LLM-written reward for control | gone; nearest analogue Wu et al., Building Simulation 2026 (LLM adapts reward *weights* over TD3) | the shape-vs-weights discriminator is the claim, and it is exactly what our weight-search plateau evidence defends |
| CPC-only beating a *tuned* ROSCO on regulation and both DELs | **survives**, no counterexample found | keep as the main result |

### What the audit adds
- **No verified residual RL in wind energy at all.** The framing is unoccupied. (The one residual-RL-on-a-classical-
  controller paper opened, arXiv:2310.14788, is the Tennessee Eastman chemical process.)
- **Anand & Bottasso, WES 11:1989 (2026)** is the closest competitor: same turbine, same simulator, adaptive economic
  NMPC whose model mismatch is repaired by an *offline* network, +9 % profit. Engage directly; our distinctions are
  online-vs-offline and the four-term objective under a per-wind load rule.
- **Corredera et al. (2026)**: ROSCO ported to a virtualised Siemens PLC at VAF > 90 % against the native
  implementation. This is the answer to "the ROSCO baseline is an academic straw man" and it is now in §2 of the paper.
- **Nilsen et al. (2026)**: a from-scratch RL wind controller starts ~12 % *below* its baseline and warm-starting from
  a model-based expert removes that phase — independent support for residual-on-a-strong-base.
- **Chen et al., Processes 14(18):2954 (2026)**: open-weight LLMs score 98-100 % on declarative control theory but
  31.8-53.2 % at ranking PI gain sets, every prespecified interval includes zero, and the paper concludes that
  simulator-based validation remains necessary. This is the citation for the fork-verification loop.
- **Reporting conventions**, all verified in print: short-term DEL as `(sum n_i S_i^m / n_ref)^(1/m)`; lifetime
  aggregation `DEL^m = sum_V sum_T DEL_bin^m P(T|V) P(V)` under a Rayleigh mean-wind distribution, IEC class 1A,
  V_ave = 0.2 V_ref, 20-year life (this is the form s42 implements, truncated to the evaluated bins); ADC =
  (1/T) int |dbeta|/dbeta_max dt with the rate limit printed because the normalisation is turbine-specific; tower
  m = 4 and blade m = 10 conventional, with 3/4/5 and 8/10/12 sweeps in print; six seeds per wind speed the de-facto
  floor, twelve in the closest-matching study. **No fetched paper does a paired bootstrap or any significance test.**

### Two checks against our own artefacts
1. **Turbulence class.** `data/wind/templates/turbsim_5mw.inp` uses `IECstandard = "1-ed3"`, `IECturbc = B`,
   `TurbModel = IECKAI`, power-law profile, `AnalysisTime` 650 s for 600 s episodes. The paper must therefore say
   **IEC 61400-1 edition 3**, and must not quote an I_ref value, which TurbSim sets from the standard.
2. **Is the 20 s discard long enough?** (`scripts/dev/transient_check.py`, no simulations — the baseline `.npz` keep
   the raw OpenFAST channels.) On eight baseline episodes the tower-base moment's RMS in the 0-20 s window is
   **+6.9 %** above the settled level and in the 20-40 s window **+0.9 %**; lengthening the discard from 20 s to 100 s
   moves the baseline's tower DEL by **-0.5 %**, its above-rated speed variance by **-2.5 %** and its power MSE by
   **-4.2 %**. The transient is over by 20 s; the residual effect is a shift of the level that both controllers of a
   comparison share, so it cancels in the ratios. Stated with these numbers in §3.1 rather than assumed.

### Consequences already applied
Manuscript v5 cites the prior art at each of the three mechanisms, carries the PLC and warm-start citations, states the
edition, the discard evidence, the DEL form and the seed count, and reports ADC with the 10 deg/s rate limit alongside
the travel ratio. `agents/rollout.py` now records the tower DEL at m = 3, 4, 5 and the blade DEL at m = 8, 10, 12 in the
same rainflow pass (`campaign_mexp.sh` re-evaluates the four headline controllers on the fresh seeds).

### Open items, to be closed before submission
- IEEE Xplore 403 blocked the full text of Liu et al. (2024); open it before any sentence about offset-free MPC.
- The IET full text of Abbas et al. (2026) should be read before the gating sentence is finalised.
- PubMed 41539907, an early-2026 DDQN pitch controller reportedly compared against ROSCO, is UNVERIFIED and is a
  potential competitor baseline; retrieve via the publisher DOI.
- Author list of the Wöhler-sensitivity paper (WES 9:799) is unverified in the bib entry.


## 44. The three blocked papers, read in full: the closest competitor, and what the two prior-art claims actually contain (2026-09-23)

The three papers the scans of s43 could not open (IEEE, IET and ScienceDirect all 403) were obtained and read. All
three open items are now closed, and one of them changes what we may say about our own numbers.

### 44.1 Espinoza, Ormaza, Tutiven & Vidal, *ISA Transactions* 169:428-435 (2026) - the closest published competitor
Double-deep-Q-network collective pitch in OpenFAST, warm-started by policy transfer from the PI, evaluated against
the shipped ROSCO. This is the paper a reviewer will hold our percentages against, so the comparison must be made
carefully - and it is **not** a comparison of the same quantity.

| | Espinoza et al. | this work |
|---|---|---|
| regulation metric | standard deviation of a **min-max normalised** power signal, normalised w.r.t. ROSCO (their Eq. 11) | MSE of power and generator speed on wind-labelled above-rated steps |
| reported gain | **15.7 %** on the test profile, **30.5 %** on a sawtooth profile (15.87 % in the rate-limiter table) | 51.5 % power MSE / 58.0 % speed MSE vs the shipped GSPI, 36 / 39 % vs the tuned one |
| loads | mean and scaled std of blade-root, tower-base and thrust channels; "largely unaffected", tower side-to-side slightly worse | tower-base and blade-root **damage-equivalent loads** (m = 4 / 10), -18.9 % / -1.8 % |
| evidence | one 400 s trace per wind profile (test + sawtooth), region 3 | 28 episodes x 600 s over seven mean wind speeds, paired bootstrap intervals, replicated on fresh seeds |
| baseline | shipped ROSCO | shipped GSPI **and** a tuned one (s29) |
| actuation | pitch **rate limiter** swept 0.1-1.0 deg/s as a design knob; 0.4 deg/s selected | rate limit at the controller's own 10 deg/s; duty reported as travel ratio and ADC |

Two things to take from it. First, **do not claim to beat their number**: a std reduction of a normalised signal and an
MSE reduction are different quantities, and the paper now says so explicitly. Second, their own ablation supports our
architecture: a **SAC agent trained from scratch under the same conditions regulates power worse than the ROSCO
baseline it was meant to improve** (scaled std 0.5288 against ROSCO's 0.1932 on the test wind), which is the same
finding as Nilsen et al. (2026) from a second direction.

### 44.2 Liu, Guo, Kong, Ma & Lee, *IEEE TII* 20(7):9487-9496 (2024) - offset-free MPC in wind
A Luenberger observer estimates the model-plant mismatch **introduced by linearisation**, inside a tube-based
stochastic MPC that enforces probabilistic rated-power constraints; validated in FAST. It is prior art for the
formulation and is cited as such. What it does not contain is the measurement we make: the comparison is between
three MPC variants (offset-free SMPC, mixed tube MPC, offset-free RMPC) over **180 s at 18 / 19 / 20 m/s**, scored by
average output power, rate of violating the rated-power constraint and average performance cost - **no fatigue metric
and no classical baseline**. Our estimator corrects an aerodynamic coefficient error rather than a linearisation
residual, and the point of s30 is that it beats a learned residual at doing so.

### 44.3 Abbas, Chasparis & Kelleher, *IET CTA* 20:e70099 (2026) - the gated residual
Residual policy on a PID expert, applied **only under critical conditions**; criticality is an **input-output hidden
Markov model trained offline on expert trajectories**, whose most probable hidden state is scored by the critic's
value function. Validated on the **Tennessee Eastman process**. Three differences are now in the manuscript, none of
them the mechanism: our gating variable is **exogenous** (the wind-speed estimate the base already computes) rather
than a hidden state inferred from the plant's response, so the gate is a fixed inspectable function with no learned
component; our base is tuned, where they write that their PID "serves as a safety net and baseline rather than the
ultimate optimized design"; and the evidence is per wind speed on held-out and fresh winds.

### Consequences applied
Manuscript v5 §2 now carries the Espinoza comparison in full (with the explicit statement that we do not present our
percentages as beating theirs), the precise content of the Liu and Abbas claims, and the Espinoza SAC ablation next to
Nilsen as support for residual-on-a-strong-base. Bibliography entries for all three are now from the PDFs rather than
from metadata records. Remaining open item from s43: the author list of the Wohler-sensitivity paper (WES 9:799).

## 45. A referee reading of manuscript v5, and what the artefacts say (2026-09-23; `scripts/dev/review_checks.py`)

A detailed critical reading of v5 raised eight points. Six are correct, one is correct as a scoring criticism but
wrong in its physical premise, and one is a scope statement. Everything below is measured on the existing evaluation
artefacts; no simulations were run.

1. **The two regulation terms of J are correlated (CORRECT as a scoring criticism, wrong as stated).** Above rated the
   torque is constant, so the referee expected power MSE and speed MSE to be the same quantity counted twice. The raw
   levels are NOT proportional: their per-episode ratio is 9.1 +- 15.1 (min 1.0, max 63), because each is normalised by
   its own rated value and the wind-labelled window contains near-rated steps at which power is still below rated; the
   correlation of the levels runs from -0.03 (tuned ROSCO) to 0.85 (MPC). But the two REDUCTIONS do move together
   (Pearson 0.94-0.95 on the MPC rows, 0.19-0.83 on the PI rows), so as a score the set does give regulation about half
   the weight. **Decisive check: dropping the power term and scoring on the remaining three changes no ranking**
   (tuned ROSCO 15.93 -> 13.07, MPC 30.46 -> 24.26, MPC + gated layer 32.54 -> 26.23); the only movement is two seeds
   of one arm 0.05 points apart. The metric set is kept because it is the published set this work is measured against,
   and both facts are now stated in the objective section.
2. **J does not price pitch activity (CORRECT).** Now stated explicitly where the objective is defined. The actuator
   argument was rewritten: "an order of magnitude below saturation" is the wrong yardstick, because pitch-bearing and
   drive fatigue accumulate with travel whether or not the actuator is near its rate limit. The duty-knee result (the
   same J at 38-41 % of the actuation) is flagged as the engineering-relevant one.
3. **Statistics (CORRECT on three of four counts).** "16 of 16" is eight runs evaluated on two wind sets, not sixteen
   independent observations, and the text now says so. Merging the two held-out sets (56 episodes, EIGHT seeds per wind
   speed) gives +4.16 / +4.27 / +4.77 on the tuned ROSCO and +1.58 / +2.17 / +2.05 on the MPC at interval widths of
   0.30-0.63, so the four-seed strata were not in fact producing suspiciously narrow intervals. What the intervals do
   NOT contain is the training-seed spread, which is of the same size: s.d. 0.33 with a 0.62-point range on the tuned
   ROSCO, 0.31 and 0.59 on the MPC. The fixed-reward arm has two seeds. All of this is now in the results section.
4. **Two internal inconsistencies and one mixed convention (CORRECT, all three were errors).**
   a. The "4 of 5 seeds hold every term" sentence belongs to s37: the UNGATED layer on the tuned PI base, held-out
      seeds, against that base. Table 2 is the GATED campaign on fresh seeds, whose agent seeds lose 1.0 / 2.7 / 1.4
      points of tower fatigue at 18 m/s and therefore miss the same 1 % rule. The paper now reports the gated runs as a
      difference of degree (two to six times smaller than the fixed reward's 5.9) and keeps the strong form only for
      the experiment it came from.
   b. Table 3 aggregates LEVELS (a ratio of means, which is what a power-mean DEL requires) while J and every figure
      average per-episode reductions. The two disagree substantially: MPC tower 22.1 % against 17.4 %, MPC power
      15.3 % against 49.1 %. The convention is now defined in the text and in the caption, both columns of the table
      are in the same convention, and the abstract says which convention the 22.1 -> 40.4 claim is in.
   c. The "+7-8 speed MSE" sentence mixed a difference in points (5.7) with a relative reduction of the remaining
      error (7-8 %). Both are now given, each with its name.
5. **The gate threshold was never swept (CORRECT).** Contribution (b) is narrowed to "restricting the layer helps",
   and the limitations state that where the boundary belongs is not established here. `campaign_gate13.sh` (gate
   13-15 m/s, two seeds per base) is that sweep and is paused mid-run.
6. **Double standard on the tower damper (CORRECT).** We insisted that the speed filter be tuned, then used ROSCO's
   damper at its shipped gain to conclude that it is worth -0.05. The row is now labelled shipped-gain only, with the
   explicit note that a damper gain is as turbine-specific as a filter corner.
7. **The "published verdict that MPC trades tower fatigue for regulation" was a straw man (CORRECT).** That -17.6 %
   tower row was OUR OWN F-era result and an artefact of the cost scaling; no citation was given because none exists.
   The sentence now attributes the impression to our earlier implementation and states that the MPC literature
   generally reports load reductions.
8. **Scope (CORRECT).** Onshore 5 MW, class B, above rated, collective pitch. Added to the limitations: the shipped
   filter corner is conservative partly because it must serve turbines we do not simulate, floating ones in
   particular, where raising the regulation bandwidth can drive negative damping of the platform pitch mode, so the
   15.4-point tuning margin does not transfer; and the flat blade-root result is what collective pitch can be expected
   to give, since the 1P blade load needs individual pitch.

Open after this pass: the gate-threshold sweep (paused campaign), a tuned tower damper if the -0.05 row is to carry
any weight, and more training seeds if the seed spread is to be separated from the wind spread.

### 45a. Zalkind, Dall'Anese & Pao (WES 5:1579-1600, 2020), added 2026-09-23

Automatic controller tuning by zeroth-order optimisation over aeroelastic simulations, on the NREL 5 MW and the SUMR
rotors in FAST. One of its three applications is the pitch controller's natural frequency and damping ratio, i.e. the
same layer of the controller our s29 search touches, and it reports **20-26 % cost-function reduction over the default
parameters** in that case (their cost, not our J - do not equate the numbers). Verified by fetching the article page.

Why it matters to us: it is the wind-energy literature's own answer to the question manuscript v5 is built around, and
it cuts both ways. It supports the tuned-baseline argument (defaults leave a lot on the table, and tuning them is
published practice rather than a contrivance of ours), and it bounds our claim, because our search covers two
parameters while theirs optimises the architecture: **15.4 points is a lower bound on what the classical controller
can be brought to, and against a fully tuned baseline every margin in the paper would be smaller.** Both directions are
now in the manuscript (baselines section, related work, limitations).

## 46. Pre-registration of the three referee-driven experiments (written 2026-09-23 22:30, BEFORE any of the three tables exists)

The three campaigns of this round were requested by two referee readings (s45) and are running. What each outcome
will do to the manuscript is fixed here, before the results, so that the write-up cannot be chosen after the fact.
The rule from the earlier probe applies again: a negative result is reported in the same place and with the same
prominence as a positive one would have been.

### A. Gate-threshold sweep (`campaign_gate13.sh`: gate 13-15 m/s, two seeds on each base)
Existing points: ungated (s39, s40) and gate 15-17 (s41). The sweep makes contribution (b) a measured trade-off
instead of one setting.

| outcome on the held-out and fresh sets, against each run's own base | what the paper says |
|---|---|
| 13-15 beats 15-17 on J in both seeds on both bases | the gate is a real design axis and our first setting was not its optimum; §4.2 becomes "where the gate closes is itself a measurable trade-off, and the better point is nearer rated", the headline rows move to 13-15, and contribution (b) claims a measured boundary |
| 13-15 and 15-17 within the seed spread (~0.3 J) | the threshold is a plateau over 13-17; contribution (b) stays "restricting the layer helps", and the limitation becomes "flat over the range tested" rather than "not established" |
| 13-15 worse, i.e. it reintroduces the near-rated damage the gate was built to remove | the s39 diagnostic is confirmed by construction rather than by selection; the paper keeps 15-17, and the sweep is reported as the evidence for it |
In every case the per-wind worst term against each run's own base is reported, because a J gain bought by damaging
12-14 m/s is exactly what this layer was gated to avoid.

### B. Wohler-exponent sensitivity (`campaign_mexp.sh`, `scripts/dev/wohler_sensitivity.py`)
Tower at m = 3, 4, 5 and blade at m = 8, 10, 12, controller side re-evaluated, baseline recomputed from the stored
raw channels.
- **If the ordering of the controllers is the same in all six columns**: one appendix table, one sentence in §4 that
  the fatigue conclusion does not depend on the exponent, and the matter is closed.
- **If the ordering changes anywhere**: the exponent joins the aggregation convention and the wind weighting as a
  third reporting choice that decides the answer, which strengthens the paper's own argument and must be given a
  paragraph in §4.4, not an appendix line.

### C. Random-structure control to n = 8 (`campaign_rand_seeds.sh`)
The open question is whether the LLM proposer beats a random draw from its own vocabulary on the per-wind load rule.
Current: agent 4/5, random structure 0/3, Fisher two-sided p = 0.14.
- **0 or 1 of the five new seeds holds the loads** (random 0-1 of 8): p = 0.007 or 0.02, the claim "the agent's
  proposals, not merely a structural search, are what hold the loads" is established on this base and is stated as
  the agent's measurable contribution.
- **2 or more of the eight hold the loads**: p > 0.1, and the claim is **withdrawn**. The paper then says that a
  verified structural search holds the loads and that the proposer is not distinguishable on either base, which
  removes the last place where the language model is credited with anything specific. The LLM would stay in the paper
  as one instance of a verified structural search, and the abstract's sentence about reward shape would be rewritten
  to credit the search rather than the model.
- Either way the level comparison is unchanged: a searched structure buys 3.5-4 points over the tuned PI and fixed
  rewards buy 1-2.

### Integrity
The machine ran out of free memory while these were running (7.5 GB total, 1.1 GB available, 17 OpenFAST processes);
no OOM kill appeared in dmesg, but `scripts/dev/integrity_check.py --since "2026-09-23 13:00"` runs before any of
these tables is read, because a killed worker leaves truncated artefacts that the pipelines would otherwise treat as
data.

## 47. The gate-threshold sweep: the setting we had is the best of the three, and the low gate fails exactly where the gate was invented (2026-09-23 13:27 - 23:14; `campaign_gate13.sh`, `scripts/dev/gate_sweep.py`, `docs/tables/gated_residual.csv`)

Outcome of experiment A of the pre-registration (s46): **branch three, "13-15 worse, it reintroduces the near-rated
damage the gate was built to remove".** The manuscript keeps the 15-17 gate and now reports the sweep as the evidence
for it, rather than reporting one setting and asserting the rest.

Arms: agent-written reward with the gate opening at 13 m/s and full at 15 (`--residual_gate 13 15`), two seeds on each
base, everything else identical to s41. Evaluated on the held-out seeds 3-6 and the fresh seeds 7-10 against each
run's own base.

| base | threshold | n | held-out mean diff | fresh mean diff | per seed (fresh) |
|---|---|---|---|---|---|
| tuned ROSCO | ungated (s37) | 5 | +4.02 | - | |
| | **13-15** | 2 | +3.81 | +3.86 | +3.80 +3.92 |
| | **15-17** | 3 | **+4.39** | **+4.41** | +4.21 +4.26 +4.75 |
| scheduled MPC | ungated (s40) | 5 | +0.81 | - | |
| | **13-15** | 2 | +1.61 | +1.37 | +2.20 +0.53 |
| | **15-17** | 3 | **+2.02** | **+1.85** | +1.52 +2.08 +1.96 |

Reading.
1. **15-17 is the best of the three settings on both bases and on both wind sets.** On the MPC base the ordering is
   monotone in the threshold (ungated +0.81, 13-15 +1.61, 15-17 +2.02 held-out); on the tuned PI base the low gate is
   worse than no gate at all, though that comparison crosses campaigns and should be read with care.
2. **The damage moves to where the gate was supposed to protect.** Over all rows of each arm, the worst per-wind term
   sits at or below 16 m/s in **6 of 8** low-gate rows against **3 of 16** high-gate rows (Fisher exact, two-sided
   p = 0.022), and the largest single per-wind shortfall in the whole study is now the low gate's **-10.3 points of
   power MSE at 16 m/s** (held out; -4.7 fresh), against -5.9 for the worst high-gate row. Opening the gate two
   metres per second earlier puts the layer back into the near-rated band where the base is already strong and where
   the s39 diagnostic first found it doing harm.
3. **Variance, not just level.** The two low-gate seeds on the MPC base are the best and the worst of all seven
   MPC-base runs (fresh +2.20 and +0.53). Widening the region the layer may act in widens the outcome distribution,
   which is its own argument for the narrower gate.
4. **What the paper may now claim.** Contribution (b) moves from "restricting the layer helps" to a measured
   trade-off over three settings, with the caveat that it is a coarse sweep: two seeds per low-gate cell, and the
   J differences (0.5 points) are the size of the seed spread measured in s45 (0.31-0.33 s.d.). The part that does
   not rest on the small J difference is the location and size of the worst per-wind term, which is what item 2
   reports.

Artefacts: `~/wtrl/exp/{mrwCwGLt_s0,mrwCwGLt_s1,trwCwGLt_s0,trwCwGLt_s1}/eval_range_TIB_s{3456,78910}.json`,
`docs/tables/gated_residual.csv` (now carries all three thresholds), `scripts/dev/gate_sweep.py`.
`integrity_check.py --since "2026-09-23 13:00"` reports no broken artefacts, which matters because the machine ran
out of free memory during this campaign.

## 48. The fatigue conclusion does not depend on the Wohler exponent (2026-09-23 23:15 - 09-24 00:10; `campaign_mexp.sh`, `scripts/dev/wohler_sensitivity.py`, `docs/tables/wohler.csv`)

Outcome of experiment B of the pre-registration (s46): **branch one, the ordering is the same in every column**, so
this is an appendix table and one sentence in the results, not a section.

The conventional exponents are 4 for the welded steel tower and 10 for the composite blade, and the field sweeps the
neighbours (WES 9:799 uses exactly 3/4/5 and 8/10/12). `agents/rollout.py` now records all six in the same rainflow
pass; the four headline controllers were re-evaluated on the fresh seeds and the paired GSPI side was recomputed from
the raw channels kept in the baseline `.npz`, so no baseline was simulated again. 70 baseline episodes, 28 per
controller.

| controller | tower m=3 | m=4 | m=5 | blade m=8 | m=10 | m=12 |
|---|---|---|---|---|---|---|
| tuned ROSCO | 6.1 | 7.1 | 7.6 | 1.5 | 1.4 | 1.3 |
| scheduled MPC | 14.5 | 17.4 | 19.2 | -0.2 | -0.0 | 0.0 |
| tuned ROSCO + gated layer (s2) | 6.1 | 7.4 | 8.0 | 2.3 | 2.3 | 2.2 |
| scheduled MPC + gated layer (s1) | **16.2** | **18.9** | **20.3** | 1.6 | 1.8 | 1.9 |

Reading.
1. **The ordering of the four controllers is identical in all six columns**, on both channels, so nothing in the
   paper's fatigue argument turns on the choice of exponent. The one tie is at tower m = 3, where the tuned ROSCO and
   the tuned ROSCO with the gated layer are both 6.1 %.
2. **The reported exponent is the conservative one, not the flattering one.** Every controller's tower reduction
   grows with m (6.1 -> 7.6, 14.5 -> 19.2, 16.2 -> 20.3), because a higher exponent weights the large cycles that the
   controllers damp most. Reporting m = 4 rather than m = 5 costs the MPC rows 1.5-1.8 points of apparent benefit.
   The blade side is flat in m, as expected for a channel the collective command barely touches.
3. Practical note for reuse: this cost one re-evaluation campaign only because the extra exponents were not recorded
   originally. They are now computed in the same rainflow pass at negligible cost, so any future evaluation carries
   them.

## 49. The proposer question, settled at n = 8: the search buys the level, the agent buys the constraint (2026-09-24 00:11 - 06:06; `campaign_rand_seeds.sh`, `scripts/dev/proposer_test.py`, `docs/tables/range_residual.csv`)

Outcome of experiment C of the pre-registration (s46): **the branch in which the claim is established.** Five more
random-structure seeds on the tuned PI base, ungated, identical base, winds, budget, objective and selection to the
existing three, so all eight pool without qualification.

| arm | n | loads held (per-wind rule, 600 s) | mean diff to the tuned ROSCO | s.d. |
|---|---|---|---|---|
| agent-written reward | 5 | **4/5** | **+4.02** | 1.42 |
| random reward structure, same fork verification | 8 | **1/8** | **+4.01** | 1.70 |
| fixed reward, J-tuned weights | 3 | 0/3 | +1.91 | 1.55 |
| fixed reward, tower weight 10 | 5 | 0/5 | +1.14 | 0.79 |

Fisher exact on the per-wind load rule: agent 4/5 against the random structures 1/8 **two-sided p = 0.032**, against
the fixed rewards 0/8 p = 0.007, against the pooled controls 1/16 p = 0.004. Every one of the thirteen searched-arm
600 s paired intervals is above zero.

Reading.
1. **The level is the search, not the proposer, and now exactly so.** +4.02 against +4.01. At n = 3 the random
   structures averaged +3.50 and it was possible to believe the agent was worth half a point of level; at n = 8 that
   reading is gone. Any statement in the paper that credits the language model with the \emph{level} would be wrong.
2. **The constraint is the proposer.** 4 of 5 against 1 of 8, p = 0.032. This is the one place in the whole study
   where the language model is distinguishable from a random draw out of its own vocabulary, and it is a constraint
   satisfaction difference, not a performance difference.
3. **The diagnostic detail worth keeping.** Going from three control seeds to eight raised the random structures'
   mean from +3.50 to +4.01 but their load-rule count only from 0/3 to 1/8. More search found the level; it did not
   find the constraint. That is the cleanest single sentence we have for what the agent is doing.
4. **Where it does not hold**: on the scheduled MPC base (s40) the two arms are indistinguishable on both level and
   the load rule, and the gated configuration was never run with a matched random-structure control, so the claim is
   scoped to the ungated tuned-PI base. Both scopes are stated in the manuscript's limitations.

Manuscript: the abstract, introduction result (iv), the ablation table and the ablation section now carry this; the
"not resolved by three control runs" wording of the previous revision is gone.

## 50. A WES-style referee reading of v6, and the four things it changed (2026-09-24; `scripts/dev/{gate_policy_test,windobs_table}.py`, `campaign_windobs.sh`)

A detailed review against the WES criteria returned Major Revision with support for publication. Four of its points
were substantive; three of those were right, and one of the three found a defect we had not noticed.

### 50.1 The wind signal the policy reads (the defect)
`envs/base_env.py` line 189 put **the simulator's true hub wind** in the PPO observation, while the MPC (line 180) and
the gate (line 266) both use the controller's own estimate. On this plant that is not a harmless substitution: over
the baseline episodes the estimate has a **-0.35 m/s bias, a 1.42 m/s error s.d. and correlation 0.73** with the
truth. The layer was trained with information a turbine does not have.

Added `EnvConfig.obs_wind_est` and `evaluate.py --obs_wind_est`, and re-evaluated the six frozen gated policies with
the estimate substituted, nothing else changed and nothing retrained (`campaign_windobs.sh`, 6 x 28 episodes):

| base | policy | J (true wind) | J (estimate) | difference [95 %] |
|---|---|---|---|---|
| tuned ROSCO | s0 / s1 / s2 | 20.14 / 20.19 / 20.68 | 20.11 / 20.24 / 20.51 | -0.03 / **+0.05** / -0.17 |
| scheduled MPC | s0 / s1 / s2 | 31.98 / 32.54 / 32.42 | 31.91 / 32.43 / 32.23 | -0.07 / -0.11 / -0.19 |

**The oracle is not load-bearing**: mean -0.09, worst -0.19, one seed improves, pitch travel unchanged to two
decimals. Against margins of +1.85 and +4.41 and a seed spread of 0.30, the true wind is worth 2-10 % of what the
layer earns. Disclosed in the setup, measured in a new section, and listed as a limitation (a policy *trained* on the
estimate was not run).

### 50.2 Best-of-three seeds as the headline (right)
Table 2 and the decomposition used the best of three training seeds. Changed to the seed mean with the seed spread:
**32.31 = 15.93 (49 %) + 14.53 (45 %) + 1.85 (6 %)**, learned rows now `20.34 +- 0.30` and `32.31 +- 0.30`. The text
now separates the two uncertainties explicitly: a paired interval is the wind realisation conditional on one policy,
the +-0.30 is the spread across policies. The message is unchanged because all three seeds were positive.

### 50.3 Pseudo-replication in the gate test (right, and the effect does not survive)
The p = 0.022 of s47 counted rows of `gated_residual.csv`, which are policy x wind-set pairs, not independent units.
Recounted with the trained policy as the unit (`gate_policy_test.py`): worst per-wind term at or below 16 m/s on
either set, **4/4 low-gate vs 2/8 high-gate, Fisher two-sided p = 0.061**; on both sets 2/4 vs 1/8, p = 0.236. The
paper no longer claims significance here. What it relies on instead is the J ordering (15-17 best on both bases and
both wind sets) and the magnitudes: low-gate worst shortfalls -10.3, -3.5, -1.8, -1.7 against high-gate -5.9 ... -1.2,
the -10.3 being a power-MSE loss at 16 m/s, inside the band the gate exists to protect.

### 50.4 Presentation and scope (right)
- The three aggregations are now **equations** (per-episode mean of ratios; power mean of levels within and across
  bins; Rayleigh weights), with the note that m enters twice, inside the episode's DEL and across bins.
- "site-weighted" is everywhere **conditional above-rated weighting**; the abstract says so.
- The baseline claim is narrowed to "a **minimal two-parameter** re-tuning already accounts for 49 %".
- "the Wohler exponent changes nothing" -> "changes the magnitudes but not the ranking".
- The proposer result is demoted to a secondary finding and restated conditionally: not that a language model finds
  better reward structures than random search, but that, **conditioned on a vocabulary distilled from its own
  output** and on this budget, its proposals satisfy the per-wind constraint more often than random recombination.
- The deployment sentence became "among the configurations examined, under the present simulation metrics, the most
  defensible performance-complexity trade-off", with what would be needed to call it a deployment recommendation.
- Bibliography placeholders cleared: IEC 61400-1 **ed. 3** (matching our TurbSim `1-ed3`), and the baseline paper
  resolved via Crossref to **IEEE TSTE 17(4):4378-4390, 2026, DOI 10.1109/TSTE.2026.3712960**.
- Fixed a pre-existing crash in `evaluate.py --help` (a literal `[%]` in a help string).

### 50.5 Requate, Wiens & Meyer, JPCS 1618:022045 (2020) - the closest precedent
Read in full (`RelatedWorks/`). They align controller evaluation with the V-model across three requirement domains
and demonstrate that evaluation parameters change the apparent quality, chiefly through the **seed count**: two
disjoint sets of six seeds give blade-root DEL reductions of 17 % and 10 % at one wind speed, 4 % and 10 % at another,
so "6 seeds ... are not necessarily a sound basis for a fatigue life evaluation". They also report their IPC costing
**+188 % pitch operating time**. Cited in three places: as the precedent for the question (with our difference stated:
they structure the evaluation, we price specific choices against each other on one plant, including two they do not
vary, the baseline's tuning state and the aggregation convention), in the statistics paragraph on seed counts, and in
the actuation section.

### Still open
A stronger classical baseline (4-6 parameters, or Zalkind-style automatic tuning) would test how far the 49 % depends
on tuning depth; a turbulence-class sensitivity (A or C) would test whether the aggregation finding holds; a matched
random-structure control for the gated configuration; and an archival release with a DOI for the code, the prompts
and the generated rewards.

## 51. The parameter search was asking the wrong question of the wrong box (2026-10-09 to 10-11; `docs/paper/design_2026-10-11_rosco-native-search.md`, `scripts/rosco_native.py`, `eval/selection.py`, `scripts/dev/search_box_bounds.py`, `campaign_rosco_native.sh`)

The proposer comparison launched on 10-09 (four proposers, three repeats, 16 candidates each, scored by the composite
on the design winds) reached 8 of its 12 runs before an audit stopped it. Three defects, in order of seriousness.

**Two of the four searched inputs were not ROSCO tuning quantities.** The search multiplied `PC_GS_KP` and `PC_GS_KI`
by scalars. ROSCO generates those tables from `omega_pc` and `zeta_pc` as `Kp = (2 zeta omega + A(beta))/B(beta)`,
`Ki = omega^2/B(beta)`. Scaling Ki by b is exactly `omega -> omega sqrt(b)`, but scaling Kp by a requires
`2 zeta' omega' = a 2 zeta omega + (a-1) A(beta)` and A varies along the schedule, so the implied damping ratio is not
constant: 6.8 % of spread for the s29 tuned baseline, 7.1-12.8 % for the three search winners. Those gain tables
cannot be produced by any tuning file. A and B are recoverable exactly from the shipped tables given the shipped
tuning (zeta 0.7, omega 0.6; round trip 8.7e-19), so the native parameterisation costs no new dependency and is
self-checking.

**Three box limits were ours, and two of them produced the result.** The filter corner was capped at 4.0 rad/s, just
above 3P (3.80), and both `llm` repeats put their winner exactly on the cap; the measured ceiling is 19.2 rad/s, where
a first-order filter passes 99.99 % of the generator-speed signal power. `FA_KI` was capped at 0.004 rad s/m, which
allows the tower damper 0.25 deg of pitch authority against a 5 deg limiter it can therefore never reach -- **so the
s23 finding that ROSCO's tower damper is worth about nothing at 600 s is withdrawn**, it was measured inside a box 50x
too narrow. `FA_HPFCornerFreq` was frozen at 0.172 rad/s, a value from our own tower-damper template; ROSCO's own
default for all three damper fields is 0.

**The composite chose the controller.** Of the nine selected designs, five lost a load term and two lost more than 11
points of tower-base DEL (`bayes r1` -11.4, `random r1` -20.3) while scoring highest, because a mean of four
reductions trades one for the others. Replaced by the standard constrained statement: maximise the generator-speed
regulation reduction subject to tower-base DEL, blade-root DEL and energy capture being no worse than the shipped
controller, with a constraint violated only where the loss exceeds its own 95 % paired stratified bootstrap interval
(measured median half-widths over the 14-episode design set: 1.30, 1.00 and 0.031 points), ranked by constraint
domination. On the 140 candidates already evaluated, 29 % are feasible under this rule and every run found at least
one -- but `local r0` and `random r1` found exactly one each and it was worse than shipped on regulation, so those
runs would report no feasible improvement in 16 simulations.

Two smaller fixes fell out. Parameters were being rounded to three significant figures before being written, which
wrote the shipped corner 1.5708 as 1.57: that is the whole of the -0.15 the identity check used to read, and at six
significant figures it now reads exactly +0.000. And `evaluate.py` only enables the wind-labelled above-rated subset
by itself when the borrowed config run carries the composite objective, so `--relabel_wind` is now passed explicitly;
without it the subset reverts to a per-controller pitch rule and the two sides stop being scored on the same samples.

What survives from the paused campaign is the motivation above and nothing else: the designs are not reachable, one
ceiling was load-bearing and the damper range was meaningless, so no proposer ranking is carried forward. The
replacement campaign re-derives the paper's tuned baseline in the native parameterisation first (the s29 recipe, 40
candidates), then runs the comparison at 24 candidates per run, then re-scores every selected tuning on held-out and
fresh winds with the per-wind load check the design set cannot resolve.

## 52. The tuned baseline, re-derived as a tuning the controller can actually take (2026-10-11)

Stage 1 of `campaign_rosco_native.sh`: the s29 recipe (40 candidates, random draws for the first half, then a local
step), the native parameterisation and the constrained selection of section 51, on the 14 design episodes
(12-24 m/s in 2 m/s steps, class B, seeds 1-2, 600 s simulated, 20 s discarded, 580 s scored). 19 of the 40
candidates held every constraint; the first one that held them and also improved regulation was candidate 30.

Selected tuning, every quantity one a tuning file carries, with the gain schedule regenerated by the toolbox
formula from plant coefficients recovered exactly from the shipped tables:

    omega_pc 0.604783 rad/s   zeta_pc 0.737849   F_LPFCornerFreq 6.52919 rad/s
    FA_KI 0.0331436 rad s/m   FA_HPFCornerFreq 0.692474 rad/s   TD_Mode 1   FA_IntSat 0.175

Against the shipped controller on the same wind fields, in percent (positive is better; brackets are 95 % paired
stratified bootstrap intervals over episodes):

| set | episodes | gen-speed MSE | power MSE | tower-base DEL | blade-root DEL | energy | travel | ADC |
|---|---|---|---|---|---|---|---|---|
| design, seeds 1-2 | 14 | +28.30 [25.35, 31.23] | +24.50 | +22.77 [22.08, 23.45] | +3.46 [2.15, 4.87] | +0.145 | 2.01x | 0.0948 (shipped 0.0471) |
| held out, seeds 3-6 | 28 | +28.32 | | +23.93 | | | | |
| fresh, seeds 7-10 | 28 | +29.75 | | +25.32 | | | | |

Per wind speed on the design set, 12 to 24 m/s: gen-speed +12.8 +34.5 +33.1 +32.0 +30.0 +28.4 +27.2, tower
+41.7 +38.6 +21.5 +14.1 +13.2 +13.7 +16.6, blade +16.5 +5.9 +0.6 +1.8 +0.4 -1.1 +0.3, power +7.2 +9.3 +26.8
+36.2 +33.5 +30.2 +28.4. The worst single-wind load term off the design winds is -0.24 (blade at 16 m/s) held
out and -0.34 fresh, so the design set did not hide a per-wind violation. Optimism of the design score: -0.02
points, i.e. none. Table `docs/tables/rosco_tuned_baseline.csv`, template
`~/wtrl/runs/template_5mw_rosco_tuned_native`.

Four consequences, three of them for the manuscript:

1. **The tower damper carries much of this.** FA_KI 0.0331 is 8.3 times the ceiling of the old box and asks about
   2.0 degrees of pitch (the limiter is at 10 and is never reached). The s23 statement that the fore-aft damper is
   worth about nothing was an artefact of a box that allowed it a quarter of a degree; it is withdrawn, and what
   replaces it is the opposite: the damper together with a speed filter opened by a factor of four is where a
   legitimate tuning finds 23 to 25 % of tower-base fatigue.
2. **The speed filter corner is the untuned part, as s29 said, but the tuned value is 6.53 rad/s**, above 3P
   (3.80), not the 3.42 of s29, which sat near a ceiling we had chosen at 4.0.
3. **Actuation is no longer what separates the model-based controller.** This baseline spends 2.01 times the
   shipped pitch travel; the scheduled model-based controller spends 2.14. Both buy their result with about twice
   the duty, so a duty argument cannot carry the comparison.
4. **Every table carrying the old tuned baseline must be re-measured against this one.** The old row was both
   selected and reported under the composite that section 51 removed, so its number is not comparable to anything
   here. On the published reductions this baseline alone takes 28 % of the regulation error and 23 % of the
   tower-base fatigue out of the shipped controller, so the margin the manuscript credits to advanced control
   shrinks and the single claim of the paper (that the baseline decides the apparent benefit) gets its strongest
   evidence so far.

### 52.1 The shipped controller is one of the candidates

The first stage-2 run exposed a defect in the selection, not in the search: with the winner taken over the
candidates only, a run that found nothing reported as its selection a tuning 21 points worse in regulation than
the controller already in the field. No engineer would ship that; the honest statement is that the arm bought
nothing. The shipped tuning is evaluated by every run anyway, free of budget, and against itself it reads exactly
zero on every term with intervals of zero width, so it is feasible by construction and an objective of zero is the
floor of any honest selection. It is now in the pool: it is the starting incumbent of the local arm, it is a
training point for the surrogates of the Bayesian arm with the best feasible objective they measure improvement
against, and a run that beats nothing selects it and reports a gain of exactly 0.00. The trace of best feasible
regulation against simulations spent reads 0.00 until a real gain appears, so each cell is what was bought over
the shipped controller.

The first run of the local arm was discarded and restarted because it had spent 17 of its 24 candidates refining
around an incumbent the fix removes (kept at `roscosearch_native/_superseded/local_r0_pre_incumbent_fix`). The
random arm is not affected, it never reads the incumbent, and its first run stands: 24 candidates, 6 of them
feasible, none better than shipped. Stage 1 was derived before the fix; the fix can only add the shipped point to
pools that lacked it, so the baseline above is a lower bound on what the recipe finds.

### 52.2 Stage 2 paused after 8 of 12 runs (2026-10-12 08:30, session 392 held with SIGSTOP)

Budget 24 candidates per run, same box, same start, same 14 design episodes, selection by constraint domination
with the shipped tuning in the pool. random r2 is mid-run at 14 candidates; local r2, bayes r2 and llm r2 have
not started. 23 of the roughly 200 candidates evaluated so far could not be run by the plant (divergence, which
is a result: the search reached the stability boundary), and no evaluation failed for any other reason.

| proposer | rep | feasible | sims to the first feasible gain | regulation | power MSE | tower DEL | blade DEL | energy | travel | ADC |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0 | 6/24 | none | 0.00 (shipped) | | | | | 1.00x | |
| random | 1 | 6/24 | none | 0.00 (shipped) | | | | | 1.00x | |
| random | 2 | 4/14 | none so far | 0.00 (shipped) | | | | | 1.00x | |
| local | 0 | 10/24 | 18 | +14.30 | +20.08 | +17.78 | +4.03 | +0.156 | 2.72x | 0.1281 |
| local | 1 | 9/24 | none | 0.00 (shipped) | | | | | 1.00x | |
| bayes | 0 | 10/24 | none | 0.00 (shipped) | | | | | 1.00x | |
| bayes | 1 | 12/24 | 7 | +46.76 | +37.77 | +4.24 | +3.84 | +0.106 | 1.53x | 0.0722 |
| llm | 0 | 10/24 | 8 | +47.79 | +38.97 | -0.61 | +3.06 | +0.111 | 1.76x | 0.0830 |
| llm | 1 | 14/24 | 8 | +41.27 | +33.55 | -0.94 | +3.33 | +0.127 | 1.74x | 0.0821 |

Runs that bought a shippable gain: random 0 of 3, local 1 of 2, bayes 1 of 2, llm 2 of 2 at a median of 8
simulations. Random search at this budget finds feasible tunings (4 to 6 of 24) but never one that also improves
regulation, so it selects the shipped tuning and buys nothing. That is the same shape as s37 on this base.

**What the selected tunings are is more interesting than who proposed them.** Sorted by regulation, with the
tower-base term beside it:

| tuning | omega_pc | zeta_pc | corner | damper | regulation | tower DEL | travel | worst per-wind tower |
|---|---|---|---|---|---|---|---|---|
| llm r0 | 0.700 | 0.750 | 5.500 | off | +47.79 | -0.61 [-1.30, +0.06] | 1.76x | -5.9 at 18 m/s |
| bayes r1 | 0.727 | 0.642 | 5.712 | 0.0014 | +46.76 | +4.24 [+3.70, +4.75] | 1.53x | -2.3 at 20 m/s |
| llm r1 | 0.610 | 0.920 | 5.100 | off | +41.27 | -0.94 [-1.88, +0.01] | 1.74x | -6.6 at 18 m/s |
| stage 1 (tune r0) | 0.605 | 0.738 | 6.529 | 0.0331 | +28.30 | +22.77 [+22.08, +23.45] | 2.01x | +13.2 at 20 m/s |
| local r0 | 0.613 | 0.607 | 8.747 | off | +14.30 | +17.78 [+17.00, +18.53] | 2.72x | +4.1 at 22 m/s |

The legitimate tuning space is not a point, it is a front, and the two ends of it differ by 33 points of
regulation and 23 points of tower-base fatigue. Every one of these is a tuning file the controller can take;
every one holds the constraints as written. The regulation objective alone walks up the front and turns the
damper off, because the damper is what buys tower fatigue and it costs regulation.

**Open decision, to be taken when stage 3 has run (it changes what the paper claims):** which of these is *the*
tuned baseline. The rule has to be stated and then applied, not chosen after seeing the comparison. The repo
already has the ingredient: s32 found that a set-mean load constraint is gamed across wind speeds and that the
per-wind constraint is the honest one, and the per-wind column above shows exactly that happening - llm r0 holds
the set mean at -0.61, inside the resolution of the measurement, while losing 5.9 points of tower fatigue at
18 m/s. Under a per-wind load constraint the two llm tunings are infeasible and the stage-1 tuning stands.
Recommended rule, stated before stage 3 is read: **the tuned baseline is the tuning with the largest regulation
reduction whose per-wind load terms hold off the design winds.** Two further notes:

- The per-wind re-selection needs no new simulations. Every history record already carries the per-wind terms of
  its candidate, so all roughly 200 evaluated candidates can be re-ranked under a per-wind constraint from the
  files on disk; only the verification of whatever that selects costs simulations.
- The front itself belongs in the manuscript. It is the cleanest statement of the single claim the paper makes:
  within the space of controllers the shipped tuning file can express, the baseline is a design choice worth
  33 points of regulation or 23 points of tower fatigue, and where the author puts it decides what the advanced
  controller appears to be worth.

### 52.3 The same 245 tunings re-ranked per wind speed, at no simulation cost (2026-10-12)

`eval/selection.py` now carries the per-wind rule (`per_wind_rank_key`, tolerance 1.0 point, the one s33 states)
and `scripts/dev/rosco_perwind_select.py` applies it to every tuning the campaign has ever evaluated, read off
the history files: 245 distinct tunings, 30 the plant could not run, 101 holding the constraints on the set
mean, **52 also holding both load terms at every single wind speed**. Table `docs/tables/rosco_perwind.csv`.

Three things came out of it, one of which corrects the recommendation of section 52.2.

**1. The per-wind screen cannot be the decision on the design set.** It rejects the stage-1 tuning, whose worst
per-wind term on the design set is -1.1 (blade at 22 m/s) against a tolerance of 1.0 - and that same tuning
measures -0.24 held out and -0.34 fresh, where there are four episodes per wind speed instead of two. The
design-set per-wind number is noise at that sample size, so the procedure has to be: **screen on the design set
at a generous tolerance, then measure per wind off the design winds for the finalists**, which is what stage 3
already does. Taking the screen as the decision would have thrown away a tuning that holds comfortably where it
can be measured, and selected +6.38 points of regulation instead of +28.30 from that run.

**2. There is a free one-knob fix in the shipped tuning file, and the duty axis of the front collapses.** The llm
arm of repeat 1 ran a deliberate one-factor sweep of the speed-filter corner at the shipped PI gains (its own
stated reasons: *isolating low-pass sensitivity at shipped PI settings*, then *probe whether further increasing
filter bandwidth beyond the 3.8 rad/s result improves regulation without exciting tower or blade loads*), which
produced a ladder nothing else in the campaign produced. All four rows hold every load term at every wind speed:

| F_LPFCornerFreq | gen-speed MSE | power MSE | tower DEL | blade DEL | energy | travel | worst per wind |
|---|---|---|---|---|---|---|---|
| 1.5708 (shipped) | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 | 1.000x | 0.0 |
| 3.000 | **+22.62** | +16.90 | +7.83 | +4.26 | +0.084 | **1.004x** | +0.7 at 22 |
| 3.800 (3P at rated) | +25.91 | +19.77 | +9.69 | +4.75 | +0.107 | 1.059x | +0.5 at 22 |
| 5.000 | +27.39 | +22.21 | +11.53 | +5.03 | +0.138 | 1.182x | +0.8 at 22 |
| 6.500 | +28.60 | +23.55 | +13.58 | +5.16 | +0.144 | 1.572x | +0.5 at 22 |

One line of the tuning file, nothing else touched, and the shipped controller gives up a fifth of its regulation
error, a sixth of its power-MSE and 8 % of its tower-base fatigue **at the same pitch travel** (1.004x, ADC
0.0473 against 0.0471) with no term negative anywhere. Nothing is traded; it is not a design choice, it is a
default left unexamined. Every comparison in the literature that uses the shipped ROSCO as its baseline is
starting from a point this far below what the same controller does with one number changed.

**3. The front under the per-wind screen, with duty as the third axis** (all per-wind feasible on the design set,
to be confirmed off the design winds):

| tuning | regulation | tower DEL | blade DEL | travel | ADC | where from |
|---|---|---|---|---|---|---|
| omega 0.600 zeta 0.850 corner 5.000, damper off | **+36.96** | +4.37 | +4.46 | 1.475x | 0.0695 | llm r1 #12 |
| omega 0.664 zeta 0.687 corner 3.408, damper 0.0012 | +34.59 | +3.66 | +3.53 | **1.152x** | 0.0543 | bayes r1 #7 |
| omega 0.600 zeta 0.700 corner 6.500, damper off | +28.60 | +13.58 | +5.16 | 1.572x | 0.0741 | llm r1 #11 |
| omega 0.605 zeta 0.738 corner 6.529, damper 0.0331 (stage 1) | +28.30 | **+22.77** | +3.46 | 2.013x | 0.0948 | tune r0 #30 |
| omega 0.600 zeta 0.700 corner 3.000, damper off | +22.62 | +7.83 | +4.26 | 1.004x | 0.0473 | llm r1 #9 |
| omega 0.613 zeta 0.607 corner 8.747, damper off | +14.30 | +17.78 | +4.03 | 2.719x | 0.1281 | local r0 #21 |

The set-mean rule would select +47.79 (llm r0 #18, -5.9 of tower-base load at 18 m/s); the per-wind screen costs
10.8 points of regulation against that. Reading down the table, the damper is what buys tower-base fatigue and it
costs both regulation and duty, while the filter corner buys regulation nearly free up to about 3.4 rad/s and
then starts costing duty.

**What this does to the manuscript.** The baseline is not a row, it is a band, and the band is wide: between
+22.6 and +36.9 of regulation and between +4.4 and +22.8 of tower-base fatigue, every member a tuning file the
shipped controller can take. The ladder figure needs the band drawn on it, and the claim that reporting choices
decide the apparent benefit of advanced control now has its sharpest form: the advanced controller is compared
against a baseline the author chooses from a 14-point interval, and the cheapest member of that band costs one
line and no actuator duty.

**Next, after the campaign finishes (no new decisions needed):** verify the three finalists off the design winds
(seeds 3-6 and 7-10, four episodes per wind speed, about 20 minutes each) - the +36.96, the +34.59 at 1.15x duty
and the one-knob +22.62 - and take as the tuned baseline the largest regulation reduction whose per-wind load
terms hold where they can be measured. Stage 1 is already verified on both sets and stands as the load-optimal
member of the band.

## 53. The ROSCO-native campaign, finished (2026-10-12 16:33; 12 runs, 321 tunings, 3 stages)

`campaign_rosco_native.sh` ran to the end: stage 1 (the re-derived baseline, section 52), stage 2 (four
proposers, three repeats, 24 candidates each) and stage 3 (every selected tuning re-scored on held-out seeds 3-6
and fresh seeds 7-10 with the per-wind load check). Tables `docs/tables/rosco_search.csv`,
`rosco_tuned_baseline.csv`, `rosco_perwind.csv`. Health: every one of the 12 identity checks read exactly 0.000,
37 of the 321 distinct tunings could not be run by the plant (divergence, a result), and **no evaluation failed
for any other reason**, so no count here is contaminated by a broken machine.

### 53.1 Who should propose the parameters of a classical controller

Same box, same start, same budget, same episodes, same selection rule; a run that found nothing selects the
shipped tuning and buys 0.00.

| proposer | runs that bought a shippable gain | regulation of the selection, mean (sd) | median simulations to the first gain |
|---|---|---|---|
| random | 0 of 3 | - | - |
| local, (1+lambda) from the shipped tuning | 1 of 3 | +14.30 | 18 |
| constrained Bayesian | 2 of 3 | +31.06 (22.20) | 14 |
| reasoning model | **3 of 3** | **+42.18 (5.22)** | **8** |

Random search at 24 candidates finds feasible tunings (6 to 7 of 24) but never one that also improves
regulation. The ordering is the s37 shape on this base, now with a mechanism that is on the record: the model
arm sweeps one quantity at a time and uses the plant frequencies as landmarks (its own words, *isolating
low-pass sensitivity at shipped PI settings*, then *probe whether further increasing filter bandwidth beyond the
3.8 rad/s result improves regulation*), which is what produced both the one-knob ladder of section 52.3 and the
low-duty family in 53.3. **Do not write that the model designs a better controller**: it was the fastest and the
most reliable at the objective as stated, and 53.2 is about the objective as stated being insufficient.

**The design set ranks candidates the way unseen winds do, here.** Over the 12 selected tunings, design against
held-out correlation **+1.00** and rank correlation **+1.00**, optimism of the design score -0.13 points on
average (worst +0.70), and 12 of 12 stay feasible on held-out and on fresh winds. The s38 warning that a small
design set cannot rank designs (rank correlation -0.9 there) does **not** transfer to this space: five smooth
tuning quantities on a deterministic controller, with paired episodes, rank cleanly. That is a statement about
the search space, not about the evaluation protocol.

### 53.2 The decisive result: the set-mean load constraint is not enough, on a classical controller too

Off the design winds, with four episodes per wind speed, the worst load reduction at any single wind speed:

| selected tuning | set-mean tower DEL | worst per wind, held out | worst per wind, fresh | shippable |
|---|---|---|---|---|
| llm r0, +47.79 regulation | -0.61 | **-4.64** (tower, 12 m/s) | -2.87 (tower, 24) | no |
| llm r1, +41.27 | -0.94 | **-7.61** (tower, 12) | -2.07 (tower, 24) | no |
| llm r2, +37.47 | +0.73 | -2.72 (tower, 16) | -3.73 (tower, 24) | no |
| bayes r1, +46.76 | +4.24 | -0.71 (tower, 22) | -1.24 (tower, 24) | marginal |
| bayes r2, +15.37 | +5.08 | +2.37 | +0.25 | yes |
| local r0, +14.30 | +17.78 | +1.12 | +0.33 | yes |
| stage 1, +28.30 | +22.77 | -0.24 | -0.34 | yes |

Every one of these holds the constraints as the search stated them, on the set mean, inside the resolution of
that measurement. The three tunings with the most regulation lose 2 to 8 points of tower-base fatigue at one
wind speed. The size of the set-mean margin predicts whether a tuning holds per wind, which is the useful rule
of thumb: a tuning that clears the load constraint by a point or two is holding it on average only. This is s32
again - a set-mean constraint is gamed across wind speeds - and it now has a second, independent instance on a
deterministic classical controller, where no learning is involved at all. **The per-wind constraint is not an RL
precaution; it is what the measurement requires.**

### 53.3 What a tuning file is worth, and the band the baseline comes from

321 distinct tunings evaluated, 132 holding the constraints on the set mean, 70 also holding both loads at every
wind speed on the design set. Four members of that band say what there is to say (design set; the first three
await the per-wind measurement off the design winds):

| tuning | regulation | power MSE | tower DEL | blade DEL | travel | what it is |
|---|---|---|---|---|---|---|
| corner 3.000 alone, shipped gains | +22.62 | +16.90 | +7.83 | +4.26 | **1.004x** | one line of the tuning file, no duty, nothing traded |
| omega 0.650 zeta 0.700 corner 3.100 | +33.34 | +26.30 | +2.63 | +3.66 | 1.118x | the most regulation available at near-shipped duty |
| omega 0.600 zeta 0.850 corner 5.000 | +36.96 | +29.60 | +4.37 | +4.46 | 1.475x | the most regulation that passes the design-set screen |
| stage 1, damper 0.0331 | +28.30 | +24.50 | **+22.77** | +3.46 | 2.013x | the load-optimal end, verified held out and fresh |

Reading the band: the speed-filter corner buys regulation nearly free to about 3.1 rad/s, the loop bandwidth
buys more regulation for duty, and the fore-aft damper is the only quantity that buys tower-base fatigue, at the
cost of regulation and duty. The band is 14 points of regulation wide and 20 points of tower-base fatigue deep,
and every member is a file the shipped controller can take.

### 53.4 What this does to the manuscript

1. The tuned-baseline row of v6 is void twice over: not a tuning file, and selected under the composite. Replace
   it with a band and name the member used for each comparison.
2. The s23 tower-damper statement is replaced by its opposite (section 52).
3. **The duty argument against the model-based controller is gone.** A PI tuning reaches +33 points of
   regulation at 1.12 times the shipped pitch travel; the scheduled model-based controller sits at 2.14 times.
   Whatever the comparison now shows, it cannot be carried by actuator duty.
4. The single claim of the paper is unchanged and much better supported: the baseline is a design choice worth
   more than the advanced controller is usually credited with, and the cheapest point in it costs one line.

### 53.5 Process conclusions, for the next search of any kind

- The incumbent must be in the candidate pool. Without it one run reported as its winner a controller 21 points
  worse in regulation than the one already in the field (section 52.1).
- A candidate the plant cannot run and a machine that is broken must be separated before any count is believed:
  37 divergences here, zero infrastructure failures, and the table says so explicitly.
- A per-wind screen on two episodes per wind speed is noise: it rejected a tuning at -1.1 that measures -0.24
  and -0.34 where there are four episodes per wind. Screen on the design set, measure off the design winds.
- Open, about one hour of simulation: the per-wind measurement of the three unverified band members above, after
  which the baseline row is fixed by the stated rule (largest regulation whose per-wind load terms hold off the
  design winds). Stage 1 holds that rule already and stands until something beats it.

### 53.6 The band off the design winds, paused after 4 of 10 members (2026-10-12 18:30)

`campaign_verify_band.sh` (session 395, `scripts/dev/rosco_verify_band.py`) measures the ten band members that
either buy more regulation than the standing baseline or cost no actuator duty, 28 held-out and 28 fresh
episodes each, and applies the stated rule. Paused at the user's request with four members measured on both
sets; eleven evaluations remain, about 90 minutes at 14 workers. Resume with
`campaign_ctl.sh resume campaign_verify_band`; nothing repeats, every evaluation is cached by parameter hash.

What the first four say, and they say it with one voice:

| tuning | design | held out | fresh | worst per wind, held out | fresh | verdict |
|---|---|---|---|---|---|---|
| omega 0.600 zeta 0.850 corner 5.000, 1.475x | +36.96 | +36.98, tower +5.09 | +37.74, tower +7.59 | +1.06 | **-1.17 at 24, tower** | fails |
| omega 0.664 zeta 0.687 corner 3.408 damper 0.0012, 1.152x | +34.59 | +35.07, tower +3.37 | +37.43, tower +5.20 | -0.47 | **-2.04 at 24, tower** | fails |
| omega 0.660 zeta 0.680 corner 3.100, 1.117x | +33.60 | +32.66, tower +2.07 | +34.83, tower +3.72 | **-1.94 at 24** | **-3.40 at 24** | fails |
| omega 0.650 zeta 0.700 corner 3.100, 1.118x | +33.34 | +32.52, tower +2.08 | +34.46, tower +4.01 | **-1.55 at 24** | **-2.98 at 24** | fails |

Four for four, always the tower-base load, always at 24 m/s, and always worse where there are more unseen
episodes. The regulation reductions themselves replicate almost exactly (+36.96 design -> +36.98 held out ->
+37.74 fresh), so this is not optimism in the objective: it is the load constraint being satisfied on the set
mean while it is violated at the top of the wind range, which is section 53.2 measured a second time on members
no run selected. The set-mean tower margin of these four is +1.9 to +5.1 points, and that margin is what the
per-wind failure tracks; the standing baseline carries +22.8.

The pattern has a mechanism, not just a correlation: every one of these four raises `omega_pc` or the filter
corner and switches the damper off, which is where the regulation comes from, and the tower-base load at 24 m/s
is exactly where the damper earns its keep. Nothing in the list above +28.3 has the damper on except the
bayes member, whose damper gain is 0.0012 against the 0.0331 of the standing baseline.

So the rule is already pointing somewhere: of the four measured, none holds, and the standing baseline
(+28.30 design, +28.32 held out, +29.75 fresh, tower +22.77, worst per wind -0.24) still stands. Six members
remain, all of them below +32 of regulation, three of them at or under the shipped pitch travel.

## 54 The ladder in the four published reductions, and what the composite was hiding (2026-10-12, no simulations)

`scripts/dev/ladder_published.py`, `docs/tables/ladder_published_{fresh,heldout}.csv`. Every rung of the
manuscript's ladder already exists as an evaluation on disk, and every evaluation carries its per-episode
reductions against the paired shipped controller and the wind file each episode used. Joining the rungs on the
wind file gives the ladder on one common set of 28 episodes with no simulation at all, and once the composite is
not used to collapse the four terms, the rungs can only be read term by term. That changes the paper's second
rung.

Fresh seeds 7-10, 28 episodes, each term a reduction against the paired shipped controller, interval a
stratified paired bootstrap over episodes:

| rung | power MSE | speed MSE | tower DEL | blade DEL | energy | travel |
|---|---|---|---|---|---|---|
| tuned baseline, superseded (hand-scaled gains) | +24.5 | +30.7 | **+7.1** | +1.4 | +0.122 | 1.120x |
| tuned baseline, native (s52) | +25.3 | +29.7 | **+25.3** | +1.7 | +0.167 | 2.016x |
| scheduled offset-free LPV-MPC | +49.1 | +55.4 | **+17.4** | -0.0 | +0.257 | 2.076x |
| MPC + wind-gated learned layer | +51.5 | +58.0 | +18.9 | +1.8 | +0.088 | 2.899x |
| superseded baseline + gated layer | +32.5 | +39.3 | +6.8 | +2.2 | +0.133 | 1.417x |

**The model-based controller's tower-fatigue advantage was an artefact of the baseline not being a tuning
file.** Against the superseded row it gains +10.3 points of tower DEL; against a baseline the controller can
actually take it *loses* 7.91 [8.85, 6.97] on the fresh seeds and 7.43 [8.37, 6.44] on the held-out seeds, both
intervals clear of zero on paired episodes. The learned layer recovers +1.44 [0.70, 2.21] of it (fresh) and
+2.03 [1.29, 2.77] (held out) - real, resolved, and not enough: the whole stack ends 6.4 points of tower
fatigue below the tuned PI loop.

Three things make this a fair comparison rather than a baseline chosen to flatter:

1. **The two baselines are the same controller in regulation.** Native minus superseded: speed MSE
   -0.91 [-1.94, +0.10], not resolved; power MSE +0.85 [+0.31, +1.39]. The native tuning buys
   +18.18 [+16.78, +19.86] points of tower DEL without giving up regulation.
2. **It is not bought for free, it is bought with pitch travel**: 1.120x shipped for the superseded row,
   2.016x for the native one, because the damper is on. The band has members that do not spend it (the one-knob
   corner at 1.004x, +22.6 regulation, +7.8 tower), so which member is named decides the duty column.
3. **At the same duty the comparison is a clean trade.** The native baseline spends 2.016x and the scheduled
   MPC 2.076x - within 3 % of each other - and over that trade the MPC buys 23.7 power / 25.6 speed and gives
   up 7.9 tower. Nothing in the comparison is carried by actuator duty any more (section 53.4).

The composite could not show any of this: it averaged the four terms, so +25.6 of regulation and -7.9 of tower
fatigue appeared as one positive number, and the rung looked like a strict improvement. Reporting the four
published reductions makes the second rung a trade and the third a small gain on every term.

Consequence for the manuscript beyond the baseline row: the learned-layer rung on the PI base (s34, s37, s41)
was trained and measured on the *superseded* baseline, so its +4.2 is a gain over a controller that is no longer
the baseline. The MPC-base rung is unaffected. Re-deriving the PI-base rung needs retraining, not re-reading, so
it is a decision rather than a table fix: either report it as what it is (a gain over the hand-scaled row, with
the row named) or re-run the three agent seeds on the native baseline.

## 55 The band verified off the design winds: the rule picks stage 1, and the band is a duty ladder (2026-10-12 22:10)

`campaign_verify_band.sh` finished: 10 members, 560 episodes, 28 held out and 28 fresh each, no infrastructure
failures (`docs/tables/rosco_band_verified.csv`). Four hold both load terms at every wind speed on both sets,
six do not, and the six fail the same way the three stage-3 winners failed in section 53.2.

### 55.1 Six of ten fail, and they fail identically

| member | design | held out | fresh | worst per wind, held out / fresh |
|---|---|---|---|---|
| omega 0.600 zeta 0.850 corner 5.000, 1.48x | +36.96 | +36.98 | +37.74 | +1.06 / **-1.17 at 24** |
| omega 0.664 zeta 0.687 corner 3.408 damper 0.0012, 1.15x | +34.59 | +35.07 | +37.43 | -0.47 / **-2.04 at 24** |
| omega 0.660 zeta 0.680 corner 3.100, 1.12x | +33.60 | +32.66 | +34.83 | **-1.94 / -3.40 at 24** |
| omega 0.650 zeta 0.700 corner 3.100, 1.12x | +33.34 | +32.52 | +34.46 | **-1.55 / -2.98 at 24** |
| omega 0.650 zeta 0.700 corner 2.900, 1.11x | +32.14 | +31.17 | +33.28 | **-1.76 / -3.04 at 24** |
| omega 0.650 zeta 0.700 corner 2.700, 1.09x | +30.69 | +29.81 | +31.80 | **-2.25 at 16 / -3.00 at 24** |

Six for six on the tower-base load, five of six at 24 m/s, and every one of them worse on the fresh seeds than
on the held-out seeds while its regulation reduction replicates to a point or two. **Every member of the pool
that buys more regulation than the standing baseline fails the per-wind load test.** The set-mean tower margin
of these six is +1.0 to +4.4 points; the four that hold carry +7.8 to +15.9. That margin is the predictor,
measured a third time (s32, s53.2, here).

### 55.2 The four that hold, in the four published reductions

`scripts/dev/rosco_band_table.py`, `docs/tables/rosco_band_rows_{fresh,heldout}.csv`. Fresh seeds 7-10, 28
episodes, against the paired shipped controller, with the actuator duty cycle against the 10 deg/s rate limit:

| member | power MSE | speed MSE | tower DEL | blade DEL | energy | travel | ADC |
|---|---|---|---|---|---|---|---|
| **stage 1: corner 6.53, damper 0.0331** | **+25.3** | **+29.7** | **+25.3** | +1.7 | +0.167 | 2.016x | 0.0968 |
| corner 6.500, damper off | +24.2 | +28.3 | +16.3 | +2.5 | +0.175 | 1.647x | 0.0791 |
| corner 3.000, damper off (one knob) | +17.7 | +23.0 | +8.7 | +2.0 | +0.103 | **1.000x** | 0.0480 |
| omega 0.562 zeta 0.663 corner 4.146 damper 0.0018 | +8.0 | +11.4 | +17.0 | +2.7 | +0.156 | **0.974x** | 0.0468 |
| omega 0.582 zeta 0.590 corner 2.595, damper off | **-1.9** | +0.7 | +12.1 | +1.8 | +0.074 | 0.884x | 0.0424 |

**The stated rule selects stage 1 and the choice is not close.** Paired against stage 1 on the same episodes,
every other member is worse in regulation and in tower fatigue with every interval clear of zero: the closest,
corner 6.500 with the damper off, is -1.40 [-1.74, -1.08] in speed MSE and -9.07 [-9.55, -8.61] in tower DEL on
the fresh seeds (-0.76 and -9.63 held out). Going down the band buys exactly one thing: actuator duty.

So the manuscript's baseline row is the stage-1 native tuning, and it is now verified three ways: feasible on
the design set, per-wind on 28 held-out episodes, per-wind on 28 fresh episodes.

### 55.3 What the band is for

The four verified rows are a duty ladder of the same controller, every row a tuning file:

- **1.000x travel, ADC 0.0480 against 0.0476 shipped**: one line changed (`F_LPFCornerFreq` 1.57 -> 3.0) buys
  +17.7 power MSE, +23.0 speed MSE, +8.7 tower DEL, +2.0 blade DEL, every term positive at every wind speed.
  **At the shipped actuator duty.**
- **0.974x travel, LESS actuation than shipped**: a slower loop with the damper on buys +8.0 / +11.4 / +17.0 /
  +2.7. All four terms positive while using less pitch travel than the controller in the field.
- **1.647x**: the filter alone at 6.5 rad/s, +24.2 / +28.3 / +16.3 / +2.5.
- **2.016x**: stage 1, the filter plus the damper, +25.3 / +29.7 / +25.3 / +1.7. The damper is what buys the
  last 9 points of tower fatigue and it costs 0.37x of travel.

The last row of the table is a warning and stays out of the band: `power MSE -1.9` [-2.6, -1.2] with speed MSE
+0.7 [-0.4, +1.9]. The selection constrains tower DEL, blade DEL and energy, and takes speed MSE as the
objective on the argument that power and speed carry much the same information above rated; on this member they
disagree in sign. The argument holds on the model-based rows (correlation 0.94-0.95) and is weaker on PI rows
(0.19-0.83), and this is an instance of the weak case. It changes no selection made here - the member is not
selected by anything - but it belongs in the manuscript's statement of that convention.

### 55.4 Lesson, process

Do not edit a shell script while bash is executing it: patching `campaign_verify_band.sh` mid-run shifted the
byte offsets and bash mis-parsed the line it had not read yet (`^\[eval|^ *$: command not found`). The run
itself completed - the edit only hit the final echo - but it could as easily have hit a line that mattered.

## 56. The learned layer re-trained on the baseline that is a tuning file (2026-10-13)

`campaign_gate_native.sh`, arm `tnrwCwG3t`, three agent seeds, finished 04:28 after 4 h 32 min. Why it had to be
re-trained rather than re-read: the manuscript's PI-base learned rung (s34, s37, s41) was trained against the
*superseded* hand-scaled baseline's own rollouts, so no table arithmetic can move it onto the baseline s55 fixed.
Everything else is identical to the old runs - same arm arguments (agent-written reward, gate 15-17 m/s,
objective Cw per wind, 300 episodes of 150 s), same wind protocol, same knob file; the base is chosen by
`WTRL_TEMPLATE` and the paired reference by `WTRL_HOME`, and the identity check of the new paired tree
(`identity_rtn_range`) read exactly 0.000 on all four terms and 0/0/0/0 at every wind speed, which is what says
the rollouts belong to this base.

### 56.1 Training side (150 s, held-out seeds 3-6, 28 episodes, vs the native base)

All three seeds at the strict tier, every one of the four terms positive in every seed:

| seed | power MSE | speed MSE | tower DEL | blade DEL | energy |
|------|-----------|-----------|-----------|-----------|--------|
| s0 | +7.49 | +7.64 | +0.81 | +0.96 | -0.00 |
| s1 | +8.01 | +8.33 | +0.23 | +0.99 | +0.00 |
| s2 | +8.31 | +8.70 | +0.84 | +0.19 | -0.00 |

The old runs on the superseded base were +1.1 / +6.2 / +7.1 in their per-wind objective, a spread of six points;
these three are within half a point of each other on every term. 3/3, no rollback loop, no degraded tier.

### 56.2 600 s, the four published reductions (`scripts/dev/gated_native_table.py`, `docs/tables/gated_native.csv`)

Fresh TurbSim seeds 7-10, 28 episodes, each reduction against the paired **shipped** controller on the same wind
file; the right-hand block is the same quantity paired against the run's **own base** episode by episode:

| row | power MSE | speed MSE | tower DEL | blade DEL | travel | ADC |
|-----|-----------|-----------|-----------|-----------|--------|-----|
| native tuned baseline (s52) | +25.3 | +29.7 | +25.3 | +1.7 | 2.016x | 0.0968 |
| + gated layer, s0 | +31.1 | +35.7 | +25.8 | +2.9 | 2.606x | 0.1251 |
| + gated layer, s1 | +31.1 | +35.9 | +25.3 | +3.2 | 2.460x | 0.1181 |
| + gated layer, s2 | +31.8 | +36.5 | +26.0 | +2.2 | 2.501x | 0.1201 |

Paired against the base, fresh / held out, every interval a stratified paired bootstrap over the 28 episodes:

- power MSE **+5.80 to +6.44** fresh (+5.40 to +6.24 held out), all six intervals clear of zero;
- speed MSE **+5.91 to +6.78** fresh (+5.54 to +6.58 held out), all six clear of zero;
- tower DEL +0.01 to +0.64 fresh, +0.06 to +0.48 held out - **positive or unresolved, never negative** (s1 is
  the unresolved one on both sets);
- blade DEL +0.47 to +1.48 fresh, +0.42 to +1.15 held out;
- per-wind violation **0.00 %** for Cw and CTw in all six evaluations: both loads hold at every wind speed on
  28 held-out and 28 fresh episodes.

Duty: the layer costs 1.22-1.29x the base's pitch travel, the same relative cost it had on the superseded base
(1.25-1.29x), which takes the whole PI route to 2.46-2.61x the shipped travel, ADC 0.118-0.131 against the
10 deg/s rate limit.

### 56.3 What changes in the ladder

`scripts/dev/ladder_published.py` now carries the rung (`docs/tables/ladder_published_{fresh,heldout}.csv`).
Fresh seeds, same 28 episodes for every row:

| rung | power MSE | speed MSE | tower DEL | blade DEL | travel |
|------|-----------|-----------|-----------|-----------|--------|
| tuned baseline, native (s52) | +25.3 | +29.7 | +25.3 | +1.7 | 2.016x |
| native baseline + gated layer | +31.1 | +35.9 | +25.3 | +3.2 | 2.460x |
| scheduled offset-free LPV-MPC | +49.1 | +55.4 | +17.4 | -0.0 | 2.076x |
| MPC + wind-gated learned layer | +51.5 | +58.0 | +18.9 | +1.8 | 2.899x |

Three readings, each per term rather than on a composite:

1. **The layer is smaller on a real tuning than it was on the hand-scaled row** - about three quarters of it:
   +5.8 / +6.2 against +8.0 / +8.6 in power and speed MSE for the same seed. The baseline took part of the room.
   It is also cleaner: on the superseded base two of three seeds lost tower DEL (-0.31 [-0.62, -0.05] and
   -0.46 [-0.77, -0.15]); on the native base no seed does on either set.
2. **The decomposition of the headline has to be written per term, and in one term the model-based rung is
   negative.** Of the 58.0 points of speed MSE the full stack buys over the shipped controller: tuning +29.7
   (51 %), model-based +25.7 (44 %), learning +2.6 (5 %). Of the 51.5 points of power MSE: +25.3 / +23.8 / +2.4
   (49 / 46 / 5 %). But of the 18.9 points of tower DEL: tuning **+25.3**, model-based **-7.9**, learning +1.4 -
   the model-based rung spends tower fatigue to buy regulation, so no single set of shares describes the ladder.
   The old "15.93 / +14.53 / +2.08, i.e. 49 / 45 / 6 %" was that average.
3. **Duty does not explain the gap between the two routes.** The PI route with its layer sits at 2.46x the
   shipped travel and the MPC alone at 2.08x; the MPC still has +19.5 [18.3, 20.7] more speed MSE at *less*
   actuation, and still gives up -7.9 [-8.9, -7.0] of tower DEL. The duty argument (s53) is gone in both
   directions: a PI tuning reaches +33 regulation at 1.12x travel, and the learned PI route reaches 2.46x
   without catching the MPC.

### 56.4 Defect found and fixed in my own table

The first version of `gated_native_table.py` labelled the three seeds of both bases "+ gated layer, seed N" and
keyed its episode dictionary by that label, so the second base's rows overwrote the first's and the "native base
+ layer" differences were silently computed from the *superseded* base's runs. It printed tower DEL -18.5
[-20.2, -17.1], which is the difference between the two bases, not what the layer does. It was caught because
the row disagreed with `evaluate.py`'s own per-wind print for the same run (+25.8 against +6.8) and because the
two bases' seed rows were identical to four digits. Rows are now keyed by path. Any table that joins several
evaluations has to key on something unique to the evaluation, and the cheapest check is that a re-derived number
matches the evaluator's own output for one row.

## 57. The main line, stated, and the documents reorganised around it (2026-10-13)

The user asked the question the record could not answer in one place: *is the main line not what agentic tuning
delivers on the shipped GSPI and on an untuned MPC, in regulation and load reduction?* It is, and the evidence
for it was spread over fifteen documents and fifty-six roadmap sections, with the top of `CLAUDE.md` still
carrying a reporting-methodology claim as the headline. Three things came out of checking it.

### 57.1 The main line is the shape of three legs, and the shape is the result

**Agentic search buys feasibility and sample efficiency on real controllers; it does not buy peak level.** The
same loop is applied to three objects and all three come out the same shape. Written up in full, with every
number sourced, in `docs/MAIN_LINE.md`; the three legs are the shipped ROSCO pitch loop (s51-s55), an untuned
offset-free LPV-MPC (s33, s38), and a residual layer's reward structure (s32-s41, s49, s56).

Two things were found while checking the claim, both of which change how leg 1 can be written:

- **The baseline row was not found by the agent.** Stage 1 came from `--proposer tune`, the fixed s29 recipe,
  which `scripts/rosco_search.py` marks in its own docstring as "used to re-derive the paper's tuned baseline
  in the native parameterisation, **not as an arm of the comparison**". So the strongest verified tuning in the
  study is a classical recipe's, not an agent's, and the paper has to say so.
- **The per-arm census under the per-wind rule is complete and it was free.** The design screen leaves exactly
  10 tunings with positive regulation - an eleventh is the shipped tuning itself at 0.00 by construction - and
  s55 measured all 10 off the design winds, so the census needs no further simulation: llm proposed 7 and **2
  hold** (`llm_r1#9`, the one-knob row at 1.000x travel, and `llm_r1#11` at 1.647x), local proposed 2 and both
  hold (but at 0.976x and 0.880x travel, buying +10.69 and +1.73 of regulation - the cheap end of the duty
  ladder, not a competitor for the baseline row), bayes proposed 1 and 0 hold, random proposed none. The s53
  count (llm 3/3 runs, random 0/3) was taken under the objective as stated; this is the count under the rule
  that now decides, and both belong in the paper.

Leg 3 needed no re-reading: **s49 already settled it at n = 8** - agent-written reward 4/5 runs holding the
per-wind load rule against random reward structures 1/8 (Fisher two-sided p = 0.032) and fixed rewards 0/8
(p = 0.007), with the *level* of the two searched arms identical. "More search found the level; it did not find
the constraint" is the main line's sharpest sentence and it was already measured.

### 57.2 Leg 2 measured in published terms, which it never had been

s38 established that the llm and es proposers both discover the wind-scheduled tower weight in 24 simulations
while random search finds nothing in 48, but it reported the discovery in composite points. In the four
published reductions, held-out seeds 3-6, 28 episodes, paired on the same wind file (no simulations - both rows
were already on disk):

| row | power MSE | speed MSE | tower DEL | blade DEL | travel |
|---|---|---|---|---|---|
| offset-free LPV-MPC, **unscheduled** | +47.3 | +52.4 | +15.0 | +3.4 | 1.981x |
| offset-free LPV-MPC, agent-found schedule | +45.7 | +50.7 | +16.5 | +2.9 | 2.140x |

Paired difference: -1.59 power, -1.63 speed, **+1.49 tower**, -0.50 blade, every interval resolved. The set
means understate what the schedule does, and the per-wind tower-base term is where it shows:

| U | unscheduled | agent-found schedule |
|---|---|---|
| 20 | +0.38 | +1.56 |
| 22 | **-2.46** | +0.24 |
| 24 | **-6.27** | +0.08 |

**The agent converts a controller that loses tower-base fatigue at high wind into one that does not, for 1.6
points of regulation.** That is a feasibility repair, the same thing it does on the other two legs, and it is
the honest way to report a search that does not beat the hand design.

The unscheduled reference had never been run on the fresh seeds 7-10 while every other rung of the ladder is
reported there. `campaign_mpc_unsched_fresh.sh` added that one evaluation (identical arguments to the scheduled
row of `campaign_gate.sh` except `qt_sched`; 28 episodes, 13 min). On the fresh seeds the leg is sharper, and
it is the fresh-seed version that the manuscript should carry (`scripts/dev/mpc_leg_table.py`,
`docs/tables/mpc_leg_fresh.csv`):

| row | power MSE | speed MSE | tower DEL | blade DEL | travel | ADC |
|---|---|---|---|---|---|---|
| tuned baseline, native (s52) | +25.3 | +29.7 | +25.3 | +1.7 | 2.016x | 0.0968 |
| offset-free LPV-MPC, **unscheduled** | +50.7 | +57.1 | +15.6 | +0.5 | 1.912x | 0.0918 |
| offset-free LPV-MPC, agent-found schedule | +49.1 | +55.4 | +17.4 | -0.0 | 2.076x | 0.0997 |

Paired on the same 28 episodes the schedule is **-1.65 power, -1.74 speed, +1.83 tower [+1.29, +2.72], -0.53
blade**. Per wind, tower-base DEL reduction:

| U | unscheduled | agent-found schedule | tuned PI baseline |
|---|---|---|---|
| 20 | +4.22 | +5.42 | +18.70 |
| 22 | +0.55 | +3.44 | +16.94 |
| 24 | **-8.76** | **-0.33** | +15.30 |

So the statement that leg 2 can carry, on winds nothing has seen: **the per-wind load rule of s28 is "no term
below -1 % at any wind speed"; the unscheduled controller fails it by 8.8 points at 24 m/s, and the agent-found
schedule brings it inside the tolerance for 1.7 points of regulation.** Two honest qualifications go with it.
At -0.33 the repaired term is *inside* the tolerance and not positive - the repair replicates on fresh seeds
but with no margin to spare (held out it was +0.08). And the tuned PI loop beats both MPC rows on tower fatigue
at **every** wind speed from 16 m/s up, at comparable duty (2.016x against 2.076x), which is the per-wind form
of s54 and s56: the model-based rung buys regulation and spends tower fatigue.

### 57.3 What was reorganised, and the rule that keeps it from happening again

- `docs/MAIN_LINE.md`, new and the only place the claim lives: the claim, the three legs with current numbers,
  **what cannot be claimed** (six items), the measurement discipline, and the open items in order.
- `docs/README.md` rewritten as an index **by leg** rather than by date, with a section map of this file so a
  2800-line log does not have to be read through to find which section serves which leg.
- `CLAUDE.md`: the flat list of 27 findings is grouped under the three legs, the discipline, the manuscript and
  the history, and its head points at `MAIN_LINE.md` first. Nothing was deleted - the regrouping was done with
  an assertion that every non-blank line survives, which caught the two intended exceptions (the old section
  heading and the manuscript entry promoted out of a nested bullet).
- The manuscript plan carries the consequence: v6's claim keeps all its evidence and becomes the paper's
  measurement discipline in the setup and discussion, and section 5 is reorganised by leg.

**The rule, now in `docs/README.md`:** a new experiment document opens with the leg it serves and one sentence
on which number in `MAIN_LINE.md` it changes; if it changes none it is a roadmap note, not a document; and when
a result does change a main-line number, `MAIN_LINE.md` is updated in the same commit. A stale claim in a short
file is worse than a long log.

## 58 The matched random-structure control for the gated layer, pre-registered (2026-10-13, launched 11:36)

Leg 3. This is open item 1 of `docs/MAIN_LINE.md`, and the reason it was run before the manuscript rewrite is
that it is the only queued experiment that can **move** the claim instead of confirming it. s49 settled "who
writes the reward" at n = 8 on the **ungated** tuned-PI base: under the per-wind load rule the agent-written
reward holds 4/5, a random structure drawn from the same grammar with the same fork verification holds 1/8, a
fixed reward 0/8, Fisher two-sided p = 0.032 against the random structures. The configuration the paper
headlines is the **wind-gated** one, and it has never been run against that control on either base - s40 and
s41 both record this as a limitation, and s56 re-trained only the agent arm on the baseline that is a real
tuning file. If random structures also hold the loads under the gate, the sharpest sentence in leg 3 has to be
scoped to the ungated base.

Arm `tnrrCwG3t` is `tnrwCwG3t` with `--supervisor random_reward` and nothing else changed; the two lines of the
arm table were diffed word by word before launch and differ in exactly that one token. Same base template
(`template_5mw_rosco_tuned_native`), same paired rollouts (`~/wtrl_rtn_range`, the 42 files whose identity check
read exactly 0.000 in s56), same winds, same knob file, same objective Cw, same gate 15-17 m/s, same 300
episodes of 150 s, and the same three training seeds 0 1 2 - so each control run pairs with an agent run seed by
seed, and the random draw is reproducible from the seed. `scripts/wsl/campaign_gate_native_rand.sh`, resumable,
pausable with `campaign_ctl.sh {pause|resume|stop|status} campaign_gate_native_rand`. Three training runs over
two lanes plus six 600 s evaluations; the reference rows of the base already exist from s56.

### 58.1 The decision rule, stated before any result

**Power first, because it decides what the outcomes are allowed to say.** With three control seeds against three
agent seeds the best possible Fisher outcome, 3/3 against 0/3, is two-sided **p = 0.1**. So this run cannot
establish the agent under the gate. It can refute the extension of s49 to the gated configuration, and that
asymmetry - the informative branch is the cheap one - is why three seeds run before any decision about five.

"Holds" means exactly what it means in s49 and s56: at 600 s, against the paired shipped controller, **no load
term below -1 % at any mean wind speed**, measured on held-out seeds 3-6 and on fresh seeds 7-10. The agent arm
is 3/3 by that rule, with a per-wind violation of 0.00 % in all six of its evaluations.

- **2 or 3 of 3 control seeds hold** - the proposer claim does not extend to the gated configuration. Leg 3 is
  scoped to the ungated PI base in `MAIN_LINE.md` and in manuscript section 5.5, which says so in the text
  rather than in a limitation sentence.
- **1 of 3 holds** - nothing is established in either direction (3/3 against 1/3 is p = 0.4). Decide by the
  per-wind load margin and by the level, and only then on whether to spend the 5-against-5 extension.
- **0 of 3 hold** - the direction agrees with s49, but at p = 0.1 it is not a result. The claim stays scoped
  until the extension is run: two more agent seeds and two more control seeds, about 6 h, which makes 5/5
  against 0/5 worth p = 0.008.
- **The level is secondary and is reported paired against the same base** (power MSE and gen-speed MSE against
  the native tuned baseline on the same episodes, as in s56). s49 found the two searched arms **identical** in
  level, so a level difference here would be a new finding and would need its own replication before being
  written anywhere.

### 58.2 The deciding rule applied to the agent arm, before the control finished

`scripts/dev/gate_proposer_control.py`, `docs/tables/gate_proposer_control.csv`. The measurement was fixed while
the control was still training, which is the point of doing it in this order.

**The rule is not the number s56 quoted.** s56 reported a per-wind violation of 0.00 %, which is the evaluator
per-wind Cw and is measured against the **shipped** controller; the rule that decides this question (s49,
implemented in `proposer_test.py` as `range600_worst_{tower,blade}_diff`) is the worst single-wind load term
against the run **own base**, tolerance -1 point, bin mean before the difference. Applying that rule unchanged,
with the s52 native tuning as the base:

| arm | fresh 7-10, worst tower / blade | held out 3-6, worst tower / blade | holds |
|---|---|---|---|
| agent-written reward, seed 0 | -0.08 / -0.08 | -0.10 / +0.00 | yes |
| agent-written reward, seed 1 | **-0.98** / +0.00 | -0.50 / +0.00 | yes |
| agent-written reward, seed 2 | -0.03 / -0.22 | -0.18 / -0.40 | yes |

So the agent arm is **3/3 on both seed sets** under the rule that decides, not only under the weaker per-wind Cw
of s56. One thing goes beside that count: **seed 1 holds by 0.02 points** (tower DEL at 18 m/s on fresh seeds),
and it is also the seed whose set-mean tower difference is the one unresolved interval of s56
(+0.01 [-0.24, +0.25]). A count of 3/3 whose worst member sits two hundredths of a point inside the tolerance is
a count a re-run could flip, and the paper says so wherever it says 3/3.

The level, paired against the same base on the same episodes, is the s56 row unchanged: power MSE +5.80 to
+6.44, speed MSE +5.91 to +6.78 on fresh seeds, at 1.22 to 1.29x the base pitch travel.

### 58.3 The result: under the gate the proposer is not distinguishable, and the gate is why

Finished 2026-10-13 15:46, 4 h 10 min, three control seeds and six 600 s evaluations, no infrastructure
failures. `docs/tables/gate_proposer_control.csv`. **This is the branch of 58.1 that moves the claim.**

Per-wind load rule, worst single-wind load term against the native base (tolerance -1 point):

| seed | agent, fresh twr / bld | random, fresh twr / bld | agent, held out | random, held out |
|---|---|---|---|---|
| 0 | -0.08 / -0.08 | -0.06 / -0.02 | -0.10 / +0.00 | **+0.00 / +0.00** |
| 1 | **-0.98** / +0.00 | -0.07 / +0.00 | -0.50 / +0.00 | **+0.00 / +0.00** |
| 2 | -0.03 / -0.22 | -0.02 / -0.16 | -0.18 / -0.40 | **+0.00 / +0.00** |

**Agent 3/3, random reward structure 3/3, on both seed sets, Fisher two-sided p = 1.0** - and the control holds
with **more** margin than the agent: no random seed is negative at any wind speed on the held-out set, while the
agent arm has its -0.98 at 18 m/s on fresh seeds. The control is a real control: both arms adopted a proposal at
6 of their 7 supervision points, and the kept expressions are structurally comparable draws from the same
grammar (the control seeds kept, for instance, `86.5*tanh(6*(p_ratio-1))*(region==0) - 27.1*tanh((d_wg/0.005)**2)*(region_w==1) - ...`).

Level, paired against the same base on the same episodes (power MSE / speed MSE added, fresh seeds): agent
+5.80 / +5.91, +5.80 / +6.16, +6.44 / +6.78; random **+7.89 / +8.20, +6.13 / +6.37, +6.74 / +6.98**. Held out
the ordering repeats (+7.44, +5.80, +6.34 against +5.40, +5.72, +6.24). The direction is consistent on both
seed sets but **it is not a result**: n = 3 per arm, the arm means are within each other spread, and the control
spends more actuation (1.23 to 1.49x the base pitch travel against 1.22 to 1.31x). The one duty-matched pair
makes the point without the duty confound - random seed 1 at 1.226x against agent seed 1 at 1.220x is
+6.13 / +6.37 against +5.80 / +6.16 with tower DEL +0.62 against +0.01. **No agent advantage is visible under
the gate in either the level or the constraint.**

**Why, and it is in the per-wind block.** Both arms are exactly +0.00 at 12 m/s by construction (the gate is
shut) and the worst cell of any of the six runs at 14 m/s is -0.09. The ungated failures of s34 and s49 were
exactly there: the fixed reward lost 3 to 7 points of tower fatigue at 12-16 m/s, and that is the damage the
agent-written reward had to avoid by writing the load terms itself. **The gate removes the bins where the
constraint was hard, so the configuration supplies the feasibility that the proposer used to supply.** Stated
as a mechanism it is consistent with every number we have: searching the structure buys the level (s49, +4.02
against +4.01); the proposer buys the constraint only where the constraint is actually binding (s49 ungated,
4/5 against 1/8, p = 0.032); under the gate it is not binding and the arms are identical (here, 3/3 against
3/3). It also explains s40, where the two arms were indistinguishable on the scheduled MPC base - a base that
already holds its own loads.

**What changes.**
1. Leg 3 of `docs/MAIN_LINE.md` is **scoped in the text, not in a limitation**: the proposer effect is a
   property of the ungated tuned-PI base. In the configuration the paper headlines, a random draw from the same
   grammar holds the loads just as well, with more margin.
2. The honest form of the agentic claim for leg 3 is about the **search space**, not the proposer: a searched
   reward *structure* is what buys both the level and, where it binds, the constraint. The language model is
   distinguishable from a random draw out of its own vocabulary in exactly one configuration of this study.
3. The gate is prior art (Kim arXiv:2609.21307, Abbas IET CTA 2026; `docs/paper/literature_2026-09-23.md`), so the
   device that supplies the feasibility is not ours either. Say so where the gate is introduced.
4. Legs 1 and 2 are untouched: their controls are random *search* in the design space (0/3 runs on leg 1,
   0/48 on leg 2), not a random draw from a reward grammar, and both still separate.

**The cheapest next question, not run:** a fixed-reward arm under the gate on the native base, three runs,
about 3 h. If the fixed reward also holds, the structure search buys nothing under the gate either and leg 3
collapses to the gate; if it does not, the search is still load-bearing and only the proposer is not. s41 has
the fixed reward under the gate on the MPC base losing 5.9 points of tower fatigue at 20 m/s, which points to
the second reading, but the native base has never had that arm.

## 59 Leg 1 at n = 6: the one proposer claim s58 did not touch, endpoints fixed first (2026-10-13, 23:11)

s58 removed the only configuration in which the language model was measurably better than a random draw out of
the same space, and it did so on leg 3. That leaves exactly one proposer statement standing anywhere in the
study, leg 1: in the ROSCO-native search of s53 the llm arm bought a shippable gain in 3 of 3 runs against
2 of 3 for constrained Bayesian optimisation, 1 of 3 for the local search and 0 of 3 for random draws, and it
reached its first gain in a median of 8 simulations against 14 and 18. **Nothing in that sentence is a test:**
3 of 3 against 2 of 3 is Fisher two-sided p = 1.0, and a median over two or three runs is a number with no
spread. The claim the manuscript would make - the same loop reaches the level of the best dedicated search in
each space and gets there in fewer simulations - rests on this leg, so this doubles it.

Nine runs: `llm`, `bayes`, `local`, repeats 3, 4, 5, through `campaign_rosco_native_n6.sh`, which sets `ARMS`,
`REPEATS` and `STAGES` and then execs `campaign_rosco_native.sh` unchanged - same box read off the plant, same
shipped start, same budget of 24 candidates, same constrained selection, same 14-episode design protocol, same
cache by parameter hash, same stage 3. About 1 h 45 min per run from the s53 log, so about 16 h, then stage 3
re-scores only the new selections on held-out seeds 3-6 and fresh seeds 7-10 because `_verify` caches by
parameter hash as well (about 12 min per new row). Pausable: `campaign_ctl.sh {pause|resume|stop|status}
campaign_rosco_native`. `random` deliberately stays at three runs: all three of its runs selected the shipped
tuning itself, its job is only to show the space is not trivially searchable, and 0 of 3 against 6 of 6 is
already p = 0.012.

### 59.1 The endpoints, the power and the outcome branches, written before any of the nine runs exists

s55 is the reason this section exists. The number the search maximises is not the number that survives: every
band member that bought more regulation than the standing baseline failed the per-wind load rule off the design
winds, and **all three llm selections of s53 are in that set** - their worst single-wind load term on fresh
seeds is -2.87, -2.07 and -3.73 points (tower, 24 or 12 m/s) against +0.25 for the bayes selection that holds
and +0.33 for the local one. So the arm that looks strongest under the set-mean selection rule is the arm whose
selections do not ship, and an endpoint had to be chosen before the new runs land rather than after.

**Primary: simulations to the first feasible gain** (`sims_to_first_feasible_gain`, already a column of
`docs/tables/rosco_search.csv`), Mann-Whitney one-sided, llm against bayes, the strongest classical arm. A run
that never finds a feasible gain within its 24 candidates is censored and enters the test tied at 25, ranked
last. Power: the smallest one-sided p available at 6 against 6 is 1/C(12,6) = 0.0011; if s53 replicates
(llm near 8-12, bayes spread 7-21 with two censored) the test reads about p = 0.031, and a weaker replication
reads about 0.054. **This endpoint is the claim**: it is sample efficiency, which is what the main line says
agentic search buys, and it is the only endpoint n = 6 can resolve.

**Secondary A: how many runs per arm deliver a shippable gain** (`selected_feasible` and not
`selected_shipped`). Underpowered and stated as such in advance: 6 of 6 against 4 of 6 is Fisher two-sided
p = 0.45, against 3 of 6 p = 0.18, and only 2 of 6 or fewer reaches 0.06 two-sided / 0.03 one-sided. A
direction, not a result, unless bayes collapses.

**Secondary B: the per-wind restatement, which is the honest version of leg 1.** For each of the 21 runs, apply
the per-wind screen to that run's own candidates as `rosco_perwind_select.py` did for the band in s52.3 -
screen on the design set, then verify off the design winds - and count the runs that deliver a tuning holding
both load terms at every wind speed with a real regulation gain, plus the simulations spent before the first
such candidate. This is the quantity a user of the tool actually gets, and it is where the llm arm currently
reads 0 of 3 on its selections while its *candidates* hold twice (`llm_r1#9` and `#11`, the corner ladder of
s52.3). It needs two evaluations per newly selected tuning off the design winds, which is cheap, and it is
reported beside the primary, never instead of it.

**Validity check, not an endpoint:** `propose_llm` fills the budget from the box when a reply is malformed and
records `filled randomly` in the history, so each llm run is audited for how many of its 24 candidates were
actually the model. A run diluted that way is reported, not quietly averaged in.

Outcome branches, so the reading is fixed in advance:

1. **Primary separates and secondary B puts llm no worse than the classical arms:** the leg-1 claim stands as
   written, it is the paper's one proposer claim, and s58 scopes the leg-3 one.
2. **Primary separates and secondary B does not:** the claim becomes narrower and must be written that way -
   the model finds the largest *set-mean* number fastest, and that number does not ship. The proposer then
   buys search efficiency against a defective objective, which is a finding about the objective (a fourth
   instance after s32, s53.2 and s55) and not about the controller.
3. **Primary does not separate:** leg 1 joins s58, and then **no leg of this study has a proposer claim left**.
   What survives is the search *space* (a searched reward structure buys the level, leg 3; a wind schedule
   repairs a per-wind violation, leg 2; one filter corner is worth a fifth of the regulation error, s52.3) and
   the measurement discipline. The manuscript would say so, and the agentic part of the title would have to go.

### 59.2 The restatement, done before the new runs land: under the rule the project ships by, no arm delivers

No simulations - `scripts/dev/rosco_proposer_test.py`, `docs/tables/rosco_proposer_test.csv`. The twelve runs of
s53 are read three ways from the files on disk, and the five tunings the per-wind rule re-selects were all
verified already as band members of s55, so the third reading costs nothing either.

            run |  set-mean rule: sims  regulation  design w/w  fresh w/w |  per-wind: sims  regulation  design w/w  heldout w/w  fresh w/w  ships
      random r0 |                    -        0.00        0.00       0.00 |                -        0.00
      random r1 |                    -        0.00        0.00       0.00 |                -        0.00
      random r2 |                    -        0.00        0.00       0.00 |                -        0.00
       local r0 |                   18       14.30       -0.00      +0.33 |               18       14.30       -0.00        +1.12      +0.33    yes
       local r1 |                    -        0.00        0.00       0.00 |                -        0.00
       local r2 |                    -        0.00        0.00       0.00 |                -        0.00
       bayes r0 |                    -        0.00        0.00       0.00 |                -        0.00
       bayes r1 |                    7       46.76       -2.30      -1.24 |                7       34.59       -0.60        -0.47      -2.04     NO
       bayes r2 |                   21       15.37       -0.70      +0.25 |               21       15.37       -0.70        +2.37      +0.25    yes
         llm r0 |                    8       47.79       -5.90      -2.87 |                -        0.00
         llm r1 |                    8       41.27       -6.60      -2.07 |                8       36.96       -0.60        +1.06      -1.17     NO
         llm r2 |                   12       37.47       -2.20      -3.73 |               12       33.60       -0.70        -1.94      -3.40     NO

Three readings of the same twelve runs, and they do not agree:

1. **The rule the search used** (the three constraints on the set mean): llm 3 of 3 runs with a shippable gain,
   median 8 simulations to it, regulation of the selections +42.18 (5.22); bayes 2 of 3, median 21, +31.06
   (22.20); local 1 of 3, 18, +14.30; random 0 of 3. Against bayes the simulations read one-sided exact
   p = 0.30 and the counts Fisher p = 1.0. **At n = 3 nothing separates the model from Bayesian optimisation**,
   which is exactly what s59 was queued to settle.
2. **The same rule with the load constraints required at every wind speed, screened on the design set:** llm
   2 of 3 (llm r0 has no per-wind-holding candidate at all), bayes 2 of 3, local 1 of 3, random 0 of 3, and the
   holding selections are llm +35.28 (2.38) against bayes +24.98 (13.59).
3. **That rule measured off the design winds - what a user of the tool actually gets: llm 0 of 3, bayes 1 of 3
   (+16.57 fresh), local 1 of 3 (+11.62), random 0 of 3.** The llm selections miss the tolerance by 0.17 points
   (llm r1 #12: design -0.60, held out +1.06, fresh -1.17) and by 2.40 (llm r2 #24: -0.70, -1.94, -3.40), and
   the bayes r1 selection misses as well (-0.47, -2.04). Every miss is on the tower-base load and every one is
   worse on fresh seeds than held out, the fourth instance of that pattern after s32, s53.2 and s55.

**And the mechanism is not the proposer.** llm r1 visited two tunings that do hold off the design winds - #9 and
#11, the filter-corner ladder of s52.3, one of them the +22.6-regulation row at 1.004x pitch travel that sits in
the verified band of s55 - and its own rule selected #12 instead, because on 14 design episodes #12 reads
+36.96 of regulation against +22.6 with a worst single wind of -0.60. **The screen is what fails, not the
search:** two episodes per wind cannot tell -0.60 from -1.20, so the regulation-maximising candidate wins the
tie and the shippable one is left in the cache. s52.3 said the per-wind screen is a screen and not the decision;
this is what that sentence costs, measured.

What it does to s59: the primary endpoint is untouched, because it asks how fast the search finds a gain under
the rule the search was given. Secondary B now has this third form, and at n = 6 it is counted for all 21 runs,
each new per-wind selection costing two evaluations of 28 episodes. What n = 6 cannot do is repair a selection
rule that resolves nothing at 14 episodes - screening with a margin (s55 found the set-mean margin predicts
holding), or spending episodes at the winds where the constraint binds, is a separate and cheap question, and
it is not this campaign.

### 59.3 Paused at 4 of 9 runs (2026-10-14 07:31, on request)

`campaign_ctl.sh pause campaign_rosco_native` SIGSTOPped all 36 processes of session 400; the machine is idle
and nothing is lost. On disk with a `best.json`: llm r3 (started 23:11), bayes r3 (00:59), local r3 (02:43),
llm r4 (04:26), each 25 history records, each about 1 h 45 min. In flight: bayes r4, 17 records of 25, started
06:12, with two 600 s evaluations frozen mid-OpenFAST. Not started: local r4, llm r5, bayes r5, local r5, and
then stage 3. About 9 h 40 min of work left.

Resume with `bash scripts/wsl/campaign_ctl.sh resume campaign_rosco_native`. If the WSL VM is shut down while
the session is stopped, the frozen processes die with it; relaunching `campaign_rosco_native_n6.sh` then
continues correctly, because a finished run is skipped on its `best.json` and bayes r4 is a deterministic
proposer whose re-walked candidates all hit its own parameter-hash cache. An interrupted run of the llm arm
would instead start its budget over, so a stop during an llm run costs that run.

**The tests of s59.1 are not to be computed on this partial set.** They are pre-registered at n = 6 per arm,
four finished runs is two arms at n = 4 and one at n = 4 in part, and reading them now is exactly the peeking
the pre-registration exists to prevent. `rosco_proposer_test.py` will be run when the ninth run and stage 3
are done.

Resumed 2026-10-14 09:57 on request, 2 h 26 min after the pause: all 36 processes continued from SIGSTOP,
bayes r4 went on from its candidate 17 of 24, and the frozen OpenFAST evaluations finished normally, so the
pause cost nothing but wall clock. Expect the ninth run and stage 3 around 19:40 today.

### 59.4 The primary endpoint is null: at n = 6 the model is not distinguishable from classical search (2026-10-14)

All nine runs finished - llm r3 23:11, bayes r3 00:59, local r3 02:43, llm r4 04:26, bayes r4 06:12, paused
07:31 to 09:57 on request, local r4 10:28, llm r5 12:12, bayes r5 13:59, local r5 15:42, stage 3 started 17:26 -
21 runs and 561 tunings in the space in total, 1 h 44 min per run, no infrastructure failures. Paused again at
17:45 with stage 3 at 1 of its 18 new evaluations, so **the primary endpoint below is final (it is computed from
the 21 finished histories and nothing pending can change it) and the delivery count is not.**

                        runs   with a gain   sims to it: median            values     regulation of the gains
    the rule the search used (the three constraints on the set mean)
            random         3        0 of 3                 25.0      [25, 25, 25]                           -
             local         6        4 of 6                 14.0  [8,9,10,18,25,25]              +29.23 (12.14)
             bayes         6        5 of 6                 18.0  [5,7,15,21,21,25]              +23.79 (14.03)
               llm         6        5 of 6                 10.0   [6,8,8,12,13,25]               +38.40 (8.47)

**Primary, pre-registered in s59.1: simulations to the first feasible gain, one-sided exact permutation.**
llm against bayes **p = 0.3117**; llm against local p = 0.1937; llm against random p = 0.0476. Counts, the
pre-declared underpowered secondary: llm 5 of 6, bayes 5 of 6 (Fisher p = 1.0), local 4 of 6 (p = 1.0), random
0 of 3 (p = 0.0476). **This is outcome branch 3 of s59.1.** The model separates from random draws and from
nothing else: at n = 6 it is not distinguishable from constrained Bayesian optimisation or from a (1+lambda)
local search, on either endpoint. The s53 impression - 3 of 3 and a median of 8 against 14 and 18 - was three
runs per arm, and it did not survive doubling: llm r3 found nothing at all, the first llm run ever to do so,
and bayes r4 found its gain in 5 simulations.

The level is the one place the model is still ahead and it is the place where being ahead is worth least:
the regulation of its selections is +38.40 (8.47) against +29.23 (12.14) for local and +23.79 (14.03) for
bayes, i.e. it walks further up the front of s52.2 - and the front above about +25 points is where s55 found
every member fails the per-wind load rule off the design winds. Which is what happens here:

    the same rule with the loads required at every wind speed, MEASURED off the design winds (what a user gets)
            random         3        0 of 3                                                              -
             local         6        1 of 6   (3 selections still unverified: r3, r4, r5)           +11.62
             bayes         6        1 of 6   (2 still unverified: r3, r5)                          +16.57
               llm         6        0 of 6   FINAL - all six resolved                                   -

The llm column is final because its three per-wind selections are all measured already: llm r1 #12 and llm r5
#11 are **the same tuning** (design -0.60, held out +1.06, fresh -1.17, missing the tolerance by 0.17 points)
and llm r2 #24 fails by 2.40, while llm r0, r3 and r4 have no per-wind-holding candidate at all. The five
pending rows can only raise local and bayes, and their design margins say they probably will: the classical
selections sit at +0.10, +1.20, +1.30, +0.60 and +1.20 points of worst-single-wind margin, where the model
selections sit at -0.60 and -0.70, and s55 established that this margin is what predicts holding. So the
likely final shape is **the classical arms deliver a shippable tuning and the model arm does not**, which is
the opposite of the s53 reading, and the mechanism is not mysterious: the model is better at maximising the
objective it was given, and the objective it was given is the one s32, s53.2 and s55 proved gameable. Fourth
instance, now with the agent on the losing side of it.

**Two things the model does win, both post hoc and neither pre-registered, so they are reported as what they
are.** First, **it never proposed a candidate the plant could not run: 0 of 144 candidates diverged, against
13 of 144 for the local search (Fisher p = 0.00018), 9 of 144 for Bayesian optimisation (p = 0.0034) and 16 of
72 for random draws (p < 1e-5)**, across six independent runs and with every one of its 144 candidates actually
proposed by the model (the `filled randomly` audit of s59.1 is 0 in all six runs). It reads the five quantities
as a controller rather than as a box, and it never asked for a gain schedule that destabilises the turbine.
That is a usability property, not a performance one, and it is the kind of thing a practitioner cares about.
Second, **its runs repeat each other**: 3 per-wind selections, 2 distinct tunings, against 4 of 4 distinct for
both classical arms - the search is reproducible across seeds in a way a GP with a random start is not.

What this does to the study: **no leg now has a proposer claim against a dedicated search.** Leg 3 lost it to
s58 under the gate, leg 2 never had it (llm = es, only random search loses), and leg 1 has now lost it at
n = 6. What survives is three statements that do not mention who proposes: the search *space* matters (a reward
structure, a wind schedule, one filter corner), the measurement discipline (set-mean constraints, baseline
tuning, aggregation and weighting) decides every apparent benefit, and the same loop runs in three different
design spaces without being rebuilt - at parity with the best dedicated search in each, not above it. s59.1
said in advance that the agentic part of the title would have to go if the primary came out null. It came out
null.

Stage 3 rebuilds `docs/tables/rosco_search.csv` for all 21 runs and re-scores every run's selection off the
design winds, after which `rosco_proposer_test.py` gives the delivery counts; it covers three of the five
pending per-wind rows and not the other two (s59.5 says which, and how much is left). Nothing else in this
section depends on it.

### 59.5 Paused in stage 3 on request, with three of the five pending per-wind rows already in (2026-10-14 23:23)

Paused at 23:23 on request, SIGSTOP of all 20 processes of session 400 (state `T`), mid-evaluation of the
bayes r5 selection on the fresh seeds. Resume with `bash scripts/wsl/campaign_ctl.sh resume
campaign_rosco_native`; the interrupted evaluation continues from where it stopped, nothing but wall clock is
lost, and if the WSL VM goes down meanwhile relaunch `campaign_rosco_native_n6.sh`, which skips stage 2 run by
run on `best.json` and re-enters stage 3 against the same parameter-hash cache.

**What stage 3 actually does, stated correctly because s59.4 got it wrong:** it re-scores the SET-MEAN
selection of each of the 21 runs on held-out seeds 3-6 and fresh seeds 7-10, two evaluations of 28 episodes
per distinct tuning, cached by parameter hash. 37 of those 42 evaluations are on disk; **5 are left** - bayes
r5 on the fresh seeds (interrupted), and llm r4 #23 and llm r5 #24 on both seed sets - at about 11 min each,
so **about 1 h**. It resolves a *per-wind* selection only where that run's per-wind selection IS its set-mean
selection, which is the case for local r3, bayes r3 and bayes r5 and NOT for local r4 (#10 against #21) or
local r5 (#18 against #9). **Those two need their own pass of four evaluations, about 45 min**, so the
delivery count of s59.4 is not final when stage 3 ends.

Three of the five pending rows landed before the pause, and they move in the direction s59.4 said the design
margins pointed:

    per-wind selection      design (worst)   held out (worst)      fresh (worst)   travel   ships
       local r3 #8           +27.64 (+0.10)   +26.76 (+0.32)    +28.66 (-0.87)      1.05x   yes
       bayes r3 #22          +24.35 (+0.60)   +24.86 (+0.57)    +26.72 (-1.15)      1.07x   NO, by 0.15
       bayes r5 #15          +22.36 (+1.20)   +24.03 (+2.17)    interrupted         1.42x   pending

So the delivery count under the rule that ships now reads **llm 0 of 6 (final), local 2 of 6, bayes 1 of 6
with one row pending, random 0 of 3**, and two local rows (r4 #10, r5 #18) are not measured at all yet. The
tolerance is -1.0 point on the worst single-wind load term, so bayes r3 misses it by 0.15 on the tower-base
load - and **every one of the three new rows is worse on fresh seeds than held out** (+0.32 to -0.87, +0.57 to
-1.15), the fifth instance of that pattern after s32, s53.2, s55 and s59.2. The primary endpoint is untouched:
it is computed from the 21 finished histories.

One row worth naming on its own: **local r3 #8 (omega 0.633, zeta 0.664, corner 3.214, damper 0.0012, hpf
0.360) holds both loads at every wind speed off the design winds with +28.66 of regulation at 1.05x shipped
pitch travel.** The standing baseline reads +29.75 fresh at 2.016x, so the selection rule still picks stage 1,
by 1.09 points - but this is a sixth row for the duty ladder of s55.2-55.3, where the one-knob member buys
+23.0 at 1.000x, and it was found by the local search, not by the model.

Resumed on request 2026-10-15 09:31, and the pause did not survive: the WSL VM had gone down overnight, so
session 400 was gone and `campaign_ctl.sh resume` had nothing to continue. `integrity_check.py --since
"2026-10-14 17:00"` reports no broken artefact, and the relaunch of `campaign_rosco_native_n6.sh` (session 409)
skipped all nine runs on their `best.json` and re-entered stage 3 against the parameter-hash cache, so the only
thing lost is the one evaluation that was in flight at the pause - bayes r5 on the fresh seeds, which starts
again: 37 of the 42 row-evaluations of stage 3 are cache hits, and 12 of the 15 distinct selections are
measured on both seed sets.

The pass stage 3 does not cover is now a script rather than a note: `scripts/dev/rosco_perwind_select.py
--verify` evaluates each run's PER-WIND selection on held-out and fresh seeds into the same cache, which closes
the gap for local r4 #10 and local r5 #18 and will close it for any future run whose two rules disagree, and
`scripts/wsl/campaign_perwind_verify.sh` (session 557) waits for stage 3 to release the seven workers, runs it,
and then prints the closing delivery table from `rosco_proposer_test.py`. Four evaluations, about 45 min,
queued behind stage 3's remaining five.

### 59.6 The campaign is closed, and the delivery count reverses the s53 reading completely (2026-10-15 11:18)

Stage 3 finished at 10:23:49 and the pass it does not cover at 11:18. **The whole of s59 is now measured: 21
runs, 561 tunings proposed, 486 of them distinct, 39 the plant could not run, 211 holding the three set-mean
constraints, 114 of those also holding both loads at every wind speed on the design screen, and every one of
the 11 per-wind selections verified on 28 held-out and 28 fresh episodes.** Stage 3's own by-products: 21 of 21
set-mean selections stay feasible on both seed sets, design against held-out correlates +1.00 (rank +1.00) over
the 21, and the design score is pessimistic on average (optimism -0.43 points, worst +0.88) - the s38 warning
that a small design set cannot rank designs still does not transfer to this space.

One infrastructure failure, reported as such: the first run of the new verification pass
(`campaign_perwind_verify.sh`) did not export `WTRL_WIND` / `WTRL_HOME`, so `evaluate.py` looked for the 150 s
wind bank, raised `FileNotFoundError: ~/wtrl/wind/U12_TIB_S3.bts` and wrote four `failure_kind:
infrastructure` records in 20 seconds. Those four cache entries were deleted, the script now exports the 600 s
bank and its paired baselines as the campaign does, and the four evaluations ran properly from 10:38 to 11:18.
No candidate and no result was affected; the counts below are from the re-run.

The three readings of the same 21 runs, final:

                     runs   with a gain   sims: median                  values   regulation of the gains
    the rule the search used (the three constraints on the set mean) - the primary endpoint lives here
            random     3        0 of 3             25.0            [25,25,25]                          -
             local     6        4 of 6             14.0     [8,9,10,18,25,25]             +29.23 (12.14)
             bayes     6        5 of 6             18.0     [5,7,15,21,21,25]             +23.79 (14.03)
               llm     6        5 of 6             10.0      [6,8,8,12,13,25]              +38.40 (8.47)
    the same rule with the loads required at every wind speed, SCREENED on the design set
             local     6        4 of 6             18.0    [8,10,18,18,25,25]              +19.81 (6.02)
             bayes     6        4 of 6             21.5    [7,15,21,22,25,25]              +24.17 (7.94)
               llm     6        3 of 6             18.5     [6,8,12,25,25,25]              +35.84 (1.94)
    the same rule MEASURED off the design winds - what a user of the tool gets
            random     3        0 of 3             25.0            [25,25,25]                          -
             local     6      **4 of 6**           18.0    [8,10,18,18,25,25]              +19.87 (7.43)
             bayes     6        2 of 6             25.0   [15,21,25,25,25,25]              +19.94 (4.76)
               llm     6      **0 of 6**           25.0   [25,25,25,25,25,25]                          -

**The model arm delivers nothing in six runs; the local search delivers in four of six.** llm against local is
Fisher two-sided p = 0.0606 on the counts (0.0303 one-sided in the direction the data actually went, which is
not the direction s59.1 set the test up to detect); llm against bayes 0 of 6 against 2 of 6, p = 0.4545. This
is secondary B of s59.1, pre-registered and declared underpowered in advance, and it came out reversed: the
hypothesis was that the model arm would be no worse than the classical arms under the per-wind rule, and it is
the only arm that hands a user nothing.

Each run's per-wind selection off the design winds, which is where the count comes from (regulation; worst
single-wind load term on the design screen, held out, fresh; tolerance -1.0 on both measured sets):

                 design   held out   fresh   design w/w   heldout w/w   fresh w/w   ships
      local r0 #21    14.30     14.20   11.62        -0.00         +1.12       +0.33   yes
      local r3  #8    27.64     26.76   28.66        +0.10         +0.32       -0.87   yes
      local r4 #10    15.97     19.07   16.40        +1.20         +1.99       +1.10   yes
      local r5 #18    21.32     22.84   22.78        +1.30         +2.10       +1.06   yes
      bayes r2 #21    15.37     18.58   16.57        -0.70         +2.37       +0.25   yes
      bayes r5 #15    22.36     24.03   23.30        +1.20         +2.17       +0.84   yes
      bayes r1  #7    34.59     35.07   37.43        -0.60         -0.47       -2.04   NO
      bayes r3 #22    24.35     24.86   26.72        +0.60         +0.57       -1.15   NO
      llm r1 #12 = llm r5 #11
                      36.96     36.98   37.74        -0.60         +1.06       -1.17   NO
      llm r2 #24      33.60     32.66   34.83        -0.70         -1.94       -3.40   NO
    llm r0, r3, r4 and the three random runs have no candidate that holds both loads per wind on the design set

**The mechanism is position on the front, not the proposer, and the table says it twice.** Everything that
ships sits between +11.6 and +28.7 points of regulation; everything that fails sits between +26.7 and +37.7.
The model's selections are the three highest rows in the table and all three fail on the tower-base load; the
local search delivers four times because its selections sit lower. The one local row high on the front
(r3 #8, +28.66 fresh) ships by 0.13 points, and the two rows stage 3 never looked at (r4 #10, r5 #18) ship with
the largest margins of all (+1.10 and +1.06 fresh), exactly as their design margins of +1.20 and +1.30 said
they would - fifth instance of the s55 rule that the margin, not the level, predicts holding.

So leg 1 closes like this: **the model reaches the best set-mean number fastest and the search space is real,
but under the rule the project ships by the agent arm delivers 0 of 6 against 4 of 6 for a plain (1+lambda)
local search, and the single row a user would actually take out of 21 runs is local r3 #8 - one filter corner
and a small damper, +28.66 of regulation at 1.05x shipped pitch travel.** The primary endpoint is unchanged
(p = 0.3117 against bayes), so s59.1's outcome branch 3 stands and the agentic part of the title goes; this
section adds that the honest restatement is not neutral but against the proposer, and that it would have been
missed by measuring only what the search itself selected.

## 60 The narrative fixed, the documents regrouped, and C1's novelty checked (2026-10-15, no simulations)

**The user's decision, in their words:** the main narrative carries only the positive results; the headline
contribution is **the first use of an LLM agent in controller tuning in the wind turbine field**; the negative
results are considered later for the appendix. This section records what that implies, what was moved, and the
one thing that had to be verified before the narrative could rest on it.

### 60.1 The documents, in three groups

Sixteen markdown files totalling 7260 lines were scattered across one directory with no indication of which
served the paper. They are now in three, and every file is in exactly one:

- **`docs/paper/`** - what the main text draws on: the new `NARRATIVE.md` (the claim, contributions C1-C5
  with a novelty status each, the ten sections with their evidence, figures F1-F6, tables T1-T7, and eight
  wording guardrails), the new `EVIDENCE.md` (the number book: every main-text number, by section, with its
  table and its roadmap section, so the paper can be written without reading this log), plus the leg-1 search
  design, the data-and-caveats file and the four literature-audit files.
- **`docs/appendix/`** - measured and real but not in the main narrative: the new `APPENDIX_PLAN.md`
  (appendices A-J) and the IPC negative result.
- **`docs/archive/`** - superseded or wrong: the new `ARCHIVE.md` (why each file is out, what replaced it, and
  the five retracted claims kept visible), the two F-era documents, the two earlier literature scans, and
  `plan_2026-10-12_manuscript-rewrite.md`, whose site-by-site table remains the correct checklist for the
  edit reasons (A)-(D) even though its framing was written under a claim that has since changed twice.

`MAIN_LINE.md`, this log and `RESTORE.md` stay at the top level. **`MAIN_LINE.md` was not narrowed to match
the paper:** it stays the project's complete record, nulls and *what cannot be claimed* included, and now says
in its header that the paper's scope is deliberately smaller and where that scope is written down. All
inbound references were repaired - 18 files, including four scripts whose docstrings name the search-design
document - and `CLAUDE.md` was edited at byte level so its CRLF line endings survived (the first `sed` pass
converted six CRLF files to LF; all six were restored and verified against `HEAD`).

The tables and figures were classified rather than moved, because the generators write to fixed paths:
`NARRATIVE.md` §4-§5 names which figure and which CSV each float comes from, which are appendix-only, and
which 17 CSVs are **superseded and may not be cited** (the `table_J_*` family, the 42-episode range tables,
and everything else measured over the composite or the hand-scaled baseline that s52 voided).

### 60.2 C1's novelty, checked before the narrative was allowed to rest on it

The 2026-09-23 audit covered LLM-written *rewards* (A(iii)) but never asked whether anyone has put an LLM in a
wind turbine control-design loop. Four targeted searches on 2026-10-15 - LLM agent + wind turbine controller
tuning; "large language model" + wind turbine + ROSCO gain tuning; the one AI-agents-in-wind-energy review
that surfaced; LLM agent + OpenFAST + simulation-in-the-loop tuning - returned **no instance**. The nearest
work is all in other domains and must be cited beside the claim: ControlAgent (arXiv:2410.19811, LLM agents
designing controllers), Shou et al. (arXiv:2607.26594, an LLM agent tuning PID for 200 chemical plants),
Chen et al. (Processes 14:2954, 2026, which benchmarks LLM gain *ranking* at 31.8-53.2 % accuracy and
concludes that simulator-based validation remains necessary - the best available justification for our
deterministic verification loop), and Wu et al. (Building Simulation 2026, LLM reward *weights* on a real
energy system). **So C1 holds as a domain-first with "to our knowledge", and one item must be opened before
it is printed:** Teke, *The Role of Artificial Intelligence Agents in Optimization and Control of Wind Energy
Systems*, ICENSOS 2025 - title confirmed, abstract not retrievable in this session.

### 60.3 What the positive narrative is, and the one thing it must still disclose

The claim: one agentic loop, three design spaces, no per-space machinery, and every candidate it proposes is
a controller the plant can run. The main text's four positive deliverables are the agent's two verified
shipped-duty rows on the pitch loop (`llm_r1#9`: +17.73 / +22.98 / +8.66 / +2.02 at **1.000x** travel, found
in 8 simulations; `llm_r1#11`: +24.20 / +28.35 / +16.25 / +2.48 at 1.647x - both holding every load term at
every wind speed on 28 held-out and 28 fresh episodes, where random search finds nothing in 72 candidates),
the MPC feasibility repair (tower DEL at 24 m/s -8.76 -> -0.33 in 24 simulations, random 0 of 48), the gated
layer's +5.8 to +6.4 power MSE over the tuned baseline with tower fatigue never negative, and the
**0 of 144 unrunnable candidates** against 13/144, 9/144 and 16/72 (p = 0.00018, 0.0034, < 1e-5), which is
the publishable form of the stability the user asked about and is reported as post hoc.

One flag, recorded because it is a risk the narrative carries rather than a disagreement with it: **s59.1
pre-registered the proposer endpoint and the delivery count before the runs existed.** A pre-registered
endpoint that is measured and then not mentioned is different in kind from a result that simply did not make
the cut, and a referee who finds it later has grounds to distrust the rest. `NARRATIVE.md` §6 therefore
specifies three sentences for the discussion - the endpoint was pre-registered, it came out null at n = 6, the
classical arms delivered more often under the shipping rule, both in full in appendix D1-D2 - and keeps every
number itself in the appendix, as the user asked. None of C1-C5 depends on either result, so the disclosure
costs the paper nothing.

### 60.4 The figure set, built the same day (no simulations)

Four of the five missing figures now exist, and both scripts are written so that a figure cannot drift away
from the table beside it.

**`scripts/dev/figures_leg1.py`** reads the campaign's own 21 `history.jsonl` files and writes two panels.
**F2 `f5_front.png`** is the front against actuator duty: (a) all 485 distinct proposed tunings on the design
screen, split into the 236 that lose a constraint, the 97 that hold it on the set mean and the 113 that also
hold both loads at every wind speed, with 39 more the plant could not run; (b) the ten verified band members
with their fresh-seed regulation, filled where they hold per wind on both seed sets, the one-line row and
stage 1 labelled. Panel (a) uses a **log duty axis and a symmetric-log regulation axis** because the box is
real and half of what a proposer can write is far worse than shipped - the median proposal reads -33 % of
regulation and the worst -17412 %, so a linear window would have hidden 56 % of the cloud rather than showing
that most of the space is bad. **F3 `f5_search.png`** is one panel per proposer, cumulative best feasible
regulation against the simulation index, the first feasible gain marked, censoring named in the title.
The script **re-derives the first feasible gain of every run and exits non-zero** if any of the 21 disagrees
with `docs/tables/rosco_proposer_test.csv`, or if any arm's count and median disagree with s59.4; it found two
of its own bugs that way (a median taken as the upper of the two central values, and the shipped identity
check counted as a proposal), both fixed, and now reports 0 mismatches.

**`scripts/dev/ladder_published.py --figdir`** draws **F5 `f5_stack.png`** (the four published reductions per
rung with paired bootstrap bars) and **F4 `f5_perwind.png`** (the same rungs per wind speed, gate ramp shaded)
*inside the script that writes the table*, from the same joined arrays, so the two cannot disagree. This
replaces regenerating `figures_v5.py`, whose stack carried a fifth panel for the composite and whose first
rung was the hand-scaled baseline s52 voided - regenerating it would have reproduced a figure built on a row
that is not a tuning file. The superseded rows stay in the table, which is where they explain why the
model-based rung used to look load-optimal, and are kept out of the figures. Re-running the generator
reproduced both committed CSVs byte for byte, which is the check that the table on disk is still the table the
code makes.

F4 carries two things the set mean hides and the text needs: the MPC rows lose tower-base fatigue from 18 m/s
up while the PI rows hold +15 to +20 points, and below the gate the gated rung lies exactly on its base.

**What is still blocking the rewrite: F1, the loop diagram**, which no script can make and which has to be
drawn by hand in TikZ. `f5_duty.png` is replaced by F2 and eight v4/F-era figures are retired. Everything
else the main text needs is measured and in `docs/tables/`.
