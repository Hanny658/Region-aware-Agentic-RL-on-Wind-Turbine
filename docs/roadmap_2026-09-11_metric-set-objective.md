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
