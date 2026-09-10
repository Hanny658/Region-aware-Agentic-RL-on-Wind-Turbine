# Individual pitch control: a documented negative result (2026-09-03 → 09-05)

The individual-pitch (dq cyclic) experiments of this project, kept as a documented negative. They
matter because they bound the supervision claim of `roadmap_2026-08-30.md` §21–§22: on the
collective channel the LLM proposer beats verified random search (+5.5 F, 7/7 seeds, exact
p = 0.0156, replicated on a second wind set); on the cyclic channel the *same* supervisor, prompt,
knobs and guardrail loses to fixed weights three times in a row. Reporting both is what makes the
claim falsifiable.

Six mechanism variants, 24 OpenFAST runs. All numbers are blade-root out-of-plane DEL reduction [%]
vs the seed-paired GSPI baseline on held-out wind (S3–S6, best checkpoint, blade objective:
`--load_signal M_oop --fitness_target blade`). The raw run artifacts of these arms were not carried
over when the machine changed; the numbers below are the record.

## 1. The channel is physically valuable — measured before any RL

`envs/coleman.py` implements the ROSCO-exact Coleman (multi-blade) transform. The residual adds an
R3-gated dq amplitude pair (±1°/axis, per-axis second-order dampers, per-blade ZMQ offsets, Fortran
unchanged); the observation gains EMA(0.5 s) of (M_d, M_q) and the action penalty gains
(|θ_dq|/κ_ipc)².

Two open-loop probes at 15 m/s, before any learning:

| probe | blade-1 DEL |
|---|---|
| static tilt θ_d = +1° | **−14 %** |
| hand-tuned I-controller on the dq moments | **−22 %** |
| same I-controller, wrong sign | +37 % |

So the channel is worth −14…−22 %, and a classical controller collects it.

## 2. Fixed-weight RL does not use it (§ IPC part 1)

`spec + guard`, λ_L = 1, 5 seeds, seed-paired:

| arm | per seed | mean ± std | U15 only | strict |
|---|---|---|---|---|
| +dq residual (±1°) | 9.0, 8.4, 10.1, 9.9, 9.1 | 9.32 ± 0.69 | 7.5 ± 1.5 | 5/5 |
| collective only | 10.0, 11.1, 10.2, 5.7, 9.5 | 9.28 ± 2.09 | 6.2 ± 2.5 | 5/5 |

A total tie (paired p = 0.98). The diagnosis is in the policy, not the score: the trained agent uses
|θ_d| ≈ 0.03°, |θ_q| ≈ 0.04° — **3–4 % of the authority it was given** — even though a constant +1°
tilt is worth −14 %. With λ_L = 1 the R3 reward is dominated by the speed term and 300 episodes of
zero-mean PPO exploration never pull the dq bias off zero.

Side finding worth keeping: under a *blade* objective, collective pitch alone reaches ~10 % DEL
reduction. The "±3–8 % ceiling" quoted in F2 was a tower-objective by-product number, not the
ceiling of collective pitch for a blade-trained policy.

## 3. The λ-curriculum hypothesis, tested and falsified (§ IPC part 2)

If the bottleneck were the reward weight, the supervisor's `lambda_load_R3` knob is exactly the
lever. `llm_fork` (K = 3, fork-verified) on top of the dq channel, 3 seeds, seed-paired:

| seed | + LLM supervision | + fixed weights | collective only | supervisor trajectory |
|---|---|---|---|---|
| s0 | 10.79 | 9.03 | 9.97 | textbook λ_R3 ramp 1→3→4.5→5.5→6.5, w_speed 20→125 |
| s1 | 5.05 | 8.44 | 11.09 | λ_R2-heavy route (up to 8), oscillating |
| s2 | 7.78 | 10.13 | 10.16 | mild λ_R3 → 2.6 |
| mean | **7.9 ± 2.9** | 9.2 ± 0.9 | 10.4 ± 0.6 | all strict everywhere |

**Negative, and the failure is informative.** Probing the best-supervised policy (s0, λ_R3 = 6.5)
gives |θ_d| ≈ 0.033°, |θ_q| ≈ 0.026° — *identical* to the fixed-λ arm. The reward landscape at
λ_R3 = 6.5 clearly pays for the coherent bias; the policy still never finds it. s0's +10.8 came from
the supervisor improving the *collective* behaviour, not from the cyclic channel.

**The bottleneck is discovery and credit assignment, not reward weighting.** Turning the weights
harder cannot help a policy that never visits the behaviour.

**Quantified afterwards (2026-09-10).** `scripts/dev/critic_health.py` showed that the R3 critic in
every run of this project collapses to a constant (its value target is O(1e4) while the critic
starts at xavier gain 0.1 under grad-norm clipping, so it saturates instead of fitting). With a
constant baseline the GAE advantage reduces to a discounted reward window of 1/(1 - gamma*lambda) =
**0.46 s**, while one rotor revolution at rated speed is **4.96 s**. The learning signal is therefore
an order of magnitude shorter than the physical effect it must credit: a cyclic pitch bias pays off
over a full revolution and is structurally invisible to it. That is the concrete mechanism behind
"credit assignment" in the paragraph above — and it means the two per-step arms (§2 and §3 here)
are **confounded**: they show that RL *in this credit-assignment configuration* does not use the
channel, not that RL cannot. The load-bearing negative is §4-§5 below, where the design fixes
exactly this.

## 4. Correct macro credit assignment: better, but not the fix (§ IPC part 3)

Following Coquelet et al. (2022), who hold the cyclic action for one full rotor rotation per
training decision precisely so its load effect becomes observable, `--ipc_hold <s>` adds a separate
slow PPO learner (act_dim 2, γ_macro = γ^K, own minimum batch) that decides (θ_d, θ_q) once per hold
window; the regional agents stay per-step. Macro reward = discounted window sum / K.

This design happens to remove **both** problems identified above. Its credit window is
1/(1 - γ_macro·λ_macro) = 1.5 macro steps = **7.7 s**, i.e. longer than one revolution rather than
ten times shorter; and its value target is r/(1 - γ_macro) ≈ **25**, small enough for the critic to
reach, so the saturation pathology that flattens the per-step R3 critic does not apply to it. The
arms below are therefore the ones on which the negative result rests.

3 seeds each, blade objective, same seeds and wind as above:

| arm | per seed | mean ± std | strict | θ utilisation (probe, U15) |
|---|---|---|---|---|
| rotation-held + fixed weights | 10.1, 11.2, 12.7 | **11.3 ± 1.3** | 3/3 | 2–4 % of 1° |
| rotation-held + LLM supervision | 11.3, 4.9, 9.9 | 8.7 ± 3.4 | 2/3 | 8 % (s0, at 2°) / 1–2 % |
| collective only (control) | 10.0, 11.1, 10.2 | 10.4 ± 0.6 | 3/3 | — |
| per-step dq (control) | 9.0, 8.4, 10.1 | 9.2 ± 0.9 | 3/3 | 3–4 % |

1. **Directionally yes, mechanistically no.** Rotation-holding makes the cyclic arm ≥ collective in
   every seed for the first time (paired +0.93 pp, p = 0.37 at n = 3) and beats per-step by 2.1 pp —
   but the channel is *still* 2–4 % utilised, so the gain cannot be attributed to 1P cancellation.
2. **Second LLM non-win**: 8.7 ± 3.4 vs 11.3 ± 1.3 (p = 0.35), with one tier slip. The attributable
   positive: on s0 the supervisor immediately raised the new `ipc_max` knob to 2° and that policy
   reached 8 % utilisation (0.17°) and the arm's best held-out (+11.3). On s1 it repeated its known
   pathology (λ_R3 = 18, w_speed = 400) and scored +4.9.
3. Rotation-held runs show mid/late-run collapse-and-rollback churn — small macro batches give
   high-variance slow-learner updates. Best and last checkpoints both survive it.

## 5. A clean-wind curriculum: the last cheap lever (§ IPC part 4)

Coquelet et al. train on laminar wind, where the 1P signal is clean. First 150 episodes on a
freshly generated TI 2 % bank (rollback disabled during the clean phase), then TI 8 %; supervisor
fitness and held-out evaluation unchanged. Rotation-held throughout, 3 seeds:

| arm | per seed | mean ± std | θ utilisation |
|---|---|---|---|
| curriculum + fixed weights | 11.6, 11.6, 9.9 | 11.0 ± 1.0 | 3–11 % |
| curriculum + LLM supervision | 8.8, 6.4, 7.7 | **7.6 ± 1.2** | 2–10 % |
| (ref) no curriculum + fixed weights | 10.1, 11.2, 12.7 | 11.3 ± 1.3 | 2–4 % |
| (ref) collective only | 10.0, 11.1, 10.2 | 10.4 ± 0.6 | — |

1. **The curriculum does not add held-out performance** (11.0 vs 11.3, p = 0.84). It does nudge the
   channel usage up (best single run 0.11–0.19°, 10–11 % of authority) — the clean-wind phase does
   teach *some* cyclic pitch — but the effect neither survives the turbulent phase strongly nor
   converts into held-out DEL.
2. **Third LLM non-win**, and the largest: 7.6 ± 1.2 vs 11.0 ± 1.0, paired p = 0.063. On seed 1 the
   supervisor entered the same extreme attractor a third time (λ_R3 → 20, w_speed → 360, and this
   time also strangling `ipc_max` to 0.008).

## 6. Verdict

**The channel is physically worth −14…−22 % and RL never took more than 11 % of it**, across
per-step and rotation-held credit assignment, with and without a clean-wind curriculum, with and
without LLM supervision — six mechanism variants, 24 runs. The single durable positive is that
rotation-held + fixed weights beats collective-only in every seed tested, at about +1 pp.

Scope of that statement, after the 2026-09-10 critic finding: the **per-step** arms are confounded
(their advantage window was 11× shorter than a rotor revolution), so they support only the weaker
reading "not under this credit assignment". The **rotation-held** arms are not confounded — their
macro learner has both a 7.7 s credit window and a reachable value target — and they are what makes
the negative result stand: correct macro credit assignment, on its own, still left the channel
2-4 % utilised.

**The supervision boundary.** Three consecutive campaigns in which the same fork-verified LLM
supervisor lost to fixed weights:

| campaign | LLM-supervised | fixed weights | seeds |
|---|---|---|---|
| per-step dq | 7.9 ± 2.9 | 9.2 ± 0.9 | 3 |
| rotation-held dq | 8.7 ± 3.4 | 11.3 ± 1.3 (p = 0.35) | 3 |
| rotation-held + low-TI curriculum | 7.6 ± 1.2 | 11.0 ± 1.0 (p = 0.063) | 3 |

with a **reproducible failure attractor** — λ_load_R3 driven to 18–20 and w_speed to 360–400 while
the channel's own authority is strangled. Fork verification cannot see it: every individual step
stays locally acceptable while the trajectory drifts somewhere globally poor. The same attractor
appeared once on the collective axis (`n1_llmfork_s0`, the only fork-supervised run there that lost
the strict tier, roadmap §21).

**Why the two axes differ, in one sentence.** Supervision that tunes reward weights helps when the
reward weighting *is* the binding constraint — the λ curriculum on the collective channel — and does
nothing when the binding constraint is exploration, because no weighting makes an unvisited
behaviour attractive.

Unlocking the cyclic channel is left as future work.

## 7. Code paths

The implementation stays in this tree and is inert by default: `envs/coleman.py`,
`--ipc_max 0` (channel off), `--ipc_hold 0` (per-step if the channel is on),
`--curriculum_ti 0` (no clean-wind phase). Migrated run configs read these keys, so they must not
be removed. The collective-pitch control arm of §2 above (`ipc_off_*`, `ipc_max = 0`) is the
blade-objective `spec + guard` run set used elsewhere in the results.
