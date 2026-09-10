# Data, tables and reporting caveats (state of 2026-09-10)

Everything a number in this repository rests on: which runs exist, which tables are derived from
them, how to rebuild those tables, and the reporting caveats that must survive into the manuscript.

## 1. Where the data lives

| what | where | size |
|---|---|---|
| run artifacts (`config.json summary.json episodes.csv evals.csv decisions.jsonl eval_*.json ckpt_best.pt ckpt_last.pt`, LLM arms also `llm_transcript.jsonl`) | WSL `~/wtrl/exp/<run>/` | 37 MB |
| wind bank `U{8,12.5,15}_TI{2,8,14,22}_S*.bts`, `U18_TI8_S{3,4}` | WSL `~/wtrl/wind/` | 1.4 GB |
| paired GSPI baselines (the denominator of every reported number) | WSL `~/wtrl/baselines/openfast/` | 4.1 GB |
| distilled knob curriculum used by the replay arms | `configs/schedules/` (in git) | 1 KB |
| derived tables | `docs/tables/` (in git) | 150 KB |
| campaign logs, including the failures | `docs/logs/` (in git) | 105 KB |
| the as-run ROSCO configuration | `docs/plant/DISCON.IN` (in git) | 20 KB |

A second copy of the wind bank, the baselines and the run artifacts is kept outside the repository
at `../wtrl-data/` (5.5 GB). **Never regenerate the wind bank**: new TurbSim realisations are
different wind, so every historical number stops being seed-paired — this has happened once and it
changed a conclusion (caveat C1).

## 2. Rebuilding the tables

```bash
~/wtrl/run.sh python scripts/dev/build_tables.py       # standard library only
```

reads `~/wtrl/exp` (override with `WTRL_EXP`) and writes `docs/tables/`:

| table | content |
|---|---|
| `all_evals.csv` | one row per (run, evaluation tag) — the raw material for everything else |
| `table_wind_sets.csv` | every arm on both held-out wind sets (the main results table) |
| `table_paired_stats.csv` | every paired comparison with its exact-permutation floor (caveat C6) |
| `table_heldout_main.csv`, `table_heldout2_s78910.csv` | per-arm aggregates, first and second wind set |
| `table_paper_metrics.csv` | the baseline paper's four metrics |
| `table_robustness.csv` | stress classes (TI14 / TI22 / U18) |
| `table_torque.csv` | the R2 torque-residual negative |
| `table_mpc.csv` | the full LPV-MPC evaluation set |
| `knob_trajectories.csv` | supervisor knob paths per decision (the curriculum figure) |

`table_night1_SUPERSEDED_by_S21.csv` is kept only as provenance of a retracted table — do not quote it.

## 3. Run inventory

| runs | arm | objective | note |
|---|---|---|---|
| `tq_off_s0-4` | **spec + guard**, the fixed-weight main line | tower | the name is historical: it is the control arm of the torque campaign |
| `n1_mono_s0-4` | mono (monolithic control) | tower | seed-paired with `tq_off` |
| `n1_llmfork_s0-6` | spec + llm_fork | tower | 7 seeds, one wind bank |
| `n1_randfork_s0-6` | spec + random_fork | tower | 7 seeds, paired with the above |
| `n1_llmsingle_s0-4` | single-proposal LLM, no fork, no dry run | tower | isolates verification |
| `sched2_ep_s0-4` | curriculum replay, episode-indexed | tower | curriculum in `configs/schedules/` |
| `sched2_comp_s0-4` | curriculum replay, competence-indexed | tower | contaminated, see C7 |
| `sched3_comp_s0-4` | the same, with the fixed competence gate | tower | the clean verdict |
| `tq_on3_s0-4` | spec + guard + R2 torque residual | tower | the negative result's main evidence |
| `tq_on_s0`, `tq_on2k_s0`, `tq_on2kg_s0`, `tq_on2kk_s0` | torque debugging sequence | tower | three false trails and one real bug |
| `toy_dtau_*` | toy torque sweep | blade | 1-DOF twin |
| `ipc_off_s0-4` | spec + guard, blade objective | blade | also the collective-only control of the cyclic-pitch chapter |
| `gspi_td` | GSPI + ROSCO tower damper | tower | gain sweep, held-out, stress classes, identity check |
| `mpc`, `mpc600`, `mpc600s` | LPV-MPC | tower | tuning grid, both wind sets, 600 s window, window/realisation decomposition |
| `_migration_check` | infrastructure | — | GSPI identity checks; provenance of the 2026-09-09 baseline restore |

## 4. Reporting caveats

- **C1 — resolved.** Two wind banks were once mixed in one paired comparison. Every arm has since
  been re-run on the single canonical bank. **That re-run changed a conclusion** (roadmap §21):
  the earlier "the supervised variants are statistically tied" was an artefact of the mixed pairing.
  Treat any historical number computed before 2026-09-09 as belonging to a different bank.
- **C2 — resolved.** The stress sweep now runs on the tower-objective arms (roadmap §17); the earlier
  table scored blade-objective runs, whose tower column was a by-product, and has been removed.
- **C3.** For any controller that alters ROSCO's *own* pitch command (the LPV-MPC, the tower damper)
  the region label must be recomputed from wind speed on **both** sides — `--relabel_wind`, tags
  `heldoutW_*` / `heldout2W_*`. The native-label MPC files are pre-fix and must not be quoted.
- **C4 — answered.** The GSPI baseline runs with ROSCO's load-mitigation features off. Turning the
  fore-aft tower damper on and sweeping its gain under the same model-selection rule gives **no
  strict configuration at any gain** (roadmap §18): it sits on the same speed-for-load trade-off as
  the MPC.
- **C5 — verified.** Re-scoring on 600 s episodes moves every RL arm by less than ±1.5 F and changes
  no tier (roadmap §19). The 130 s window does not bias the comparison. The MPC is the exception:
  its advantage is window-dependent.
- **C6 — reporting rule.** The exact sign-flip permutation test cannot go below 2/2ⁿ (n = 5 → 0.0625,
  n = 7 → 0.0156). `table_paired_stats.csv` prints `exact_perm_p` next to `exact_floor`; when they
  are equal the result must be written as "≤ this value at n = …". Always report the parametric and
  the exact test together.
- **C7.** The competence-indexed replay arm `sched2_comp_*` is contaminated: after a guardrail
  rollback the trainer masks F with the historical best and the competence gate read that masked
  value, firing aggressive curriculum entries into a crash (5 of 24 entries; s2 alone 4 of 5 —
  `scripts/dev/gate_audit.py` reproduces the count). The gate was fixed on 2026-09-02 to read
  `F_measured`; `sched3_comp_*` is the clean re-run and is the arm to quote.
- **C8.** `ckpt_last` is frequently degraded — every number uses `ckpt_best`, and each run keeps its
  `eval_*_ckpt_last.json` as the evidence.
- **C9.** The replayed curriculum was distilled from `n1_llmfork_s3`. An earlier claim that schedule
  replay was the lowest-variance method has been retracted: variance belongs to the curriculum's
  content, not to the replay paradigm.
- **C10.** `tq_off_*` *is* the main guard arm, despite the name.
- **C11.** **Tier counts are not a robust statistic.** `mono` is 0/5 strict on the first held-out set
  and 4/5 on the second, because those realisations lower every controller's speed-std ratio by
  about 0.05 and mono sits within 2 % of the boundary. Report the **paired continuous quantity**
  (mono spends +0.05 speed-std relative to spec, 10/10 seeds across both sets, p = 0.0202 / 0.0066)
  and let the tier follow.
- **C12.** The LLM-supervised arms depend on an external model (`gpt-5.6-luna`) called at two
  different dates; model drift cannot be verified and must be stated.

## 5. Environment

| component | version |
|---|---|
| OpenFAST | 4.2.1 (conda-forge) |
| ROSCO | 2.10.5 + `controllers/rosco_patch/` (22-channel ZMQ, torque offset, WSE sees applied torque) |
| Python | 3.11; torch 2.13.0; numpy 2.4.6; gymnasium 1.2.3; sb3 2.9.0; fatpack 0.7.8; openai 3.8.0; osqp 1.1.3 |
| plant | NREL 5 MW onshore; constant torque (`VS_ConstPower=0`); `PS_Mode=1`; 10 ms control step |
| cost | 300-episode run ≈ 66–80 min at 8 workers, two lanes in parallel |

Re-bootstrap on a fresh machine: `WTRL_SKIP_WIND=1 bash scripts/wsl/bootstrap.sh` from the repo root
inside WSL, then copy the wind bank and baselines into `~/wtrl/{wind,baselines/openfast}` rather than
regenerating them. `.env` (`LLM_BASE_URL / LLM_API_KEY / LLM_MODEL`) is required for the
LLM-supervised arms and is never committed.
