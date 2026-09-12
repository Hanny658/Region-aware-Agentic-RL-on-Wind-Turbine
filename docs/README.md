# Documentation index

Files are named by the date the work they describe was completed, so reading them in order follows
the project's progress.

| file | what it is | read it for |
|---|---|---|
| `roadmap_2026-08-30.md` | the day-by-day experiment log, §1–23 | what was tried, in order, including everything that failed and why |
| `REPORT_2026-09-01.md` | the consolidated findings F1–F7 | the verified conclusions, each bound to its constraint tier |
| `litreview_schedule_paradigm_2026-09-01.md` | literature behind the supervision design | why the supervisor is shaped the way it is |
| `RESULTS_2026-09-05_ipc-negative.md` | the individual-pitch chapter, a documented negative | the boundary of the supervision claim: where the same supervisor stops winning, and why |
| `RESULTS_2026-09-10_data-and-caveats.md` | data provenance, run inventory, table rebuild, caveats C1–C12 | before quoting any number in a manuscript |
| `roadmap_2026-09-11_metric-set-objective.md` | roadmap v2: the objective switch to the paper's metric set J, the critic/MPC-scaling findings, the J campaigns and the fairness steps (§1–§10) | the current headline and every J-era table |
| `tables/` | every derived table as CSV | the numbers themselves (`scripts/dev/build_tables.py` regenerates them) |
| `figures/` | R3 trajectory figures | the qualitative comparison at a single above-rated episode |
| `logs/` | campaign logs, including the failed attempts | provenance of specific runs |
| `plant/DISCON.IN` | the ROSCO configuration as actually run | which load-mitigation features were on or off (caveat C4) |

## Where to start

1. `REPORT_2026-09-01.md` — the findings, in the language the manuscript will use.
2. `RESULTS_2026-09-10_data-and-caveats.md` §4 — the caveats. Several conclusions are tier- or
   wind-set-dependent by construction, and two of them changed once a confound was removed; the
   caveat list is what keeps the claims honest.
3. `roadmap_2026-09-11_metric-set-objective.md` §8–§10 — the current (J-era) headline: the fixed MPC, the fair RL comparison, and which agentic lever survives it; `roadmap_2026-08-30.md` §21–§23 for the F-era mechanism.

## A note on retracted claims

Three statements in the earlier record were superseded by later evidence and are marked as such
where they appear: "the supervised variants are statistically tied" (§13 → reversed in §21 once both
arms shared one wind bank), "schedule replay is the lowest-variance method" (§11 → retracted in §14),
and "mono never satisfies the speed constraint" (→ restated in §22 as a paired ratio, because the
tier count flips between wind sets). They are kept visible rather than edited away: the reason each
one changed is itself a result.
