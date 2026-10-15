# Archive - what is in here and why it is out (2026-10-15)

Nothing here is deleted and nothing here is cited. Each file is kept because the *reason* it went out of date
is itself a result, and because the mechanism chains in them are still the best explanation of why the
current design looks the way it does. **No number from this directory may enter the manuscript.**

| file | what it is | why it is out | what replaced it |
|---|---|---|---|
| `roadmap_2026-08-30.md` | the F-era day-by-day log, §1-§23 | objective (F, tower-DEL priority with constraint tiers), baseline (shipped GSPI only) and aggregation are all superseded | `../roadmap_2026-09-11_metric-set-objective.md` |
| `REPORT_2026-09-01.md` | the F-era consolidated findings F1-F7 | same three reasons; read it for the mechanism chain (critic scale, credit window, reward gating), never for numbers | `../MAIN_LINE.md` |
| `plan_2026-10-12_manuscript-rewrite.md` | the site-by-site edit pass for `main.tex`, with change reasons (A)-(D) | written under v6's claim, which the 2026-10-13 decision replaced, and then under the 2026-10-15 decision that the main text carries the positive results only | `../paper/NARRATIVE.md`; **its site-by-site table is still correct** for reasons (A)-(D) and is the checklist to work through |
| `literature_2026-09-16.md` | the first literature scan | superseded by the 2026-09-23 audit, which verified every entry and recorded a verification status per entry | `../paper/literature_2026-09-23.md` |
| `litreview_schedule_paradigm_2026-09-01.md` | the earliest scan, on the schedule-replay paradigm | predates the objective switch | as above |

## Retracted claims, kept visible

Statements that later evidence reversed are left where they appear rather than edited away. The live list is
in `../MAIN_LINE.md` under *what cannot be claimed*; these are the F-era three, all in `roadmap_2026-08-30.md`:

- "the supervised variants are statistically tied" (§13) - reversed in §21 once both arms shared one wind bank;
- "schedule replay is the lowest-variance method" (§11) - retracted in §14;
- "mono never satisfies the speed constraint" (§22) - restated as a paired ratio, because the tier count
  flips between wind sets.

And the two that the current log retracted:

- **"ROSCO's tower damper is worth about nothing"** (s23) - withdrawn in s52: its gain had been capped at
  0.004, worth 0.25 deg of authority. Searched, it asks about 2.0 deg and buys most of the +25.3 tower DEL.
- **The duty argument against the model-based controller** (s22, s26) - gone in both directions (s53, s56):
  a PI tuning reaches +33 regulation at 1.12x travel, and the learned PI route reaches 2.46x without
  catching the MPC.
