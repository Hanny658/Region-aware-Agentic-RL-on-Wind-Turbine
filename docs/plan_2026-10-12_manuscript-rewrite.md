# What has to change in main.tex, site by site (2026-10-12)

Written before the edit so the pass is mechanical and nothing is half-converted. Three separate reasons for a
change, and they must not be mixed up:

**(A) The composite must leave the reporting.** The user's rule: report the four published reductions, never a
measure we invented, and the symbol $J$ is reserved for the MPC cost function. The composite is still a true
part of the *method* -- it is what the MPC and residual campaigns selected and trained on -- so it stays in the
setup and the appendix as the selection objective under its own name, and leaves every results table and every
sentence that quotes a margin. Nothing is recomputed to do this: the four reductions are already in every table
beside the composite column (`docs/tables/ladder_published_{fresh,heldout}.csv`, roadmap s54).

**(B) The baseline row is void and becomes a band.** It was not a tuning file (roadmap s52) and it was selected
under the composite. The replacement rows are measured (s52, s54) and the band member used in each comparison
has to be named.

**(C) Two statements are now known to be false.** The tower damper is not worth nothing (s52), and the duty
argument against the model-based controller is gone (s53.4).

## Site by site

| line | what it says now | treatment |
|---|---|---|
| 16 | `\newcommand{\J}{J}` | (A) rename the macro; $J$ is the MPC cost function in §3.2 |
| 75--76 | eq:J, the composite | (A) keep as the *selection and training objective* of the campaigns, renamed, with one sentence saying it is never used to report a result |
| 80--82 | properties of eq:J, "how numbers are reported" | (A) rewrite: the four reductions are the report; the composite is what the searches maximised |
| 90 | shipped GSPI, damper "worth $-0.05$ points" | (A) the $-0.05$ is composite points -> give the four reductions of that row (table has them: $-0.3$ / $-0.6$ / $+0.5$ / $+0.2$); (C) the sentence that the tuned baseline turns the damper on is right and gets the s52 number: the damper asks $\approx2.0^\circ$ and buys most of the $+25.3$ tower DEL |
| 92 | the four-parameter search: multipliers on the gain tables, corner capped at 4 rad/s, damper gain to 0.004 | (B) replace with the native parameterisation (`omega_pc`, `zeta_pc`, `F_LPFCornerFreq`, `FA_KI`, `FA_HPFCornerFreq`), the box read off the plant, and the constrained selection (one objective, three constraints at no worse than shipped, resolution of the measurement as the tolerance, constraint domination) |
| 116--128 (tab:ladder) | range-set ladder, composite column, tuned row 15.4 | (A) drop the column; (B) the tuned row is re-measured -- the native tuning has no 42-episode range evaluation, so either run it (free of decisions, 42 episodes) or move the table to the 28 + 28 sets of tab:top |
| 135--143 (tab:top) | fresh-seed ladder, composite column, tuned row 15.9 / 24.5 / 30.7 / 7.1 / 1.4 at 1.12x | (A) drop the column; (B) two baseline rows, both verified and both named: the duty-free member (one knob, 1.004x) and the tower-strong member (s52, 2.02x); the MPC and gated rows are unchanged and already in published terms |
| 143 | "decomposition of the 32.3 points: tuning 15.9 (49 %), model-based $+14.5$ (45 %), learning $+1.9$ (6 %)" | (A) delete: it is arithmetic on the composite. Replace with the per-term ladder of s54, where the second rung is a trade, not an addition: $+23.7$ power, $+25.6$ speed, $-7.9$ tower |
| 149 (fig:stack) | "the four terms of the objective and \J" | (A) regenerate `f5_stack.png` without the composite panel (`scripts/dev/figures_v5.py`) |
| 153 | "the same damper that is worth nothing at its shipped gain" | (C) keep the scoping (shipped gain only) but state what it is worth when its gain is searched: s52 |
| 178, 209 | the aggregation section | (A) the three aggregations are of the four reductions, not of the composite; check every number in §5.2 |
| 252, 284--288, 325, 414 | ablation tables in $\Delta$ composite | (A) these are *selection* results (which arm found what), so they may keep the objective they were selected on, renamed, but each table also needs its four reductions, which `docs/tables/*.csv` already carry |
| 317 | wind-estimate substitution "$-0.09$ points of \J" | (A) restate per term |
| 346--349 (fig:duty) | composite against pitch travel; "the tuned baseline uses 1.14x" | (A) y-axis per term or the regulation term alone; (B)(C) the band spans 1.00x to 2.72x and the duty argument is gone -- the figure becomes the band against duty, which is the honest version of this figure |
| 374--377 | eq:Cw, the per-wind training objective | (A) keep, renamed, as the training objective; it is method, not reporting |
| 392 | "a grid over the corner frequency and a multiplier on the proportional gain, scored by eq:J" | (B) replace with the native search (as line 92) |
| 442 | "fresh-seed \J 32.5" | (A) restate per term |

## Order of the pass

1. Fix (C) first: two sentences, no numbers to re-measure.
2. Then (B): the baseline paragraph and the two baseline rows. **Gated on the band verification now running**
   (the one-knob member is in that batch and is not verified yet).
3. Then (A): the macro, the two table columns, the decomposition line, the two figures, then every sentence
   that quotes a margin in composite points. One pass, no partial state -- the paper must not report two
   different kinds of number.
