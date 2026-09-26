# Restoring this project from `G:\wind-n-openfast-backup`

Written 2026-09-26, when the working copies on C: and inside WSL were archived to the external drive and removed.
Everything needed to reproduce or continue the work in `docs/manuscript/main.tex` is in that folder.

## What is in the backup

```
G:\wind-n-openfast-backup\
  repo\                     the full working copy, including .git, and the three things git does not carry:
                              .env                        LLM credentials (never commit this)
                              docs\manuscript\authors.tex author block; without it the manuscript builds anonymous
                              RelatedWorks\               reference PDFs
  wsl\
    wtrl.tar                ~/wtrl          17 GB  environment, ROSCO build, templates, 600 s wind bank, runs
    wtrl600.tar             ~/wtrl600      3.8 GB  paired baselines for the 600 s range protocol
    wtrl_rt.tar             ~/wtrl_rt      293 MB  paired baselines, tuned-ROSCO base, low-turbulence set
    wtrl_rt_range.tar       ~/wtrl_rt_range 409 MB paired baselines, tuned-ROSCO base, range protocol
    wtrl_mpc_range.tar      ~/wtrl_mpc_range 409 MB paired baselines, scheduled-MPC base, range protocol
    MANIFEST.txt            size, entry count and sha256 of each archive
```

`~/wtrl/tmp` was excluded on purpose (transient scratch). Nothing else was.

## What each piece is for

**Irreplaceable, or expensive enough to count as irreplaceable**

- `~/wtrl/wind600/*.bts` (70 fields, 5.0 GB). The 600 s wind bank: seven mean wind speeds, class B, TurbSim seeds
  1--10. Seeds 7--10 were generated after the controllers were frozen and every headline number is evaluated on
  them. Regenerating them would break that property even if the fields came out identical.
- `baselines/openfast/*.npz` in all five roots (~9 GB, 278 files). The paired GSPI and paired base-controller
  rollouts. Every percentage in the paper is a ratio against these, and they keep the raw OpenFAST channels, which
  is what let the W\"ohler-exponent sweep and the start-up-transient check be done without new simulations.
- `~/wtrl/exp/*/` (2.0 GB, 256 run directories). Per run: `config.json`, `ckpt_best.pt` (the only checkpoint any
  reported number uses), `decisions.jsonl`, `llm_transcript.jsonl`, `episodes.csv`, `evals.csv` and the
  `eval_*.json` / `eval_*.csv` the tables are built from.

**Recoverable but tedious**

- `~/wtrl/mamba` (5.8 GB), the micromamba environment, and `~/wtrl/ROSCO`, `rosco_build`, `rosco_install`, `bin`
  (~135 MB), the patched 22-channel ZeroMQ ROSCO. `scripts/wsl/bootstrap.sh` rebuilds all of it, but the archived
  copy is the exact environment the results were produced in, which a fresh conda-forge solve may no longer be.
- `~/wtrl/runs/template_5mw*` (26 MB), the OpenFAST case templates, including `template_5mw_rosco_tuned`, which
  carries the tuned baseline (speed-filter corner 3.42 rad/s, proportional gain x1.13).
- `~/wtrl/wind/*.inp` (192 files, 3.1 MB). The low-turbulence bank's TurbSim input files. The `.bts` fields
  themselves were deleted during an earlier disk-space cleanup; TurbSim is deterministic given an input file, so
  these `.inp` reproduce the same fields. They matter only for the historical low-turbulence results, not for
  anything in the current manuscript.

## Restoring

```bash
# 1. the repository
robocopy G:\wind-n-openfast-backup\repo C:\path\to\Region-aware-Agentic-RL-on-Wind-Turbine /E

# 2. the WSL trees, inside WSL Ubuntu-24.04
cd ~
for r in wtrl wtrl600 wtrl_rt wtrl_rt_range wtrl_mpc_range; do
  tar -xf /mnt/g/wind-n-openfast-backup/wsl/$r.tar
done

# 3. point the runner at wherever the repo now lives, then check nothing is truncated
~/wtrl/run.sh python scripts/dev/integrity_check.py
```

`~/wtrl/run.sh` contains the repo path it was built for; if the repo moved, re-run
`WTRL_SKIP_WIND=1 bash scripts/wsl/bootstrap.sh` from the repo root, which rewrites it and rebuilds anything missing
without touching the wind bank.

To verify an archive before trusting it:

```bash
sha256sum /mnt/g/wind-n-openfast-backup/wsl/wtrl.tar   # compare with MANIFEST.txt
tar -tf   /mnt/g/wind-n-openfast-backup/wsl/wtrl.tar | wc -l
```

## Reproducing the manuscript's numbers without re-running anything

Every table is built by a script in `scripts/dev/` from the archived evaluation files, so with the tars unpacked
these all run in seconds and should print what the paper says:

| table or claim | script |
|---|---|
| controller ladder, range set | `range_table.py` |
| gated layer per seed, held-out and fresh | `gate_table.py` |
| gate-threshold sweep, policy-level test | `gate_sweep.py`, `gate_policy_test.py` |
| aggregation conventions and their intervals | `convention_flip.py` |
| conditional Rayleigh weighting | `lifetime_del.py` |
| W\"ohler-exponent sensitivity | `wohler_sensitivity.py` |
| reward proposer, level and load rule | `range_residual_table.py`, `proposer_test.py` |
| the wind-signal realism check | `windobs_table.py` |
| simulation budget per arm | `budget_table.py` |
| figures | `figures_v5.py` |
