# Reporting conventions in the published load / control literature (2026-09-23)

Scope: what the field actually prints when it reports fatigue, seeds, episode length, Wöhler
exponents, actuator duty and statistics — collected to justify (or correct) our own protocol
(12–24 m/s in 2 m/s steps, IEC class B, 600 s, TurbSim seeds, tower-base FA DEL m = 4, blade-root
OoP DEL m = 10, energy constraint, pitch travel as actuator cost).

Verification discipline used here:
- **VERIFIED** — I fetched the page and read the quoted text/number on it.
- **VERIFIED (bib only)** — the page was fetched and identifies the work, but the specific number
  was not on the part I read.
- **UNVERIFIED** — seen only in a search-result summary, or the page could not be parsed. **No
  number is quoted from these.**

Environment note (affects what could be checked): `www.nrel.gov` and `docs.nrel.gov` did not
resolve from this machine, so MLife could not be read. Official / mirrored **IEC 61400-1 PDFs and
most Copernicus *preprint* PDFs returned unparseable binary**; Copernicus and arXiv **HTML article
pages** fetch cleanly and are the practical source. MDPI returned HTTP 403.

---

## 1. Aggregating fatigue over a wind distribution

### Short-term DEL, as printed
- **VERIFIED** `https://wes.copernicus.org/articles/9/1885/2024/` (probabilistic DEL surrogates,
  mixture density networks): "DEL_ST := (Σ n_i S_i^m / n_ref)^(1/m)", with **n_ref = 600** for
  1 Hz DELs over 10 min. This is exactly the canonical form.
- **VERIFIED** `https://wes.copernicus.org/articles/10/2903/2025/` Sect. 3.3.1, Eq. (5):
  "DEL = [Σ_i n_i L_i^R m / n_eq]^(1/m)".
- **VERIFIED** `https://wes.copernicus.org/articles/10/2005/2025/` Sect. 3.1, Eq. (9):
  "DEL = [Σ n_i R_i^m / n_eq]^(1/m)".

### Palmgren–Miner, as printed
- **VERIFIED** `https://arxiv.org/html/1411.3925` Sect. 3.1: S-N curve "s^k N = K" (Eq. 3.1),
  damage "D(T) = Σ_{i=1}^{N(T)} 1/N_i" (Eq. 3.2) → "D(T) = Σ s_i^k / K" (Eq. 3.3); the paper uses
  **k = 4, K = 6.25e37**, stating k = 4 is "adequate for steel structures".
- **VERIFIED** `https://arxiv.org/html/2601.01657` (FLOAT, floating tower fatigue design) Eq. (2):
  "D_j = Σ_i n_i(Δσ_i)/N_i(Δσ_i)"; lifetime Eq. (8): "D_t ≈ LT · Σ_j w_j · D(U_j, H_s,j, T_p,j,
  M_ww,j)" — lifetime damage = lifetime × probability-weighted sum of per-condition damage.
- **VERIFIED** `https://wes.copernicus.org/articles/7/1171/2022/` Sect. 1: "The Palmgren–Miner rule
  is the standard approach followed in the design of wind turbines by which it is ensured that the
  linear damage sum over an intended lifetime is lower than unity after considering required safety
  margins." Design lifetime quoted as **25 years**. Its Eq. (1) weights the per-wind-speed DEL by
  N_v, the annual hours at that mean wind speed. (The equation came back partly garbled through the
  markdown conversion — re-read the page before transcribing it literally.)

### Who uses the Rayleigh weighting
- **VERIFIED** `https://wes.copernicus.org/articles/9/799/2024/` (sensitivity of fatigue reliability:
  design turbulence and the Wöhler exponent) — the cleanest printed lifetime aggregation:
  "DEL_lifetime^m = Σ_{V_bin} Σ_{T_bin} [(DEL_bin)^m · P(T_bin|V_bin) · P(V_bin)]", with **a Rayleigh
  distribution for the mean wind speed**, IEC **class 1A: V_ref = 50 m/s, V_ave = 0.2·V_ref = 10 m/s**,
  **design lifetime 20 years**, load cases **IEC DLC 1.2**. Note it also integrates a turbulence bin
  conditional on wind speed, which the plain IEC Rayleigh-only scheme does not.
- **VERIFIED** `https://wes.copernicus.org/articles/8/1299/2023/` (Guo & Schlipf — confirmation of a
  source we already knew): site-specific **Weibull**, not Rayleigh. Eq. (10):
  "DEL = (Σ_i Σ_j A_eq,ij^m · N_10min/N_ref · P_Uhub,i · P_sta,j)^(1/m)", **Weibull shape 2.02, scale
  9.41 m/s at 10 m**, three atmospheric stability classes with FINO1 probabilities, IEC 61400-1 (2019)
  class C. Short-term form Eq. (9): "A_eq = (Σ A_i^m n_i)^(1/m)". They also define an
  "extended lifetime" ratio, **EL = 20·(DEL_i/DEL_j)^(-m-1)** — a way to turn a % DEL change into
  equivalent lifetime years.
- **VERIFIED** `https://wes.copernicus.org/articles/10/2865/2025/` (floating LDEL surrogate):
  25-year lifetime DEL, Eq. (11), **n_eq = 1e6 reference cycles**, n_L = number of 10-min periods in
  L = 25 years; deliberately uses **ERA5 hourly conditions instead of a fitted Weibull/Rayleigh** — a
  live alternative convention.
- **UNVERIFIED** NREL MLife Theory Manual (`https://www.nrel.gov/docs/libraries/wind-docs/mlife-theory.pdf`)
  — the canonical printed lifetime-DEL formula (follows IEC 61400-1 ed. 3 Annex G; a Weibull shape
  factor of 2 gives the Rayleigh case). Domain unreachable here; quote nothing from it until read.

**For our paper.** A defensible lifetime aggregation is
`LDEL = [Σ_j p_j (DEL_j)^m]^(1/m)` with p_j the Rayleigh bin probability for the IEC class
(V_ave = 0.2 V_ref), stated together with the design lifetime (20 or 25 yr) and n_eq (600 for
1 Hz / 10 min, or 1e6 for lifetime). Our 12–24 m/s span covers only part of the Rayleigh mass, so we
must either renormalise a truncated weighting or keep reporting per-bin DELs — the literature does
both, and several of the papers above report per-wind-speed DELs only.

---

## 2. Number of turbulent seeds, and what IEC requires

**The IEC ed. 4 clause text itself is UNVERIFIED** — no fetchable copy of IEC 61400-1:2019 could be
parsed (official preview, an iteh.ai sample and a university mirror all returned binary). Do **not**
write "IEC 61400-1 ed. 4 requires six seeds for DLC 1.2" as a quotation from the standard.

What *is* verified is that peer-reviewed work attributes six to the standard and uses it:
- **VERIFIED** `https://wes.copernicus.org/articles/9/799/2024/`: "we use a sample size of six
  (**recommended number of samples by the IEC 61400-1**)"; simulations "based on the IEC standard
  design load case (DLC) 1.2"; their underlying database holds 200 realizations per wind/turbulence bin.
- **VERIFIED** `https://wes.copernicus.org/articles/4/397/2019/` (non-intrusive UQ in aeroservoelastic
  simulation): "**six seeds were used to limit the computational cost of the MC analysis, following
  accepted international standards (IEC61400-1, 2005)**".
- **VERIFIED** `https://wes.copernicus.org/articles/6/1401/2021/` Sect. 3 — a fully specified DLC 1.2
  matrix: "18 10 min load simulations (three yaw directions: 0° ± 10°, and **six turbulent wind
  seeds**) for each mean wind speed ranging from **4 to 26 m/s in the interval of 2 m/s**, which
  results in a total of **216 simulations**"; "sampling frequency of 50 Hz".
- **VERIFIED** `https://wes.copernicus.org/articles/9/1791/2024/` Sect. 3.2: "**six turbulence seeds**
  were generated to improve statistical convergence" per 10-min bin.
- **VERIFIED** `https://arxiv.org/html/2601.01657`: six independent realizations per wind speed bin,
  22 wind bins, IEC Class A.
- **VERIFIED** `https://ar5iv.labs.arxiv.org/html/2110.14169` (controller comparison, floating):
  "**6 random turbulence seeds at each whole-numbered wind speed**" 12–24 m/s, "**78 simulation cases
  for each controller**", NTM "as outlined in design load case 1.1".
- **VERIFIED** `https://wes.copernicus.org/articles/8/1299/2023/` (Guo & Schlipf, confirming):
  "six independent simulations are performed that have different random seed numbers".

Papers that use **more** than six:
- **VERIFIED** `https://wes.copernicus.org/articles/8/149/2023/` Sect. 5.1.3 (lidar-assisted control
  under various turbulence characteristics) — the closest match to our own protocol: "hub height mean
  wind speed **from 12 to 24 m/s with a step of 2 m/s**", "**12 different random seed numbers**" per
  stability class.
- **VERIFIED** `https://wes.copernicus.org/articles/10/2005/2025/`: 10 seeds (single wind speed, 15 m/s).
- **VERIFIED** `https://wes.copernicus.org/articles/10/2865/2025/`: 44 seeds per test condition.
- **VERIFIED** `https://wes.copernicus.org/articles/9/1885/2024/`: 300 seeds per test condition for
  reference distributions.

**For our paper.** 6 seeds × 7 wind speeds = 42 episodes sits at the de-facto floor and below the
12 seeds of the closest-matching control study. Phrase it as "six seeds per wind speed, the number
commonly used for DLC 1.2 (refs)", not as a quotation of the standard.

---

## 3. Episode length and discarded transient

All rows **VERIFIED**. The convention is unambiguous: **simulate transient + 600 s and analyse the
last 600 s**; the discarded transient is 60–400 s, most often 100 s or 300 s.

| Source (URL) | Total | Discarded | Analysed |
|---|---|---|---|
| `https://wes.copernicus.org/articles/9/799/2024/` | 700 s | first **100 s** ("recognized as transient time and is omitted") | 600 s |
| `https://wes.copernicus.org/articles/8/1299/2023/` (Guo & Schlipf, confirm) | 700 s | "the initial **100 s** results are ignored" | 600 s |
| `https://wes.copernicus.org/articles/9/1885/2024/` | 900 s | "the first **300 s** is discarded to exclude the initial transient" | 600 s |
| `https://wes.copernicus.org/articles/10/2865/2025/` | 300 s ramped init + 600 s | **300 s** | 600 s |
| `https://arxiv.org/html/2601.01657` (FLOAT) | 1000 s | first **400 s** | 600 s ("yielding 10 minutes effective data") |
| `https://ar5iv.labs.arxiv.org/html/2110.14169` | 800 s | "the first **200 seconds** of transient settling discarded" | 600 s |
| `https://wes.copernicus.org/articles/9/1791/2024/` Sect. 3.3 | 10-min bins | "the first **minute** of each 10 min bin was discarded to remove transients" | ~9 min |
| `https://wes.copernicus.org/articles/8/149/2023/` Sect. 5.1.3 | **31 min** | "the initial **60 s** time series, which contains the initialization" | 30 min |
| `https://wes.copernicus.org/articles/10/2005/2025/` Sect. 3.3 | **2100 s** | first **300 s** | 1800 s |
| `https://wes.copernicus.org/articles/10/2903/2025/` | **4600 s** | first 1000 s | final **3600 s** |
| `https://wes.copernicus.org/articles/7/523/2022/` Sect. 4.1 | 1000 s (dt = 0.01 s) | turbulent cases analysed over **750–1000 s** | 250 s |

On the 600-s norm itself: **VERIFIED** `https://wes.copernicus.org/articles/8/575/2023/`
(prognostics-based adaptive lifetime control) — "**Following the IEC 61400-1 recommendation for
fatigue load evaluation, a 600 s stochastic wind profile generated using TurbSim software is
used**", with "six profiles with mean wind speeds of 18 and 14 m/s, each having **three seeds**".

**For our paper.** If our 600 s episodes *include* the startup transient we are out of step with
every row above. Either simulate 700–900 s and score the last 600 s, or state in the protocol
exactly how many seconds are discarded before the metrics are formed.

---

## 4. Wöhler / S-N exponents

- **Tower, m = 4** — **VERIFIED** `https://wes.copernicus.org/articles/10/2903/2025/`: blades m = 10,
  tower **m = 4**, "a compromise based on an investigation of tower stress cycles ... showing stress
  cycles distributed on both sides of the transition point of the bi-linear S–N curve".
  **VERIFIED** `https://wes.copernicus.org/articles/8/1299/2023/`: tower and shaft **m = 4**, blades
  **m = 10**, mooring chain m = 3.
  **VERIFIED** `https://arxiv.org/html/2601.01657` Sect. 3.7.1: DNV-RP-C203 Type E curve, "m set by
  default to **4**, the average slope of the two S-N curve regimes" (slopes **3 and 5**, transition at
  1e7 cycles; thickness exponent k = 0.20, t_ref = 25 mm).
  **VERIFIED** `https://arxiv.org/html/1411.3925` Sect. 3.1: k = 4, "adequate for steel structures".
- **Tower, m = 3** — **VERIFIED** `https://wes.copernicus.org/articles/9/799/2024/`: "equal to **3** in
  the case of steel components". **VERIFIED** `https://wes.copernicus.org/articles/8/575/2023/`:
  "Wöhler exponent (typically **3 for steel materials like the tower and 10 for composites like the
  blade**)".
- **Tower, m = 3.5** — **VERIFIED** `https://wes.copernicus.org/articles/9/1885/2024/` and
  `https://wes.copernicus.org/articles/10/2865/2025/`: "tower m = **3.5**, blade flapwise m = **10**,
  blade edgewise m = **8**".
- **Blade, m = 10 (composite)** — **VERIFIED** `https://wes.copernicus.org/articles/10/2005/2025/`
  Sect. 3.1 ("m = 10 for composites"), plus all of the above.
- **Sensitivity ranges actually studied** — **VERIFIED** `https://wes.copernicus.org/articles/9/799/2024/`
  examined **m = 8, 10, 12 for blades and m = 3, 4, 5 for the tower base**. This is the citation for
  our m = 4 / m = 10 being a mid-range convention, and the template for a cheap robustness line.
- Other components for context — **VERIFIED** `https://wes.copernicus.org/articles/6/1401/2021/`:
  **m = 6** for the cast-iron main shaft, mean-stress correction M = 0.19.

**For our paper.** m = 4 (tower FA) and m = 10 (blade OoP) are squarely conventional; 3 and 3.5 are
the competing tower choices. An m ∈ {3, 4, 5} sensitivity row on the range set would pre-empt a
reviewer.

---

## 5. Actuator duty reported alongside load claims — yes, routinely

- **VERIFIED** `https://wes.copernicus.org/articles/10/2005/2025/` Sect. 3.1, **Eq. (10)**:
  "ADC = (1/T) ∫_0^T |u̇(t)|/u̇_max dt", with **u̇_max = 2° s⁻¹**, the maximum pitch rate of the
  IEA 15 MW turbine. The paper's framing is explicitly the trade-off between actuation effort and
  blade fatigue reduction, reported jointly — the same front we report in roadmap s26.
- **VERIFIED** `https://wes.copernicus.org/articles/7/523/2022/` **Eqs. (38)–(39)**:
  "ADC = (1/T) ∫_0^T β̇(t)/β_max dt", with **β_max = ±8° s⁻¹** for the 10 MW turbine.
  → ADC is the time-averaged pitch rate **normalised by the actuator rate limit** (dimensionless), not
  raw pitch travel in degrees, and the normalisation is turbine-specific (2 vs 8 °/s), so ADC values
  are not comparable across papers unless the limit is printed.
- **VERIFIED** `https://wes.copernicus.org/articles/8/575/2023/` reports "the **average total pitch
  travel** marginally increases by **0.13 %**" (and 0.5 % near rated) next to its fatigue claims —
  i.e. pitch travel as a % change against the baseline controller, which is our convention.
- **VERIFIED** `https://wes.copernicus.org/articles/8/1299/2023/` (Guo & Schlipf): reports blade pitch
  *rate* statistics rather than an ADC, and flags that their optimally tuned feedback "gives higher
  blade pitch rates, which are even doubled for very high mean wind speed ranges".
- **UNVERIFIED** (search summaries only; no numbers taken): the closed-form "pitch travel
  PT = ∫|dθ/dt| dt" definition, and a "50 % DEL reduction at 16.4 % of the actuator effort" headline.

**For our paper.** Our "pitch travel ratio vs GSPI" (1.40–2.9x rows) is in line with practice.
Adding ADC with the rate limit stated would make the actuator cost directly comparable to the two
definitions above.

---

## 6. Statistical practice — the field's bar is low

- **VERIFIED** `https://wes.copernicus.org/articles/4/397/2019/`: "six seeds were used to limit the
  computational cost of the MC analysis, following accepted international standards"; and the caveat
  "**the use of only six seeds does not guarantee the full convergence of all quantities, especially
  in terms of standard deviations**", while "the differences in AEP and DELs are indeed small, this is
  not true for the ultimate loads". This is the citation for "six seeds converge DELs but not higher
  moments".
- **VERIFIED** `https://wes.copernicus.org/articles/10/2903/2025/`: individual seed results as
  transparent markers, seed-averaged values opaque, **standard deviation across seeds as error bars**,
  and explicitly **no significance testing or confidence intervals**.
- **VERIFIED** `https://wes.copernicus.org/articles/10/2005/2025/` Sect. 3.3: means "averaged over 10
  turbulent wind field instances", **1σ error bars / shaded bands**.
- **VERIFIED** `https://wes.copernicus.org/articles/9/1791/2024/` Sect. 3.3: **median and interquartile
  range**.
- **VERIFIED** `https://wes.copernicus.org/articles/8/1299/2023/`: results are the "average value of
  A_eq by six random seeds" — plain seed means, no intervals.
- **VERIFIED** `https://ar5iv.labs.arxiv.org/html/2110.14169`: both controllers are run over the
  **identical 78-case matrix** (6 seeds × 13 wind speeds), i.e. paired by construction; the paper
  reports std/max per case and applies **no paired statistical test**.
- **UNVERIFIED** (search summaries only): bootstrap CIs for seed-driven load uncertainty, and 95 % CIs
  on the mean via the t distribution. These appeared in Copernicus *preprint* PDFs that could not be
  parsed (`wes-2024-68`, `wes-2025-112`) — chase them if we want a citation for bootstrap CIs.

**For our paper.** Nothing fetched here does a paired bootstrap over episodes. Our per-wind-speed
terms with paired bootstrap intervals over a common seed set (`scripts/dev/paired_bootstrap.py`) are
**above** the field's norm; saying so in the methods is defensible and also explains why we can
resolve ~1 J differences that seed-mean-only papers could not.

---

## Open items (not verified; do not cite from here)
1. **IEC 61400-1 ed. 4 DLC 1.2 clause** — seed count and 10-min duration as printed in the standard.
   Needs an offline copy; every fetchable PDF was unparseable.
2. **NREL MLife theory manual** — the canonical lifetime-DEL formula; nrel.gov unreachable from this
   machine.
3. **IEC class table** — A/B/C I_ref = 0.16 / 0.14 / 0.12 and V_ref = 50 / 42.5 / 37.5 m/s appeared
   only in search summaries. The one anchor actually read:
   **VERIFIED** `https://wes.copernicus.org/articles/9/2001/2024/` states "a reference value of
   turbulence intensity was taken to be **0.12, for the least turbulent wind turbine class C**" and
   cites "the Weibull form of the normal turbulence model (NTM) in Eq. (11) of IEC 61400-1:2019".
   Our class B I_ref = 0.14 is therefore **not yet backed by a fetched page** — check the TurbSim
   input we actually used and cite the standard directly.
