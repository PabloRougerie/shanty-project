# Vessel trajectory forecasting for maritime search and rescue - Project Summary

**Repo**: [github.com/PabloRougerie](https://github.com/PabloRougerie) · **Demo**: _link to fill in_

---

## Executive Summary

### Case

Ships broadcast their position through AIS (Automatic Identification System), a radio system most commercial vessels are required to run. When a ship stops sending that signal, the people responsible for it lose track of where it is.

This matters beyond emergencies. A ship that goes silent may simply have a technical problem while still moving normally: knowing where it likely is stays useful for safety and situational awareness, not only once a rescue is underway.

Without a filed route or a known destination to assume, on the ship's own recent movement forward is one of the only usable estimate of where it has gone. Assuming a simple "keep going the way you were going" guess functions as the operational floor for this task: whatever more complex prediction model must do better than that, with the little amount of data available in the AIS when in was stil transmitting.

### Requirements

The end user is a maritime coordination center. To be useful, the tool has to hand over a sized, circular search area around the predicted position, not an open-ended corridor stretching along every route the ship could have taken. A sized area tells searchers roughly how big a zone to cover and gives them a sense of how confident the estimate is.The larger the area, the less confident a model is about its position forecast. 

Beyond that, the tool needs to:
- work for any horizon, from a couple of hours to three days, without being rebuilt for each one,
- rely only on the ship's own signal history (no weather, sea current, or destination data, which are harder to obtain reliably),
- use a fixed amount of that history for every prediction, so the system behaves the same way regardless of the situation.

### Key Outcomes

- One prediction model that covers every waiting time from 2 hours to 3 days: no separate model to retrain or maintain per horizon.
- The tool shrinks the search area by **60% at short horizon, growing to 82% for the longest horizon tested** (3 days), which is when the search area is largest and a reduction matters most.
- The tool provides a 90% confidence disk. Checked on data it never trained on, that holds up well overall, though for the fastest-moving ships over the longest horizon the true share is a bit lower than announced (about 86%, not the full 90%).

### Deliverable

- A model that predicts a ship's position at the requested waiting time, together with a matching search radius around that point.
- A demo interface (UI) showing the prediction on a set of real, representative ship tracks.

### Important insights

- A ship's movement over time falls into a few recognizable patterns: sitting still, sailing in a straight line, or maneuvering (turning, changing speed). The longer the horizon before the prediction, the more likely the ship has switched between these patterns during that time, which is what makes long waits harder to predict.
- For short horizon forecast of ships moving in a straight line, a simple guess based on the ship's last known course and speed already works very well.
- The model built for this project does not do much better than that simple guess on those easy cases. Its value shows up on the harder ones: it strongly reduces the size of the worst-case errors, the rare but very large misses that would otherwise force a huge search area.
- It is not possible to tell, from a ship's past movement alone, whether the simple guess or the model will do better for that specific ship, because that depends on what the ship does next during the wait itself, which cannot be known in advance. What does help is adjusting the search disk to the ship's past speed, something that we found bringing more accurate prediction area.

---

## Repo structure

```
shanty_project/
├── configs/                  # default.yaml (horizons, lookback, split), demo_vessels.yaml (UI vessel list)
├── src/vessel_tracker/       # package: preprocessing, features, baseline, evaluation, calibration, metrics, paths
├── scripts/
│   ├── preprocess.py         # ingest → clean → resample → split
│   ├── train.py              # single LGBM + R90 lookups (lgbm + baseline) → artifacts/models/
│   ├── evaluate_test.py      # held-out test scoring, lgbm vs baseline → artifacts/reports/
│   └── _internal/
│       └── build_demo_bundle.py   # static UI bundle (logs + model + lookup)
├── notebooks/                # NB01-NB07, narrative analysis, source of truth for methods
├── tests/                    # pytest suite (package + script plumbing)
├── docs/                     # PACKAGE_API
├── data/
│   ├── raw/                  # downloaded AIS (gitignored)
│   ├── processed/            # cleaned splits (gitignored)
│   └── output/                # cached notebook results (CSV/parquet), source of the numbers below
├── artifacts/
│   ├── models/                # lgbm_final.pkl, lgbm_final.metadata.json, r90_lookup_lgbm/baseline.parquet (generated)
│   ├── reports/                # test_report.json (generated)
│   └── demo_bundle/            # curated logs + copies for the UI (generated)
├── UI/                        # Streamlit demo app
└── visualizations/             # figures referenced from this document
```

| Notebook | Focus | Outcome |
|---|---|---|
| NB01 | Framing, metrics, ingestion | Scope fixed: AIS-only, cargo/tanker, Gulf of Mexico, November 2024. Raw load 7,532,915 rows / 2,441 vessels. |
| NB02 | Cleaning, resampling, split, EDA | Cleaning: 7,532,915 → 6,167,266 rows, 2,441 → 2,200 vessels. After resampling and a minimal-length filter: 3,661,057 rows / 2,173 vessels. Vessel-level 70/15/15 split (train 1,521 / val 326 / test 326), no shared vessel. |
| NB03 | Deterministic baselines | Constant-velocity extrapolation kept as baseline; lookback fixed at 12h for every horizon and every later model. Quadratic extrapolation strictly underperformed. |
| NB04 | Linear models (Ridge) | Ridge improves MAE over baseline (~13% at 48-72h) and roughly halves the R90 search area from 12h on, but has a heavier worst-case tail; the ceiling motivates a non-linear model. |
| NB05 | Non-linear models (LGBM) | LGBM has lower MAE and lower R90 than both baseline and Ridge at every horizon; retained as the model. |
| NB06 | Feature engineering & selection | Lookback displacement (`dx`, `dy`) carries nearly all the engineered gain; ablation retains lags + `dx`/`dy` + vessel geometry. |
| NB07 | Final test evaluation & calibration | Test-set confirmation of the OOF gains; conditional radius R(p, h, speed) built on a held-out split and checked on test. |

---

## Technical details

### Data

**Dataset.** NOAA AIS position reports for cargo and tanker vessels (type codes 70-89), Gulf of Mexico, November 2024, filtered to that bounding box and vessel type at ingestion. Raw: 7,532,915 position reports, 2,441 vessels.

<!-- FIGURE: visualizations/gulf_map.png - study area (NOAA AIS, Gulf of Mexico, November 2024) -->

**Why this dataset.** Public and free, dense enough over one region and one month to support a held-out test split by vessel, and restricted to two vessel types with broadly comparable dynamics (no fishing boats or pleasure craft), which keeps the underlying kinematics reasonably homogeneous.

**Target.** Velocity in degrees per minute (`vx`, `vy`), not absolute position (see Model selection for why).

**Metrics.**
- Haversine MAE (km): ranks models against each other.
- R(p, h): the p-th percentile of the haversine error distribution at horizon h. R(90,5) = x means "90% of prediction error at horizon 5h are below 5km".  This is a descriptive statistic about past errors, not automatically a coverage guarantee for a new case; Notebook 07 checks it against held-out data.
- External reference: the IAMSAR 10 nautical mile (18.5 km) default immediate-search radius for *drifting* vessels, used as a fixed benchmark.

**Cleaning** [overall, incompréhensible pour qqn qui a pas lu le notebook. il faut etre plus général du genre "encoder en nan les encodage "bizarre" de nan dans le dataset basé sur recherhce internet, vitesse interpolé jusqu'uà un point] (Notebook 02), applied in this order, each step reported so the attrition [pa compris] is traceable: drop exact duplicate rows; median-impute vessel dimensions (Length, Width, Draft) where missing, since each field's missing values are sentinel-encoded [pa compris et c pas un terme que j'utiliserais] and concentrated on vessels with no dimension data at all, ruling out a per-vessel median; gap-limited linear interpolation of speed over gaps under 30 minutes; drop `COG` (recoverable from the LAT/LON sequence, and its missingness concentrates on a handful of vessels, so keeping the column would cost more vessels than dropping it); a two-pass filter flagging any inter-ping speed above 50 knots, which separates isolated GPS glitches (the single bad ping is dropped) from two vessels sharing one identifier (the whole vessel is dropped if jumps persist on the second pass) [il faut etre un peu plus narraitf: on a vu des jump de vitese aberrante indiquent 2 vaisseux sur une meme track ou gps glitch]; drop any remaining rows with missing values. Net effect: 7,532,915 to 6,167,266 rows, 2,441 to 2,200 vessels. The GPS-jump filter alone accounts for 626,526 rows and 161 vessels of that drop.

Resampled to a fixed 10-minute step, then filtered to vessels with at least 2 hours of track [on s'en fout] (the shortest horizon used downstream): 3,661,057 rows, 2,173 vessels. Split by vessel, 70/15/15 (train 1,521 / val 326 / test 326 vessels), so no vessel appears in more than one split [important]. The test set is untouched until NB07.

### Vessel behavior analysis

Track shapes in the training set fall into three patterns: stationary,  straight-line transit, and maneuvering (turning, changing speed). At the case level in the final test evaluation, close to half the population is stationary (47%), 35% straight, 18% maneuvering (NB07 §5).

<!-- FIGURE: visualizations/tracks_examples.png - sample of train-set vessel tracks illustrating the three patterns (NB02 §8.1) -->

Widening the observation window shifts that mix: straightness decreases slightly and the speed distribution changes, so a longer window is more likely to contain a mix of patterns than a single one. Working hypothesis: prediction difficulties comes from the changes between pattern during the horizon window, not by the duration of the window per se.

### Model selection

**Target: velocity, not position.** The model predicts a velocity vector (`vx`, `vy`, degrees per minute) and the future position is reconstructed as present position plus velocity times horizon [math formula would be easier to understand]. 

Predicting speed vector rather than position was selected because: 
- it stays comparable with baseline, which compute the same quantity
- it performs better on boundaries: a zero horizon returns the present position with no special case
- the scale of prediction is more similar across horizon: a prediction at short horizon and a long horizon necessarily would have to predict latitude and longitude of differente values, whereas predicting the speed is more invariant to h. [je me suis mal exprimé là, à améliorer]

**Baseline: constant-velocity extrapolation.** Assumes the vessel holds its current course and speed. Tested across five lookback windows (1 to 24h) and a horizon grid up to 72h (NB03): a short lookback is marginally better for the baseline itself, but the project fixes 12h for every horizon and every later model, trading a small, bounded loss in baseline accuracy for one shared preprocessing pipeline and more history available to later, learned models. A quadratic extrapolation (fitting curvature explicitly over the lookback) was tested and strictly underperformed the linear version: the future position is not a simple geometric continuation of the recent track shape, which motivates a learned model.

**Linear model (Ridge).** A Ridge regression on the base features (position lags) modestly improved MAE over the baseline at long horizons (about 13% at 48-72h, slightly negative at 2h) and roughly halved the R90 search area from 12h on. The baseline kept a lower median error, and Ridge's own worst-case error at 72h reached 9,422 km, against 5,538 km for the baseline: a heavier extreme tail than the baseline itself. Testing many more lags (up to 72) and a longer lookback did not close that gap; the ceiling looked informational, a limit on what a global linear map of the features can express (the L2 penalty is a reasonable choice given the correlation between lag features, though regularization strength was not itself the limiting factor). That ceiling motivated testing a non-linear model.

**Non-linear model (LightGBM).** A single LGBM regressor has lower MAE than both the baseline and Ridge at every horizon, and the margin over the baseline grows with horizon (about 6% at 2h to about 38% at 72h). On the error tail (R90), LGBM is lowest at every horizon; on the median (R50), the baseline stays lowest through short and medium horizons, since a large share of tracks are stationary or straight, where constant-velocity extrapolation is close to exact by construction. LGBM was retained as the final model.

**Why not a sequential model (LSTM/RNN).** Not tested. Stopping at LGBM was a simplicity call: its gain over the baseline covered the project's need, and a lighter model is preferable once it does. 

### Model improvements and test

**Hyperparameter tuning.** A narrowed grid search (probe: 24h horizon, 12h lookback, 6 lags) improved MAE by about 2.4% over untuned LGBM. Not pursued further: the ceiling looked informational, not a tuning problem.

**Feature engineering.** Candidate features: the lookback displacement (`dx`, `dy`), and four local-dynamics features (curvature rate, effective speed, speed trend, recent turn rate). Adding `dx`/`dy` cut MAE by 9 to 23% depending on horizon, several times what tuning gave; the four local-dynamics features contributed nothing measurable and turned slightly negative at long horizons.

**Feature selection method: ablation, not permutation importance (decision).** The project retrains the model with one feature group removed at a time (leave-one-group-out) rather than reading feature importance inside a single fitted model.  Importance inside one fit can be misleading when features are correlated, since either one of two correlated features can look dispensable alone while the pair together carries real signal. The one correlated cluster here is vessel geometry (Length-Width at 0.93, Draft-either at 0.73), removed as one group for that reason.

**Retained feature set:** all position lags, `dx`, `dy`, vessel geometry (Length, Width, Draft), and the horizon `h`. Heading was tested and dropped: it gave a small short-horizon gain that turned slightly negative at 24h and 72h, and the intermediate lags already recover part of the same short-horizon signal.

**Test-set confirmation (NB07).** LGBM has lower MAE and lower R90 than the baseline at every horizon from 2h to 72h. Search area (proportional to R90 squared) drops 60.5% at 2h and 81.5% at 72h, the gain growing with horizon. The out-of-fold to test gap stays under 3.2% on MAE and 5.3% on R90, in the same range as the baseline's own sampling drift between splits, which argues against overfitting rather than proving its absence outright. This evaluation covers 3,832,949 position-horizon rows across 304 test vessels.

| Horizon | Baseline MAE | LGBM MAE | Baseline R90 | LGBM R90 | Search-area reduction |
|--------:|-------------:|---------:|-------------:|---------:|----------------------:|
| 2h  | 6.09   | 5.07   | 23.98   | 15.07  | 60.5% |
| 6h  | 20.14  | 15.08  | 77.67   | 42.31  | 70.3% |
| 8h  | 27.95  | 20.26  | 107.29  | 56.09  | 72.7% |
| 10h | 36.19  | 25.59  | 137.83  | 70.42  | 73.9% |
| 12h | 44.81  | 31.13  | 169.03  | 86.14  | 74.0% |
| 24h | 101.24 | 67.15  | 363.89  | 178.74 | 75.9% |
| 48h | 225.04 | 135.01 | 753.13  | 344.65 | 79.1% |
| 72h | 350.96 | 194.00 | 1139.44 | 489.52 | 81.5% |

_All distances in km, from `notebooks/outputs/test_summary_final.csv` (NB07). Search area scales with R90 squared._

<!-- FIGURE: visualizations/MAE_test_vs_oof.png - MAE by horizon, baseline vs LGBM, test vs out-of-fold -->
<!-- FIGURE: visualizations/R90_test_vs_oof.png - R90 by horizon, baseline vs LGBM, test vs out-of-fold, with the 10 NM reference -->

At 72h, LGBM's worst observed error (1,595 km) is roughly a third of the baseline's (4,914 km), and its 95th percentile (688 km) is below the baseline's 90th (1,139 km). On pooled R50 the baseline stays lower through most horizons, explained by regime: on stationary tracks (47% of test cases) the baseline is closer to the truth than LGBM on 97% of cases at 2h, down to 71% at 72h. On maneuvering tracks LGBM has lower MAE and lower R90 at every horizon, and the gap widens with horizon (MAE at 72h: 604 km baseline against 236 km LGBM). [dire explicitement du coup que LGBM ajoute de la valeur entre autre sur la limitation des erreurs catastrophique]

<!-- FIGURE: visualizations/CDF_test_error.png - test-set error CDF at 2h, 24h, 72h, baseline vs LGBM -->

### Prediction confidence

**Per-vessel difficulty is real and persistent.** Per-vessel MAE ranks consistently across horizons: Spearman 0.72 between 2h and 72h per-vessel MAE. The hardest decile of vessels averages 7.6 knots and 27% maneuvering cases; the easiest decile averages 0.5 knots and is 72% stationary. [ok mais ca dit quoi? qu'on a l'air de pouvoir indentifier les vx forte erreur ou basse erreur par leur speed? et donc peut etre qu'on peut jouer sur ca pour améliorer la prediction ou le rayon?]

**Effective speed relates to error, but not through a shape that supports a hard rule.** [ok mais tre plus narratif. ici on a jjuste une phrase qui decrit une fig. il faut dire ce qu'on en conclue. vitesse reliée à error mais pass d'une maniere où on peut etablir un cutoff baseline/lgbm pour routing. en revanche on doit pouvoir utiliser ca pour conditonnné l'erreur à al vitesse?] Speed's rank correlation with LGBM error falls from 0.62 at 2h to 0.20 at 72h, and at long horizons the true relationship is a U shape: error is high at the slowest deciles, lowest around the middle, and rises again at the fastest.  
[add fig]

**Why routing fails and conditioning works.** [réécrire tout ca. en gros: pas de routing parceque : effective speed a pas une courbe facile à exploiter pour ca (pas de coude) et les erreurs proviennent surement de vaiseaus avec changement dramatiques dans le futur. donc un routing fondé sur le passé sans information extra (une destinaton par ex) n'est pas jouable] Sending an easy-looking vessel to the baseline and a hard-looking one to the model would require knowing in advance whether that vessel changes pattern during the horizon. That fact belongs to the unobserved futur [tournure LLM je ne dirais pas ca]. 

By contrast, we can use the relationship observe with effspide to calibrate the confidence radius: it only requires how predictable a vessel has looked so far, which the past track does carry. Replacing one fleet-wide R(90, h) with R(90, h, speed) uses effective speed as a non-linear stand-in for that predictability, tightening or widening the disk per vessel without touching the point prediction.

**Coverage, before and after conditioning.** [il faut que ce soit clair dans le parag qu'on parle ici des erreurs ou incertitude "90% coverage is really 99%, the fastest bin's is 73%" c'est vague. ] The marginal (fleet-wide) radius over-covers slow vessels and under-covers fast ones at the same time, regardless of horizon. At 2h the gap is 26 points: the slowest bin's [oarlé olutot de slowest vessels] announced 90% coverage is really 99%, the fastest bin's is 73%. The gap narrows with horizon but does not close: at 72h the fastest bin under the marginal radius is at 79.5%. 

Conditioning on speed brings every bin close to the announced 90% (the fastest bin at 72h moves from 79.5% to 86.2%), though a residual gap of a few points remains for the fastest vessels at the longest horizons, a known limit rather than one conditioning fully removes. [c le plus important faut le metter en avant] [ce paragraphe et long et en fait décrit la figure qu'il y aura. il faut resserer sur les valeurs numérique et dire (en gros) "over confident sur les fast, under confident sur les slow]

<!-- FIGURE: visualizations/Rmarginal vs conditional.png - marginal vs speed-conditioned R90 coverage on test -->


---

## Technical annexe [à developper, y'a surement plus à dire sur la steructure du package non? pytest, pydantic etc etc qQUE SAIS JE]

**Stack.** Python, pandas, numpy, scikit-learn (MultiOutputRegressor, GroupKFold), LightGBM. Packaged as `vessel_tracker` (src layout, editable install via `uv`). Fixed random seed (273), the same vessel split replayed identically across notebooks, cached intermediate outputs with a force-recompute switch so a notebook can be re-read without re-running every cell.

**"MLOps"-level decisions.** [reformuler]
- Package code (`src/vessel_tracker/`) is separate from the narrative notebooks; the notebooks import from the package rather than redefining functions inline, so the analysis and the shippable code do not drift apart.
- Runnable scripts reproduce the pipeline outside a notebook: `scripts/preprocess.py`, `scripts/train.py` (fit + calibration, serving artifacts only), `scripts/evaluate_test.py` (held-out test scoring, lgbm vs baseline); `scripts/_internal/build_demo_bundle.py` builds the static bundle the UI reads [pas nécessaire].
- A pytest suite covers the package and script plumbing; CI runs pytest and ruff on every push.
- Configuration (horizons, lookback, split ratios, random seed) is defined in `configs/default.yaml`, not hardcoded in scripts or notebooks.
