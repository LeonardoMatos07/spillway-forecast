# Model card: spillway-forecast curtailment risk

## Model

Two scikit-learn `HistGradientBoosting` estimators trained jointly on the same features:
a classifier for the **risk** (hour with curtailment >= 10% of potential output) and a
regressor for the **expected curtailment percentage**. One global model for all 27
(state/subsystem x source) series, with series id and source as categorical features.
Registered in MLflow as `spillway-risk-classifier` and `spillway-pct-regressor`; the serving
bundle is `models/bundle.joblib`. Parameters: 250 iterations, learning rate 0.06, 31 leaves,
min 200 samples per leaf, L2 = 1, no early stopping, seed 42.

## Intended use

* Decision support and analysis: which hours of the next 1-7 days are more likely to be curtailed
  for wind or solar in a state or subsystem; shown in the members-only `/previsao` page of Spillway.
* **Not** intended for: trading or dispatch decisions without human review, regulatory or
  contractual claims about compensation, plant-level (single plant) forecasts, horizons
  beyond 7 days.

## Data

Hybrid: the silver layer of `spillway-data` (typed, deduplicated, invalid rows quarantined; read
from the lake over S3) for every complete month, `wind_cluster_curtailment` from 2021-10 and
`solar_cluster_curtailment` from 2024-04 up to 2026-09-30, plus a provisional tail from the ONS
open data (same aggregation) for the days after that, up to 2026-10-27. 30-minute cluster rows are
aggregated to hourly state, subsystem and SIN series. Curtailed energy is the official "geracao
nao realizada apurada" (GNRa) and generation is the verified generation, exactly the pipeline's
`curtailed_mwh` / `generation_mwh`; the monthly sums of the silver part reconcile with the warehouse
tables (max difference 0.0005 MWh) and silver agrees with a fresh ONS download to 5e-12 MWh.
27 modelled series (solar MS has under 120 days of history and is dropped), 1.62 M forecast rows
(origins every 3 days x 168 horizons), 251 k rows in the final test. The provisional tail is
revised by ONS; its rows are flagged `provisional`.
Hours where the denominator (curtailed + generated) is under 1 MWh (night-time solar) are
excluded from training and evaluation and reported as "not applicable".

## Features

Calendar of the target hour (hour, weekday, month, weekend, Brazilian holiday, seasonal
sin/cos) plus history available at 00:00 of the origin day: curtailment share over the last
day / 7 / 28 / 90 days and trend, max hourly share of the last day, generation level (1/7/28 days),
same-hour share yesterday and averaged over 7 and 28 days, same-hour generation over 7 days,
same hour last week. No weather, no price, no load, no outage information.

## Validation protocol

* Rolling-origin, expanding-window backtest with 5 folds of 3 months (2025-01 .. 2026-03).
  Training rows must have their *target hour* before the fold start (7-day embargo).
* Final held-out test 2026-04-01 .. 2026-10-27, untouched by development; hyper-parameters
  were not tuned.
* Baselines: climatology by series x month x weekday/weekend x hour (fitted on unique hourly
  observations before the split) and seasonal naive (same hour last week).
* Uncertainty: paired block bootstrap by week for the model-minus-baseline loss difference.
* Leakage tests: corrupting all data at/after the origin does not change history features;
  training targets never reach the evaluation block.

## Metrics (held-out test)

| Model | MAE (pp) | RMSE (pp) | ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| Climatology | 13.90 | 23.98 | 0.833 | 0.770 | 0.2204 | 0.218 |
| Seasonal naive | 12.07 | 21.23 | 0.801 | 0.691 | 0.1927 | 0.193 |
| HistGradientBoosting | 10.15 | 16.43 | 0.915 | 0.877 | 0.1167 | 0.037 |

Paired block bootstrap (weeks), model minus climatology: MAE -3.7 pp [-4.5, -2.8], Brier -0.104
[-0.119, -0.087]; minus seasonal naive: MAE -1.9 pp [-2.3, -1.5].
Risk base rate in the test: 42% (32% on average across the CV folds). Per target month the model
MAE is 9.2-10.7 pp from April to September and 10.9 pp in October (provisional data).
Rolling CV (mean over folds): MAE 9.45 (model) vs 11.54 (climatology) vs 10.61 (seasonal naive);
AUC 0.891 vs 0.804 vs 0.757. Source of all numbers: `reports/results.json`, MLflow run
`backtest`.

**Where the model does not clearly win:** on the curtailment percentage the improvement over
climatology is small in some periods; in CV fold 1 (2025-01..03) the model's MAE (10.57 pp) was
slightly worse than climatology's (10.45 pp). The advantage is robust for the risk probability
and its calibration.

## Limitations

* Weather is not an input. The model mostly extrapolates the recent regime and the hour-of-day
  pattern; it will miss events (a storm, a transmission outage, a rule change). Skill hardly
  decays with the horizon because there is no horizon-specific information to lose.
* Non-stationarity: curtailment grew strongly in 2024-2026 and depends on grid expansion and
  regulation; the model can lag behind such shifts and degrade silently. Retrain regularly and
  monitor the live error.
* Solar history is only ~2.5 years; per-state series can be short. New states or plants are
  unknown to the model (series are fixed at training time).
* The ONS "apurada" figures are revised; the last month of data may change.
* The risk threshold (10%) and the display levels (low < 0.33 <= medium < 0.66 <= high) are
  conventions, not calibrated decision thresholds.
* Operational: the forecast starts the day after the last observation. The lake receives complete
  months, so the forecast can be weeks stale; the API returns `stale_days` and `generated_at` and
  the UI warns the user.
* Hourly values are in ONS local time and aggregated from 30-minute rows.

## Ethical and operational considerations

Data is public and aggregated (no personal data). The forecast could influence market behaviour
or be misread as a guarantee; the UI labels it as a statistical risk indicator. It must not be
the sole basis of financial or operational decisions, and it says nothing about *why* energy is
curtailed (reliability vs energy reasons) or whether it will be compensated.

## How to retrain

```bash
uv run spillway retrain --force   # ingest + train + promotion gate + publish (automatic monthly)
uv run spillway train             # the same protocol without the gate, overwrites the bundle
uv run spillway reconcile
```

Update `test_start`/`fold_starts` in `src/spillway_forecast/config.py` when moving the held-out
period forward (the old test period should then be used as development data), and update the
tables above from `reports/results.json`.
