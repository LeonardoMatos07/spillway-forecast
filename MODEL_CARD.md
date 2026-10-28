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
  for wind or solar in a state or subsystem; shown in the members Analyses page of Spillway.
* **Not** intended for: trading or dispatch decisions without human review, regulatory or
  contractual claims about compensation, plant-level (single plant) forecasts, horizons
  beyond 7 days.

## Data

ONS open data `restricao_coff_eolica_tm` (from 2021-10) and `restricao_coff_fotovoltaica_tm`
(from 2024-04) up to 2026-10-27, 30-minute cluster rows aggregated to hourly state, subsystem and
SIN series. Curtailed energy is the official "geração não realizada apurada" (GNRa) and
generation is the verified generation, as in `energia-data`. 27 series, 1.62 M forecast rows
(origins every 3 days x 168 horizons), 251 k rows in the final test.
Hours where the denominator (curtailed + generated) is under 1 MWh (night-time solar) are
excluded from training and evaluation and reported as "not applicable" by the API.

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
| HistGradientBoosting | 10.09 | 16.30 | 0.915 | 0.875 | 0.1173 | 0.038 |

Risk base rate in the test: 42% (32% on average across the CV folds).
By source (test): wind AUC 0.922 / Brier 0.110 / MAE 9.7 pp; solar AUC 0.894 / Brier 0.132 /
MAE 10.9 pp. By forecast day: AUC 0.918 (day 1) to 0.905 (day 7); MAE 9.7 to 10.2 pp.
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
* Operational: the forecast starts the day after the last observation. If ingestion is behind,
  the API returns `stale_days > 0` and the UI warns the user.
* Hourly values are in ONS local time and aggregated from 30-minute rows.

## Ethical and operational considerations

Data is public and aggregated (no personal data). The forecast could influence market behaviour
or be misread as a guarantee; the UI labels it as a statistical risk indicator. It must not be
the sole basis of financial or operational decisions, and it says nothing about *why* energy is
curtailed (reliability vs energy reasons) or whether it will be compensated.

## How to retrain

```bash
uv run spillway ingest   # refreshes the last 2 months and fetches new ones
uv run spillway train    # backtest, report, MLflow run, registry, models/bundle.joblib
```

Update `test_start`/`fold_starts` in `src/spillway_forecast/config.py` when moving the held-out
period forward (the old test period should then be used as development data), and update the
tables above from `reports/results.json`.
