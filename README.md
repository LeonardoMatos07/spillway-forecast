# spillway-forecast

Hourly **risk of wind/solar curtailment** (energy cut by the Brazilian grid operator ONS) for the
next 7 days, per state / subsystem and source, with honest temporal validation, MLflow tracking and a
FastAPI service that the Spillway API and web app can call.

> Curtailment ("constrained-off", *corte de geração*) is energy that wind and solar plants could
> have produced but were told not to. Knowing which hours are likely to be cut is useful to
> traders, plant operators and analysts. This project predicts, for every hour of the next 7 days:
> the probability that the hour loses at least 10% of its potential output (the **risk**) and the
> expected curtailment percentage.

## Architecture

The forecast reaches the website through the **warehouse**, not through an HTTP call
(ADR 0007 in `spillway-api`): the API stays up with the last forecast when the ML job is down,
its latency does not depend on scoring, and every forecast is versioned.

```mermaid
flowchart LR
    ONS[(ONS open data)] --> DATA[spillway-data<br/>bronze -> silver -> gold]
    DATA -->|silver parquet<br/>S3 API of the lake| H[ingest<br/>hourly state / subsystem / SIN]
    H --> F[features.py<br/>leak-free, origin = 00:00]
    F --> V[validation.py<br/>5 rolling-origin folds + held-out test]
    V --> M[HistGradientBoosting<br/>classifier + regressor]
    V --> B[baselines.py<br/>climatology, seasonal naive]
    M --> ML[(MLflow<br/>sqlite + registry)]
    M --> S[bundle.joblib + reports/results.json]
    S --> P[spillway publish<br/>batch job, one transaction]
    H --> P
    P -->|forecast_model / forecast_run / forecast_hourly| PG[(Postgres warehouse)]
    PG --> API[spillway-api<br/>GET /v1/forecast, /v1/forecast/model]
    API --> W[spillway-web<br/>/previsao]
    S -. optional, debugging only .-> SRV[spillway serve<br/>FastAPI :8100]
```

Production flow: **train offline** (manual, about monthly, ~20 min) -> **`spillway publish` on a
schedule** (daily, after the monthly `spillway-data` publish) writes the next 168 h of every series
to Postgres -> the API reads the latest `ready` run -> the web only calls the API. The publisher
connects as its own role `energia_forecast` (INSERT/DELETE on the three tables, UPDATE only on
`forecast_model.is_current` and `forecast_run.status`, SELECT on the rest; grants live in
`spillway-infra`, tables and migration `0007` in `spillway-api`). `spillway serve` is only a
debugging tool and is not part of the production path.

## Data

* **Source: the silver layer of `spillway-data`** (SeaweedFS lake, S3 API at
  `http://localhost:8333`, bucket `energy-lake`):
  `silver/{wind,solar}_cluster_curtailment/year=YYYY/month=MM/data.parquet`. Silver is typed,
  deduplicated, has invalid rows quarantined, keeps Brasilia local time (`measured_at`) and UTC,
  and already models curtailment: `curtailed_mwh = max(GNRa, 0) x 0.5`, `generation_mwh =
  verified generation x 0.5`. Nothing else is needed from the pipeline.
* What is derived here: the hourly aggregation per state, subsystem and SIN (the pipeline only
  publishes monthly aggregates), written to `data/processed/hourly.parquet` (~850k rows, a few MB).
* **Lake access** (`src/spillway_forecast/silver.py`): boto3 with *path-style* addressing, explicit
  region `us-east-1` and the static keys (`LAKE_S3_ENDPOINT_URL`, `LAKE_BUCKET`,
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`; defaults are the local
  docker-compose values). The earlier `AccessDenied` came from not sending the keys / using the
  wrong addressing; with the pipeline's configuration it works. Install with `uv sync --extra lake`.
* The old direct-ONS loader (`spillway ingest --from ons`) is kept as a fallback only (e.g. when
  the lake is not available); it produces the same schema and the same definitions.
* **Reconciliation** (`uv run spillway reconcile`, read-only SELECT on the warehouse): the monthly
  sums of the hourly series equal `state_monthly_curtailment` (805 state-month rows) and
  `subsystem_monthly_curtailment` incl. the SIN rows (346 rows) for every wind and solar month:
  no missing months in either direction, maximum absolute difference 0.0005 MWh (the warehouse
  rounds to 3 decimals), maximum relative difference 1.6e-4 on tiny months and 2.7e-8 for
  generation. See `reports/reconcile.txt`.
* Real history actually used: **wind 2021-10 to 2026-09-30 (~5 years, 15 series)**, **solar 2024-04
  to 2026-09-30 (~2.5 years, 12 modelled series)**, 28 series exist and 27 are modelled (solar MS has under 120 days of history and is dropped). The lake holds
  complete months only, so the data now ends on 2026-09-30 (the previous direct-ONS run went to
  2026-10-27): the final test period is shorter (2026-04..2026-09) and a forecast made today starts
  from stale data. The API/web report `stale_days`.
* Nothing is synthetic. Unit tests use synthetic frames only to exercise code paths.
* Target: `curtailed / (curtailed + generation)` per hour (undefined at night for solar, those
  hours are dropped and flagged `applicable=false`). Risk label: that value >= 10%.

## Features (all strictly past, except deterministic calendar)

Forecasts are issued at 00:00 of an *origin day* and cover the next 168 hours. Per row:
horizon, hour, weekday, month, weekend, Brazilian holiday, day-of-year sin/cos (calendar of the
target hour, known in advance); curtailment share of the last day, 7/28/90 days and their trend;
max hourly share of the last day; generation level over 1/7/28 days and its ratio; same-hour
curtailment share yesterday and averaged over 7/28 days; same-hour generation over 7 days;
same hour one week earlier; series id and source. Weather is **not** used (no free, aligned
source was integrated), which is the biggest known limitation.

## Validation protocol

* **No random splits.** Rolling-origin, expanding window: 5 folds of 3 months each
  (2025-01 .. 2025-12, plus 2026-01..03), each trained only on rows whose *target hour* is before
  the fold start (embargo of the 7-day horizon).
* **Held-out final test: 2026-04-01 to 2026-09-30**, never used for any decision (no tuning was
  done on it; hyper-parameters are fixed, early stopping is off because sklearn's internal
  validation split is random).
* Baselines are fitted on data before the split only: *climatology* (mean by series x month x
  weekday/weekend x hour) and *seasonal naive* (same hour last week).
* `tests/test_features.py` proves no future data leaks into features: corrupting every
  observation at or after the origin leaves all history features unchanged.

## Results (held-out test, 218k hour-forecasts, 2026-04 .. 2026-09)

| Model | MAE (pp) | RMSE (pp) | ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| Climatology (baseline) | 13.65 | 23.76 | 0.841 | 0.775 | 0.2174 | 0.218 |
| Seasonal naive (baseline) | 11.84 | 21.15 | 0.803 | 0.691 | 0.1900 | 0.190 |
| **HistGradientBoosting** | **10.05** | **16.45** | **0.917** | **0.882** | **0.1156** | **0.036** |

Base rate of risk hours in the test period: 42%. Brier skill vs the base-rate forecast: 0.52
(model), 0.22 (seasonal naive), 0.11 (climatology). Block-bootstrap (by week) 95% CI of
model minus climatology: MAE -3.6 pp [-4.6, -2.5]; Brier -0.102 [-0.123, -0.081]. Versus
seasonal naive: MAE -1.8 pp [-2.3, -1.4].

**Versus the previous run (direct ONS download, test to 2026-10-27, 251k rows):** MAE 10.09 /
13.90 / 12.07, ROC-AUC 0.915 / 0.833 / 0.801, ECE 0.038 / 0.218 / 0.193. The differences are small
and come from the test window (October 2026 is no longer in the lake, so the test has 218k instead
of 251k rows and the climatology baseline changed with it); the cross-validation folds are
identical to the previous run (mean MAE 9.45 pp both times) because the lake and the direct ONS
files agree. We do not claim any gain from the new source: it changes provenance, not accuracy.

Rolling-origin CV (mean of 5 folds): MAE 9.45 / 11.54 / 10.61 pp and ROC-AUC 0.891 / 0.804 /
0.757 for model / climatology / seasonal naive. Reliability plot: `reports/figures/calibration.png`
(the bins are also published to the website). Full numbers (per fold, source, forecast day,
permutation importance, calibration bins): `reports/results.json`.

Honest caveats:

* The gain is much clearer for the **risk classification** (AUC 0.92 vs 0.83; calibration error
  0.04 vs 0.22) than for the **curtailment percentage**. In fold 1 (2025-01..03) the model's MAE
  (10.57 pp) was slightly *worse* than climatology (10.45 pp).
* Skill barely decays from day 1 to day 7 (MAE 9.7 -> 10.3 pp) because most of the signal is the
  recent level and the hour-of-day pattern, not weather. That also means the model cannot
  anticipate sudden regime changes (new transmission limits, weather events).
* The most important feature by far is the same-hour 28-day average share. The model is largely
  a good, adaptive climatology plus calendar effects, not a physical model.
* Curtailment rose sharply over 2024-2026 so periods differ in base rate (32% CV vs 42% test).

## Run it

```bash
uv sync --extra lake --extra warehouse
uv run spillway ingest            # silver layer (lake) -> data/processed/hourly.parquet
uv run spillway ingest --from ons # fallback: download the raw ONS files directly
uv run spillway reconcile         # monthly sums vs the warehouse tables (read-only)
uv run spillway train             # backtest + MLflow + models/bundle.joblib + reports/results.json (~20 min)
uv run spillway publish --dsn postgresql://USER:PASS@HOST:5433/warehouse   # writes the forecast (see below)
uv run spillway register --run-id <id>   # only if registration was skipped / failed
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse runs and the model registry

uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy
```

**Publishing.** `spillway publish` needs the `spillway-api` migration `0007` applied. It scores all
series from one common origin (the day after the latest observed day), inserts the model row (once
per MLflow run), a run in `loading`, copies the hourly rows, flips the run to `ready`, marks the
model current and keeps the last 14 runs, all in **one transaction**; running it twice for the same
model and origin is a no-op. It reads `reports/results.json` and `models/bundle.joblib`
(the MLflow registry holds the same estimators; the bundle is what the service loads). Use
`--dsn` or `PUBLISH_DSN` (role `energia_forecast` in production, the warehouse user locally).
Schedule it daily after the monthly `spillway-data` publish; retraining is manual.

The API serves it: `GET /v1/forecast?scope=NE&source=wind&days=7`, `GET /v1/forecast/scopes` and
`GET /v1/forecast/model` (training results for the website: data period, metrics baseline vs model
with bootstrap intervals, calibration bins, feature importance, limitations).

**Debugging service (optional).** `uv run spillway serve --port 8100` exposes the same scoring over
HTTP (`GET /v1/forecast`, `/v1/scopes`, `/v1/model`, `/health`); it is not used in production.

## What I learned

* In this data **leakage is easy to create by accident**: rolling 28-day windows must be shifted
  by a day, the "same hour last week" feature is only valid up to a 7-day horizon, and
  scikit-learn's default early stopping uses a random validation split.
* Training rows overlap heavily (every origin produces 168 horizons), so the baseline has to be
  fitted on unique hourly observations, and the significance test must resample *blocks* of time
  (weeks) instead of rows.
* Beating a climatology on MAE is much harder than beating it on classification metrics; reporting
  only AUC would have hidden that. Reliability (ECE) is where the model shines.
* MLflow 3's default skops serialization refuses HistGradientBoosting models; cloudpickle is used.
* Without weather inputs this is a statistical "recent regime" model; the next step is adding
  forecast wind speed / irradiance and ONS load and price (CMO) forecasts.

See `MODEL_CARD.md` for intended use and limitations.
