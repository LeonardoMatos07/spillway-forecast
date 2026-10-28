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

Production flow, all automatic once the worker runs (`docker compose up -d forecast-worker`):

1. **`spillway watch`** checks every 60 minutes (a few HEAD requests and one SQL query, ~3 s when
   nothing changed). When the ONS files of the current month changed, or silver has a new month,
   it rebuilds the series (silver history + ONS tail, see Data), and **only if the data reach a
   day later than the latest published run** it publishes: `forecast_model` / `forecast_run` /
   `forecast_hourly` in one transaction, then `POST /v1/admin/revalidate {"tags": ["forecast"]}`
   so the API drops its cache and the web refetches at once.
2. **`spillway retrain`** is started by the same worker once a day in a subprocess (the hourly
   refresh is never blocked). It retrains only when silver holds a complete month newer than the
   model's data and 25 days passed since the last training; a promotion gate decides whether the
   new model replaces the current one (see Monthly retraining).
3. The API serves the hours from *now* (Brasilia time) onwards of the latest ready run, with
   `valid_from`/`valid_to`, `data_through` and `generated_at`; the web shows the real week.

The publisher connects as its own role `energia_forecast` (INSERT/DELETE on the three tables,
UPDATE only on `forecast_model.is_current` and `forecast_run.status`, SELECT on the rest; grants
live in `spillway-infra`, tables and migration `0007` in `spillway-api`). `spillway serve` is only
a debugging tool and is not part of the production path. Airflow is deliberately not involved
(no ML dependencies in its image): the worker polls, so a new ONS day reaches the page within
the polling interval plus ~40 s, and the monthly pipeline run is noticed through the new silver
month.

## Data

* **Source: hybrid.** The hourly series are the **silver layer of `spillway-data`** (SeaweedFS
  lake, S3 API at `http://localhost:8333`, bucket `energy-lake`:
  `silver/{wind,solar}_cluster_curtailment/year=YYYY/month=MM/data.parquet`) for every complete
  month, **plus a provisional tail taken straight from the ONS open data** for the months after
  the last silver month. Silver only holds complete months (the pipeline loads the previous
  month), which would make every forecast start ~4 weeks in the past; ONS fills the current
  month's file up to about yesterday.
  * Silver is typed, deduplicated, has invalid rows quarantined, keeps Brasilia local time and UTC,
    and models curtailment: `curtailed_mwh = max(GNRa, 0) x 0.5`, `generation_mwh = verified
    generation x 0.5`. The ONS tail uses the same aggregation (`aggregate_raw`), so definitions
    match.
  * Tail rows are flagged `provisional` in `hourly.parquet` (ONS revises them) and trimmed to
    complete days; `data/processed/hourly_meta.json` records the boundary, the ONS ETags and the
    overlap check.
  * **Overlap check**: for the last silver month the direct ONS file is aggregated and compared
    with silver. Measured on 2026-09: maximum absolute difference 5e-12 MWh (floating point
    noise) on 10,800 wind and 9,360 solar hours: the two sources agree. If they ever disagree,
    silver wins and the deviation is logged.
  * The reconcile command ignores provisional rows (they are not in the warehouse).
* What is derived here: the hourly aggregation per state, subsystem and SIN (the pipeline only
  publishes monthly aggregates), written to `data/processed/hourly.parquet` (~850k rows).
* **Lake access** (`src/spillway_forecast/silver.py`): boto3 with *path-style* addressing, explicit
  region `us-east-1` and the static keys (`LAKE_S3_ENDPOINT_URL`, `LAKE_BUCKET`,
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`; defaults are the local
  docker-compose values). The earlier `AccessDenied` came from not sending the keys / using the
  wrong addressing; with the pipeline's configuration it works. Install with `uv sync --extra lake`.
* `spillway ingest` is hybrid by default; `--from silver` (validated months only) and
  `--from ons` (everything from ONS, a fallback when the lake is unavailable) are still available.
* **Reconciliation** (`uv run spillway reconcile`, read-only SELECT on the warehouse): the monthly
  sums of the hourly series equal `state_monthly_curtailment` (805 state-month rows) and
  `subsystem_monthly_curtailment` incl. the SIN rows (346 rows) for every wind and solar month:
  no missing months in either direction, maximum absolute difference 0.0005 MWh (the warehouse
  rounds to 3 decimals), maximum relative difference 1.6e-4 on tiny months and 2.7e-8 for
  generation. See `reports/reconcile.txt`.
* **History actually used (all of it, every year):** wind **2021-10-01 to 2026-10-27** (~5 years,
  15 series: 10 states, 4 subsystems, SIN) and solar **2024-04-01 to 2026-10-27** (~2.5 years, when
  ONS started publishing it; 13 series, of which 12 are modelled: solar MS has under 120 days of
  history and is dropped). Validated silver covers up to 2026-09-30; the 27 days after it
  (18,144 hourly rows) are the provisional ONS tail. Nothing is sampled out: training rows are
  built from every origin (every 3 days) of every series.
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

## Results (held-out test, 251k hour-forecasts, 2026-04-01 .. 2026-10-27)

| Model | MAE (pp) | RMSE (pp) | ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| Climatology (baseline) | 13.90 | 23.98 | 0.833 | 0.770 | 0.2204 | 0.218 |
| Seasonal naive (baseline) | 12.07 | 21.23 | 0.801 | 0.691 | 0.1927 | 0.193 |
| **HistGradientBoosting** | **10.15** | **16.43** | **0.915** | **0.877** | **0.1167** | **0.037** |

Base rate of risk hours in the test period: 42%. Block-bootstrap (by week) 95% CI of model minus
climatology: MAE -3.7 pp [-4.5, -2.8]; Brier -0.104 [-0.119, -0.087]. Versus seasonal naive: MAE
-1.9 pp [-2.3, -1.5]; Brier -0.076 [-0.085, -0.067].

**History of the numbers.** First run (direct ONS download, same test window): MAE 10.09, AUC
0.915, ECE 0.038. Silver only (test to 2026-09-30, 218k rows): MAE 10.05, AUC 0.917, ECE 0.036.
Hybrid (silver + provisional ONS tail, this table): the test window is again the one of the first
run and the numbers agree with it within noise (the model is refitted, not identical). The
source change brings freshness, not accuracy, and no gain is claimed from it.

**Effect of the provisional tail.** The only test targets that are provisional are those of
October 2026 (the last 27 days). Per target month, model MAE is 10.3 / 9.9 / 10.2 / 9.2 / 10.1 /
10.7 pp for April-September and **10.9 pp for October** (ROC-AUC 0.90, inside the 0.88-0.94 range of
the other months); climatology is 15.5 pp in October. So the provisional month is
slightly harder but nowhere near enough to change any conclusion; we cannot separate "provisional"
from "October" with this data. Silver and a fresh ONS download agree to floating-point precision
for September, so revisions of already published days look small, but ONS may still revise the
newest days after the forecast was made.

Rolling-origin CV (mean of 5 folds): MAE 9.45 / 11.54 / 10.61 pp and ROC-AUC 0.891 / 0.804 /
0.757 for model / climatology / seasonal naive (identical to the previous runs: the folds end
before 2026-04). Reliability plot: `reports/figures/calibration.png`. Full numbers (per fold,
source, forecast day, target month, permutation importance, calibration bins):
`reports/results.json`. A full retrain took **12.5 minutes** (752 s) on this machine.

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
uv run spillway ingest            # silver history + ONS tail -> data/processed/hourly.parquet
uv run spillway reconcile         # silver months vs the warehouse tables (read-only)
uv run spillway train             # backtest + MLflow + models/bundle.joblib + reports/results.json (~13 min)
uv run spillway publish --dsn postgresql://USER:PASS@HOST:5433/warehouse   # one-off publish
uv run spillway refresh           # publish only if new ONS data arrived, then tell the API
uv run spillway watch --every 60  # the loop the worker runs
uv run spillway retrain --dry-run # what the monthly retrain would do now
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy
```

`publish` needs the `spillway-api` migration `0007`. It scores all series from one common origin
(the day after the latest observed day), inserts the model row (once per MLflow run), a run in
`loading`, copies the hourly rows, flips the run to `ready`, marks the model current and keeps
the last 14 runs, all in **one transaction**; running it twice for the same model and origin is a
no-op.

### The worker (starts by itself, you decide when)

```bash
docker compose up -d --build forecast-worker   # builds the image, starts `spillway watch --every 60`
docker compose logs -f forecast-worker
docker compose stop forecast-worker            # SIGTERM: it finishes the current step and exits
```

`docker-compose.yml` joins the external network `energia-data_default` so the container reaches
`lake:8333` and `warehouse-db:5432` with the pipeline's credentials (env `LAKE_ACCESS_KEY`,
`LAKE_SECRET_KEY`, `PUBLISH_DSN`; defaults `lakeadmin`/`lakesecret` and `warehouse`/`warehouse`),
and posts to `API_URL` (default `http://host.docker.internal:8000`, optional Cognito machine token
through `COGNITO_TOKEN_URL`, `DATA_CLIENT_ID`, `DATA_CLIENT_SECRET`, as in `spillway-data`).
`models/` and `reports/` are mounted from the host (the trained bundle and results); a named
volume keeps the hourly series and the MLflow database. In production the same image runs as one
long-running ECS service (one task is enough; the advisory locks make extra tasks harmless) with a
persistent volume (EFS) for `models/`, `reports/` and `/data`, `PUBLISH_DSN` of the
`energia_forecast` role from Secrets Manager, and `API_URL` of the API service; no infra changed
here.

Each check logs one line, e.g. `refresh: up-to-date: no change in ONS files or silver (2.8 s)` or
`refresh: new data -> published run 2 (data through 2026-10-27, 4536 rows, 31.4 s, api=...)`.
Errors never stop the loop: exponential backoff (1, 2, 4... minutes, capped at the interval), one
log line per attempt. `FORECAST_MODEL_MAX_AGE_DAYS` (45) makes the refresh log a warning when the
published model is older than that.

## Monthly retraining (automatic)

```
ONS monthly file -> spillway-data DAG (day 3, 06:00): silver -> gold -> report published and e-mailed
  -> the worker's daily check sees a complete month newer than the model's data
  -> spillway retrain: ingest -> same protocol as `train` -> promotion gate
  -> promote + publish (forecast_model.is_current flips in the publish transaction) + API revalidate
  -> the website shows the new model card and metrics
```

* **Trigger** (`retrain.decide_trigger`): silver has a complete month ending after the data the
  current model was trained on, **and** at least 25 days since the last training (a DAG re-run
  never retrains twice). `--force` skips both rules, `--dry-run` only reports.
* **Gate** (`retrain.gate`, constants `GATE_MAX_AUC_DROP=0.01`, `GATE_MAX_MAE_RISE_PP=0.3`,
  `GATE_MAX_ECE_RISE=0.02`, `MIN_GATE_ROWS=2000`): candidate and incumbent are scored on the same
  new window, the days after the incumbent's training data, which neither has seen (the candidate
  is re-fitted only on data before that window). The candidate is promoted only if it is not
  worse beyond those tolerances **and** on its own final test it beats climatology with a 95%
  block-bootstrap interval below zero for both absolute error and Brier score.
* **Rejection**: the current model and bundle stay, the reasons are logged, written to
  `reports/retrain_last.json` and tagged on the MLflow run (`gate`, `gate_reasons`). They are not
  surfaced on the website: that would need a schema change and the contract was kept as is.
* **Promotion** is atomic (temp file + rename), keeps `bundle.previous.joblib` and
  `results.previous.json`, registers the model in MLflow and publishes under the publish lock.
  The committed `MODEL_CARD.md` and this README are never rewritten by the worker; the website
  reads the numbers from the database.
* **Safety**: Postgres advisory locks (a second retrain or a publish during a promotion waits or
  skips), retraining runs in a subprocess with a time limit (3 h) so the hourly refresh keeps
  serving with the old model, and any failure leaves bundle, results and database untouched
  (tested). Manual override at any time: `uv run spillway retrain --force`.

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
