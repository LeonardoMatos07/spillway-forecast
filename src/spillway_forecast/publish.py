"""Batch publisher: score every series and write the result to the warehouse (ADR 0007).

The production path of the forecast is *not* an HTTP call. After training, this job loads the
model, builds features from the silver-derived hourly series, scores the next 168 hours of every
series and writes everything to Postgres **in one transaction**:

``forecast_model`` (training results, one row per trained model, one ``is_current``),
``forecast_run`` (one scoring batch) and ``forecast_hourly`` (the hourly predictions).

Readers (the API) only ever see committed, complete runs. The job is idempotent: the same
model and origin never creates a second run. Old runs beyond ``keep_runs`` are deleted.

Database role: in production the job uses ``energia_forecast`` (SELECT on everything, INSERT/
SELECT/DELETE on the three forecast tables, column-level UPDATE on ``forecast_model.is_current``
and ``forecast_run.status`` only). Tables and migration are owned by ``spillway-api``.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from spillway_forecast import config
from spillway_forecast.service import ForecastService

DEFAULT_KEEP_RUNS = 14
LIMITATIONS = [
    "No weather inputs: wind speed, irradiance and temperature forecasts are not used.",
    "The output is a risk indicator, not a guarantee; calibration is checked on history only.",
    "Operator decisions (transmission outages, load, hydro dispatch) are not modelled.",
    "Solar has about 2.5 years of history, wind about 5 years.",
    "Quality degrades with the horizon and with the age of the latest data.",
]

HourlyRow = tuple[str, str, datetime, float, float, bool]


@dataclass(frozen=True)
class Payload:
    model: dict[str, Any]
    origin: datetime
    data_through: date
    hourly: list[HourlyRow]


def results_path() -> Path:
    base = Path(os.environ.get("SPILLWAY_REPORTS_DIR", config.REPO_ROOT / "reports"))
    return base / "results.json"


def model_row(
    results: dict[str, Any],
    n_series: int,
    n_rows: int,
    series_per_source: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Map ``reports/results.json`` (written by training) to the ``forecast_model`` columns."""
    trained_at = datetime.fromisoformat(results["trained_at"])
    return {
        "mlflow_run_id": results["run_id"],
        "model_version": f"hgb-{trained_at:%Y%m%d}-{results['run_id'][:8]}",
        "trained_at": trained_at,
        "data_period": {
            source: {
                **period,
                "n_series": (series_per_source or {}).get(source, period["n_series"]),
            }
            for source, period in results["period_by_source"].items()
        },
        "n_series": n_series,
        "n_rows": n_rows,
        "metrics": {
            "test": results["test"],
            "test_by_source": results["test_by_source"],
            "test_by_horizon_day": results["test_by_horizon_day"],
            "test_bootstrap_vs_baseline": results["test_bootstrap"],
            "cv_mean_std": results["cv"],
            "test_start": results["data"]["test_start"],
            "test_rows": results["data"]["test_rows"],
            "test_base_rate": results["test_base_rate"],
            "models": {
                "climatology": "mean by series, month, weekday/weekend and hour",
                "seasonal_naive": "same hour one week earlier",
                "hist_gbm": "gradient boosting (classifier + regressor)",
            },
        },
        "calibration": {"bins": results["test_calibration"], "n_bins": 10},
        "feature_importance": results["permutation_importance_brier"],
        "model_card": {
            "protocol": (
                "Rolling-origin validation (5 expanding folds of 3 months, 7-day embargo) and an "
                f"untouched final test from {results['data']['test_start']}."
            ),
            "risk_definition": f"hour with curtailment >= {config.RISK_THRESHOLD:.0%}",
            "data_source": results.get("data_source"),
            "limitations": LIMITATIONS,
        },
    }


def build_payload(service: ForecastService, results: dict[str, Any]) -> Payload:
    """Score all series from one common origin: the day after the latest observed day."""
    data_through = max(s.last_day for s in service.series.values())
    origin = (pd.Timestamp(data_through) + pd.Timedelta(days=1)).to_pydatetime()
    hourly: list[HourlyRow] = []
    for series in service.series.values():
        out = service.forecast(series.scope, series.source, 7, origin=origin.date())
        for p in out["points"]:
            hourly.append(
                (
                    series.source,
                    series.scope,
                    datetime.fromisoformat(p["ts"]),
                    p["risk_probability"],
                    p["expected_curtailment_pct"],
                    p["applicable"],
                )
            )
    return Payload(
        model=model_row(
            results,
            len(service.series),
            int(results["data"]["rows"]),
            dict(Counter(s.source for s in service.series.values())),
        ),
        origin=origin,
        data_through=data_through,
        hourly=hourly,
    )


def write(dsn: str, payload: Payload, keep_runs: int = DEFAULT_KEEP_RUNS) -> dict[str, Any]:
    """Write model, run and hourly rows in a single transaction. Returns what happened."""
    import psycopg
    from psycopg.types.json import Jsonb

    m = payload.model
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:  # commits on clean exit
        cur.execute("select id from forecast_model where mlflow_run_id = %s", (m["mlflow_run_id"],))
        found = cur.fetchone()
        if found:
            model_id = found[0]
        else:
            cur.execute(
                "insert into forecast_model (mlflow_run_id, model_version, trained_at, "
                "data_period, n_series, n_rows, metrics, calibration, feature_importance, "
                "model_card, is_current) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,false) "
                "returning id",
                (
                    m["mlflow_run_id"],
                    m["model_version"],
                    m["trained_at"],
                    Jsonb(m["data_period"]),
                    m["n_series"],
                    m["n_rows"],
                    Jsonb(m["metrics"]),
                    Jsonb(m["calibration"]),
                    Jsonb(m["feature_importance"]),
                    Jsonb(m["model_card"]),
                ),
            )
            row = cur.fetchone()
            assert row is not None
            model_id = row[0]
        cur.execute(
            "select id, status from forecast_run where model_id = %s and origin_ts = %s",
            (model_id, payload.origin),
        )
        existing = cur.fetchone()
        if existing and existing[1] == "ready":
            return {"status": "already-published", "run_id": existing[0], "model_id": model_id}
        if existing:  # a leftover ``loading`` run (cannot survive a transaction normally)
            cur.execute("delete from forecast_run where id = %s", (existing[0],))
        cur.execute(
            "insert into forecast_run (model_id, origin_ts, data_through, horizon_hours, status) "
            "values (%s,%s,%s,%s,'loading') returning id",
            (model_id, payload.origin, payload.data_through, config.HORIZON_HOURS),
        )
        run_row = cur.fetchone()
        assert run_row is not None
        run_id = run_row[0]
        with cur.copy(
            "copy forecast_hourly (run_id, source, scope, hour_ts, risk_probability, "
            "expected_curtailment_pct, applicable) from stdin"
        ) as copy:
            for source, scope, ts, risk, pct, ok in payload.hourly:
                copy.write_row((run_id, source, scope, ts, risk, pct, ok))
        cur.execute("update forecast_run set status = 'ready' where id = %s", (run_id,))
        cur.execute("update forecast_model set is_current = false where is_current")
        cur.execute("update forecast_model set is_current = true where id = %s", (model_id,))
        cur.execute(
            "delete from forecast_run where id not in "
            "(select id from forecast_run order by origin_ts desc, id desc limit %s)",
            (keep_runs,),
        )
        purged = cur.rowcount
    return {
        "status": "published",
        "run_id": run_id,
        "model_id": model_id,
        "rows": len(payload.hourly),
        "purged_runs": purged,
    }


def publish(dsn: str | None = None, keep_runs: int = DEFAULT_KEEP_RUNS) -> dict[str, Any]:
    dsn = dsn or os.environ.get("PUBLISH_DSN") or os.environ.get("WAREHOUSE_DSN")
    if not dsn:
        raise SystemExit("pass --dsn or set PUBLISH_DSN (role energia_forecast in production)")
    results = json.loads(results_path().read_text(encoding="utf-8"))
    service = ForecastService.from_disk()
    result = write(dsn, build_payload(service, results), keep_runs)
    result["published_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return result
