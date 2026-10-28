"""Silver-layer ingestion and the warehouse publisher (synthetic frames, fake S3 client)."""

from __future__ import annotations

import io
import os
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import polars as pl
import pytest
from botocore.exceptions import ClientError

from spillway_forecast.features import build_dataset, to_series
from spillway_forecast.model import ForecastModel
from spillway_forecast.publish import Payload, build_payload, model_row, write
from spillway_forecast.service import ForecastService
from spillway_forecast.silver import aggregate_silver, read_silver_hourly, silver_key


def silver_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "measured_at": [
                datetime(2026, 1, 1, 0, 0),
                datetime(2026, 1, 1, 0, 30),
                datetime(2026, 1, 1, 0, 0),
            ],
            "state_id": ["RN", "RN", "BA"],
            "subsystem_id": ["NE", "NE", "NE"],
            "curtailed_mwh": [50.0, 0.0, float("nan")],
            "generation_mwh": [100.0, 100.0, 25.0],
        }
    )


def test_aggregate_silver_uses_pipeline_energy_columns() -> None:
    out = aggregate_silver(silver_frame(), "wind")
    rn = out.filter((pl.col("scope") == "RN") & (pl.col("scope_type") == "state")).row(
        0, named=True
    )
    assert rn["curtailed_mwh"] == pytest.approx(50.0)  # already MWh in silver: no x0.5 here
    assert rn["generation_mwh"] == pytest.approx(200.0)
    ne = out.filter(pl.col("scope") == "NE").row(0, named=True)
    assert ne["generation_mwh"] == pytest.approx(225.0)
    assert out.filter(pl.col("scope") == "SIN").height == 1
    assert set(out["scope_type"]) == {"state", "subsystem", "sin"}
    assert out["ts"].dt.hour().max() == 0  # truncated to the hour


class FakeS3:
    """Serves one month of wind silver; everything else is 'not in the lake'."""

    def __init__(self, key: str, frame: pl.DataFrame) -> None:
        buffer = io.BytesIO()
        frame.write_parquet(buffer)
        self.objects = {key: buffer.getvalue()}

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "x"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}


def test_read_silver_hourly_skips_missing_months() -> None:
    client = FakeS3(silver_key("wind", 2026, 1), silver_frame())
    out = read_silver_hourly(date(2026, 2, 15), client=client, bucket="b")
    assert set(out["source"]) == {"wind"}
    assert out.height == 4  # states RN and BA, subsystem NE and SIN at hour 00:00


def test_read_silver_hourly_fails_without_any_file() -> None:
    with pytest.raises(FileNotFoundError):
        read_silver_hourly(date(2026, 2, 15), client=FakeS3("x", silver_frame()), bucket="b")


@pytest.fixture(scope="module")
def trained(hourly: pd.DataFrame) -> tuple[ForecastService, Any]:
    series = to_series(hourly)
    data = build_dataset(series, 3)
    model = ForecastModel(series_ids=[s.series_id for s in series]).fit(
        data.sample(min(20000, len(data)), random_state=0)
    )
    return ForecastService.from_frames(model, hourly), model


def results_fixture() -> dict[str, Any]:
    metrics = {"mae_pp": 10.0, "roc_auc": 0.9, "brier": 0.1, "ece": 0.04}
    return {
        "run_id": "abcdef1234567890",
        "trained_at": datetime(2026, 10, 28, 12, tzinfo=UTC).isoformat(),
        "data_source": "silver",
        "period_by_source": {"wind": {"start": "2021-10-01", "end": "2026-09-30", "n_series": 3}},
        "test": {"hist_gbm": metrics, "climatology": metrics, "seasonal_naive": metrics},
        "test_by_source": {},
        "test_by_horizon_day": {},
        "test_bootstrap": {},
        "cv": {},
        "test_base_rate": 0.4,
        "test_calibration": {"hist_gbm": [{"bin": 0, "n": 1, "mean_pred": 0.1, "obs": 0.1}]},
        "permutation_importance_brier": {"hod28": 0.1},
        "data": {"test_start": "2026-04-01", "test_rows": 10, "rows": 100},
    }


def test_model_row_maps_training_results() -> None:
    row = model_row(results_fixture(), n_series=3, n_rows=100)
    assert row["mlflow_run_id"] == "abcdef1234567890"
    assert row["model_version"] == "hgb-20261028-abcdef12"
    assert row["metrics"]["test"]["hist_gbm"]["mae_pp"] == 10.0
    assert row["calibration"]["bins"]["hist_gbm"][0]["obs"] == 0.1
    assert row["model_card"]["limitations"]
    assert row["feature_importance"] == {"hod28": 0.1}


def test_build_payload_uses_one_common_origin(trained: tuple[ForecastService, Any]) -> None:
    service = trained[0]
    payload = build_payload(service, results_fixture())
    assert payload.origin.hour == 0
    assert payload.data_through < payload.origin.date()
    series_count = len(service.series)
    assert len(payload.hourly) == series_count * 168
    assert {h[0] for h in payload.hourly} <= {"wind", "solar"}
    assert all(0.0 <= h[3] <= 1.0 for h in payload.hourly)
    assert all(h[2] >= payload.origin for h in payload.hourly)


@pytest.mark.skipif(not os.environ.get("PUBLISH_TEST_DSN"), reason="needs a throwaway Postgres")
def test_publish_is_transactional_and_idempotent(trained: tuple[ForecastService, Any]) -> None:
    import psycopg

    dsn = os.environ["PUBLISH_TEST_DSN"]
    payload = build_payload(trained[0], results_fixture())
    first = write(dsn, payload, keep_runs=2)
    again = write(dsn, payload, keep_runs=2)
    assert first["status"] == "published" and again["status"] == "already-published"
    assert again["run_id"] == first["run_id"]
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("select count(*) from forecast_run where status = 'ready'")
        assert cur.fetchone() == (1,)
        cur.execute("select count(*) from forecast_model where is_current")
        assert cur.fetchone() == (1,)
        cur.execute("select count(*) from forecast_hourly")
        assert cur.fetchone() == (len(payload.hourly),)
        # cleanup so the test can be re-run against the same database
        cur.execute("delete from forecast_run")
        cur.execute("delete from forecast_model")
        conn.commit()


def test_payload_dataclass_is_frozen() -> None:
    payload = Payload(
        model={}, origin=datetime(2026, 1, 1), data_through=date(2025, 12, 31), hourly=[]
    )
    with pytest.raises(AttributeError):
        payload.origin = datetime(2026, 1, 2)  # type: ignore[misc]
