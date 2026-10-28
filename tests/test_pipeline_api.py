from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from fastapi.testclient import TestClient

from spillway_forecast.api import create_app
from spillway_forecast.config import Protocol
from spillway_forecast.features import build_dataset, to_series
from spillway_forecast.ingest import aggregate_raw
from spillway_forecast.model import ForecastModel
from spillway_forecast.service import ForecastService, risk_level


def test_aggregate_raw_units_and_clipping() -> None:
    raw = pl.DataFrame(
        {
            "din_instante": ["2026-01-01 00:00:00", "2026-01-01 00:30:00", "2026-01-01 00:00:00"],
            "id_estado": ["RN", "RN", "BA"],
            "id_subsistema": ["NE", "NE", "NE"],
            "val_geracaonaorealizadaapurada": [100.0, -5.0, float("nan")],
            "val_geracao": [200.0, 200.0, 50.0],
        }
    ).with_columns(pl.col("din_instante").str.to_datetime())
    out = aggregate_raw(raw, "wind")
    rn = out.filter((pl.col("scope") == "RN") & (pl.col("scope_type") == "state")).row(
        0, named=True
    )
    assert rn["curtailed_mwh"] == pytest.approx(50.0)  # (100 + 0) MW * 0.5 h
    assert rn["generation_mwh"] == pytest.approx(200.0)
    ne = out.filter(pl.col("scope") == "NE").row(0, named=True)
    assert ne["generation_mwh"] == pytest.approx(225.0)
    assert set(out["scope_type"]) == {"state", "subsystem", "sin"}


@pytest.fixture(scope="module")
def service(hourly):  # type: ignore[no-untyped-def]
    series = to_series(hourly)
    data = build_dataset(series, 3)
    model = ForecastModel(series_ids=[s.series_id for s in series]).fit(
        data.sample(min(20000, len(data)), random_state=0)
    )
    return ForecastService.from_frames(model, hourly)


def test_api_forecast_contract(service: ForecastService) -> None:
    client = TestClient(create_app(lambda: service))
    r = client.get("/v1/forecast", params={"scope": "NE", "source": "wind", "days": 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["points"]) == 48
    assert all(0 <= p["risk_probability"] <= 1 for p in body["points"])
    assert body["points"][0]["ts"] > body["data_through"]


def test_api_solar_night_is_not_applicable(service: ForecastService) -> None:
    body = service.forecast("NE", "solar", days=1)
    night = [p for p in body["points"] if p["ts"][11:13] in {"01", "02", "23"}]
    assert night and all(not p["applicable"] and p["risk_probability"] == 0 for p in night)


def test_api_errors(service: ForecastService) -> None:
    client = TestClient(create_app(lambda: service))
    assert client.get("/v1/forecast", params={"scope": "ZZ"}).status_code == 404
    assert client.get("/v1/forecast", params={"scope": "NE", "days": 9}).status_code == 422
    assert client.get("/v1/forecast", params={"scope": "NE", "source": "coal"}).status_code == 422
    assert client.get("/health").json() == {"status": "ok"}
    assert len(client.get("/v1/scopes").json()) == 3


def test_api_503_without_model() -> None:
    def missing() -> ForecastService:
        raise FileNotFoundError

    assert (
        TestClient(create_app(missing)).get("/v1/forecast", params={"scope": "NE"}).status_code
        == 503
    )


def test_risk_level_bands() -> None:
    assert [risk_level(p) for p in (0.1, 0.5, 0.9)] == ["baixo", "medio", "alto"]


def test_train_end_to_end_smoke(
    hourly: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spillway_forecast.train as train_mod

    monkeypatch.setenv("SPILLWAY_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(train_mod, "REPORTS", tmp_path / "reports")
    protocol = Protocol(
        test_start=date(2025, 1, 15),
        fold_starts=(date(2024, 10, 1), date(2024, 11, 15)),
        fold_months=1,
        origin_stride_days=7,
    )
    results = train_mod.train(protocol, register=False, hourly=hourly)
    assert set(results["test"]) == {"climatology", "seasonal_naive", "hist_gbm"}
    assert (tmp_path / "models" / "bundle.joblib").exists()
    assert (tmp_path / "reports" / "results.json").exists()
