"""FastAPI surface. ``create_router`` can be mounted inside another app (e.g. energia-api)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query
from pydantic import BaseModel

from spillway_forecast.service import MAX_DAYS, ForecastService, UnknownSeriesError


class ForecastPoint(BaseModel):
    ts: str
    risk_probability: float
    expected_curtailment_pct: float
    level: Literal["baixo", "medio", "alto"]
    applicable: bool


class ForecastResponse(BaseModel):
    scope: str
    source: Literal["wind", "solar"]
    risk_definition: str
    origin: str
    data_through: str
    stale_days: int
    model_trained_through: str | None
    points: list[ForecastPoint]


class ScopeInfo(BaseModel):
    scope: str
    source: Literal["wind", "solar"]
    last_observation: str


@lru_cache(maxsize=1)
def default_service() -> ForecastService:
    return ForecastService.from_disk()


def create_router(get_service: Callable[[], ForecastService] = default_service) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["forecast"])

    @router.get("/forecast", response_model=ForecastResponse)
    def forecast(
        scope: Annotated[
            str, Query(min_length=1, max_length=3, description="State, subsystem or SIN")
        ],
        source: Literal["wind", "solar"] = "wind",
        days: Annotated[int, Query(ge=1, le=MAX_DAYS)] = 7,
    ) -> dict[str, object]:
        try:
            service = get_service()
        except FileNotFoundError as exc:
            raise HTTPException(503, "model or history not available; run training/ingest") from exc
        try:
            return service.forecast(scope, source, days)
        except UnknownSeriesError as exc:
            raise HTTPException(
                404, f"no forecast series for scope={scope} source={source}"
            ) from exc

    @router.get("/scopes", response_model=list[ScopeInfo])
    def scopes() -> list[dict[str, str]]:
        try:
            return get_service().scopes()
        except FileNotFoundError as exc:
            raise HTTPException(503, "model or history not available") from exc

    @router.get("/model")
    def model_info() -> dict[str, object]:
        """Training results (same content the publisher writes to ``forecast_model``)."""
        import json

        from spillway_forecast.publish import model_row, results_path

        try:
            results = json.loads(results_path().read_text(encoding="utf-8"))
            service = get_service()
        except FileNotFoundError as exc:
            raise HTTPException(503, "no training results available; run training") from exc
        per_source = dict(Counter(x.source for x in service.series.values()))
        row = model_row(results, len(service.series), int(results["data"]["rows"]), per_source)
        row["trained_at"] = row["trained_at"].isoformat()
        return row

    return router


def create_app(get_service: Callable[[], ForecastService] = default_service) -> FastAPI:
    app = FastAPI(title="spillway-forecast", version="0.1.0")
    app.include_router(create_router(get_service))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
