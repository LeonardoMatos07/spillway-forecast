"""Inference service: turns a fitted model plus recent history into hourly risk forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spillway_forecast import config
from spillway_forecast.config import HORIZON_HOURS, RISK_THRESHOLD
from spillway_forecast.features import SeriesData, build_rows, make_series_id, to_series
from spillway_forecast.model import ForecastModel

MAX_DAYS = HORIZON_HOURS // 24
# Probability cut-offs for the human-readable level. They are NOT calibrated decision
# thresholds; see MODEL_CARD.md.
LEVEL_BANDS = ((0.66, "alto"), (0.33, "medio"))
MIN_EXPECTED_MWH = 0.5  # solar hours with less typical generation are "not applicable"


class UnknownSeriesError(LookupError):
    pass


def risk_level(p: float) -> str:
    for cut, label in LEVEL_BANDS:
        if p >= cut:
            return label
    return "baixo"


@dataclass
class ForecastService:
    model: ForecastModel
    series: dict[str, SeriesData]
    data_end: datetime

    @staticmethod
    def from_disk(
        model_path: Path | None = None, hourly_path: Path | None = None
    ) -> ForecastService:
        model = ForecastModel.load(model_path)
        hourly = pd.read_parquet(hourly_path or config.hourly_path())
        return ForecastService.from_frames(model, hourly)

    @staticmethod
    def from_frames(model: ForecastModel, hourly: pd.DataFrame) -> ForecastService:
        series = {s.series_id: s for s in to_series(hourly) if s.series_id in model.series_ids}
        return ForecastService(
            model=model, series=series, data_end=hourly["ts"].max().to_pydatetime()
        )

    def scopes(self) -> list[dict[str, str]]:
        return [
            {"scope": s.scope, "source": s.source, "last_observation": str(s.last_day)}
            for s in self.series.values()
        ]

    def forecast(
        self, scope: str, source: str, days: int = 7, origin: date | None = None
    ) -> dict[str, Any]:
        if not 1 <= days <= MAX_DAYS:
            raise ValueError(f"days must be between 1 and {MAX_DAYS}")
        key = make_series_id(scope.upper(), source.lower())
        series = self.series.get(key)
        if series is None:
            raise UnknownSeriesError(key)
        # Operational origin: the day after the last day with data (features only use the past).
        issue = origin or (pd.Timestamp(series.last_day) + pd.Timedelta(days=1)).date()
        rows = build_rows(series, [issue], with_target=False).head(days * 24)
        pct, risk = self.model.predict(rows)
        expected_gen = rows["hodgen7"].to_numpy(dtype=float)
        applicable = ~(np.isnan(expected_gen) | (expected_gen < MIN_EXPECTED_MWH))
        points = []
        for ts, p, e, ok in zip(rows["target_ts"], risk, pct, applicable, strict=True):
            prob = float(p) if ok else 0.0
            points.append(
                {
                    "ts": ts.isoformat(),
                    "risk_probability": round(prob, 4),
                    "expected_curtailment_pct": round(float(e) * 100, 2) if ok else 0.0,
                    "level": risk_level(prob),
                    "applicable": bool(ok),
                }
            )
        return {
            "scope": series.scope,
            "source": series.source,
            "risk_definition": f"hour with curtailment >= {RISK_THRESHOLD:.0%} of potential output",
            "origin": pd.Timestamp(issue).isoformat(),
            "data_through": str(series.last_day),
            "stale_days": max(0, (date.today() - series.last_day).days),
            "model_trained_through": self.model.meta.get("trained_through"),
            "points": points,
        }
