"""Leak-free feature construction for a 7-day-ahead hourly forecast.

Forecast design
---------------
A forecast is *issued* at 00:00 of an **origin day** ``d`` and covers the next 168 hours
(origin day plus six more days). Every input feature of a row is computed only from days
strictly before ``d`` (``shift(1)`` on all daily/rolling aggregates). Only deterministic
calendar attributes of the *target* hour (hour, weekday, month, holiday) are allowed to
look "into the future", because they are known in advance.

The "same hour last week" value for target day ``d + k`` (``k <= 6``) lives on day
``d + k - 7 <= d - 1``, so it is also available at issue time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import holidays
import numpy as np
import pandas as pd

from spillway_forecast.config import (
    HORIZON_HOURS,
    MIN_DENOMINATOR_MWH,
    MIN_SERIES_HOURS,
    RISK_THRESHOLD,
)

PAD_DAYS = 8  # NaN days appended so that an origin right after the data end can be built

NUMERIC_FEATURES = [
    "horizon_h",
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "doy_sin",
    "doy_cos",
    "r1",
    "r7",
    "r28",
    "r90",
    "r1_max",
    "r7_trend",
    "gen_d1",
    "gen7",
    "gen28",
    "gen_ratio",
    "hod_last",
    "hod7",
    "hod28",
    "hodgen7",
    "snaive_7d",
]
CATEGORICAL_FEATURES = ["series_id", "source"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class SeriesData:
    """Dense day x hour matrices for one series."""

    series_id: str
    scope: str
    source: str
    first_day: date
    curtailed: np.ndarray  # (days, 24), NaN where the hour is missing
    generation: np.ndarray  # (days, 24)

    @property
    def n_days(self) -> int:
        return int(self.curtailed.shape[0])

    @property
    def last_day(self) -> date:
        return (pd.Timestamp(self.first_day) + pd.Timedelta(days=self.n_days - 1)).date()


def make_series_id(scope: str, source: str) -> str:
    return f"{source}:{scope}"


def to_series(hourly: pd.DataFrame) -> list[SeriesData]:
    """Pivot a long hourly frame (ts, scope, source, curtailed_mwh, generation_mwh)."""
    out: list[SeriesData] = []
    for (scope, source), grp in hourly.groupby(["scope", "source"], sort=True):
        grp = grp.sort_values("ts")
        if len(grp) < MIN_SERIES_HOURS:
            continue
        first = grp["ts"].min().normalize()
        last = grp["ts"].max().normalize()
        index = pd.date_range(first, last + pd.Timedelta(hours=23), freq="h")
        grp = grp.set_index("ts").reindex(index)
        out.append(
            SeriesData(
                series_id=make_series_id(str(scope), str(source)),
                scope=str(scope),
                source=str(source),
                first_day=first.date(),
                curtailed=grp["curtailed_mwh"].to_numpy(dtype=float).reshape(-1, 24),
                generation=grp["generation_mwh"].to_numpy(dtype=float).reshape(-1, 24),
            )
        )
    return out


def _nan_roll_mean(frame: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=min_periods).mean()


def _ratio(curt: pd.Series, denom: pd.Series) -> pd.Series:
    return curt / denom.where(denom >= MIN_DENOMINATOR_MWH)


def _brazil_holidays(years: Sequence[int]) -> set[date]:
    return set(holidays.country_holidays("BR", years=list(years)).keys())


def _origin_features(series: SeriesData) -> dict[str, np.ndarray]:
    """Per-origin-day feature arrays. Row ``d`` only uses days ``<= d - 1``."""
    pad = np.full((PAD_DAYS, 24), np.nan)
    curt = pd.DataFrame(np.vstack([series.curtailed, pad]))
    gen = pd.DataFrame(np.vstack([series.generation, pad]))
    denom = curt + gen
    pct = curt / denom.where(denom >= MIN_DENOMINATOR_MWH)

    day_curt = curt.sum(axis=1, min_count=1)
    day_denom = denom.sum(axis=1, min_count=1)
    day_gen = gen.sum(axis=1, min_count=1)

    def window_ratio(w: int, min_p: int) -> pd.Series:
        return _ratio(
            day_curt.rolling(w, min_periods=min_p).sum(),
            day_denom.rolling(w, min_periods=min_p).sum(),
        )

    # All of the following are shifted by one day: row d describes days <= d-1.
    r1 = _ratio(day_curt, day_denom).shift(1)
    r7 = window_ratio(7, 3).shift(1)
    r28 = window_ratio(28, 10).shift(1)
    r90 = window_ratio(90, 30).shift(1)
    r1_max = pct.max(axis=1).shift(1)
    gen_d1 = (day_gen / 24.0).shift(1)
    gen7 = (day_gen.rolling(7, min_periods=3).mean() / 24.0).shift(1)
    gen28 = (day_gen.rolling(28, min_periods=10).mean() / 24.0).shift(1)

    pct_prev = pct.shift(1)
    return {
        "r1": r1.to_numpy(),
        "r7": r7.to_numpy(),
        "r28": r28.to_numpy(),
        "r90": r90.to_numpy(),
        "r1_max": r1_max.to_numpy(),
        "r7_trend": (r7 - r28).to_numpy(),
        "gen_d1": gen_d1.to_numpy(),
        "gen7": gen7.to_numpy(),
        "gen28": gen28.to_numpy(),
        "gen_ratio": (gen7 / gen28.where(gen28 > 0)).to_numpy(),
        "hod_last": pct_prev.to_numpy(),
        "hod7": _nan_roll_mean(pct_prev, 7, 3).to_numpy(),
        "hod28": _nan_roll_mean(pct_prev, 28, 10).to_numpy(),
        "hodgen7": _nan_roll_mean(gen.shift(1), 7, 3).to_numpy(),
        "pct": pct.to_numpy(),
    }


def build_rows(
    series: SeriesData,
    origins: Sequence[date],
    *,
    with_target: bool = True,
) -> pd.DataFrame:
    """Build the (origin x horizon) feature table of one series.

    ``origins`` are the days at 00:00 of which forecasts are issued. They must lie within
    ``[first_day + 1, last_day + 1]`` (``last_day + 1`` is the operational forecast origin).
    """
    feats = _origin_features(series)
    day0 = pd.Timestamp(series.first_day)
    o_idx = np.array([(pd.Timestamp(o) - day0).days for o in origins], dtype=int)
    if len(o_idx) == 0:
        return pd.DataFrame(columns=["series_id", *FEATURES])
    if o_idx.min() < 1 or o_idx.max() > series.n_days:
        raise ValueError("origin outside the available range of the series")

    j = np.arange(HORIZON_HOURS)
    hr = j % 24
    k = j // 24
    n_o = len(o_idx)
    day_of_target = o_idx[:, None] + k[None, :]  # (n_o, 168)
    hour_grid = np.broadcast_to(hr, (n_o, HORIZON_HOURS))

    pct = feats["pct"]
    n_total = pct.shape[0]

    def gather(name: str) -> np.ndarray:
        return np.asarray(feats[name][o_idx[:, None], hour_grid])

    def broadcast(name: str) -> np.ndarray:
        return np.broadcast_to(feats[name][o_idx][:, None], (n_o, HORIZON_HOURS))

    prev_week = day_of_target - 7
    snaive = np.where(prev_week >= 0, pct[np.clip(prev_week, 0, n_total - 1), hour_grid], np.nan)
    if with_target:
        in_range = day_of_target < n_total
        target = np.where(in_range, pct[np.clip(day_of_target, 0, n_total - 1), hour_grid], np.nan)
    else:
        target = np.full((n_o, HORIZON_HOURS), np.nan)

    target_ts = (
        day0
        + pd.to_timedelta(day_of_target.ravel(), unit="D")
        + pd.to_timedelta(hour_grid.ravel(), unit="h")
    )
    origin_ts = day0 + pd.to_timedelta(np.repeat(o_idx, HORIZON_HOURS), unit="D")
    target_days = pd.DatetimeIndex(target_ts).normalize()
    hol = _brazil_holidays(sorted({int(y) for y in target_days.year.unique()}))
    is_holiday = np.array([d.date() in hol for d in target_days], dtype=float)
    doy = np.asarray(target_ts.day_of_year, dtype=float)
    dow = np.asarray(target_ts.day_of_week, dtype=float)

    frame = pd.DataFrame(
        {
            "series_id": series.series_id,
            "scope": series.scope,
            "source": series.source,
            "origin": origin_ts,
            "target_ts": target_ts,
            "horizon_h": np.tile(j + 1, n_o).astype(float),
            "hour": hour_grid.ravel().astype(float),
            "dow": dow,
            "month": np.asarray(target_ts.month, dtype=float),
            "is_weekend": (dow >= 5).astype(float),
            "is_holiday": is_holiday,
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "r1": broadcast("r1").ravel(),
            "r7": broadcast("r7").ravel(),
            "r28": broadcast("r28").ravel(),
            "r90": broadcast("r90").ravel(),
            "r1_max": broadcast("r1_max").ravel(),
            "r7_trend": broadcast("r7_trend").ravel(),
            "gen_d1": broadcast("gen_d1").ravel(),
            "gen7": broadcast("gen7").ravel(),
            "gen28": broadcast("gen28").ravel(),
            "gen_ratio": broadcast("gen_ratio").ravel(),
            "hod_last": gather("hod_last").ravel(),
            "hod7": gather("hod7").ravel(),
            "hod28": gather("hod28").ravel(),
            "hodgen7": gather("hodgen7").ravel(),
            "snaive_7d": snaive.ravel(),
            "pct": target.ravel(),
        }
    )
    frame["risk"] = np.where(frame["pct"].isna(), np.nan, frame["pct"] >= RISK_THRESHOLD)
    return frame


def default_origins(series: SeriesData, stride_days: int, *, up_to_last: bool = True) -> list[date]:
    """Origins every ``stride_days`` days, anchored on the calendar (not on series start)."""
    start = pd.Timestamp(series.first_day) + pd.Timedelta(days=90)  # allow r90 warm-up
    end = pd.Timestamp(series.last_day) + pd.Timedelta(days=0 if up_to_last else 1)
    days = pd.date_range(start, end, freq="D")
    epoch = pd.Timestamp("2020-01-01")
    return [d.date() for d in days if ((d - epoch).days % stride_days) == 0]


def build_dataset(
    series_list: Sequence[SeriesData], stride_days: int, *, min_origin: date | None = None
) -> pd.DataFrame:
    frames = []
    for s in series_list:
        origins = [
            o for o in default_origins(s, stride_days) if min_origin is None or o >= min_origin
        ]
        frames.append(build_rows(s, origins))
    data = pd.concat(frames, ignore_index=True)
    data = data[data["pct"].notna()].reset_index(drop=True)
    return data


def as_model_frame(data: pd.DataFrame, series_categories: Sequence[str]) -> pd.DataFrame:
    """Select model features with fixed categorical vocabularies (train/serve parity)."""
    x = data[FEATURES].copy()
    x["series_id"] = pd.Categorical(x["series_id"], categories=list(series_categories))
    x["source"] = pd.Categorical(x["source"], categories=["wind", "solar"])
    return x
