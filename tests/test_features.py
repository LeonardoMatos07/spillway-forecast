from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from spillway_forecast.config import HORIZON_HOURS
from spillway_forecast.features import FEATURES, build_rows, to_series

# Calendar attributes of the *target* hour are deterministic and legitimately "future".
CALENDAR = {"horizon_h", "hour", "dow", "month", "is_weekend", "is_holiday", "doy_sin", "doy_cos"}
HISTORY_FEATURES = [f for f in FEATURES if f not in CALENDAR and f not in {"series_id", "source"}]


def _series(hourly: pd.DataFrame, scope: str = "NE", source: str = "wind"):  # type: ignore[no-untyped-def]
    return next(s for s in to_series(hourly) if s.scope == scope and s.source == source)


def _corrupt_future(hourly: pd.DataFrame, origin: date) -> pd.DataFrame:
    """Replace every observation at or after the origin with garbage."""
    out = hourly.copy()
    future = out["ts"] >= pd.Timestamp(origin)
    rng = np.random.default_rng(123)
    out.loc[future, "curtailed_mwh"] = rng.uniform(0, 5000, future.sum())
    out.loc[future, "generation_mwh"] = rng.uniform(0, 5000, future.sum())
    return out


def test_history_features_do_not_depend_on_data_at_or_after_origin(hourly: pd.DataFrame) -> None:
    origin = date(2024, 9, 15)
    clean = build_rows(_series(hourly), [origin])
    dirty = build_rows(_series(_corrupt_future(hourly, origin)), [origin])
    pd.testing.assert_frame_equal(clean[HISTORY_FEATURES], dirty[HISTORY_FEATURES])
    assert len(clean) == HORIZON_HOURS


def test_leak_check_has_power(hourly: pd.DataFrame) -> None:
    """Changing the day BEFORE the origin must change features, otherwise the test is vacuous."""
    origin = date(2024, 9, 15)
    changed = hourly.copy()
    mask = (changed["ts"] >= pd.Timestamp("2024-09-14")) & (changed["ts"] < pd.Timestamp(origin))
    changed.loc[mask, "curtailed_mwh"] += 1000.0
    a = build_rows(_series(hourly), [origin])
    b = build_rows(_series(changed), [origin])
    assert not np.allclose(a["r1"], b["r1"])


def test_target_is_the_future_and_features_are_not(hourly: pd.DataFrame) -> None:
    series = _series(hourly)
    origin = date(2024, 9, 15)
    rows = build_rows(series, [origin])
    assert (rows["target_ts"] >= pd.Timestamp(origin)).all()
    assert rows["target_ts"].min() == pd.Timestamp(origin)
    assert rows["target_ts"].max() == pd.Timestamp(origin) + pd.Timedelta(hours=HORIZON_HOURS - 1)
    # target equals the realised value of the hourly table
    sub = hourly[(hourly["scope"] == "NE") & (hourly["source"] == "wind")].set_index("ts")
    sample = rows.iloc[100]
    row = sub.loc[sample["target_ts"]]
    expected = row["curtailed_mwh"] / (row["curtailed_mwh"] + row["generation_mwh"])
    assert np.isclose(sample["pct"], expected)


def test_seasonal_naive_feature_is_exactly_one_week_earlier(hourly: pd.DataFrame) -> None:
    series = _series(hourly)
    rows = build_rows(series, [date(2024, 9, 15)])
    sub = hourly[(hourly["scope"] == "NE") & (hourly["source"] == "wind")].set_index("ts")
    for i in (0, 25, 167):
        ts = rows.iloc[i]["target_ts"] - pd.Timedelta(days=7)
        row = sub.loc[ts]
        expected = row["curtailed_mwh"] / (row["curtailed_mwh"] + row["generation_mwh"])
        assert np.isclose(rows.iloc[i]["snaive_7d"], expected)
        assert ts < pd.Timestamp("2024-09-15")  # available at issue time


def test_operational_origin_after_last_observation(hourly: pd.DataFrame) -> None:
    series = _series(hourly)
    origin = date.fromordinal(series.last_day.toordinal() + 1)
    rows = build_rows(series, [origin], with_target=False)
    assert len(rows) == HORIZON_HOURS
    assert rows["pct"].isna().all()
    assert rows["r7"].notna().all()
