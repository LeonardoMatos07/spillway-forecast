"""Leakage tests for the inflow training set: the future must never reach a feature."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from spillway_forecast.inflow_dataset import DatasetConfig, build_features, write_dataset

START, END = date(2001, 1, 1), date(2026, 3, 1)
CFG = DatasetConfig()


def synthetic(seed: int = 0, ids: tuple[str, ...] = ("AAA", "BBB")) -> tuple[pl.DataFrame, ...]:
    rng = np.random.default_rng(seed)
    days = [START + timedelta(days=i) for i in range((END - START).days + 1)]
    rows, rain_rows = [], []
    for rid in ids:
        base = 100 + 50 * np.sin(np.arange(len(days)) * 2 * np.pi / 365.25)
        q = base + rng.normal(0, 5, len(days))
        precip = rng.gamma(0.5, 6, len(days))
        for d, qv, p in zip(days, q, precip, strict=True):
            rows.append(
                {
                    "reservoir_id": rid,
                    "day": d,
                    "natural_inflow_m3s": float(qv),
                    "storage_pct": 50.0,
                }
            )
            rain_rows.append(
                {
                    "reservoir_id": rid,
                    "day": d,
                    "precip_mm": float(p),
                    "run": "final" if (END - d).days > 120 else "late",
                }
            )
    register = pl.DataFrame(
        {
            "reservoir_id": list(ids),
            "basin": ["B"] * len(ids),
            "subsystem": ["SE"] * len(ids),
            "useful_volume_hm3": [1000.0] * len(ids),
            "latitude": [-20.0] * len(ids),
            "longitude": [-46.0] * len(ids),
        }
    )
    return pl.DataFrame(rows), register, pl.DataFrame(rain_rows)


def perturb_after(df: pl.DataFrame, cutoff: date, column: str) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("day") > cutoff).then(pl.col(column) + 1000.0).otherwise(pl.col(column))
    )


def knock_out(daily: pl.DataFrame, first: date, days: int) -> pl.DataFrame:
    return daily.with_columns(
        pl.when(
            (pl.col("reservoir_id") == "AAA")
            & pl.col("day").is_between(first, first + timedelta(days=days - 1))
        )
        .then(None)
        .otherwise(pl.col("natural_inflow_m3s"))
        .alias("natural_inflow_m3s")
    )


def test_splits_and_purge() -> None:
    daily, register, rain = synthetic()
    out = build_features(daily, register, CFG, rain)
    assert set(out["split"]) == {"train", "valid", "test", "holdout"}
    by = out.group_by("split").agg(
        pl.col("origin_day").min().alias("lo"), pl.col("origin_day").max().alias("hi")
    )
    ranges = {r["split"]: (r["lo"], r["hi"]) for r in by.iter_rows(named=True)}
    assert ranges["train"][0] >= CFG.train_start
    # the 14-day window of the last origin of each split ends inside the split
    assert ranges["train"][1] + timedelta(days=14) <= CFG.train_end
    assert ranges["valid"][1] + timedelta(days=14) <= CFG.valid_end
    assert ranges["test"][1] + timedelta(days=14) <= CFG.test_end
    assert ranges["valid"][0] > CFG.train_end
    assert ranges["test"][0] > CFG.valid_end


def test_features_do_not_depend_on_the_future() -> None:
    daily, register, rain = synthetic()
    cutoff = date(2019, 6, 30)
    base = build_features(daily, register, CFG, rain)
    changed = build_features(
        perturb_after(daily, cutoff, "natural_inflow_m3s"),
        register,
        CFG,
        perturb_after(rain, cutoff, "precip_mm"),
    )
    feature_cols = [
        c for c in base.columns if not c.startswith("y_") and c not in ("q_mean_train", "split")
    ]
    a = base.filter(pl.col("origin_day") <= cutoff).select(feature_cols)
    b = changed.filter(pl.col("origin_day") <= cutoff).select(feature_cols)
    assert a.height == b.height > 1000
    assert a.equals(b)


def test_targets_are_the_flow_h_days_later() -> None:
    daily, register, rain = synthetic()
    out = build_features(daily, register, CFG, rain).filter(pl.col("reservoir_id") == "AAA")
    flows = {
        r["day"]: r["natural_inflow_m3s"]
        for r in daily.filter(pl.col("reservoir_id") == "AAA").iter_rows(named=True)
    }
    row = out.filter(pl.col("origin_day") == date(2010, 5, 5)).row(0, named=True)
    for h in (1, 7, 14):
        assert row[f"y_{h}"] == flows[date(2010, 5, 5) + timedelta(days=h)]
    assert row["q_lag0"] == flows[date(2010, 5, 5)]
    assert row["q_lag1"] == flows[date(2010, 5, 4)]


def test_long_term_mean_uses_only_training_years() -> None:
    daily, register, rain = synthetic()
    base = build_features(daily, register, CFG, rain)
    tripled = daily.with_columns(
        pl.when(pl.col("day") > CFG.train_end)
        .then(pl.col("natural_inflow_m3s") * 3)
        .otherwise(pl.col("natural_inflow_m3s"))
        .alias("natural_inflow_m3s")
    )
    other = build_features(tripled, register, CFG, rain)
    a = base.group_by("reservoir_id").agg(pl.col("q_mean_train").first()).sort("reservoir_id")
    b = other.group_by("reservoir_id").agg(pl.col("q_mean_train").first()).sort("reservoir_id")
    assert a.equals(b)


def test_rain_is_lagged_by_the_publication_delay() -> None:
    daily, register, rain = synthetic()
    lag = CFG.rain_lag_days
    origin = date(2012, 7, 15)
    base = build_features(daily, register, CFG, rain)
    changed = build_features(
        daily, register, CFG, perturb_after(rain, origin - timedelta(days=lag), "precip_mm")
    )
    cols = ["rain_sum7", "rain_sum30", "rain_sum90"]
    one = pl.col("origin_day") == origin
    assert base.filter(one).select(cols).equals(changed.filter(one).select(cols))
    # and the rain exactly at origin - lag is the newest one that is used
    seen = rain.filter(pl.col("reservoir_id") == "AAA")
    window = seen.filter(
        (pl.col("day") <= origin - timedelta(days=lag))
        & (pl.col("day") > origin - timedelta(days=lag + 7))
    )["precip_mm"].sum()
    got = base.filter(one & (pl.col("reservoir_id") == "AAA"))["rain_sum7"][0]
    assert abs(got - window) < 1e-6


def test_short_gaps_are_filled_and_flagged_but_never_become_targets() -> None:
    daily, register, rain = synthetic()
    gap = date(2010, 3, 10)
    out = build_features(knock_out(daily, gap, 2), register, CFG, rain).filter(
        pl.col("reservoir_id") == "AAA"
    )
    filled = out.filter(pl.col("origin_day") == gap).row(0, named=True)
    assert filled["q_filled"] is True
    assert filled["q_lag0"] is not None
    before = out.filter(pl.col("origin_day") == gap - timedelta(days=1)).row(0, named=True)
    assert before["y_1"] is None
    assert before["y_3"] is not None


def test_long_gaps_remove_the_origins_that_need_them() -> None:
    daily, register, rain = synthetic()
    start = date(2010, 3, 1)
    out = build_features(knock_out(daily, start, 10), register, CFG, rain).filter(
        pl.col("reservoir_id") == "AAA"
    )
    days = set(out["origin_day"].to_list())
    assert start + timedelta(days=4) not in days  # inside the gap
    assert start + timedelta(days=30) not in days  # the 90-day mean still needs the gap
    assert date(2010, 9, 15) in days  # far enough, the windows are clean again


def test_dataset_id_is_stable_and_changes_with_the_inputs(tmp_path: Path) -> None:
    daily, register, rain = synthetic()
    cfg = DatasetConfig(reservoirs=("AAA",))
    inputs = {"reservoir_daily": daily, "register": register, "rain": rain}
    d1 = write_dataset(build_features(daily, register, cfg, rain), cfg, inputs, tmp_path)
    d2 = write_dataset(build_features(daily, register, cfg, rain), cfg, inputs, tmp_path)
    assert d1 == d2
    assert (d1 / "manifest.json").exists()
    altered = daily.with_columns(
        pl.when(pl.col("day") == date(2005, 1, 1))
        .then(pl.col("natural_inflow_m3s") + 1)
        .otherwise(pl.col("natural_inflow_m3s"))
        .alias("natural_inflow_m3s")
    )
    d3 = write_dataset(
        build_features(altered, register, cfg, rain),
        cfg,
        {**inputs, "reservoir_daily": altered},
        tmp_path,
    )
    assert d3 != d1


def test_rain_is_optional() -> None:
    daily, register, _ = synthetic()
    out = build_features(daily, register, CFG, None)
    assert not any(c.startswith("rain_") for c in out.columns)
