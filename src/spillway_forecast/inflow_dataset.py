"""Training set builder for the inflow forecast (stage 1): one row per reservoir and origin day.

No model is trained here. The builder only decides what is known on the origin day and what is
the target, and it does so without leaking the future:

* An origin day ``t`` is the last day with ONS data (publication delay of about one day already
  counted). Features use days ``<= t`` only: flows, storage, calendar.
* Rain is shifted by ``rain_lag_days`` (default 46, the real delay of the IMERG store in the NASA
  POWER bucket, measured on 2026-10-30): on day ``t`` only rain up to ``t - lag`` is used.
* Targets are the observed natural inflow at ``t + h``, ``h = 1..14``. Interpolated days are used
  as features only, never as targets.
* Splits are by origin year (train 2002-2017, validation 2018-2020, test 2021-2025, holdout 2026+).
  An origin whose 14-day target window crosses the end of its split is dropped (purge).
* Statistics that need history (long-term mean flow) are computed on the training years only.
* Short gaps (up to ``max_gap_days``) are interpolated and flagged; longer gaps are never filled
  and remove every origin whose windows need them.

The output is a Parquet file plus a manifest, in a directory named after a hash of the config, the
inputs and the output, so the same inputs always give the same dataset id.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

HORIZONS = 14
FLOW_WINDOWS = (7, 30, 90)
RAIN_WINDOWS = (7, 30, 90)
FLOW_LAGS = (1, 2, 6)
STATIC_COLUMNS = ("basin", "subsystem", "useful_volume_hm3", "latitude", "longitude")
MODEL_VERSION = "inflow-dataset-1"


@dataclass(frozen=True)
class DatasetConfig:
    train_start: date = date(2002, 1, 1)
    train_end: date = date(2017, 12, 31)
    valid_end: date = date(2020, 12, 31)
    test_end: date = date(2025, 12, 31)
    horizons: int = HORIZONS
    max_gap_days: int = 3
    rain_lag_days: int = 46
    final_only_rain: bool = False
    reservoirs: tuple[str, ...] | None = None
    version: str = MODEL_VERSION

    def split_of(self, origin: date) -> str | None:
        if origin < self.train_start:
            return None
        if origin <= self.train_end:
            return "train"
        if origin <= self.valid_end:
            return "valid"
        if origin <= self.test_end:
            return "test"
        return "holdout"

    def split_end(self, split: str) -> date | None:
        return {"train": self.train_end, "valid": self.valid_end, "test": self.test_end}.get(split)


def frame_hash(df: pl.DataFrame) -> str:
    buffer = io.BytesIO()
    df.sort(df.columns).write_parquet(buffer, compression="zstd", statistics=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _full_grid(daily: pl.DataFrame) -> pl.DataFrame:
    """One row per reservoir and calendar day between its first and last day."""
    span = daily.group_by("reservoir_id").agg(
        pl.col("day").min().alias("first"), pl.col("day").max().alias("last")
    )
    return (
        span.with_columns(pl.date_ranges("first", "last").alias("day"))
        .explode("day")
        .select("reservoir_id", "day")
        .sort("reservoir_id", "day")
    )


def _fill_short_gaps(df: pl.DataFrame, column: str, max_gap: int) -> pl.DataFrame:
    """Linear interpolation of null runs of at most ``max_gap`` days between two observed days.
    Adds ``{column}_filled`` (bool). Longer runs stay null."""
    null = pl.col(column).is_null()
    run_id = (
        (null != null.shift(1).over("reservoir_id")).fill_null(True).cum_sum().over("reservoir_id")
    )
    out = df.with_columns(run_id.alias("_run")).with_columns(
        pl.len().over("reservoir_id", "_run").alias("_len"),
        pl.col(column).interpolate().over("reservoir_id").alias("_interp"),
        pl.col(column).is_not_null().cum_sum().over("reservoir_id").alias("_seen"),
        pl.col(column)
        .is_not_null()
        .reverse()
        .cum_sum()
        .reverse()
        .over("reservoir_id")
        .alias("_ahead"),
    )
    fill = null & (pl.col("_len") <= max_gap) & (pl.col("_seen") > 0) & (pl.col("_ahead") > 0)
    return out.with_columns(
        pl.when(fill).then(pl.col("_interp")).otherwise(pl.col(column)).alias(column),
        fill.alias(f"{column}_filled"),
    ).drop("_run", "_len", "_interp", "_seen", "_ahead")


def _rolling(column: str, window: int, how: str) -> pl.Expr:
    expr = pl.col(column)
    rolled = (
        expr.rolling_mean(window, min_samples=window)
        if how == "mean"
        else expr.rolling_sum(window, min_samples=window)
    )
    return rolled.over("reservoir_id")


def build_features(
    reservoir_daily: pl.DataFrame,
    register: pl.DataFrame,
    config: DatasetConfig,
    rain: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Rows: reservoir_id, origin_day, split, features, targets y_1..y_H."""
    daily = reservoir_daily.select(
        "reservoir_id",
        "day",
        pl.col("natural_inflow_m3s").cast(pl.Float64).alias("q_obs"),
        pl.col("storage_pct").cast(pl.Float64).alias("ear_pct"),
    )
    if config.reservoirs is not None:
        daily = daily.filter(pl.col("reservoir_id").is_in(config.reservoirs))
    if daily.is_empty():
        raise ValueError("no reservoir-days to build a dataset from")
    grid = _full_grid(daily)
    df = grid.join(daily, on=["reservoir_id", "day"], how="left").with_columns(
        pl.col("q_obs").alias("q")
    )
    df = _fill_short_gaps(df, "q", config.max_gap_days)
    df = _fill_short_gaps(df, "ear_pct", config.max_gap_days)

    exprs: list[pl.Expr] = [pl.col("q").alias("q_lag0")]
    exprs += [pl.col("q").shift(k).over("reservoir_id").alias(f"q_lag{k}") for k in FLOW_LAGS]
    exprs += [_rolling("q", w, "mean").alias(f"q_mean{w}") for w in FLOW_WINDOWS]
    exprs += [
        pl.col("q_filled")
        .cast(pl.Int32)
        .rolling_sum(30, min_samples=1)
        .over("reservoir_id")
        .alias("q_filled_30d")
    ]
    day = pl.col("day")
    angle = 2 * math.pi * day.dt.ordinal_day() / 365.25
    exprs += [angle.sin().alias("doy_sin"), angle.cos().alias("doy_cos")]
    # hydrological year runs from October to September
    exprs += [
        pl.when(day.dt.month() >= 10)
        .then(day.dt.year() + 1)
        .otherwise(day.dt.year())
        .alias("hydro_year")
    ]
    exprs += [
        pl.col("q_obs").shift(-h).over("reservoir_id").alias(f"y_{h}")
        for h in range(1, config.horizons + 1)
    ]
    df = df.with_columns(exprs)

    if rain is not None:
        r = rain.select("reservoir_id", "day", "precip_mm", "run")
        if config.final_only_rain:
            r = r.with_columns(
                pl.when(pl.col("run") == "final").then(pl.col("precip_mm")).alias("precip_mm")
            )
        df = df.join(r.drop("run"), on=["reservoir_id", "day"], how="left")
        df = df.with_columns(
            [_rolling("precip_mm", w, "sum").alias(f"_rain{w}") for w in RAIN_WINDOWS]
        ).with_columns(
            [
                pl.col(f"_rain{w}")
                .shift(config.rain_lag_days)
                .over("reservoir_id")
                .alias(f"rain_sum{w}")
                for w in RAIN_WINDOWS
            ]
        )
        df = df.drop("precip_mm", *[f"_rain{w}" for w in RAIN_WINDOWS])

    # long-term mean flow from the training years only
    train_mean = (
        df.filter((pl.col("day") >= config.train_start) & (pl.col("day") <= config.train_end))
        .group_by("reservoir_id")
        .agg(pl.col("q_obs").mean().alias("q_mean_train"))
    )
    df = df.join(train_mean, on="reservoir_id", how="left")
    statics = (
        register.select("reservoir_id", *STATIC_COLUMNS)
        if "reservoir_id" in register
        else (register.rename({"id": "reservoir_id"}).select("reservoir_id", *STATIC_COLUMNS))
    )
    df = df.join(statics, on="reservoir_id", how="left")

    targets = [f"y_{h}" for h in range(1, config.horizons + 1)]
    df = df.rename({"day": "origin_day"}).with_columns(
        pl.col("origin_day")
        .map_elements(lambda d: config.split_of(d) or "", return_dtype=pl.String)
        .alias("split")
    )
    df = df.filter(pl.col("split") != "")
    # purge: the target window of an origin must end inside its own split
    for split in ("train", "valid", "test"):
        end = config.split_end(split)
        assert end is not None
        cutoff = end - timedelta(days=config.horizons)
        df = df.filter((pl.col("split") != split) | (pl.col("origin_day") <= cutoff))
    required = ["q_lag0", *[f"q_mean{w}" for w in FLOW_WINDOWS], "q_mean_train"]
    df = df.filter(
        pl.all_horizontal(pl.col(c).is_not_null() for c in required)
        & pl.any_horizontal(pl.col(t).is_not_null() for t in targets)
    )
    return df.drop("q_obs", "q").sort("reservoir_id", "origin_day")


def write_dataset(
    features: pl.DataFrame,
    config: DatasetConfig,
    inputs: Mapping[str, pl.DataFrame | None],
    out_dir: Path,
) -> Path:
    """Writes ``features.parquet`` and ``manifest.json`` under ``out_dir/<dataset_id>/``."""
    buffer = io.BytesIO()
    features.write_parquet(buffer, compression="zstd", statistics=False)
    data = buffer.getvalue()
    file_hash = hashlib.sha256(data).hexdigest()
    input_hashes = {k: (frame_hash(v) if v is not None else None) for k, v in inputs.items()}
    cfg: dict[str, Any] = {
        k: (v.isoformat() if isinstance(v, date) else v) for k, v in asdict(config).items()
    }
    identity = json.dumps(
        {"config": cfg, "inputs": input_hashes, "features": file_hash}, sort_keys=True, default=str
    )
    dataset_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
    target = out_dir / dataset_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "features.parquet").write_bytes(data)
    counts = {split: int(n) for split, n in features.group_by("split").len().iter_rows()}
    manifest = {
        "dataset_id": dataset_id,
        "config": cfg,
        "inputs_sha256": input_hashes,
        "features_sha256": file_hash,
        "rows": features.height,
        "rows_by_split": counts,
        "reservoirs": features["reservoir_id"].n_unique(),
        "columns": features.columns,
        "rain_attribution": "NASA POWER IMERG daily precipitation (CC BY 4.0); see spillway-data",
        "note": "Statistical dataset. Test years are never used to choose anything.",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return target
