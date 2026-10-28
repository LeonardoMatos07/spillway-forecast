"""Read the typed silver layer of ``spillway-data`` (S3 API of the SeaweedFS lake).

Silver files: ``silver/{wind,solar}_cluster_curtailment/year=YYYY/month=MM/data.parquet``,
one row per cluster and 30 minutes, already typed, deduplicated and with invalid rows
quarantined. Relevant columns (see ``energy_pipeline/schema.py`` and ``curtailment.py``):

* ``measured_at``: Brasilia local time (naive); ``measured_at_utc`` is also kept
* ``state_id``, ``subsystem_id``
* ``generation_mwh`` = verified generation x 0.5 and ``curtailed_mwh`` =
  max(GNRa, 0) x 0.5, i.e. exactly the definitions the warehouse uses.

Silver already carries everything we need, so nothing is re-derived here except the hourly
aggregation per state / subsystem / SIN (the pipeline only publishes monthly aggregates).

Access note: SeaweedFS needs **path-style addressing**, an explicit region and the static
keys. With virtual-host style (``bucket.localhost``) or without keys requests fail with
``AccessDenied``/DNS errors.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterator
from datetime import date
from typing import Any

import polars as pl

from spillway_forecast.config import LAKE_DEFAULTS, LAKE_PREFIXES, ONS_DATASETS
from spillway_forecast.ingest import HOURLY_COLUMNS, month_range


def lake_client() -> tuple[Any, str]:
    """boto3 S3 client + bucket, configured from ``LAKE_*`` / ``AWS_*`` env vars."""
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("LAKE_S3_ENDPOINT_URL", LAKE_DEFAULTS["endpoint"]),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", LAKE_DEFAULTS["access_key"]),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", LAKE_DEFAULTS["secret_key"]),
        region_name=os.environ.get("AWS_DEFAULT_REGION", LAKE_DEFAULTS["region"]),
        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 3}),
    )
    return client, os.environ.get("LAKE_BUCKET", LAKE_DEFAULTS["bucket"])


def silver_key(source: str, year: int, month: int) -> str:
    return f"{LAKE_PREFIXES[source]}/year={year}/month={month:02d}/data.parquet"


def aggregate_silver(raw: pl.DataFrame, source: str) -> pl.DataFrame:
    """Aggregate 30-minute silver rows to hourly rows per state, subsystem and SIN."""
    base = raw.select(
        pl.col("measured_at").dt.truncate("1h").alias("ts"),
        pl.col("state_id").alias("state"),
        pl.col("subsystem_id").alias("subsystem"),
        pl.col("curtailed_mwh").fill_nan(None).fill_null(0.0),
        pl.col("generation_mwh").fill_nan(None).fill_null(0.0),
    )
    parts = []
    for scope_type, keys in (("state", ["state"]), ("subsystem", ["subsystem"]), ("sin", [])):
        grouped = base.group_by(["ts", *keys]).agg(
            pl.col("curtailed_mwh").sum(), pl.col("generation_mwh").sum()
        )
        scope = pl.col(keys[0]) if keys else pl.lit("SIN")
        parts.append(
            grouped.select(
                "ts",
                pl.lit(scope_type).alias("scope_type"),
                scope.alias("scope"),
                pl.lit(source).alias("source"),
                "curtailed_mwh",
                "generation_mwh",
            )
        )
    return pl.concat(parts).select(HOURLY_COLUMNS).sort(["scope", "ts"])


def iter_months(today: date) -> Iterator[tuple[str, int, int]]:
    for source, (_, _, first) in ONS_DATASETS.items():
        for year, month in month_range(first, today):
            yield source, year, month


def read_silver_hourly(
    today: date | None = None, client: Any = None, bucket: str | None = None
) -> pl.DataFrame:
    """Read every available silver month and return the combined hourly frame."""
    from botocore.exceptions import ClientError

    if client is None:
        client, default_bucket = lake_client()
        bucket = bucket or default_bucket
    frames = []
    for source, year, month in iter_months(today or date.today()):
        key = silver_key(source, year, month)
        try:
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        except ClientError as error:
            if error.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
                continue  # month not in the lake (yet)
            raise
        raw = pl.read_parquet(io.BytesIO(body))
        frames.append(aggregate_silver(raw, source))
        print(f"[silver] {source} {year}-{month:02d}: {raw.height} rows")
    if not frames:
        raise FileNotFoundError("no silver curtailment files found in the lake")
    return (
        pl.concat(frames)
        .unique(subset=["ts", "scope", "scope_type", "source"], keep="last")
        .sort(["source", "scope_type", "scope", "ts"])
    )
