"""Download ONS constrained-off (curtailment) open data and aggregate it to hourly series.

Source: ONS open data, datasets ``restricao_coff_eolica_tm`` and
``restricao_coff_fotovoltaica_tm`` (one parquet per month, 30-minute rows per wind/solar
cluster). The same datasets feed ``energia-data``; the definitions match:

* curtailed MWh = max(``val_geracaonaorealizadaapurada``, 0) * 0.5 (the official GNRa)
* generation MWh = ``val_geracao`` * 0.5 (verified generation)

Each monthly file is downloaded to a temporary location, aggregated to hourly rows per
scope (state, subsystem, and SIN) and discarded, so only a few MB per month are kept.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from spillway_forecast.config import HALF_HOUR, ONS_BASE_URL, ONS_DATASETS, data_dir, hourly_path

HOURLY_COLUMNS = ["ts", "scope_type", "scope", "source", "curtailed_mwh", "generation_mwh"]


def month_range(start: tuple[int, int], end: date) -> Iterator[tuple[int, int]]:
    year, month = start
    while (year, month) <= (end.year, end.month):
        yield year, month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def file_url(source: str, year: int, month: int) -> str:
    folder, prefix, _ = ONS_DATASETS[source]
    return f"{ONS_BASE_URL}/{folder}/{prefix}_{year}_{month:02d}.parquet"


def aggregate_raw(raw: pl.DataFrame, source: str) -> pl.DataFrame:
    """Aggregate 30-minute cluster rows into hourly rows per state, subsystem and SIN."""
    base = raw.select(
        pl.col("din_instante").dt.truncate("1h").alias("ts"),
        pl.col("id_estado").alias("state"),
        pl.col("id_subsistema").alias("subsystem"),
        (
            pl.col("val_geracaonaorealizadaapurada")
            .fill_nan(None)
            .fill_null(0.0)
            .clip(lower_bound=0.0)
            * HALF_HOUR
        ).alias("curtailed_mwh"),
        (pl.col("val_geracao").fill_nan(None).fill_null(0.0) * HALF_HOUR).alias("generation_mwh"),
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


def _download(url: str, destination: Path) -> bool:
    with (
        httpx.stream("GET", url, timeout=120, follow_redirects=True) as response,
    ):
        if response.status_code == 404:
            return False
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    return True


def ingest(
    sources: list[str] | None = None, refresh_last: int = 2, today: date | None = None
) -> Path:
    """Download missing months, refresh the last ``refresh_last`` and write hourly parquet."""
    today = today or date.today()
    interim = data_dir() / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    chosen = sources or list(ONS_DATASETS)
    for source in chosen:
        first = ONS_DATASETS[source][2]
        months = list(month_range(first, today))
        for idx, (year, month) in enumerate(months):
            target = interim / f"{source}_{year}_{month:02d}.parquet"
            stale = idx >= len(months) - refresh_last
            if target.exists() and not stale:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                raw_path = Path(tmp) / "raw.parquet"
                if not _download(file_url(source, year, month), raw_path):
                    print(f"[ingest] {source} {year}-{month:02d}: not published yet")
                    continue
                aggregate_raw(pl.read_parquet(raw_path), source).write_parquet(target)
            print(f"[ingest] {source} {year}-{month:02d}: ok")
    return combine()


def combine() -> Path:
    files = sorted((data_dir() / "interim").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No interim files; run `spillway ingest` first.")
    frame = pl.concat([pl.read_parquet(f) for f in files]).unique(
        subset=["ts", "scope", "scope_type", "source"], keep="last"
    )
    out = hourly_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.sort(["source", "scope_type", "scope", "ts"]).write_parquet(out)
    return out
