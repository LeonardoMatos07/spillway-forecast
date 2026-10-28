"""Fallback ingestion straight from ONS (default is the silver layer, see ``silver.py``).

Download ONS constrained-off (curtailment) open data and aggregate it to hourly series.

Source: ONS open data, datasets ``restricao_coff_eolica_tm`` and
``restricao_coff_fotovoltaica_tm`` (one parquet per month, 30-minute rows per wind/solar
cluster). The same datasets feed ``energia-data``; the definitions match:

* curtailed MWh = max(``val_geracaonaorealizadaapurada``, 0) * 0.5 (the official GNRa)
* generation MWh = ``val_geracao`` * 0.5 (verified generation)

Each monthly file is downloaded to a temporary location, aggregated to hourly rows per
scope (state, subsystem, and SIN) and discarded, so only a few MB per month are kept.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

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


def ingest_silver(today: date | None = None) -> Path:
    """Default ingestion: build ``hourly.parquet`` from the silver layer of the lake."""
    from spillway_forecast.silver import read_silver_hourly

    frame = read_silver_hourly(today)
    out = hourly_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out)
    return out


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


# --------------------------------------------------------------------------------------------
# Hybrid ingestion: validated silver history + a provisional tail straight from ONS
# --------------------------------------------------------------------------------------------

META_NAME = "hourly_meta.json"


def meta_path() -> Path:
    return hourly_path().with_name(META_NAME)


def read_meta() -> dict[str, Any]:
    path = meta_path()
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def ons_signature(today: date) -> dict[str, str]:
    """ETag of the ONS files of the current and previous month (HEAD requests, a few KB)."""
    sig: dict[str, str] = {}
    prev = date(today.year - 1, 12, 1) if today.month == 1 else date(today.year, today.month - 1, 1)
    for source in ONS_DATASETS:
        for year, month in ((prev.year, prev.month), (today.year, today.month)):
            url = file_url(source, year, month)
            response = httpx.head(url, timeout=30, follow_redirects=True)
            sig[f"{source}:{year}-{month:02d}"] = (
                response.headers.get("etag", "") if response.status_code == 200 else "missing"
            )
    return sig


def read_ons_month(source: str, year: int, month: int) -> pl.DataFrame | None:
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "raw.parquet"
        if not _download(file_url(source, year, month), raw_path):
            return None
        out = aggregate_raw(pl.read_parquet(raw_path), source)
        return out.with_columns(pl.col("ts").cast(pl.Datetime("us")))


def complete_days_only(frame: pl.DataFrame) -> pl.DataFrame:
    """Drop the trailing days that do not have all 24 hours (the series are dense day x 24)."""
    sin = frame.filter(pl.col("scope_type") == "sin").with_columns(
        pl.col("ts").dt.date().alias("d")
    )
    full = sin.group_by("d").agg(pl.col("ts").n_unique().alias("h")).filter(pl.col("h") == 24)
    if full.is_empty():
        return frame.clear()
    last_full = full["d"].max()
    return frame.filter(pl.col("ts").dt.date() <= last_full)


def compare_overlap(silver: pl.DataFrame, ons: pl.DataFrame) -> dict[str, float]:
    """Largest deviation between silver and a direct ONS aggregation over the same hours."""
    keys = ["ts", "scope_type", "scope", "source"]
    joined = silver.join(ons, on=keys, how="inner", suffix="_ons")
    if joined.is_empty():
        return {"hours": 0, "max_abs_curtailed_mwh": float("nan")}
    dev = joined.select(
        (pl.col("curtailed_mwh") - pl.col("curtailed_mwh_ons")).abs().max().alias("c"),
        (pl.col("generation_mwh") - pl.col("generation_mwh_ons")).abs().max().alias("g"),
        pl.col("curtailed_mwh").sum().alias("sc"),
        (pl.col("curtailed_mwh") - pl.col("curtailed_mwh_ons")).abs().sum().alias("dc"),
    ).row(0, named=True)
    return {
        "hours": float(joined.height),
        "max_abs_curtailed_mwh": float(dev["c"]),
        "max_abs_generation_mwh": float(dev["g"]),
        "relative_curtailed_sum": float(dev["dc"] / dev["sc"]) if dev["sc"] else 0.0,
    }


def ingest_hybrid(today: date | None = None, ons_check: dict[str, str] | None = None) -> Path:
    """Silver history (validated) + provisional tail from ONS for the months after silver.

    Silver only holds complete months (the pipeline loads the previous month), so the newest
    weeks come from the ONS file of the current month, which ONS fills up to about yesterday.
    Tail rows are marked ``provisional`` (ONS revises them) and trimmed to complete days. Where
    both sources cover the same hours (the last silver month) the deviation is measured and
    logged; silver always wins.
    """
    from spillway_forecast.silver import read_silver_hourly

    today = today or date.today()
    silver = read_silver_hourly(today).with_columns(pl.lit(False).alias("provisional"))
    meta: dict[str, Any] = {"built_at": datetime.now(UTC).isoformat(timespec="seconds")}
    tails: list[pl.DataFrame] = []
    for source in ONS_DATASETS:
        sub = silver.filter(pl.col("source") == source)
        last_ts = cast("datetime | None", sub["ts"].max())
        silver_through = last_ts.date() if last_ts is not None else None
        meta[f"silver_through_{source}"] = str(silver_through)
        if silver_through is None or last_ts is None:
            continue
        last_month = (silver_through.year, silver_through.month)
        overlap = read_ons_month(source, *last_month)
        if overlap is not None:
            dev = compare_overlap(sub, overlap)
            meta[f"overlap_{source}"] = {"month": f"{last_month[0]}-{last_month[1]:02d}", **dev}
            print(f"[hybrid] {source} {last_month}: silver vs ONS deviation {dev}", flush=True)
        for year, month in month_range(last_month, today):
            if (year, month) == last_month:
                continue
            part = read_ons_month(source, year, month)
            if part is None:
                print(f"[hybrid] {source} {year}-{month:02d}: not published", flush=True)
                continue
            part = complete_days_only(part).filter(pl.col("ts") > pl.lit(last_ts))
            tails.append(part.with_columns(pl.lit(True).alias("provisional")))
        source_tail = [t for t in tails if t.height and t["source"][0] == source]
        newest = max((cast("datetime", t["ts"].max()) for t in source_tail), default=last_ts)
        meta[f"data_through_{source}"] = str(newest.date())
    frames = [silver, *[t.select(silver.columns) for t in tails if t.height]]
    frame = (
        pl.concat(frames)
        .unique(subset=["ts", "scope", "scope_type", "source"], keep="first")
        .sort(["source", "scope_type", "scope", "ts"])
    )
    out = hourly_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    frame.write_parquet(tmp)
    tmp.replace(out)  # atomic: readers never see a half-written file
    meta["provisional_rows"] = int(frame["provisional"].sum())
    meta["ons_signature"] = ons_check if ons_check is not None else ons_signature(today)
    meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out
