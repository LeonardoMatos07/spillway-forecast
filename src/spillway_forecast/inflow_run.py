"""Runs the stage 2 comparison: references, ridge and gradient boosting on one dataset.

Fit on train, choose the few hyper-parameters on validation, score validation and test. The
``holdout`` split (2026 onward) is never read. Writes ``results.json`` and the per-reservoir
table so that the report can be rebuilt from files.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from spillway_forecast import inflow_eval as ev
from spillway_forecast import inflow_models as im

REFERENCES = ["persistence", "climatology", "recession", "moving_average_7d"]
CANDIDATES = ["ridge", "gbm_quantile"]


def run(dataset_dir: Path, out_dir: Path) -> dict[str, Any]:
    features = pl.read_parquet(dataset_dir / "features.parquet")
    train = features.filter(pl.col("split") == "train")
    valid = features.filter(pl.col("split") == "valid")
    test = features.filter(pl.col("split") == "test")
    scored = pl.concat([valid, test])

    builders: dict[str, Callable[[], Any]] = {
        "persistence": lambda: im.Persistence().fit(train),
        "climatology": lambda: im.Climatology().fit(train),
        "recession": lambda: im.Recession().fit(train),
        "moving_average_7d": lambda: im.MovingAverage().fit(train),
        "ridge": lambda: im.RidgeDirect().fit(train, valid),
        "gbm_quantile": lambda: im.GradientBoostingQuantile().fit(train, valid),
    }
    frames: list[pl.DataFrame] = []
    chosen: dict[str, Any] = {}
    for name, build in builders.items():
        model = build()
        frames.append(im.to_forecast_frame(scored, name, model.predict(scored)))
        if name == "ridge":
            chosen[name] = {"alpha_by_horizon": model.chosen}
        if name == "gbm_quantile":
            chosen[name] = {"max_iter_max_depth": list(model.params)}
    # ablation: the same boosting model without the (46-day lagged) rain features
    no_rain_cols = [c for c in features.columns if c.startswith("rain_sum")]
    tr0, va0, sc0 = (d.drop(no_rain_cols) for d in (train, valid, scored))
    gbm0 = im.GradientBoostingQuantile().fit(tr0, va0)
    frames.append(im.to_forecast_frame(sc0, "gbm_no_rain", gbm0.predict(sc0)))

    forecasts = pl.concat(frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    forecasts.write_parquet(out_dir / "forecasts.parquet")  # large, local only
    tables = {s: ev.per_reservoir(forecasts, s) for s in ("valid", "test")}
    for s, t in tables.items():
        t.write_parquet(out_dir / f"per_reservoir_{s}.parquet")
    agg = {s: ev.aggregate(t) for s, t in tables.items()}
    for s, a in agg.items():
        a.write_csv(out_dir / f"aggregate_{s}.csv")

    refs = [*REFERENCES]
    gates = {
        cand: ev.gate(tables["test"], tables["valid"], cand, refs)
        for cand in (*CANDIDATES, "gbm_no_rain")
    }
    for cand, g in gates.items():
        g.write_csv(out_dir / f"gate_{cand}.csv")

    by_year = _by_year(forecasts, test)
    by_year.write_csv(out_dir / "by_year_test.csv")
    result = {
        "dataset_id": dataset_dir.name,
        "rows": {"train": train.height, "valid": valid.height, "test": test.height},
        "reservoirs": features["reservoir_id"].n_unique(),
        "chosen_on_validation": chosen,
        "publishable_horizons": {
            cand: [int(h) for h in g.filter(pl.col("publish"))["h"].to_list()]
            for cand, g in gates.items()
        },
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def _by_year(forecasts: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    """Median NSE over reservoirs by hydrological year (Oct-Sep) of the TARGET day, test years."""
    f = forecasts.filter((pl.col("split") == "test") & pl.col("obs").is_not_null())
    f = f.filter(pl.col("h").is_in([1, 3, 7, 14]))
    target_day = pl.col("origin_day") + pl.duration(days=pl.col("h"))
    f = f.with_columns(
        pl.when(target_day.dt.month() >= 10)
        .then(target_day.dt.year() + 1)
        .otherwise(target_day.dt.year())
        .alias("hydro_year")
    )
    rows: list[dict[str, Any]] = []
    for (model, h, hy, rid), g in f.group_by("model", "h", "hydro_year", "reservoir_id"):
        o = g["obs"].to_numpy()
        if len(o) >= 60:
            rows.append(
                {
                    "model": model,
                    "h": h,
                    "hydro_year": hy,
                    "reservoir_id": rid,
                    "nse": ev.nse(o, g["p50"].to_numpy()),
                }
            )
    t = pl.DataFrame(rows)
    return (
        t.group_by("model", "h", "hydro_year")
        .agg(pl.col("nse").median().alias("nse_med"), pl.len().alias("reservoirs"))
        .sort("model", "h", "hydro_year")
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="stage 2 comparison on a built inflow dataset")
    p.add_argument("--dataset", required=True, help="directory with features.parquet")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    print(json.dumps(run(Path(a.dataset), Path(a.out)), indent=2))


if __name__ == "__main__":
    main()
