"""Evaluation framework for the inflow forecast (stage 2): metrics per reservoir and horizon,
aggregation across reservoirs (median and spread) and the publication gate of the spec.

Everything here is model agnostic: a forecast is a frame with the observed flow and P10/P50/P90.
The gate is computed per horizon against the best reference, which is chosen on VALIDATION years
and then judged on TEST years, so test data never picks a winner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import polars as pl

QUANTILES = (0.1, 0.5, 0.9)
FORECAST_COLUMNS = ("reservoir_id", "origin_day", "split", "h", "obs", "p10", "p50", "p90")


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((obs - obs.mean()) ** 2))
    return float("nan") if denom <= 0 else 1.0 - float(np.sum((obs - pred) ** 2)) / denom


def kge(obs: np.ndarray, pred: np.ndarray) -> float:
    """Kling-Gupta efficiency (2009): correlation, variability ratio and bias ratio."""
    if len(obs) < 3 or obs.std() == 0 or pred.std() == 0 or obs.mean() == 0:
        return float("nan")
    r = float(np.corrcoef(obs, pred)[0, 1])
    alpha = float(pred.std() / obs.std())
    beta = float(pred.mean() / obs.mean())
    return 1.0 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def pinball(obs: np.ndarray, pred: np.ndarray, tau: float) -> float:
    diff = obs - pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def series_metrics(
    obs: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray
) -> dict[str, float]:
    """Metrics of one reservoir at one horizon. Peak error uses the observed top 5% of days."""
    if len(obs) == 0:
        return {}
    total = float(obs.sum())
    peaks = obs >= np.quantile(obs, 0.95)
    peaks &= obs > 0
    scale = float(obs.mean()) if obs.mean() > 0 else 1.0
    return {
        "n": float(len(obs)),
        "nse": nse(obs, p50),
        "kge": kge(obs, p50),
        "rmse": float(np.sqrt(np.mean((obs - p50) ** 2))),
        "bias_pct": float(100 * (p50.sum() - total) / total) if total > 0 else float("nan"),
        "peak_err_pct": (
            float(100 * np.mean((p50[peaks] - obs[peaks]) / obs[peaks]))
            if peaks.any()
            else float("nan")
        ),
        "coverage": float(np.mean((obs >= p10) & (obs <= p90))),
        "pinball_rel": float(
            np.mean([pinball(obs, q, t) for q, t in zip((p10, p50, p90), QUANTILES, strict=True)])
            / scale
        ),
    }


def per_reservoir(forecasts: pl.DataFrame, split: str) -> pl.DataFrame:
    """Rows: model, reservoir_id, h, metrics. ``forecasts`` has a ``model`` column."""
    rows: list[dict[str, object]] = []
    sub = forecasts.filter((pl.col("split") == split) & pl.col("obs").is_not_null())
    for (model, rid, h), g in sub.group_by("model", "reservoir_id", "h", maintain_order=True):
        m = series_metrics(
            g["obs"].to_numpy(), g["p10"].to_numpy(), g["p50"].to_numpy(), g["p90"].to_numpy()
        )
        if m:
            rows.append({"model": model, "reservoir_id": rid, "h": h, **m})
    return pl.DataFrame(rows)


def aggregate(table: pl.DataFrame) -> pl.DataFrame:
    """Median and inter-quartile spread across reservoirs, per model and horizon."""
    metrics = ["nse", "kge", "rmse", "bias_pct", "peak_err_pct", "coverage", "pinball_rel"]
    aggs: list[pl.Expr] = [pl.len().alias("reservoirs")]
    for c in metrics:
        aggs += [
            pl.col(c).median().alias(f"{c}_med"),
            pl.col(c).quantile(0.25).alias(f"{c}_q25"),
            pl.col(c).quantile(0.75).alias(f"{c}_q75"),
        ]
    return table.group_by("model", "h").agg(aggs).sort("model", "h")


def _med(df: pl.DataFrame, col: str) -> float:
    return float(cast(float, df[col].median()))


def best_reference(valid_table: pl.DataFrame, references: list[str], h: int) -> str:
    """Reference with the highest median NSE on VALIDATION at horizon ``h``."""
    sub = valid_table.filter((pl.col("h") == h) & pl.col("model").is_in(references))
    med = sub.group_by("model").agg(pl.col("nse").median().alias("m")).sort("m", descending=True)
    return str(med["model"][0])


@dataclass(frozen=True)
class GateConfig:
    """Publication criterion of the spec (to be confirmed by the owner at stage 3)."""

    coverage_low: float = 0.70
    coverage_high: float = 0.90
    max_worse_share: float = 0.30
    worse_margin: float = 0.0  # a reservoir counts as worse when its NSE is below the reference


def gate(
    test_table: pl.DataFrame,
    valid_table: pl.DataFrame,
    candidate: str,
    references: list[str],
    config: GateConfig | None = None,
) -> pl.DataFrame:
    """One row per horizon with each condition and the verdict."""
    config = config or GateConfig()
    out: list[dict[str, object]] = []
    for h in sorted(test_table["h"].unique().to_list()):
        ref = best_reference(valid_table, references, h)
        c = test_table.filter((pl.col("model") == candidate) & (pl.col("h") == h))
        r = test_table.filter((pl.col("model") == ref) & (pl.col("h") == h))
        j = c.join(r, on="reservoir_id", suffix="_ref")
        if j.is_empty():
            continue
        wins_nse = _med(j, "nse") > _med(j, "nse_ref")
        wins_kge = _med(j, "kge") > _med(j, "kge_ref")
        bias_ok = abs(_med(j, "bias_pct")) <= abs(_med(j, "bias_pct_ref")) + 1e-9
        peak_ok = abs(_med(j, "peak_err_pct")) <= abs(_med(j, "peak_err_pct_ref")) + 1e-9
        cov = float(_med(j, "coverage"))
        cov_ok = config.coverage_low <= cov <= config.coverage_high
        worse = float((j["nse"] < j["nse_ref"] - config.worse_margin).sum() / j.height)
        worse_ok = worse <= config.max_worse_share
        out.append(
            {
                "h": h,
                "best_reference": ref,
                "wins_nse_or_kge": bool(wins_nse or wins_kge),
                "bias_not_worse": bool(bias_ok),
                "peak_not_worse": bool(peak_ok),
                "coverage_median": cov,
                "coverage_ok": cov_ok,
                "share_worse": worse,
                "share_worse_ok": worse_ok,
                "publish": bool(
                    (wins_nse or wins_kge) and bias_ok and peak_ok and cov_ok and worse_ok
                ),
            }
        )
    return pl.DataFrame(out)
