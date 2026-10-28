"""Simple baselines the gradient boosting model has to beat.

* ``ClimatologyBaseline``: historical mean curtailment % (and risk rate) by
  series x month x day-type x hour. It is fitted on the *unique hourly observations*
  before the split (not on the overlapping forecast rows), with back-off to coarser cells.
* ``seasonal_naive``: the value of the same hour one week earlier (``snaive_7d`` feature).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spillway_forecast.config import MIN_DENOMINATOR_MWH, RISK_THRESHOLD

KEYS_FULL = ["series_id", "month", "is_weekend", "hour"]
KEYS_MID = ["series_id", "is_weekend", "hour"]
KEYS_COARSE = ["series_id"]
MIN_CELL_COUNT = 5


def hourly_observations(hourly: pd.DataFrame) -> pd.DataFrame:
    """Unique (series, ts) observations with pct, risk and calendar columns."""
    obs = hourly.copy()
    denom = obs["curtailed_mwh"] + obs["generation_mwh"]
    obs["pct"] = obs["curtailed_mwh"] / denom.where(denom >= MIN_DENOMINATOR_MWH)
    obs = obs[obs["pct"].notna()].copy()
    obs["series_id"] = obs["source"] + ":" + obs["scope"]
    obs["risk"] = (obs["pct"] >= RISK_THRESHOLD).astype(float)
    obs["month"] = obs["ts"].dt.month.astype(float)
    obs["is_weekend"] = (obs["ts"].dt.day_of_week >= 5).astype(float)
    obs["hour"] = obs["ts"].dt.hour.astype(float)
    return obs


class ClimatologyBaseline:
    def __init__(self) -> None:
        self._tables: list[tuple[list[str], pd.DataFrame]] = []

    def fit(self, obs: pd.DataFrame) -> ClimatologyBaseline:
        self._tables = []
        for keys in (KEYS_FULL, KEYS_MID, KEYS_COARSE):
            agg = obs.groupby(keys)[["pct", "risk"]].agg(["mean", "count"])
            agg.columns = ["pct", "n_pct", "risk", "n_risk"]
            agg = agg[agg["n_pct"] >= (MIN_CELL_COUNT if keys is not KEYS_COARSE else 1)]
            self._tables.append((keys, agg[["pct", "risk"]].reset_index()))
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Return columns ``pct`` and ``risk`` (probability) aligned with ``rows``."""
        out = pd.DataFrame({"pct": np.nan, "risk": np.nan}, index=rows.index)
        for keys, table in self._tables:
            missing = out["pct"].isna()
            if not missing.any():
                break
            merged = rows.loc[missing, keys].merge(table, on=keys, how="left")
            merged.index = rows.index[missing]
            out.loc[missing, "pct"] = merged["pct"]
            out.loc[missing, "risk"] = merged["risk"]
        return out


def seasonal_naive(rows: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    """Same hour last week; falls back to climatology where it is missing."""
    pct = rows["snaive_7d"].where(rows["snaive_7d"].notna(), fallback["pct"])
    risk = (
        (rows["snaive_7d"] >= RISK_THRESHOLD)
        .astype(float)
        .where(rows["snaive_7d"].notna(), fallback["risk"])
    )
    return pd.DataFrame({"pct": pct, "risk": risk})
