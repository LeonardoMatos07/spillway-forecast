"""Reference forecasters and statistical competitors for the inflow forecast (stage 2).

Order fixed by the spec: simple references (persistence, day-of-year climatology, exponential
recession, moving average), regularised linear regression per horizon, gradient boosting with
quantiles per horizon. No neural network here. Every model is fitted on TRAIN rows only; the
validation rows choose the few hyper-parameters; test rows are only scored.

All models return an array ``(n, H, 3)`` of P10/P50/P90 in m3/s. Reference models are point
forecasts, so their band comes from the empirical log-residual quantiles on the training rows
(per reservoir and horizon, with the horizon-wide pool as fallback).
"""

from __future__ import annotations

import warnings
from datetime import timedelta

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

H = 14
MIN_RESIDUALS = 200
SEED = 20260101
Z_FEATURES = ("q_lag0", "q_lag1", "q_lag2", "q_lag6", "q_mean7", "q_mean30", "q_mean90")


def _z(a: np.ndarray) -> np.ndarray:
    return np.asarray(np.log1p(np.clip(a, 0.0, None)), dtype=float)


def targets(df: pl.DataFrame) -> np.ndarray:
    return df.select([f"y_{h}" for h in range(1, H + 1)]).to_numpy().astype(float)


class ResidualBand:
    """Empirical P10/P90 of log-space residuals, per (reservoir, horizon)."""

    def __init__(self) -> None:
        self.by_key: dict[tuple[str, int], tuple[float, float]] = {}
        self.pool: dict[int, tuple[float, float]] = {}

    def fit(self, reservoirs: np.ndarray, obs: np.ndarray, median: np.ndarray) -> ResidualBand:
        resid = _z(obs) - _z(median)
        for h in range(H):
            col = resid[:, h]
            ok = np.isfinite(col)
            if ok.sum() == 0:
                self.pool[h] = (0.0, 0.0)
                continue
            self.pool[h] = (float(np.quantile(col[ok], 0.1)), float(np.quantile(col[ok], 0.9)))
            for rid in np.unique(reservoirs):
                m = ok & (reservoirs == rid)
                if m.sum() >= MIN_RESIDUALS:
                    self.by_key[(str(rid), h)] = (
                        float(np.quantile(col[m], 0.1)),
                        float(np.quantile(col[m], 0.9)),
                    )
        return self

    def apply(self, reservoirs: np.ndarray, median: np.ndarray) -> np.ndarray:
        out = np.empty((*median.shape, 3))
        zmed = _z(median)
        for h in range(H):
            lo = np.array([self.by_key.get((str(r), h), self.pool[h])[0] for r in reservoirs])
            hi = np.array([self.by_key.get((str(r), h), self.pool[h])[1] for r in reservoirs])
            out[:, h, 0] = np.minimum(np.expm1(zmed[:, h] + lo), median[:, h])
            out[:, h, 1] = median[:, h]
            out[:, h, 2] = np.maximum(np.expm1(zmed[:, h] + hi), median[:, h])
        out[:, :, 0] = np.clip(out[:, :, 0], 0.0, None)
        return out


class PointReference:
    """Base class: subclasses define ``point``; the band comes from training residuals."""

    name = "reference"

    def fit(self, train: pl.DataFrame) -> PointReference:
        self._fit_point(train)
        self.band = ResidualBand().fit(
            train["reservoir_id"].to_numpy(), targets(train), self.point(train)
        )
        return self

    def _fit_point(self, train: pl.DataFrame) -> None:
        pass

    def point(self, df: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return self.band.apply(df["reservoir_id"].to_numpy(), self.point(df))


class Persistence(PointReference):
    name = "persistence"

    def point(self, df: pl.DataFrame) -> np.ndarray:
        return np.repeat(df["q_lag0"].to_numpy()[:, None], H, axis=1).astype(float)


class MovingAverage(PointReference):
    name = "moving_average_7d"

    def point(self, df: pl.DataFrame) -> np.ndarray:
        return np.repeat(df["q_mean7"].to_numpy()[:, None], H, axis=1).astype(float)


class Recession(PointReference):
    """q(t+h) = q(t) * k^h, with k the median day-to-day ratio of falling days, per reservoir."""

    name = "recession"

    def _fit_point(self, train: pl.DataFrame) -> None:
        ratio = train.filter(
            (pl.col("q_lag0") > 0)
            & pl.col("y_1").is_not_null()
            & (pl.col("y_1") < pl.col("q_lag0"))
        ).with_columns((pl.col("y_1") / pl.col("q_lag0")).alias("r"))
        self.k = {
            str(r): float(k)
            for r, k in ratio.group_by("reservoir_id").agg(pl.col("r").median()).iter_rows()
        }
        median = ratio["r"].median() if ratio.height else None
        self.k_default = float(median) if isinstance(median, (int, float)) else 1.0

    def point(self, df: pl.DataFrame) -> np.ndarray:
        k = np.array([self.k.get(str(r), self.k_default) for r in df["reservoir_id"].to_numpy()])
        steps = np.arange(1, H + 1)[None, :]
        return df["q_lag0"].to_numpy()[:, None].astype(float) * k[:, None] ** steps


class Climatology(PointReference):
    """Mean flow for the target day of year, from training years only (+-7 day window)."""

    name = "climatology"

    def _fit_point(self, train: pl.DataFrame) -> None:
        d = train.select(
            "reservoir_id",
            pl.col("origin_day").dt.ordinal_day().alias("doy"),
            pl.col("q_lag0").alias("q"),
        )
        self.table: dict[str, np.ndarray] = {}
        for key, g in d.group_by("reservoir_id"):
            sums = np.zeros(366)
            cnt = np.zeros(366)
            for doy, q in zip(g["doy"].to_numpy(), g["q"].to_numpy(), strict=True):
                if np.isfinite(q):
                    sums[doy - 1] += q
                    cnt[doy - 1] += 1
            kernel = np.ones(15)
            ss = np.convolve(np.r_[sums[-7:], sums, sums[:7]], kernel, mode="valid")
            cc = np.convolve(np.r_[cnt[-7:], cnt, cnt[:7]], kernel, mode="valid")
            self.table[str(key[0])] = np.where(cc > 0, ss / np.maximum(cc, 1), np.nan)

    def point(self, df: pl.DataFrame) -> np.ndarray:
        origins = df["origin_day"].to_list()
        rids = df["reservoir_id"].to_numpy()
        out = np.full((len(origins), H), np.nan)
        for i, (o, r) in enumerate(zip(origins, rids, strict=True)):
            tab = self.table.get(str(r))
            if tab is None:
                continue
            for h in range(H):
                out[i, h] = tab[(o + timedelta(days=h + 1)).timetuple().tm_yday - 1]
        return out


def feature_matrix(df: pl.DataFrame, basins: list[str]) -> np.ndarray:
    cols = [_z(df[c].to_numpy().astype(float)) for c in Z_FEATURES]
    cols.append(_z(df["q_mean_train"].to_numpy().astype(float)))
    cols.append(df["ear_pct"].to_numpy().astype(float))
    for c in ("rain_sum7", "rain_sum30", "rain_sum90"):
        if c in df.columns:  # absent in the no-rain ablation
            cols.append(_z(df[c].to_numpy().astype(float)))
    cols += [df["doy_sin"].to_numpy(), df["doy_cos"].to_numpy()]
    cols.append(np.log(df["useful_volume_hm3"].to_numpy().astype(float).clip(1.0)))
    basin = df["basin"].to_numpy()
    cols += [(basin == b).astype(float) for b in basins]
    return np.column_stack(cols).astype(float)


def _delta_targets(df: pl.DataFrame) -> np.ndarray:
    """log1p(y_h) - log1p(q_lag0): the model learns the change, not the level."""
    return np.asarray(
        _z(targets(df)) - _z(df["q_lag0"].to_numpy().astype(float))[:, None], dtype=float
    )


def _from_delta(df: pl.DataFrame, delta: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.expm1(_z(df["q_lag0"].to_numpy().astype(float))[:, None] + delta), dtype=float
    )


class RidgeDirect:
    """Regularised linear regression per horizon on the log change; band from train residuals."""

    name = "ridge"

    def __init__(self, alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)) -> None:
        self.alphas = alphas

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame) -> RidgeDirect:
        self.basins = sorted(train["basin"].unique().to_list())
        xt0, xv0 = feature_matrix(train, self.basins), feature_matrix(valid, self.basins)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.fill = np.nan_to_num(np.nanmedian(xt0, axis=0), nan=0.0)
        xt, xv = self._prep(xt0), self._prep(xv0)
        self.mu, self.sd = xt.mean(axis=0), xt.std(axis=0) + 1e-9
        xt, xv = (xt - self.mu) / self.sd, (xv - self.mu) / self.sd
        yt, yv = _delta_targets(train), _delta_targets(valid)
        self.models: list[Ridge] = []
        self.chosen: list[float] = []
        for h in range(H):
            ok_t, ok_v = np.isfinite(yt[:, h]), np.isfinite(yv[:, h])

            def loss(
                a: float, h: int = h, ok_t: np.ndarray = ok_t, ok_v: np.ndarray = ok_v
            ) -> float:
                fit = Ridge(alpha=a).fit(xt[ok_t], yt[ok_t, h])
                return float(np.mean((fit.predict(xv[ok_v]) - yv[ok_v, h]) ** 2))

            best = min(self.alphas, key=loss)
            self.chosen.append(best)
            self.models.append(Ridge(alpha=best).fit(xt[ok_t], yt[ok_t, h]))
        self.band = ResidualBand().fit(
            train["reservoir_id"].to_numpy(), targets(train), self._point(train)
        )
        return self

    def _prep(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        idx = np.where(np.isnan(x))
        x[idx] = np.take(self.fill, idx[1])
        return x

    def _point(self, df: pl.DataFrame) -> np.ndarray:
        x = (self._prep(feature_matrix(df, self.basins)) - self.mu) / self.sd
        delta = np.column_stack([m.predict(x) for m in self.models])
        return np.asarray(np.clip(_from_delta(df, delta), 0.0, None), dtype=float)

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return self.band.apply(df["reservoir_id"].to_numpy(), self._point(df))


class GradientBoostingQuantile:
    """One histogram gradient boosting model per horizon and quantile, on the log change.

    Depth and number of iterations are chosen on validation absolute error (median, three
    horizons), never on test."""

    name = "gbm_quantile"
    GRID: tuple[tuple[int, int], ...] = ((150, 3), (300, 4))

    @staticmethod
    def _make(q: float, it: int, depth: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_iter=it,
            max_depth=depth,
            learning_rate=0.05,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=SEED,
        )

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame) -> GradientBoostingQuantile:
        self.basins = sorted(train["basin"].unique().to_list())
        xt, xv = feature_matrix(train, self.basins), feature_matrix(valid, self.basins)
        yt, yv = _delta_targets(train), _delta_targets(valid)

        def valid_loss(it: int, depth: int) -> float:
            total = 0.0
            for h in (0, 6, 13):
                ok_t, ok_v = np.isfinite(yt[:, h]), np.isfinite(yv[:, h])
                m = self._make(0.5, it, depth).fit(xt[ok_t], yt[ok_t, h])
                total += float(np.mean(np.abs(m.predict(xv[ok_v]) - yv[ok_v, h])))
            return total

        self.params = min(self.GRID, key=lambda p: valid_loss(*p))
        it, depth = self.params
        self.models: dict[tuple[int, float], HistGradientBoostingRegressor] = {}
        for h in range(H):
            ok = np.isfinite(yt[:, h])
            for q in (0.1, 0.5, 0.9):
                self.models[(h, q)] = self._make(q, it, depth).fit(xt[ok], yt[ok, h])
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        x = feature_matrix(df, self.basins)
        delta = np.stack(
            [
                np.column_stack([self.models[(h, q)].predict(x) for h in range(H)])
                for q in (0.1, 0.5, 0.9)
            ],
            axis=2,
        )
        delta.sort(axis=2)  # remove quantile crossing
        out = np.stack([_from_delta(df, delta[:, :, i]) for i in range(3)], axis=2)
        return np.asarray(np.clip(out, 0.0, None), dtype=float)


def to_forecast_frame(df: pl.DataFrame, model: str, pred: np.ndarray) -> pl.DataFrame:
    """Long format: one row per origin, reservoir, horizon (observed value, P10/P50/P90)."""
    obs = targets(df)
    n = df.height
    return pl.DataFrame(
        {
            "model": [model] * (n * H),
            "reservoir_id": np.repeat(df["reservoir_id"].to_numpy(), H),
            "origin_day": np.repeat(df["origin_day"].to_numpy(), H),
            "split": np.repeat(df["split"].to_numpy(), H),
            "h": np.tile(np.arange(1, H + 1), n),
            "obs": obs.reshape(-1),
            "p10": pred[:, :, 0].reshape(-1),
            "p50": pred[:, :, 1].reshape(-1),
            "p90": pred[:, :, 2].reshape(-1),
        }
    ).with_columns(pl.col("obs").fill_nan(None))
