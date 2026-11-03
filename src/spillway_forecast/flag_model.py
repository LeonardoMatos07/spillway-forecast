"""Stage 4 model code for the tariff flag forecast: features, predictive distributions, colours.

The model predicts the official PLD trigger (a regression, with uncertainty) and the GSF of the
month; the colour probabilities come from applying the month's bands (``flag_rule``) to samples
of both. The colour is never a training label (spec).

Candidates, in the order fixed by the spec: (1) regularised linear regression with leave-one-out
residuals, (2) linear quantile regression, (3) a small gradient boosting for quantiles, which is
only tried when a linear candidate already beats the references (see ``flag_eval``).

Honest limits: no DECOMP rounds and no ONS hydro forecast are available, so the trigger and the
GSF are approximated from storage (EAR), inflow (ENA) and PLD already published.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor

from spillway_forecast.flag_rule import FLAG_ORDER, RuleVersion, compute_bands

FloatArray = NDArray[np.float64]

FEATURES = (
    "log_pld_last",
    "log_pld_mean4",
    "ear_se",
    "ear_sin",
    "ena_se_30",
    "ear_chg_30",
    "prev_flag_ord",
    "moy_sin",
    "moy_cos",
)
CORE_FEATURES = ("log_pld_last", "ear_se")  # a month without these is not scored
RIDGE_ALPHA = 5.0  # fixed in advance, no tuning
QUANTILE_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
GSF_BOUNDS = (0.30, 1.40)
LABEL_LAG_DAYS = 10  # a month's official trigger is read from the bulletin ~6 days after it starts
FLAG_ORDINAL = {flag.value: i for i, flag in enumerate(FLAG_ORDER)}
FLAG_ORDINAL["Escassez Hídrica"] = 3


def colour_index(name: str) -> int:
    return FLAG_ORDINAL[name]


def default_announcement(month: date) -> date:
    """Typical ANEEL announcement: the last weekday up to 3 days before the month starts."""
    day = month - timedelta(days=3)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def label_available_on(month: date) -> date:
    return month + timedelta(days=LABEL_LAG_DAYS)


def build_features(
    cutoff: date,
    target_month: date,
    prev_flag: str | None,
    pld_weekly: pd.DataFrame,
    storage_daily: pd.DataFrame,
) -> dict[str, float]:
    """Features for ``target_month`` using ONLY data available at ``cutoff``.

    ``pld_weekly``: week_start (date), subsystem, pld. A week is usable once it has started.
    ``storage_daily``: day (date), subsystem, storage_pct, ena_gross_pct_mlt. Usable up to D-1.
    """
    out = dict.fromkeys(FEATURES, float("nan"))
    pld = pld_weekly[(pld_weekly["subsystem"] == "SE") & (pld_weekly["week_start"] <= cutoff)]
    pld = pld.sort_values("week_start")
    if len(pld):
        last4 = pld["pld"].to_numpy(dtype=float)[-4:]
        out["log_pld_last"] = float(np.log(max(last4[-1], 1.0)))
        out["log_pld_mean4"] = float(np.log(max(last4.mean(), 1.0)))
    cut = cutoff - timedelta(days=1)
    stor = storage_daily[storage_daily["day"] <= cut]
    se = stor[stor["subsystem"] == "SE"].sort_values("day")
    if len(se):
        out["ear_se"] = float(se["storage_pct"].iloc[-1])
        recent = se[se["day"] > cut - timedelta(days=30)]
        if len(recent):
            out["ena_se_30"] = float(recent["ena_gross_pct_mlt"].mean())
        old = se[se["day"] <= cut - timedelta(days=30)]
        if len(old):
            out["ear_chg_30"] = float(se["storage_pct"].iloc[-1] - old["storage_pct"].iloc[-1])
        latest = stor[stor["day"] == se["day"].iloc[-1]]
        if len(latest) == 4:
            out["ear_sin"] = float(latest["storage_pct"].mean())
    if prev_flag is not None:
        out["prev_flag_ord"] = float(colour_index(prev_flag))
    angle = 2 * np.pi * (target_month.month - 1) / 12
    out["moy_sin"] = float(np.sin(angle))
    out["moy_cos"] = float(np.cos(angle))
    return out


@dataclass(frozen=True)
class Predictive:
    """Equal-weight samples of the trigger (R$/MWh) and of the GSF for one month."""

    trigger: FloatArray
    gsf: FloatArray


def colour_probabilities(
    pred: Predictive,
    pld_min: float | Decimal,
    pld_max: float | Decimal,
    rule: RuleVersion,
) -> FloatArray:
    """P(colour) for [green, yellow, red 1, red 2]: every trigger x GSF pair is one scenario."""
    lo, hi = float(pld_min), float(pld_max)
    trig = np.clip(np.round(pred.trigger, 2), lo, hi)
    gsf = np.round(np.clip(pred.gsf, *GSF_BOUNDS), 2)
    counts = np.zeros(4)
    unique, n_each = np.unique(gsf, return_counts=True)
    for g, k in zip(unique, n_each, strict=True):
        bands = compute_bands(str(lo), str(hi), str(g), rule)
        edges = np.array(
            [float(bands.green_upper), float(bands.yellow_upper), float(bands.red1_upper)]
        )
        idx = np.searchsorted(edges, trig, side="left")  # edges are inclusive on the upper side
        counts += np.bincount(idx, minlength=4)[:4] * k
    return counts / counts.sum()


class Model(Protocol):
    name: str

    def fit(self, x: FloatArray, y_trigger_log: FloatArray, y_gsf: FloatArray) -> None: ...

    def predict(self, x_row: FloatArray) -> Predictive: ...


def _prepare(x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    filled = np.where(np.isnan(x), med, x)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return med, mean, std


class _Standardiser:
    def fit(self, x: FloatArray) -> FloatArray:
        self.med, self.mean, self.std = _prepare(x)
        return self.transform(x)

    def transform(self, x: FloatArray) -> FloatArray:
        filled = np.where(np.isnan(x), self.med, x)
        return np.asarray((filled - self.mean) / self.std, dtype=np.float64)


class RidgeLoo:
    """Ridge on standardised features; predictive spread = leave-one-out residuals."""

    name = "ridge"

    def fit(self, x: FloatArray, y_trigger_log: FloatArray, y_gsf: FloatArray) -> None:
        self._std = _Standardiser()
        z = self._std.fit(x)
        n = len(z)
        self._zbar = z.mean(axis=0)
        zc = z - self._zbar
        gram = zc.T @ zc + RIDGE_ALPHA * np.eye(z.shape[1])
        inv = np.linalg.inv(gram)
        hat = 1.0 / n + np.einsum("ij,jk,ik->i", zc, inv, zc)
        self._coefs: list[tuple[FloatArray, float]] = []
        self._loo: list[FloatArray] = []
        for y in (y_trigger_log, y_gsf):
            ybar = float(y.mean())
            beta = inv @ zc.T @ (y - ybar)
            resid = y - (ybar + zc @ beta)
            self._coefs.append((beta, ybar))
            self._loo.append(resid / np.clip(1.0 - hat, 0.05, None))

    def predict(self, x_row: FloatArray) -> Predictive:
        z = self._std.transform(x_row.reshape(1, -1))[0] - self._zbar
        mu = [float(z @ beta + ybar) for beta, ybar in self._coefs]
        return Predictive(
            trigger=np.exp(mu[0] + self._loo[0]), gsf=np.asarray(mu[1] + self._loo[1])
        )


class LinearQuantile:
    """Linear quantile regression on a fixed grid of quantiles (L1-penalised, alpha fixed)."""

    name = "linear_quantile"

    def __init__(self, quantiles: Sequence[float] = QUANTILE_GRID) -> None:
        self.quantiles = tuple(quantiles)

    def fit(self, x: FloatArray, y_trigger_log: FloatArray, y_gsf: FloatArray) -> None:
        self._std = _Standardiser()
        z = self._std.fit(x)
        self._models = [
            [
                QuantileRegressor(quantile=q, alpha=0.02, solver="highs").fit(z, y)
                for q in self.quantiles
            ]
            for y in (y_trigger_log, y_gsf)
        ]

    def predict(self, x_row: FloatArray) -> Predictive:
        z = self._std.transform(x_row.reshape(1, -1))
        trig, gsf = (
            np.sort(np.array([float(m.predict(z)[0]) for m in models])) for models in self._models
        )
        return Predictive(trigger=np.exp(trig), gsf=gsf)


class BoostingQuantile:
    """Small gradient boosting for quantiles (depth 2, 50 trees). Only tried if linear wins."""

    name = "boosting_quantile"

    def __init__(self, quantiles: Sequence[float] = QUANTILE_GRID, n_estimators: int = 50) -> None:
        self.quantiles = tuple(quantiles)
        self.n_estimators = n_estimators

    def fit(self, x: FloatArray, y_trigger_log: FloatArray, y_gsf: FloatArray) -> None:
        self._std = _Standardiser()
        z = self._std.fit(x)
        self._models = [
            [
                GradientBoostingRegressor(
                    loss="quantile",
                    alpha=q,
                    n_estimators=self.n_estimators,
                    max_depth=2,
                    learning_rate=0.05,
                    min_samples_leaf=3,
                    random_state=0,
                ).fit(z, y)
                for q in self.quantiles
            ]
            for y in (y_trigger_log, y_gsf)
        ]

    def predict(self, x_row: FloatArray) -> Predictive:
        z = self._std.transform(x_row.reshape(1, -1))
        trig, gsf = (
            np.sort(np.array([float(m.predict(z)[0]) for m in models])) for models in self._models
        )
        return Predictive(trigger=np.exp(trig), gsf=gsf)


def make_model(name: str) -> Model:
    if name == "ridge":
        return RidgeLoo()
    if name == "linear_quantile":
        return LinearQuantile()
    if name == "boosting_quantile":
        return BoostingQuantile()
    raise ValueError(f"unknown model {name!r}")
