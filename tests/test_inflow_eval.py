"""Stage 2: metrics, reference models and the publication gate (synthetic data, no network)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from spillway_forecast import inflow_eval as ev
from spillway_forecast import inflow_models as im


def test_nse_kge_known_values() -> None:
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    assert ev.nse(obs, obs) == pytest.approx(1.0)
    assert ev.nse(obs, np.full(4, obs.mean())) == pytest.approx(0.0)
    assert ev.kge(obs, obs) == pytest.approx(1.0)
    assert ev.kge(obs, obs * 2) < 1.0  # bias and variability ratios are 2


def test_series_metrics_coverage_bias_and_peak() -> None:
    obs = np.arange(1.0, 101.0)
    m = ev.series_metrics(obs, obs - 1, obs, obs + 1)
    assert m["coverage"] == 1.0
    assert m["bias_pct"] == pytest.approx(0.0)
    assert m["nse"] == pytest.approx(1.0)
    over = ev.series_metrics(obs, obs, obs * 1.1, obs * 1.2)
    assert over["bias_pct"] == pytest.approx(10.0)
    assert over["peak_err_pct"] == pytest.approx(10.0)


def _synthetic(n_days: int = 3500, seed: int = 1) -> pl.DataFrame:
    """Two reservoirs with seasonal flow and noise, in the dataset-builder layout."""
    rng = np.random.default_rng(seed)
    start = date(2002, 1, 1)
    rows = []
    for rid, level in (("AAA", 100.0), ("BBB", 40.0)):
        days = [start + timedelta(days=i) for i in range(n_days + 15)]
        doy = np.array([d.timetuple().tm_yday for d in days])
        q = (
            level
            * (1.5 + np.sin(2 * np.pi * doy / 365.25))
            * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))) * 0.2)
        )
        for i in range(n_days):
            origin = days[i]
            if origin <= date(2009, 12, 31):
                split = "train"
            elif origin <= date(2012, 12, 31):
                split = "valid"
            else:
                split = "test"
            row = {
                "reservoir_id": rid,
                "origin_day": origin,
                "split": split,
                "q_lag0": q[i],
                "q_lag1": q[i - 1] if i else q[i],
                "q_lag2": q[i - 2] if i > 1 else q[i],
                "q_lag6": q[i - 6] if i > 5 else q[i],
                "q_mean7": q[max(0, i - 6) : i + 1].mean(),
                "q_mean30": q[max(0, i - 29) : i + 1].mean(),
                "q_mean90": q[max(0, i - 89) : i + 1].mean(),
                "q_mean_train": level * 1.5,
                "ear_pct": 50.0,
                "doy_sin": float(np.sin(2 * np.pi * doy[i] / 365.25)),
                "doy_cos": float(np.cos(2 * np.pi * doy[i] / 365.25)),
                "basin": "X",
                "useful_volume_hm3": 1000.0,
            }
            row |= {f"y_{h}": q[i + h] for h in range(1, 15)}
            rows.append(row)
    return pl.DataFrame(rows)


def test_persistence_and_recession_shapes_and_band_order() -> None:
    df = _synthetic()
    train = df.filter(pl.col("split") == "train")
    for model in (im.Persistence(), im.Recession(), im.MovingAverage(), im.Climatology()):
        pred = model.fit(train).predict(df.filter(pl.col("split") == "valid"))
        assert pred.shape[1:] == (14, 3)
        ok = np.isfinite(pred).all(axis=(1, 2))
        assert ok.all()
        assert (pred[:, :, 0] <= pred[:, :, 1] + 1e-9).all()
        assert (pred[:, :, 1] <= pred[:, :, 2] + 1e-9).all()


def test_persistence_is_best_at_one_day_and_climatology_wins_later_on_pure_season() -> None:
    df = _synthetic()
    train = df.filter(pl.col("split") == "train")
    valid = df.filter(pl.col("split") == "valid")
    p = im.Persistence().fit(train).predict(valid)[:, :, 1]
    c = im.Climatology().fit(train).predict(valid)[:, :, 1]
    obs = im.targets(valid)
    err_p = np.abs(p - obs).mean(axis=0)
    err_c = np.abs(c - obs).mean(axis=0)
    assert err_p[0] < err_c[0]


def test_models_do_not_see_validation_or_test_rows() -> None:
    df = _synthetic()
    train = df.filter(pl.col("split") == "train")
    valid = df.filter(pl.col("split") == "valid")
    corrupted = valid.with_columns(
        [pl.col(f"y_{h}") * 5 for h in range(1, 15)]
    )  # reference models never read validation
    a = im.Climatology().fit(train).predict(valid)
    b = im.Climatology().fit(train).predict(corrupted)
    assert np.allclose(a, b, equal_nan=True)
    persist_a = im.Persistence().fit(train).predict(valid)
    persist_b = im.Persistence().fit(train).predict(corrupted)
    assert np.allclose(persist_a, persist_b)


def test_ridge_and_gbm_fit_and_gbm_quantiles_do_not_cross() -> None:
    df = _synthetic()
    train = df.filter(pl.col("split") == "train")
    valid = df.filter(pl.col("split") == "valid")
    ridge = im.RidgeDirect().fit(train, valid)
    assert ridge.predict(valid).shape[1:] == (14, 3)
    gbm = im.GradientBoostingQuantile()
    gbm.GRID = ((20, 2),)
    pred = gbm.fit(train, valid).predict(valid)
    assert (pred[:, :, 0] <= pred[:, :, 1] + 1e-9).all()
    assert (pred[:, :, 1] <= pred[:, :, 2] + 1e-9).all()


def _table(
    model: str, h: int, nse: float, bias: float, peak: float, cov: float, n: int = 10
) -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "reservoir_id": f"R{i}",
            "h": h,
            "nse": nse,
            "kge": nse,
            "bias_pct": bias,
            "peak_err_pct": peak,
            "coverage": cov,
        }
        for i in range(n)
    ]


def test_gate_passes_and_fails_on_each_condition() -> None:
    valid = pl.DataFrame(_table("ref", 1, 0.5, 5, 5, 0.8) + _table("cand", 1, 0.7, 1, 1, 0.8))
    base = _table("ref", 1, 0.5, 5, 5, 0.8)
    good = ev.gate(pl.DataFrame(base + _table("cand", 1, 0.7, 1, 1, 0.8)), valid, "cand", ["ref"])
    assert good["publish"][0]
    cases = {
        "nse": _table("cand", 1, 0.4, 1, 1, 0.8),
        "bias": _table("cand", 1, 0.7, 9, 1, 0.8),
        "peak": _table("cand", 1, 0.7, 1, 9, 0.8),
        "coverage": _table("cand", 1, 0.7, 1, 1, 0.95),
    }
    for name, cand in cases.items():
        res = ev.gate(pl.DataFrame(base + cand), valid, "cand", ["ref"])
        assert not res["publish"][0], name
    mixed = _table("cand", 1, 0.7, 1, 1, 0.8)
    for row in mixed[:4]:  # 40% of reservoirs worse than the reference
        row["nse"] = 0.1
    res = ev.gate(pl.DataFrame(base + mixed), valid, "cand", ["ref"])
    assert not res["share_worse_ok"][0]


def test_best_reference_is_chosen_on_validation_only() -> None:
    valid = pl.DataFrame(_table("a", 1, 0.9, 0, 0, 0.8) + _table("b", 1, 0.2, 0, 0, 0.8))
    assert ev.best_reference(valid, ["a", "b"], 1) == "a"
