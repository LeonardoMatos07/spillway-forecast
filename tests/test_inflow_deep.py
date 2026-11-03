"""Stage 3: the network's data path (no leakage) and a tiny CPU training smoke test."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from spillway_forecast import inflow_dataset as ds  # noqa: E402
from spillway_forecast import inflow_deep as dp  # noqa: E402


def _inputs(n: int = 700) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(0)
    days = [date(2002, 1, 1) + timedelta(days=i) for i in range(n)]
    q = 50 + 20 * np.sin(np.arange(n) / 40) + rng.normal(0, 1, n)
    rd = pl.DataFrame(
        {
            "reservoir_id": ["AAA"] * n,
            "day": days,
            "natural_inflow_m3s": q,
            "storage_pct": [50.0] * n,
        }
    )
    reg = pl.DataFrame(
        {
            "reservoir_id": ["AAA"],
            "basin": ["X"],
            "subsystem": ["SE"],
            "useful_volume_hm3": [500.0],
            "latitude": [-20.0],
            "longitude": [-47.0],
        }
    )
    rain = pl.DataFrame(
        {
            "reservoir_id": ["AAA"] * n,
            "day": days,
            "precip_mm": rng.gamma(1.0, 3.0, n),
            "run": ["final"] * n,
        }
    )
    return rd, reg, rain


def _config() -> ds.DatasetConfig:
    return ds.DatasetConfig(
        train_start=date(2002, 1, 1),
        train_end=date(2002, 12, 31),
        valid_end=date(2003, 6, 30),
        test_end=date(2003, 12, 31),
    )


def test_window_ends_on_origin_and_rain_is_lagged() -> None:
    rd, reg, rain = _inputs()
    cfg = _config()
    panel = dp.build_panel(rd, reg, rain, cfg)
    row = panel.index[("AAA", date(2002, 9, 1))]
    # channel 0 is the standardised log flow of the day itself
    mu, sd = panel.scale["AAA"]
    z = (np.log1p(rd.filter(pl.col("day") == date(2002, 9, 1))["natural_inflow_m3s"][0]) - mu) / sd
    assert float(panel.x[row, 0]) == pytest.approx(z, abs=1e-4)
    # channel 3 on the origin day is the rain of origin - 46 days
    lagged = rain.filter(pl.col("day") == date(2002, 9, 1) - timedelta(days=46))["precip_mm"][0]
    assert float(panel.x[row, 3]) == pytest.approx(np.log1p(lagged), abs=1e-4)


def test_changing_future_does_not_change_past_inputs_or_scale() -> None:
    rd, reg, rain = _inputs()
    cfg = _config()
    a = dp.build_panel(rd, reg, rain, cfg)
    later = rd.with_columns(
        pl.when(pl.col("day") > date(2003, 1, 1))
        .then(pl.col("natural_inflow_m3s") * 9)
        .otherwise(pl.col("natural_inflow_m3s"))
        .alias("natural_inflow_m3s")
    )
    b = dp.build_panel(later, reg, rain, cfg)
    assert a.scale == b.scale  # normalisation uses training years only
    cut = a.index[("AAA", date(2003, 1, 1))]
    assert torch.equal(a.x[: cut + 1], b.x[: cut + 1])


def test_no_rain_panel_has_no_rain_signal() -> None:
    rd, reg, rain = _inputs()
    panel = dp.build_panel(rd, reg, rain, _config(), use_rain=False)
    assert float(panel.x[:, 3].abs().sum()) == 0.0


def test_pinball_loss_masks_missing_and_trains() -> None:
    pred = torch.zeros(2, 14, 3)
    y = torch.full((2, 14), float("nan"))
    assert float(dp.pinball_loss(pred, y)) == 0.0
    y[0, 0] = 1.0
    assert float(dp.pinball_loss(pred, y)) > 0

    rd, reg, rain = _inputs()
    cfg = _config()
    feats = ds.build_features(rd, reg, cfg, rain)
    panel = dp.build_panel(rd, reg, rain, cfg)
    tr = dp.make_samples(panel, feats.filter(pl.col("split") == "train"))
    va = dp.make_samples(panel, feats.filter(pl.col("split") == "valid"))
    small = replace(
        dp.DeepConfig(), epochs=2, samples_per_epoch=256, hidden=8, window=30, threads=1
    )
    net, info = dp.fit_one(panel, tr, va, small, seed=1)
    out = dp.to_flow(dp.predict_raw(net, panel, va, small), va)
    assert out.shape == (va.rows.shape[0], 14, 3)
    assert (out[:, :, 0] <= out[:, :, 1] + 1e-9).all() and (
        out[:, :, 1] <= out[:, :, 2] + 1e-9
    ).all()
    _, info2 = dp.fit_one(panel, tr, va, small, seed=1)
    assert info["valid_pinball"] == pytest.approx(info2["valid_pinball"])  # fixed seed
