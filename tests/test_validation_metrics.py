from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from spillway_forecast.baselines import ClimatologyBaseline, hourly_observations
from spillway_forecast.config import PROTOCOL, Protocol
from spillway_forecast.features import build_dataset, to_series
from spillway_forecast.metrics import (
    classification_metrics,
    expected_calibration_error,
    paired_block_bootstrap,
    regression_metrics,
)
from spillway_forecast.validation import cv_splits, final_split


def test_folds_are_expanding_ordered_and_inside_dev_period() -> None:
    splits = cv_splits(PROTOCOL)
    starts = [s.eval_start for s in splits]
    assert starts == sorted(starts)
    for s in splits:
        assert s.eval_end is not None
        assert s.eval_start < s.eval_end <= PROTOCOL.test_start
    assert final_split(PROTOCOL).eval_start == PROTOCOL.test_start


def test_no_training_target_reaches_the_evaluation_block(hourly: pd.DataFrame) -> None:
    data = build_dataset(to_series(hourly), stride_days=3)
    protocol = Protocol(test_start=date(2024, 12, 1))
    split = final_split(protocol)
    train = data[split.train_mask(data)]
    evald = data[split.eval_mask(data)]
    assert train["target_ts"].max() < pd.Timestamp(split.eval_start)
    assert train["origin"].max() < evald["origin"].min()
    assert not train.empty and not evald.empty


def test_climatology_fit_ignores_future_observations(hourly: pd.DataFrame) -> None:
    cutoff = pd.Timestamp("2024-10-01")
    obs = hourly_observations(hourly)
    rows = obs[obs["ts"] >= cutoff].head(50)
    a = ClimatologyBaseline().fit(obs[obs["ts"] < cutoff]).predict(rows)
    shuffled = hourly.copy()
    future = shuffled["ts"] >= cutoff
    shuffled.loc[future, "curtailed_mwh"] *= 3
    obs2 = hourly_observations(shuffled)
    b = ClimatologyBaseline().fit(obs2[obs2["ts"] < cutoff]).predict(rows)
    pd.testing.assert_frame_equal(a, b)


def test_metrics_perfect_and_calibration() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    perfect = classification_metrics(y, y)
    assert perfect["roc_auc"] == 1.0 and perfect["brier"] == 0.0 and perfect["pr_auc"] == 1.0
    const = classification_metrics(y, np.full(4, 0.5))
    assert abs(const["brier_skill"]) < 1e-9
    assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
    reg = regression_metrics(np.array([0.1, 0.2]), np.array([0.2, 0.2]))
    assert np.isclose(reg["mae_pp"], 5.0)


def test_bootstrap_detects_a_clearly_better_model() -> None:
    rng = np.random.default_rng(0)
    blocks = np.repeat(np.arange(40), 25).astype(np.int64)
    a = rng.uniform(0, 0.1, len(blocks))
    b = a + 0.2
    res = paired_block_bootstrap(a, b, blocks, n_boot=200)
    assert res["ci_high"] < 0
