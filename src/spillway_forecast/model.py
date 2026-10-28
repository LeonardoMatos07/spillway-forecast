"""Gradient boosting model (sklearn HistGradientBoosting) and its on-disk bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from spillway_forecast.config import RISK_THRESHOLD, models_dir
from spillway_forecast.features import FEATURES, as_model_frame

BUNDLE_NAME = "bundle.joblib"

GBM_PARAMS: dict[str, Any] = {
    "max_iter": 250,
    "learning_rate": 0.06,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 200,
    "l2_regularization": 1.0,
    # No internal early stopping: sklearn would carve a *random* validation split out of
    # the training rows, which is exactly the temporal leakage this project avoids.
    "early_stopping": False,
    "categorical_features": "from_dtype",
}


@dataclass
class ForecastModel:
    series_ids: list[str]
    random_state: int = 42
    classifier: HistGradientBoostingClassifier | None = None
    regressor: HistGradientBoostingRegressor | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def fit(self, data: pd.DataFrame) -> ForecastModel:
        x = as_model_frame(data, self.series_ids)
        self.classifier = HistGradientBoostingClassifier(
            random_state=self.random_state, **GBM_PARAMS
        ).fit(x, data["risk"].astype(int))
        self.regressor = HistGradientBoostingRegressor(
            random_state=self.random_state, **GBM_PARAMS
        ).fit(x, data["pct"])
        return self

    def predict(self, data: pd.DataFrame) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(expected curtailment fraction, risk probability)``."""
        if self.classifier is None or self.regressor is None:
            raise RuntimeError("model is not fitted")
        x = as_model_frame(data, self.series_ids)
        pct = np.clip(self.regressor.predict(x), 0.0, 1.0)
        risk = self.classifier.predict_proba(x)[:, 1]
        return pct.astype(np.float64), risk.astype(np.float64)

    def save(self, path: Path | None = None) -> Path:
        path = path or models_dir() / BUNDLE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self,
                "features": FEATURES,
                "risk_threshold": RISK_THRESHOLD,
            },
            path,
        )
        return path

    @staticmethod
    def load(path: Path | None = None) -> ForecastModel:
        path = path or models_dir() / BUNDLE_NAME
        bundle = joblib.load(path)
        model = bundle["model"]
        if not isinstance(model, ForecastModel):
            raise TypeError("invalid bundle")
        return model
