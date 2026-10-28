"""Synthetic fixtures. They only exercise code paths; no result is reported from them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_hourly(days: int = 420, start: str = "2024-01-01", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    frames = []
    for source, scope in (("wind", "NE"), ("wind", "RS"), ("solar", "NE")):
        hour = idx.hour.to_numpy()
        if source == "solar":
            gen = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 800 + rng.normal(
                0, 5, len(idx)
            )
            gen = np.clip(gen, 0, None)
            night = (hour < 6) | (hour > 18)
            gen[night] = 0.0
        else:
            gen = 500 + 200 * np.sin(idx.day_of_year.to_numpy() / 58) + rng.normal(0, 40, len(idx))
        share = np.clip(
            0.05 + 0.2 * (hour >= 10) * (hour <= 15) + rng.normal(0, 0.05, len(idx)), 0, 0.9
        )
        curt = gen * share
        if source == "solar":
            curt[(hour < 6) | (hour > 18)] = 0.0
        frames.append(
            pd.DataFrame(
                {
                    "ts": idx,
                    "scope_type": "subsystem",
                    "scope": scope,
                    "source": source,
                    "curtailed_mwh": curt,
                    "generation_mwh": gen,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def hourly() -> pd.DataFrame:
    return make_hourly()
