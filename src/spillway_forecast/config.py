"""Central configuration: paths, modelling constants and the validation protocol."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ONS_BASE_URL = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset"
HALF_HOUR = 0.5  # raw rows are 30-minute average MW -> MWh per row

# source -> (ONS dataset folder, file prefix, first month with data)
ONS_DATASETS: dict[str, tuple[str, str, tuple[int, int]]] = {
    "wind": ("restricao_coff_eolica_tm", "RESTRICAO_COFF_EOLICA", (2021, 10)),
    "solar": ("restricao_coff_fotovoltaica_tm", "RESTRICAO_COFF_FOTOVOLTAICA", (2024, 4)),
}

HORIZON_HOURS = 168  # forecast issued at 00:00 covers the next 7 days
MIN_DENOMINATOR_MWH = 1.0  # below this (night-time solar) the curtailment % is undefined
RISK_THRESHOLD = 0.10  # an hour is "at risk" when >= 10% of potential output is cut
MIN_SERIES_HOURS = 24 * 120  # drop series with less than ~4 months of history


def data_dir() -> Path:
    return Path(os.environ.get("SPILLWAY_DATA_DIR", REPO_ROOT / "data"))


def models_dir() -> Path:
    return Path(os.environ.get("SPILLWAY_MODELS_DIR", REPO_ROOT / "models"))


def hourly_path() -> Path:
    return data_dir() / "processed" / "hourly.parquet"


def mlflow_uri() -> str:
    default = f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
    return os.environ.get("MLFLOW_TRACKING_URI", default)


@dataclass(frozen=True)
class Protocol:
    """Temporal validation protocol.

    The final test period is never touched during development. Rolling-origin folds
    use expanding training windows that end strictly before each validation block.
    """

    test_start: date = date(2026, 4, 1)
    fold_starts: tuple[date, ...] = (
        date(2025, 1, 1),
        date(2025, 4, 1),
        date(2025, 7, 1),
        date(2025, 10, 1),
        date(2026, 1, 1),
    )
    fold_months: int = 3
    origin_stride_days: int = 3
    random_state: int = 42


PROTOCOL = Protocol()
