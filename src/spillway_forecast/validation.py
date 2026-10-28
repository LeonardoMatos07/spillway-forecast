"""Temporal splits: rolling-origin (expanding window) folds and a held-out final test."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from spillway_forecast.config import Protocol


@dataclass(frozen=True)
class Split:
    name: str
    eval_start: date
    eval_end: date | None  # exclusive; None = open ended (final test)

    def train_mask(self, data: pd.DataFrame) -> pd.Series:
        """Training rows: the *target hour* must lie strictly before the evaluation block.

        This is stricter than filtering on the origin: a row issued 3 days before the
        split whose horizon reaches into the evaluation block is excluded (embargo).
        """
        return data["target_ts"] < pd.Timestamp(self.eval_start)

    def eval_mask(self, data: pd.DataFrame) -> pd.Series:
        mask = data["origin"] >= pd.Timestamp(self.eval_start)
        if self.eval_end is not None:
            mask &= data["origin"] < pd.Timestamp(self.eval_end)
        return mask


def _add_months(d: date, months: int) -> date:
    idx = d.year * 12 + d.month - 1 + months
    return date(idx // 12, idx % 12 + 1, 1)


def cv_splits(protocol: Protocol) -> list[Split]:
    """Expanding-window folds; every fold ends no later than the final test start."""
    splits = []
    for i, start in enumerate(protocol.fold_starts):
        end = min(_add_months(start, protocol.fold_months), protocol.test_start)
        splits.append(Split(f"fold{i + 1}", start, end))
    return splits


def final_split(protocol: Protocol) -> Split:
    return Split("test", protocol.test_start, None)
