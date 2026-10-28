"""Regression and probabilistic classification metrics, plus a paired block bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

FloatArray = NDArray[np.float64]


def reliability_table(y: FloatArray, p: FloatArray, bins: int = 10) -> pd.DataFrame:
    """Equal-width reliability table: mean predicted vs observed frequency per bin."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append(
                {
                    "bin": b,
                    "n": int(m.sum()),
                    "mean_pred": float(p[m].mean()),
                    "obs": float(y[m].mean()),
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(y: FloatArray, p: FloatArray, bins: int = 10) -> float:
    table = reliability_table(y, p, bins)
    if table.empty:
        return float("nan")
    return float((table["n"] * (table["mean_pred"] - table["obs"]).abs()).sum() / table["n"].sum())


def regression_metrics(y: FloatArray, pred: FloatArray) -> dict[str, float]:
    err = pred - y
    return {
        "mae_pp": float(np.mean(np.abs(err)) * 100),  # percentage points of curtailment
        "rmse_pp": float(np.sqrt(np.mean(err**2)) * 100),
        "bias_pp": float(np.mean(err) * 100),
    }


def classification_metrics(y: FloatArray, p: FloatArray) -> dict[str, float]:
    base = float(y.mean())
    out = {"n": float(len(y)), "base_rate": base}
    if len(np.unique(y)) < 2:
        return {
            **out,
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "brier": float(brier_score_loss(y, p)),
            "ece": expected_calibration_error(y, p),
            "brier_skill": float("nan"),
        }
    brier = float(brier_score_loss(y, p))
    ref = base * (1 - base)  # Brier of the constant "base rate" forecast
    return {
        **out,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": brier,
        "ece": expected_calibration_error(y, p),
        "brier_skill": float(1 - brier / ref) if ref > 0 else float("nan"),
    }


def evaluate(rows: pd.DataFrame, pred_pct: FloatArray, pred_risk: FloatArray) -> dict[str, float]:
    y_pct = rows["pct"].to_numpy(dtype=float)
    y_risk = rows["risk"].to_numpy(dtype=float)
    return {
        **regression_metrics(y_pct, pred_pct),
        **classification_metrics(y_risk, np.clip(pred_risk, 0.0, 1.0)),
    }


def paired_block_bootstrap(
    loss_a: FloatArray,
    loss_b: FloatArray,
    blocks: NDArray[np.int64],
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """CI of ``mean(loss_a) - mean(loss_b)`` resampling whole time blocks (e.g. weeks).

    Negative values mean model A has the lower loss. Blocks respect serial correlation.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(blocks, return_inverse=True)
    diff = loss_a - loss_b
    sums = np.bincount(inv, weights=diff, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        stats[i] = sums[pick].sum() / cnts[pick].sum()
    return {
        "delta": float(diff.mean()),
        "ci_low": float(np.percentile(stats, 2.5)),
        "ci_high": float(np.percentile(stats, 97.5)),
    }
