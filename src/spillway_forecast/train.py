"""Backtest (rolling origin + held-out test), MLflow tracking and model registration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from spillway_forecast import config
from spillway_forecast.baselines import (
    ClimatologyBaseline,
    hourly_observations,
    seasonal_naive,
)
from spillway_forecast.config import PROTOCOL, REPO_ROOT, Protocol
from spillway_forecast.features import FEATURES, as_model_frame, build_dataset, to_series
from spillway_forecast.metrics import (
    evaluate,
    paired_block_bootstrap,
    reliability_table,
)
from spillway_forecast.model import GBM_PARAMS, ForecastModel
from spillway_forecast.validation import Split, cv_splits, final_split

MODELS = ("climatology", "seasonal_naive", "hist_gbm")
EXPERIMENT = "spillway-forecast"
REPORTS = Path(os.environ.get("SPILLWAY_REPORTS_DIR", REPO_ROOT / "reports"))


def predict_all(
    model: ForecastModel, climatology: ClimatologyBaseline, rows: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    clim = climatology.predict(rows)
    snaive = seasonal_naive(rows, clim)
    pct, risk = model.predict(rows)
    gbm = pd.DataFrame({"pct": pct, "risk": risk}, index=rows.index)
    return {"climatology": clim, "seasonal_naive": snaive, "hist_gbm": gbm}


def run_split(
    split: Split, data: pd.DataFrame, obs: pd.DataFrame, series_ids: list[str], seed: int
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit everything on data strictly before ``split`` and predict the evaluation block."""
    train = data[split.train_mask(data)]
    evald = data[split.eval_mask(data)].copy()
    if train.empty or evald.empty:
        raise ValueError(f"split {split.name} has no train or eval rows")
    model = ForecastModel(series_ids=series_ids, random_state=seed).fit(train)
    climatology = ClimatologyBaseline().fit(obs[obs["ts"] < pd.Timestamp(split.eval_start)])
    preds = predict_all(model, climatology, evald)
    ok = pd.concat([p.notna().all(axis=1) for p in preds.values()], axis=1).all(axis=1)
    evald = evald[ok]
    return evald, {k: v[ok] for k, v in preds.items()}


def score(rows: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    return {
        name: evaluate(rows, p["pct"].to_numpy(float), p["risk"].to_numpy(float))
        for name, p in preds.items()
    }


def _breakdown(
    rows: pd.DataFrame, preds: dict[str, pd.DataFrame], by: pd.Series
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for key in sorted(by.unique()):
        m = (by == key).to_numpy()
        sub = rows[m]
        out[str(key)] = score(sub, {k: v[m] for k, v in preds.items()})
    return out


def calibration_figure(rows: pd.DataFrame, preds: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    y = rows["risk"].to_numpy(float)
    for name in ("climatology", "hist_gbm"):
        table = reliability_table(y, np.clip(preds[name]["risk"].to_numpy(float), 0, 1))
        ax.plot(table["mean_pred"], table["obs"], marker="o", label=name)
    ax.set_xlabel("predicted risk")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability on the held-out test period")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _agg_cv(folds: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for model in MODELS:
        out[model] = {}
        for metric in folds[0][model]:
            vals = np.array([f[model][metric] for f in folds], dtype=float)
            out[model][metric] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return out


def train(
    protocol: Protocol = PROTOCOL,
    *,
    register: bool = True,
    hourly: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if hourly is None:
        hourly = pd.read_parquet(config.hourly_path())
    series_list = to_series(hourly)
    series_ids = [s.series_id for s in series_list]
    data = build_dataset(series_list, protocol.origin_stride_days)
    obs = hourly_observations(hourly)
    obs = obs[obs["series_id"].isin(series_ids)]

    mlflow.set_tracking_uri(config.mlflow_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="backtest") as run:
        mlflow.log_params(
            {
                **{f"gbm_{k}": v for k, v in GBM_PARAMS.items()},
                "test_start": protocol.test_start.isoformat(),
                "fold_starts": ",".join(d.isoformat() for d in protocol.fold_starts),
                "fold_months": protocol.fold_months,
                "origin_stride_days": protocol.origin_stride_days,
                "horizon_hours": config.HORIZON_HOURS,
                "risk_threshold": config.RISK_THRESHOLD,
                "n_series": len(series_ids),
                "n_rows": len(data),
                "data_start": str(hourly["ts"].min()),
                "data_end": str(hourly["ts"].max()),
            }
        )

        fold_scores: list[dict[str, dict[str, float]]] = []
        for i, split in enumerate(cv_splits(protocol)):
            rows, preds = run_split(split, data, obs, series_ids, protocol.random_state)
            s = score(rows, preds)
            fold_scores.append(s)
            for m in MODELS:
                mlflow.log_metrics({f"cv_{m}_{k}": v for k, v in s[m].items()}, step=i)
            print(
                f"[{split.name}] n={len(rows)} "
                + " | ".join(
                    f"{m}: MAE={s[m]['mae_pp']:.2f} AUC={s[m]['roc_auc']:.3f} "
                    f"Brier={s[m]['brier']:.4f}"
                    for m in MODELS
                )
            )
        cv = _agg_cv(fold_scores)
        for m in MODELS:
            mlflow.log_metrics({f"cv_mean_{m}_{k}": v["mean"] for k, v in cv[m].items()})

        # ---- Held-out final test (never used for any decision above) -----------------
        t_rows, t_preds = run_split(
            final_split(protocol), data, obs, series_ids, protocol.random_state
        )
        test = score(t_rows, t_preds)
        for m in MODELS:
            mlflow.log_metrics({f"test_{m}_{k}": v for k, v in test[m].items()})
        by_source = _breakdown(t_rows, t_preds, t_rows["source"])
        by_day = _breakdown(t_rows, t_preds, ((t_rows["horizon_h"] - 1) // 24 + 1).astype(int))
        weeks = t_rows["origin"].dt.to_period("W").astype(str).to_numpy()
        blocks = pd.factorize(weeks)[0].astype(np.int64)
        y_pct = t_rows["pct"].to_numpy(float)
        y_risk = t_rows["risk"].to_numpy(float)
        bootstrap: dict[str, dict[str, dict[str, float]]] = {}
        for base in ("climatology", "seasonal_naive"):
            bootstrap[f"hist_gbm_minus_{base}"] = {
                "abs_error": paired_block_bootstrap(
                    np.abs(t_preds["hist_gbm"]["pct"].to_numpy(float) - y_pct),
                    np.abs(t_preds[base]["pct"].to_numpy(float) - y_pct),
                    blocks,
                ),
                "brier": paired_block_bootstrap(
                    (t_preds["hist_gbm"]["risk"].to_numpy(float) - y_risk) ** 2,
                    (t_preds[base]["risk"].to_numpy(float) - y_risk) ** 2,
                    blocks,
                ),
            }

        # ---- Production model: refit on everything available -------------------------
        final = ForecastModel(series_ids=series_ids, random_state=protocol.random_state).fit(data)
        final.meta = {
            "trained_through": str(hourly["ts"].max()),
            "n_rows": len(data),
            "test_metrics": test["hist_gbm"],
        }
        sample = t_rows.sample(min(15000, len(t_rows)), random_state=0)
        eval_model = ForecastModel(series_ids=series_ids, random_state=protocol.random_state).fit(
            data[final_split(protocol).train_mask(data)]
        )
        assert eval_model.classifier is not None
        imp = permutation_importance(
            eval_model.classifier,
            as_model_frame(sample, series_ids),
            sample["risk"].astype(int),
            n_repeats=3,
            random_state=0,
            scoring="neg_brier_score",
        )
        importances = (
            pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False).round(5)
        )

        results = {
            "run_id": run.info.run_id,
            "data": {
                "start": str(hourly["ts"].min()),
                "end": str(hourly["ts"].max()),
                "n_series": len(series_ids),
                "series": series_ids,
                "rows": len(data),
                "test_rows": len(t_rows),
                "test_start": protocol.test_start.isoformat(),
                "risk_threshold": config.RISK_THRESHOLD,
            },
            "cv": cv,
            "cv_folds": fold_scores,
            "test": test,
            "test_by_source": by_source,
            "test_by_horizon_day": by_day,
            "test_bootstrap": bootstrap,
            "permutation_importance_brier": importances.to_dict(),
        }
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        calibration_figure(t_rows, t_preds, REPORTS / "figures" / "calibration.png")
        mlflow.log_artifact(str(REPORTS / "results.json"))
        mlflow.log_artifact(str(REPORTS / "figures" / "calibration.png"))

        bundle_path = final.save()
        mlflow.log_artifact(str(bundle_path))
        if register:
            assert final.classifier is not None and final.regressor is not None
            mlflow.sklearn.log_model(
                final.classifier,
                name="risk_classifier",
                registered_model_name="spillway-risk-classifier",
                serialization_format="cloudpickle",
            )
            mlflow.sklearn.log_model(
                final.regressor,
                name="pct_regressor",
                registered_model_name="spillway-pct-regressor",
                serialization_format="cloudpickle",
            )
    return results


def register_bundle(run_id: str | None = None) -> None:
    """Register the saved bundle's two estimators (resume the backtest run when given)."""
    model = ForecastModel.load()
    assert model.classifier is not None and model.regressor is not None
    mlflow.set_tracking_uri(config.mlflow_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else "register"):
        mlflow.sklearn.log_model(
            model.classifier,
            name="risk_classifier",
            registered_model_name="spillway-risk-classifier",
            serialization_format="cloudpickle",
        )
        mlflow.sklearn.log_model(
            model.regressor,
            name="pct_regressor",
            registered_model_name="spillway-pct-regressor",
            serialization_format="cloudpickle",
        )
