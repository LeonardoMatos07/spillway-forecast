"""Automatic monthly retraining with a promotion gate.

Trigger (checked once a day by ``spillway watch``): silver holds a complete month that ends after
the data the current model was trained on, and at least ``MIN_DAYS_BETWEEN_TRAININGS`` days have
passed since the last training (a re-run of the pipeline never retrains twice).

The candidate goes through exactly the same protocol as ``spillway train`` (rolling-origin folds
with embargo, untouched final test, baselines, bootstrap). It is promoted only if

1. on the *new* window, i.e. the days after the incumbent's training data, which neither the
   incumbent nor the candidate (re-fitted only on data before that window) has seen, it is not
   worse than the incumbent beyond small tolerances (ROC-AUC, MAE, ECE), and
2. on its own final test it still beats climatology with a 95% bootstrap interval below zero for
   both absolute error and Brier score.

On rejection nothing on disk or in the database changes; the outcome is logged, written to
``reports/retrain_last.json`` and tagged on the MLflow run. Promotion writes the bundle and the
results atomically (temp file + rename), keeps ``*.previous.*`` copies, registers the model in
MLflow, publishes (``forecast_model.is_current`` flips in the publish transaction) and asks the
API to revalidate.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from spillway_forecast import config
from spillway_forecast.baselines import hourly_observations
from spillway_forecast.config import PROTOCOL, Protocol
from spillway_forecast.features import build_dataset, to_series
from spillway_forecast.model import BUNDLE_NAME, ForecastModel
from spillway_forecast.refresh import (
    LOCK_PUBLISH,
    LOCK_RETRAIN,
    advisory_lock,
    default_dsn,
    month_end,
    notify_api,
    silver_latest_month,
)
from spillway_forecast.validation import Split

log = logging.getLogger("spillway.retrain")

MIN_DAYS_BETWEEN_TRAININGS = 25
GATE_MAX_AUC_DROP = 0.01
GATE_MAX_MAE_RISE_PP = 0.3
GATE_MAX_ECE_RISE = 0.02
MIN_GATE_ROWS = 2000


@dataclass(frozen=True)
class Trigger:
    go: bool
    reason: str


def decide_trigger(
    *,
    silver_complete_through: date | None,
    trained_through: date | None,
    trained_at: datetime | None,
    now: datetime,
    force: bool = False,
) -> Trigger:
    if force:
        return Trigger(True, "forced")
    if silver_complete_through is None:
        return Trigger(False, "silver has no complete month")
    if trained_through is not None and silver_complete_through <= trained_through:
        return Trigger(
            False,
            f"no complete month newer than the training data "
            f"(silver through {silver_complete_through}, model through {trained_through})",
        )
    if trained_at is not None:
        ref = trained_at if trained_at.tzinfo else trained_at.replace(tzinfo=UTC)
        days = (now - ref).days
        if days < MIN_DAYS_BETWEEN_TRAININGS:
            return Trigger(False, f"trained {days} days ago (minimum {MIN_DAYS_BETWEEN_TRAININGS})")
    return Trigger(True, f"new complete month in silver (through {silver_complete_through})")


@dataclass
class GateDecision:
    promote: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)


def gate(
    candidate: dict[str, float],
    incumbent: dict[str, float],
    bootstrap_vs_climatology: dict[str, dict[str, float]] | None,
    n_rows: int,
) -> GateDecision:
    """Apply the promotion rules to metrics measured on the same new window."""
    reasons: list[str] = []
    checks: dict[str, Any] = {"n_rows": n_rows}
    if n_rows < MIN_GATE_ROWS:
        reasons.append(f"new window too small to compare ({n_rows} < {MIN_GATE_ROWS} rows)")
    auc_drop = incumbent["roc_auc"] - candidate["roc_auc"]
    mae_rise = candidate["mae_pp"] - incumbent["mae_pp"]
    ece_rise = candidate["ece"] - incumbent["ece"]
    checks.update(auc_drop=auc_drop, mae_rise_pp=mae_rise, ece_rise=ece_rise)
    if auc_drop > GATE_MAX_AUC_DROP:
        reasons.append(f"ROC-AUC dropped {auc_drop:.4f} (> {GATE_MAX_AUC_DROP})")
    if mae_rise > GATE_MAX_MAE_RISE_PP:
        reasons.append(f"MAE rose {mae_rise:.2f} pp (> {GATE_MAX_MAE_RISE_PP})")
    if ece_rise > GATE_MAX_ECE_RISE:
        reasons.append(f"ECE rose {ece_rise:.4f} (> {GATE_MAX_ECE_RISE})")
    beats = bootstrap_vs_climatology is not None and all(
        bootstrap_vs_climatology[m]["ci_high"] < 0 for m in ("abs_error", "brier")
    )
    checks["beats_climatology"] = beats
    if not beats:
        reasons.append("does not beat climatology with a bootstrap interval below zero")
    return GateDecision(promote=not reasons, reasons=reasons, checks=checks)


def model_state(results_file: Path) -> tuple[date | None, datetime | None, str | None]:
    """``(trained_through, trained_at, mlflow run id)`` of the model in production."""
    if not results_file.exists():
        return None, None, None
    results = json.loads(results_file.read_text(encoding="utf-8"))
    end = results.get("data", {}).get("end")
    trained_through = pd.Timestamp(end).date() if end else None
    trained_at = datetime.fromisoformat(results["trained_at"]) if "trained_at" in results else None
    return trained_through, trained_at, results.get("run_id")


def _gate_metrics(
    hourly: pd.DataFrame, incumbent_path: Path, gate_start: date, protocol: Protocol
) -> tuple[dict[str, float], dict[str, float], int]:
    """Score candidate (fit before the window) and incumbent on the same unseen window."""
    from spillway_forecast.metrics import evaluate
    from spillway_forecast.train import run_split

    series_list = to_series(hourly)
    series_ids = [s.series_id for s in series_list]
    data = build_dataset(series_list, protocol.origin_stride_days)
    obs = hourly_observations(hourly)
    obs = obs[obs["series_id"].isin(series_ids)]
    split = Split("gate", gate_start, None)
    rows, preds = run_split(split, data, obs, series_ids, protocol.random_state)
    incumbent = ForecastModel.load(incumbent_path)
    pct, risk = incumbent.predict(rows)
    ok = ~(pd.isna(pct) | pd.isna(risk))
    rows, pct, risk = rows[ok], pct[ok], risk[ok]
    cand = preds["hist_gbm"][ok]
    cand_scores = evaluate(rows, cand["pct"].to_numpy(float), cand["risk"].to_numpy(float))
    inc_scores = evaluate(rows, pct, risk)
    return cand_scores, inc_scores, len(rows)


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def promote_files(staging_models: Path, staging_reports: Path, models: Path, reports: Path) -> None:
    """Swap the candidate in atomically, keeping the previous bundle and results."""
    models.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    bundle, results = models / BUNDLE_NAME, reports / "results.json"
    if bundle.exists():
        _atomic_copy(bundle, models / "bundle.previous.joblib")
    if results.exists():
        _atomic_copy(results, reports / "results.previous.json")
    _atomic_copy(staging_models / BUNDLE_NAME, bundle)
    _atomic_copy(staging_reports / "results.json", results)
    figure = staging_reports / "figures" / "calibration.png"
    if figure.exists():
        (reports / "figures").mkdir(exist_ok=True)
        _atomic_copy(figure, reports / "figures" / "calibration.png")


def retrain(
    dsn: str | None = None,
    *,
    force: bool = False,
    dry_run: bool = False,
    fast: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    from spillway_forecast import ingest, publish
    from spillway_forecast import model as model_mod
    from spillway_forecast import train as train_mod

    dsn = dsn or default_dsn()
    now = now or datetime.now(UTC)
    models, reports = config.models_dir(), publish.results_path().parent
    trained_through, trained_at, _ = model_state(reports / "results.json")
    latest = silver_latest_month()
    trigger = decide_trigger(
        silver_complete_through=month_end(latest) if latest else None,
        trained_through=trained_through,
        trained_at=trained_at,
        now=now,
        force=force,
    )
    if dry_run or not trigger.go:
        verb = "would retrain" if trigger.go else "no retrain"
        log.info("%s: %s", verb, trigger.reason)
        return {
            "status": "dry-run" if dry_run else "skipped",
            "go": trigger.go,
            "reason": trigger.reason,
        }
    with advisory_lock(dsn, LOCK_RETRAIN) as got:
        if not got:
            log.info("another retrain is running; nothing to do")
            return {"status": "busy"}
        log.info("new month -> retraining (%s)", trigger.reason)
        started = datetime.now(UTC)
        ingest.ingest_hybrid(today=date.today())
        protocol = PROTOCOL
        saved_params = dict(model_mod.GBM_PARAMS)
        if fast:  # demo configuration only: NOT a valid model, never used in production
            trees = int(os.environ.get("FORECAST_DEMO_TREES", "25"))
            log.warning("FAST demo configuration: %d trees, 1 fold, coarse origins", trees)
            model_mod.GBM_PARAMS["max_iter"] = trees
            protocol = Protocol(fold_starts=(date(2026, 1, 1),), origin_stride_days=9)
        staging = models / f"staging-{started:%Y%m%d%H%M%S}"
        staging_reports = staging / "reports"
        try:
            candidate = train_mod.train(
                protocol, register=False, models_dir=staging, reports_dir=staging_reports
            )
            hourly = pd.read_parquet(config.hourly_path())
            incumbent_file = models / BUNDLE_NAME
            gate_start = (trained_through or protocol.test_start) + timedelta(days=1)
            cand_scores, inc_scores, n_rows = _gate_metrics(
                hourly, incumbent_file, gate_start, protocol
            )
            decision = gate(
                cand_scores,
                inc_scores,
                candidate["test_bootstrap"].get("hist_gbm_minus_climatology"),
                n_rows,
            )
            outcome: dict[str, Any] = {
                "status": "promoted" if decision.promote else "rejected",
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "candidate_run_id": candidate["run_id"],
                "gate_window_start": str(gate_start),
                "candidate": cand_scores,
                "incumbent": inc_scores,
                "checks": decision.checks,
                "reasons": decision.reasons,
                "fast_demo": fast,
            }
            _record(candidate["run_id"], outcome)
            if not decision.promote:
                log.warning(
                    "gate rejected the candidate, keeping the current model: %s",
                    "; ".join(decision.reasons),
                )
                (reports / "retrain_last.json").write_text(
                    json.dumps(outcome, indent=2), encoding="utf-8"
                )
                return outcome
            log.info(
                "gate passed (%s)",
                {k: round(v, 4) if isinstance(v, float) else v for k, v in decision.checks.items()},
            )
            with advisory_lock(dsn, LOCK_PUBLISH, wait_seconds=600) as publish_got:
                if not publish_got:
                    raise RuntimeError("could not take the publish lock to promote the model")
                promote_files(staging, staging_reports, models, reports)
                train_mod.register_bundle(candidate["run_id"], models / BUNDLE_NAME)
                result = publish.publish(dsn)
            outcome["published"] = result
            log.info(
                "promoted model %s; published run %s", candidate["run_id"][:8], result.get("run_id")
            )
            (reports / "retrain_last.json").write_text(
                json.dumps(outcome, indent=2), encoding="utf-8"
            )
            try:
                outcome["api"] = notify_api()
            except Exception as exc:
                log.warning("API revalidation failed (%s); its cache expires on its own", exc)
            return outcome
        finally:
            model_mod.GBM_PARAMS.clear()
            model_mod.GBM_PARAMS.update(saved_params)
            shutil.rmtree(staging, ignore_errors=True)


def _record(run_id: str, outcome: dict[str, Any]) -> None:
    """Tag the MLflow run with the gate outcome (best effort)."""
    try:
        import mlflow

        mlflow.set_tracking_uri(config.mlflow_uri())
        client = mlflow.MlflowClient()
        client.set_tag(run_id, "gate", outcome["status"])
        client.set_tag(run_id, "gate_reasons", "; ".join(outcome["reasons"]) or "all checks passed")
    except Exception as exc:
        log.warning("could not tag the MLflow run: %s", exc)
