"""Trigger rules, promotion gate, refresh decision, hybrid ingest and failure safety."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from spillway_forecast import ingest
from spillway_forecast import model as model_mod
from spillway_forecast import retrain as rt
from spillway_forecast import train as train_mod
from spillway_forecast.refresh import RunState, decide_refresh, month_end

NOW = datetime(2026, 11, 4, 12, tzinfo=UTC)
GOOD = {"roc_auc": 0.917, "mae_pp": 10.0, "ece": 0.036}
BOOT_OK = {"abs_error": {"ci_high": -0.01}, "brier": {"ci_high": -0.05}}


# ------------------------------------------------------------------ trigger
def trigger(**kw: Any) -> rt.Trigger:
    base: dict[str, Any] = {
        "silver_complete_through": date(2026, 10, 31),
        "trained_through": date(2026, 9, 30),
        "trained_at": NOW - timedelta(days=36),
        "now": NOW,
        "force": False,
    }
    return rt.decide_trigger(**{**base, **kw})


def test_trigger_new_complete_month_after_25_days() -> None:
    assert trigger().go


def test_trigger_needs_a_newer_complete_month() -> None:
    t = trigger(silver_complete_through=date(2026, 9, 30))
    assert not t.go and "no complete month newer" in t.reason
    assert not trigger(silver_complete_through=None).go


def test_trigger_hybrid_tail_counts_as_trained_data() -> None:
    # trained on a provisional tail through Nov 2: the October silver month is not new
    assert not trigger(trained_through=date(2026, 11, 2)).go


def test_trigger_25_day_rule() -> None:
    assert not trigger(trained_at=NOW - timedelta(days=24)).go
    assert trigger(trained_at=NOW - timedelta(days=25)).go


def test_trigger_force_overrides_everything() -> None:
    assert trigger(silver_complete_through=None, trained_at=NOW, force=True).go


# --------------------------------------------------------------------- gate
def test_gate_promotes_a_candidate_that_is_not_worse() -> None:
    d = rt.gate(GOOD, GOOD, BOOT_OK, 5000)
    assert d.promote and d.reasons == []


@pytest.mark.parametrize(
    ("candidate", "text"),
    [
        ({"roc_auc": 0.917 - 0.0101, "mae_pp": 10.0, "ece": 0.036}, "ROC-AUC"),
        ({"roc_auc": 0.917, "mae_pp": 10.31, "ece": 0.036}, "MAE"),
        ({"roc_auc": 0.917, "mae_pp": 10.0, "ece": 0.057}, "ECE"),
    ],
)
def test_gate_rejects_each_tolerance(candidate: dict[str, float], text: str) -> None:
    d = rt.gate(candidate, GOOD, BOOT_OK, 5000)
    assert not d.promote and any(text in r for r in d.reasons)


def test_gate_accepts_changes_inside_the_tolerances() -> None:
    inside = {"roc_auc": 0.917 - 0.009, "mae_pp": 10.29, "ece": 0.055}
    assert rt.gate(inside, GOOD, BOOT_OK, 5000).promote


def test_gate_requires_beating_climatology_with_interval_below_zero() -> None:
    crossing = {"abs_error": {"ci_high": 0.002}, "brier": {"ci_high": -0.05}}
    assert not rt.gate(GOOD, GOOD, crossing, 5000).promote
    assert not rt.gate(GOOD, GOOD, None, 5000).promote


def test_gate_rejects_a_window_that_is_too_small() -> None:
    d = rt.gate(GOOD, GOOD, BOOT_OK, rt.MIN_GATE_ROWS - 1)
    assert not d.promote and "too small" in d.reasons[0]


def test_gate_constants_are_the_documented_ones() -> None:
    assert (rt.GATE_MAX_AUC_DROP, rt.GATE_MAX_MAE_RISE_PP, rt.GATE_MAX_ECE_RISE) == (
        0.01,
        0.3,
        0.02,
    )
    assert rt.MIN_DAYS_BETWEEN_TRAININGS == 25


# ------------------------------------------------------------------ refresh
def state(**kw: Any) -> RunState:
    base: dict[str, Any] = {
        "run_id": 1,
        "origin_ts": datetime(2026, 10, 28),
        "data_through": date(2026, 10, 27),
        "created_at": NOW,
        "model_run_id": "abc",
        "model_trained_at": NOW,
    }
    return RunState(**{**base, **kw})


def decide(**kw: Any) -> tuple[bool, str]:
    base: dict[str, Any] = {
        "bundle_run_id": "abc",
        "signature": {"wind:2026-10": "e1"},
        "stored_signature": {"wind:2026-10": "e1"},
        "silver_month": "2026-09",
        "stored_silver_month": "2026-09",
    }
    return decide_refresh(kw.pop("state", state()), **{**base, **kw})


def test_refresh_does_nothing_when_nothing_changed() -> None:
    go, reason = decide()
    assert not go and "no change" in reason


def test_refresh_reacts_to_each_change() -> None:
    assert decide(state=None)[0]
    assert decide(bundle_run_id="new")[0]
    assert decide(signature={"wind:2026-10": "e2"})[0]
    assert decide(silver_month="2026-10")[0]


def test_refresh_survives_an_unreachable_ons() -> None:
    assert not decide(signature=None)[0]  # keeps the last published forecast


def test_month_end() -> None:
    assert month_end((2026, 10)) == date(2026, 10, 31)
    assert month_end((2026, 12)) == date(2026, 12, 31)
    assert month_end((2028, 2)) == date(2028, 2, 29)


# ------------------------------------------------------------- hybrid ingest
def hourly_frame(day: date, hours: int, value: float = 1.0) -> pl.DataFrame:
    ts = [datetime(day.year, day.month, day.day) + timedelta(hours=h) for h in range(hours)]
    return pl.DataFrame(
        {
            "ts": ts,
            "scope_type": "sin",
            "scope": "SIN",
            "source": "wind",
            "curtailed_mwh": value,
            "generation_mwh": 10.0,
        }
    )


def test_complete_days_only_trims_the_partial_last_day() -> None:
    frame = pl.concat([hourly_frame(date(2026, 10, 1), 24), hourly_frame(date(2026, 10, 2), 9)])
    out = ingest.complete_days_only(frame)
    assert out["ts"].max() == datetime(2026, 10, 1, 23)


def test_compare_overlap_reports_the_largest_deviation() -> None:
    a, b = hourly_frame(date(2026, 9, 30), 24), hourly_frame(date(2026, 9, 30), 24, value=1.5)
    dev = ingest.compare_overlap(a, b)
    assert dev["hours"] == 24 and dev["max_abs_curtailed_mwh"] == pytest.approx(0.5)
    assert ingest.compare_overlap(a, a)["max_abs_curtailed_mwh"] == 0


def test_ingest_hybrid_appends_a_provisional_tail_and_prefers_silver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import spillway_forecast.silver as silver

    monkeypatch.setenv("SPILLWAY_DATA_DIR", str(tmp_path))
    silver_frame = pl.concat(
        [hourly_frame(date(2026, 9, 29), 24), hourly_frame(date(2026, 9, 30), 24)]
    )
    monkeypatch.setattr(silver, "read_silver_hourly", lambda today: silver_frame)

    def fake_ons(source: str, year: int, month: int) -> pl.DataFrame | None:
        if source != "wind":
            return None
        if (year, month) == (2026, 9):  # same hours as silver, slightly different values
            return hourly_frame(date(2026, 9, 30), 24, value=1.0)
        return pl.concat(
            [hourly_frame(date(2026, 10, 1), 24, 2.0), hourly_frame(date(2026, 10, 2), 5, 2.0)]
        )

    monkeypatch.setattr(ingest, "read_ons_month", fake_ons)
    path = ingest.ingest_hybrid(today=date(2026, 10, 2), ons_check={"x": "y"})
    out = pl.read_parquet(path)
    assert out.filter(pl.col("provisional"))["ts"].min() == datetime(2026, 10, 1)
    assert out["ts"].max() == datetime(2026, 10, 1, 23)  # partial 2 Oct dropped
    assert not out.filter(pl.col("ts") < datetime(2026, 10, 1))["provisional"].any()
    meta = json.loads(ingest.meta_path().read_text(encoding="utf-8"))
    assert meta["data_through_wind"] == "2026-10-01" and meta["provisional_rows"] == 24
    assert meta["overlap_wind"]["max_abs_curtailed_mwh"] == 0


# ----------------------------------------------------- promotion and safety
def test_promote_files_is_atomic_and_keeps_the_previous_copies(tmp_path: Path) -> None:
    models, reports = tmp_path / "m", tmp_path / "r"
    st_m, st_r = tmp_path / "s" / "m", tmp_path / "s" / "r"
    for d in (models, reports, st_m, st_r):
        d.mkdir(parents=True)
    (models / "bundle.joblib").write_bytes(b"old")
    (reports / "results.json").write_text("old")
    (st_m / "bundle.joblib").write_bytes(b"new")
    (st_r / "results.json").write_text("new")
    rt.promote_files(st_m, st_r, models, reports)
    assert (models / "bundle.joblib").read_bytes() == b"new"
    assert (models / "bundle.previous.joblib").read_bytes() == b"old"
    assert (reports / "results.previous.json").read_text() == "old"
    assert not list(models.glob("*.tmp"))


@contextmanager
def _always_locked(*_a: Any, **_k: Any) -> Iterator[bool]:
    yield True


def test_failed_training_leaves_bundle_and_results_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models, reports = tmp_path / "models", tmp_path / "reports"
    models.mkdir()
    reports.mkdir()
    (models / "bundle.joblib").write_bytes(b"production bundle")
    (reports / "results.json").write_text(
        json.dumps(
            {
                "run_id": "old",
                "trained_at": "2026-09-01T00:00:00+00:00",
                "data": {"end": "2026-09-30"},
            }
        )
    )
    monkeypatch.setenv("SPILLWAY_MODELS_DIR", str(models))
    monkeypatch.setenv("SPILLWAY_REPORTS_DIR", str(reports))
    monkeypatch.setattr(rt, "advisory_lock", _always_locked)
    monkeypatch.setattr(rt, "silver_latest_month", lambda: (2026, 10))
    monkeypatch.setattr(ingest, "ingest_hybrid", lambda **_k: tmp_path / "h.parquet")

    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("training exploded")

    monkeypatch.setattr(train_mod, "train", boom)
    params_before = dict(model_mod.GBM_PARAMS)
    with pytest.raises(RuntimeError, match="exploded"):
        rt.retrain("postgresql://unused", force=True, fast=True)
    assert (models / "bundle.joblib").read_bytes() == b"production bundle"
    assert json.loads((reports / "results.json").read_text())["run_id"] == "old"
    assert not list(models.glob("staging-*"))
    assert params_before == model_mod.GBM_PARAMS  # the demo override never leaks


def test_dry_run_trains_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "results.json").write_text(
        json.dumps(
            {
                "run_id": "old",
                "trained_at": "2026-09-01T00:00:00+00:00",
                "data": {"end": "2026-09-30"},
            }
        )
    )
    monkeypatch.setenv("SPILLWAY_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("SPILLWAY_REPORTS_DIR", str(reports))
    monkeypatch.setattr(rt, "silver_latest_month", lambda: (2026, 10))

    def must_not_run(*_a: Any, **_k: Any) -> None:
        raise AssertionError("dry run must not train")

    monkeypatch.setattr(train_mod, "train", must_not_run)
    out = rt.retrain("postgresql://unused", dry_run=True, now=datetime(2026, 11, 4, tzinfo=UTC))
    assert out["status"] == "dry-run" and out["go"] is True
