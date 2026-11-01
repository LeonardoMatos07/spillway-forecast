# mypy: disable-error-code="arg-type,assignment,operator,index,call-overload,union-attr"
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from spillway_forecast import flag_publish as fp
from test_flag_model_eval import synthetic_inputs


class FakeCursor:
    """Records statements; answers the few selects the publisher issues."""

    def __init__(self, ready_exists: bool = False, pending: list[tuple[int, str]] | None = None):
        self.calls: list[tuple[str, Any]] = []
        self.ready_exists = ready_exists
        self.pending = pending or []
        self._next: Any = None
        self.probs: dict[int, list[tuple[str, float]]] = {}

    def execute(self, query: str, params: Any = None) -> None:
        self.calls.append((query, params))
        q = query.lower()
        if q.startswith("select id from flag_forecast_run"):
            self._next = (1,) if self.ready_exists else None
        elif q.startswith("insert into flag_forecast_run"):
            self._next = (42,)
        elif q.startswith("select r.id"):
            self._next = self.pending
        elif q.startswith("select flag, prob"):
            self._next = self.probs[params[0]]

    def fetchone(self) -> Any:
        return self._next

    def fetchall(self) -> Any:
        return self._next


def write_results(path: Path, passed: dict[str, str | None], rate: float = 1.0) -> Path:
    leads = {
        lead: {"gates": [{"candidate": cand, "passed": cand is not None}] if cand else []}
        for lead, cand in passed.items()
    }
    path.write_text(json.dumps({"reproduction_rate": rate, "leads": leads}), encoding="utf-8")
    return path


def test_only_leads_that_passed_are_publishable(tmp_path: Path) -> None:
    results = write_results(
        tmp_path / "r.json", {"21": None, "14": "ridge", "7": "ridge", "3": None}
    )
    assert fp.publishable_leads(results) == {14: "ridge", 7: "ridge"}
    assert fp.publishable_leads(write_results(tmp_path / "none.json", {"14": None})) == {}


def test_a_divergent_engine_blocks_publication(tmp_path: Path) -> None:
    results = write_results(tmp_path / "r.json", {"14": "ridge"}, rate=0.97)
    with pytest.raises(fp.PublicationBlockedError, match="reproduce"):
        fp.publishable_leads(results)


def test_due_leads_follow_the_announcement_date() -> None:
    leads = {21: "ridge", 14: "ridge", 7: "ridge", 3: "ridge"}
    month = date(2026, 11, 1)  # announced around 2026-10-29
    assert fp.due_leads(month, date(2026, 10, 1), leads) == []
    assert fp.due_leads(month, date(2026, 10, 12), leads) == [21]
    assert fp.due_leads(month, date(2026, 10, 15), leads) == [14, 21]
    assert fp.due_leads(month, date(2026, 10, 28), leads) == [3, 7, 14, 21]


def test_plan_run_probabilities_sum_to_one_and_bands_are_ordered() -> None:
    inputs = synthetic_inputs(60, signal=3.0, seed=3)
    target = date(2025, 1, 1)
    plan = fp.plan_run(target, 14, "ridge", inputs, 61.07, 716.80, today=date(2024, 12, 20))
    assert sum(p for _, p, _, _ in plan.probabilities) == pytest.approx(1.0, abs=1e-3)
    assert [f for f, *_ in plan.probabilities] == ["Verde", "Amarela", "Vermelha P1", "Vermelha P2"]
    lowers = [lo for *_, lo, _ in plan.probabilities]
    assert lowers == sorted(lowers)
    p10, p50, p90 = plan.trigger_quantiles
    assert p10 <= p50 <= p90
    assert plan.data_through <= date(2024, 12, 20)
    assert plan.inputs["note"].startswith("approximation")


def test_plan_run_uses_only_data_up_to_the_cutoff() -> None:
    inputs = synthetic_inputs(60, signal=3.0, seed=3)
    target = date(2025, 1, 1)
    base = fp.plan_run(target, 14, "ridge", inputs, 61.07, 716.80, today=date(2024, 12, 20))
    inputs.pld_weekly.loc[inputs.pld_weekly["week_start"] > date(2024, 12, 15), "pld"] = 9999.0
    inputs.storage_daily.loc[inputs.storage_daily["day"] >= date(2024, 12, 15), "storage_pct"] = 1.0
    after = fp.plan_run(target, 14, "ridge", inputs, 61.07, 716.80, today=date(2024, 12, 20))
    assert after.probabilities == base.probabilities


def test_write_run_is_one_unit_and_flips_to_ready_last() -> None:
    inputs = synthetic_inputs(60, signal=3.0, seed=3)
    plan = fp.plan_run(date(2025, 1, 1), 14, "ridge", inputs, 61.07, 716.80, date(2024, 12, 20))
    cur = FakeCursor()
    run_id = fp.write_run(cur, plan, datetime(2024, 12, 20, tzinfo=UTC))
    assert run_id == 42
    statements = [c[0].split("(")[0].strip().lower() for c in cur.calls]
    assert statements[1] == "insert into flag_forecast_run"
    assert statements.count("insert into flag_forecast_prob") == 4
    assert cur.calls[-1][0].lower().startswith("update flag_forecast_run set status = 'ready'")
    again = fp.write_run(FakeCursor(ready_exists=True), plan, datetime(2024, 12, 21, tzinfo=UTC))
    assert again is None


def test_score_pending_inserts_once_per_announced_run() -> None:
    cur = FakeCursor(pending=[(7, "Amarela")])
    cur.probs[7] = [("Verde", 0.2), ("Amarela", 0.6), ("Vermelha P1", 0.2), ("Vermelha P2", 0.0)]
    assert fp.score_pending(cur) == 1
    query, params = cur.calls[-1]
    assert "on conflict (run_id) do nothing" in query
    assert params[1] == "Amarela"
    assert params[2] == 0.6
    assert params[3] == pytest.approx(0.04 + 0.16 + 0.04, abs=1e-6)
    assert params[4] is True
    assert fp.score_pending(FakeCursor(pending=[])) == 0


def test_check_bulletin_flags_a_divergent_row() -> None:
    class Cur(FakeCursor):
        def execute(self, query: str, params: Any = None) -> None:
            self._next = [
                (date(2026, 8, 1), 57.31, 785.27, 0.72, 98.14, 246.40, 339.47),
                (date(2026, 7, 1), 57.31, 785.27, 0.70, 91.60, 229.97, 400.00),
            ]

    problems = fp.check_bulletin(Cur())
    assert len(problems) == 1
    assert "2026-07-01" in problems[0]
