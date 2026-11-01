"""Runners for the flag forecast: the reproduction report and the stage 4 evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from spillway_forecast import flag_backtest as bt
from spillway_forecast import flag_eval as ev


def backtest_report() -> str:
    rules, aneel = bt.read_rules(), bt.read_aneel_flags()
    return bt.report_markdown(bt.reproduce(rules, aneel), rules)


def gate_to_dict(result: ev.GateResult) -> dict[str, Any]:
    return {
        "lead": result.lead,
        "candidate": result.candidate,
        "best_reference": result.best_reference,
        "passed": result.passed,
        "reasons": result.reasons,
        "metrics": result.metrics,
    }


def evaluate(dsn: str, out_dir: Path) -> dict[str, Any]:
    """Real evaluation: rolling origin at 21/14/7/3 days with warehouse inputs (SELECT only).

    Writes ``results.json``, ``scored.csv`` and ``summary_<lead>.csv`` in ``out_dir``. The gate
    constants come from ``flag_eval.GATE`` and were fixed before any result was seen.
    """
    pld, storage, flags = bt.load_warehouse_inputs(dsn)
    rules = bt.read_rules()
    repro = bt.reproduce(rules, flags if len(flags) else bt.read_aneel_flags())
    rate = bt.reproduction_rate(repro)
    inputs = ev.Inputs(rules=rules, flags=flags, pld_weekly=pld, storage_daily=storage)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"reproduction_rate": rate, "gate": vars(ev.GATE), "leads": {}}
    scored_all = []
    for lead in ev.LEADS:
        scored, gates = ev.run_lead(inputs, lead, rate)
        if scored.empty:
            results["leads"][str(lead)] = {"error": "not enough training months"}
            continue
        scored = scored.assign(lead=lead)
        scored_all.append(scored)
        summary = ev.summarise(scored)
        summary.to_csv(out_dir / f"summary_{lead}.csv", index=False)
        results["leads"][str(lead)] = {
            "summary": summary.to_dict(orient="records"),
            "gates": [gate_to_dict(g) for g in gates],
            "publishable": any(g.passed for g in gates),
        }
    if scored_all:
        pd.concat(scored_all).to_csv(out_dir / "scored.csv", index=False)
    results["publishable_leads"] = [
        int(k) for k, v in results["leads"].items() if v.get("publishable")
    ]
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    return results
