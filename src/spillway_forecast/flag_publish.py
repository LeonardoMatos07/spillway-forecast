"""Stage 5 publisher of the flag forecast: runs, probabilities, inputs and the scoreboard.

Only leads that passed the fixed publication criterion are published (``results.json`` of
``spillway flag-evaluate``). Everything for one run is written in a single transaction and the run
becomes visible only when its status flips to ``ready``. Runs are never rewritten; the scoreboard
row is inserted once, after ANEEL announces the flag. Database role in production:
``energia_forecast`` (grants in spillway-infra; tables are owned by spillway-api).

If the latest bulletin diverges from our rule engine the job stops publishing and says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from spillway_forecast import flag_eval as ev
from spillway_forecast import flag_model as fm
from spillway_forecast.flag_rule import FLAG_ORDER, compute_bands, rule_for_month


class Cursor(Protocol):
    def execute(self, query: str, params: Any = ...) -> Any: ...

    def fetchone(self) -> Any: ...

    def fetchall(self) -> Any: ...


@dataclass(frozen=True)
class RunPlan:
    target_month: date
    lead_days: int
    data_through: date
    model_version: str
    rule_version: str
    probabilities: list[tuple[str, float, float, float]]  # flag, prob, band lower, band upper
    trigger_quantiles: tuple[float, float, float]
    inputs: dict[str, Any]


class PublicationBlockedError(RuntimeError):
    """The engine diverges from a bulletin, or no lead passed the criterion."""


def publishable_leads(results_path: Path) -> dict[int, str]:
    """Lead days that passed the criterion, with the candidate model that passed (first wins)."""
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("reproduction_rate", 0) < 1.0:
        raise PublicationBlockedError("the rule engine does not reproduce every bulletin")
    out: dict[int, str] = {}
    for lead, info in results.get("leads", {}).items():
        for gate in info.get("gates", []):
            if gate["passed"]:
                out[int(lead)] = gate["candidate"]
                break
    return out


def due_leads(target_month: date, today: date, leads: dict[int, str]) -> list[int]:
    """Leads whose cutoff has been reached (announcement minus lead days)."""
    announce = fm.default_announcement(target_month)
    return sorted(lead for lead in leads if today >= announce - timedelta(days=lead))


def plan_run(
    target_month: date,
    lead: int,
    model_name: str,
    inputs: ev.Inputs,
    pld_min: float,
    pld_max: float,
    today: date,
    model_version: str | None = None,
) -> RunPlan:
    """Fit on every readable bulletin and predict the target month from data up to the cutoff."""
    announce = fm.default_announcement(target_month)
    cutoff = min(today, announce - timedelta(days=lead))
    flag_by_month = dict(zip(inputs.flags["month"], inputs.flags["flag"], strict=True))
    prev = (target_month - timedelta(days=1)).replace(day=1)
    feats = fm.build_features(
        cutoff, target_month, flag_by_month.get(prev), inputs.pld_weekly, inputs.storage_daily
    )
    table = ev.build_table(inputs, lead)
    train = table[table["valid"] & (table["month"].map(fm.label_available_on) <= cutoff)]
    if len(train) < ev.GATE.min_train:
        raise PublicationBlockedError(f"only {len(train)} readable bulletins to train on")
    model = fm.make_model(model_name)
    model.fit(
        train[list(fm.FEATURES)].to_numpy(dtype=float),
        train["y_trigger_log"].to_numpy(dtype=float),
        train["y_gsf"].to_numpy(dtype=float),
    )
    x_row = np.array([feats[name] for name in fm.FEATURES], dtype=float)
    pred = model.predict(x_row)
    rule = rule_for_month(target_month)
    probs = fm.colour_probabilities(pred, pld_min, pld_max, rule)
    gsf_med = float(np.round(np.median(np.clip(pred.gsf, *fm.GSF_BOUNDS)), 2))
    bands = compute_bands(f"{pld_min:.2f}", f"{pld_max:.2f}", f"{gsf_med:.2f}", rule)
    lowers = [
        float(bands.pld_min),
        float(bands.green_upper) + 0.01,
        float(bands.yellow_upper) + 0.01,
        float(bands.red1_upper) + 0.01,
    ]
    uppers = [
        float(bands.green_upper),
        float(bands.yellow_upper),
        float(bands.red1_upper),
        float(bands.pld_max),
    ]
    rows = [
        (flag.value, round(float(p), 4), round(lo, 2), round(min(up, float(bands.pld_max)), 2))
        for flag, p, lo, up in zip(FLAG_ORDER, probs, lowers, uppers, strict=True)
    ]
    q10, q50, q90 = (float(np.quantile(pred.trigger, q)) for q in (0.1, 0.5, 0.9))
    return RunPlan(
        target_month=target_month,
        lead_days=lead,
        data_through=cutoff,
        model_version=model_version or f"{model_name}-{today:%Y%m%d}",
        rule_version=rule.name,
        probabilities=rows,
        trigger_quantiles=(round(q10, 2), round(q50, 2), round(q90, 2)),
        inputs={
            "cutoff": cutoff.isoformat(),
            "features": {
                k: (None if np.isnan(v) else round(float(v), 4)) for k, v in feats.items()
            },
            "n_train": len(train),
            "gsf_median": gsf_med,
            "pld_min": pld_min,
            "pld_max": pld_max,
            "sources": ["spillway-data: pld_weekly, subsystem_storage_daily, tariff_flag_monthly"],
            "note": "approximation: no DECOMP rounds and no ONS hydro forecast",
        },
    )


def write_run(cur: Cursor, plan: RunPlan, generated_at: datetime) -> int | None:
    """Insert one run, its probabilities and inputs, then flip it to ready. None if it exists."""
    cur.execute(
        "select id from flag_forecast_run where target_month = %s and lead_days = %s "
        "and data_through = %s and status = 'ready'",
        (plan.target_month, plan.lead_days, plan.data_through),
    )
    if cur.fetchone():
        return None
    p10, p50, p90 = plan.trigger_quantiles
    cur.execute(
        "insert into flag_forecast_run (target_month, lead_days, generated_at, data_through, "
        "model_version, rule_version, trigger_p10_brl_per_mwh, trigger_p50_brl_per_mwh, "
        "trigger_p90_brl_per_mwh, status) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,'loading') "
        "returning id",
        (
            plan.target_month,
            plan.lead_days,
            generated_at,
            plan.data_through,
            plan.model_version,
            plan.rule_version,
            p10,
            p50,
            p90,
        ),
    )
    run_id = int(cur.fetchone()[0])
    for flag, prob, lower, upper in plan.probabilities:
        cur.execute(
            "insert into flag_forecast_prob (run_id, flag, prob, band_lower_brl_per_mwh, "
            "band_upper_brl_per_mwh) values (%s,%s,%s,%s,%s)",
            (run_id, flag, prob, lower, upper),
        )
    cur.execute(
        "insert into flag_forecast_input (run_id, inputs) values (%s, %s::jsonb)",
        (run_id, json.dumps(plan.inputs)),
    )
    cur.execute("update flag_forecast_run set status = 'ready' where id = %s", (run_id,))
    return run_id


def score_pending(cur: Cursor) -> int:
    """Score ready runs whose month was announced. Inserted once, never rewritten."""
    cur.execute(
        "select r.id, t.flag from flag_forecast_run r "
        "join tariff_flag_monthly t on t.month = r.target_month "
        "left join flag_forecast_score s on s.run_id = r.id "
        "where r.status = 'ready' and s.run_id is null"
    )
    pending = cur.fetchall()
    scored = 0
    for run_id, actual in pending:
        cur.execute("select flag, prob from flag_forecast_prob where run_id = %s", (run_id,))
        probs = {flag: float(p) for flag, p in cur.fetchall()}
        vector = [probs.get(f.value, 0.0) for f in FLAG_ORDER]
        row = ev.score_run(vector, actual)
        cur.execute(
            "insert into flag_forecast_score (run_id, actual_flag, prob_actual, brier, top_hit) "
            "values (%s,%s,%s,%s,%s) on conflict (run_id) do nothing",
            (
                run_id,
                row["actual_flag"],
                round(row["prob_actual"], 4),
                round(row["brier"], 6),
                row["top_hit"],
            ),
        )
        scored += 1
    return scored


def check_bulletin(cur: Cursor) -> list[str]:
    """Months whose stored bands differ from the engine by more than a cent (halts publication)."""
    from spillway_forecast.flag_rule import compare_bands

    cur.execute(
        "select month, pld_min_brl_per_mwh, pld_max_brl_per_mwh, gsf, green_upper_brl_per_mwh, "
        "yellow_upper_brl_per_mwh, red1_upper_brl_per_mwh from flag_rule_month order by month"
    )
    problems = []
    for month, lo, hi, gsf, g, y, r1 in cur.fetchall():
        bands = compute_bands(str(lo), str(hi), str(gsf), rule_for_month(month))
        diffs = compare_bands(bands, g, y, r1)
        if diffs:
            problems.append(f"{month}: {'; '.join(diffs)}")
    return problems


def publish(
    dsn: str, results_path: Path, target_month: date, today: date | None = None
) -> dict[str, Any]:
    """Publish the due leads of the target month; score announced months; notify the API."""
    import psycopg

    from spillway_forecast import flag_backtest as bt
    from spillway_forecast.refresh import notify_api

    today = today or datetime.now(UTC).date()
    leads = publishable_leads(results_path)
    pld, storage, flags = bt.load_warehouse_inputs(dsn)
    rules = bt.read_rules()
    inputs = ev.Inputs(rules=rules, flags=flags, pld_weekly=pld, storage_daily=storage)
    written: list[int] = []
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:  # one transaction
        problems = check_bulletin(cur)
        if problems:
            raise PublicationBlockedError(
                "bulletin diverges from the engine: " + " | ".join(problems)
            )
        scored = score_pending(cur)
        cur.execute(
            "select pld_min_brl_per_mwh, pld_max_brl_per_mwh from flag_rule_month "
            "order by month desc limit 1"
        )
        limits = cur.fetchone()
        if limits and leads:
            for lead in due_leads(target_month, today, leads):
                plan = plan_run(
                    target_month,
                    lead,
                    leads[lead],
                    inputs,
                    float(limits[0]),
                    float(limits[1]),
                    today,
                )
                run_id = write_run(cur, plan, datetime.now(UTC))
                if run_id is not None:
                    written.append(run_id)
    result: dict[str, Any] = {"published_runs": written, "scored": scored, "leads": sorted(leads)}
    if written or scored:
        result["revalidate"] = notify_api(["flag"])
    return result
