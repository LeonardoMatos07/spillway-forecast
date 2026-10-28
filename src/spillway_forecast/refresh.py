"""Automatic refresh: detect new ONS data, rebuild the series, publish, tell the API.

``refresh`` is cheap when nothing changed (a few HEAD requests and one SQL query) and
idempotent: it only recomputes when

* nothing has been published yet, or the model in the bundle is not the published one, or
* the ONS files of the current / previous month changed (ETag) or silver has a new month,
  *and* the rebuilt series reach a day later than the latest published run.

``watch`` runs it in a loop (default every 60 minutes) and starts ``spillway retrain`` in a
subprocess once a day, without ever blocking the refresh loop.

Locks: Postgres advisory locks (shared by every worker)
  * ``LOCK_PUBLISH``: held while publishing or promoting a model, so refresh and retrain never
    write the forecast tables or the bundle at the same time;
  * ``LOCK_RETRAIN``: held for the whole retraining, so two workers never retrain at once.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from spillway_forecast import config

log = logging.getLogger("spillway.refresh")

LOCK_PUBLISH = 727001
LOCK_RETRAIN = 727002
MODEL_MAX_AGE_DAYS = int(os.environ.get("FORECAST_MODEL_MAX_AGE_DAYS", "45"))
RETRAIN_CHECK_EVERY = timedelta(hours=24)
RETRAIN_TIMEOUT_SECONDS = int(os.environ.get("FORECAST_RETRAIN_TIMEOUT_SECONDS", str(3 * 3600)))


def default_dsn() -> str:
    return (
        os.environ.get("PUBLISH_DSN")
        or os.environ.get("WAREHOUSE_DSN")
        or "postgresql://warehouse:warehouse@localhost:5433/warehouse"
    )


@contextmanager
def advisory_lock(dsn: str, key: int, wait_seconds: float = 0.0) -> Iterator[bool]:
    """Session-level Postgres advisory lock; yields whether it was acquired."""
    import psycopg

    deadline = time.monotonic() + wait_seconds
    with psycopg.connect(dsn, autocommit=True) as conn:
        acquired = False
        while True:
            row = conn.execute("select pg_try_advisory_lock(%s)", (key,)).fetchone()
            acquired = bool(row and row[0])
            if acquired or time.monotonic() >= deadline:
                break
            time.sleep(2.0)
        try:
            yield acquired
        finally:
            if acquired:
                conn.execute("select pg_advisory_unlock(%s)", (key,))


@dataclass(frozen=True)
class RunState:
    run_id: int
    origin_ts: datetime
    data_through: date
    created_at: datetime
    model_run_id: str
    model_trained_at: datetime


def latest_run_state(dsn: str) -> RunState | None:
    import psycopg

    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "select r.id, r.origin_ts, r.data_through, r.created_at, m.mlflow_run_id, "
            "m.trained_at from forecast_run r join forecast_model m on m.id = r.model_id "
            "where r.status = 'ready' order by r.origin_ts desc, r.id desc limit 1"
        ).fetchone()
    return None if row is None else RunState(*row)


def silver_latest_month() -> tuple[int, int] | None:
    """Newest month partition in the lake (a cheap listing, no parquet is read)."""
    from spillway_forecast.silver import lake_client

    client, bucket = lake_client()
    newest: tuple[int, int] | None = None
    year = date.today().year
    for source_prefix in config.LAKE_PREFIXES.values():
        for y in (year, year - 1):
            resp = client.list_objects_v2(Bucket=bucket, Prefix=f"{source_prefix}/year={y}/")
            for item in resp.get("Contents", []):
                key = item["Key"]
                try:
                    month = int(key.split("month=")[1][:2])
                except (IndexError, ValueError):
                    continue
                if newest is None or (y, month) > newest:
                    newest = (y, month)
    return newest


def month_end(month: tuple[int, int]) -> date:
    y, m = month
    nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return nxt - timedelta(days=1)


def decide_refresh(
    state: RunState | None,
    *,
    bundle_run_id: str | None,
    signature: dict[str, str] | None,
    stored_signature: dict[str, str] | None,
    silver_month: str | None,
    stored_silver_month: str | None,
) -> tuple[bool, str]:
    """Is it worth rebuilding the series? (A rebuild may still end in "no new day".)"""
    if state is None:
        return True, "nothing published yet"
    if bundle_run_id and bundle_run_id != state.model_run_id:
        return True, "the model in the bundle is newer than the published one"
    if signature is not None and signature != (stored_signature or {}):
        return True, "ONS files changed"
    if silver_month and silver_month != stored_silver_month:
        return True, f"silver has a new month ({silver_month})"
    return False, "no change in ONS files or silver"


def machine_token() -> str | None:
    url = os.environ.get("COGNITO_TOKEN_URL")
    client_id = os.environ.get("DATA_CLIENT_ID")
    secret = os.environ.get("DATA_CLIENT_SECRET")
    if not (url and client_id and secret):
        return None
    response = httpx.post(
        url,
        data={
            "grant_type": "client_credentials",
            "scope": os.environ.get("API_SCOPE", "energia-api/admin"),
        },
        auth=(client_id, secret),
        timeout=15,
    )
    response.raise_for_status()
    token: str = response.json()["access_token"]
    return token


def notify_api(tags: list[str] | None = None) -> dict[str, Any]:
    """Ask the API to drop its cache and the web to revalidate (same mechanism as the pipeline)."""
    api_url = os.environ.get("API_URL")
    if not api_url:
        return {"skipped": "API_URL is not set"}
    token = machine_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = httpx.post(
        f"{api_url.rstrip('/')}/v1/admin/revalidate",
        json={"tags": tags or ["forecast"]},
        headers=headers,
        timeout=20,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def model_age_days(trained_at: datetime, now: datetime | None = None) -> int:
    now = now or datetime.now(UTC)
    ref = trained_at if trained_at.tzinfo else trained_at.replace(tzinfo=UTC)
    return (now - ref).days


def refresh(
    dsn: str | None = None, *, force: bool = False, today: date | None = None
) -> dict[str, Any]:
    """One check. Returns ``{"status": "up-to-date" | "published" | ..., ...}``."""
    from spillway_forecast import ingest, publish

    dsn = dsn or default_dsn()
    today = today or date.today()
    started = time.monotonic()
    state = latest_run_state(dsn)
    results = json.loads(publish.results_path().read_text(encoding="utf-8"))
    if state is not None:
        age = model_age_days(state.model_trained_at)
        if age > MODEL_MAX_AGE_DAYS:
            log.warning(
                "the published model is %d days old (limit %d): retrain it", age, MODEL_MAX_AGE_DAYS
            )
    meta = ingest.read_meta()
    try:
        signature: dict[str, str] | None = ingest.ons_signature(today)
    except httpx.HTTPError as exc:
        log.warning("could not reach ONS (%s); keeping the last published forecast", exc)
        signature = None
    latest = silver_latest_month()
    silver_month = f"{latest[0]}-{latest[1]:02d}" if latest else None
    stored_silver = max(
        (str(v)[:7] for k, v in meta.items() if k.startswith("silver_through_")), default=None
    )
    go, reason = decide_refresh(
        state,
        bundle_run_id=results.get("run_id"),
        signature=signature,
        stored_signature=meta.get("ons_signature"),
        silver_month=silver_month,
        stored_silver_month=stored_silver,
    )
    if force:
        go, reason = True, "forced"
    if not go:
        return {
            "status": "up-to-date",
            "reason": reason,
            "seconds": round(time.monotonic() - started, 1),
        }
    ingest.ingest_hybrid(today=today, ons_check=signature)
    meta = ingest.read_meta()
    through = max(date.fromisoformat(v) for k, v in meta.items() if k.startswith("data_through_"))
    if (
        state is not None
        and not force
        and bundle_run_id_same(results, state)
        and through <= state.data_through
    ):
        return {
            "status": "up-to-date",
            "reason": f"no new day (data through {through})",
            "seconds": round(time.monotonic() - started, 1),
        }
    try:
        _warn_on_reconcile(dsn)
    except Exception as exc:  # reconcile is a sanity check, never a blocker
        log.warning("reconcile skipped: %s", exc)
    with advisory_lock(dsn, LOCK_PUBLISH, wait_seconds=600) as got:
        if not got:
            return {"status": "busy", "reason": "another publish or model promotion is running"}
        outcome = publish.publish(dsn)
    outcome["reason"] = reason
    outcome["data_through"] = str(through)
    try:
        outcome["api"] = notify_api()
    except (httpx.HTTPError, KeyError) as exc:
        log.warning("API revalidation failed (%s); the API cache expires on its own", exc)
        outcome["api"] = {"error": str(exc)}
    outcome["seconds"] = round(time.monotonic() - started, 1)
    return outcome


def bundle_run_id_same(results: dict[str, Any], state: RunState) -> bool:
    return results.get("run_id") == state.model_run_id


def _warn_on_reconcile(dsn: str) -> None:
    from spillway_forecast.cli import reconcile

    bad = reconcile(dsn)
    if bad:
        log.warning("reconcile: %d month rows deviate from the warehouse", bad)


def _format(result: dict[str, Any]) -> str:
    status = result.get("status")
    if status == "published":
        return (
            f"new data -> published run {result.get('run_id')} "
            f"(data through {result.get('data_through')}, {result.get('rows')} rows, "
            f"{result.get('seconds')} s, api={result.get('api')})"
        )
    return f"{status}: {result.get('reason')} ({result.get('seconds', '-')} s)"


def spawn_retrain() -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, "-m", "spillway_forecast.cli", "retrain"])


def watch(every_minutes: float = 60.0, *, dsn: str | None = None, once: bool = False) -> None:
    """Refresh every ``every_minutes``; retrain check once a day; stop cleanly on SIGTERM."""
    stop = threading.Event()

    def _stop(signum: int, _frame: object) -> None:
        log.info("signal %s received, stopping", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    failures = 0
    child: subprocess.Popen[bytes] | None = None
    child_started = 0.0
    next_retrain_check = datetime.now(UTC)
    interval = every_minutes * 60
    log.info("watch started: refresh every %s min, retrain check every 24 h", every_minutes)
    while not stop.is_set():
        try:
            result = refresh(dsn)
            log.info("refresh: %s", _format(result))
            failures = 0
        except Exception as exc:
            failures += 1
            log.error("refresh failed (attempt %d): %s: %s", failures, type(exc).__name__, exc)
        if child is not None:
            code = child.poll()
            if code is not None:
                log.info("retrain subprocess finished with exit code %s", code)
                child = None
            elif time.monotonic() - child_started > RETRAIN_TIMEOUT_SECONDS:
                log.error(
                    "retrain exceeded %d s; killing it (model and bundle untouched)",
                    RETRAIN_TIMEOUT_SECONDS,
                )
                child.kill()
                child = None
        if child is None and datetime.now(UTC) >= next_retrain_check:
            next_retrain_check = datetime.now(UTC) + RETRAIN_CHECK_EVERY
            log.info("daily retrain check: starting `spillway retrain` in the background")
            child = spawn_retrain()
            child_started = time.monotonic()
        if once:
            break
        wait = interval if failures == 0 else min(interval, 60.0 * 2**failures)
        stop.wait(wait)
    if child is not None and child.poll() is None:
        log.info("terminating the running retrain subprocess")
        child.terminate()
