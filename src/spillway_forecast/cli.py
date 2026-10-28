"""Command line: ``spillway ingest | train | serve | reconcile``."""

from __future__ import annotations

import argparse
import os
from datetime import date

import pandas as pd


def _reconcile() -> None:
    """Compare monthly sums with the warehouse (read-only SELECT)."""
    import psycopg

    from spillway_forecast import config

    dsn = os.environ.get(
        "WAREHOUSE_DSN", "postgresql://warehouse:warehouse@localhost:5433/warehouse"
    )
    hourly = pd.read_parquet(config.hourly_path())
    hourly = hourly[hourly["scope_type"] == "state"]
    hourly["month"] = hourly["ts"].dt.to_period("M").dt.to_timestamp()
    ours = hourly.groupby(["scope", "source", "month"])[["curtailed_mwh", "generation_mwh"]].sum()
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "select state, source, month, curtailed_mwh, verified_generation_mwh "
            "from state_monthly_curtailment where source in ('wind','solar')"
        )
        wh = pd.DataFrame(cur.fetchall(), columns=["scope", "source", "month", "wh_curt", "wh_gen"])
    wh["month"] = pd.to_datetime(wh["month"])
    for c in ("wh_curt", "wh_gen"):
        wh[c] = wh[c].astype(float)
    merged = wh.merge(ours.reset_index(), on=["scope", "source", "month"], how="inner")
    merged["curt_rel_diff"] = (merged["curtailed_mwh"] - merged["wh_curt"]) / merged["wh_curt"]
    print(f"{len(merged)} state-month rows compared")
    print(merged["curt_rel_diff"].describe().round(5).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(prog="spillway")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="download ONS open data and build hourly series")
    p_ing.add_argument("--refresh-last", type=int, default=2)
    p_train = sub.add_parser("train", help="backtest, track in MLflow and register the model")
    p_train.add_argument("--no-register", action="store_true")
    p_serve = sub.add_parser("serve", help="run the FastAPI service")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8100)
    p_reg = sub.add_parser("register", help="register the saved model bundle in MLflow")
    p_reg.add_argument("--run-id", default=None)
    sub.add_parser("reconcile", help="check hourly aggregation against the warehouse")
    args = parser.parse_args()

    if args.cmd == "ingest":
        from spillway_forecast.ingest import ingest

        print(ingest(refresh_last=args.refresh_last, today=date.today()))
    elif args.cmd == "train":
        from spillway_forecast.train import train

        train(register=not args.no_register)
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("spillway_forecast.api:app", host=args.host, port=args.port)
    elif args.cmd == "register":
        from spillway_forecast.train import register_bundle

        register_bundle(args.run_id)
    else:
        _reconcile()


if __name__ == "__main__":
    main()
