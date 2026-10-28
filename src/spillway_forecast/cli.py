"""Command line: ``spillway ingest | train | publish | serve | reconcile``."""

from __future__ import annotations

import argparse
import os
from datetime import date

import pandas as pd


def reconcile(dsn: str | None = None, tolerance: float = 0.001) -> int:
    """Compare monthly sums of the hourly series with the warehouse tables (SELECT only).

    Returns the number of month rows whose relative deviation exceeds ``tolerance``.
    """
    import psycopg

    from spillway_forecast import config

    dsn = dsn or os.environ.get(
        "WAREHOUSE_DSN", "postgresql://warehouse:warehouse@localhost:5433/warehouse"
    )
    hourly = pd.read_parquet(config.hourly_path())
    hourly["month"] = hourly["ts"].dt.to_period("M").dt.to_timestamp()
    bad_total = 0
    for scope_type, table, col in (
        ("state", "state_monthly_curtailment", "state"),
        ("subsystem+SIN", "subsystem_monthly_curtailment", "subsystem"),
    ):
        # the warehouse subsystem table also carries a SIN row per month
        types = ["state"] if scope_type == "state" else ["subsystem", "sin"]
        sub = hourly[hourly["scope_type"].isin(types)]
        ours = sub.groupby(["scope", "source", "month"])[["curtailed_mwh", "generation_mwh"]].sum()
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"select {col}, source, month, curtailed_mwh, verified_generation_mwh "
                f"from {table} where source in ('wind','solar')"
            )
            wh = pd.DataFrame(
                cur.fetchall(), columns=["scope", "source", "month", "wh_curt", "wh_gen"]
            )
        wh["scope"] = wh["scope"].str.strip()
        wh["month"] = pd.to_datetime(wh["month"])
        for c in ("wh_curt", "wh_gen"):
            wh[c] = wh[c].astype(float)
        merged = wh.merge(ours.reset_index(), on=["scope", "source", "month"], how="outer")
        only_wh = merged["curtailed_mwh"].isna().sum()
        only_ours = merged["wh_curt"].isna().sum()
        both = merged.dropna(subset=["curtailed_mwh", "wh_curt"]).copy()
        both["curt_diff"] = (both["curtailed_mwh"] - both["wh_curt"]).abs()
        both["gen_diff"] = (both["generation_mwh"] - both["wh_gen"]).abs()
        both["curt_rel"] = both["curt_diff"] / both["wh_curt"].where(both["wh_curt"] > 0)
        both["gen_rel"] = both["gen_diff"] / both["wh_gen"].where(both["wh_gen"] > 0)
        bad = both[(both["curt_rel"] > tolerance) | (both["gen_rel"] > tolerance)]
        bad_total += len(bad) + int(only_wh) + int(only_ours)
        print(f"== {scope_type}: {len(both)} month rows compared (tolerance {tolerance:.2%})")
        print(f"   only in warehouse: {only_wh}, only in hourly series: {only_ours}")
        print(
            f"   max |diff| curtailed MWh: {both['curt_diff'].max():.4f}, "
            f"max rel: {both['curt_rel'].max():.2e}"
        )
        print(
            f"   max |diff| generation MWh: {both['gen_diff'].max():.4f}, "
            f"max rel: {both['gen_rel'].max():.2e}"
        )
        print(f"   rows above tolerance: {len(bad)}")
        if len(bad):
            print(bad.head(10).to_string())
    return bad_total


def main() -> None:
    parser = argparse.ArgumentParser(prog="spillway")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="build the hourly series (default: silver layer)")
    p_ing.add_argument("--from", dest="origin", choices=["silver", "ons"], default="silver")
    p_ing.add_argument("--refresh-last", type=int, default=2)
    p_train = sub.add_parser("train", help="backtest, track in MLflow and register the model")
    p_train.add_argument("--no-register", action="store_true")
    p_serve = sub.add_parser("serve", help="run the FastAPI service")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8100)
    p_pub = sub.add_parser("publish", help="score all series and write them to the warehouse")
    p_pub.add_argument("--dsn", default=None, help="Postgres DSN (or env PUBLISH_DSN)")
    p_pub.add_argument("--keep-runs", type=int, default=14)
    p_reg = sub.add_parser("register", help="register the saved model bundle in MLflow")
    p_reg.add_argument("--run-id", default=None)
    sub.add_parser("reconcile", help="check hourly aggregation against the warehouse")
    args = parser.parse_args()

    if args.cmd == "ingest":
        from spillway_forecast import ingest as ing

        if args.origin == "silver":
            print(ing.ingest_silver(today=date.today()))
        else:
            print(ing.ingest(refresh_last=args.refresh_last, today=date.today()))
    elif args.cmd == "train":
        from spillway_forecast.train import train

        train(register=not args.no_register)
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("spillway_forecast.api:app", host=args.host, port=args.port)
    elif args.cmd == "publish":
        from spillway_forecast.publish import publish

        print(publish(args.dsn, args.keep_runs))
    elif args.cmd == "register":
        from spillway_forecast.train import register_bundle

        register_bundle(args.run_id)
    else:
        raise SystemExit(1 if reconcile() else 0)


if __name__ == "__main__":
    main()
