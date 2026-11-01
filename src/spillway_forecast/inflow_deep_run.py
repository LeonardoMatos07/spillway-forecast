"""Stage 3 runner: compares the recurrent network with the stage 2 models under the strict gate.

Order: (1) choose LSTM or GRU on validation pinball with one seed; (2) train the seed ensemble
with and without rain; (3) score validation and test; (4) apply the publication gate against the
best of ALL stage 2 models (references, ridge, boosting), the best being picked on validation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import polars as pl

from spillway_forecast import inflow_dataset as ds
from spillway_forecast import inflow_deep as dp
from spillway_forecast import inflow_eval as ev

SEEDS = (11, 22, 33, 44, 55)
BASELINES = [
    "persistence",
    "climatology",
    "recession",
    "moving_average_7d",
    "ridge",
    "gbm_quantile",
]


def run(
    dataset_dir: Path,
    reservoir_daily: Path,
    register: Path,
    rain: Path,
    baseline_forecasts: Path,
    out_dir: Path,
    seeds: tuple[int, ...] = SEEDS,
    epochs: int = 20,
) -> dict[str, Any]:
    features = pl.read_parquet(dataset_dir / "features.parquet")
    cfg_ds = ds.DatasetConfig()
    rd, reg, rn = (pl.read_parquet(p) for p in (reservoir_daily, register, rain))
    out_dir.mkdir(parents=True, exist_ok=True)
    log: dict[str, Any] = {"dataset_id": dataset_dir.name, "seeds": list(seeds)}
    base = replace(dp.DeepConfig(), epochs=epochs)

    parts = {s: features.filter(pl.col("split") == s) for s in ("train", "valid", "test")}
    frames: list[pl.DataFrame] = []
    panels = {True: dp.build_panel(rd, reg, rn, cfg_ds, use_rain=True)}
    panels[False] = dp.build_panel(rd, reg, rn, cfg_ds, use_rain=False)

    # (1) cell type, one seed, validation pinball only
    samples = {
        use: {s: dp.make_samples(panels[use], parts[s]) for s in parts} for use in (True, False)
    }
    cell_scores: dict[str, float] = {}
    for cell in ("lstm", "gru"):
        t = time.time()
        cfg = replace(base, cell=cell)
        _, info = dp.fit_one(
            panels[True], samples[True]["train"], samples[True]["valid"], cfg, seeds[0]
        )
        cell_scores[cell] = info["valid_pinball"]
        log[f"cell_{cell}"] = {**info, "seconds": time.time() - t}
        print(cell, info, flush=True)
    best_cell = min(cell_scores, key=lambda c: cell_scores[c])
    log["chosen_cell"] = best_cell

    # (2) ensembles with and without rain
    for use_rain, name in ((True, "net"), (False, "net_no_rain")):
        cfg = replace(base, cell=best_cell, use_rain=use_rain)
        nets, infos = [], []
        for sd in seeds:
            t = time.time()
            net, info = dp.fit_one(
                panels[use_rain], samples[use_rain]["train"], samples[use_rain]["valid"], cfg, sd
            )
            nets.append(net)
            infos.append({**info, "seed": sd, "seconds": time.time() - t})
            print(name, infos[-1], flush=True)
        log[name] = {"config": asdict(cfg), "runs": infos}
        # (3) scoring: validation and test rows, test is only predicted here
        for split in ("valid", "test"):
            frames.append(
                dp.ensemble_forecast(name, nets, panels[use_rain], samples[use_rain][split], cfg)
            )

    # (4) strict gate against every stage 2 model
    old = pl.read_parquet(baseline_forecasts).filter(pl.col("model").is_in(BASELINES))
    forecasts = pl.concat([old, *frames])
    tables = {s: ev.per_reservoir(forecasts, s) for s in ("valid", "test")}
    for split_name, tab in tables.items():
        tab.write_parquet(out_dir / f"per_reservoir_{split_name}.parquet")
        ev.aggregate(tab).write_csv(out_dir / f"aggregate_{split_name}.csv")
    gates = {
        c: ev.gate(tables["test"], tables["valid"], c, BASELINES) for c in ("net", "net_no_rain")
    }
    for c, g in gates.items():
        g.write_csv(out_dir / f"gate_{c}.csv")
    log["publishable_horizons"] = {
        c: [int(h) for h in g.filter(pl.col("publish"))["h"].to_list()] for c, g in gates.items()
    }
    (out_dir / "results.json").write_text(json.dumps(log, indent=2, sort_keys=True))
    return log


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="stage 3: recurrent network vs stage 2 models")
    p.add_argument("--dataset", required=True)
    p.add_argument("--reservoir-daily", required=True)
    p.add_argument("--register", required=True)
    p.add_argument("--rain", required=True)
    p.add_argument("--baseline-forecasts", required=True, help="forecasts.parquet of stage 2")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=len(SEEDS))
    p.add_argument("--epochs", type=int, default=20)
    a = p.parse_args(argv)
    res = run(
        Path(a.dataset),
        Path(a.reservoir_daily),
        Path(a.register),
        Path(a.rain),
        Path(a.baseline_forecasts),
        Path(a.out),
        SEEDS[: a.seeds],
        a.epochs,
    )
    print(json.dumps(res["publishable_horizons"]))


if __name__ == "__main__":
    main()
