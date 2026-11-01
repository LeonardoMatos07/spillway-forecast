"""Global recurrent network (LSTM or GRU) for the inflow forecast (stage 3), CPU only.

One model for all reservoirs: the last ``window`` days of flow, storage, lagged rain and calendar
plus static attributes go in; P10/P50/P90 of the next 1..14 days come out (pinball loss). The
origins, splits and targets are exactly those of the stage 1 dataset, so the comparison with the
stage 2 models is like for like.

No leakage: normalisation uses training years only; rain is shifted by the real delay (46 days);
early stopping uses the validation years; test rows are only predicted at the very end.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
import torch
from torch import nn

from spillway_forecast import inflow_dataset as ds
from spillway_forecast.inflow_models import H, to_forecast_frame

QUANTILES = (0.1, 0.5, 0.9)
N_CHANNELS = 7


@dataclass(frozen=True)
class DeepConfig:
    cell: str = "lstm"  # or "gru"
    window: int = 90
    hidden: int = 48
    dropout: float = 0.2
    lr: float = 2e-3
    batch: int = 256
    epochs: int = 20
    samples_per_epoch: int = 30000
    patience: int = 4
    use_rain: bool = True
    threads: int = 4


@dataclass
class Panel:
    """All reservoir-days in one array so that any window is a slice."""

    x: torch.Tensor  # (rows, channels)
    index: dict[tuple[str, date], int]
    scale: dict[str, tuple[float, float]]  # reservoir -> (mean z, std z) from train years
    statics: dict[str, np.ndarray]


def build_panel(
    reservoir_daily: pl.DataFrame,
    register: pl.DataFrame,
    rain: pl.DataFrame | None,
    config: ds.DatasetConfig,
    use_rain: bool = True,
) -> Panel:
    daily = reservoir_daily.select(
        "reservoir_id",
        "day",
        pl.col("natural_inflow_m3s").cast(pl.Float64).alias("q"),
        pl.col("storage_pct").cast(pl.Float64).alias("ear"),
    )
    if config.reservoirs is not None:
        daily = daily.filter(pl.col("reservoir_id").is_in(config.reservoirs))
    grid = ds._full_grid(daily)
    df = grid.join(daily, on=["reservoir_id", "day"], how="left")
    df = ds._fill_short_gaps(df, "q", config.max_gap_days)
    df = ds._fill_short_gaps(df, "ear", config.max_gap_days)
    if rain is not None and use_rain:
        r = rain.select("reservoir_id", "day", "precip_mm")
        df = df.join(r, on=["reservoir_id", "day"], how="left").with_columns(
            pl.col("precip_mm").shift(config.rain_lag_days).over("reservoir_id").alias("rain")
        )
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("rain"))
    df = df.sort("reservoir_id", "day").with_columns(
        pl.col("q").clip(lower_bound=0).log1p().alias("z")
    )
    train = df.filter((pl.col("day") >= config.train_start) & (pl.col("day") <= config.train_end))
    stats = train.group_by("reservoir_id").agg(
        pl.col("z").mean().alias("mu"), pl.col("z").std().alias("sd")
    )
    df = df.join(stats, on="reservoir_id")
    ang = 2 * math.pi * pl.col("day").dt.ordinal_day() / 365.25
    df = df.with_columns(
        ((pl.col("z") - pl.col("mu")) / pl.col("sd").clip(lower_bound=1e-3)).alias("c_z"),
        (pl.col("ear") / 100.0).alias("c_ear"),
        pl.col("ear").is_null().cast(pl.Float64).alias("c_ear_missing"),
        (pl.col("rain").clip(lower_bound=0).log1p()).alias("c_rain"),
        pl.col("rain").is_null().cast(pl.Float64).alias("c_rain_missing"),
        ang.sin().alias("c_sin"),
        ang.cos().alias("c_cos"),
    )
    channels = ["c_z", "c_ear", "c_ear_missing", "c_rain", "c_rain_missing", "c_sin", "c_cos"]
    arr = df.select([pl.col(c).fill_null(0.0).fill_nan(0.0) for c in channels]).to_numpy()
    keys = df.select("reservoir_id", "day").iter_rows()
    index = {(str(r), d): i for i, (r, d) in enumerate(keys)}
    scale = {str(r): (float(m), float(max(s, 1e-3))) for r, m, s in stats.iter_rows()}
    reg = (
        register if "reservoir_id" in register.columns else register.rename({"id": "reservoir_id"})
    )
    basins = sorted(reg["basin"].unique().to_list())
    logv = np.log(reg["useful_volume_hm3"].to_numpy().astype(float).clip(1.0))
    lv = (logv - logv.mean()) / (logv.std() + 1e-9)
    statics = {
        str(rid): np.array(
            [lv[i], *[float(reg["basin"][i] == b) for b in basins]], dtype=np.float32
        )
        for i, rid in enumerate(reg["reservoir_id"].to_list())
    }
    return Panel(torch.tensor(arr, dtype=torch.float32), index, scale, statics)


@dataclass
class Samples:
    rows: torch.Tensor  # panel row of the origin day
    static: torch.Tensor
    y: torch.Tensor  # (n, H) standardised change; NaN where unobserved
    z0: np.ndarray  # log1p of the flow on the origin day
    sd: np.ndarray
    frame: pl.DataFrame


def make_samples(panel: Panel, features: pl.DataFrame) -> Samples:
    rid = features["reservoir_id"].to_list()
    origin = features["origin_day"].to_list()
    rows = np.array([panel.index[(str(r), o)] for r, o in zip(rid, origin, strict=True)])
    sd = np.array([panel.scale[str(r)][1] for r in rid])
    z0 = np.log1p(features["q_lag0"].to_numpy().astype(float).clip(0))
    y = features.select([f"y_{h}" for h in range(1, H + 1)]).to_numpy().astype(float)
    ychg = (np.log1p(y.clip(0)) - z0[:, None]) / sd[:, None]
    static = np.stack([panel.statics[str(r)] for r in rid])
    return Samples(
        torch.tensor(rows),
        torch.tensor(static),
        torch.tensor(ychg, dtype=torch.float32),
        z0,
        sd,
        features,
    )


class Net(nn.Module):
    def __init__(self, cfg: DeepConfig, n_static: int) -> None:
        super().__init__()
        rnn = nn.LSTM if cfg.cell == "lstm" else nn.GRU
        self.rnn = rnn(N_CHANNELS, cfg.hidden, batch_first=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden + n_static, 64), nn.ReLU(), nn.Linear(64, H * 3)
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(seq)
        h = self.drop(out[:, -1])
        out_q: torch.Tensor = self.head(torch.cat([h, static], dim=1)).view(-1, H, 3)
        return out_q


def pinball_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(y)
    yy = torch.where(mask, y, torch.zeros_like(y))
    taus = torch.tensor(QUANTILES).view(1, 1, 3)
    diff = yy.unsqueeze(2) - pred
    loss = torch.maximum(taus * diff, (taus - 1) * diff) * mask.unsqueeze(2)
    return loss.sum() / (mask.sum() * 3).clamp(min=1)


def _batch(panel: Panel, s: Samples, idx: torch.Tensor, window: int) -> torch.Tensor:
    rows = s.rows[idx]
    offs = torch.arange(-window + 1, 1)
    return panel.x[rows.unsqueeze(1) + offs.unsqueeze(0)]


def predict_raw(net: Net, panel: Panel, s: Samples, cfg: DeepConfig) -> torch.Tensor:
    net.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(s.rows), 2048):
            idx = torch.arange(start, min(start + 2048, len(s.rows)))
            outs.append(net(_batch(panel, s, idx, cfg.window), s.static[idx]))
    return torch.cat(outs)


def fit_one(
    panel: Panel, train: Samples, valid: Samples, cfg: DeepConfig, seed: int
) -> tuple[Net, dict[str, float]]:
    torch.manual_seed(seed)
    torch.set_num_threads(cfg.threads)
    gen = torch.Generator().manual_seed(seed)
    net = Net(cfg, train.static.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    best, best_state, bad, history = float("inf"), copy.deepcopy(net.state_dict()), 0, []
    usable = torch.nonzero(~torch.isnan(train.y).all(dim=1)).squeeze(1)
    for _ in range(cfg.epochs):
        net.train()
        pick = usable[torch.randperm(len(usable), generator=gen)[: cfg.samples_per_epoch]]
        for start in range(0, len(pick), cfg.batch):
            idx = pick[start : start + cfg.batch]
            opt.zero_grad()
            loss = pinball_loss(
                net(_batch(panel, train, idx, cfg.window), train.static[idx]), train.y[idx]
            )
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        val = float(pinball_loss(predict_raw(net, panel, valid, cfg), valid.y))
        history.append(val)
        if val < best - 1e-5:
            best, best_state, bad = val, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    net.load_state_dict(best_state)
    return net, {"valid_pinball": best, "epochs": float(len(history))}


def to_flow(raw: torch.Tensor, s: Samples) -> np.ndarray:
    """Standardised log change -> m3/s, quantiles sorted (no crossing)."""
    q = np.sort(raw.numpy().astype(float), axis=2)
    z = s.z0[:, None, None] + q * s.sd[:, None, None]
    return np.asarray(np.clip(np.expm1(z), 0.0, None), dtype=float)


def ensemble_forecast(
    name: str, nets: list[Net], panel: Panel, s: Samples, cfg: DeepConfig
) -> pl.DataFrame:
    raws = torch.stack([predict_raw(n, panel, s, cfg) for n in nets])
    mean = torch.sort(raws, dim=3).values.mean(dim=0)  # average of sorted quantiles
    return to_forecast_frame(s.frame, name, to_flow(mean, s))
