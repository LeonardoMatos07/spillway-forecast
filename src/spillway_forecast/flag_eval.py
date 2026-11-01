"""Stage 4 evaluation: rolling origin per lead, three references, metrics and the strict gate.

THE PUBLICATION CRITERION IS FIXED HERE, BEFORE ANY REAL RESULT IS SEEN (see
docs/flag-publication-criterion.md). Do not edit ``GATE`` after looking at results; a change
needs a new commit that says why, and the model card must report the change.

A lead is publishable only when a candidate passes ALL of the following against the best
reference of that lead (the best reference is chosen on the same evaluation months, which is
the hardest choice for the candidate):

1. at least ``min_scored`` months scored at that lead;
2. mean Brier (sum over 4 colours) at least ``min_rel_gain`` lower than the best reference;
3. paired block bootstrap of the monthly Brier difference: the ``1 - alpha / k`` quantile of the
   difference is below zero (``k`` candidates tried at the lead, Bonferroni);
4. top-colour hit rate not below the best reference;
5. pooled one-vs-rest expected calibration error at most ``max_ece``;
6. the rule engine reproduces every bulletin used (``reproduction_rate == 1``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from spillway_forecast.flag_model import (
    CORE_FEATURES,
    FEATURES,
    build_features,
    colour_index,
    colour_probabilities,
    label_available_on,
    make_model,
)
from spillway_forecast.flag_rule import FLAG_ORDER, rule_for_month

FloatArray = NDArray[np.float64]

LEADS = (21, 14, 7, 3)
REFERENCES = (
    "ref_prev_colour",
    "ref_month_climatology",
    "ref_current_pld",
    "ref_overall_frequency",
)
CANDIDATES = ("ridge", "linear_quantile", "boosting_quantile")
LOG_LOSS_EPS = 1e-3


@dataclass(frozen=True)
class Gate:
    min_scored: int = 24
    min_rel_gain: float = 0.05
    alpha: float = 0.10
    boot_block_months: int = 3
    n_boot: int = 2000
    top_hit_margin: float = 0.0
    max_ece: float = 0.15
    ece_bins: int = 5
    min_train: int = 12
    seed: int = 20261101


GATE = Gate()


# --- metrics -----------------------------------------------------------------------------------


def brier(probs: FloatArray, actual: int) -> float:
    """Multi-class Brier: sum over the 4 colours of (p - 1[actual])^2, range 0 to 2."""
    one_hot = np.zeros(4)
    one_hot[actual] = 1.0
    return float(((probs - one_hot) ** 2).sum())


def log_loss(probs: FloatArray, actual: int) -> float:
    return float(-np.log(max(probs[actual], LOG_LOSS_EPS)))


def top_hit(probs: FloatArray, actual: int) -> bool:
    return int(np.argmax(probs)) == actual


def score_run(probs: Sequence[float], actual: str) -> dict[str, Any]:
    """The columns of ``flag_forecast_score`` for one run (scoreboard row)."""
    p = np.asarray(probs, dtype=float)
    a = colour_index(actual)
    return {
        "actual_flag": FLAG_ORDER[a].value,
        "prob_actual": float(p[a]),
        "brier": brier(p, a),
        "top_hit": top_hit(p, a),
    }


def calibration_table(probs: FloatArray, actual: NDArray[np.int64], bins: int) -> pd.DataFrame:
    """Pooled one-vs-rest reliability: every (month, colour) pair is one forecast."""
    p = probs.reshape(-1)
    y = np.zeros_like(probs)
    y[np.arange(len(actual)), actual] = 1.0
    y = y.reshape(-1)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append(
                {
                    "bin": b,
                    "n": int(m.sum()),
                    "mean_pred": float(p[m].mean()),
                    "obs": float(y[m].mean()),
                }
            )
    return pd.DataFrame(rows)


def ece(probs: FloatArray, actual: NDArray[np.int64], bins: int) -> float:
    table = calibration_table(probs, actual, bins)
    if table.empty:
        return float("nan")
    return float((table["n"] * (table["mean_pred"] - table["obs"]).abs()).sum() / table["n"].sum())


def minimum_detectable_difference(
    n: int, sd_diff: float, alpha: float = 0.10, power: float = 0.80
) -> float:
    """Smallest true mean Brier difference a paired one-sided test detects (normal approx.)."""
    from statistics import NormalDist

    z = NormalDist()
    return (z.inv_cdf(1 - alpha) + z.inv_cdf(power)) * sd_diff / np.sqrt(n)


# --- references ----------------------------------------------------------------------------------


def _one_hot(i: int) -> FloatArray:
    out = np.zeros(4)
    out[i] = 1.0
    return out


def ref_prev_colour(prev_flag: str) -> FloatArray:
    return _one_hot(colour_index(prev_flag))


def ref_month_climatology(month: date, history: pd.DataFrame) -> FloatArray:
    """Share of each colour in the same calendar month of earlier years (add-half smoothing)."""
    same = history[
        (history["month"].map(lambda d: d.month) == month.month) & (history["month"] < month)
    ]
    counts = np.full(4, 0.5)
    for name in same["flag"]:
        counts[colour_index(name)] += 1.0
    return counts / counts.sum()


def ref_overall_frequency(train_flags: Sequence[str]) -> FloatArray:
    """Share of each colour over the same training months the models see (add-half smoothing).

    Stricter than the spec's three references: the no-skill forecast with the model's own sample.
    """
    counts = np.full(4, 0.5)
    for name in train_flags:
        counts[colour_index(name)] += 1.0
    return counts / counts.sum()


# --- dataset and rolling origin ----------------------------------------------------------------


@dataclass
class Inputs:
    rules: pd.DataFrame  # seed: month, flag, pld_min, pld_max, trigger, gsf, announced_on, ...
    flags: pd.DataFrame  # ANEEL series: month, flag
    pld_weekly: pd.DataFrame  # week_start, subsystem, pld
    storage_daily: pd.DataFrame  # day, subsystem, storage_pct, ena_gross_pct_mlt


def build_table(inputs: Inputs, lead: int) -> pd.DataFrame:
    """One row per bulletin month with the features that exist ``lead`` days before the news."""
    flag_by_month = dict(zip(inputs.flags["month"], inputs.flags["flag"], strict=True))
    rows = []
    for r in inputs.rules.sort_values("month").itertuples():
        month: date = r.month
        prev = (month.replace(day=1) - timedelta(days=1)).replace(day=1)
        cutoff = r.announced_on - timedelta(days=lead)
        feats = build_features(
            cutoff, month, flag_by_month.get(prev), inputs.pld_weekly, inputs.storage_daily
        )
        rows.append(
            {
                "month": month,
                "cutoff": cutoff,
                "flag": r.flag,
                "prev_flag": flag_by_month.get(prev),
                "pld_min": r.pld_min,
                "pld_max": r.pld_max,
                "y_trigger_log": float(np.log(r.trigger)),
                "y_gsf": float(r.gsf),
                "trigger": float(r.trigger),
                **feats,
            }
        )
    table = pd.DataFrame(rows)
    table["valid"] = table[list(CORE_FEATURES)].notna().all(axis=1) & table["prev_flag"].notna()
    return table


def _features_matrix(table: pd.DataFrame) -> FloatArray:
    return table[list(FEATURES)].to_numpy(dtype=float)


def rolling_origin(
    table: pd.DataFrame,
    inputs: Inputs,
    model_names: Sequence[str],
    gate: Gate = GATE,
    model_factory: Callable[[str], Any] = make_model,
) -> pd.DataFrame:
    """Predict each valid month using ONLY months whose label was readable at its cutoff.

    Returns one row per (month, model) with the 4 colour probabilities and the actual colour.
    """
    out: list[dict[str, Any]] = []
    valid = table[table["valid"]].reset_index(drop=True)
    for _, row in valid.iterrows():
        cutoff: date = row["cutoff"]
        train = valid[valid["month"].map(label_available_on) <= cutoff]
        train = train[train["month"] < row["month"]]
        if len(train) < gate.min_train:
            continue
        actual = colour_index(row["flag"])
        base = {"month": row["month"], "actual": actual, "n_train": len(train)}
        rule = rule_for_month(row["month"])
        x_train = _features_matrix(train)
        x_row = row[list(FEATURES)].to_numpy(dtype=float)
        for name in model_names:
            model = model_factory(name)
            model.fit(
                x_train,
                train["y_trigger_log"].to_numpy(dtype=float),
                train["y_gsf"].to_numpy(dtype=float),
            )
            probs = colour_probabilities(model.predict(x_row), row["pld_min"], row["pld_max"], rule)
            out.append(
                {**base, "model": name, **dict(zip(("p0", "p1", "p2", "p3"), probs, strict=True))}
            )
        # references
        history = inputs.flags[inputs.flags["month"].map(label_available_on) <= cutoff]
        refs = {
            "ref_prev_colour": ref_prev_colour(row["prev_flag"]),
            "ref_month_climatology": ref_month_climatology(row["month"], history),
            "ref_overall_frequency": ref_overall_frequency(list(train["flag"])),
        }
        gsf_model = model_factory("ridge")
        gsf_model.fit(
            x_train,
            train["y_trigger_log"].to_numpy(dtype=float),
            train["y_gsf"].to_numpy(dtype=float),
        )
        pld_now = float(np.exp(row["log_pld_last"]))
        pred = gsf_model.predict(x_row)
        point = type(pred)(trigger=np.array([pld_now]), gsf=np.array([float(np.median(pred.gsf))]))
        refs["ref_current_pld"] = colour_probabilities(point, row["pld_min"], row["pld_max"], rule)
        for name, p in refs.items():
            out.append(
                {**base, "model": name, **dict(zip(("p0", "p1", "p2", "p3"), p, strict=True))}
            )
    return pd.DataFrame(out)


def score_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    probs = forecasts[["p0", "p1", "p2", "p3"]].to_numpy(dtype=float)
    actual = forecasts["actual"].to_numpy(dtype=int)
    scored = forecasts.copy()
    scored["brier"] = [brier(p, a) for p, a in zip(probs, actual, strict=True)]
    scored["log_loss"] = [log_loss(p, a) for p, a in zip(probs, actual, strict=True)]
    scored["top_hit"] = [top_hit(p, a) for p, a in zip(probs, actual, strict=True)]
    return scored


def summarise(scored: pd.DataFrame, gate: Gate = GATE) -> pd.DataFrame:
    rows = []
    for model, g in scored.groupby("model"):
        probs = g[["p0", "p1", "p2", "p3"]].to_numpy(dtype=float)
        actual = g["actual"].to_numpy(dtype=int)
        rows.append(
            {
                "model": model,
                "n": len(g),
                "brier": float(g["brier"].mean()),
                "log_loss": float(g["log_loss"].mean()),
                "top_hit": float(g["top_hit"].mean()),
                "ece": ece(probs, actual, gate.ece_bins),
            }
        )
    return pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)


# --- the gate ------------------------------------------------------------------------------------


def bootstrap_upper(diff: FloatArray, quantile: float, gate: Gate = GATE) -> float:
    """``quantile`` of the mean of ``diff`` (candidate minus reference), resampling month blocks."""
    rng = np.random.default_rng(gate.seed)
    n = len(diff)
    size = gate.boot_block_months
    starts = np.arange(0, n, size)
    blocks = [diff[s : s + size] for s in starts]
    stats = np.empty(gate.n_boot)
    for i in range(gate.n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        sample = np.concatenate([blocks[j] for j in pick])
        stats[i] = sample.mean()
    return float(np.quantile(stats, quantile))


@dataclass(frozen=True)
class GateResult:
    lead: int
    candidate: str
    best_reference: str
    passed: bool
    reasons: list[str]
    metrics: dict[str, float]


def evaluate_gate(
    scored: pd.DataFrame,
    lead: int,
    candidate: str,
    n_candidates: int,
    reproduction_rate: float,
    gate: Gate = GATE,
) -> GateResult:
    """Apply the fixed criterion to one candidate at one lead (``scored`` is that lead only)."""
    summary = summarise(scored, gate).set_index("model")
    refs = summary.loc[[r for r in REFERENCES if r in summary.index]]
    best_ref = str(refs["brier"].idxmin())
    cand = scored[scored["model"] == candidate].sort_values("month")
    ref = scored[scored["model"] == best_ref].sort_values("month")
    months = sorted(set(cand["month"]) & set(ref["month"]))
    cand = cand[cand["month"].isin(months)]
    ref = ref[ref["month"].isin(months)]
    diff = cand["brier"].to_numpy() - ref["brier"].to_numpy()
    n = len(months)
    c_brier, r_brier = float(cand["brier"].mean()), float(ref["brier"].mean())
    rel_gain = (r_brier - c_brier) / r_brier if r_brier > 0 else 0.0
    upper = (
        bootstrap_upper(diff, 1 - gate.alpha / max(n_candidates, 1), gate) if n else float("nan")
    )
    cand_sum = summary.loc[candidate]
    reasons = []
    if n < gate.min_scored:
        reasons.append(f"only {n} months scored (min {gate.min_scored})")
    if rel_gain < gate.min_rel_gain:
        reasons.append(f"Brier gain {rel_gain:.1%} over {best_ref} (min {gate.min_rel_gain:.0%})")
    if not upper < 0:
        reasons.append(f"bootstrap upper bound {upper:.4f} is not below zero")
    if cand_sum["top_hit"] < refs.loc[best_ref, "top_hit"] - gate.top_hit_margin:
        reasons.append("top-colour hit rate below the best reference")
    if not cand_sum["ece"] <= gate.max_ece:
        reasons.append(f"calibration error {cand_sum['ece']:.3f} above {gate.max_ece}")
    if reproduction_rate < 1.0:
        reasons.append(f"rule engine reproduces only {reproduction_rate:.1%} of the bulletins")
    sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
    return GateResult(
        lead=lead,
        candidate=candidate,
        best_reference=best_ref,
        passed=not reasons,
        reasons=reasons,
        metrics={
            "n": float(n),
            "brier_candidate": c_brier,
            "brier_reference": r_brier,
            "rel_gain": rel_gain,
            "bootstrap_upper": upper,
            "sd_paired_diff": sd,
            "mdd": minimum_detectable_difference(n, sd, gate.alpha) if n > 1 else float("nan"),
            "top_hit_candidate": float(cand_sum["top_hit"]),
            "top_hit_reference": float(refs.loc[best_ref, "top_hit"]),
            "ece": float(cand_sum["ece"]),
        },
    )


def run_lead(
    inputs: Inputs,
    lead: int,
    reproduction_rate: float,
    gate: Gate = GATE,
    model_factory: Callable[[str], Any] = make_model,
) -> tuple[pd.DataFrame, list[GateResult]]:
    """All candidates at one lead. Boosting is tried only when a linear candidate beats the
    best reference on mean Brier (a descriptive look, no selection by it)."""
    table = build_table(inputs, lead)
    linear = ["ridge", "linear_quantile"]
    forecasts = rolling_origin(table, inputs, linear, gate, model_factory)
    if forecasts.empty:
        return forecasts, []
    scored = score_table(forecasts)
    summary = summarise(scored, gate).set_index("model")
    best_ref = float(summary.loc[[r for r in REFERENCES if r in summary.index], "brier"].min())
    names = list(linear)
    if float(summary.loc[linear, "brier"].min()) < best_ref:
        extra = rolling_origin(table, inputs, ["boosting_quantile"], gate, model_factory)
        extra = extra[extra["model"] == "boosting_quantile"]
        scored = pd.concat([scored, score_table(extra)], ignore_index=True)
        names.append("boosting_quantile")
    results = [evaluate_gate(scored, lead, c, len(names), reproduction_rate, gate) for c in names]
    return scored, results
