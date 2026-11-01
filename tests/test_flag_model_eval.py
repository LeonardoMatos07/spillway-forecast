# mypy: disable-error-code="arg-type,assignment,operator,index,call-overload,union-attr"
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from spillway_forecast import flag_backtest as bt
from spillway_forecast import flag_eval as ev
from spillway_forecast import flag_model as fm
from spillway_forecast.flag_rule import RULE_C, Flag, compute_bands, flag_from_trigger


def light(name: str) -> Any:
    """Same models with a coarser quantile grid, to keep the tests fast."""
    if name == "linear_quantile":
        return fm.LinearQuantile((0.1, 0.3, 0.5, 0.7, 0.9))
    if name == "boosting_quantile":
        return fm.BoostingQuantile((0.1, 0.5, 0.9), n_estimators=6)
    return fm.make_model(name)


def month_range(start: date, n: int) -> list[date]:
    out, m = [], start
    for _ in range(n):
        out.append(m)
        m = (m.replace(day=28) + timedelta(days=4)).replace(day=1)
    return out


def synthetic_inputs(n_months: int = 60, signal: float = 1.0, seed: int = 1) -> ev.Inputs:
    """Months whose trigger depends on the storage level (signal) plus noise."""
    rng = np.random.default_rng(seed)
    months = month_range(date(2020, 1, 1), n_months)
    ear = {m: float(np.clip(rng.uniform(15, 90), 10, 95)) for m in months}
    storage, weekly, rules = [], [], []
    start = months[0] - timedelta(days=60)
    span = (months[-1] - start).days + 40
    for i in range(span):
        day = start + timedelta(days=i)
        level = ear.get(day.replace(day=1), 60.0)
        for sub in ("N", "NE", "S", "SE"):
            storage.append((day, sub, level, 100.0 - level))
    for i in range(0, span, 7):
        day = start + timedelta(days=i)
        level = ear.get(day.replace(day=1), 60.0)
        weekly.append((day, "SE", float(np.clip(400 - 4 * level + rng.normal(0, 20), 60, 700))))
    flags = []
    for i, m in enumerate(months):
        noise = rng.normal(0, 0.25 / max(signal, 0.1))
        trig = float(
            np.clip(
                np.exp(4.0 + signal * 0.03 * (80 - ear[months[max(i - 1, 0)]]) + noise),
                61.07,
                716.8,
            )
        )
        gsf = float(np.round(np.clip(0.75 + rng.normal(0, 0.02), 0.4, 1.2), 2))
        bands = compute_bands("61.07", "716.80", str(gsf), RULE_C)
        flag = flag_from_trigger(bands, str(round(trig, 2))).value
        flags.append((m, flag))
        rules.append(
            {
                "month": m,
                "flag": flag,
                "pld_min": 61.07,
                "pld_max": 716.80,
                "trigger": round(trig, 2),
                "gsf": gsf,
                "announced_on": fm.default_announcement(m),
            }
        )
    return ev.Inputs(
        rules=pd.DataFrame(rules),
        flags=pd.DataFrame(flags, columns=["month", "flag"]),
        pld_weekly=pd.DataFrame(weekly, columns=["week_start", "subsystem", "pld"]),
        storage_daily=pd.DataFrame(
            storage, columns=["day", "subsystem", "storage_pct", "ena_gross_pct_mlt"]
        ),
    )


# --- engine backtest on the real seed ---------------------------------------------------------


def test_engine_reproduces_every_seeded_bulletin_and_the_aneel_colour() -> None:
    rules, aneel = bt.read_rules(), bt.read_aneel_flags()
    result = bt.reproduce(rules, aneel)
    assert len(result) == 37
    assert bt.reproduction_rate(result) == 1.0
    text = bt.report_markdown(result, rules)
    assert "37/37" in text
    assert "nenhuma" in text


def test_reproduction_detects_a_wrong_band() -> None:
    rules = bt.read_rules()
    rules.loc[rules["month"] == date(2026, 8, 1), "yellow_upper"] = 250.0
    result = bt.reproduce(rules, bt.read_aneel_flags())
    assert bt.reproduction_rate(result) == pytest.approx(36 / 37)
    assert "yellow_upper" in " ".join(result["diffs"])


# --- features: no look-ahead ---------------------------------------------------------------------


def test_features_ignore_data_after_the_cutoff() -> None:
    inputs = synthetic_inputs(30)
    cutoff = date(2022, 3, 10)
    base = fm.build_features(
        cutoff, date(2022, 4, 1), "Verde", inputs.pld_weekly, inputs.storage_daily
    )
    pld = inputs.pld_weekly.copy()
    sto = inputs.storage_daily.copy()
    pld.loc[pld["week_start"] > cutoff, "pld"] = 9999.0
    sto.loc[sto["day"] >= cutoff, ["storage_pct", "ena_gross_pct_mlt"]] = -50.0
    shifted = fm.build_features(cutoff, date(2022, 4, 1), "Verde", pld, sto)
    assert shifted == base
    assert not np.isnan(base["log_pld_last"])
    assert not np.isnan(base["ear_se"])


def test_default_announcement_is_a_weekday_before_the_month() -> None:
    for m in month_range(date(2025, 1, 1), 24):
        a = fm.default_announcement(m)
        assert a.weekday() < 5
        assert a < m


# --- predictive distribution to colours ----------------------------------------------------------


def test_colour_probabilities_follow_the_bands() -> None:
    low = fm.Predictive(trigger=np.array([70.0, 80.0]), gsf=np.array([0.72]))
    high = fm.Predictive(trigger=np.array([500.0, 600.0]), gsf=np.array([0.72]))
    p_low = fm.colour_probabilities(low, 57.31, 785.27, RULE_C)
    p_high = fm.colour_probabilities(high, 57.31, 785.27, RULE_C)
    assert p_low.sum() == pytest.approx(1.0)
    assert p_low[0] == 1.0
    assert p_high[3] == 1.0
    # bulletin 100: green up to 98.14, yellow from 98.15
    mid = fm.Predictive(trigger=np.array([98.14, 98.15]), gsf=np.array([0.72]))
    assert list(fm.colour_probabilities(mid, 57.31, 785.27, RULE_C)[:2]) == [0.5, 0.5]
    # GSF uncertainty shifts the colour: GSF >= 1 makes everything green
    both = fm.Predictive(trigger=np.array([300.0]), gsf=np.array([0.72, 1.05]))
    p = fm.colour_probabilities(both, 57.31, 785.27, RULE_C)
    assert p[0] == 0.5
    assert p[2] == 0.5


@pytest.mark.parametrize("name", ["ridge", "linear_quantile", "boosting_quantile"])
def test_models_fit_and_return_distributions(name: str) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, len(fm.FEATURES)))
    x[3, 2] = np.nan
    y_t = 5 + 0.5 * x[:, 0] + rng.normal(0, 0.2, 30)
    y_g = 0.8 + rng.normal(0, 0.02, 30)
    model = fm.make_model(name)
    model.fit(x, y_t, y_g)
    pred = model.predict(x[5])
    assert len(pred.trigger) > 5
    assert (pred.trigger > 0).all()
    if name != "ridge":
        assert (np.diff(pred.trigger) >= 0).all()
    with pytest.raises(ValueError, match="unknown"):
        fm.make_model("nope")


def test_ridge_predicts_higher_trigger_when_the_signal_is_higher() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, len(fm.FEATURES)))
    y_t = 5 + 0.8 * x[:, 0] + rng.normal(0, 0.1, 40)
    model = fm.RidgeLoo()
    model.fit(x, y_t, np.full(40, 0.8))
    lo, hi = np.zeros(len(fm.FEATURES)), np.zeros(len(fm.FEATURES))
    lo[0], hi[0] = -1.5, 1.5
    assert np.median(model.predict(hi).trigger) > np.median(model.predict(lo).trigger)


# --- metrics --------------------------------------------------------------------------------------


def test_brier_logloss_hit_and_score_run() -> None:
    p = np.array([0.1, 0.7, 0.2, 0.0])
    assert ev.brier(p, 1) == pytest.approx(0.01 + 0.09 + 0.04)
    assert ev.log_loss(p, 3) == pytest.approx(-np.log(ev.LOG_LOSS_EPS))
    assert ev.top_hit(p, 1)
    assert not ev.top_hit(p, 0)
    row = ev.score_run(p, "Amarela")
    assert row["actual_flag"] == "Amarela"
    assert row["top_hit"]
    assert row["prob_actual"] == 0.7
    assert ev.score_run([0.25] * 4, "Escassez Hídrica")["actual_flag"] == "Vermelha P2"


def test_calibration_of_perfect_and_wrong_forecasts() -> None:
    actual = np.array([0, 1, 2, 3, 0, 1] * 5)
    assert ev.ece(np.eye(4)[actual], actual, 5) == pytest.approx(0.0)
    assert ev.ece(np.eye(4)[(actual + 1) % 4], actual, 5) == pytest.approx(0.5)


def test_minimum_detectable_difference_numbers() -> None:
    # assumption of docs/flag-publication-criterion.md: sd of paired monthly Brier diff = 0.35
    assert ev.minimum_detectable_difference(24, 0.35) == pytest.approx(0.1517, abs=1e-3)
    assert ev.minimum_detectable_difference(100, 0.35) < ev.minimum_detectable_difference(24, 0.35)


def test_references_are_probability_vectors() -> None:
    history = pd.DataFrame(
        {
            "month": [date(2020, 8, 1), date(2021, 8, 1), date(2021, 9, 1)],
            "flag": ["Verde", "Amarela", "Verde"],
        }
    )
    clim = ev.ref_month_climatology(date(2022, 8, 1), history)
    assert clim.sum() == pytest.approx(1.0)
    assert clim[0] == clim[1]
    assert clim[0] > clim[2]
    assert ev.ref_prev_colour("Vermelha P1").tolist() == [0, 0, 1, 0]
    overall = ev.ref_overall_frequency(["Verde", "Verde", "Amarela"])
    assert overall.sum() == pytest.approx(1.0)
    assert overall[0] > overall[1] > overall[2]
    # a later year is never in the climatology
    assert ev.ref_month_climatology(date(2020, 8, 1), history)[1] == pytest.approx(0.5 / 2.0)


# --- rolling origin and the gate -----------------------------------------------------------------


def test_rolling_origin_trains_only_on_labels_already_readable() -> None:
    inputs = synthetic_inputs(40)
    table = ev.build_table(inputs, 14)

    class Spy:
        name = "spy"

        def fit(self, x: Any, yt: Any, yg: Any) -> None:
            self.n = len(x)

        def predict(self, row: Any) -> fm.Predictive:
            return fm.Predictive(trigger=np.array([100.0]), gsf=np.array([0.7]))

    out = ev.rolling_origin(table, inputs, ["spy"], model_factory=lambda _name: Spy())
    spy = out[out["model"] == "spy"]
    assert len(spy) > 0
    valid = table[table["valid"]].reset_index(drop=True)
    for _, row in spy.iterrows():
        month_row = valid[valid["month"] == row["month"]].iloc[0]
        trainable = valid[
            (valid["month"] < row["month"])
            & (valid["month"].map(fm.label_available_on) <= month_row["cutoff"])
        ]
        assert row["n_train"] == len(trainable)
        assert row["n_train"] >= ev.GATE.min_train


def test_gate_passes_with_a_strong_signal_and_long_history() -> None:
    inputs = synthetic_inputs(90, signal=3.0, seed=3)
    scored, results = ev.run_lead(inputs, 14, reproduction_rate=1.0, model_factory=light)
    assert not scored.empty
    summary = ev.summarise(scored).set_index("model")
    assert summary.loc["ridge", "brier"] < summary.loc["ref_prev_colour", "brier"]
    ridge = next(r for r in results if r.candidate == "ridge")
    assert ridge.metrics["n"] >= ev.GATE.min_scored
    assert ridge.passed, ridge.reasons


def test_gate_refuses_short_histories_and_engine_divergence() -> None:
    short = synthetic_inputs(26, signal=3.0, seed=5)
    _, res_short = ev.run_lead(short, 14, reproduction_rate=1.0, model_factory=light)
    assert all(not r.passed for r in res_short)
    assert any("months scored" in why for r in res_short for why in r.reasons)
    strong = synthetic_inputs(90, signal=3.0, seed=3)
    _, res_div = ev.run_lead(strong, 14, reproduction_rate=0.95, model_factory=light)
    assert all(not r.passed for r in res_div)
    assert any("reproduces only" in why for r in res_div for why in r.reasons)


def test_gate_refuses_a_model_that_does_not_beat_the_references() -> None:
    inputs = synthetic_inputs(90, signal=0.0, seed=4)
    # no information in the features: shuffle the labels in time
    rng = np.random.default_rng(0)
    rules = inputs.rules.copy()
    for col in ("flag", "trigger", "gsf"):
        rules[col] = rng.permutation(rules[col].to_numpy())
    flags = inputs.flags.copy()
    flags["flag"] = rules["flag"].to_numpy()
    noisy = ev.Inputs(rules, flags, inputs.pld_weekly, inputs.storage_daily)
    _, results = ev.run_lead(noisy, 14, reproduction_rate=1.0, model_factory=light)
    assert results
    assert all(not r.passed for r in results)


def test_flag_enum_values_match_the_database_labels() -> None:
    assert [f.value for f in (Flag.GREEN, Flag.YELLOW, Flag.RED1, Flag.RED2)] == [
        "Verde",
        "Amarela",
        "Vermelha P1",
        "Vermelha P2",
    ]
