from __future__ import annotations

from datetime import date
from decimal import Decimal as Dec

import pytest

from spillway_forecast.flag_rule import (
    RULE_B,
    RULE_C,
    Flag,
    RuleVersion,
    apply_gfom,
    compare_bands,
    compute_bands,
    flag_for_month,
    flag_from_trigger,
    flag_probabilities,
    gsf_band,
    rule_for_month,
)

# (rule, pld_min, pld_max, gsf, trigger, green, yellow, red1, colour): the six reproduced
# bulletins of docs/flag-forecast-discovery.md section 4.
BULLETINS = [
    (RULE_B, "55.70", "646.58", "0.83", "55.70", "187.81", "406.23", "543.93", Flag.GREEN),
    (RULE_B, "69.04", "684.73", "0.74", "69.04", "122.80", "265.61", "355.65", Flag.GREEN),
    (RULE_C, "61.07", "716.80", "0.65", "254.18", "78.51", "197.12", "271.58", Flag.RED1),
    (RULE_C, "58.60", "751.73", "0.94", "208.03", "458.00", "751.73", "751.73", Flag.GREEN),
    (RULE_C, "58.60", "751.73", "0.84", "304.02", "171.75", "431.20", "594.08", Flag.YELLOW),
    (RULE_C, "57.31", "758.27", "0.72", "128.67", "98.14", "246.40", "339.47", Flag.YELLOW),
]


@pytest.mark.parametrize(("rule", "lo", "hi", "gsf", "trig", "g", "y", "r1", "colour"), BULLETINS)
def test_reproduces_bulletins(
    rule: RuleVersion, lo: str, hi: str, gsf: str, trig: str, g: str, y: str, r1: str, colour: Flag
) -> None:
    bands = compute_bands(lo, hi, gsf, rule)
    assert (bands.green_upper, bands.yellow_upper, bands.red1_upper) == (Dec(g), Dec(y), Dec(r1))
    assert flag_from_trigger(bands, trig) is colour
    assert compare_bands(bands, g, y, r1) == []


def test_gsf_rounds_half_up_to_two_decimals() -> None:
    assert gsf_band(Dec("72.5"), Dec("100")) == Dec("0.73")
    assert gsf_band(Dec("0.7249"), Dec("1")) == Dec("0.72")
    with pytest.raises(ValueError, match="positive"):
        gsf_band(1, 0)


def test_edges_are_inclusive_on_the_upper_side() -> None:
    bands = compute_bands("57.31", "758.27", "0.72")
    assert flag_from_trigger(bands, "98.14") is Flag.GREEN
    assert flag_from_trigger(bands, "98.15") is Flag.YELLOW
    assert flag_from_trigger(bands, "246.40") is Flag.YELLOW
    assert flag_from_trigger(bands, "246.41") is Flag.RED1
    assert flag_from_trigger(bands, "339.47") is Flag.RED1
    assert flag_from_trigger(bands, "339.48") is Flag.RED2


def test_edges_clamp_to_pld_range() -> None:
    low = compute_bands("55.70", "646.58", "0.0")  # Lim itself is below PLDmin
    assert low.green_upper == Dec("55.70")
    high = compute_bands("58.60", "751.73", "0.94")
    assert high.yellow_upper == high.red1_upper == Dec("751.73")
    # With red 1 and red 2 empty, a trigger at PLDmax stays yellow.
    assert flag_from_trigger(high, "751.73") is Flag.YELLOW


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="exceed"):
        compute_bands("10", "5", "0.5")
    with pytest.raises(ValueError, match="gsf"):
        compute_bands("5", "10", "1.0")


def test_full_chain_from_gh_gf() -> None:
    flag, bands = flag_for_month("57.31", "758.27", "38088", "52900", "128.67")
    assert bands.gsf == Dec("0.72")
    assert flag is Flag.YELLOW


def test_gfom_composition() -> None:
    assert apply_gfom(Flag.GREEN, 6962) is Flag.GREEN
    assert apply_gfom(Flag.GREEN, 6963) is Flag.YELLOW
    assert apply_gfom(Flag.GREEN, 16562) is Flag.RED2
    assert apply_gfom(Flag.GREEN, 99999) is Flag.RED2
    assert apply_gfom(Flag.YELLOW, 9262) is Flag.YELLOW
    assert apply_gfom(Flag.YELLOW, 9263) is Flag.RED1
    assert apply_gfom(Flag.RED1, 11463) is Flag.RED2
    assert apply_gfom(Flag.RED2, 0) is Flag.RED2
    flag, _ = flag_for_month("57.31", "758.27", "38088", "52900", "50", 12000)
    assert flag is Flag.RED1  # green + 12.000 MWm
    # rule B has no GFOM: the dispatch is ignored
    flag_b, _ = flag_for_month("55.70", "646.58", "83", "100", "55.70", 12000, RULE_B)
    assert flag_b is Flag.GREEN


def test_compare_bands_reports_divergence() -> None:
    bands = compute_bands("57.31", "758.27", "0.72")
    assert compare_bands(bands, "98.14", "246.40", "339.47") == []
    assert compare_bands(bands, "98.14", "246.40", "339.48") == []  # within one cent
    assert len(compare_bands(bands, "98.14", "246.40", "340.00")) == 1


def test_rule_version_by_month() -> None:
    assert rule_for_month(date(2024, 4, 1)) is RULE_C
    assert rule_for_month(date(2024, 3, 1)) is RULE_B
    assert "to be confirmed" in RULE_C.status


def test_probabilities_sum_to_one_and_follow_bands() -> None:
    bands = compute_bands("57.31", "758.27", "0.72")
    p = flag_probabilities(bands, [50, 100, 100, 300, 500])
    assert sum(p.values()) == pytest.approx(1.0)
    assert p[Flag.GREEN] == 0.2
    assert p[Flag.YELLOW] == 0.4
    assert p[Flag.RED1] == 0.2
    assert p[Flag.RED2] == 0.2
    with pytest.raises(ValueError, match="empty"):
        flag_probabilities(bands, [])
