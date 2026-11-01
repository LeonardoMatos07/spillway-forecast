"""Tariff flag rule engine (bandeira tarifaria): a pure function of published numbers.

The CCEE InfoBandeira bulletin applies a rule to numbers: the GSF band (GH / GF, two decimals)
turns three constant limits per pattern into price bands, and the colour is the band that holds
the monthly "PLD trigger". This module reproduces that rule; it does no I/O.

Honesty note: the ``Lim`` constants below were FITTED on the band edges printed in the
bulletins themselves (see docs/flag-forecast-discovery.md, section 4). They are NOT read from
Annex I of ANEEL REH 3.306/2024 (the ANEEL site returned HTTP 403). Status: "adjusted, to be
confirmed in Annex I of REH 3.306/2024". The engine is reproducible against the bulletins, and
any month where it diverges from the bulletin must stop publication (see ``compare_bands``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

CENT = Decimal("0.01")


class Flag(StrEnum):
    """Colour labels as stored in ``tariff_flag_monthly`` / ``flag_rule_month``."""

    GREEN = "Verde"
    YELLOW = "Amarela"
    RED1 = "Vermelha P1"
    RED2 = "Vermelha P2"


FLAG_ORDER: tuple[Flag, ...] = (Flag.GREEN, Flag.YELLOW, Flag.RED1, Flag.RED2)

Num = Decimal | float | int | str


@dataclass(frozen=True)
class RuleVersion:
    """Constants of one rule version (numerators of ``Lim / (1 - GSF)``, R$/MWh)."""

    name: str
    lim_green: Decimal
    lim_yellow: Decimal
    lim_red1: Decimal
    gfom: bool
    effective_from: date | None
    status: str


_STATUS = "adjusted, to be confirmed in Annex I of REH 3.306/2024"

# Version B: fitted on only two bulletins (out/2022 and ago/2023). Validity range unknown.
RULE_B = RuleVersion(
    name="B",
    lim_green=Decimal("31.9275"),
    lim_yellow=Decimal("69.059"),
    lim_red1=Decimal("92.4683"),
    gfom=False,
    effective_from=None,
    status=_STATUS,
)
# Version C: fitted on four GSFs (0.65, 0.72, 0.84, 0.94); with GFOM; since 2024-04-01.
RULE_C = RuleVersion(
    name="C",
    lim_green=Decimal("27.4798"),
    lim_yellow=Decimal("68.992"),
    lim_red1=Decimal("95.0522"),
    gfom=True,
    effective_from=date(2024, 4, 1),
    status=_STATUS,
)
RULES: dict[str, RuleVersion] = {RULE_B.name: RULE_B, RULE_C.name: RULE_C}

# GFOM (out-of-merit thermal dispatch by energy security), InfoBandeira 92 (dec/2025), Table 2.
# Per original colour: successive upper limits (MWm) of the composed colour. Above the last
# limit the composed colour is Red 2. To be confirmed in REN 1.084/2024.
GFOM_LIMITS_MWM: dict[Flag, tuple[tuple[Decimal, Flag], ...]] = {
    Flag.GREEN: (
        (Decimal(6962), Flag.GREEN),
        (Decimal(11362), Flag.YELLOW),
        (Decimal(14262), Flag.RED1),
        (Decimal(16562), Flag.RED2),
    ),
    Flag.YELLOW: (
        (Decimal(9262), Flag.YELLOW),
        (Decimal(12162), Flag.RED1),
        (Decimal(16562), Flag.RED2),
    ),
    Flag.RED1: ((Decimal(11462), Flag.RED1), (Decimal(14262), Flag.RED2)),
    Flag.RED2: ((Decimal(11762), Flag.RED2),),
}
# Tariff addition (R$/100 kWh) of the composed colour, same table.
FLAG_ADDITION: dict[Flag, Decimal] = {
    Flag.GREEN: Decimal("0"),
    Flag.YELLOW: Decimal("18.85"),
    Flag.RED1: Decimal("44.63"),
    Flag.RED2: Decimal("78.77"),
}


@dataclass(frozen=True)
class Bands:
    """Upper edges of the first three colours; Red 2 runs up to ``pld_max``."""

    green_upper: Decimal
    yellow_upper: Decimal
    red1_upper: Decimal
    pld_min: Decimal
    pld_max: Decimal
    gsf: Decimal


def _dec(value: Num) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def round_half_up(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def gsf_band(gh_mwmed: Num, gf_mwmed: Num) -> Decimal:
    """GSF = GH / GF rounded to 2 decimals (half up), as ANEEL asked."""
    gf = _dec(gf_mwmed)
    if gf <= 0:
        raise ValueError("gf_mwmed must be positive")
    return round_half_up(_dec(gh_mwmed) / gf)


def rule_for_month(month: date) -> RuleVersion:
    """Rule version for a month: C from 2024-04, otherwise B (B is only verified for 2022-23)."""
    assert RULE_C.effective_from is not None
    return RULE_C if month >= RULE_C.effective_from else RULE_B


def compute_bands(pld_min: Num, pld_max: Num, gsf: Num, rule: RuleVersion = RULE_C) -> Bands:
    """Price bands: ``min(PLDmax, max(PLDmin, Lim / (1 - GSF)))`` per pattern, to the cent."""
    lo, hi, g = _dec(pld_min), _dec(pld_max), _dec(gsf)
    if lo > hi:
        raise ValueError("pld_min must not exceed pld_max")
    if g < 0:
        raise ValueError("gsf must not be negative")

    def edge(lim: Decimal) -> Decimal:
        # GSF >= 1: the limit is unbounded (division by zero or negative), so it clamps to PLDmax.
        if g >= 1:
            return round_half_up(hi)
        return round_half_up(min(hi, max(lo, lim / (Decimal(1) - g))))

    return Bands(
        green_upper=edge(rule.lim_green),
        yellow_upper=edge(rule.lim_yellow),
        red1_upper=edge(rule.lim_red1),
        pld_min=lo,
        pld_max=hi,
        gsf=g,
    )


def flag_from_trigger(bands: Bands, pld_trigger: Num) -> Flag:
    """Colour of the band that holds the PLD trigger (edges inclusive on the upper side)."""
    p = round_half_up(_dec(pld_trigger))
    if p <= bands.green_upper:
        return Flag.GREEN
    if p <= bands.yellow_upper:
        return Flag.YELLOW
    if p <= bands.red1_upper:
        return Flag.RED1
    return Flag.RED2


def apply_gfom(flag: Flag, thermal_dispatch_mwm: Num) -> Flag:
    """Compose the original colour with the out-of-merit thermal dispatch (GFOM, since 04/2024)."""
    d = _dec(thermal_dispatch_mwm)
    for upper, composed in GFOM_LIMITS_MWM[flag]:
        if d <= upper:
            return composed
    return Flag.RED2


def flag_for_month(
    pld_min: Num,
    pld_max: Num,
    gh_mwmed: Num,
    gf_mwmed: Num,
    pld_trigger: Num,
    thermal_dispatch_mwm: Num | None = None,
    rule: RuleVersion = RULE_C,
) -> tuple[Flag, Bands]:
    """Full chain GH, GF -> GSF -> bands -> colour (-> GFOM composition when the rule has it)."""
    bands = compute_bands(pld_min, pld_max, gsf_band(gh_mwmed, gf_mwmed), rule)
    flag = flag_from_trigger(bands, pld_trigger)
    if rule.gfom and thermal_dispatch_mwm is not None:
        flag = apply_gfom(flag, thermal_dispatch_mwm)
    return flag, bands


def compare_bands(
    computed: Bands,
    bulletin_green: Num,
    bulletin_yellow: Num,
    bulletin_red1: Num,
    tolerance: Decimal = CENT,
) -> list[str]:
    """Divergences (empty when the engine matches the bulletin). Use it to halt publication."""
    out: list[str] = []
    for name, mine, theirs in (
        ("green_upper", computed.green_upper, _dec(bulletin_green)),
        ("yellow_upper", computed.yellow_upper, _dec(bulletin_yellow)),
        ("red1_upper", computed.red1_upper, _dec(bulletin_red1)),
    ):
        if abs(mine - theirs) > tolerance:
            out.append(f"{name}: engine {mine} vs bulletin {theirs}")
    return out


def flag_probabilities(bands: Bands, trigger_samples: list[Num]) -> dict[Flag, float]:
    """Colour probabilities: the month's bands applied to a predicted trigger distribution."""
    if not trigger_samples:
        raise ValueError("trigger_samples must not be empty")
    counts = dict.fromkeys(FLAG_ORDER, 0)
    for s in trigger_samples:
        counts[flag_from_trigger(bands, s)] += 1
    n = len(trigger_samples)
    return {f: c / n for f, c in counts.items()}
