"""Backtest of the rule engine against the bulletins, plus the data loaders of the flag forecast.

``reproduce`` applies ``flag_rule`` to every month of the seed (numeric facts read from the CCEE
bulletins, loaded into ``flag_rule_month`` by spillway-data) and compares bands and colour with
the bulletin and with the ANEEL series. Decision point 1 of the spec is its reproduction rate.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from spillway_forecast.flag_rule import (
    Flag,
    compare_bands,
    compute_bands,
    flag_from_trigger,
    rule_for_month,
)

SEED_DIR = Path(__file__).parent / "flag_seed"
RULES_CSV = SEED_DIR / "flag_rule_month.csv"
ANEEL_CSV = SEED_DIR / "aneel_flags.csv"


def read_rules(path: Path = RULES_CSV) -> pd.DataFrame:
    """Seed rows with the names the model code uses."""
    raw = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "month": pd.to_datetime(raw["month"]).dt.date,
            "rule_version": raw["rule_version"],
            "pld_min": raw["pld_min_brl_per_mwh"],
            "pld_max": raw["pld_max_brl_per_mwh"],
            "gh": raw["gh_mwmed"],
            "gf": raw["gf_mwmed"],
            "gsf": raw["gsf"],
            "trigger": raw["pld_trigger_brl_per_mwh"],
            "gfom": raw["gfom_mwmed"],
            "green_upper": raw["green_upper_brl_per_mwh"],
            "yellow_upper": raw["yellow_upper_brl_per_mwh"],
            "red1_upper": raw["red1_upper_brl_per_mwh"],
            "flag": raw["flag"],
            "bulletin_number": raw["bulletin_number"],
            "announced_on": pd.to_datetime(raw["announced_on"]).dt.date,
            "revised_on": pd.to_datetime(raw["revised_on"]).dt.date,
        }
    )
    return out.sort_values("month").reset_index(drop=True)


def read_aneel_flags(path: Path = ANEEL_CSV) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return pd.DataFrame({"month": pd.to_datetime(raw["month"]).dt.date, "flag": raw["flag"]})


def _dec(x: Any) -> str:
    return f"{float(x):.2f}"


def reproduce(rules: pd.DataFrame, aneel: pd.DataFrame) -> pd.DataFrame:
    """One row per month: does the engine match the bulletin bands and the published colour?"""
    aneel_flag = dict(zip(aneel["month"], aneel["flag"], strict=True))
    rows = []
    for r in rules.itertuples():
        month: date = r.month
        rule = rule_for_month(month)
        bands = compute_bands(_dec(r.pld_min), _dec(r.pld_max), _dec(r.gsf), rule)
        diffs = compare_bands(
            bands,
            Decimal(_dec(r.green_upper)),
            Decimal(_dec(r.yellow_upper)),
            Decimal(_dec(r.red1_upper)),
        )
        colour = flag_from_trigger(bands, _dec(r.trigger))
        official = aneel_flag.get(month)
        rows.append(
            {
                "month": month,
                "bulletin": int(r.bulletin_number),
                "rule_version": rule.name,
                "bands_match": not diffs,
                "engine_flag": colour.value,
                "bulletin_flag": r.flag,
                "aneel_flag": official,
                "colour_match": colour.value == official,
                "revised": pd.notna(r.revised_on),
                "diffs": "; ".join(diffs),
            }
        )
    return pd.DataFrame(rows)


def reproduction_rate(result: pd.DataFrame) -> float:
    ok = result["bands_match"] & result["colour_match"]
    return float(ok.mean()) if len(result) else 0.0


def report_markdown(result: pd.DataFrame, rules: pd.DataFrame) -> str:
    """Reproduction report (Portuguese, numbers only)."""
    n = len(result)
    bands = int(result["bands_match"].sum())
    colours = int(result["colour_match"].sum())
    both = int((result["bands_match"] & result["colour_match"]).sum())
    by_colour = result.groupby("aneel_flag").size().to_dict()
    lines = [
        "# Reprodução da regra da bandeira: motor contra os boletins",
        "",
        f"Meses testados: {n} (boletins nº {int(result['bulletin'].min())} a "
        f"{int(result['bulletin'].max())}, {result['month'].min():%m/%Y} a "
        f"{result['month'].max():%m/%Y}).",
        "",
        "| Verificação | Acertos | Taxa |",
        "|---|---|---|",
        f"| Faixas ao centavo (tolerância 0,01) | {bands}/{n} | {bands / n:.1%} |",
        f"| Cor igual à série da ANEEL | {colours}/{n} | {colours / n:.1%} |",
        f"| Faixas e cor | {both}/{n} | {both / n:.1%} |",
        "",
        "Cores na amostra: " + ", ".join(f"{k} {v}" for k, v in sorted(by_colour.items())) + ".",
        "",
        "| Versão da regra | Meses | Faixas ok |",
        "|---|---|---|",
    ]
    for version, g in result.groupby("rule_version"):
        lines.append(f"| {version} | {len(g)} | {int(g['bands_match'].sum())}/{len(g)} |")
    bad = result[~(result["bands_match"] & result["colour_match"])]
    lines += ["", "Divergências: " + ("nenhuma." if bad.empty else "")]
    for r in bad.itertuples():
        lines.append(f"- {r.month:%m/%Y} (boletim {r.bulletin}): {r.diffs or 'cor'}")
    both_rules = int(
        sum(
            compute_bands(
                _dec(r.pld_min), _dec(r.pld_max), _dec(r.gsf), rule_for_month(r.month)
            ).green_upper
            == compute_bands(_dec(r.pld_min), _dec(r.pld_max), _dec(r.gsf), alt).green_upper
            and compare_bands(
                compute_bands(_dec(r.pld_min), _dec(r.pld_max), _dec(r.gsf), alt),
                _dec(r.green_upper),
                _dec(r.yellow_upper),
                _dec(r.red1_upper),
            )
            == []
            for r in rules.itertuples()
            for alt in [_other_rule(r.month)]
        )
    )
    lines += [
        "",
        "Meses em que a versão da regra NÃO escolhida pela data também reproduz as faixas: "
        f"{both_rules} (GSF alto deixa tudo verde, então as duas versões coincidem). "
        "Essa troca B para C só é "
        "provada pela norma (abr/2024), não pelos dados.",
        "",
        "Limites desta checagem: as constantes de cada versão foram ajustadas nas faixas dos "
        "próprios "
        "boletins (a confirmar no Anexo I da REH 3.306/2024); a reprodução prova consistência, não "
        "que as constantes oficiais sejam essas. Nenhum mês da amostra tem GFOM maior que zero.",
        "",
    ]
    return "\n".join(lines)


def _other_rule(month: date) -> Any:
    from spillway_forecast.flag_rule import RULE_B, RULE_C

    return RULE_B if rule_for_month(month) is RULE_C else RULE_C


def check_flag_values() -> list[str]:
    return [f.value for f in Flag]


def load_warehouse_inputs(dsn: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """SELECT-only: weekly PLD (SE), daily storage and the ANEEL colours from the warehouse."""
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT week_start, subsystem, pld_avg_brl_per_mwh FROM pld_weekly ORDER BY week_start"
        )
        pld = pd.DataFrame(cur.fetchall(), columns=["week_start", "subsystem", "pld"])
        cur.execute(
            "SELECT day, subsystem, storage_pct, ena_gross_pct_mlt FROM subsystem_storage_daily "
            "ORDER BY day"
        )
        storage = pd.DataFrame(
            cur.fetchall(), columns=["day", "subsystem", "storage_pct", "ena_gross_pct_mlt"]
        )
        cur.execute("SELECT month, flag FROM tariff_flag_monthly ORDER BY month")
        flags = pd.DataFrame(cur.fetchall(), columns=["month", "flag"])
    pld["pld"] = pld["pld"].astype(float)
    for col in ("storage_pct", "ena_gross_pct_mlt"):
        storage[col] = storage[col].astype(float)
    return pld, storage, flags
