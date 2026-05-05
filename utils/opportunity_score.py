"""Stock Opportunity Score calculation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.local_data import connect_readonly
from utils.scoring_common import available_watchlists, load_prices, load_profiles, load_watchlist


FACTOR_WEIGHTS: dict[str, float] = {
    "price_momentum": 0.25,
    "revenue_growth": 0.20,
    "eps_fcf_improvement": 0.20,
    "valuation_reasonableness": 0.15,
    "financial_safety": 0.10,
    "industry_strength": 0.10,
}


FACTOR_LABELS: dict[str, str] = {
    "price_momentum": "Price Momentum",
    "revenue_growth": "Revenue Growth",
    "eps_fcf_improvement": "EPS / FCF Improvement",
    "valuation_reasonableness": "Valuation Reasonableness",
    "financial_safety": "Financial Safety",
    "industry_strength": "Industry Strength",
}

INDUSTRY_FACTOR_WEIGHTS: dict[str, float] = {
    "quality_score": 0.30,
    "growth_score": 0.25,
    "valuation_score": 0.20,
    "risk_protection_score": 0.15,
    "upside_momentum_score": 0.10,
}


INDUSTRY_FACTOR_LABELS: dict[str, str] = {
    "quality_score": "Quality",
    "growth_score": "Growth",
    "valuation_score": "Valuation",
    "risk_protection_score": "Risk Protection",
    "upside_momentum_score": "Upside Momentum",
}


MARKET_THEME_RULES: list[dict[str, Any]] = [
    {
        "theme": "AI Infrastructure",
        "tickers": {
            "NVDA",
            "AMD",
            "AVGO",
            "MRVL",
            "TSM",
            "ASML",
            "AMAT",
            "KLAC",
            "LRCX",
            "SMCI",
            "DELL",
            "HPE",
            "VRT",
            "ETN",
            "PWR",
            "ANET",
            "CIEN",
            "COHR",
            "LITE",
            "MU",
            "WDC",
            "STX",
            "PSTG",
            "NTAP",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "ORCL",
        },
        "keywords": [
            "semiconductor",
            "computer hardware",
            "communication equipment",
            "electrical equipment",
            "data center",
            "cloud",
        ],
    },
    {
        "theme": "Power & Grid",
        "tickers": {
            "VST",
            "CEG",
            "NRG",
            "NEE",
            "SO",
            "DUK",
            "AEP",
            "EXC",
            "XEL",
            "PCG",
            "PEG",
            "ETR",
            "FE",
            "CNP",
            "SRE",
            "EIX",
            "PWR",
            "ETN",
            "HUBB",
            "VRT",
            "GEV",
            "GNRC",
        },
        "keywords": [
            "utilities",
            "renewable utilities",
            "independent power",
            "electrical equipment",
            "engineering & construction",
            "regulated electric",
        ],
    },
    {
        "theme": "Memory & Storage",
        "tickers": {
            "MU",
            "WDC",
            "STX",
            "SNDK",
            "PSTG",
            "NTAP",
            "HPE",
            "DELL",
            "SAMSUNG",
            "SKHYNIX",
        },
        "keywords": ["memory", "storage", "disk drive", "data storage"],
    },
    {
        "theme": "Optical Communications",
        "tickers": {"CIEN", "COHR", "LITE", "NOK", "ERIC", "AAOI", "INFN", "FNSR", "ACIA", "AVGO", "MRVL"},
        "keywords": ["communication equipment", "optical", "photonics", "networking"],
    },
    {
        "theme": "AI Semiconductors",
        "tickers": {
            "NVDA",
            "AMD",
            "AVGO",
            "MRVL",
            "TSM",
            "ASML",
            "AMAT",
            "KLAC",
            "LRCX",
            "ARM",
            "QCOM",
            "INTC",
            "MU",
            "MCHP",
            "ON",
            "NXPI",
            "ADI",
            "TXN",
        },
        "keywords": ["semiconductor"],
    },
    {
        "theme": "Data Center Equipment",
        "tickers": {"SMCI", "DELL", "HPE", "VRT", "ETN", "PWR", "ANET", "NTAP", "PSTG", "WDC", "STX", "CSCO"},
        "keywords": ["computer hardware", "electrical equipment", "networking", "data center"],
    },
]


@dataclass(frozen=True)
class ScoreInputs:
    watchlist_name: str | None = None
    min_price_rows: int = 60


def _rank_score(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.notna().sum() <= 1:
        return pd.Series(np.where(values.notna(), 50.0, np.nan), index=series.index)
    ranked = values.rank(pct=True, ascending=higher_is_better, na_option="keep") * 100.0
    return ranked.clip(0, 100)


def _group_rank_score(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    *,
    higher_is_better: bool = True,
) -> pd.Series:
    values = pd.to_numeric(df.get(value_col, pd.Series(index=df.index)), errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    grouped = df[group_col].fillna("Unknown")

    def rank_group(group: pd.Series) -> pd.Series:
        if group.notna().sum() <= 1:
            return pd.Series(np.where(group.notna(), 50.0, np.nan), index=group.index)
        return group.rank(pct=True, ascending=higher_is_better, na_option="keep").mul(100).clip(0, 100)

    return values.groupby(grouped, group_keys=False).apply(rank_group)


def _mean_score(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].mean(axis=1, skipna=True)


def _json_float(payload: Any, key: str) -> float | None:
    if not payload:
        return None
    try:
        data = json.loads(str(payload))
    except (TypeError, json.JSONDecodeError):
        return None
    value = data.get(key)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _period_return(grp: pd.DataFrame, latest_price: float, latest_dt: datetime, days: int) -> float | None:
    target = (latest_dt - timedelta(days=days)).strftime("%Y-%m-%d")
    past = grp[grp["price_date"] <= target]
    if past.empty:
        return None
    base = past.iloc[-1]["price"]
    if pd.isna(base) or not base:
        return None
    return (float(latest_price) - float(base)) / float(base) * 100.0


def _price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for ticker, grp in price_df.groupby("ticker"):
        grp = grp.dropna(subset=["price"]).sort_values("price_date")
        if grp.empty:
            continue
        latest = grp.iloc[-1]
        latest_dt = datetime.strptime(str(latest["price_date"]), "%Y-%m-%d")
        price = float(latest["price"])
        trailing_252 = grp.tail(252)["price"].dropna()
        trailing_126 = grp.tail(126)["price"].dropna()
        high_52w = float(trailing_252.max()) if not trailing_252.empty else np.nan
        low_52w = float(trailing_252.min()) if not trailing_252.empty else np.nan
        daily_returns = trailing_126.pct_change().dropna()
        volatility_6m = float(daily_returns.std() * np.sqrt(252) * 100.0) if not daily_returns.empty else np.nan
        rows.append(
            {
                "ticker": ticker,
                "latest_price": price,
                "price_date": latest["price_date"],
                "high_52w": high_52w,
                "low_52w": low_52w,
                "drawdown_from_52w_high_pct": (price / high_52w - 1.0) * 100.0 if high_52w else None,
                "downside_to_52w_low_pct": (low_52w / price - 1.0) * 100.0 if price else None,
                "volatility_6m_pct": volatility_6m,
                "ret_1m": _period_return(grp, price, latest_dt, 30),
                "ret_3m": _period_return(grp, price, latest_dt, 91),
                "ret_6m": _period_return(grp, price, latest_dt, 182),
                "ret_12m": _period_return(grp, price, latest_dt, 365),
            }
        )
    return pd.DataFrame(rows)


def _load_fundamentals(conn: sqlite3.Connection, tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(tickers))
    return pd.read_sql_query(
        f"""
        SELECT ticker, report_date, revenue, eps, free_cash_flow
        FROM fundamental_data
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, report_date
        """,
        conn,
        params=tickers,
    )


def _safe_growth(now: Any, old: Any) -> float | None:
    try:
        n = float(now)
        o = float(old)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(n) or not np.isfinite(o) or o == 0:
        return None
    return (n - o) / abs(o) * 100.0


def _fundamental_features(fund_df: pd.DataFrame) -> pd.DataFrame:
    if fund_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for ticker, grp in fund_df.groupby("ticker"):
        grp = grp.sort_values("report_date").copy()
        for col in ["revenue", "eps", "free_cash_flow"]:
            grp[col] = pd.to_numeric(grp[col], errors="coerce")
        if grp.empty:
            continue
        latest = grp.iloc[-1]
        prior_q = grp.iloc[-2] if len(grp) >= 2 else None
        prior_y = grp.iloc[-5] if len(grp) >= 5 else None
        rows.append(
            {
                "ticker": ticker,
                "latest_report_date": latest["report_date"],
                "revenue_yoy_pct": _safe_growth(latest["revenue"], prior_y["revenue"] if prior_y is not None else None),
                "revenue_qoq_pct": _safe_growth(latest["revenue"], prior_q["revenue"] if prior_q is not None else None),
                "eps_yoy_pct": _safe_growth(latest["eps"], prior_y["eps"] if prior_y is not None else None),
                "fcf_yoy_pct": _safe_growth(
                    latest["free_cash_flow"],
                    prior_y["free_cash_flow"] if prior_y is not None else None,
                ),
            }
        )
    return pd.DataFrame(rows)


def _load_key_metrics(conn: sqlite3.Connection, tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(tickers))
    df = pd.read_sql_query(
        f"""
        SELECT ticker, as_of_date, market_cap, enterprise_value, ev_to_sales,
               ev_to_operating_cash_flow, ev_to_free_cash_flow,
               net_debt_to_ebitda, payload_json
        FROM key_metrics_ttm
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, as_of_date
        """,
        conn,
        params=tickers,
    )
    if df.empty:
        return df
    latest = df.sort_values("as_of_date").drop_duplicates("ticker", keep="last").copy()
    latest["earnings_yield"] = latest["payload_json"].apply(lambda v: _json_float(v, "earningsYieldTTM"))
    latest["free_cash_flow_yield"] = latest["payload_json"].apply(lambda v: _json_float(v, "freeCashFlowYieldTTM"))
    latest["income_quality"] = latest["payload_json"].apply(lambda v: _json_float(v, "incomeQualityTTM"))
    return latest.drop(columns=["payload_json"])


def _load_price_targets(conn: sqlite3.Connection, tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(tickers))
    return pd.read_sql_query(
        f"""
        SELECT ticker, target_low, target_consensus, target_median, target_high, updated_at AS target_updated_at
        FROM price_target_consensus
        WHERE ticker IN ({placeholders})
        """,
        conn,
        params=tickers,
    )


def _classify_industry_candidate(row: pd.Series) -> str:
    score = row.get("industry_opportunity_score", np.nan)
    risk = row.get("risk_protection_score", np.nan)
    upside = row.get("upside_momentum_score", np.nan)
    valuation = row.get("valuation_score", np.nan)
    if pd.isna(score):
        return "Insufficient Data"
    if score >= 80 and risk >= 65 and upside >= 60:
        return "Attractive"
    if score >= 70 and risk >= 55:
        return "Watchlist"
    if valuation >= 70 and risk < 50:
        return "Cheap but Risky"
    if row.get("quality_score", 0) >= 75 and valuation < 45:
        return "High Quality but Expensive"
    if risk < 40:
        return "High Risk"
    return "Fair"


def _industry_thesis(row: pd.Series) -> str:
    notes: list[str] = []
    if row.get("industry_rank", np.nan) == 1:
        notes.append("Top industry rank")
    if row.get("risk_protection_score", 0) >= 65:
        notes.append("lower relative risk")
    if row.get("downside_protection_score", 0) >= 65:
        notes.append("limited downside profile")
    if row.get("upside_momentum_score", 0) >= 65:
        notes.append("continued upside support")
    if row.get("valuation_score", 0) >= 65:
        notes.append("reasonable valuation")
    if row.get("quality_score", 0) >= 65:
        notes.append("solid quality")
    return ", ".join(notes) if notes else "Needs further research"


def _themes_for_row(row: pd.Series) -> list[str]:
    ticker = str(row.get("ticker", "")).upper().strip()
    classification_text = " ".join(
        str(row.get(col, "") or "").lower()
        for col in ["sector", "industry"]
    )
    themes: list[str] = []
    for rule in MARKET_THEME_RULES:
        if ticker in rule["tickers"] or any(keyword in classification_text for keyword in rule["keywords"]):
            themes.append(str(rule["theme"]))
    return sorted(set(themes))


def _classify_theme_candidate(row: pd.Series) -> str:
    score = row.get("theme_opportunity_score", np.nan)
    risk = row.get("theme_risk_protection_score", np.nan)
    upside = row.get("theme_upside_momentum_score", np.nan)
    valuation = row.get("theme_valuation_score", np.nan)
    if pd.isna(score):
        return "Insufficient Data"
    if score >= 82 and risk >= 60 and upside >= 60:
        return "Recommended"
    if score >= 72 and risk >= 55:
        return "Positive Watchlist"
    if score >= 68 and upside >= 70 and risk < 50:
        return "High Upside High Risk"
    if valuation >= 70 and risk < 50:
        return "Cheap but Risky"
    if row.get("theme_quality_score", 0) >= 75 and valuation < 45:
        return "High Quality but Expensive"
    if risk < 40:
        return "High Risk"
    return "Neutral"


def _theme_thesis(row: pd.Series) -> str:
    notes: list[str] = []
    if row.get("theme_rank", np.nan) == 1:
        notes.append("Top theme rank")
    if row.get("theme_quality_score", 0) >= 65:
        notes.append("strong relative quality")
    if row.get("theme_growth_score", 0) >= 65:
        notes.append("above-theme growth")
    if row.get("theme_valuation_score", 0) >= 65:
        notes.append("reasonable valuation")
    if row.get("theme_risk_protection_score", 0) >= 65:
        notes.append("lower theme risk")
    if row.get("theme_upside_momentum_score", 0) >= 65:
        notes.append("upside momentum")
    return ", ".join(notes) if notes else "Needs further research"


def compute_opportunity_scores(db_path: Path, inputs: ScoreInputs | None = None) -> pd.DataFrame:
    inputs = inputs or ScoreInputs()
    if not db_path.exists():
        return pd.DataFrame()

    with connect_readonly(db_path, timeout=30) as conn:
        watch = load_watchlist(conn, inputs.watchlist_name)
        if watch.empty:
            return pd.DataFrame()
        tickers = [str(x).upper() for x in watch["ticker"].dropna().tolist()]
        profiles = load_profiles(conn)
        prices = _price_features(load_prices(conn, tickers, inputs.min_price_rows))
        fundamentals = _fundamental_features(_load_fundamentals(conn, tickers))
        key_metrics = _load_key_metrics(conn, tickers)
        price_targets = _load_price_targets(conn, tickers)

    df = watch.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for part in (profiles, prices, fundamentals, key_metrics, price_targets):
        if not part.empty:
            df = df.merge(part, on="ticker", how="left")

    if df.empty:
        return df

    for col in ["ret_1m", "ret_3m", "ret_6m", "ret_12m"]:
        df[f"{col}_score"] = _rank_score(df[col], higher_is_better=True) if col in df else np.nan
    df["price_momentum"] = _mean_score(df, ["ret_1m_score", "ret_3m_score", "ret_6m_score", "ret_12m_score"])

    df["revenue_yoy_score"] = _rank_score(df.get("revenue_yoy_pct", pd.Series(index=df.index)), higher_is_better=True)
    df["revenue_qoq_score"] = _rank_score(df.get("revenue_qoq_pct", pd.Series(index=df.index)), higher_is_better=True)
    df["revenue_growth"] = _mean_score(df, ["revenue_yoy_score", "revenue_qoq_score"])

    df["eps_yoy_score"] = _rank_score(df.get("eps_yoy_pct", pd.Series(index=df.index)), higher_is_better=True)
    df["fcf_yoy_score"] = _rank_score(df.get("fcf_yoy_pct", pd.Series(index=df.index)), higher_is_better=True)
    df["income_quality_score"] = _rank_score(df.get("income_quality", pd.Series(index=df.index)), higher_is_better=True)
    df["eps_fcf_improvement"] = _mean_score(df, ["eps_yoy_score", "fcf_yoy_score", "income_quality_score"])

    for col in ["ev_to_sales", "ev_to_operating_cash_flow", "ev_to_free_cash_flow"]:
        if col in df:
            df.loc[pd.to_numeric(df[col], errors="coerce") <= 0, col] = np.nan
        df[f"{col}_score"] = _rank_score(df.get(col, pd.Series(index=df.index)), higher_is_better=False)
    df["earnings_yield_score"] = _rank_score(df.get("earnings_yield", pd.Series(index=df.index)), higher_is_better=True)
    df["free_cash_flow_yield_score"] = _rank_score(df.get("free_cash_flow_yield", pd.Series(index=df.index)), higher_is_better=True)
    df["valuation_reasonableness"] = _mean_score(
        df,
        [
            "ev_to_sales_score",
            "ev_to_operating_cash_flow_score",
            "ev_to_free_cash_flow_score",
            "earnings_yield_score",
            "free_cash_flow_yield_score",
        ],
    )

    df["net_debt_to_ebitda_score"] = _rank_score(
        df.get("net_debt_to_ebitda", pd.Series(index=df.index)),
        higher_is_better=False,
    )
    df["financial_safety"] = _mean_score(df, ["net_debt_to_ebitda_score", "income_quality_score"])

    sector_strength = (
        df.groupby("sector", dropna=False)["price_momentum"]
        .mean()
        .rank(pct=True, ascending=True)
        .mul(100)
    )
    df["industry_strength"] = df["sector"].map(sector_strength)

    weighted = pd.Series(0.0, index=df.index)
    weight_seen = pd.Series(0.0, index=df.index)
    for factor, weight in FACTOR_WEIGHTS.items():
        vals = pd.to_numeric(df[factor], errors="coerce")
        weighted += vals.fillna(0) * weight
        weight_seen += np.where(vals.notna(), weight, 0.0)
    df["score"] = np.where(weight_seen > 0, weighted / weight_seen, np.nan)
    df["score"] = df["score"].round(0)
    df["type"] = df.apply(_classify_row, axis=1)
    df["judgment"] = df.apply(_judgment_row, axis=1)
    df = _add_industry_rankings(df)
    df["market_themes"] = df.apply(lambda row: ", ".join(_themes_for_row(row)), axis=1)
    df["primary_market_theme"] = df["market_themes"].str.split(", ").str[0].fillna("")

    return df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)


def build_theme_rankings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        for theme in _themes_for_row(row):
            item = row.to_dict()
            item["market_theme"] = theme
            rows.append(item)
    if not rows:
        return pd.DataFrame()

    themed = pd.DataFrame(rows)
    raw_to_final = {
        "quality_raw": "theme_quality_score",
        "growth_raw": "theme_growth_score",
        "valuation_raw": "theme_valuation_score",
        "risk_protection_raw": "theme_risk_protection_score",
        "upside_raw": "theme_upside_momentum_score",
    }
    for raw_col, final_col in raw_to_final.items():
        themed[final_col] = _group_rank_score(themed, raw_col, "market_theme", higher_is_better=True)

    weighted = pd.Series(0.0, index=themed.index)
    weight_seen = pd.Series(0.0, index=themed.index)
    factor_map = {
        "theme_quality_score": INDUSTRY_FACTOR_WEIGHTS["quality_score"],
        "theme_growth_score": INDUSTRY_FACTOR_WEIGHTS["growth_score"],
        "theme_valuation_score": INDUSTRY_FACTOR_WEIGHTS["valuation_score"],
        "theme_risk_protection_score": INDUSTRY_FACTOR_WEIGHTS["risk_protection_score"],
        "theme_upside_momentum_score": INDUSTRY_FACTOR_WEIGHTS["upside_momentum_score"],
    }
    for factor, weight in factor_map.items():
        vals = pd.to_numeric(themed[factor], errors="coerce")
        weighted += vals.fillna(0) * weight
        weight_seen += np.where(vals.notna(), weight, 0.0)

    themed["theme_opportunity_score"] = np.where(weight_seen > 0, weighted / weight_seen, np.nan)
    themed["theme_opportunity_score"] = pd.to_numeric(themed["theme_opportunity_score"], errors="coerce").round(0)
    themed["theme_peer_count"] = themed.groupby("market_theme")["ticker"].transform("count")
    themed["theme_rank"] = (
        themed.groupby("market_theme")["theme_opportunity_score"]
        .rank(method="first", ascending=False, na_option="bottom")
        .astype("Int64")
    )
    themed["theme_candidate_type"] = themed.apply(_classify_theme_candidate, axis=1)
    themed["theme_thesis"] = themed.apply(_theme_thesis, axis=1)
    themed["top_theme_candidate"] = np.where(
        (themed["theme_rank"] <= 5)
        & (themed["theme_opportunity_score"] >= 70)
        & (themed["theme_risk_protection_score"] >= 55),
        "Yes",
        "No",
    )
    return themed.sort_values(
        ["market_theme", "theme_rank", "theme_opportunity_score"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def available_market_themes() -> list[str]:
    return [str(rule["theme"]) for rule in MARKET_THEME_RULES]


def _add_industry_rankings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["industry_group"] = (
        df.get("industry", pd.Series(index=df.index, dtype="object"))
        .fillna(df.get("sector", pd.Series(index=df.index, dtype="object")))
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    latest_price = pd.to_numeric(df.get("latest_price", pd.Series(index=df.index)), errors="coerce")
    target_consensus = pd.to_numeric(df.get("target_consensus", pd.Series(index=df.index)), errors="coerce")
    target_low = pd.to_numeric(df.get("target_low", pd.Series(index=df.index)), errors="coerce")
    df["target_upside_pct"] = np.where(
        (latest_price > 0) & target_consensus.notna(),
        (target_consensus / latest_price - 1.0) * 100.0,
        np.nan,
    )
    df["target_low_downside_pct"] = np.where(
        (latest_price > 0) & target_low.notna(),
        (target_low / latest_price - 1.0) * 100.0,
        np.nan,
    )

    df["quality_raw"] = _mean_score(df, ["eps_fcf_improvement", "financial_safety"])
    df["growth_raw"] = _mean_score(df, ["revenue_growth", "eps_fcf_improvement"])
    df["valuation_raw"] = df.get("valuation_reasonableness", pd.Series(index=df.index))
    df["target_upside_score"] = _group_rank_score(
        df,
        "target_upside_pct",
        "industry_group",
        higher_is_better=True,
    )
    df["upside_raw"] = _mean_score(df, ["price_momentum", "target_upside_score"])

    df["low_volatility_score"] = _group_rank_score(
        df,
        "volatility_6m_pct",
        "industry_group",
        higher_is_better=False,
    )
    df["low_drawdown_score"] = _group_rank_score(
        df,
        "drawdown_from_52w_high_pct",
        "industry_group",
        higher_is_better=True,
    )
    df["target_downside_score"] = _group_rank_score(
        df,
        "target_low_downside_pct",
        "industry_group",
        higher_is_better=True,
    )
    df["downside_protection_score"] = _mean_score(
        df,
        ["low_volatility_score", "low_drawdown_score", "target_downside_score"],
    )
    df["risk_protection_raw"] = _mean_score(df, ["financial_safety", "downside_protection_score"])

    raw_to_final = {
        "quality_raw": "quality_score",
        "growth_raw": "growth_score",
        "valuation_raw": "valuation_score",
        "risk_protection_raw": "risk_protection_score",
        "upside_raw": "upside_momentum_score",
    }
    for raw_col, final_col in raw_to_final.items():
        df[final_col] = _group_rank_score(df, raw_col, "industry_group", higher_is_better=True)

    weighted = pd.Series(0.0, index=df.index)
    weight_seen = pd.Series(0.0, index=df.index)
    for factor, weight in INDUSTRY_FACTOR_WEIGHTS.items():
        vals = pd.to_numeric(df[factor], errors="coerce")
        weighted += vals.fillna(0) * weight
        weight_seen += np.where(vals.notna(), weight, 0.0)
    df["industry_opportunity_score"] = np.where(weight_seen > 0, weighted / weight_seen, np.nan)
    df["industry_opportunity_score"] = pd.to_numeric(df["industry_opportunity_score"], errors="coerce").round(0)
    df["industry_peer_count"] = df.groupby("industry_group")["ticker"].transform("count")
    df["industry_rank"] = (
        df.groupby("industry_group")["industry_opportunity_score"]
        .rank(method="first", ascending=False, na_option="bottom")
        .astype("Int64")
    )
    df["industry_candidate_type"] = df.apply(_classify_industry_candidate, axis=1)
    df["industry_thesis"] = df.apply(_industry_thesis, axis=1)
    df["top_industry_candidate"] = np.where(
        (df["industry_rank"] <= 5)
        & (df["industry_opportunity_score"] >= 70)
        & (df["risk_protection_score"] >= 55),
        "Yes",
        "No",
    )
    return df


def _classify_row(row: pd.Series) -> str:
    if row.get("revenue_growth", 0) >= 70 and row.get("price_momentum", 0) >= 70:
        return "Growth"
    if row.get("valuation_reasonableness", 0) >= 70 and row.get("financial_safety", 0) >= 70:
        return "Quality Value"
    if row.get("eps_fcf_improvement", 0) >= 65 and row.get("price_momentum", 100) < 65:
        return "Turnaround"
    if row.get("score", 0) >= 75:
        return "High-Quality Candidate"
    if row.get("score", 0) >= 60:
        return "Watchlist"
    return "Low Priority"


def _judgment_row(row: pd.Series) -> str:
    notes: list[str] = []
    if row.get("price_momentum", 0) >= 75:
        notes.append("Strong momentum")
    elif row.get("price_momentum", 100) <= 40:
        notes.append("Weak momentum")
    if row.get("valuation_reasonableness", 100) <= 40:
        notes.append("High valuation")
    elif row.get("valuation_reasonableness", 0) >= 70:
        notes.append("Reasonable valuation")
    if row.get("financial_safety", 100) <= 40:
        notes.append("High financial risk")
    if row.get("eps_fcf_improvement", 0) >= 70:
        notes.append("Earnings/cash flow improving")
    if row.get("score", 0) >= 80 and "High valuation" in notes:
        return "Strong but expensive"
    if row.get("score", 0) >= 75 and "Reasonable valuation" in notes:
        return "Reasonable valuation"
    if row.get("score", 100) < 65 and row.get("eps_fcf_improvement", 0) >= 65:
        return "High-risk turnaround"
    return ", ".join(notes) if notes else "Needs further research"
