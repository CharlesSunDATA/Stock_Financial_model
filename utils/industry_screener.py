"""Industry ranking and stock screening engine."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.local_data import connect_readonly
from utils.scoring_common import load_prices


SCORE_WEIGHTS: dict[str, float] = {
    "growth_quality": 0.25,
    "profitability_quality": 0.25,
    "balance_sheet_safety": 0.20,
    "valuation_reasonableness": 0.15,
    "technical_downside_risk": 0.15,
}

SCORE_LABELS: dict[str, str] = {
    "growth_quality": "Growth Quality",
    "profitability_quality": "Profitability Quality",
    "balance_sheet_safety": "Balance Sheet Safety",
    "valuation_reasonableness": "Valuation Reasonableness",
    "technical_downside_risk": "Technical Downside Risk",
}


@dataclass(frozen=True)
class ScreenerInputs:
    watchlist_names: tuple[str, ...] = ("sp500_ndx", "sp500", "ndx", "nasdaq100", "QQQ")
    min_price_rows: int = 200


def default_db_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "quant_data.db"


def _now() -> str:
    return datetime.utcnow().replace(tzinfo=None).isoformat(timespec="seconds")


def _clean_ticker(value: Any) -> str:
    return str(value or "").strip().upper().replace(".", "-")


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _json_float(payload: Any, *keys: str) -> float | None:
    if not payload:
        return None
    try:
        data = json.loads(str(payload))
    except (TypeError, json.JSONDecodeError):
        return None
    for key in keys:
        value = data.get(key)
        parsed = _safe_float(value)
        if parsed is not None:
            return parsed
    return None


def _rank_score(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.notna().sum() <= 1:
        return pd.Series(np.where(values.notna(), 50.0, np.nan), index=series.index)
    return values.rank(pct=True, ascending=higher_is_better, na_option="keep").mul(100).clip(0, 100)


def _industry_rank_score(
    df: pd.DataFrame,
    column: str,
    *,
    higher_is_better: bool = True,
) -> pd.Series:
    values = pd.to_numeric(df.get(column, pd.Series(index=df.index)), errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    groups = df["industry"].fillna("Unknown").replace("", "Unknown")

    def _rank_group(group: pd.Series) -> pd.Series:
        if group.notna().sum() <= 1:
            return pd.Series(np.where(group.notna(), 50.0, np.nan), index=group.index)
        return group.rank(pct=True, ascending=higher_is_better, na_option="keep").mul(100).clip(0, 100)

    return values.groupby(groups, group_keys=False).apply(_rank_group)


def _mean_score(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].mean(axis=1, skipna=True)


def _weighted_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    weighted = pd.Series(0.0, index=df.index)
    weight_seen = pd.Series(0.0, index=df.index)
    for column, weight in weights.items():
        values = pd.to_numeric(df.get(column, pd.Series(index=df.index)), errors="coerce")
        weighted += values.fillna(0) * weight
        weight_seen += np.where(values.notna(), weight, 0.0)
    return pd.Series(np.where(weight_seen > 0, weighted / weight_seen, np.nan), index=df.index)


def _pct_change(now: Any, old: Any) -> float | None:
    current = _safe_float(now)
    previous = _safe_float(old)
    if current is None or previous is None or previous == 0:
        return None
    return (current - previous) / abs(previous) * 100.0


def available_watchlists(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    with connect_readonly(db_path) as conn:
        try:
            df = pd.read_sql_query(
                "SELECT DISTINCT watchlist_name FROM fmp_watchlist ORDER BY watchlist_name",
                conn,
            )
        except Exception:
            return []
    return [str(x) for x in df["watchlist_name"].dropna().tolist()]


def _load_universe(conn: sqlite3.Connection, names: tuple[str, ...]) -> pd.DataFrame:
    available = pd.read_sql_query(
        "SELECT DISTINCT watchlist_name FROM fmp_watchlist ORDER BY watchlist_name",
        conn,
    )
    available_names = set(available["watchlist_name"].dropna().astype(str))
    chosen = [name for name in names if name in available_names]
    if not chosen:
        return pd.DataFrame(columns=["ticker", "universe"])
    placeholders = ",".join("?" * len(chosen))
    rows = pd.read_sql_query(
        f"""
        SELECT ticker, watchlist_name
        FROM fmp_watchlist
        WHERE watchlist_name IN ({placeholders})
        """,
        conn,
        params=chosen,
    )
    if rows.empty:
        return pd.DataFrame(columns=["ticker", "universe"])
    rows["ticker"] = rows["ticker"].apply(_clean_ticker)
    universe = (
        rows.groupby("ticker")["watchlist_name"]
        .apply(lambda vals: ", ".join(sorted(set(str(v) for v in vals if str(v).strip()))))
        .reset_index(name="universe")
    )
    return universe


def _latest_by_ticker(conn: sqlite3.Connection, table: str, date_col: str, columns: list[str], tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(tickers))
    selected = ", ".join(["ticker", date_col] + columns)
    df = pd.read_sql_query(
        f"""
        SELECT {selected}
        FROM {table}
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, {date_col}
        """,
        conn,
        params=tickers,
    )
    if df.empty:
        return df
    return df.sort_values(date_col).drop_duplicates("ticker", keep="last").copy()


def _financial_features(conn: sqlite3.Connection, tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(tickers))

    income = pd.read_sql_query(
        f"""
        SELECT ticker, report_date, revenue, gross_profit, operating_income, eps
        FROM income_statement
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, report_date
        """,
        conn,
        params=tickers,
    )
    cash_flow = pd.read_sql_query(
        f"""
        SELECT ticker, report_date, free_cash_flow
        FROM cash_flow_statement
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, report_date
        """,
        conn,
        params=tickers,
    )
    balance = pd.read_sql_query(
        f"""
        SELECT ticker, report_date, total_assets, total_liabilities, total_debt
        FROM balance_sheet
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, report_date
        """,
        conn,
        params=tickers,
    )
    ratios = _latest_by_ticker(
        conn,
        "financial_ratios",
        "report_date",
        ["roe", "roic", "pe_ratio", "ev_to_ebitda"],
        tickers,
    )
    key_metrics = pd.read_sql_query(
        f"""
        SELECT ticker, as_of_date, market_cap, payload_json
        FROM key_metrics_ttm
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, as_of_date
        """,
        conn,
        params=tickers,
    )

    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        row: dict[str, Any] = {"ticker": ticker}

        inc = income[income["ticker"] == ticker].sort_values("report_date").copy()
        for col in ["revenue", "gross_profit", "operating_income", "eps"]:
            if col in inc:
                inc[col] = pd.to_numeric(inc[col], errors="coerce")
        if not inc.empty:
            latest = inc.iloc[-1]
            prior_year = inc.iloc[-5] if len(inc) >= 5 else None
            revenue = latest.get("revenue")
            row.update(
                {
                    "as_of_date": latest.get("report_date"),
                    "revenue_growth": _pct_change(revenue, prior_year.get("revenue") if prior_year is not None else None),
                    "eps_growth": _pct_change(latest.get("eps"), prior_year.get("eps") if prior_year is not None else None),
                    "gross_margin": (
                        _safe_float(latest.get("gross_profit")) / _safe_float(revenue) * 100.0
                        if _safe_float(revenue)
                        else None
                    ),
                    "operating_margin": (
                        _safe_float(latest.get("operating_income")) / _safe_float(revenue) * 100.0
                        if _safe_float(revenue)
                        else None
                    ),
                }
            )

        cf = cash_flow[cash_flow["ticker"] == ticker].sort_values("report_date").copy()
        if not cf.empty:
            row["free_cash_flow"] = _safe_float(cf.iloc[-1].get("free_cash_flow"))

        bal = balance[balance["ticker"] == ticker].sort_values("report_date").copy()
        if not bal.empty:
            latest_bal = bal.iloc[-1]
            assets = _safe_float(latest_bal.get("total_assets"))
            liabilities = _safe_float(latest_bal.get("total_liabilities"))
            debt = _safe_float(latest_bal.get("total_debt"))
            equity = assets - liabilities if assets is not None and liabilities is not None else None
            row["debt_to_equity"] = debt / equity if debt is not None and equity and equity > 0 else None

        ratio = ratios[ratios["ticker"] == ticker]
        if not ratio.empty:
            latest_ratio = ratio.iloc[-1]
            row.update(
                {
                    "roe": _safe_float(latest_ratio.get("roe")),
                    "roic": _safe_float(latest_ratio.get("roic")),
                    "pe_ratio": _safe_float(latest_ratio.get("pe_ratio")),
                    "ev_ebitda": _safe_float(latest_ratio.get("ev_to_ebitda")),
                }
            )

        km = key_metrics[key_metrics["ticker"] == ticker].sort_values("as_of_date")
        if not km.empty:
            latest_km = km.iloc[-1]
            row["market_cap"] = _safe_float(latest_km.get("market_cap"))
            row["forward_pe"] = _json_float(
                latest_km.get("payload_json"),
                "forwardPERatio",
                "forwardPE",
                "forwardPriceToEarningsRatio",
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _price_features(conn: sqlite3.Connection, tickers: list[str], min_price_rows: int) -> pd.DataFrame:
    price_df = load_prices(conn, tickers + ["SPY"], min_price_rows=min_price_rows)
    if price_df.empty:
        return pd.DataFrame(columns=["ticker"])
    market = price_df[price_df["ticker"] == "SPY"][["price_date", "price"]].rename(columns={"price": "market_price"})
    if market.empty:
        market = (
            price_df[price_df["ticker"].isin(tickers)]
            .pivot(index="price_date", columns="ticker", values="price")
            .mean(axis=1)
            .reset_index(name="market_price")
        )
    market_returns = market.sort_values("price_date").set_index("price_date")["market_price"].pct_change().dropna()

    rows: list[dict[str, Any]] = []
    for ticker, grp in price_df[price_df["ticker"].isin(tickers)].groupby("ticker"):
        grp = grp.dropna(subset=["price"]).sort_values("price_date").copy()
        if grp.empty:
            continue
        prices = pd.to_numeric(grp["price"], errors="coerce").dropna()
        if prices.empty:
            continue
        latest = grp.iloc[-1]
        trailing_252 = prices.tail(252)
        returns = prices.pct_change().dropna()
        joined_returns = pd.DataFrame(
            {
                "stock": returns.to_numpy(),
            },
            index=grp.loc[returns.index, "price_date"],
        ).join(market_returns.rename("market"), how="inner")
        market_var = joined_returns["market"].var() if not joined_returns.empty else np.nan
        beta = joined_returns["stock"].cov(joined_returns["market"]) / market_var if market_var and market_var > 0 else np.nan
        drawdown = prices / prices.cummax() - 1.0
        rows.append(
            {
                "ticker": ticker,
                "price_date": latest["price_date"],
                "price": float(prices.iloc[-1]),
                "high_52w": float(trailing_252.max()) if not trailing_252.empty else np.nan,
                "low_52w": float(trailing_252.min()) if not trailing_252.empty else np.nan,
                "ma50": float(prices.tail(50).mean()) if len(prices) >= 50 else np.nan,
                "ma200": float(prices.tail(200).mean()) if len(prices) >= 200 else np.nan,
                "beta": beta,
                "max_drawdown": float(drawdown.min() * 100.0) if not drawdown.empty else np.nan,
                "volatility": float(returns.tail(126).std() * np.sqrt(252) * 100.0) if not returns.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _load_companies(conn: sqlite3.Connection, universe: pd.DataFrame) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame()
    tickers = universe["ticker"].tolist()
    placeholders = ",".join("?" * len(tickers))
    profiles = pd.read_sql_query(
        f"""
        SELECT ticker, company_name, sector, industry
        FROM company_profile
        WHERE ticker IN ({placeholders})
        """,
        conn,
        params=tickers,
    )
    if not profiles.empty:
        profiles["ticker"] = profiles["ticker"].apply(_clean_ticker)
    out = universe.merge(profiles, on="ticker", how="left")
    out["company_name"] = out["company_name"].fillna(out["ticker"])
    out["sector"] = out["sector"].fillna("Unknown").replace("", "Unknown")
    out["industry"] = out["industry"].fillna(out["sector"]).replace("", "Unknown")
    return out


def compute_industry_screen(db_path: Path | None = None, inputs: ScreenerInputs | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    db = db_path or default_db_path()
    inputs = inputs or ScreenerInputs()
    if not db.exists():
        return pd.DataFrame(), pd.DataFrame()

    with sqlite3.connect(str(db), timeout=30) as conn:
        universe = _load_universe(conn, inputs.watchlist_names)
        companies = _load_companies(conn, universe)
        if companies.empty or "ticker" not in companies.columns:
            return pd.DataFrame(), pd.DataFrame()
        tickers = companies["ticker"].dropna().astype(str).tolist()
        financials = _financial_features(conn, tickers)
        prices = _price_features(conn, tickers, inputs.min_price_rows)

    if companies.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = companies.merge(financials, on="ticker", how="left").merge(prices, on="ticker", how="left")
    for metric in [
        "revenue_growth",
        "eps_growth",
        "gross_margin",
        "operating_margin",
        "free_cash_flow",
        "debt_to_equity",
        "roe",
        "roic",
        "pe_ratio",
        "forward_pe",
        "ev_ebitda",
        "price",
        "high_52w",
        "low_52w",
        "ma50",
        "ma200",
        "beta",
        "max_drawdown",
        "volatility",
        "market_cap",
    ]:
        if metric in df:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

    for col in ["pe_ratio", "forward_pe", "ev_ebitda"]:
        df.loc[pd.to_numeric(df[col], errors="coerce") <= 0, col] = np.nan

    df["revenue_growth_score"] = _industry_rank_score(df, "revenue_growth", higher_is_better=True)
    df["eps_growth_score"] = _industry_rank_score(df, "eps_growth", higher_is_better=True)
    df["growth_quality"] = _mean_score(df, ["revenue_growth_score", "eps_growth_score"])

    df["gross_margin_score"] = _industry_rank_score(df, "gross_margin", higher_is_better=True)
    df["operating_margin_score"] = _industry_rank_score(df, "operating_margin", higher_is_better=True)
    df["roe_score"] = _industry_rank_score(df, "roe", higher_is_better=True)
    df["roic_score"] = _industry_rank_score(df, "roic", higher_is_better=True)
    df["profitability_quality"] = _mean_score(df, ["gross_margin_score", "operating_margin_score", "roe_score", "roic_score"])

    df["debt_safety_score"] = _industry_rank_score(df, "debt_to_equity", higher_is_better=False)
    df["fcf_safety_score"] = _industry_rank_score(df, "free_cash_flow", higher_is_better=True)
    df["low_beta_score"] = _industry_rank_score(df, "beta", higher_is_better=False)
    df["balance_sheet_safety"] = _mean_score(df, ["debt_safety_score", "fcf_safety_score", "low_beta_score"])

    df["pe_score"] = _industry_rank_score(df, "pe_ratio", higher_is_better=False)
    df["forward_pe_score"] = _industry_rank_score(df, "forward_pe", higher_is_better=False)
    df["ev_ebitda_score"] = _industry_rank_score(df, "ev_ebitda", higher_is_better=False)
    df["valuation_reasonableness"] = _mean_score(df, ["pe_score", "forward_pe_score", "ev_ebitda_score"])

    df["max_drawdown_score"] = _industry_rank_score(df, "max_drawdown", higher_is_better=True)
    df["volatility_score"] = _industry_rank_score(df, "volatility", higher_is_better=False)
    df["ma200_distance_pct"] = np.where((df["price"] > 0) & (df["ma200"] > 0), (df["price"] / df["ma200"] - 1.0) * 100.0, np.nan)
    df["trend_score"] = _industry_rank_score(df, "ma200_distance_pct", higher_is_better=True)
    df["technical_downside_risk"] = _mean_score(df, ["max_drawdown_score", "volatility_score", "low_beta_score", "trend_score"])

    df["total_score"] = _weighted_score(df, SCORE_WEIGHTS).round(0)
    df["risk_level"] = df.apply(_risk_level, axis=1)
    df["upside_factors"] = df.apply(_upside_factors, axis=1)
    df["downside_risks"] = df.apply(_downside_risks, axis=1)
    df["valuation_comment"] = df.apply(_valuation_comment, axis=1)
    df["overall_rank"] = df["total_score"].rank(method="first", ascending=False, na_option="bottom").astype("Int64")
    df["industry_rank"] = (
        df.groupby("industry")["total_score"]
        .rank(method="first", ascending=False, na_option="bottom")
        .astype("Int64")
    )

    industry = _build_industry_rankings(df)
    return (
        df.sort_values(["overall_rank", "ticker"], na_position="last").reset_index(drop=True),
        industry,
    )


def _risk_level(row: pd.Series) -> str:
    score = row.get("total_score")
    technical = row.get("technical_downside_risk")
    safety = row.get("balance_sheet_safety")
    if pd.isna(score):
        return "Unknown"
    if score >= 75 and technical >= 60 and safety >= 55:
        return "Low"
    if score >= 55 and technical >= 45:
        return "Medium"
    return "High"


def _upside_factors(row: pd.Series) -> str:
    notes: list[str] = []
    if row.get("growth_quality", 0) >= 70:
        notes.append("above-peer growth")
    if row.get("profitability_quality", 0) >= 70:
        notes.append("strong profitability")
    if row.get("valuation_reasonableness", 0) >= 70:
        notes.append("reasonable valuation")
    if row.get("price", 0) > row.get("ma200", np.inf):
        notes.append("price above 200-day average")
    return ", ".join(notes).capitalize() if notes else "No clear upside edge"


def _downside_risks(row: pd.Series) -> str:
    risks: list[str] = []
    if row.get("balance_sheet_safety", 100) < 40:
        risks.append("balance-sheet risk")
    if row.get("valuation_reasonableness", 100) < 35:
        risks.append("expensive versus peers")
    if row.get("technical_downside_risk", 100) < 40:
        risks.append("weak technical protection")
    if row.get("max_drawdown", 0) < -35:
        risks.append("large historical drawdown")
    if row.get("volatility", 0) > 45:
        risks.append("high volatility")
    return ", ".join(risks).capitalize() if risks else "No major peer-relative risk flags"


def _valuation_comment(row: pd.Series) -> str:
    score = row.get("valuation_reasonableness")
    pe = row.get("pe_ratio")
    fpe = row.get("forward_pe")
    if pd.isna(score):
        return "Valuation data is limited"
    multiples: list[str] = []
    if pd.notna(pe):
        multiples.append(f"PE {pe:.1f}x")
    if pd.notna(fpe):
        multiples.append(f"Forward PE {fpe:.1f}x")
    prefix = ", ".join(multiples) if multiples else "Multiple data is limited"
    if score >= 70:
        return f"{prefix}; attractive versus industry peers"
    if score >= 45:
        return f"{prefix}; fair versus industry peers"
    return f"{prefix}; expensive versus industry peers"


def _build_industry_rankings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for industry, grp in df.groupby("industry", dropna=False):
        grp = grp.sort_values("total_score", ascending=False, na_position="last")
        sector = grp["sector"].mode(dropna=True)
        quality = _mean_score(grp, ["growth_quality", "profitability_quality", "balance_sheet_safety"]).mean()
        risk = grp["technical_downside_risk"].mean()
        upside = _mean_score(grp, ["growth_quality", "technical_downside_risk"]).mean()
        valuation = grp["valuation_reasonableness"].mean()
        industry_score = (
            grp["total_score"].mean() * 0.60
            + quality * 0.15
            + risk * 0.10
            + upside * 0.10
            + valuation * 0.05
        )
        rows.append(
            {
                "industry": str(industry or "Unknown"),
                "sector": str(sector.iloc[0]) if not sector.empty else "Unknown",
                "industry_score": round(float(industry_score), 0) if pd.notna(industry_score) else np.nan,
                "quality_score": round(float(quality), 0) if pd.notna(quality) else np.nan,
                "risk_score": round(float(risk), 0) if pd.notna(risk) else np.nan,
                "upside_score": round(float(upside), 0) if pd.notna(upside) else np.nan,
                "valuation_score": round(float(valuation), 0) if pd.notna(valuation) else np.nan,
                "stock_count": int(len(grp)),
                "top_stocks": ", ".join(grp["ticker"].head(5).tolist()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("industry_score", ascending=False, na_position="last").reset_index(drop=True)


def write_screening_tables(db_path: Path | None = None, inputs: ScreenerInputs | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    db = db_path or default_db_path()
    stock_df, industry_df = compute_industry_screen(db, inputs)
    if stock_df.empty:
        return stock_df, industry_df

    from scripts.init_db import init_db

    init_db(db)
    now = _now()
    score_date = datetime.utcnow().date().isoformat()
    with sqlite3.connect(str(db)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        companies = stock_df[
            ["ticker", "company_name", "sector", "industry", "market_cap", "universe"]
        ].copy()
        companies["updated_at"] = now
        _replace_rows(conn, "companies", companies)

        financial_cols = [
            "ticker",
            "as_of_date",
            "revenue_growth",
            "eps_growth",
            "gross_margin",
            "operating_margin",
            "free_cash_flow",
            "debt_to_equity",
            "roe",
            "roic",
            "pe_ratio",
            "forward_pe",
            "ev_ebitda",
        ]
        financials = stock_df[[col for col in financial_cols if col in stock_df.columns]].copy()
        financials["updated_at"] = now
        _replace_rows(conn, "financials", financials)

        price_cols = [
            "ticker",
            "price_date",
            "price",
            "high_52w",
            "low_52w",
            "ma50",
            "ma200",
            "beta",
            "max_drawdown",
            "volatility",
        ]
        prices = stock_df[[col for col in price_cols if col in stock_df.columns]].copy()
        prices["updated_at"] = now
        _replace_rows(conn, "prices", prices)

        score_cols = [
            "ticker",
            "total_score",
            "growth_quality",
            "profitability_quality",
            "balance_sheet_safety",
            "valuation_reasonableness",
            "technical_downside_risk",
            "upside_factors",
            "downside_risks",
            "valuation_comment",
            "risk_level",
            "industry_rank",
            "overall_rank",
        ]
        scores = stock_df[[col for col in score_cols if col in stock_df.columns]].copy()
        scores.insert(1, "score_date", score_date)
        scores["updated_at"] = now
        _replace_rows(conn, "scores", scores)

        industries = industry_df.copy()
        industries.insert(2, "score_date", score_date)
        industries["updated_at"] = now
        _replace_rows(conn, "industry_rankings", industries)
        conn.commit()

    return stock_df, industry_df


def _replace_rows(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    clean = df.replace({np.nan: None}).copy()
    columns = [str(col) for col in clean.columns]
    placeholders = ",".join("?" * len(columns))
    column_sql = ", ".join(columns)
    conn.execute(f"DELETE FROM {table}")
    if clean.empty:
        return
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({column_sql}) VALUES ({placeholders})",
        clean.itertuples(index=False, name=None),
    )


def export_screening_results(
    db_path: Path | None = None,
    output_dir: Path | None = None,
    inputs: ScreenerInputs | None = None,
) -> dict[str, Path]:
    db = db_path or default_db_path()
    out_dir = output_dir or db.parents[0] / "industry_screen_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stock_df, industry_df = write_screening_tables(db, inputs)
    if stock_df.empty:
        return {}
    stock_path = out_dir / "stock_scores.csv"
    industry_path = out_dir / "industry_rankings.csv"
    top5_path = out_dir / "industry_top5_stocks.csv"
    html_path = out_dir / "industry_dashboard.html"

    stock_df.to_csv(stock_path, index=False)
    industry_df.to_csv(industry_path, index=False)
    top5 = (
        stock_df.sort_values(["industry", "industry_rank"])
        .groupby("industry", group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )
    top5.to_csv(top5_path, index=False)
    html_path.write_text(_build_html_dashboard(stock_df, industry_df), encoding="utf-8")
    return {
        "stock_scores": stock_path,
        "industry_rankings": industry_path,
        "industry_top5_stocks": top5_path,
        "html_dashboard": html_path,
    }


def _build_html_dashboard(stock_df: pd.DataFrame, industry_df: pd.DataFrame) -> str:
    industries = industry_df.head(50).copy()
    top_stocks = stock_df[stock_df["industry_rank"] <= 5].copy()
    cols = [
        "industry",
        "industry_rank",
        "ticker",
        "company_name",
        "total_score",
        "upside_factors",
        "downside_risks",
        "valuation_comment",
        "risk_level",
    ]
    top_stocks = top_stocks[[col for col in cols if col in top_stocks.columns]].sort_values(["industry", "industry_rank"])
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Industry Ranking Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #17202a; background: #f7f8fa; }}
    header {{ padding: 28px 36px; background: #102030; color: white; }}
    main {{ padding: 28px 36px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 30px 0 12px; font-size: 20px; }}
    .meta {{ color: #cbd5df; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .metric {{ background: white; border: 1px solid #e4e7eb; border-radius: 8px; padding: 16px; }}
    .metric strong {{ display: block; font-size: 26px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #e4e7eb; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #edf0f2; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f5; font-weight: 650; }}
    tr:hover {{ background: #f8fbff; }}
  </style>
</head>
<body>
  <header>
    <h1>Industry Ranking Dashboard</h1>
    <div class="meta">Generated {generated_at}</div>
  </header>
  <main>
    <section class="grid">
      <div class="metric"><span>Stocks Ranked</span><strong>{len(stock_df):,}</strong></div>
      <div class="metric"><span>Industries Ranked</span><strong>{len(industry_df):,}</strong></div>
      <div class="metric"><span>Average Stock Score</span><strong>{stock_df["total_score"].mean():.0f}</strong></div>
      <div class="metric"><span>Low Risk Stocks</span><strong>{int((stock_df["risk_level"] == "Low").sum()):,}</strong></div>
    </section>
    <h2>Industry Rankings</h2>
    {industries.to_html(index=False, escape=True)}
    <h2>Top 5 Stocks by Industry</h2>
    {top_stocks.to_html(index=False, escape=True)}
  </main>
</body>
</html>
"""
