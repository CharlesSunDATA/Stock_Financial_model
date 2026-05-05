"""Stock Risk Score calculation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.local_data import connect_readonly
from utils.scoring_common import available_watchlists, load_prices, load_profiles, load_watchlist


RISK_WEIGHTS: dict[str, float] = {
    "cash_flow_risk": 1 / 6,
    "debt_risk": 1 / 6,
    "valuation_risk": 1 / 6,
    "technical_risk": 1 / 6,
    "fundamental_risk": 1 / 6,
    "systemic_risk": 1 / 6,
}


RISK_LABELS: dict[str, str] = {
    "cash_flow_risk": "Cash Flow Risk",
    "debt_risk": "Debt Risk",
    "valuation_risk": "Valuation Risk",
    "technical_risk": "Technical Risk",
    "fundamental_risk": "Fundamental Risk",
    "systemic_risk": "Systemic Risk",
}


@dataclass(frozen=True)
class RiskInputs:
    watchlist_name: str | None = None
    min_price_rows: int = 200


def _rank_score(series: pd.Series, *, higher_is_riskier: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.notna().sum() <= 1:
        return pd.Series(np.where(values.notna(), 50.0, np.nan), index=series.index)
    ranked = values.rank(pct=True, ascending=higher_is_riskier, na_option="keep") * 100.0
    return ranked.clip(0, 100)


def _price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for ticker, grp in price_df.groupby("ticker"):
        grp = grp.dropna(subset=["price"]).sort_values("price_date")
        if grp.empty:
            continue
        latest = grp.iloc[-1]
        latest_price = float(latest["price"])
        ma200 = grp["price"].tail(200).mean() if len(grp) >= 200 else np.nan
        ret_6m = _period_return(grp, latest_price, str(latest["price_date"]), 182)
        rows.append(
            {
                "ticker": ticker,
                "latest_price": latest_price,
                "price_date": latest["price_date"],
                "ma200": ma200,
                "below_200ma": bool(pd.notna(ma200) and latest_price < float(ma200)),
                "ret_6m": ret_6m,
            }
        )
    return pd.DataFrame(rows)


def _period_return(grp: pd.DataFrame, latest_price: float, latest_date: str, days: int) -> float | None:
    latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
    target = (latest_dt - timedelta(days=days)).strftime("%Y-%m-%d")
    past = grp[grp["price_date"] <= target]
    if past.empty:
        return None
    base = past.iloc[-1]["price"]
    if pd.isna(base) or not base:
        return None
    return (latest_price - float(base)) / float(base) * 100.0


def _load_fundamentals(conn: sqlite3.Connection, tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(tickers))
    return pd.read_sql_query(
        f"""
        SELECT ticker, report_date, eps, free_cash_flow
        FROM fundamental_data
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, report_date
        """,
        conn,
        params=tickers,
    )


def _fundamental_features(fund_df: pd.DataFrame) -> pd.DataFrame:
    if fund_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for ticker, grp in fund_df.groupby("ticker"):
        grp = grp.sort_values("report_date").copy()
        grp["eps"] = pd.to_numeric(grp["eps"], errors="coerce")
        grp["free_cash_flow"] = pd.to_numeric(grp["free_cash_flow"], errors="coerce")
        latest = grp.iloc[-1]
        eps = grp["eps"].dropna().tail(3).tolist()
        latest_eps = eps[-1] if eps else np.nan
        prev_eps = eps[-2] if len(eps) >= 2 else np.nan
        eps_decline_streak = len(eps) >= 3 and eps[-1] < eps[-2] < eps[-3]
        eps_latest_decline = len(eps) >= 2 and eps[-1] < eps[-2]
        rows.append(
            {
                "ticker": ticker,
                "latest_report_date": latest["report_date"],
                "latest_fcf": latest["free_cash_flow"],
                "latest_eps": latest_eps,
                "previous_eps": prev_eps,
                "eps_decline_streak": eps_decline_streak,
                "eps_latest_decline": eps_latest_decline,
            }
        )
    return pd.DataFrame(rows)


def _load_key_metrics(conn: sqlite3.Connection, tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(tickers))
    df = pd.read_sql_query(
        f"""
        SELECT ticker, as_of_date, market_cap, ev_to_sales, net_debt_to_ebitda
        FROM key_metrics_ttm
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, as_of_date
        """,
        conn,
        params=tickers,
    )
    if df.empty:
        return df
    return df.sort_values("as_of_date").drop_duplicates("ticker", keep="last").copy()


def compute_risk_scores(db_path: Path, inputs: RiskInputs | None = None) -> pd.DataFrame:
    inputs = inputs or RiskInputs()
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

    df = watch.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for part in (profiles, prices, fundamentals, key_metrics):
        if not part.empty:
            df = df.merge(part, on="ticker", how="left")

    if df.empty:
        return df

    df["cash_flow_risk"] = np.where(
        pd.to_numeric(df.get("latest_fcf", pd.Series(index=df.index)), errors="coerce") < 0,
        100.0,
        0.0,
    )

    leverage = pd.to_numeric(df.get("net_debt_to_ebitda", pd.Series(index=df.index)), errors="coerce")
    df["debt_risk"] = ((leverage - 2.0) / 3.0 * 100.0).clip(0, 100)

    ev_to_sales = pd.to_numeric(df.get("ev_to_sales", pd.Series(index=df.index)), errors="coerce")
    df["valuation_risk"] = _rank_score(ev_to_sales.where(ev_to_sales > 0), higher_is_riskier=True)

    df["technical_risk"] = np.where(df.get("below_200ma", pd.Series(False, index=df.index)).fillna(False), 100.0, 0.0)

    eps_decline_streak = df.get("eps_decline_streak", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    eps_latest_decline = df.get("eps_latest_decline", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    df["fundamental_risk"] = np.select(
        [eps_decline_streak.to_numpy(), eps_latest_decline.to_numpy()],
        [100.0, 50.0],
        default=0.0,
    )

    sector_momentum = df.groupby("sector", dropna=False)["ret_6m"].mean()
    sector_strength = sector_momentum.rank(pct=True, ascending=True).mul(100)
    strength = df["sector"].map(sector_strength)
    df["systemic_risk"] = (100.0 - strength).clip(0, 100)

    weighted = pd.Series(0.0, index=df.index)
    weight_seen = pd.Series(0.0, index=df.index)
    for factor, weight in RISK_WEIGHTS.items():
        vals = pd.to_numeric(df[factor], errors="coerce")
        weighted += vals.fillna(0) * weight
        weight_seen += np.where(vals.notna(), weight, 0.0)
    df["risk_score"] = np.where(weight_seen > 0, weighted / weight_seen, np.nan)
    df["risk_score"] = df["risk_score"].round(0)
    df["risk_level"] = df["risk_score"].apply(classify_risk_level)
    df["risk_drivers"] = df.apply(_risk_drivers, axis=1)

    return df.sort_values("risk_score", ascending=False, na_position="last").reset_index(drop=True)


def classify_risk_level(score: Any) -> str:
    if pd.isna(score):
        return "Unknown"
    value = float(score)
    if value <= 30:
        return "Low Risk"
    if value <= 60:
        return "Medium Risk"
    return "High Risk"


def _risk_drivers(row: pd.Series) -> str:
    drivers: list[str] = []
    if row.get("cash_flow_risk", 0) >= 75:
        drivers.append("Negative FCF")
    if row.get("debt_risk", 0) >= 50:
        drivers.append("High leverage")
    if row.get("valuation_risk", 0) >= 75:
        drivers.append("High EV/Sales")
    if row.get("technical_risk", 0) >= 75:
        drivers.append("Below 200MA")
    if row.get("fundamental_risk", 0) >= 75:
        drivers.append("EPS decline streak")
    if row.get("systemic_risk", 0) >= 75:
        drivers.append("Weak sector")
    return ", ".join(drivers) if drivers else "No major risk flags"
