"""Local SQLite data access helpers.

Application pages should prefer these helpers before reaching out to Yahoo,
FMP, SEC, or other external sources.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_db_path() -> Path:
    return project_root() / "data" / "quant_data.db"


def connect_readonly(db_path: Path | None = None, *, timeout: int = 30) -> sqlite3.Connection:
    db = db_path or default_db_path()
    quoted_path = quote(str(db.resolve()), safe="/:")
    conn = sqlite3.connect(
        f"file:{quoted_path}?mode=ro&immutable=1",
        timeout=timeout,
        uri=True,
    )
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def load_price_history(
    tickers: list[str],
    start: date,
    end: date,
    *,
    db_path: Path | None = None,
    min_rows: int = 1,
) -> pd.DataFrame:
    symbols = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not symbols:
        return pd.DataFrame()
    db = db_path or default_db_path()
    if not db.exists():
        return pd.DataFrame()

    placeholders = ",".join("?" * len(symbols))
    with connect_readonly(db) as conn:
        if not table_exists(conn, "prices_eod"):
            return pd.DataFrame()
        df = pd.read_sql_query(
            f"""
            SELECT ticker, price_date, open, high, low, close, adj_close, volume
            FROM prices_eod
            WHERE ticker IN ({placeholders})
              AND price_date >= ?
              AND price_date <= ?
            ORDER BY ticker, price_date
            """,
            conn,
            params=symbols + [start.isoformat(), end.isoformat()],
        )

    if df.empty:
        return df
    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")
    df = df.dropna(subset=["price_date"]).copy()
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    counts = df.groupby("ticker")["close"].count()
    valid = counts[counts >= min_rows].index
    return df[df["ticker"].isin(valid)].copy()


def load_adjusted_close(
    tickers: list[str],
    start: date,
    end: date,
    *,
    db_path: Path | None = None,
    min_rows: int = 1,
) -> pd.DataFrame:
    df = load_price_history(tickers, start, end, db_path=db_path, min_rows=min_rows)
    if df.empty:
        return pd.DataFrame()
    df["price"] = df["adj_close"].fillna(df["close"])
    prices = df.pivot(index="price_date", columns="ticker", values="price")
    prices = prices.sort_index().dropna(how="all").ffill().dropna(how="all")
    ordered = [t for t in [s.upper() for s in tickers] if t in prices.columns]
    return prices[ordered].astype(float) if ordered else pd.DataFrame()


def load_ohlcv(
    ticker: str,
    start: date,
    end: date,
    *,
    db_path: Path | None = None,
    min_rows: int = 1,
) -> pd.DataFrame:
    sym = ticker.strip().upper()
    df = load_price_history([sym], start, end, db_path=db_path, min_rows=min_rows)
    if df.empty:
        return pd.DataFrame()
    df = df[df["ticker"] == sym].sort_values("price_date").copy()
    df = df.set_index("price_date")
    out = pd.DataFrame(
        {
            "Open": df["open"],
            "High": df["high"],
            "Low": df["low"],
            "Close": df["adj_close"].fillna(df["close"]),
            "Volume": df["volume"],
        }
    )
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def latest_quotes(
    tickers: list[str],
    *,
    db_path: Path | None = None,
    lookback_days: int = 10,
) -> dict[str, dict[str, Any]]:
    end = date.today()
    start = end - timedelta(days=lookback_days)
    df = load_adjusted_close(tickers, start, end, db_path=db_path, min_rows=1)
    out: dict[str, dict[str, Any]] = {}
    for sym in [t.upper() for t in tickers]:
        if sym not in df.columns:
            continue
        s = df[sym].dropna()
        if s.empty:
            continue
        price = float(s.iloc[-1])
        prev = float(s.iloc[-2]) if len(s) >= 2 else price
        change = price - prev
        pct = change / prev * 100.0 if prev else 0.0
        out[sym] = {"price": price, "change": change, "pct": pct}
    return out


def latest_company_snapshot(ticker: str, *, db_path: Path | None = None) -> dict[str, Any]:
    sym = ticker.strip().upper()
    db = db_path or default_db_path()
    if not sym or not db.exists():
        return {}
    out: dict[str, Any] = {}
    with connect_readonly(db) as conn:
        if table_exists(conn, "company_profile"):
            row = conn.execute(
                """
                SELECT company_name, sector, industry, full_time_employees
                FROM company_profile
                WHERE ticker = ?
                """,
                (sym,),
            ).fetchone()
            if row:
                out.update(
                    {
                        "longName": row[0],
                        "sector": row[1],
                        "industry": row[2],
                        "fullTimeEmployees": row[3],
                    }
                )
        if table_exists(conn, "key_metrics_ttm"):
            row = conn.execute(
                """
                SELECT market_cap, enterprise_value, ev_to_sales, net_debt_to_ebitda
                FROM key_metrics_ttm
                WHERE ticker = ?
                ORDER BY as_of_date DESC
                LIMIT 1
                """,
                (sym,),
            ).fetchone()
            if row:
                out.update(
                    {
                        "marketCap": row[0],
                        "enterpriseValue": row[1],
                        "enterpriseToRevenue": row[2],
                        "netDebtToEbitda": row[3],
                    }
                )
    quote = latest_quotes([sym], db_path=db).get(sym)
    if quote and quote.get("price") is not None:
        out["currentPrice"] = quote["price"]
        out["regularMarketPrice"] = quote["price"]
    return out

