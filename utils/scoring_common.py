"""Shared data-loading helpers for score calculations."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from utils.local_data import connect_readonly


# Calendar lookback that reliably covers one trading year plus weekends and holidays.
PRICE_LOOKBACK_DAYS = 430


def available_watchlists(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    try:
        with connect_readonly(db_path, timeout=15) as conn:
            df = pd.read_sql_query(
                "SELECT DISTINCT watchlist_name FROM fmp_watchlist ORDER BY watchlist_name",
                conn,
            )
    except Exception:
        return []
    return [str(x) for x in df["watchlist_name"].dropna().tolist()]


def load_watchlist(conn: sqlite3.Connection, watchlist_name: str | None) -> pd.DataFrame:
    if watchlist_name:
        return pd.read_sql_query(
            """
            SELECT DISTINCT ticker
            FROM fmp_watchlist
            WHERE watchlist_name = ?
            ORDER BY ticker
            """,
            conn,
            params=(watchlist_name,),
        )
    return pd.read_sql_query("SELECT DISTINCT ticker FROM fmp_watchlist ORDER BY ticker", conn)


def load_profiles(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT ticker, company_name, sector, industry
        FROM company_profile
        """,
        conn,
    ).drop_duplicates("ticker")


def load_prices(conn: sqlite3.Connection, tickers: list[str], min_price_rows: int) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    latest = conn.execute("SELECT MAX(price_date) FROM prices_eod").fetchone()[0]
    if not latest:
        return pd.DataFrame()
    latest_dt = datetime.strptime(latest, "%Y-%m-%d")
    start = (latest_dt - timedelta(days=PRICE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    placeholders = ",".join("?" * len(tickers))
    df = pd.read_sql_query(
        f"""
        SELECT ticker, price_date, COALESCE(adj_close, close) AS price
        FROM prices_eod
        WHERE ticker IN ({placeholders})
          AND price_date >= ?
        ORDER BY ticker, price_date
        """,
        conn,
        params=tickers + [start],
    )
    if df.empty:
        return df
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    counts = df.groupby("ticker")["price"].count()
    valid_tickers = counts[counts >= min_price_rows].index
    return df[df["ticker"].isin(valid_tickers)].copy()
