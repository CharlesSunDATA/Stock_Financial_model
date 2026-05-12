"""
Fetch recent FMP quarterly financial highlights for watchlist tickers.

The output table feeds the Streamlit single-stock report so the earnings
section can show concrete revenue, EPS, and margin context instead of only
calendar events.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from typing import Any

try:
    from scripts.init_db import init_db
    from scripts.market_loader_fmp import FMP_API_KEY, FMP_BASE, FmpClient, _safe_float, _utc_now_iso, load_watchlist_tickers
except ModuleNotFoundError:
    from init_db import init_db  # type: ignore
    from market_loader_fmp import FMP_API_KEY, FMP_BASE, FmpClient, _safe_float, _utc_now_iso, load_watchlist_tickers  # type: ignore


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _pct_change(current: float | None, prior: float | None) -> float | None:
    if current is None or prior in (None, 0):
        return None
    try:
        return (float(current) / float(prior) - 1.0) * 100.0
    except Exception:
        return None


def fetch_income_statement(client: FmpClient, ticker: str, *, limit: int) -> list[dict[str, Any]]:
    data = client.get_data(
        f"{FMP_BASE}/income-statement",
        {"symbol": ticker, "period": "quarter", "limit": max(2, int(limit))},
    )
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_financial_highlights (
          ticker TEXT NOT NULL,
          report_date TEXT NOT NULL,
          fiscal_year INTEGER,
          fiscal_period TEXT,
          revenue REAL,
          revenue_yoy REAL,
          eps REAL,
          eps_yoy REAL,
          net_income REAL,
          gross_profit_ratio REAL,
          operating_income REAL,
          source TEXT,
          payload_json TEXT,
          updated_at TEXT NOT NULL,
          UNIQUE(ticker, report_date)
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_stock_financial_highlights_ticker_date
        ON stock_financial_highlights(ticker, report_date);
        """
    )


def upsert_highlights(conn: sqlite3.Connection, ticker: str, rows: list[dict[str, Any]]) -> int:
    now = _utc_now_iso()
    sym = ticker.strip().upper()
    rows = sorted(rows, key=lambda row: str(row.get("date") or ""), reverse=True)
    inserted = 0
    for index, row in enumerate(rows):
        report_date = str(row.get("date") or row.get("reportedCurrency") or "").strip()
        if not report_date:
            continue
        prior = rows[index + 4] if index + 4 < len(rows) else None
        revenue = _safe_float(row.get("revenue"))
        eps = _safe_float(row.get("eps") or row.get("epsdiluted"))
        revenue_yoy = _pct_change(revenue, _safe_float(prior.get("revenue")) if prior else None)
        eps_yoy = _pct_change(eps, _safe_float(prior.get("eps") or prior.get("epsdiluted")) if prior else None)
        conn.execute(
            """
            INSERT OR REPLACE INTO stock_financial_highlights
              (ticker, report_date, fiscal_year, fiscal_period, revenue,
               revenue_yoy, eps, eps_yoy, net_income, gross_profit_ratio,
               operating_income, source, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sym,
                report_date,
                _safe_int(row.get("calendarYear") or row.get("fiscalYear")),
                row.get("period"),
                revenue,
                revenue_yoy,
                eps,
                eps_yoy,
                _safe_float(row.get("netIncome")),
                _safe_float(row.get("grossProfitRatio")),
                _safe_float(row.get("operatingIncome")),
                "FMP income-statement",
                json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        inserted += 1
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FMP quarterly financial highlights.")
    parser.add_argument("tickers", nargs="*")
    parser.add_argument("--watchlist-name", default="default")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--calls-per-minute", type=int, default=240)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    key = os.getenv("FMP_API_KEY", "").strip() or FMP_API_KEY
    if not key:
        raise SystemExit("FMP_API_KEY is not set.")

    db_path = init_db()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        ensure_schema(conn)
        tickers = [ticker.upper() for ticker in args.tickers]
        if not tickers:
            tickers = load_watchlist_tickers(conn, watchlist_name=str(args.watchlist_name))

        client = FmpClient(key, calls_per_minute=int(args.calls_per_minute))
        total = 0
        for index, ticker in enumerate(tickers, start=1):
            try:
                rows = fetch_income_statement(client, ticker, limit=int(args.limit))
            except Exception as exc:
                print(f"    WARN FMP financial highlights {ticker}: {exc}")
                continue
            inserted = upsert_highlights(conn, ticker, rows)
            total += inserted
            conn.commit()
            print(f"  [{index}/{len(tickers)}] {ticker}: financial highlights upserted={inserted}")
            if args.sleep > 0:
                time.sleep(float(args.sleep))
        print(f"Total financial highlight rows: {total}")


if __name__ == "__main__":
    main()
