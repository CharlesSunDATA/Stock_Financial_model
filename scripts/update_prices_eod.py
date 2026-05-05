#!/usr/bin/env python3
"""
Update local prices_eod for market trend workflows.

Default mode uses FMP's daily EOD bulk endpoint, which is the fastest way to
keep breadth and moving-average dashboards current after the US close.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

try:
    from scripts.market_loader_fmp import (  # type: ignore
        FMP_API_KEY,
        FmpClient,
        _db_path,
        _is_day_done,
        _mark_day_done,
        _parse_iso_date,
        _safe_float,
        _yesterday,
        fetch_eod_bulk,
        fetch_historical_price_eod_full,
        init_db,
        upsert_prices_eod,
    )
except ModuleNotFoundError:  # pragma: no cover
    from market_loader_fmp import (  # type: ignore
        FMP_API_KEY,
        FmpClient,
        _db_path,
        _is_day_done,
        _mark_day_done,
        _parse_iso_date,
        _safe_float,
        _yesterday,
        fetch_eod_bulk,
        fetch_historical_price_eod_full,
        init_db,
        upsert_prices_eod,
    )


def _latest_prices_date(conn: sqlite3.Connection) -> date | None:
    row = conn.execute("SELECT MAX(price_date) FROM prices_eod").fetchone()
    if not row or not row[0]:
        return None
    return datetime.strptime(str(row[0])[:10], "%Y-%m-%d").date()


def _resolve_start(conn: sqlite3.Connection, *, days: int, date_from: str) -> date:
    if date_from:
        return datetime.strptime(date_from[:10], "%Y-%m-%d").date()

    latest = _latest_prices_date(conn)
    if latest is not None:
        return latest + timedelta(days=1)

    return _yesterday() - timedelta(days=max(0, days - 1))


def _is_weekend(day: date) -> bool:
    return day.weekday() >= 5


def _upsert_symbol_rows(conn: sqlite3.Connection, *, ticker: str, rows: list[dict[str, Any]]) -> int:
    n_ins = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = _parse_iso_date(r.get("date"))
        if not d:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO prices_eod
              (ticker, price_date, open, high, low, close, adj_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                d,
                _safe_float(r.get("open")),
                _safe_float(r.get("high")),
                _safe_float(r.get("low")),
                _safe_float(r.get("close")),
                _safe_float(r.get("adjClose")),
                _safe_float(r.get("volume")),
            ),
        )
        n_ins += 1
    return n_ins


def _update_bulk(
    conn: sqlite3.Connection,
    client: FmpClient,
    *,
    start: date,
    end: date,
    skip_completed: bool,
) -> int:
    total = 0
    day = start
    while day <= end:
        if _is_weekend(day):
            print(f"{day.isoformat()}: weekend; skip")
            day += timedelta(days=1)
            continue

        if skip_completed and _is_day_done(conn, "eod", day):
            print(f"{day.isoformat()}: already marked complete; skip")
            day += timedelta(days=1)
            continue

        print(f"{day.isoformat()}: fetching eod-bulk...")
        rows = fetch_eod_bulk(client, day=day)
        print(f"{day.isoformat()}: rows received={len(rows):,}")
        n_rows = upsert_prices_eod(conn, day=day, rows=rows)
        _mark_day_done(conn, "eod", day, len(rows))
        conn.commit()
        total += n_rows
        print(f"{day.isoformat()}: upserted prices_eod={n_rows:,}")
        day += timedelta(days=1)
    return total


def _update_per_symbol(
    conn: sqlite3.Connection,
    client: FmpClient,
    *,
    start: date,
    end: date,
    batch_size: int,
) -> int:
    tickers = [
        str(r[0]).upper()
        for r in conn.execute(
            """
            SELECT ticker
            FROM prices_eod
            GROUP BY ticker
            HAVING MAX(price_date) < ?
            ORDER BY ticker
            """,
            (end.isoformat(),),
        ).fetchall()
        if r and r[0]
    ]
    if batch_size > 0:
        tickers = tickers[:batch_size]

    print(f"per-symbol update: tickers={len(tickers):,}, range={start.isoformat()}->{end.isoformat()}")
    total = 0
    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {ticker}: fetching historical-price-eod/full...")
        rows = fetch_historical_price_eod_full(
            client,
            symbol=ticker,
            date_from=start.isoformat(),
            date_to=end.isoformat(),
        )
        n_rows = _upsert_symbol_rows(conn, ticker=ticker, rows=rows)
        conn.commit()
        total += n_rows
        print(f"[{i}/{len(tickers)}] {ticker}: upserted={n_rows:,}")
        time.sleep(0.05)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Update prices_eod for market trend analysis.")
    ap.add_argument("--mode", choices=["bulk", "per-symbol"], default="bulk")
    ap.add_argument("--days", type=int, default=7, help="Fallback lookback when DB has no prices yet.")
    ap.add_argument("--from", dest="date_from", default="", help="Explicit start date YYYY-MM-DD.")
    ap.add_argument("--to", dest="date_to", default="", help="Explicit end date YYYY-MM-DD. Default: yesterday.")
    ap.add_argument("--calls-per-minute", type=int, default=280)
    ap.add_argument("--batch-size", type=int, default=0, help="Per-symbol mode only. 0 means all lagging tickers.")
    ap.add_argument("--no-skip-completed", action="store_true", help="Bulk mode only.")
    args = ap.parse_args()

    if FMP_API_KEY.strip() in ("", "YOUR_API_KEY_HERE"):
        raise SystemExit("FMP_API_KEY is not set. Configure .streamlit/secrets.toml or environment first.")

    db_path = init_db(_db_path())
    client = FmpClient(FMP_API_KEY, calls_per_minute=int(args.calls_per_minute))

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA journal_mode=WAL;")

        start = _resolve_start(conn, days=max(1, int(args.days)), date_from=str(args.date_from or "").strip())
        end = datetime.strptime(str(args.date_to)[:10], "%Y-%m-%d").date() if args.date_to else _yesterday()
        latest_before = _latest_prices_date(conn)
        print(f"DB: {Path(db_path)}")
        print(f"Latest prices_eod before update: {latest_before.isoformat() if latest_before else 'none'}")

        if start > end:
            print(f"prices_eod already current through {end.isoformat()}; nothing to update.")
            return

        try:
            if args.mode == "bulk":
                total = _update_bulk(
                    conn,
                    client,
                    start=start,
                    end=end,
                    skip_completed=not bool(args.no_skip_completed),
                )
            else:
                total = _update_per_symbol(
                    conn,
                    client,
                    start=start,
                    end=end,
                    batch_size=max(0, int(args.batch_size)),
                )
        except requests.HTTPError as e:
            if args.mode == "bulk":
                print(f"Bulk EOD update failed: {e}")
                print("Your FMP plan may not include eod-bulk. Try: --mode per-symbol")
                raise SystemExit(2) from e
            raise

        latest_after = _latest_prices_date(conn)
        print(f"Total upserted rows: {total:,}")
        print(f"Latest prices_eod after update: {latest_after.isoformat() if latest_after else 'none'}")


if __name__ == "__main__":
    main()
