"""
update_news_fmp.py

Fetch recent stock news from FMP and store in stock_news table.

Endpoint: GET /stable/news/stock?symbol=&limit=&page=&apikey=
  - One symbol per request; pagination via page=0,1,2,...
  - Returns up to 'limit' articles per page (max 100)
  - ~30-60 days of rolling history depending on ticker

Run:
  python3 scripts/update_news_fmp.py
  python3 scripts/update_news_fmp.py --watchlist default --pages 3
  python3 scripts/update_news_fmp.py AAPL MSFT NVDA
  python3 scripts/update_news_fmp.py --force   # re-fetch regardless of age
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.market_loader_fmp import (
        FMP_API_KEY,
        FMP_BASE,
        FmpClient,
        _utc_now_iso,
        load_watchlist_tickers,
    )
    from scripts.init_db import init_db
except ModuleNotFoundError:
    from market_loader_fmp import (  # type: ignore
        FMP_API_KEY,
        FMP_BASE,
        FmpClient,
        _utc_now_iso,
        load_watchlist_tickers,
    )
    from init_db import init_db  # type: ignore

DEFAULT_PAGES = 2
PAGE_SIZE = 50
MAX_AGE_HOURS_DEFAULT = 24.0


# ── Staleness helper ─────────────────────────────────────────────────────────

def _ticker_latest_news(conn: sqlite3.Connection, ticker: str) -> str | None:
    cur = conn.execute(
        "SELECT MAX(updated_at) FROM stock_news WHERE ticker = ?", (ticker,)
    )
    row = cur.fetchone()
    return str(row[0]) if row and row[0] else None


def _is_fresh(last_updated: str | None, max_age_hours: float) -> bool:
    if not last_updated or max_age_hours <= 0:
        return False
    try:
        t = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (now - t).total_seconds() / 3600.0 <= max_age_hours
    except Exception:
        return False


# ── Fetch / upsert ───────────────────────────────────────────────────────────

def fetch_news_page(client: FmpClient, ticker: str, page: int, limit: int = PAGE_SIZE) -> list[dict]:
    try:
        data = client.get_data(
            f"{FMP_BASE}/news/stock",
            {"symbol": ticker, "limit": limit, "page": page},
        )
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []
    except Exception as e:
        print(f"    WARN news/{ticker} page={page}: {e}")
        return []


def upsert_news(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        pub_date = str(r.get("publishedDate") or "").strip()
        url = str(r.get("url") or "").strip()
        title = str(r.get("title") or "").strip()
        if not pub_date or not title:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO stock_news
              (ticker, published_date, publisher, title, site, text,
               url, image_url, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                pub_date,
                r.get("publisher"),
                title,
                r.get("site"),
                r.get("text"),
                url or None,
                r.get("image"),
                json.dumps({k: v for k, v in r.items() if k != "text"},
                           ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


# ── Orchestration ────────────────────────────────────────────────────────────

def run(
    *,
    tickers: list[str] | None = None,
    watchlist_name: str = "default",
    max_age_hours: float = MAX_AGE_HOURS_DEFAULT,
    force: bool = False,
    pages: int = DEFAULT_PAGES,
    sleep_s: float = 0.12,
    log_every: int = 50,
    api_key: str = "",
) -> None:
    key = api_key.strip() or os.getenv("FMP_API_KEY", "").strip() or FMP_API_KEY
    if not key:
        raise ValueError("FMP_API_KEY not set.")

    db = init_db()
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL;")

    if not tickers:
        tickers = load_watchlist_tickers(conn, watchlist_name=watchlist_name)
    if not tickers:
        print(f"No tickers found (watchlist='{watchlist_name}').")
        conn.close()
        return

    client = FmpClient(key, calls_per_minute=250)

    print(f"Tickers : {len(tickers)}")
    print(f"Pages/ticker: {pages}  (up to {pages * PAGE_SIZE} articles each)")
    print(f"max_age_hours={max_age_hours}, force={force}")
    print()

    grand_total = 0

    for i, ticker in enumerate(tickers, start=1):
        if not force:
            last = _ticker_latest_news(conn, ticker)
            if _is_fresh(last, max_age_hours):
                continue

        inserted = 0
        for page in range(pages):
            rows = fetch_news_page(client, ticker, page)
            if not rows:
                break
            inserted += upsert_news(conn, ticker, rows)
            if sleep_s > 0:
                time.sleep(sleep_s)
            if len(rows) < PAGE_SIZE:
                break

        if inserted:
            conn.commit()
            grand_total += inserted

        if i % log_every == 0 or i == len(tickers):
            print(f"  [{i:,}/{len(tickers):,}] {ticker}  total articles: {grand_total:,}")

    conn.close()
    print(f"\nDone. Total articles inserted: {grand_total:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch stock news from FMP into stock_news table")
    parser.add_argument("tickers", nargs="*", help="Explicit tickers. Omit to use --watchlist.")
    parser.add_argument("--watchlist", default="default")
    parser.add_argument(
        "--max-age-hours", type=float, default=MAX_AGE_HOURS_DEFAULT,
        help="Skip ticker if news was fetched within this many hours (default 24; 0 = always fetch)",
    )
    parser.add_argument("--force", action="store_true", help="Re-fetch regardless of age")
    parser.add_argument(
        "--pages", type=int, default=DEFAULT_PAGES,
        help=f"Pages of news to fetch per ticker (default {DEFAULT_PAGES}; each page = {PAGE_SIZE} articles)",
    )
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    run(
        tickers=[t.upper() for t in args.tickers] if args.tickers else None,
        watchlist_name=args.watchlist,
        max_age_hours=args.max_age_hours,
        force=args.force,
        pages=args.pages,
        sleep_s=args.sleep,
        log_every=args.log_every,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
