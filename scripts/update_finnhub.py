#!/usr/bin/env python3
"""
Fetch Finnhub enrichment data into the local SQLite database.

Finnhub is used as a supplemental source for company news and earnings
surprises. It does not replace FMP as the primary fundamentals source.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

try:
    from scripts.init_db import init_db
except ModuleNotFoundError:  # pragma: no cover
    from init_db import init_db  # type: ignore


FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
DEFAULT_CALLS_PER_MINUTE = 55


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_secret_value(*keys: str) -> str:
    candidates = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            try:
                import tomllib
            except ImportError:  # pragma: no cover
                import tomli as tomllib  # type: ignore

            data = tomllib.loads(path.read_text())
            for key in keys:
                value = str(data.get(key, "") or "").strip()
                if value:
                    return value
        except Exception:
            for line in path.read_text().splitlines():
                stripped = line.strip()
                for key in keys:
                    if stripped.startswith(key):
                        parts = stripped.split("=", 1)
                        if len(parts) == 2:
                            value = parts[1].strip().strip('"').strip("'")
                            if value:
                                return value
    return ""


def get_finnhub_key() -> str:
    for env_key in ("FINNHUB_API_KEY", "FINNHUB_KEY"):
        value = os.getenv(env_key, "").strip()
        if value:
            return value
    return _read_secret_value("FINNHUB_API_KEY", "FINNHUB_KEY")


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _news_timestamp_to_iso(value: Any) -> str:
    try:
        ts = int(value)
        return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return ""


class FinnhubClient:
    def __init__(self, api_key: str, *, calls_per_minute: int = DEFAULT_CALLS_PER_MINUTE) -> None:
        self.api_key = api_key
        self.calls_per_minute = max(1, int(calls_per_minute))
        self._last_call = 0.0

    def _wait(self) -> None:
        min_interval = 60.0 / self.calls_per_minute
        elapsed = time.monotonic() - self._last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call = time.monotonic()

    def get(self, path: str, params: dict[str, Any]) -> Any:
        self._wait()
        payload = dict(params)
        payload["token"] = self.api_key
        resp = requests.get(f"{FINNHUB_BASE_URL}{path}", params=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data.get("error")))
        return data

    def company_news(self, *, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
        data = self.get(
            "/company-news",
            {
                "symbol": symbol,
                "from": start.isoformat(),
                "to": end.isoformat(),
            },
        )
        return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []

    def earnings_surprises(self, *, symbol: str) -> list[dict[str, Any]]:
        data = self.get("/stock/earnings", {"symbol": symbol})
        return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def load_watchlist_tickers(conn: sqlite3.Connection, *, watchlist_name: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT ticker
        FROM fmp_watchlist
        WHERE watchlist_name = ?
        ORDER BY ticker
        """,
        (watchlist_name,),
    ).fetchall()
    return [str(r[0]).strip().upper() for r in rows if r and str(r[0]).strip()]


def upsert_company_news(conn: sqlite3.Connection, *, ticker: str, rows: list[dict[str, Any]]) -> int:
    now = _utc_now_iso()
    n_rows = 0
    for row in rows:
        published = _news_timestamp_to_iso(row.get("datetime"))
        title = str(row.get("headline") or "").strip()
        url = str(row.get("url") or "").strip()
        if not published or not title:
            continue
        payload = dict(row)
        payload["provider"] = "finnhub"
        conn.execute(
            """
            INSERT OR REPLACE INTO stock_news
              (ticker, published_date, publisher, title, site, text,
               url, image_url, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                published,
                row.get("source"),
                title,
                "finnhub",
                row.get("summary"),
                url or None,
                row.get("image"),
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n_rows += 1
    return n_rows


def upsert_earnings_surprises(conn: sqlite3.Connection, *, ticker: str, rows: list[dict[str, Any]]) -> int:
    now = _utc_now_iso()
    n_rows = 0
    for row in rows:
        surprise_date = str(row.get("period") or "").strip()[:10]
        if not surprise_date:
            continue
        payload = dict(row)
        payload["provider"] = "finnhub"
        conn.execute(
            """
            INSERT OR REPLACE INTO earnings_surprises
              (ticker, surprise_date, actual_eps, estimated_eps, surprise_percent,
               payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                surprise_date,
                _safe_float(row.get("actual")),
                _safe_float(row.get("estimate")),
                _safe_float(row.get("surprisePercent")),
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n_rows += 1
    return n_rows


def run(
    *,
    tickers: list[str] | None,
    watchlist_name: str,
    days: int,
    batch_size: int,
    include_news: bool,
    include_earnings: bool,
    calls_per_minute: int,
    api_key: str = "",
    required: bool = False,
) -> None:
    key = api_key.strip() or get_finnhub_key()
    if not key:
        message = "FINNHUB_API_KEY is not set; Finnhub enrichment skipped."
        if required:
            raise ValueError(message)
        print(message)
        return

    db_path = init_db()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")

    if not tickers:
        tickers = load_watchlist_tickers(conn, watchlist_name=watchlist_name)
    tickers = sorted({t.strip().upper() for t in tickers if t.strip()})
    if batch_size > 0:
        tickers = tickers[:batch_size]
    if not tickers:
        print(f"No tickers found for watchlist='{watchlist_name}'.")
        conn.close()
        return

    end = date.today()
    start = end - timedelta(days=max(1, int(days)))
    client = FinnhubClient(key, calls_per_minute=calls_per_minute)

    print(f"DB: {db_path}")
    print(f"Finnhub tickers: {len(tickers):,}")
    print(f"Company news range: {start.isoformat()}->{end.isoformat()}")
    print(f"include_news={include_news}, include_earnings={include_earnings}")

    total_news = 0
    total_earnings = 0
    for idx, ticker in enumerate(tickers, start=1):
        try:
            news_count = 0
            earnings_count = 0
            if include_news:
                news_count = upsert_company_news(
                    conn,
                    ticker=ticker,
                    rows=client.company_news(symbol=ticker, start=start, end=end),
                )
            if include_earnings:
                earnings_count = upsert_earnings_surprises(
                    conn,
                    ticker=ticker,
                    rows=client.earnings_surprises(symbol=ticker),
                )
            conn.commit()
            total_news += news_count
            total_earnings += earnings_count
            print(f"[{idx}/{len(tickers)}] {ticker}: news={news_count:,}, earnings={earnings_count:,}", flush=True)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            print(f"[{idx}/{len(tickers)}] {ticker}: Finnhub failed ({type(exc).__name__}): {exc}", flush=True)

    conn.close()
    print(f"Done. Finnhub news rows={total_news:,}; earnings rows={total_earnings:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Finnhub enrichment data into local SQLite tables.")
    parser.add_argument("tickers", nargs="*", help="Explicit tickers. Omit to use --watchlist.")
    parser.add_argument("--watchlist", default="default")
    parser.add_argument("--days", type=int, default=14, help="Lookback window for company news.")
    parser.add_argument("--batch-size", type=int, default=0, help="0 means all tickers.")
    parser.add_argument("--calls-per-minute", type=int, default=DEFAULT_CALLS_PER_MINUTE)
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-earnings", action="store_true")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--required", action="store_true", help="Fail if FINNHUB_API_KEY is missing.")
    args = parser.parse_args()

    run(
        tickers=[str(t).upper() for t in args.tickers] if args.tickers else None,
        watchlist_name=str(args.watchlist),
        days=max(1, int(args.days)),
        batch_size=max(0, int(args.batch_size)),
        include_news=not bool(args.skip_news),
        include_earnings=not bool(args.skip_earnings),
        calls_per_minute=max(1, int(args.calls_per_minute)),
        api_key=str(args.api_key or ""),
        required=bool(args.required),
    )


if __name__ == "__main__":
    main()
