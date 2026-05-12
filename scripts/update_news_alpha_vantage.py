"""
Fetch Alpha Vantage news sentiment for watchlist tickers into stock_news.

This supplements FMP stock news in GitHub Actions. Alpha Vantage has tighter
rate limits, so the workflow can cap the number of tickers per run.
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

import requests

try:
    from scripts.init_db import init_db
    from scripts.market_loader_fmp import _utc_now_iso, load_watchlist_tickers
except ModuleNotFoundError:
    from init_db import init_db  # type: ignore
    from market_loader_fmp import _utc_now_iso, load_watchlist_tickers  # type: ignore


ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


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
            continue
    return ""


def _alpha_key() -> str:
    for env_key in ("ALPHA_VANTAGE_API_KEY", "ALPHAVANTAGE_API_KEY", "ALPHA_VANTAGE_KEY"):
        value = os.getenv(env_key, "").strip()
        if value:
            return value
    return _read_secret_value("ALPHA_VANTAGE_API_KEY", "ALPHAVANTAGE_API_KEY", "ALPHA_VANTAGE_KEY")


def _parse_alpha_time(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        return datetime.strptime(raw[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        return raw


class AlphaNewsClient:
    def __init__(self, api_key: str, *, calls_per_minute: int) -> None:
        self.api_key = api_key
        self.calls_per_minute = max(1, int(calls_per_minute))
        self._last_call = 0.0

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        min_interval = 60.0 / self.calls_per_minute
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call = time.monotonic()

    def fetch(self, ticker: str, *, limit: int) -> list[dict[str, Any]]:
        self._wait()
        resp = requests.get(
            ALPHA_VANTAGE_BASE_URL,
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "sort": "LATEST",
                "limit": max(1, int(limit)),
                "apikey": self.api_key,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "Note" in data or "Information" in data:
            print(f"    WARN Alpha Vantage limit/info for {ticker}: {data.get('Note') or data.get('Information')}")
            return []
        feed = data.get("feed", [])
        return [row for row in feed if isinstance(row, dict)]


def _ticker_relevance(row: dict[str, Any], ticker: str) -> dict[str, Any] | None:
    target = ticker.upper()
    for item in row.get("ticker_sentiment", []) or []:
        if str(item.get("ticker", "")).upper() == target:
            return item
    return None


def upsert_news(conn: sqlite3.Connection, ticker: str, rows: list[dict[str, Any]]) -> int:
    now = _utc_now_iso()
    sym = ticker.strip().upper()
    inserted = 0
    for row in rows:
        title = str(row.get("title") or "").strip()
        url = str(row.get("url") or "").strip()
        published = _parse_alpha_time(str(row.get("time_published") or ""))
        if not title or not url:
            continue
        relevance = _ticker_relevance(row, sym)
        summary = str(row.get("summary") or "").strip()
        payload = {
            "provider": "Alpha Vantage",
            "overall_sentiment_score": row.get("overall_sentiment_score"),
            "overall_sentiment_label": row.get("overall_sentiment_label"),
            "ticker_sentiment": relevance,
            "topics": row.get("topics", []),
        }
        conn.execute(
            """
            INSERT OR REPLACE INTO stock_news
              (ticker, published_date, publisher, title, site, text,
               url, image_url, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sym,
                published,
                row.get("source"),
                title,
                "Alpha Vantage",
                summary,
                url,
                row.get("banner_image"),
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        inserted += 1
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Alpha Vantage news sentiment into stock_news.")
    parser.add_argument("tickers", nargs="*")
    parser.add_argument("--watchlist-name", default="default")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--limit-tickers", type=int, default=25)
    parser.add_argument("--calls-per-minute", type=int, default=5)
    args = parser.parse_args()

    api_key = _alpha_key()
    if not api_key:
        raise SystemExit("ALPHA_VANTAGE_API_KEY is not set.")

    db_path = init_db()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        tickers = [ticker.upper() for ticker in args.tickers]
        if not tickers:
            tickers = load_watchlist_tickers(conn, watchlist_name=str(args.watchlist_name))
        if args.limit_tickers > 0:
            tickers = tickers[: int(args.limit_tickers)]

        client = AlphaNewsClient(api_key, calls_per_minute=int(args.calls_per_minute))
        total = 0
        for index, ticker in enumerate(tickers, start=1):
            try:
                rows = client.fetch(ticker, limit=int(args.limit))
            except Exception as exc:
                print(f"    WARN Alpha Vantage news {ticker}: {exc}")
                continue
            inserted = upsert_news(conn, ticker, rows)
            total += inserted
            conn.commit()
            print(f"  [{index}/{len(tickers)}] {ticker}: alpha news upserted={inserted}")
        print(f"Total Alpha Vantage news rows: {total}")


if __name__ == "__main__":
    main()
