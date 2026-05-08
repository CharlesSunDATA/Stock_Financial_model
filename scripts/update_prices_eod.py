#!/usr/bin/env python3
"""
Update local prices_eod for market trend workflows.

Default mode uses FMP's daily EOD bulk endpoint, which is the fastest way to
keep breadth and moving-average dashboards current after the US close.
"""

from __future__ import annotations

import argparse
import os
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


def _get_alpha_vantage_key() -> str:
    for env_key in ("ALPHA_VANTAGE_API_KEY", "ALPHAVANTAGE_API_KEY", "ALPHA_VANTAGE_KEY"):
        value = os.getenv(env_key, "").strip()
        if value:
            return value
    return _read_secret_value("ALPHA_VANTAGE_API_KEY", "ALPHAVANTAGE_API_KEY", "ALPHA_VANTAGE_KEY")


class AlphaVantageClient:
    def __init__(self, api_key: str, *, calls_per_minute: int = 5) -> None:
        self.api_key = api_key
        self.calls_per_minute = max(1, int(calls_per_minute))
        self._last_call = 0.0

    def _wait(self) -> None:
        min_interval = 60.0 / self.calls_per_minute
        elapsed = time.monotonic() - self._last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call = time.monotonic()

    def _fetch_daily_payload(self, *, function_name: str, symbol: str) -> dict[str, Any]:
        self._wait()
        resp = requests.get(
            ALPHA_VANTAGE_BASE_URL,
            params={
                "function": function_name,
                "symbol": symbol,
                "outputsize": "compact",
                "apikey": self.api_key,
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            return {}
        return payload

    def fetch_daily_adjusted(self, *, symbol: str, date_from: date, date_to: date) -> list[dict[str, Any]]:
        payload = self._fetch_daily_payload(function_name="TIME_SERIES_DAILY_ADJUSTED", symbol=symbol)
        if payload.get("Error Message"):
            raise ValueError(str(payload.get("Error Message")))
        info = str(payload.get("Information") or "")
        if info and "premium" in info.lower():
            payload = self._fetch_daily_payload(function_name="TIME_SERIES_DAILY", symbol=symbol)
            if payload.get("Error Message"):
                raise ValueError(str(payload.get("Error Message")))
            if payload.get("Note") or payload.get("Information"):
                raise RuntimeError(str(payload.get("Note") or payload.get("Information")))
        elif payload.get("Note") or payload.get("Information"):
            raise RuntimeError(str(payload.get("Note") or payload.get("Information")))

        series = payload.get("Time Series (Daily)", {})
        if not isinstance(series, dict):
            return []

        rows: list[dict[str, Any]] = []
        for day_text, values in series.items():
            day = _parse_iso_date(day_text)
            if day is None or day < date_from or day > date_to or not isinstance(values, dict):
                continue
            rows.append(
                {
                    "date": day.isoformat(),
                    "open": values.get("1. open"),
                    "high": values.get("2. high"),
                    "low": values.get("3. low"),
                    "close": values.get("4. close"),
                    "adjClose": values.get("5. adjusted close") or values.get("4. close"),
                    "volume": values.get("6. volume") or values.get("5. volume"),
                }
            )
        rows.sort(key=lambda r: str(r.get("date", "")))
        return rows


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
    universe: str,
    max_retries: int,
    fallback_source: str,
    alpha_client: AlphaVantageClient | None,
) -> int:
    if universe == "profile":
        query = """
            SELECT p.ticker
            FROM prices_eod p
            JOIN company_profile cp ON cp.ticker = p.ticker
            WHERE cp.company_name IS NOT NULL
              AND cp.sector IS NOT NULL
              AND length(p.ticker) <= 5
              AND p.ticker NOT LIKE '%.%'
              AND p.ticker NOT LIKE '%-%'
            GROUP BY p.ticker
            HAVING MAX(p.price_date) < ?
            ORDER BY p.ticker
        """
        params = (end.isoformat(),)
    elif universe == "watchlist":
        query = """
            SELECT p.ticker
            FROM prices_eod p
            JOIN fmp_watchlist w ON w.ticker = p.ticker
            GROUP BY p.ticker
            HAVING MAX(p.price_date) < ?
            ORDER BY p.ticker
        """
        params = (end.isoformat(),)
    else:
        query = """
            SELECT ticker
            FROM prices_eod
            GROUP BY ticker
            HAVING MAX(price_date) < ?
            ORDER BY ticker
        """
        params = (end.isoformat(),)

    tickers = [str(r[0]).upper() for r in conn.execute(query, params).fetchall() if r and r[0]]
    if batch_size > 0:
        tickers = tickers[:batch_size]

    print(f"per-symbol update: universe={universe}, tickers={len(tickers):,}, range={start.isoformat()}->{end.isoformat()}")
    if fallback_source == "alpha-vantage" and alpha_client is None:
        print("Alpha Vantage fallback requested, but no Alpha Vantage API key was found; fallback is disabled.", flush=True)
    total = 0
    failed: list[str] = []
    for i, ticker in enumerate(tickers, start=1):
        rows: list[dict[str, Any]] = []
        fmp_failed = False
        for attempt in range(1, max(1, max_retries) + 1):
            try:
                print(f"[{i}/{len(tickers)}] {ticker}: fetching historical-price-eod/full... attempt={attempt}", flush=True)
                rows = fetch_historical_price_eod_full(
                    client,
                    symbol=ticker,
                    date_from=start.isoformat(),
                    date_to=end.isoformat(),
                )
                break
            except requests.RequestException as e:
                print(f"[{i}/{len(tickers)}] {ticker}: request failed ({type(e).__name__}): {e}", flush=True)
                if attempt >= max(1, max_retries):
                    fmp_failed = True
                    rows = []
                else:
                    time.sleep(min(10, attempt * 2))

        if fallback_source == "alpha-vantage" and alpha_client is not None:
            fmp_upsertable = any(_parse_iso_date(r.get("date")) for r in rows if isinstance(r, dict))
            if fmp_failed or not fmp_upsertable:
                try:
                    print(f"[{i}/{len(tickers)}] {ticker}: trying Alpha Vantage fallback...", flush=True)
                    rows = alpha_client.fetch_daily_adjusted(symbol=ticker, date_from=start, date_to=end)
                    print(f"[{i}/{len(tickers)}] {ticker}: Alpha Vantage rows={len(rows):,}", flush=True)
                except (requests.RequestException, RuntimeError, ValueError) as e:
                    print(f"[{i}/{len(tickers)}] {ticker}: Alpha Vantage fallback failed ({type(e).__name__}): {e}", flush=True)
                    if fmp_failed:
                        failed.append(ticker)
                    rows = []
        elif fmp_failed:
            failed.append(ticker)

        n_rows = _upsert_symbol_rows(conn, ticker=ticker, rows=rows)
        conn.commit()
        total += n_rows
        print(f"[{i}/{len(tickers)}] {ticker}: upserted={n_rows:,}", flush=True)
        time.sleep(0.05)
    if failed:
        print(f"per-symbol update: failed tickers={len(failed):,}: {', '.join(failed[:50])}", flush=True)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Update prices_eod for market trend analysis.")
    ap.add_argument("--mode", choices=["bulk", "per-symbol"], default="bulk")
    ap.add_argument("--days", type=int, default=7, help="Fallback lookback when DB has no prices yet.")
    ap.add_argument("--from", dest="date_from", default="", help="Explicit start date YYYY-MM-DD.")
    ap.add_argument("--to", dest="date_to", default="", help="Explicit end date YYYY-MM-DD. Default: yesterday.")
    ap.add_argument("--calls-per-minute", type=int, default=280)
    ap.add_argument("--batch-size", type=int, default=0, help="Per-symbol mode only. 0 means all lagging tickers.")
    ap.add_argument("--universe", choices=["all", "profile", "watchlist"], default="all", help="Per-symbol ticker universe.")
    ap.add_argument("--max-retries", type=int, default=3, help="Per-symbol request retries before skipping a ticker.")
    ap.add_argument(
        "--fallback-source",
        choices=["none", "alpha-vantage"],
        default="none",
        help="Per-symbol mode only. Try this source when FMP returns no rows or fails.",
    )
    ap.add_argument("--alpha-calls-per-minute", type=int, default=5, help="Alpha Vantage fallback rate limit.")
    ap.add_argument("--no-skip-completed", action="store_true", help="Bulk mode only.")
    args = ap.parse_args()

    if FMP_API_KEY.strip() in ("", "YOUR_API_KEY_HERE"):
        raise SystemExit("FMP_API_KEY is not set. Configure .streamlit/secrets.toml or environment first.")

    db_path = init_db(_db_path())
    client = FmpClient(FMP_API_KEY, calls_per_minute=int(args.calls_per_minute))
    alpha_client = None
    if str(args.fallback_source) == "alpha-vantage":
        alpha_key = _get_alpha_vantage_key()
        if alpha_key:
            alpha_client = AlphaVantageClient(alpha_key, calls_per_minute=int(args.alpha_calls_per_minute))

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
                    universe=str(args.universe),
                    max_retries=max(1, int(args.max_retries)),
                    fallback_source=str(args.fallback_source),
                    alpha_client=alpha_client,
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
