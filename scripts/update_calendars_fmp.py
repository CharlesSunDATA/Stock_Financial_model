"""
update_calendars_fmp.py

Fetch and persist calendar / corporate-event data from FMP for watchlist stocks.

Tables populated:
  earnings_calendar     v3  /historical/earning_calendar/{sym}      (per-symbol)
  historical_dividends  v3  /historical-price-full/stock_dividend/{sym}  (per-symbol)
  historical_splits     v3  /historical-price-full/stock_split/{sym}     (per-symbol)
  ipo_calendar          v3  /ipo_calendar?from=&to=                  (global)
  ipo_prospectus        v4  /ipo-calendar-prospectus?from=&to=       (global)
  economic_calendar     v3  /economic_calendar?from=&to=             (global)
  mergers_acquisitions  v4  /mergers-acquisitions-rss-feed           (global; filtered to watchlist)

Run:
  python3 scripts/update_calendars_fmp.py
  python3 scripts/update_calendars_fmp.py --watchlist default --max-age-hours 24
  python3 scripts/update_calendars_fmp.py --force --datasets earnings_calendar,historical_dividends
  python3 scripts/update_calendars_fmp.py --from-date 2024-01-01 --to-date 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from scripts.market_loader_fmp import (
        FMP_API_KEY,
        FMP_BASE,
        FMP_V3_ROOT,
        FMP_V4_ROOT,
        FmpClient,
        _parse_iso_date,
        _safe_float,
        _safe_int,
        _today,
        _utc_now_iso,
        load_watchlist_tickers,
    )
    from scripts.init_db import init_db
except ModuleNotFoundError:
    from market_loader_fmp import (  # type: ignore
        FMP_API_KEY,
        FMP_BASE,
        FMP_V3_ROOT,
        FMP_V4_ROOT,
        FmpClient,
        _parse_iso_date,
        _safe_float,
        _safe_int,
        _today,
        _utc_now_iso,
        load_watchlist_tickers,
    )
    from init_db import init_db  # type: ignore


# ── Staleness helpers ────────────────────────────────────────────────────────

def _ticker_last_updated(conn: sqlite3.Connection, table: str, ticker: str) -> str | None:
    try:
        cur = conn.execute(
            f"SELECT MAX(updated_at) FROM {table} WHERE ticker = ?", (ticker,)
        )
        row = cur.fetchone()
        return str(row[0]) if row and row[0] else None
    except Exception:
        return None


def _global_last_updated(conn: sqlite3.Connection, table: str) -> str | None:
    try:
        cur = conn.execute(f"SELECT MAX(updated_at) FROM {table}")
        row = cur.fetchone()
        return str(row[0]) if row and row[0] else None
    except Exception:
        return None


def _is_fresh(last_updated: str | None, max_age_hours: float) -> bool:
    if not last_updated or max_age_hours <= 0:
        return False
    try:
        t = datetime.fromisoformat(last_updated.replace("Z", ""))
        return (datetime.utcnow() - t).total_seconds() / 3600.0 <= max_age_hours
    except Exception:
        return False


# ── Generic fetch helper ─────────────────────────────────────────────────────

def _get(client: FmpClient, url: str, params: dict | None = None) -> list[dict[str, Any]]:
    try:
        data = client.get_data(url, params)
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        if isinstance(data, dict) and not data.get("Error Message") and not data.get("error"):
            return [data]
        return []
    except Exception as e:
        print(f"    WARN {url}: {e}")
        return []


# ── Per-symbol fetch functions ───────────────────────────────────────────────

def fetch_earnings_calendar(client: FmpClient, from_date: str, to_date: str) -> list[dict]:
    return _get(client, f"{FMP_BASE}/earnings-calendar", {"from": from_date, "to": to_date})


def fetch_historical_dividends(client: FmpClient, sym: str) -> list[dict]:
    data = _get(client, f"{FMP_BASE}/dividends", {"symbol": sym})
    # stable endpoint returns {"symbol": ..., "historical": [...]}
    if data and isinstance(data[0], dict) and "historical" in data[0]:
        return [r for r in data[0]["historical"] if isinstance(r, dict)]
    return data


def fetch_historical_splits(client: FmpClient, sym: str) -> list[dict]:
    data = _get(client, f"{FMP_BASE}/splits", {"symbol": sym})
    # stable endpoint returns {"symbol": ..., "historical": [...]}
    if data and isinstance(data[0], dict) and "historical" in data[0]:
        return [r for r in data[0]["historical"] if isinstance(r, dict)]
    return data


# ── Global fetch functions ───────────────────────────────────────────────────

def fetch_ipo_calendar(client: FmpClient, from_date: str, to_date: str) -> list[dict]:
    return _get(client, f"{FMP_BASE}/ipo-calendar", {"from": from_date, "to": to_date})


def fetch_ipo_prospectus(client: FmpClient, from_date: str, to_date: str) -> list[dict]:
    return _get(client, f"{FMP_BASE}/ipo-calendar-prospectus", {"from": from_date, "to": to_date})


def fetch_economic_calendar(client: FmpClient, from_date: str, to_date: str) -> list[dict]:
    return _get(client, f"{FMP_BASE}/economic-calendar", {"from": from_date, "to": to_date})


def fetch_mergers_acquisitions(
    client: FmpClient,
    watchlist: set[str],
    *,
    max_pages: int = 20,
    sleep_s: float = 0.1,
) -> list[dict]:
    rows = _get(client, f"{FMP_BASE}/mergers-acquisitions-latest", {"limit": 500})
    return [
        r for r in rows
        if str(r.get("symbol") or "").upper() in watchlist
        or str(r.get("targetedSymbol") or "").upper() in watchlist
    ]


# ── Upsert functions ─────────────────────────────────────────────────────────

def upsert_earnings_calendar(conn: sqlite3.Connection, rows: list[dict], watchlist: set[str] | None = None) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        ticker = str(r.get("symbol") or "").strip().upper()
        if not ticker:
            continue
        if watchlist and ticker not in watchlist:
            continue
        d = _parse_iso_date(r.get("date"))
        if not d:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO earnings_calendar
              (ticker, event_date, eps, eps_estimated, time,
               revenue, revenue_estimated, fiscal_date_ending, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker, d,
                _safe_float(r.get("epsActual") if r.get("epsActual") is not None else r.get("eps")),
                _safe_float(r.get("epsEstimated")),
                r.get("time"),
                _safe_float(r.get("revenueActual") if r.get("revenueActual") is not None else r.get("revenue")),
                _safe_float(r.get("revenueEstimated")),
                r.get("fiscalDateEnding"),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


def upsert_historical_dividends(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        d = _parse_iso_date(r.get("date"))
        if not d:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO historical_dividends
              (ticker, ex_dividend_date, label, adj_dividend, dividend,
               record_date, payment_date, declaration_date, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker, d,
                r.get("label"),
                _safe_float(r.get("adjDividend")),
                _safe_float(r.get("dividend")),
                _parse_iso_date(r.get("recordDate")),
                _parse_iso_date(r.get("paymentDate")),
                _parse_iso_date(r.get("declarationDate")),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


def upsert_historical_splits(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        d = _parse_iso_date(r.get("date"))
        if not d:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO historical_splits
              (ticker, split_date, label, numerator, denominator, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker, d,
                r.get("label"),
                _safe_float(r.get("numerator")),
                _safe_float(r.get("denominator")),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


def upsert_ipo_calendar(conn: sqlite3.Connection, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        d = _parse_iso_date(r.get("date") or r.get("ipoDate"))
        sym = str(r.get("symbol") or "").strip().upper()
        if not d:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO ipo_calendar
              (symbol, ipo_date, company, exchange, actions, shares,
               price_range, market_cap, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sym, d,
                r.get("company") or r.get("companyName"),
                r.get("exchange"),
                r.get("actions"),
                _safe_float(r.get("shares")),
                r.get("priceRange"),
                _safe_float(r.get("marketCap")),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


def upsert_ipo_prospectus(conn: sqlite3.Connection, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        d = _parse_iso_date(r.get("ipoDate") or r.get("date"))
        sym = str(r.get("symbol") or "").strip().upper()
        if not d:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO ipo_prospectus
              (symbol, ipo_date, company, cik, form_type, filing_date, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sym, d,
                r.get("company") or r.get("companyName"),
                r.get("cik"),
                r.get("form") or r.get("formType"),
                _parse_iso_date(r.get("filingDate") or r.get("date")),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


def upsert_economic_calendar(conn: sqlite3.Connection, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        raw_date = str(r.get("date") or "")
        d = _parse_iso_date(raw_date[:10]) if raw_date else None
        event_name = str(r.get("event") or "").strip()
        country = str(r.get("country") or "").strip().upper()
        if not d or not event_name:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO economic_calendar
              (event_date, event_name, country, currency, previous, estimate, actual,
               impact, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                d, event_name, country,
                r.get("currency"),
                _safe_float(r.get("previous")),
                _safe_float(r.get("estimate")),
                _safe_float(r.get("actual")),
                r.get("impact"),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


def upsert_mergers_acquisitions(conn: sqlite3.Connection, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        url = str(r.get("link") or r.get("url") or r.get("transactionUrl") or "").strip()
        if not url:
            continue
        d = _parse_iso_date(str(r.get("transactionDate") or r.get("date") or r.get("acceptanceTime") or "")[:10])
        acquirer_sym = str(r.get("symbol") or "").strip().upper() or None
        target_sym = str(r.get("targetedSymbol") or "").strip().upper() or None
        conn.execute(
            """
            INSERT OR REPLACE INTO mergers_acquisitions
              (transaction_url, event_date, title, acquirer_ticker, acquirer_name,
               target_ticker, target_name, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                url, d,
                r.get("title"),
                acquirer_sym,
                r.get("companyName") or r.get("acquirerName"),
                target_sym,
                r.get("targetedCompanyName") or r.get("targetName"),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


# ── Dataset registry ─────────────────────────────────────────────────────────

PER_SYMBOL_DATASETS = ["historical_dividends", "historical_splits"]
GLOBAL_DATASETS = ["earnings_calendar", "ipo_calendar", "ipo_prospectus", "economic_calendar", "mergers_acquisitions"]
DATASETS = PER_SYMBOL_DATASETS + GLOBAL_DATASETS

_PER_SYMBOL_FETCH = {
    "historical_dividends": fetch_historical_dividends,
    "historical_splits":   fetch_historical_splits,
}

_PER_SYMBOL_UPSERT = {
    "historical_dividends": upsert_historical_dividends,
    "historical_splits":   upsert_historical_splits,
}


# ── Orchestration ────────────────────────────────────────────────────────────

def _process_per_symbol(
    client: FmpClient,
    conn: sqlite3.Connection,
    tickers: list[str],
    *,
    datasets: list[str],
    max_age_hours: float,
    force: bool,
    sleep_s: float,
    log_every: int,
) -> dict[str, int]:
    totals: dict[str, int] = {}
    active = [d for d in datasets if d in PER_SYMBOL_DATASETS]
    if not active:
        return totals

    for i, ticker in enumerate(tickers, start=1):
        for ds in active:
            if not force:
                last = _ticker_last_updated(conn, ds, ticker)
                if _is_fresh(last, max_age_hours):
                    continue
            rows = _PER_SYMBOL_FETCH[ds](client, ticker)
            n = _PER_SYMBOL_UPSERT[ds](conn, ticker, rows)
            totals[ds] = totals.get(ds, 0) + n
            if sleep_s > 0:
                time.sleep(sleep_s)
        conn.commit()
        if i % log_every == 0 or i == len(tickers):
            print(f"  [{i:,}/{len(tickers):,}] {ticker}  per-symbol rows: {sum(totals.values()):,}")

    return totals


def _process_global(
    client: FmpClient,
    conn: sqlite3.Connection,
    watchlist: set[str],
    *,
    datasets: list[str],
    from_date: str,
    to_date: str,
    max_age_hours: float,
    force: bool,
    sleep_s: float,
) -> dict[str, int]:
    totals: dict[str, int] = {}
    active = [d for d in datasets if d in GLOBAL_DATASETS]
    if not active:
        return totals

    for ds in active:
        if not force:
            last = _global_last_updated(conn, ds)
            if _is_fresh(last, max_age_hours):
                print(f"  [skip] {ds} is fresh (max_age_hours={max_age_hours})")
                continue

        print(f"  Fetching {ds} …", end=" ", flush=True)
        if ds == "earnings_calendar":
            rows = fetch_earnings_calendar(client, from_date, to_date)
            n = upsert_earnings_calendar(conn, rows, watchlist)
        elif ds == "ipo_calendar":
            rows = fetch_ipo_calendar(client, from_date, to_date)
            n = upsert_ipo_calendar(conn, rows)
        elif ds == "ipo_prospectus":
            rows = fetch_ipo_prospectus(client, from_date, to_date)
            n = upsert_ipo_prospectus(conn, rows)
        elif ds == "economic_calendar":
            rows = fetch_economic_calendar(client, from_date, to_date)
            n = upsert_economic_calendar(conn, rows)
        elif ds == "mergers_acquisitions":
            rows = fetch_mergers_acquisitions(client, watchlist, sleep_s=sleep_s)
            n = upsert_mergers_acquisitions(conn, rows)
        else:
            n = 0

        conn.commit()
        totals[ds] = n
        print(f"{n:,} rows")
        if sleep_s > 0:
            time.sleep(sleep_s)

    return totals


def run(
    *,
    watchlist_name: str = "default",
    max_age_hours: float = 24.0,
    force: bool = False,
    datasets: list[str] | None = None,
    from_date: str = "",
    to_date: str = "",
    sleep_s: float = 0.1,
    log_every: int = 10,
    api_key: str = "",
) -> None:
    key = api_key.strip() or os.getenv("FMP_API_KEY", "").strip() or FMP_API_KEY
    if not key:
        raise ValueError("FMP_API_KEY not set. Pass --api-key or export FMP_API_KEY.")

    today = _today()
    f_date = from_date or (today - timedelta(days=365)).isoformat()
    t_date = to_date or (today + timedelta(days=90)).isoformat()

    db = init_db()
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL;")

    tickers = load_watchlist_tickers(conn, watchlist_name=watchlist_name)
    if not tickers:
        print(f"No tickers in watchlist '{watchlist_name}'. Populate fmp_watchlist first.")
        conn.close()
        return

    active = datasets or DATASETS
    unknown = [d for d in active if d not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}. Valid: {', '.join(DATASETS)}")

    watchlist_set = set(tickers)
    client = FmpClient(key, calls_per_minute=250)

    print(f"Watchlist '{watchlist_name}': {len(tickers)} tickers")
    print(f"Datasets : {', '.join(active)}")
    print(f"Date range: {f_date} → {t_date}")
    print(f"max_age_hours={max_age_hours}, force={force}")
    print()

    print("── Per-symbol datasets ──")
    sym_totals = _process_per_symbol(
        client, conn, tickers,
        datasets=active,
        max_age_hours=max_age_hours,
        force=force,
        sleep_s=sleep_s,
        log_every=log_every,
    )

    print("\n── Global datasets ──")
    glob_totals = _process_global(
        client, conn, watchlist_set,
        datasets=active,
        from_date=f_date,
        to_date=t_date,
        max_age_hours=max_age_hours,
        force=force,
        sleep_s=sleep_s,
    )

    all_totals = {**sym_totals, **glob_totals}
    print("\nDone. Row totals per dataset (DB):")
    for ds in active:
        try:
            db_count = conn.execute(f"SELECT COUNT(*) FROM {ds}").fetchone()[0]
        except Exception:
            db_count = all_totals.get(ds, 0)
        inserted = all_totals.get(ds)
        suffix = f"  (+{inserted:,} this run)" if inserted is not None else "  (skipped)"
        print(f"  {ds:<30} {db_count:>8,}{suffix}")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download calendar & event data from FMP for watchlist stocks"
    )
    parser.add_argument("--watchlist", default="default")
    parser.add_argument(
        "--max-age-hours", type=float, default=24.0,
        help="Skip if data is fresher than this many hours (0 = always fetch)",
    )
    parser.add_argument("--force", action="store_true", help="Re-fetch regardless of age")
    parser.add_argument(
        "--datasets", default="",
        help=f"Comma-separated subset. Options: {', '.join(DATASETS)}",
    )
    parser.add_argument("--from-date", default="", help="Start date YYYY-MM-DD (global datasets)")
    parser.add_argument("--to-date", default="", help="End date YYYY-MM-DD (global datasets)")
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds between API calls")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--api-key", default="", help="FMP API key (overrides env)")
    args = parser.parse_args()

    ds = [d.strip() for d in args.datasets.split(",") if d.strip()] if args.datasets.strip() else None
    run(
        watchlist_name=args.watchlist,
        max_age_hours=args.max_age_hours,
        force=args.force,
        datasets=ds,
        from_date=args.from_date,
        to_date=args.to_date,
        sleep_s=args.sleep,
        log_every=args.log_every,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
