"""
update_compliance_fmp.py

Download company-level disclosure data from FMP for watchlist stocks.

Tables populated:
  company_executives       /stable/key-executives?symbol=    (per-symbol)
  employee_count           /stable/employee-count?symbol=    (per-symbol historical)

Run:
  python3 scripts/update_compliance_fmp.py
  python3 scripts/update_compliance_fmp.py --watchlist default --force
  python3 scripts/update_compliance_fmp.py AAPL MSFT NVDA   # explicit tickers
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
        FmpClient,
        _parse_iso_date,
        _safe_float,
        _safe_int,
        _today,
        _utc_now_iso,
        load_watchlist_tickers,
    )
    from init_db import init_db  # type: ignore

DATASETS = [
    "company_executives",
    "employee_count",
]


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


# ── Fetch functions ──────────────────────────────────────────────────────────

def fetch_company_executives(client: FmpClient, sym: str) -> list[dict]:
    return _get(client, f"{FMP_BASE}/key-executives", {"symbol": sym})


def fetch_employee_count(client: FmpClient, sym: str) -> list[dict]:
    return _get(client, f"{FMP_BASE}/employee-count", {"symbol": sym})


# ── Upsert functions ─────────────────────────────────────────────────────────

def upsert_company_executives(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        name = str(r.get("name") or "").strip()
        if not name:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO company_executives
              (ticker, name, title, pay, currency_pay, gender,
               year_born, title_since, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                name,
                r.get("title"),
                _safe_float(r.get("pay")),
                r.get("currencyPay"),
                r.get("gender"),
                _safe_int(r.get("yearBorn")),
                r.get("titleSince"),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


def upsert_employee_count(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        period = _parse_iso_date(r.get("periodOfReport") or r.get("period") or r.get("date"))
        if not period:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO employee_count
              (ticker, period_of_report, form_type, filing_date, employee_count, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                period,
                r.get("formType") or r.get("form_type"),
                _parse_iso_date(r.get("filingDate") or r.get("filing_date")),
                _safe_int(r.get("employeeCount") or r.get("employee_count")),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


# ── Orchestration ────────────────────────────────────────────────────────────

def _process_ticker(
    client: FmpClient,
    conn: sqlite3.Connection,
    ticker: str,
    *,
    datasets: list[str],
    max_age_hours: float,
    force: bool,
    sleep_s: float,
) -> dict[str, int]:
    totals: dict[str, int] = {}

    for ds in datasets:
        if not force:
            last = _ticker_last_updated(conn, ds, ticker)
            if _is_fresh(last, max_age_hours):
                continue

        if ds == "company_executives":
            rows = fetch_company_executives(client, ticker)
            n = upsert_company_executives(conn, ticker, rows)
        elif ds == "employee_count":
            rows = fetch_employee_count(client, ticker)
            n = upsert_employee_count(conn, ticker, rows)
        else:
            n = 0

        totals[ds] = n
        if sleep_s > 0:
            time.sleep(sleep_s)

    conn.commit()
    return totals


def run(
    *,
    tickers: list[str] | None = None,
    watchlist_name: str = "default",
    max_age_hours: float = 168.0,
    force: bool = False,
    datasets: list[str] | None = None,
    sleep_s: float = 0.15,
    log_every: int = 25,
    api_key: str = "",
) -> None:
    key = api_key.strip() or os.getenv("FMP_API_KEY", "").strip() or FMP_API_KEY
    if not key:
        raise ValueError("FMP_API_KEY not set. Pass --api-key or export FMP_API_KEY.")

    db = init_db()
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL;")

    if not tickers:
        tickers = load_watchlist_tickers(conn, watchlist_name=watchlist_name)
    if not tickers:
        print(f"No tickers found (watchlist='{watchlist_name}'). Populate fmp_watchlist first.")
        conn.close()
        return

    active_ds = datasets or DATASETS
    unknown = [d for d in active_ds if d not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}. Valid: {', '.join(DATASETS)}")

    client = FmpClient(key, calls_per_minute=250)

    print(f"Tickers : {len(tickers)}")
    print(f"Datasets : {', '.join(active_ds)}")
    print(f"max_age_hours={max_age_hours}, force={force}")
    print()

    grand_totals: dict[str, int] = {}

    for i, ticker in enumerate(tickers, start=1):
        row_totals = _process_ticker(
            client, conn, ticker,
            datasets=active_ds,
            max_age_hours=max_age_hours,
            force=force,
            sleep_s=sleep_s,
        )
        for ds, n in row_totals.items():
            grand_totals[ds] = grand_totals.get(ds, 0) + n

        if i % log_every == 0 or i == len(tickers):
            print(f"  [{i:,}/{len(tickers):,}] {ticker}  total rows: {sum(grand_totals.values()):,}")

    conn.close()

    print("\nDone. Row totals per dataset:")
    for ds in active_ds:
        print(f"  {ds:<30} {grand_totals.get(ds, 0):>8,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download company executives and employee count from FMP"
    )
    parser.add_argument(
        "tickers", nargs="*",
        help="Explicit ticker list. If omitted, uses --watchlist.",
    )
    parser.add_argument("--watchlist", default="default")
    parser.add_argument(
        "--max-age-hours", type=float, default=168.0,
        help="Skip if data is fresher than this many hours (default 168 = 7 days; 0 = always fetch)",
    )
    parser.add_argument("--force", action="store_true", help="Re-fetch regardless of age")
    parser.add_argument(
        "--datasets", default="",
        help=f"Comma-separated subset. Options: {', '.join(DATASETS)}",
    )
    parser.add_argument("--sleep", type=float, default=0.15, help="Seconds between API calls")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--api-key", default="", help="FMP API key (overrides env)")
    args = parser.parse_args()

    ds = [d.strip() for d in args.datasets.split(",") if d.strip()] if args.datasets.strip() else None

    run(
        tickers=[t.upper() for t in args.tickers] if args.tickers else None,
        watchlist_name=args.watchlist,
        max_age_hours=args.max_age_hours,
        force=args.force,
        datasets=ds,
        sleep_s=args.sleep,
        log_every=args.log_every,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
