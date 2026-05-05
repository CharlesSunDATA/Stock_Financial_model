"""
update_statements.py

Fetch and store the three core financial statements plus their *_growth and
*_as_reported variants from FinancialModelingPrep (FMP).

Tables populated
  income_statement
  cash_flow_statement
  income_statement_growth
  cash_flow_statement_growth
  balance_sheet_growth
  income_statement_as_reported
  cash_flow_statement_as_reported
  balance_sheet_as_reported

Run
  python3 scripts/update_statements.py AAPL
  python3 scripts/update_statements.py AAPL MSFT NVDA --limit 8 --period annual
  python3 scripts/update_statements.py AAPL --force          # skip staleness check
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

try:
    from pages.modules.fmp_key import get_fmp_key as _get_fmp_key
except ImportError:
    from modules.fmp_key import get_fmp_key as _get_fmp_key  # type: ignore

FMP_STABLE = "https://financialmodelingprep.com/stable"

STALENESS_DAYS = 90
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


# ── helpers ─────────────────────────────────────────────────────────────────

def _api_key() -> str:
    return _get_fmp_key()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_db() -> Path:
    return _project_root() / "data" / "quant_data.db"


def _today() -> date:
    return date.today()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _is_retryable_request_error(exc: BaseException) -> bool:
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        return response is not None and response.status_code in RETRYABLE_HTTP_STATUS_CODES
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))


@retry(
    retry=retry_if_exception(_is_retryable_request_error),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _get_json(url: str, timeout_s: int = 30) -> Any:
    r = requests.get(url, timeout=timeout_s)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(f"{e} | body={r.text[:800]}", response=r) from e
    return r.json()


def _url(endpoint: str, ticker: str, period: str, limit: int) -> str:
    key = _api_key()
    return f"{FMP_STABLE}/{endpoint}?symbol={ticker}&period={period}&limit={limit}&apikey={key}"


# ── staleness check ──────────────────────────────────────────────────────────

def _is_fresh(conn: sqlite3.Connection, ticker: str) -> bool:
    cur = conn.execute(
        "SELECT MAX(report_date) FROM income_statement WHERE ticker = ?", (ticker,)
    )
    row = cur.fetchone()
    if not row or not row[0]:
        return False
    try:
        latest = datetime.strptime(row[0][:10], "%Y-%m-%d").date()
        return (_today() - latest).days < STALENESS_DAYS
    except Exception:
        return False


# ── upsert helpers ───────────────────────────────────────────────────────────

def _upsert_income_statement(
    conn: sqlite3.Connection, ticker: str, rows: list[dict[str, Any]]
) -> int:
    now = _now_iso()
    n = 0
    for r in rows:
        d = str(r.get("date") or "").strip()[:10]
        if not d:
            continue
        period = str(r.get("period") or "").strip()
        conn.execute(
            """
            INSERT OR REPLACE INTO income_statement
              (ticker, report_date, period,
               revenue, gross_profit, operating_income, ebitda, net_income,
               eps, eps_diluted, weighted_avg_shares_out,
               payload_json, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ticker, d, period,
                _safe_float(r.get("revenue")),
                _safe_float(r.get("grossProfit")),
                _safe_float(r.get("operatingIncome")),
                _safe_float(r.get("ebitda")),
                _safe_float(r.get("netIncome")),
                _safe_float(r.get("eps")),
                _safe_float(r.get("epsdiluted") or r.get("epsDiluted")),
                _safe_float(
                    r.get("weightedAverageShsOutDil")
                    or r.get("weightedAverageShsOut")
                ),
                json.dumps(r, default=str),
                now,
            ),
        )
        n += 1
    return n


def _upsert_cash_flow(
    conn: sqlite3.Connection, ticker: str, rows: list[dict[str, Any]]
) -> int:
    now = _now_iso()
    n = 0
    for r in rows:
        d = str(r.get("date") or "").strip()[:10]
        if not d:
            continue
        period = str(r.get("period") or "").strip()
        conn.execute(
            """
            INSERT OR REPLACE INTO cash_flow_statement
              (ticker, report_date, period,
               operating_cash_flow, capital_expenditure, free_cash_flow,
               investing_cash_flow, financing_cash_flow, net_change_in_cash,
               payload_json, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ticker, d, period,
                _safe_float(r.get("operatingCashFlow")),
                _safe_float(r.get("capitalExpenditure")),
                _safe_float(r.get("freeCashFlow")),
                _safe_float(
                    r.get("netCashUsedForInvestingActivites")
                    or r.get("investmentsInPropertyPlantAndEquipment")
                ),
                _safe_float(r.get("netCashUsedProvidedByFinancingActivities")),
                _safe_float(r.get("netChangeInCash")),
                json.dumps(r, default=str),
                now,
            ),
        )
        n += 1
    return n


def _upsert_json_table(
    conn: sqlite3.Connection, table: str, ticker: str, rows: list[dict[str, Any]]
) -> int:
    now = _now_iso()
    n = 0
    for r in rows:
        d = str(r.get("date") or "").strip()[:10]
        if not d:
            continue
        period = str(r.get("period") or "").strip()
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {table}
              (ticker, report_date, period, payload_json, updated_at)
            VALUES (?,?,?,?,?)
            """,
            (ticker, d, period, json.dumps(r, default=str), now),
        )
        n += 1
    return n


# ── per-ticker fetch ─────────────────────────────────────────────────────────

def fetch_ticker(
    conn: sqlite3.Connection,
    ticker: str,
    period: str = "quarter",
    limit: int = 12,
    force: bool = False,
) -> None:
    t = ticker.strip().upper()
    if not t:
        return

    if not force and _is_fresh(conn, t):
        print(f"  {t}: up to date, skipping")
        return

    print(f"  {t}: fetching statements (period={period}, limit={limit})...")

    datasets: dict[str, tuple[str, str]] = {
        "income_statement":               ("income-statement",              "income_statement"),
        "cash_flow_statement":            ("cash-flow-statement",           "cash_flow_statement"),
        "income_statement_growth":        ("income-statement-growth",       "income_statement_growth"),
        "cash_flow_statement_growth":     ("cash-flow-statement-growth",    "cash_flow_statement_growth"),
        "balance_sheet_growth":           ("balance-sheet-statement-growth","balance_sheet_growth"),
        "income_statement_as_reported":   ("income-statement-as-reported",  "income_statement_as_reported"),
        "cash_flow_statement_as_reported":("cash-flow-statement-as-reported","cash_flow_statement_as_reported"),
        "balance_sheet_as_reported":      ("balance-sheet-statement-as-reported","balance_sheet_as_reported"),
    }

    def _fetch_one(item: tuple[str, tuple[str, str]]) -> tuple[str, list[dict]]:
        key, (endpoint, table) = item
        url = _url(endpoint, t, period, limit)
        data = _get_json(url)
        return table, data if isinstance(data, list) else []

    fetched: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_one, item): item[0] for item in datasets.items()}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                table, rows = fut.result()
                fetched[table] = rows
            except Exception as e:
                print(f"    WARNING {key}: {e}")
                fetched[datasets[key][1]] = []

    counts: dict[str, int] = {}
    try:
        conn.execute("BEGIN IMMEDIATE")
        for key, (endpoint, table) in datasets.items():
            rows = fetched.get(table, [])
            if table == "income_statement":
                counts[table] = _upsert_income_statement(conn, t, rows)
            elif table == "cash_flow_statement":
                counts[table] = _upsert_cash_flow(conn, t, rows)
            else:
                counts[table] = _upsert_json_table(conn, table, t, rows)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"    WARNING upsert batch failed: {e}")
        counts = {datasets[key][1]: 0 for key in datasets}

    summary = ", ".join(f"{k.replace('_statement','').replace('_','+')}={v}" for k, v in counts.items())
    print(f"  {t}: done — {summary}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch FMP financial statements into quant_data.db")
    ap.add_argument("tickers", nargs="*", default=[], help="Ticker symbols (omit with --watchlist)")
    ap.add_argument("--watchlist", action="store_true", help="Pull tickers from fmp_watchlist table")
    ap.add_argument("--period", default="quarter", choices=["quarter", "annual"])
    ap.add_argument("--limit", type=int, default=12, help="Quarters/years to fetch per endpoint")
    ap.add_argument("--force", action="store_true", help="Skip staleness check")
    ap.add_argument("--sleep", type=float, default=0.4, help="Seconds between tickers (default 0.4)")
    args = ap.parse_args()

    if not _api_key() or _api_key() == "YOUR_API_KEY_HERE":
        print("FMP_API_KEY is not set.")
        return

    try:
        from scripts.init_db import init_db
    except ModuleNotFoundError:  # pragma: no cover
        from init_db import init_db  # type: ignore
    db_path = init_db(_default_db())
    print(f"DB: {db_path}")

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout = 30000")

        if args.watchlist:
            cur = conn.execute("SELECT ticker FROM fmp_watchlist ORDER BY ticker")
            tickers = [r[0] for r in cur.fetchall()]
            print(f"Watchlist: {len(tickers)} tickers")
        else:
            tickers = [t.strip().upper() for t in args.tickers if t.strip()]
            if not tickers:
                tickers = ["AAPL"]

        total = len(tickers)
        for i, t in enumerate(tickers):
            try:
                fetch_ticker(conn, t, period=args.period, limit=args.limit, force=args.force)
            except Exception as e:
                print(f"  ERROR {t}: {e}")
            if i < total - 1:
                time.sleep(args.sleep)
            if (i + 1) % 100 == 0:
                print(f"  --- progress: {i+1}/{total} ---")

    print("Done.")


if __name__ == "__main__":
    main()
