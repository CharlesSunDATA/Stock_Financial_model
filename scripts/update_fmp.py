"""
update_fmp.py

Fetch quarterly fundamentals from FinancialModelingPrep (FMP) and populate a local SQLite database.

Key features
- Requests-based FMP API client
- Smart caching: if the latest fundamental_data.report_date is < 90 days old, skip the ticker
- INSERT OR REPLACE upserts into:
  - fundamental_data
  - balance_sheet
  - financial_ratios
  - analyst_estimates
  - company_profile (fetched once per ticker)
- Batch processing with a 0.5s sleep per ticker

Run
  python3 scripts/update_fmp.py AAPL
  python3 scripts/update_fmp.py AAPL MSFT NVDA TSLA
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

try:
    from scripts.init_db import init_db, default_db_path
except ModuleNotFoundError:  # pragma: no cover
    from init_db import init_db, default_db_path  # type: ignore


from pages.modules.fmp_key import get_fmp_key as _get_fmp_key
FMP_API_KEY = _get_fmp_key()
FMP_BASE_V3 = "https://financialmodelingprep.com/api/v3"
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


def _today() -> date:
    return date.today()


def _parse_iso_date(s: Any) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(float(x))
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
def _get_json(url: str, timeout_s: int = 25) -> Any:
    r = requests.get(url, timeout=timeout_s)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        body = ""
        try:
            body = r.text[:1200]
        except Exception:
            body = ""
        raise requests.HTTPError(f"{e} | response_body={body}", response=r) from e
    return r.json()


def _url_income_statement(ticker: str, limit: int = 4) -> str:
    return f"{FMP_BASE_V3}/income-statement/{ticker}?period=quarter&limit={int(limit)}&apikey={FMP_API_KEY}"


def _url_cash_flow(ticker: str, limit: int = 4) -> str:
    return f"{FMP_BASE_V3}/cash-flow-statement/{ticker}?period=quarter&limit={int(limit)}&apikey={FMP_API_KEY}"


def _url_balance_sheet(ticker: str, limit: int = 4) -> str:
    return f"{FMP_BASE_V3}/balance-sheet-statement/{ticker}?period=quarter&limit={int(limit)}&apikey={FMP_API_KEY}"


def _url_ratios(ticker: str, limit: int = 4) -> str:
    return f"{FMP_BASE_V3}/ratios/{ticker}?period=quarter&limit={int(limit)}&apikey={FMP_API_KEY}"


def _url_analyst_estimates(ticker: str, limit: int = 4) -> str:
    return f"{FMP_BASE_V3}/analyst-estimates/{ticker}?period=quarter&limit={int(limit)}&apikey={FMP_API_KEY}"


def _url_profile(ticker: str) -> str:
    return f"{FMP_BASE_V3}/profile/{ticker}?apikey={FMP_API_KEY}"


def _latest_fundamental_date(conn: sqlite3.Connection, ticker: str) -> date | None:
    cur = conn.execute("SELECT MAX(report_date) FROM fundamental_data WHERE ticker = ?", (ticker,))
    row = cur.fetchone()
    return _parse_iso_date(row[0]) if row and row[0] else None


def _is_up_to_date(conn: sqlite3.Connection, ticker: str, lookback_days: int = 90) -> tuple[bool, str]:
    latest = _latest_fundamental_date(conn, ticker)
    if latest is None:
        return False, f"No existing fundamentals found for {ticker}. Fetching..."
    age = (_today() - latest).days
    if age < lookback_days:
        return True, f"Data for {ticker} is up to date (latest={latest.isoformat()}, age={age} days). Skipping..."
    return False, f"Data for {ticker} is stale (latest={latest.isoformat()}, age={age} days). Fetching..."


def _upsert_fundamental_data(conn: sqlite3.Connection, ticker: str, rows: list[dict[str, Any]]) -> int:
    n = 0
    for r in rows:
        d = str(r.get("date") or "").strip()
        if not d:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO fundamental_data (ticker, report_date, revenue, eps, free_cash_flow)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                ticker,
                d,
                _safe_float(r.get("revenue")),
                _safe_float(r.get("eps")),
                _safe_float(r.get("free_cash_flow")),
            ),
        )
        n += 1
    return n


def _upsert_balance_sheet(conn: sqlite3.Connection, ticker: str, rows: list[dict[str, Any]]) -> int:
    n = 0
    for r in rows:
        d = str(r.get("date") or "").strip()
        if not d:
            continue
        cash = r.get("cashAndCashEquivalents") or r.get("cashAndShortTermInvestments")
        shares = r.get("commonStockSharesOutstanding") or r.get("weightedAverageShsOutDil") or r.get("weightedAverageShsOut")
        conn.execute(
            """
            INSERT OR REPLACE INTO balance_sheet
              (ticker, report_date, cash_and_equivalents, total_assets, total_liabilities, outstanding_shares)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                d,
                _safe_float(cash),
                _safe_float(r.get("totalAssets")),
                _safe_float(r.get("totalLiabilities")),
                _safe_float(shares),
            ),
        )
        n += 1
    return n


def _upsert_financial_ratios(conn: sqlite3.Connection, ticker: str, rows: list[dict[str, Any]]) -> int:
    n = 0
    for r in rows:
        d = str(r.get("date") or "").strip()
        if not d:
            continue
        # Common FMP ratio keys (best-effort fallbacks)
        roe = r.get("returnOnEquity") or r.get("returnOnEquityTTM")
        roic = r.get("returnOnInvestedCapital") or r.get("roic")
        pe = r.get("priceEarningsRatio") or r.get("priceEarningsRatioTTM") or r.get("peRatio")
        pb = r.get("priceToBookRatio") or r.get("pbRatio")
        ev_ebitda = r.get("enterpriseValueMultiple") or r.get("evToEbitda") or r.get("evToEBITDA")

        conn.execute(
            """
            INSERT OR REPLACE INTO financial_ratios
              (ticker, report_date, roe, roic, pe_ratio, pb_ratio, ev_to_ebitda)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                d,
                _safe_float(roe),
                _safe_float(roic),
                _safe_float(pe),
                _safe_float(pb),
                _safe_float(ev_ebitda),
            ),
        )
        n += 1
    return n


def _upsert_analyst_estimates(conn: sqlite3.Connection, ticker: str, rows: list[dict[str, Any]]) -> int:
    n = 0
    for r in rows:
        d = str(r.get("date") or "").strip()
        if not d:
            continue
        est_rev = (
            r.get("estimatedRevenueAvg")
            or r.get("estimatedRevenue")
            or r.get("estimatedRevenueLow")
            or r.get("estimatedRevenueHigh")
        )
        est_eps = r.get("estimatedEpsAvg") or r.get("estimatedEps") or r.get("estimatedEPSAvg") or r.get("estimatedEPS")
        conn.execute(
            """
            INSERT OR REPLACE INTO analyst_estimates
              (ticker, estimated_date, estimated_revenue, estimated_eps)
            VALUES (?, ?, ?, ?)
            """,
            (ticker, d, _safe_float(est_rev), _safe_float(est_eps)),
        )
        n += 1
    return n


def _upsert_company_profile(conn: sqlite3.Connection, ticker: str, profile_rows: list[dict[str, Any]]) -> bool:
    if not profile_rows:
        return False
    p = profile_rows[0] if isinstance(profile_rows, list) else {}
    if not isinstance(p, dict):
        return False
    conn.execute(
        """
        INSERT OR REPLACE INTO company_profile
          (ticker, company_name, sector, industry, full_time_employees, image_url)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            ticker,
            str(p.get("companyName") or "") or None,
            str(p.get("sector") or "") or None,
            str(p.get("industry") or "") or None,
            _safe_int(p.get("fullTimeEmployees")),
            str(p.get("image") or p.get("imageUrl") or "") or None,
        ),
    )
    return True


def _has_profile(conn: sqlite3.Connection, ticker: str) -> bool:
    cur = conn.execute("SELECT 1 FROM company_profile WHERE ticker = ? LIMIT 1", (ticker,))
    return cur.fetchone() is not None


def fetch_and_update_ticker(conn: sqlite3.Connection, ticker: str) -> None:
    t = (ticker or "").strip().upper()
    if not t:
        return

    up_to_date, msg = _is_up_to_date(conn, t, lookback_days=90)
    print(msg)
    if up_to_date:
        return

    print(f"Fetching data for {t}...")

    income = _get_json(_url_income_statement(t, limit=4))
    cashflow = _get_json(_url_cash_flow(t, limit=4))
    balance = _get_json(_url_balance_sheet(t, limit=4))
    ratios = _get_json(_url_ratios(t, limit=4))
    estimates = _get_json(_url_analyst_estimates(t, limit=4))

    # Merge income + cashflow by date for fundamental_data
    inc_by_date = {str(r.get("date")): r for r in income if isinstance(r, dict) and r.get("date")}
    cf_by_date = {str(r.get("date")): r for r in cashflow if isinstance(r, dict) and r.get("date")}
    all_dates = sorted(set(inc_by_date.keys()) | set(cf_by_date.keys()))
    fundamentals_merged: list[dict[str, Any]] = []
    for d in all_dates:
        inc = inc_by_date.get(d, {})
        cf = cf_by_date.get(d, {})
        fundamentals_merged.append(
            {
                "date": d,
                "revenue": inc.get("revenue"),
                "eps": inc.get("epsdiluted") if "epsdiluted" in inc else inc.get("epsDiluted"),
                "free_cash_flow": cf.get("freeCashFlow"),
            }
        )

    # Fetch profile only once per ticker
    profile_rows: list[dict[str, Any]] = []
    has_profile = _has_profile(conn, t)
    if not has_profile:
        profile = _get_json(_url_profile(t))
        profile_rows = profile if isinstance(profile, list) else []

    try:
        conn.execute("BEGIN IMMEDIATE")
        n_fund = _upsert_fundamental_data(conn, t, fundamentals_merged)
        n_bs = _upsert_balance_sheet(conn, t, balance if isinstance(balance, list) else [])
        n_rat = _upsert_financial_ratios(conn, t, ratios if isinstance(ratios, list) else [])
        n_est = _upsert_analyst_estimates(conn, t, estimates if isinstance(estimates, list) else [])
        prof_ok = True if has_profile else _upsert_company_profile(conn, t, profile_rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    print(
        "Successfully updated records for "
        f"{t}: fundamentals={n_fund}, balance_sheet={n_bs}, ratios={n_rat}, estimates={n_est}, profile={'yes' if prof_ok else 'no'}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch FMP data and populate quant_data.db")
    ap.add_argument("tickers", nargs="*", help="Tickers, e.g. AAPL MSFT NVDA TSLA")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in (args.tickers if args.tickers else ["AAPL", "MSFT", "NVDA", "TSLA"]) if t.strip()]

    if FMP_API_KEY.strip() in ("", "YOUR_API_KEY_HERE"):
        print("FMP_API_KEY is not set. Please edit the script and set FMP_API_KEY before running.")
        return

    print("Connecting to DB...")
    db_path = init_db(default_db_path())
    print(f"DB path: {db_path}")

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA busy_timeout = 30000")
        for i, t in enumerate(tickers):
            try:
                fetch_and_update_ticker(conn, t)
            except Exception as e:
                print(f"Error while updating {t}: {e}")
            if i < len(tickers) - 1:
                time.sleep(0.5)

    print("Done.")


if __name__ == "__main__":
    main()
