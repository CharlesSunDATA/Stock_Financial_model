"""
update_valuation_analyst_fmp.py

Fetch and persist valuation + analyst-sentiment data for watchlist stocks.

Tables populated:
  enterprise_values              v3  /enterprise-values/{sym}
  historical_market_cap          stable  /historical-market-capitalization
  dcf                            v3  /discounted-cash-flow/{sym}
  advanced_dcf                   v4  /advanced_dcf
  levered_dcf                    v4  /advanced_levered_dcf
  price_target                   v4  /price-target
  price_target_consensus         v4  /price-target-consensus
  price_target_summary           v4  /price-target-summary
  upgrades_downgrades            v4  /upgrades-downgrades
  upgrades_downgrades_consensus  v4  /upgrades-downgrades-consensus
  stock_grade                    v3  /grade/{sym}
  stock_peers                    v4  /stock_peers

Run:
  python3 scripts/update_valuation_analyst_fmp.py
  python3 scripts/update_valuation_analyst_fmp.py --watchlist default --max-age-hours 24
  python3 scripts/update_valuation_analyst_fmp.py --force --datasets price_target,price_target_consensus
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime
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


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
        t = datetime.fromisoformat(last_updated.replace("Z", ""))
        return (datetime.utcnow() - t).total_seconds() / 3600.0 <= max_age_hours
    except Exception:
        return False


# ── Fetch functions ──────────────────────────────────────────────────────────

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


# Datasets that require a higher-tier FMP plan (404 on standard plan; tables removed from DB).
PLAN_RESTRICTED = {
    "price_target",
    "upgrades_downgrades",
    "upgrades_downgrades_consensus",
    "stock_grade",
    "advanced_dcf",
    "levered_dcf",
}


def fetch_enterprise_values(client: FmpClient, sym: str, *, period: str = "quarter", limit: int = 40) -> list[dict]:
    return _get(client, f"{FMP_BASE}/enterprise-values", {"symbol": sym, "period": period, "limit": limit})


def fetch_historical_market_cap(client: FmpClient, sym: str, *, limit: int = 500) -> list[dict]:
    return _get(client, f"{FMP_BASE}/historical-market-capitalization", {"symbol": sym, "limit": limit})


def fetch_dcf(client: FmpClient, sym: str) -> list[dict]:
    return _get(client, f"{FMP_BASE}/discounted-cash-flow", {"symbol": sym})


def fetch_advanced_dcf(client: FmpClient, sym: str) -> list[dict]:
    return _get(client, f"{FMP_V4_ROOT}/advanced_dcf", {"symbol": sym})


def fetch_levered_dcf(client: FmpClient, sym: str) -> list[dict]:
    return _get(client, f"{FMP_V4_ROOT}/advanced_levered_dcf", {"symbol": sym})


def fetch_price_target(client: FmpClient, sym: str) -> list[dict]:
    return _get(client, f"{FMP_V4_ROOT}/price-target", {"symbol": sym})


def fetch_price_target_consensus(client: FmpClient, sym: str) -> list[dict]:
    return _get(client, f"{FMP_BASE}/price-target-consensus", {"symbol": sym})


def fetch_price_target_summary(client: FmpClient, sym: str) -> list[dict]:
    return _get(client, f"{FMP_BASE}/price-target-summary", {"symbol": sym})


def fetch_upgrades_downgrades(client: FmpClient, sym: str) -> list[dict]:
    return _get(client, f"{FMP_V4_ROOT}/upgrades-downgrades", {"symbol": sym})


def fetch_upgrades_downgrades_consensus(client: FmpClient, sym: str) -> list[dict]:
    return _get(client, f"{FMP_V4_ROOT}/upgrades-downgrades-consensus", {"symbol": sym})


def fetch_stock_grade(client: FmpClient, sym: str, *, limit: int = 500) -> list[dict]:
    return _get(client, f"{FMP_V3_ROOT}/grade/{sym}", {"limit": limit})


def fetch_stock_peers(client: FmpClient, sym: str) -> list[dict]:
    # Stable endpoint returns [{symbol, companyName, price, mktCap}, ...]
    return _get(client, f"{FMP_BASE}/stock-peers", {"symbol": sym})


# ── Upsert functions ─────────────────────────────────────────────────────────

def upsert_enterprise_values(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        d = _parse_iso_date(r.get("date"))
        if not d:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO enterprise_values
              (ticker, report_date, stock_price, number_of_shares, market_cap,
               minus_cash, add_total_debt, enterprise_value, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker, d,
                _safe_float(r.get("stockPrice")),
                _safe_float(r.get("numberOfShares")),
                _safe_float(r.get("marketCapitalization")),
                _safe_float(r.get("minusCashAndCashEquivalents")),
                _safe_float(r.get("addTotalDebt")),
                _safe_float(r.get("enterpriseValue")),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


def upsert_historical_market_cap(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        d = _parse_iso_date(r.get("date"))
        if not d:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO historical_market_cap (ticker, price_date, market_cap, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (ticker, d, _safe_float(r.get("marketCap")), now),
        )
        n += 1
    return n


def upsert_dcf(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    asof = _today().isoformat()
    n = 0
    for r in rows:
        d = _parse_iso_date(r.get("date")) or asof
        conn.execute(
            """
            INSERT OR REPLACE INTO dcf
              (ticker, as_of_date, dcf_value, stock_price, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ticker, d,
                _safe_float(r.get("dcf")),
                _safe_float(r.get("Stock Price") or r.get("stockPrice") or r.get("price")),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


def upsert_advanced_dcf(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    asof = _today().isoformat()
    n = 0
    for r in rows:
        d = _parse_iso_date(r.get("date") or r.get("year")) or asof
        conn.execute(
            """
            INSERT OR REPLACE INTO advanced_dcf (ticker, as_of_date, payload_json, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (ticker, d, json.dumps(r, ensure_ascii=False, separators=(",", ":")), now),
        )
        n += 1
    return n


def upsert_levered_dcf(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    asof = _today().isoformat()
    n = 0
    for r in rows:
        d = _parse_iso_date(r.get("date") or r.get("year")) or asof
        conn.execute(
            """
            INSERT OR REPLACE INTO levered_dcf (ticker, as_of_date, payload_json, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (ticker, d, json.dumps(r, ensure_ascii=False, separators=(",", ":")), now),
        )
        n += 1
    return n


def upsert_price_target(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        d = _parse_iso_date(r.get("publishedDate") or r.get("date"))
        if not d:
            continue
        company = str(r.get("analystCompany") or r.get("newsPublisher") or "").strip()
        conn.execute(
            """
            INSERT OR REPLACE INTO price_target
              (ticker, published_date, analyst_company, analyst_name, price_target,
               adj_price_target, price_when_posted, news_url, news_title, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker, d, company,
                r.get("analystName"),
                _safe_float(r.get("priceTarget")),
                _safe_float(r.get("adjPriceTarget")),
                _safe_float(r.get("priceWhenPosted")),
                r.get("newsURL") or r.get("newsUrl"),
                r.get("newsTitle"),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


def upsert_price_target_consensus(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        conn.execute(
            """
            INSERT OR REPLACE INTO price_target_consensus
              (ticker, target_high, target_low, target_consensus, target_median, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                _safe_float(r.get("targetHigh")),
                _safe_float(r.get("targetLow")),
                _safe_float(r.get("targetConsensus")),
                _safe_float(r.get("targetMedian")),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


def upsert_price_target_summary(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        publishers = r.get("publishers")
        if isinstance(publishers, (list, dict)):
            publishers = json.dumps(publishers, ensure_ascii=False)
        conn.execute(
            """
            INSERT OR REPLACE INTO price_target_summary
              (ticker, last_month, last_quarter, last_year, all_time, publishers, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                _safe_float(r.get("lastMonth") or r.get("lastMonthAvgPriceTarget")),
                _safe_float(r.get("lastQuarter") or r.get("lastQuarterAvgPriceTarget")),
                _safe_float(r.get("lastYear") or r.get("lastYearAvgPriceTarget")),
                _safe_float(r.get("allTime") or r.get("allTimeAvgPriceTarget")),
                str(publishers) if publishers else None,
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


def upsert_upgrades_downgrades(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        d = _parse_iso_date(r.get("publishedDate") or r.get("date"))
        if not d:
            continue
        company = str(r.get("gradingCompany") or r.get("newsPublisher") or "").strip()
        conn.execute(
            """
            INSERT OR REPLACE INTO upgrades_downgrades
              (ticker, published_date, grading_company, new_grade, previous_grade, action,
               price_when_posted, news_url, news_title, news_publisher, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker, d, company,
                r.get("newGrade"),
                r.get("previousGrade"),
                r.get("action"),
                _safe_float(r.get("priceWhenPosted")),
                r.get("newsURL") or r.get("newsUrl"),
                r.get("newsTitle"),
                r.get("newsPublisher") or r.get("newsBaseURL"),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


def upsert_upgrades_downgrades_consensus(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        conn.execute(
            """
            INSERT OR REPLACE INTO upgrades_downgrades_consensus
              (ticker, strong_buy, buy, hold, sell, strong_sell, consensus, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                _safe_int(r.get("strongBuy")),
                _safe_int(r.get("buy")),
                _safe_int(r.get("hold")),
                _safe_int(r.get("sell")),
                _safe_int(r.get("strongSell")),
                r.get("consensus"),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


def upsert_stock_grade(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    n = 0
    for r in rows:
        d = _parse_iso_date(r.get("date"))
        if not d:
            continue
        company = str(r.get("gradingCompany") or "").strip()
        conn.execute(
            """
            INSERT OR REPLACE INTO stock_grade
              (ticker, grade_date, grading_company, previous_grade, new_grade, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker, d, company,
                r.get("previousGrade"),
                r.get("newGrade"),
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
                now,
            ),
        )
        n += 1
    return n


def upsert_stock_peers(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    now = _utc_now_iso()
    conn.execute("DELETE FROM stock_peers WHERE ticker = ?", (ticker,))
    n = 0
    for r in rows:
        # Stable /stock-peers returns [{symbol, companyName, price, mktCap}, ...]
        # Legacy v4 returned [{peersList: ["AAPL", ...]}, ...]
        peer_sym = str(r.get("symbol") or "").strip().upper()
        if peer_sym and peer_sym != ticker:
            conn.execute(
                "INSERT OR REPLACE INTO stock_peers (ticker, peer_ticker, updated_at) VALUES (?, ?, ?)",
                (ticker, peer_sym, now),
            )
            n += 1
            continue
        # Fallback for legacy v4 format
        peers = r.get("peersList") or r.get("peers") or []
        if isinstance(peers, str):
            try:
                peers = json.loads(peers)
            except Exception:
                peers = [p.strip() for p in peers.split(",") if p.strip()]
        for p in peers:
            peer = str(p or "").strip().upper()
            if peer and peer != ticker:
                conn.execute(
                    "INSERT OR REPLACE INTO stock_peers (ticker, peer_ticker, updated_at) VALUES (?, ?, ?)",
                    (ticker, peer, now),
                )
                n += 1
    return n


# ── Orchestration ────────────────────────────────────────────────────────────

DATASETS = [
    "enterprise_values",
    "historical_market_cap",
    "dcf",
    "price_target_consensus",
    "price_target_summary",
    "stock_peers",
]

_FETCH = {
    "enterprise_values":            fetch_enterprise_values,
    "historical_market_cap":        fetch_historical_market_cap,
    "dcf":                          fetch_dcf,
    "price_target_consensus":       fetch_price_target_consensus,
    "price_target_summary":         fetch_price_target_summary,
    "stock_peers":                  fetch_stock_peers,
}

_UPSERT = {
    "enterprise_values":            upsert_enterprise_values,
    "historical_market_cap":        upsert_historical_market_cap,
    "dcf":                          upsert_dcf,
    "price_target_consensus":       upsert_price_target_consensus,
    "price_target_summary":         upsert_price_target_summary,
    "stock_peers":                  upsert_stock_peers,
}


def process_ticker(
    client: FmpClient,
    conn: sqlite3.Connection,
    ticker: str,
    *,
    max_age_hours: float,
    force: bool,
    datasets: list[str],
    sleep_s: float,
) -> dict[str, int]:
    stats: dict[str, int] = {}
    for ds in datasets:
        table = ds
        if not force and ds != "stock_peers":
            last = _ticker_last_updated(conn, table, ticker)
            if _is_fresh(last, max_age_hours):
                continue
        rows = _FETCH[ds](client, ticker)
        n = _UPSERT[ds](conn, ticker, rows)
        stats[ds] = n
        if sleep_s > 0:
            time.sleep(sleep_s)
    conn.commit()
    return stats


def run(
    *,
    watchlist_name: str = "default",
    max_age_hours: float = 24.0,
    force: bool = False,
    datasets: list[str] | None = None,
    sleep_s: float = 0.05,
    log_every: int = 10,
    api_key: str = "",
) -> None:
    key = api_key.strip() or os.getenv("FMP_API_KEY", "").strip() or FMP_API_KEY
    if not key:
        raise ValueError("FMP_API_KEY not set. Pass --api-key or export FMP_API_KEY.")

    db = init_db()
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL;")

    tickers = load_watchlist_tickers(conn, watchlist_name=watchlist_name)
    if not tickers:
        print(f"No tickers in watchlist '{watchlist_name}'. Populate fmp_watchlist first.")
        conn.close()
        return

    active = datasets or DATASETS
    unknown = [d for d in active if d not in _FETCH]
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}. Valid: {', '.join(DATASETS)}")

    client = FmpClient(key, calls_per_minute=250)
    print(f"Watchlist '{watchlist_name}': {len(tickers)} tickers")
    print(f"Datasets : {', '.join(active)}")
    print(f"max_age_hours={max_age_hours}, force={force}")

    totals: dict[str, int] = {}
    for i, ticker in enumerate(tickers, start=1):
        stats = process_ticker(
            client, conn, ticker,
            max_age_hours=max_age_hours,
            force=force,
            datasets=active,
            sleep_s=sleep_s,
        )
        for ds, cnt in stats.items():
            totals[ds] = totals.get(ds, 0) + cnt
        if i % log_every == 0 or i == len(tickers):
            print(f"  [{i:,}/{len(tickers):,}] {ticker}  rows this run: {sum(totals.values()):,}")

    conn.close()
    print("\nDone. Row totals per dataset:")
    for ds in active:
        print(f"  {ds:<38} {totals.get(ds, 0):>8,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update valuation + analyst data from FMP for watchlist stocks")
    parser.add_argument("--watchlist", default="default")
    parser.add_argument("--max-age-hours", type=float, default=24.0,
                        help="Skip ticker/dataset if fresher than this many hours (0 = always fetch)")
    parser.add_argument("--force", action="store_true", help="Re-fetch regardless of age")
    parser.add_argument("--datasets", default="",
                        help=f"Comma-separated subset (default: all). Options: {', '.join(DATASETS)}")
    parser.add_argument("--sleep", type=float, default=0.05, help="Seconds between API calls")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--api-key", default="", help="FMP API key (overrides env + hardcoded)")
    args = parser.parse_args()

    ds = [d.strip() for d in args.datasets.split(",") if d.strip()] if args.datasets.strip() else None
    run(
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
