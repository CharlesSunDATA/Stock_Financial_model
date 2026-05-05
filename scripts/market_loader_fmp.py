"""
market_loader_fmp.py

Production-style bulk loader for the US equity universe using Financial Modeling Prep (FMP).

Design goals
- High throughput without exceeding a target rate (default: 300 calls/min)
- Prefer stable BULK endpoints to minimize API calls
- Write results into a local SQLite database (data/quant_data.db)
- Idempotent upserts (INSERT OR REPLACE) on UNIQUE(ticker, report_date) constraints

What it loads
- Universe: financial statement symbol list (stable)
- Last 4 quarters:
  - Income Statement Bulk  -> fundamental_data (revenue, eps)
  - Cash Flow Statement Bulk -> fundamental_data (free_cash_flow)
  - Balance Sheet Statement Bulk -> balance_sheet
- Latest:
  - Ratios TTM Bulk -> financial_ratios (best-effort mapping)
  - Company profile: /stable/profile?symbol= -> company_profile (watchlist-driven; one symbol per call; v3 batch is legacy-only)

Run
  python3 scripts/market_loader_fmp.py
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
import csv
import io
from pathlib import Path
from typing import Any, Iterable

import requests

# Support both execution modes:
# - `streamlit run app.py` (imports from project root)
# - `python3 scripts/market_loader_fmp.py` (executes with scripts/ on sys.path)
try:
    from scripts.init_db import init_db  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from init_db import init_db  # type: ignore


try:
    from pages.modules.fmp_key import get_fmp_key
except ImportError:
    from modules.fmp_key import get_fmp_key  # type: ignore
FMP_API_KEY = get_fmp_key()

FMP_BASE = "https://financialmodelingprep.com/stable"
# Legacy v3/v4 paths (some subscriptions return 403; callers catch and skip per symbol).
FMP_V3_ROOT = "https://financialmodelingprep.com/api/v3"
FMP_V4_ROOT = "https://financialmodelingprep.com/api/v4"


def _today() -> date:
    return date.today()

def _yesterday() -> date:
    return _today() - timedelta(days=1)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _singleton_lock_path() -> Path:
    # Keeps one loader instance per machine/workspace root by default.
    return _project_root() / ".market_loader_fmp.lock"


class SingleInstanceLock:
    """
    Prevent accidental concurrent runs (e.g., two terminals, cron + manual).
    Uses an exclusive flock on a lock file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(
                f"Another market_loader_fmp.py instance appears to be running "
                f"(lock file: {self.path}). Stop the other process or delete the lock if stale."
            ) from e
        # Write pid for debugging
        try:
            self._fh.seek(0)
            self._fh.truncate(0)
            self._fh.write(f"pid={__import__('os').getpid()}\n")
            self._fh.flush()
        except Exception:
            pass

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        self._fh = None


def _db_path() -> Path:
    return _project_root() / "data" / "quant_data.db"


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


def _parse_iso_date(s: Any) -> str | None:
    if not s:
        return None
    ss = str(s)[:10]
    try:
        datetime.strptime(ss, "%Y-%m-%d")
        return ss
    except Exception:
        return None


def _add_days_iso(d: str, days: int) -> str:
    dt = datetime.strptime(d, "%Y-%m-%d").date()
    return (dt + timedelta(days=int(days))).isoformat()


@dataclass(frozen=True)
class Quarter:
    year: int
    period: str  # Q1..Q4

    def __str__(self) -> str:
        return f"{self.year}-{self.period}"


def last_n_quarters(n: int = 4, *, as_of: date | None = None) -> list[Quarter]:
    """
    Compute last N calendar quarters ending before/as_of.
    This is a pragmatic approximation for bulk endpoints.
    """
    if as_of is None:
        as_of = _today()

    q = (as_of.month - 1) // 3 + 1
    year = as_of.year

    # If currently in a quarter, the latest completed quarter may be the previous one.
    # We keep it simple: include current quarter as a target; bulk data is still useful.
    out: list[Quarter] = []
    cur_q = q
    cur_y = year
    for _ in range(max(1, n)):
        out.append(Quarter(cur_y, f"Q{cur_q}"))
        cur_q -= 1
        if cur_q == 0:
            cur_q = 4
            cur_y -= 1
    return out


def _ensure_backfill_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fmp_bulk_backfill (
          dataset TEXT NOT NULL,
          year INTEGER NOT NULL,
          period TEXT NOT NULL,
          completed_at TEXT NOT NULL,
          rows_received INTEGER,
          PRIMARY KEY (dataset, year, period)
        );
        """
    )


def _is_quarter_done(conn: sqlite3.Connection, dataset: str, q: Quarter) -> bool:
    _ensure_backfill_table(conn)
    cur = conn.execute(
        "SELECT 1 FROM fmp_bulk_backfill WHERE dataset = ? AND year = ? AND period = ? LIMIT 1",
        (dataset, int(q.year), str(q.period)),
    )
    return cur.fetchone() is not None


def _mark_quarter_done(conn: sqlite3.Connection, dataset: str, q: Quarter, rows_received: int) -> None:
    _ensure_backfill_table(conn)
    conn.execute(
        """
        INSERT OR REPLACE INTO fmp_bulk_backfill (dataset, year, period, completed_at, rows_received)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            dataset,
            int(q.year),
            str(q.period),
            datetime.utcnow().replace(tzinfo=None).isoformat(timespec="seconds"),
            int(rows_received),
        ),
    )


def _ensure_daily_backfill_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fmp_daily_backfill (
          dataset TEXT NOT NULL,
          as_of_date TEXT NOT NULL,
          completed_at TEXT NOT NULL,
          rows_received INTEGER,
          PRIMARY KEY (dataset, as_of_date)
        );
        """
    )


def _is_day_done(conn: sqlite3.Connection, dataset: str, day: date) -> bool:
    _ensure_daily_backfill_table(conn)
    cur = conn.execute(
        "SELECT 1 FROM fmp_daily_backfill WHERE dataset = ? AND as_of_date = ? LIMIT 1",
        (dataset, day.isoformat()),
    )
    return cur.fetchone() is not None


def _mark_day_done(conn: sqlite3.Connection, dataset: str, day: date, rows_received: int) -> None:
    _ensure_daily_backfill_table(conn)
    conn.execute(
        """
        INSERT OR REPLACE INTO fmp_daily_backfill (dataset, as_of_date, completed_at, rows_received)
        VALUES (?, ?, ?, ?)
        """,
        (
            dataset,
            day.isoformat(),
            datetime.utcnow().replace(tzinfo=None).isoformat(timespec="seconds"),
            int(rows_received),
        ),
    )


class RateLimiter:
    """Token bucket rate limiter (calls per minute)."""

    def __init__(self, calls_per_minute: int) -> None:
        self.capacity = float(max(1, calls_per_minute))
        self.tokens = float(self.capacity)
        self.refill_per_sec = float(self.capacity) / 60.0
        self.last = time.monotonic()

    def acquire(self, tokens: float = 1.0) -> None:
        while True:
            now = time.monotonic()
            elapsed = max(0.0, now - self.last)
            self.last = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            need = tokens - self.tokens
            sleep_s = max(0.01, need / self.refill_per_sec)
            time.sleep(sleep_s)


class FmpClient:
    def __init__(self, api_key: str, *, calls_per_minute: int = 300) -> None:
        self.api_key = api_key.strip()
        self.session = requests.Session()
        self.limiter = RateLimiter(calls_per_minute)

    def get_data(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: int = 45,
        max_retries: int = 6,
        base_backoff_s: float = 5.0,
    ) -> Any:
        """
        GET data with:
        - rate limiting (calls/min)
        - 429 backoff retries (bulk endpoints can be aggressively throttled)
        - defensive parsing:
          - JSON (application/json)
          - CSV (text/csv) for bulk endpoints
        """
        p = dict(params or {})
        p["apikey"] = self.api_key

        for attempt in range(max_retries + 1):
            self.limiter.acquire(1.0)
            r = self.session.get(url, params=p, timeout=timeout_s)

            if r.status_code == 429:
                wait_s = min(120.0, base_backoff_s * (2.0**attempt))
                msg = ""
                try:
                    msg = r.text[:200].replace("\n", " ")
                except Exception:
                    msg = ""
                if attempt >= max_retries:
                    raise requests.HTTPError(f"429 Too Many Requests after retries | body={msg}")
                time.sleep(wait_s)
                continue

            try:
                r.raise_for_status()
            except requests.HTTPError as e:
                body = ""
                try:
                    body = r.text[:2000]
                except Exception:
                    body = ""
                raise requests.HTTPError(f"{e} | response_body={body}") from e

            ctype = (r.headers.get("content-type") or "").lower()

            # Bulk endpoints commonly return CSV.
            if "text/csv" in ctype:
                buf = io.StringIO(r.text)
                reader = csv.DictReader(buf)
                return [row for row in reader]

            # Otherwise treat as JSON (some endpoints can return non-standard bodies).
            try:
                return r.json()
            except requests.exceptions.JSONDecodeError:
                text = ""
                try:
                    text = r.text
                except Exception:
                    text = ""
                if not text:
                    raise
                dec = json.JSONDecoder()
                try:
                    obj, _idx = dec.raw_decode(text.lstrip())
                    return obj
                except Exception as e:
                    snippet = text[:2000].replace("\n", "\\n")
                    raise ValueError(f"Non-JSON response body (first 2000 chars): {snippet}") from e


def fetch_financial_statement_symbol_list(client: FmpClient) -> list[str]:
    """
    Stable endpoint: Financial Statement Symbols List
    """
    url = f"{FMP_BASE}/financial-statement-symbol-list"
    data = client.get_data(url)
    if not isinstance(data, list):
        return []
    syms: list[str] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        s = str(row.get("symbol") or "").strip().upper()
        if s:
            syms.append(s)
    return sorted(set(syms))


def fetch_sp500_constituents(client: FmpClient) -> list[str]:
    url = f"{FMP_BASE}/sp500-constituent"
    data = client.get_data(url)
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        s = str(row.get("symbol") or "").strip().upper()
        if s:
            out.append(s)
    return sorted(set(out))


def fetch_nasdaq_constituents(client: FmpClient) -> list[str]:
    url = f"{FMP_BASE}/nasdaq-constituent"
    data = client.get_data(url)
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        s = str(row.get("symbol") or "").strip().upper()
        if s:
            out.append(s)
    return sorted(set(out))


def fetch_etf_equity_holdings_symbols(client: FmpClient, etf_symbol: str) -> list[str]:
    """
    Uses: GET /stable/etf/holdings?symbol=ETF
    Extracts equity-like symbols from holdings rows (best-effort).
    """
    sym = (etf_symbol or "").strip().upper()
    if not sym:
        return []
    url = f"{FMP_BASE}/etf/holdings"
    data = client.get_data(url, params={"symbol": sym})
    if not isinstance(data, list):
        return []

    etf = sym
    out: list[str] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        # FMP ETF holdings commonly encode the *holding* ticker in `asset`, while `symbol` is the ETF itself.
        # Example row:
        # {"symbol":"QQQ","asset":"NVDA", ...}
        s = (
            row.get("asset")
            or row.get("underlyingSymbol")
            or row.get("holdingSymbol")
            or row.get("stockSymbol")
            or row.get("assetSymbol")
            or row.get("ticker")
            or row.get("symbol")
        )
        ss = str(s or "").strip().upper()
        if not ss:
            continue
        if ss == etf:
            continue
        # Filter obvious non-equities
        if ss in {"CASH", "USD", "MMDA", "MONEY MARKET"}:
            continue
        # Many ETFs include bonds/futures as odd symbols; keep only plausible tickers.
        if len(ss) > 12:
            continue
        out.append(ss)
    return sorted(set(out))


def fetch_company_screener(client: FmpClient, params: dict[str, Any]) -> list[dict[str, Any]]:
    url = f"{FMP_BASE}/company-screener"
    data = client.get_data(url, params=params)
    return data if isinstance(data, list) else []


def _exchange_allowed(exchange_short: str | None, exchange: str | None) -> bool:
    """
    Allow major US listing venues; exclude OTC-ish exchanges by denylist.
    """
    x = (exchange_short or exchange or "").strip().upper()
    if not x:
        return False
    deny = {"OTC", "OTCM", "OTCQB", "OTCQX", "PINK", "GREY", "OTCN"}
    if x in deny:
        return False
    allow = {"NASDAQ", "NYSE", "AMEX", "NYSEAMERICAN", "NYSE ARCA", "ARCA"}
    return x in allow


def screen_us_liquid_equities(
    client: FmpClient,
    *,
    price_min: float,
    market_cap_min: float,
    limit_per_exchange: int = 5000,
) -> list[str]:
    """
    Pull a broad US equity pool from company-screener and apply exchange filtering.
    Note: Some accounts may not support `exchange` query filtering; we post-filter using fields.
    """
    syms: set[str] = set()
    # Pull per major exchange to reduce payload issues and improve determinism.
    for exch in ("NASDAQ", "NYSE", "AMEX"):
        rows = fetch_company_screener(
            client,
            {
                "country": "US",
                "isActivelyTrading": True,
                "isEtf": False,
                "isFund": False,
                "priceMoreThan": float(price_min),
                "marketCapMoreThan": int(market_cap_min),
                "exchange": exch,
                "limit": int(limit_per_exchange),
            },
        )
        for r in rows:
            if not isinstance(r, dict):
                continue
            sym = str(r.get("symbol") or "").strip().upper()
            if not sym:
                continue
            es = r.get("exchangeShortName")
            ex = r.get("exchange")
            if not _exchange_allowed(str(es) if es is not None else None, str(ex) if ex is not None else None):
                continue
            syms.add(sym)
    return sorted(syms)


def avg_dollar_volume_usd(
    client: FmpClient,
    *,
    symbol: str,
    last_n_trading_days: int,
    calendar_lookback_days: int,
) -> float | None:
    """
    Approximate ADV$ using recent daily bars from historical-price-eod/full.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return None
    end = _yesterday()
    start = end - timedelta(days=max(7, int(calendar_lookback_days)))
    rows = fetch_historical_price_eod_full(client, symbol=sym, date_from=start.isoformat(), date_to=end.isoformat())
    if not isinstance(rows, list) or not rows:
        return None

    bars: list[tuple[str, float, float]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = _parse_iso_date(r.get("date"))
        if not d:
            continue
        px = _safe_float(r.get("close"))
        vol = _safe_float(r.get("volume"))
        if px is None or vol is None:
            continue
        bars.append((d, float(px), float(vol)))
    bars.sort(key=lambda x: x[0])
    if len(bars) < int(last_n_trading_days):
        return None
    tail = bars[-int(last_n_trading_days) :]
    dv = sum(px * vol for (_d, px, vol) in tail) / float(len(tail))
    return float(dv)


def filter_symbols_by_adv(
    client: FmpClient,
    symbols: list[str],
    *,
    min_adv_usd: float,
    adv_days: int,
    adv_lookback_days: int,
    max_symbols: int | None = None,
    sleep_s: float = 0.05,
) -> list[str]:
    """
    Keep symbols whose rolling average dollar volume meets threshold.
    """
    out: list[str] = []
    cap = int(max_symbols) if max_symbols is not None else len(symbols)
    for i, sym in enumerate(symbols, start=1):
        if len(out) >= cap:
            break
        adv = avg_dollar_volume_usd(
            client,
            symbol=sym,
            last_n_trading_days=int(adv_days),
            calendar_lookback_days=int(adv_lookback_days),
        )
        if adv is None:
            continue
        if adv >= float(min_adv_usd):
            out.append(sym)
        if sleep_s > 0:
            time.sleep(float(sleep_s))
        if i % 250 == 0:
            print(f"ADV filter progress: {i:,}/{len(symbols):,} scanned, kept {len(out):,}")
    return sorted(set(out))


def build_strategy_universe_symbols(
    client: FmpClient,
    *,
    strategy: str,
    ndx_etf: str,
    russell1000_etf: str,
    russell3000_etf: str,
    price_min: float,
    market_cap_min: float,
    adv_min_usd: float,
    adv_days: int,
    adv_lookback_days: int,
    adv_max_candidates: int | None,
) -> list[str]:
    """
    Implements four common research universes:
    - sp500
    - russell1000 (via ETF holdings proxy)
    - russell3000 (via ETF holdings proxy)
    - ndx (via ETF holdings proxy; default QQQ)

    Plus a convenience combo:
    - sp500_ndx

    After forming the raw basket, apply liquidity / listing hygiene filters via screener + ADV$.
    """
    s = (strategy or "").strip().lower()
    raw: set[str] = set()

    if s in ("sp500", "sp500_ndx"):
        raw.update(fetch_sp500_constituents(client))
    if s in ("ndx", "sp500_ndx"):
        raw.update(fetch_etf_equity_holdings_symbols(client, ndx_etf))
    if s in ("russell1000",):
        raw.update(fetch_etf_equity_holdings_symbols(client, russell1000_etf))
    if s in ("russell3000",):
        raw.update(fetch_etf_equity_holdings_symbols(client, russell3000_etf))

    raw_list = sorted(raw)
    print(f"Strategy '{s}' raw symbols: {len(raw_list):,}")

    # Stage 1: exchange / microcap / penny filters via screener universe membership (cheap bulk screen).
    screened = set(screen_us_liquid_equities(client, price_min=price_min, market_cap_min=market_cap_min))
    staged = [sym for sym in raw_list if sym in screened]
    print(f"After screener intersection: {len(staged):,}")

    # Stage 2: ADV$ filter (more faithful than single-day volume fields).
    # NOTE: This is API-heavy; cap candidates if requested.
    candidates = staged
    if adv_max_candidates is not None:
        candidates = staged[: int(adv_max_candidates)]

    kept = filter_symbols_by_adv(
        client,
        candidates,
        min_adv_usd=adv_min_usd,
        adv_days=adv_days,
        adv_lookback_days=adv_lookback_days,
        max_symbols=None,
        sleep_s=0.03,
    )

    # If we capped candidates, keep the remainder unstaged for transparency (do not silently drop).
    if adv_max_candidates is not None and len(staged) > len(candidates):
        print(f"ADV scan capped to first {len(candidates):,} candidates (of {len(staged):,}). Re-run to scan more.")

    print(f"After ADV$ filter (>={adv_min_usd:,.0f} on {adv_days}d avg): {len(kept):,}")
    return sorted(set(kept))


def parse_strategy_list(s: str) -> list[str]:
    raw = (s or "").strip()
    if not raw:
        return []
    out: list[str] = []
    for p in raw.replace(";", ",").split(","):
        pp = p.strip().lower()
        if pp:
            out.append(pp)
    return out


def build_multi_strategy_universe_symbols(
    client: FmpClient,
    *,
    strategies: list[str],
    ndx_etf: str,
    russell1000_etf: str,
    russell3000_etf: str,
    price_min: float,
    market_cap_min: float,
    adv_min_usd: float,
    adv_days: int,
    adv_lookback_days: int,
    adv_max_candidates: int | None,
) -> list[str]:
    """
    Run multiple strategies and union/deduplicate the resulting symbol sets.
    """
    allowed = {"sp500", "russell1000", "russell3000", "ndx", "sp500_ndx"}
    cleaned = [x for x in strategies if x]
    unknown = sorted({x for x in cleaned if x not in allowed})
    if unknown:
        raise ValueError(f"Unknown universe-strategy tokens: {', '.join(unknown)}. Allowed: {', '.join(sorted(allowed))}")

    uniq_strats = []
    seen: set[str] = set()
    for x in cleaned:
        if x not in seen:
            uniq_strats.append(x)
            seen.add(x)

    union: set[str] = set()
    print(f"Multi-strategy union requested ({len(uniq_strats)}): {', '.join(uniq_strats)}")
    for st in uniq_strats:
        part = build_strategy_universe_symbols(
            client,
            strategy=st,
            ndx_etf=ndx_etf,
            russell1000_etf=russell1000_etf,
            russell3000_etf=russell3000_etf,
            price_min=price_min,
            market_cap_min=market_cap_min,
            adv_min_usd=adv_min_usd,
            adv_days=adv_days,
            adv_lookback_days=adv_lookback_days,
            adv_max_candidates=adv_max_candidates,
        )
        union.update(part)

    out = sorted(union)
    print(f"Union universe size (unique symbols): {len(out):,}")
    return out


def parse_universe_tokens(s: str) -> list[str]:
    raw = (s or "").strip()
    if not raw:
        return []
    parts: list[str] = []
    for p in raw.replace(";", ",").split(","):
        pp = p.strip()
        if pp:
            parts.append(pp.lower())
    return parts


def build_custom_universe_symbols(
    client: FmpClient,
    *,
    tokens: list[str],
    ndx_etf: str,
    semi_etfs: list[str],
) -> list[str]:
    """
    Supported tokens (comma-separated):
    - sp500
    - nasdaq / nasdaq-listed / nasdaq_total (all Nasdaq-listed constituents via nasdaq-constituent)
    - ndx / nasdaq100 / qqq (holdings of ndx_etf, default QQQ)
    - semi / semis / sox / philly-sox (union of semi_etfs holdings, default SMH,SOXX)
    """
    syms: set[str] = set()
    ndx_etf_u = (ndx_etf or "QQQ").strip().upper()

    semi_syms: list[str] = []
    for etf in semi_etfs:
        etf_u = (etf or "").strip().upper()
        if etf_u:
            semi_syms.append(etf_u)

    for t in tokens:
        if t in ("sp500", "s&p500", "spx"):
            syms.update(fetch_sp500_constituents(client))
        elif t in ("nasdaq", "nasdaq-listed", "nasdaq_total", "nasdaq-all"):
            syms.update(fetch_nasdaq_constituents(client))
        elif t in ("ndx", "nasdaq100", "nasdaq-100", "qqq"):
            syms.update(fetch_etf_equity_holdings_symbols(client, ndx_etf_u))
        elif t in ("semi", "semis", "sox", "philly-sox", "phlx-sox"):
            for etf_u in semi_syms:
                syms.update(fetch_etf_equity_holdings_symbols(client, etf_u))
        else:
            # Allow passing raw ETF tickers directly (e.g., SMH)
            if t.isalpha() and 1 <= len(t) <= 5:
                syms.update(fetch_etf_equity_holdings_symbols(client, t.upper()))

    return sorted(syms)


def replace_watchlist(conn: sqlite3.Connection, *, watchlist_name: str, tickers: Iterable[str]) -> int:
    nm = (watchlist_name or "").strip() or "default"
    now = datetime.utcnow().replace(tzinfo=None).isoformat(timespec="seconds")
    conn.execute("DELETE FROM fmp_watchlist WHERE watchlist_name = ?", (nm,))
    rows = [(nm, (t or "").strip().upper(), now) for t in tickers if (t or "").strip()]
    conn.executemany(
        "INSERT OR REPLACE INTO fmp_watchlist (watchlist_name, ticker, updated_at) VALUES (?, ?, ?)",
        rows,
    )
    return len(rows)


def load_watchlist_tickers(conn: sqlite3.Connection, *, watchlist_name: str) -> list[str]:
    """Tickers in fmp_watchlist for a given list name (uppercased, stable order)."""
    nm = (watchlist_name or "").strip() or "default"
    return [
        str(r[0]).strip().upper()
        for r in conn.execute(
            "SELECT ticker FROM fmp_watchlist WHERE watchlist_name = ? ORDER BY ticker ASC",
            (nm,),
        ).fetchall()
        if str(r[0]).strip()
    ]


def ensure_fmp_watchlist_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fmp_watchlist (
          watchlist_name TEXT NOT NULL,
          ticker TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (watchlist_name, ticker)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fmp_watchlist_manifest (
          watchlist_name TEXT PRIMARY KEY,
          fingerprint TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          symbol_count INTEGER NOT NULL
        );
        """
    )


def _parse_watchlist_timestamp(s: str | None) -> datetime | None:
    if not s:
        return None
    t = str(s).strip()
    if not t:
        return None
    if t.endswith("Z"):
        t = t[:-1]
    try:
        return datetime.fromisoformat(t)
    except ValueError:
        return None


def build_strategy_universe_fingerprint(
    *,
    strategies: list[str],
    ndx_etf: str,
    russell1000_etf: str,
    russell3000_etf: str,
    price_min: float,
    market_cap_min: float,
    adv_min_usd: float,
    adv_days: int,
    adv_lookback_days: int,
    adv_max_candidates: int | None,
) -> str:
    adv_part = "none" if adv_max_candidates is None else str(int(adv_max_candidates))
    blob = "|".join(
        [
            "strategy_universe_v1",
            ",".join(sorted(str(x).strip().lower() for x in strategies if str(x).strip())),
            (ndx_etf or "").strip().upper(),
            (russell1000_etf or "").strip().upper(),
            (russell3000_etf or "").strip().upper(),
            f"{float(price_min):.12g}",
            f"{float(market_cap_min):.12g}",
            f"{float(adv_min_usd):.12g}",
            str(int(adv_days)),
            str(int(adv_lookback_days)),
            adv_part,
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_custom_universe_fingerprint(*, tokens: list[str], ndx_etf: str, semi_etfs: list[str]) -> str:
    blob = "|".join(
        [
            "custom_universe_v1",
            ",".join(sorted(str(t).strip().lower() for t in tokens if str(t).strip())),
            (ndx_etf or "").strip().upper(),
            ",".join(sorted((e or "").strip().upper() for e in semi_etfs if (e or "").strip())),
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def upsert_watchlist_manifest(
    conn: sqlite3.Connection, *, watchlist_name: str, fingerprint: str, symbol_count: int
) -> None:
    nm = (watchlist_name or "").strip() or "default"
    now = datetime.utcnow().replace(tzinfo=None).isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO fmp_watchlist_manifest (watchlist_name, fingerprint, updated_at, symbol_count)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(watchlist_name) DO UPDATE SET
          fingerprint = excluded.fingerprint,
          updated_at = excluded.updated_at,
          symbol_count = excluded.symbol_count
        ;
        """,
        (nm, fingerprint, now, int(symbol_count)),
    )


def maybe_load_cached_watchlist(
    conn: sqlite3.Connection,
    *,
    watchlist_name: str,
    fingerprint: str,
    max_age_hours: float,
    force_rebuild: bool,
) -> tuple[bool, list[str]]:
    """
    If manifest fingerprint matches, row count matches fmp_watchlist, and build age <= max_age_hours,
    return tickers from fmp_watchlist without calling screener/ADV (or custom ETF fetches).
    Set max_age_hours <= 0 or force_rebuild to disable reuse.
    """
    if force_rebuild or float(max_age_hours) <= 0:
        return False, []
    nm = (watchlist_name or "").strip() or "default"
    row = conn.execute(
        "SELECT fingerprint, updated_at, symbol_count FROM fmp_watchlist_manifest WHERE watchlist_name = ?",
        (nm,),
    ).fetchone()
    if row is None:
        return False, []
    stored_fp = str(row[0] or "")
    updated_at_s = str(row[1] or "")
    sym_count = int(row[2] or 0)
    if not stored_fp or stored_fp != fingerprint or sym_count <= 0:
        return False, []
    built_at = _parse_watchlist_timestamp(updated_at_s)
    if built_at is None:
        return False, []
    age_hours = (datetime.utcnow() - built_at).total_seconds() / 3600.0
    if age_hours > float(max_age_hours):
        return False, []
    tickers = load_watchlist_tickers(conn, watchlist_name=nm)
    if len(tickers) != sym_count:
        return False, []
    print(
        f"Reusing fmp_watchlist '{nm}' ({len(tickers):,} symbols, built {age_hours:.1f}h ago <= "
        f"{float(max_age_hours):g}h). Skipping universe API build (screener / ADV or custom fetches)."
    )
    return True, tickers


def fetch_income_statement_bulk(client: FmpClient, q: Quarter) -> list[dict[str, Any]]:
    url = f"{FMP_BASE}/income-statement-bulk"
    data = client.get_data(url, params={"year": q.year, "period": q.period})
    return data if isinstance(data, list) else []


def fetch_cash_flow_bulk(client: FmpClient, q: Quarter) -> list[dict[str, Any]]:
    url = f"{FMP_BASE}/cash-flow-statement-bulk"
    data = client.get_data(url, params={"year": q.year, "period": q.period})
    return data if isinstance(data, list) else []


def fetch_balance_sheet_bulk(client: FmpClient, q: Quarter) -> list[dict[str, Any]]:
    url = f"{FMP_BASE}/balance-sheet-statement-bulk"
    data = client.get_data(url, params={"year": q.year, "period": q.period})
    return data if isinstance(data, list) else []


def fetch_ratios_ttm_bulk(client: FmpClient) -> list[dict[str, Any]]:
    url = f"{FMP_BASE}/ratios-ttm-bulk"
    data = client.get_data(url)
    return data if isinstance(data, list) else []


def fetch_key_metrics_ttm_bulk(client: FmpClient) -> list[dict[str, Any]]:
    url = f"{FMP_BASE}/key-metrics-ttm-bulk"
    data = client.get_data(url)
    return data if isinstance(data, list) else []


def fetch_eod_bulk(client: FmpClient, *, day: date) -> list[dict[str, Any]]:
    """
    Stable endpoint: /stable/eod-bulk?date=YYYY-MM-DD
    Often returns CSV.
    """
    url = f"{FMP_BASE}/eod-bulk"
    data = client.get_data(url, params={"date": day.isoformat()})
    return data if isinstance(data, list) else []


def fetch_analyst_estimates(
    client: FmpClient,
    *,
    symbol: str,
    period: str = "quarter",
    limit: int = 20,
    page: int = 0,
) -> list[dict[str, Any]]:
    """
    Stable endpoint (non-bulk): /stable/analyst-estimates
    Example (per docs): ?symbol=AAPL&period=annual&page=0&limit=10
    """
    url = f"{FMP_BASE}/analyst-estimates"
    data = client.get_data(
        url,
        params={"symbol": symbol, "period": period, "page": int(page), "limit": int(limit)},
        timeout_s=45,
    )
    return data if isinstance(data, list) else []

def fetch_historical_price_eod_full(
    client: FmpClient,
    *,
    symbol: str,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """
    Stable endpoint (per-symbol): /stable/historical-price-eod/full
    Supports optional from/to query parameters.
    """
    url = f"{FMP_BASE}/historical-price-eod/full"
    params: dict[str, Any] = {"symbol": symbol}
    if date_from:
        params["from"] = date_from
    if date_to:
        params["to"] = date_to
    data = client.get_data(url, params=params, timeout_s=45)
    return data if isinstance(data, list) else []


def fetch_profiles_stable_for_watchlist(
    client: FmpClient,
    symbols: list[str],
    *,
    log_every: int = 100,
    sleep_s: float = 0.03,
) -> list[dict[str, Any]]:
    """
    GET /stable/profile?symbol=TICKER (one symbol per request).

    FMP has disabled legacy /api/v3/profile batch for many newer subscriptions (403). Stable profile
    is per-symbol only in public docs; we still batch *progress logging* via log_every.
    """
    url = f"{FMP_BASE}/profile"
    clean = sorted({(s or "").strip().upper() for s in symbols if (s or "").strip()})
    out: list[dict[str, Any]] = []
    n = len(clean)
    if n == 0:
        return out
    log_every = max(1, int(log_every))
    for i, sym in enumerate(clean, start=1):
        try:
            data = client.get_data(url, params={"symbol": sym}, timeout_s=45)
        except requests.HTTPError as e:
            print(f"    WARN: profile stable failed for {sym}: {e}")
            if sleep_s > 0:
                time.sleep(float(sleep_s))
            continue
        rows: list[dict[str, Any]] = []
        if isinstance(data, list):
            rows = [r for r in data if isinstance(r, dict)]
        elif isinstance(data, dict):
            if data.get("Error Message") or data.get("error"):
                print(f"    WARN: profile stable error payload for {sym}: {str(data.get('Error Message') or data)[:200]}")
            elif data.get("symbol") or data.get("ticker"):
                rows = [data]
        out.extend(rows)
        if i % log_every == 0 or i == n:
            print(f"  profile stable: {i:,}/{n:,} symbols requested, {len(out):,} profiles stored so far")
        if sleep_s > 0:
            time.sleep(float(sleep_s))
    return out


def fetch_historical_ratios_for_watchlist(
    client: FmpClient,
    conn: sqlite3.Connection,
    symbols: list[str],
    *,
    period: str = "quarter",
    limit: int = 40,
    log_every: int = 100,
    sleep_s: float = 0.03,
) -> int:
    """
    Fetch per-symbol historical financial ratios and upsert into historical_financial_ratios.
    Tries v3 /ratios/{sym}?period=quarter first; falls back to stable /stable/ratios?symbol=.
    Returns total rows upserted.
    """
    clean = sorted({(s or "").strip().upper() for s in symbols if (s or "").strip()})
    n_total = len(clean)
    if n_total == 0:
        return 0
    halt_v3 = False
    log_every = max(1, int(log_every))
    total_rows = 0
    per = (period or "quarter").strip().lower()
    if per not in ("quarter", "annual"):
        per = "quarter"

    for i, sym in enumerate(clean, start=1):
        rows_upserted = 0
        if not halt_v3:
            try:
                url = f"{FMP_V3_ROOT}/ratios/{sym}"
                data = client.get_data(url, params={"period": per, "limit": int(limit)}, timeout_s=60)
                rows_upserted = upsert_historical_ratios_response(conn, ticker=sym, period=per, data=data)
            except Exception as e:
                if _is_legacy_endpoint_forbidden(e):
                    print(
                        "INFO: v3 /ratios blocked (legacy-only for your plan). "
                        "Falling back to stable /ratios for all remaining symbols."
                    )
                    halt_v3 = True
                elif i % log_every == 0 or i <= 3:
                    print(f"    WARN: historical-ratios v3 failed for {sym}: {e}")
        if halt_v3:
            try:
                url = f"{FMP_BASE}/ratios"
                data = client.get_data(url, params={"symbol": sym, "limit": int(limit)}, timeout_s=60)
                rows_upserted = upsert_historical_ratios_response(conn, ticker=sym, period="annual", data=data)
            except Exception as e:
                if i % log_every == 0 or i <= 3:
                    print(f"    WARN: historical-ratios stable failed for {sym}: {e}")

        total_rows += rows_upserted
        try:
            conn.commit()
        except Exception as e:
            print(f"    WARN commit after {sym}: {e}")

        if i % log_every == 0 or i == n_total:
            print(f"  historical-ratios: {i:,}/{n_total:,} symbols processed, {total_rows:,} rows so far")

        if sleep_s > 0:
            time.sleep(float(sleep_s))

    return total_rows


def upsert_fundamentals_from_bulk(
    conn: sqlite3.Connection,
    *,
    income_rows: Iterable[dict[str, Any]] = (),
    cashflow_rows: Iterable[dict[str, Any]] = (),
) -> int:
    """
    Upsert fundamental_data by merging income + cashflow on (symbol, date).
    """
    inc: dict[tuple[str, str], dict[str, Any]] = {}
    cf: dict[tuple[str, str], dict[str, Any]] = {}

    for r in income_rows:
        if not isinstance(r, dict):
            continue
        sym = str(r.get("symbol") or r.get("ticker") or "").strip().upper()
        d = _parse_iso_date(r.get("date"))
        if sym and d:
            inc[(sym, d)] = r

    for r in cashflow_rows:
        if not isinstance(r, dict):
            continue
        sym = str(r.get("symbol") or r.get("ticker") or "").strip().upper()
        d = _parse_iso_date(r.get("date"))
        if sym and d:
            cf[(sym, d)] = r

    keys = sorted(set(inc.keys()) | set(cf.keys()))
    n = 0
    for sym, d in keys:
        r_inc = inc.get((sym, d), {})
        r_cf = cf.get((sym, d), {})
        revenue = _safe_float(r_inc.get("revenue"))
        eps = _safe_float(r_inc.get("epsdiluted") if "epsdiluted" in r_inc else r_inc.get("epsDiluted"))
        fcf = _safe_float(r_cf.get("freeCashFlow"))

        conn.execute(
            """
            INSERT OR REPLACE INTO fundamental_data (ticker, report_date, revenue, eps, free_cash_flow)
            VALUES (?, ?, ?, ?, ?)
            """,
            (sym, d, revenue, eps, fcf),
        )
        n += 1
    return n


def upsert_balance_sheet_from_bulk(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        sym = str(r.get("symbol") or "").strip().upper()
        d = _parse_iso_date(r.get("date"))
        if not sym or not d:
            continue
        cash = r.get("cashAndCashEquivalents") or r.get("cashAndShortTermInvestments")
        shares = r.get("commonStockSharesOutstanding")
        total_debt = r.get("totalDebt")
        if total_debt is None:
            st_debt = _safe_float(r.get("shortTermDebt"))
            lt_debt = _safe_float(r.get("longTermDebt"))
            if st_debt is not None or lt_debt is not None:
                total_debt = (st_debt or 0.0) + (lt_debt or 0.0)
        conn.execute(
            """
            INSERT OR REPLACE INTO balance_sheet
              (ticker, report_date, cash_and_equivalents, total_assets, total_liabilities, total_debt, outstanding_shares)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sym,
                d,
                _safe_float(cash),
                _safe_float(r.get("totalAssets")),
                _safe_float(r.get("totalLiabilities")),
                _safe_float(total_debt),
                _safe_float(shares),
            ),
        )
        n += 1
    return n


def upsert_ratios_ttm(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    """
    Map TTM ratios into financial_ratios using today's date as report_date
    (TTM is not tied to a single quarter end; this is a pragmatic store-as-of approach).
    """
    asof = _today().isoformat()
    n = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        sym = str(r.get("symbol") or "").strip().upper()
        if not sym:
            continue

        roe = r.get("returnOnEquityTTM") or r.get("returnOnEquity")
        roic = r.get("returnOnInvestedCapitalTTM") or r.get("roicTTM") or r.get("roic")
        pe = r.get("priceEarningsRatioTTM") or r.get("peRatioTTM") or r.get("priceEarningsRatio")
        pb = r.get("priceToBookRatioTTM") or r.get("pbRatioTTM") or r.get("priceToBookRatio")
        ev_ebitda = r.get("enterpriseValueMultipleTTM") or r.get("evToEbitdaTTM") or r.get("enterpriseValueMultiple")

        conn.execute(
            """
            INSERT OR REPLACE INTO financial_ratios
              (ticker, report_date, roe, roic, pe_ratio, pb_ratio, ev_to_ebitda)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sym,
                asof,
                _safe_float(roe),
                _safe_float(roic),
                _safe_float(pe),
                _safe_float(pb),
                _safe_float(ev_ebitda),
            ),
        )
        n += 1
    return n


def upsert_profiles(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        sym = str(r.get("symbol") or "").strip().upper()
        if not sym:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO company_profile
              (ticker, company_name, sector, industry, full_time_employees, image_url)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                sym,
                r.get("companyName"),
                r.get("sector"),
                r.get("industry"),
                _safe_int(r.get("fullTimeEmployees")),
                r.get("image"),
            ),
        )
        n += 1
    return n


def upsert_key_metrics_ttm(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    """
    Store key metrics TTM as-of today. The endpoint is non-filterable and returns
    the latest TTM metrics per company.
    """
    asof = _today().isoformat()
    n = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        sym = str(r.get("symbol") or "").strip().upper()
        if not sym:
            continue

        market_cap = r.get("marketCapTTM") or r.get("marketCap")
        ev = r.get("enterpriseValueTTM") or r.get("enterpriseValue")
        ev_sales = r.get("evToSalesTTM") or r.get("enterpriseValueOverRevenueTTM") or r.get("evToSales")
        ev_ocf = r.get("evToOperatingCashFlowTTM") or r.get("enterpriseValueOverOperatingCashFlowTTM")
        ev_fcf = r.get("evToFreeCashFlowTTM") or r.get("enterpriseValueOverFreeCashFlowTTM")
        net_debt_ebitda = r.get("netDebtToEBITDATTM") or r.get("netDebtToEbitdaTTM") or r.get("netDebtToEBITDA")

        payload = None
        try:
            payload = json.dumps(r, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            payload = None

        conn.execute(
            """
            INSERT OR REPLACE INTO key_metrics_ttm
              (ticker, as_of_date, market_cap, enterprise_value, ev_to_sales, ev_to_operating_cash_flow,
               ev_to_free_cash_flow, net_debt_to_ebitda, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sym,
                asof,
                _safe_float(market_cap),
                _safe_float(ev),
                _safe_float(ev_sales),
                _safe_float(ev_ocf),
                _safe_float(ev_fcf),
                _safe_float(net_debt_ebitda),
                payload,
            ),
        )
        n += 1
    return n


def upsert_prices_eod(conn: sqlite3.Connection, *, day: date, rows: Iterable[dict[str, Any]]) -> int:
    """
    Upsert a single-day EOD bulk snapshot into prices_eod.
    Handles both JSON and CSV dict keys (case/alias tolerant).
    """
    d = day.isoformat()
    n = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        sym = str(r.get("symbol") or r.get("Symbol") or r.get("ticker") or "").strip().upper()
        if not sym:
            continue
        # Prefer fields commonly returned by FMP EOD datasets.
        o = r.get("open") or r.get("Open")
        h = r.get("high") or r.get("High")
        l = r.get("low") or r.get("Low")
        c = r.get("close") or r.get("Close") or r.get("price")
        adj = r.get("adjClose") or r.get("adj_close") or r.get("Adj Close") or r.get("adjClosePrice")
        vol = r.get("volume") or r.get("Volume")

        conn.execute(
            """
            INSERT OR REPLACE INTO prices_eod
              (ticker, price_date, open, high, low, close, adj_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sym,
                d,
                _safe_float(o),
                _safe_float(h),
                _safe_float(l),
                _safe_float(c),
                _safe_float(adj),
                _safe_float(vol),
            ),
        )
        n += 1
    return n


def upsert_prices_progress(conn: sqlite3.Connection, *, ticker: str, last_price_date: str | None) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO prices_backfill_progress (ticker, last_price_date, updated_at)
        VALUES (?, ?, ?)
        """,
        (
            (ticker or "").strip().upper(),
            last_price_date,
            datetime.utcnow().replace(tzinfo=None).isoformat(timespec="seconds"),
        ),
    )


def get_prices_progress(conn: sqlite3.Connection, *, ticker: str) -> str | None:
    cur = conn.execute(
        "SELECT last_price_date FROM prices_backfill_progress WHERE ticker = ? LIMIT 1",
        ((ticker or "").strip().upper(),),
    )
    row = cur.fetchone()
    if not row:
        return None
    return str(row[0]) if row[0] else None


def _ensure_prices_cursor_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fmp_prices_cursor (
          cursor_name TEXT PRIMARY KEY,
          last_ticker TEXT,
          updated_at TEXT NOT NULL
        );
        """
    )


def get_prices_cursor(conn: sqlite3.Connection, *, cursor_name: str) -> str | None:
    _ensure_prices_cursor_table(conn)
    cur = conn.execute(
        "SELECT last_ticker FROM fmp_prices_cursor WHERE cursor_name = ? LIMIT 1",
        (cursor_name,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return str(row[0]) if row[0] else None


def set_prices_cursor(conn: sqlite3.Connection, *, cursor_name: str, last_ticker: str) -> None:
    _ensure_prices_cursor_table(conn)
    conn.execute(
        """
        INSERT OR REPLACE INTO fmp_prices_cursor (cursor_name, last_ticker, updated_at)
        VALUES (?, ?, ?)
        """,
        (
            cursor_name,
            (last_ticker or "").strip().upper(),
            datetime.utcnow().replace(tzinfo=None).isoformat(timespec="seconds"),
        ),
    )


def upsert_analyst_estimates(conn: sqlite3.Connection, *, symbol: str, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    sym = (symbol or "").strip().upper()
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = _parse_iso_date(r.get("date"))
        if not d:
            continue
        # Stable `analyst-estimates` returns fields like revenueAvg, estimatedEpsAvg, etc.
        est_rev = r.get("estimatedRevenue") or r.get("revenueAvg") or r.get("revenueLow") or r.get("revenueHigh")
        est_eps = r.get("estimatedEps") or r.get("estimatedEPS") or r.get("estimatedEpsAvg") or r.get("epsAvg")
        conn.execute(
            """
            INSERT OR REPLACE INTO analyst_estimates (ticker, estimated_date, estimated_revenue, estimated_eps)
            VALUES (?, ?, ?, ?)
            """,
            (sym, d, _safe_float(est_rev), _safe_float(est_eps)),
        )
        n += 1
    return n


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(tzinfo=None).isoformat(timespec="seconds")


def _insider_row_key(ticker: str, row: dict[str, Any]) -> str:
    blob = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(f"{ticker}|{blob}".encode("utf-8")).hexdigest()[:48]


def _flatten_fmp_list_payload(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("Error Message"), str):
            return []
        for k in (
            "data",
            "suppliers",
            "holders",
            "ownership",
            "transactions",
            "insiderTrading",
            "results",
            "positions",
            "symbolPositionsSummary",
            "investors",
        ):
            v = data.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def _is_legacy_endpoint_forbidden(exc: BaseException) -> bool:
    resp = getattr(exc, "response", None)
    if isinstance(exc, requests.HTTPError) and resp is not None:
        try:
            code = int(resp.status_code)
        except Exception:
            code = 0
        if code == 403:
            try:
                body = (resp.text or "")[:800]
            except Exception:
                body = ""
            return "Legacy Endpoint" in body or "legacy endpoints" in body.lower()
    msg = str(exc)
    return "403" in msg and ("Legacy Endpoint" in msg or "legacy endpoints" in msg.lower())


def _prior_calendar_quarter(d: date) -> tuple[int, int]:
    """Most recently completed calendar quarter (year, quarter 1..4)."""
    q = (d.month - 1) // 3 + 1
    y = d.year
    if q == 1:
        return y - 1, 4
    return y, q - 1


def upsert_suppliers_response(conn: sqlite3.Connection, *, ticker: str, data: Any) -> int:
    sym = (ticker or "").strip().upper()
    now = _utc_now_iso()
    n = 0
    rows = _flatten_fmp_list_payload(data)
    for r in rows:
        sup = str(r.get("symbol") or r.get("supplierSymbol") or r.get("ticker") or "").strip().upper()
        if not sup:
            continue
        name = r.get("companyName") or r.get("name") or r.get("supplierName")
        rel = r.get("relationship") or r.get("relation") or r.get("type")
        try:
            pj = json.dumps(r, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            pj = None
        conn.execute(
            """
            INSERT OR REPLACE INTO suppliers
              (ticker, supplier_symbol, supplier_name, relationship, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (sym, sup, name, rel, pj, now),
        )
        n += 1
    return n


def upsert_institutional_ownership_response(conn: sqlite3.Connection, *, ticker: str, data: Any) -> int:
    sym = (ticker or "").strip().upper()
    now = _utc_now_iso()
    n = 0
    rows = _flatten_fmp_list_payload(data)
    for r in rows:
        holder = str(r.get("investorName") or r.get("holder") or r.get("name") or r.get("institutionName") or "").strip()
        if not holder:
            holder = str(r.get("cik") or r.get("id") or "UNKNOWN")[:200]
        as_of = _parse_iso_date(r.get("date") or r.get("reportDate") or r.get("filingDate")) or ""
        pct = _safe_float(r.get("ownershipPercent") or r.get("weightedPercent") or r.get("percent"))
        sh = _safe_float(r.get("shares") or r.get("sharesNumber") or r.get("weightedShares"))
        try:
            pj = json.dumps(r, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            pj = None
        conn.execute(
            """
            INSERT OR REPLACE INTO institutional_ownership
              (ticker, holder_name, as_of_date, ownership_percent, shares, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (sym, holder[:500], as_of, pct, sh, pj, now),
        )
        n += 1
    return n


def upsert_insider_trading_response(conn: sqlite3.Connection, *, ticker: str, data: Any) -> int:
    sym = (ticker or "").strip().upper()
    now = _utc_now_iso()
    n = 0
    rows = _flatten_fmp_list_payload(data)
    for r in rows:
        if not isinstance(r, dict):
            continue
        rk = _insider_row_key(sym, r)
        fd = _parse_iso_date(r.get("filingDate") or r.get("transactionDate") or r.get("date"))
        insider = str(r.get("reportingName") or r.get("name") or r.get("insiderName") or "").strip()[:400]
        tx = str(r.get("transactionType") or r.get("type") or "").strip()[:120]
        sh = _safe_float(r.get("securitiesTransacted") or r.get("securitiesOwned") or r.get("shares"))
        px = _safe_float(r.get("price"))
        try:
            pj = json.dumps(r, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            pj = "{}"
        conn.execute(
            """
            INSERT OR REPLACE INTO insider_trading
              (ticker, row_key, filing_date, insider_name, transaction_type, shares, price, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (sym, rk, fd, insider or None, tx or None, sh, px, pj, now),
        )
        n += 1
    return n


def upsert_historical_ratios_response(
    conn: sqlite3.Connection, *, ticker: str, period: str, data: Any
) -> int:
    sym = (ticker or "").strip().upper()
    per = (period or "quarter").strip().lower()[:16]
    now = _utc_now_iso()
    n = 0
    rows = _flatten_fmp_list_payload(data)
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = _parse_iso_date(r.get("date") or r.get("fillingDate"))
        if not d:
            continue
        try:
            pj = json.dumps(r, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            pj = "{}"
        conn.execute(
            """
            INSERT OR REPLACE INTO historical_financial_ratios
              (ticker, period, report_date, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (sym, per, d, pj, now),
        )
        n += 1
    return n


def upsert_historical_key_metrics_response(
    conn: sqlite3.Connection, *, ticker: str, period: str, data: Any
) -> int:
    sym = (ticker or "").strip().upper()
    per = (period or "quarter").strip().lower()[:16]
    now = _utc_now_iso()
    n = 0
    rows = _flatten_fmp_list_payload(data)
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = _parse_iso_date(r.get("date") or r.get("fillingDate"))
        if not d:
            continue
        try:
            pj = json.dumps(r, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            pj = "{}"
        conn.execute(
            """
            INSERT OR REPLACE INTO historical_key_metrics
              (ticker, period, report_date, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (sym, per, d, pj, now),
        )
        n += 1
    return n


def upsert_analyst_estimates_detail_response(conn: sqlite3.Connection, *, ticker: str, data: Any) -> int:
    sym = (ticker or "").strip().upper()
    now = _utc_now_iso()
    n = 0
    rows = _flatten_fmp_list_payload(data)
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = _parse_iso_date(r.get("date"))
        if not d:
            continue
        pnm = str(r.get("period") or r.get("periodYear") or "").strip()[:64]
        est_rev = _safe_float(
            r.get("estimatedRevenueAvg")
            or r.get("revenueAvg")
            or r.get("estimatedRevenue")
            or r.get("revenueAverage")
        )
        est_eps = _safe_float(
            r.get("estimatedEpsAvg") or r.get("epsAvg") or r.get("estimatedEps") or r.get("estimatedEPS")
        )
        try:
            pj = json.dumps(r, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            pj = None
        conn.execute(
            """
            INSERT OR REPLACE INTO analyst_estimates_detail
              (ticker, estimate_date, period_name, estimated_revenue_avg, estimated_eps_avg, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (sym, d, pnm, est_rev, est_eps, pj, now),
        )
        n += 1
    # Also upsert legacy analyst_estimates (UNIQUE is ticker+estimated_date only; last row wins if v3 returns duplicates per date).
    if rows:
        try:
            upsert_analyst_estimates(conn, symbol=sym, rows=rows)
        except Exception:
            pass
    return n


def upsert_earnings_surprises_response(conn: sqlite3.Connection, *, ticker: str, data: Any) -> int:
    sym = (ticker or "").strip().upper()
    now = _utc_now_iso()
    n = 0
    rows = _flatten_fmp_list_payload(data)
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = _parse_iso_date(r.get("date"))
        if not d:
            continue
        act = _safe_float(r.get("actualEarningResult") or r.get("actualEps") or r.get("epsActual"))
        est = _safe_float(r.get("estimatedEarning") or r.get("estimatedEps") or r.get("epsEstimated"))
        sp = _safe_float(r.get("surprisePercent") or r.get("surprisePercentage"))
        try:
            pj = json.dumps(r, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            pj = None
        conn.execute(
            """
            INSERT OR REPLACE INTO earnings_surprises
              (ticker, surprise_date, actual_eps, estimated_eps, surprise_percent, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (sym, d, act, est, sp, pj, now),
        )
        n += 1
    return n


def _institutional_payload_as_rows(data: Any) -> list[dict[str, Any]]:
    rows = _flatten_fmp_list_payload(data)
    if rows:
        return rows
    if isinstance(data, dict) and not data.get("Error Message"):
        return [data]
    return []


def _ensure_quant_feature_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS suppliers (
          ticker TEXT NOT NULL,
          supplier_symbol TEXT NOT NULL,
          supplier_name TEXT,
          relationship TEXT,
          payload_json TEXT,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (ticker, supplier_symbol)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS institutional_ownership (
          ticker TEXT NOT NULL,
          holder_name TEXT NOT NULL,
          as_of_date TEXT NOT NULL DEFAULT '',
          ownership_percent REAL,
          shares REAL,
          payload_json TEXT,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (ticker, holder_name, as_of_date)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS insider_trading (
          ticker TEXT NOT NULL,
          row_key TEXT NOT NULL,
          filing_date TEXT,
          insider_name TEXT,
          transaction_type TEXT,
          shares REAL,
          price REAL,
          payload_json TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (ticker, row_key)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_financial_ratios (
          ticker TEXT NOT NULL,
          period TEXT NOT NULL,
          report_date TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(ticker, period, report_date)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_key_metrics (
          ticker TEXT NOT NULL,
          period TEXT NOT NULL,
          report_date TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(ticker, period, report_date)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analyst_estimates (
          ticker TEXT NOT NULL,
          estimated_date TEXT NOT NULL,
          estimated_revenue REAL,
          estimated_eps REAL,
          UNIQUE(ticker, estimated_date)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analyst_estimates_detail (
          ticker TEXT NOT NULL,
          estimate_date TEXT NOT NULL,
          period_name TEXT NOT NULL DEFAULT '',
          estimated_revenue_avg REAL,
          estimated_eps_avg REAL,
          payload_json TEXT,
          updated_at TEXT NOT NULL,
          UNIQUE(ticker, estimate_date, period_name)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS earnings_surprises (
          ticker TEXT NOT NULL,
          surprise_date TEXT NOT NULL,
          actual_eps REAL,
          estimated_eps REAL,
          surprise_percent REAL,
          payload_json TEXT,
          updated_at TEXT NOT NULL,
          UNIQUE(ticker, surprise_date)
        );
        """
    )


def run_watchlist_quant_features(
    conn: sqlite3.Connection,
    client: FmpClient,
    *,
    watchlist_name: str,
    args: argparse.Namespace,
) -> None:
    """
    Per-watchlist quant features. Prefer stable /stable/* endpoints; v4 supplier is legacy-only on
    many new plans — after the first legacy 403 we skip the rest of that job to avoid log spam.
    """
    flags = [
        bool(getattr(args, "fetch_supply_chain", False)),
        bool(getattr(args, "fetch_ownership", False)),
        bool(getattr(args, "fetch_historical_ratios", False)),
        bool(getattr(args, "fetch_historical_metrics", False)),
        bool(getattr(args, "fetch_estimates", False)),
        bool(getattr(args, "fetch_surprises", False)),
    ]
    if not any(flags):
        return

    ensure_fmp_watchlist_schema(conn)
    _ensure_quant_feature_tables(conn)
    syms = load_watchlist_tickers(conn, watchlist_name=watchlist_name)
    if not syms:
        print(f"WARN: fmp_watchlist has no rows for {watchlist_name!r}; skipping quant feature fetch.")
        return

    prog = max(1, int(getattr(args, "watchlist_fetch_progress_every", 100) or 100))
    hist_period = str(getattr(args, "historical_fundamental_period", "quarter") or "quarter").strip().lower()
    if hist_period not in ("quarter", "annual"):
        hist_period = "quarter"
    sleep_s = float(getattr(args, "watchlist_fetch_sleep_s", 0.02) or 0.02)
    io_y = int(getattr(args, "ownership_13f_year", 0) or 0)
    io_q = int(getattr(args, "ownership_13f_quarter", 0) or 0)
    if io_y <= 0 or io_q <= 0:
        io_y, io_q = _prior_calendar_quarter(_today())
    est_pages = max(1, int(getattr(args, "analyst_estimates_watchlist_pages", 8) or 8))
    est_limit = max(1, min(100, int(getattr(args, "analyst_estimates_watchlist_limit", 50) or 50)))

    halt_supply_v4 = False
    halt_ownership_v4 = False
    halt_insider_v4 = False
    halt_ratios_v3 = False
    halt_metrics_v3 = False
    halt_estimates_v3 = False
    halt_surprises_v3 = False
    halt_surprises_stable = False

    n_total = len(syms)
    print(
        f"Quant features for watchlist {watchlist_name!r}: {n_total:,} tickers "
        f"(progress every {prog}); historical period={hist_period!r}; "
        f"13F summary year/quarter={io_y}-Q{io_q}"
    )
    if getattr(args, "fetch_supply_chain", False):
        print(
            "Note: --fetch-supply-chain uses legacy v4 /supplier; many post-2025-08 subscriptions get 403 — "
            "the run will stop calling it after the first such error (no per-ticker spam)."
        )

    for i, sym in enumerate(syms, start=1):
        if i == 1 or i % prog == 0 or i == n_total:
            print(f"  quant-features progress: {i:,}/{n_total:,} ({sym})")

        if getattr(args, "fetch_supply_chain", False) and not halt_supply_v4:
            try:
                url = f"{FMP_V4_ROOT}/supplier"
                data = client.get_data(url, params={"symbol": sym}, timeout_s=60)
                upsert_suppliers_response(conn, ticker=sym, data=data)
            except Exception as e:
                if _is_legacy_endpoint_forbidden(e):
                    print(
                        "INFO: v4 /supplier blocked (legacy-only for your plan). "
                        "Skipping remaining --fetch-supply-chain calls for this run."
                    )
                    halt_supply_v4 = True
                elif i % prog == 0 or i <= 3:
                    print(f"    WARN supply-chain {sym}: {e}")

        if getattr(args, "fetch_ownership", False):
            if not halt_ownership_v4:
                try:
                    url = f"{FMP_V4_ROOT}/institutional-ownership/institutional-holders/symbol-ownership-percent"
                    data = client.get_data(url, params={"symbol": sym}, timeout_s=60)
                    upsert_institutional_ownership_response(conn, ticker=sym, data=data)
                except Exception as e:
                    if _is_legacy_endpoint_forbidden(e):
                        print("INFO: v4 institutional-ownership blocked; falling back to stable endpoint.")
                        halt_ownership_v4 = True
                    elif i % prog == 0 or i <= 3:
                        print(f"    WARN institutional-ownership v4 {sym}: {e}")
            if halt_ownership_v4:
                try:
                    url = f"{FMP_BASE}/institutional-ownership/symbol-positions-summary"
                    data = client.get_data(url, params={"symbol": sym, "year": io_y, "quarter": io_q}, timeout_s=60)
                    rows_io = _institutional_payload_as_rows(data)
                    upsert_institutional_ownership_response(conn, ticker=sym, data=rows_io)
                except Exception as e:
                    if i % prog == 0 or i <= 3:
                        print(f"    WARN institutional-ownership stable {sym}: {e}")

        if getattr(args, "fetch_ownership", False):
            if not halt_insider_v4:
                try:
                    url = f"{FMP_V4_ROOT}/insider-trading"
                    data = client.get_data(url, params={"symbol": sym}, timeout_s=60)
                    upsert_insider_trading_response(conn, ticker=sym, data=data)
                except Exception as e:
                    if _is_legacy_endpoint_forbidden(e):
                        print("INFO: v4 insider-trading blocked; falling back to stable endpoint.")
                        halt_insider_v4 = True
                    elif i % prog == 0 or i <= 3:
                        print(f"    WARN insider-trading v4 {sym}: {e}")
            if halt_insider_v4:
                try:
                    url = f"{FMP_BASE}/insider-trading/search"
                    data = client.get_data(url, params={"symbol": sym, "page": 0, "limit": 100}, timeout_s=60)
                    upsert_insider_trading_response(conn, ticker=sym, data=data)
                except Exception as e:
                    if i % prog == 0 or i <= 3:
                        print(f"    WARN insider-trading stable {sym}: {e}")

        if getattr(args, "fetch_historical_ratios", False):
            if not halt_ratios_v3:
                try:
                    url = f"{FMP_V3_ROOT}/ratios/{sym}"
                    data = client.get_data(url, params={"period": hist_period}, timeout_s=60)
                    upsert_historical_ratios_response(conn, ticker=sym, period=hist_period, data=data)
                except Exception as e:
                    if _is_legacy_endpoint_forbidden(e):
                        print("INFO: v3 /ratios blocked; falling back to stable endpoint.")
                        halt_ratios_v3 = True
                    elif i % prog == 0 or i <= 3:
                        print(f"    WARN historical-ratios v3 {sym}: {e}")
            if halt_ratios_v3:
                try:
                    url = f"{FMP_BASE}/ratios"
                    data = client.get_data(url, params={"symbol": sym}, timeout_s=60)
                    upsert_historical_ratios_response(conn, ticker=sym, period="annual", data=data)
                except Exception as e:
                    if i % prog == 0 or i <= 3:
                        print(f"    WARN historical-ratios stable {sym}: {e}")

        if getattr(args, "fetch_historical_metrics", False):
            if not halt_metrics_v3:
                try:
                    url = f"{FMP_V3_ROOT}/key-metrics/{sym}"
                    data = client.get_data(url, params={"period": hist_period}, timeout_s=60)
                    upsert_historical_key_metrics_response(conn, ticker=sym, period=hist_period, data=data)
                except Exception as e:
                    if _is_legacy_endpoint_forbidden(e):
                        print("INFO: v3 /key-metrics blocked; falling back to stable endpoint.")
                        halt_metrics_v3 = True
                    elif i % prog == 0 or i <= 3:
                        print(f"    WARN historical-key-metrics v3 {sym}: {e}")
            if halt_metrics_v3:
                try:
                    url = f"{FMP_BASE}/key-metrics"
                    data = client.get_data(url, params={"symbol": sym}, timeout_s=60)
                    upsert_historical_key_metrics_response(conn, ticker=sym, period="annual", data=data)
                except Exception as e:
                    if i % prog == 0 or i <= 3:
                        print(f"    WARN historical-key-metrics stable {sym}: {e}")

        if getattr(args, "fetch_estimates", False):
            est_ok = False
            if not halt_estimates_v3:
                try:
                    url = f"{FMP_V3_ROOT}/analyst-estimates/{sym}"
                    data = client.get_data(url, params={"period": hist_period}, timeout_s=60)
                    if isinstance(data, list) and len(data) > 0:
                        upsert_analyst_estimates_detail_response(conn, ticker=sym, data=data)
                        est_ok = True
                except Exception as e:
                    if _is_legacy_endpoint_forbidden(e):
                        print("INFO: v3 /analyst-estimates blocked; falling back to stable endpoint.")
                        halt_estimates_v3 = True
                    elif i % prog == 0 or i <= 3:
                        print(f"    WARN analyst-estimates v3 {sym}: {e}")
            if not est_ok:
                try:
                    for page in range(est_pages):
                        rows = fetch_analyst_estimates(
                            client,
                            symbol=sym,
                            period="annual",
                            limit=min(10, est_limit),
                            page=page,
                        )
                        if not rows:
                            break
                        upsert_analyst_estimates_detail_response(conn, ticker=sym, data=rows)
                        est_ok = True
                except Exception as e2:
                    if i % prog == 0 or i <= 3:
                        print(f"    WARN analyst-estimates stable {sym}: {e2}")

        if getattr(args, "fetch_surprises", False):
            es_ok = False
            if not halt_surprises_v3:
                try:
                    url = f"{FMP_V3_ROOT}/earnings-surprises/{sym}"
                    data = client.get_data(url, params=None, timeout_s=60)
                    if isinstance(data, list) and len(data) > 0:
                        upsert_earnings_surprises_response(conn, ticker=sym, data=data)
                        es_ok = True
                except Exception as e:
                    if _is_legacy_endpoint_forbidden(e):
                        print("INFO: v3 /earnings-surprises blocked; falling back to stable endpoint.")
                        halt_surprises_v3 = True
                    elif i % prog == 0 or i <= 3:
                        print(f"    WARN earnings-surprises v3 {sym}: {e}")
            if not es_ok and not halt_surprises_stable:
                try:
                    url = f"{FMP_BASE}/earnings-surprises"
                    data = client.get_data(url, params={"symbol": sym}, timeout_s=60)
                    if isinstance(data, list) and len(data) > 0:
                        upsert_earnings_surprises_response(conn, ticker=sym, data=data)
                except requests.HTTPError as e2:
                    if "404" in str(e2):
                        print("INFO: stable /earnings-surprises returns 404; endpoint not available for this plan.")
                        halt_surprises_stable = True
                    elif i % prog == 0 or i <= 3:
                        print(f"    WARN earnings-surprises stable {sym}: {e2}")
                except Exception as e2:
                    if i % prog == 0 or i <= 3:
                        print(f"    WARN earnings-surprises stable {sym}: {e2}")

        try:
            conn.commit()
        except Exception as e:
            print(f"    WARN commit after {sym}: {e}")

        if sleep_s > 0:
            time.sleep(sleep_s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bulk-load US equity fundamentals from FMP stable endpoints.")
    ap.add_argument("--calls-per-minute", type=int, default=300, help="Target rate limit for requests (calls/min).")
    ap.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip company profile fetch (stable /profile for tickers in --watchlist-name).",
    )
    ap.add_argument(
        "--profile-batch-size",
        type=int,
        default=100,
        help=(
            "When not using --skip-profile: stable /profile is one symbol per API call; this value is only used "
            "as a progress log interval (log every N symbols). Default: 100."
        ),
    )
    ap.add_argument(
        "--skip-historical-ratios",
        action="store_true",
        help="Skip historical financial ratios fetch for watchlist tickers (stable /profile pattern).",
    )
    ap.add_argument(
        "--historical-ratios-limit",
        type=int,
        default=40,
        help="Max periods to fetch per symbol for historical ratios (default: 40, i.e. ~10 years quarterly).",
    )
    ap.add_argument(
        "--skip-ratios-bulk",
        action="store_true",
        help="Skip ratios-ttm-bulk download (use when only running watchlist quant features).",
    )
    ap.add_argument(
        "--skip-key-metrics",
        action="store_true",
        help="Skip key-metrics-ttm-bulk (recommended if throttled).",
    )
    ap.add_argument(
        "--eod-days",
        type=int,
        default=0,
        help="Backfill EOD bulk for the last N days (uses daily tracker to skip already loaded dates).",
    )
    ap.add_argument(
        "--eod-from",
        type=str,
        default="",
        help="Optional start date for EOD backfill (YYYY-MM-DD). If set, loads from this date to yesterday.",
    )
    ap.add_argument(
        "--analyst-estimates-tech",
        action="store_true",
        help="Fetch analyst estimates (non-bulk) for a small tech-first list and store into analyst_estimates.",
    )
    ap.add_argument(
        "--analyst-estimates-period",
        type=str,
        default="annual",
        help="Analyst estimates period to request (annual is more commonly available on lower plans).",
    )
    ap.add_argument(
        "--analyst-estimates-limit",
        type=int,
        default=10,
        help="Analyst estimates limit (some plans restrict this to <= 10).",
    )
    ap.add_argument(
        "--skip-eod",
        action="store_true",
        help="Skip eod-bulk backfill (some plans restrict this endpoint).",
    )
    ap.add_argument(
        "--no-single-instance-lock",
        action="store_true",
        help="Disable the exclusive lock that prevents concurrent runs.",
    )
    ap.add_argument(
        "--prices-tech",
        action="store_true",
        help="Backfill historical EOD prices for a small tech-first list using /historical-price-eod/full.",
    )
    ap.add_argument(
        "--prices-universe",
        action="store_true",
        help="Backfill historical EOD prices for a batch of tickers from the universe (incremental; uses DB progress).",
    )
    ap.add_argument(
        "--prices-batch-size",
        type=int,
        default=100,
        help="How many tickers to backfill per run when using --prices-universe.",
    )
    ap.add_argument(
        "--prices-universe-size",
        type=int,
        default=0,
        help="Alias for --prices-batch-size (some users remember this name). If set (>0), it overrides batch size.",
    )
    ap.add_argument(
        "--prices-sequential",
        action="store_true",
        help="When using --prices-universe, advance ticker-by-ticker alphabetically from the last cursor (instead of 'catch-up' sorting).",
    )
    ap.add_argument(
        "--prices-cursor-name",
        type=str,
        default="universe",
        help="Cursor key stored in fmp_prices_cursor (allows multiple parallel strategies if needed).",
    )
    ap.add_argument(
        "--universe-custom",
        type=str,
        default="",
        help=(
            "Comma-separated universe preset tokens for price backfills, e.g. "
            "'sp500,nasdaq,ndx,semi'. "
            "ndx uses ETF holdings (see --ndx-etf). semi uses ETF holdings union (see --semi-etfs)."
        ),
    )
    ap.add_argument(
        "--universe-strategy",
        type=str,
        default="",
        help=(
            "High-level universe preset for quantitative workflows. "
            "Choices: sp500 | russell1000 | russell3000 | ndx | sp500_ndx. "
            "You may pass multiple comma-separated strategies to union/deduplicate symbols in one run. "
            "Russell universes use ETF holdings proxies (see --russell1000-etf / --russell3000-etf)."
        ),
    )
    ap.add_argument(
        "--watchlist-name",
        type=str,
        default="default",
        help="Name for fmp_watchlist rows created from --universe-custom (used to filter --prices-universe).",
    )
    ap.add_argument(
        "--reuse-watchlist-max-age-hours",
        type=float,
        default=72.0,
        help=(
            "If fmp_watchlist + manifest exist for --watchlist-name, fingerprint matches current universe settings, "
            "and build age is within this many hours, skip screener/ADV (strategy) or ETF fetch (custom) and load "
            "tickers from DB. Use <=0 to disable reuse (always rebuild). Default: 72."
        ),
    )
    ap.add_argument(
        "--force-rebuild-universe",
        action="store_true",
        help="Ignore cached watchlist/manifest and always rebuild universe (screener/ADV or custom fetches).",
    )
    ap.add_argument(
        "--ndx-etf",
        type=str,
        default="QQQ",
        help="ETF symbol used when token includes ndx/nasdaq100/qqq (default: QQQ).",
    )
    ap.add_argument(
        "--russell1000-etf",
        type=str,
        default="IWB",
        help="ETF holdings proxy for Russell 1000 universe (default: IWB).",
    )
    ap.add_argument(
        "--russell3000-etf",
        type=str,
        default="IWV",
        help="ETF holdings proxy for Russell 3000 universe (default: IWV).",
    )
    ap.add_argument(
        "--semi-etfs",
        type=str,
        default="SMH,SOXX",
        help="Comma-separated ETF symbols used when token includes semi/sox (default: SMH,SOXX).",
    )
    ap.add_argument(
        "--filter-price-min",
        type=float,
        default=5.0,
        help="Minimum stock price for universe hygiene filters (default: 5).",
    )
    ap.add_argument(
        "--filter-market-cap-min",
        type=float,
        default=300_000_000.0,
        help="Minimum market cap (USD) for universe hygiene filters (default: 300_000_000).",
    )
    ap.add_argument(
        "--filter-adv-min-usd",
        type=float,
        default=1_000_000.0,
        help="Minimum trailing average dollar volume (USD) over --filter-adv-days (default: 1_000_000).",
    )
    ap.add_argument(
        "--filter-adv-days",
        type=int,
        default=20,
        help="Trading days used for ADV$ rolling average (default: 20).",
    )
    ap.add_argument(
        "--filter-adv-lookback-days",
        type=int,
        default=45,
        help="Calendar lookback window for fetching bars before ADV$ measurement (default: 45).",
    )
    ap.add_argument(
        "--filter-adv-max-candidates",
        type=int,
        default=0,
        help="Optional cap on how many symbols receive ADV$ screening (0 means no cap).",
    )
    ap.add_argument(
        "--skip-heavy-bulk-on-watchlist",
        action="store_true",
        help=(
            "When --universe-custom / --universe-strategy is set, skip ratios/key-metrics bulk downloads for this run "
            "(speeds up watchlist-only price backfills)."
        ),
    )
    ap.add_argument(
        "--prices-window-days",
        type=int,
        default=31,
        help="Max calendar days per ticker per run (keeps each call reasonably small).",
    )
    ap.add_argument(
        "--prices-start",
        type=str,
        default="2016-01-01",
        help="Default start date for new tickers without progress (YYYY-MM-DD).",
    )
    ap.add_argument(
        "--prices-from",
        type=str,
        default="",
        help="Start date for per-symbol EOD history (YYYY-MM-DD).",
    )
    ap.add_argument(
        "--prices-to",
        type=str,
        default="",
        help="End date for per-symbol EOD history (YYYY-MM-DD).",
    )
    ap.add_argument(
        "--quarters",
        type=int,
        default=20,
        help="How many most-recent quarters to consider as the target window (e.g. 20 ~= ~5 years).",
    )
    ap.add_argument(
        "--max-quarters-per-run",
        type=int,
        default=2,
        help="Process at most N quarters per run (incremental backfill without re-fetching).",
    )
    ap.add_argument(
        "--fetch-supply-chain",
        action="store_true",
        help="For --watchlist-name: fetch /api/v4/supplier and upsert into suppliers.",
    )
    ap.add_argument(
        "--fetch-supply-chain-edgar",
        action="store_true",
        help=(
            "For --watchlist-name: parse SEC EDGAR 10-K filings to extract major-customer "
            "relationships and upsert into suppliers (free alternative to FMP v4/supplier)."
        ),
    )
    ap.add_argument(
        "--edgar-max-tickers",
        type=int,
        default=0,
        help="When --fetch-supply-chain-edgar: process at most N tickers (0=all).",
    )
    ap.add_argument(
        "--fetch-ownership",
        action="store_true",
        help="For --watchlist-name: fetch institutional holders + insider trading (v4) into institutional_ownership and insider_trading.",
    )
    ap.add_argument(
        "--fetch-historical-ratios",
        action="store_true",
        help="For --watchlist-name: fetch /api/v3/ratios/{symbol} into historical_financial_ratios (legacy endpoint).",
    )
    ap.add_argument(
        "--fetch-historical-metrics",
        action="store_true",
        help="For --watchlist-name: fetch /api/v3/key-metrics/{symbol} into historical_key_metrics (legacy endpoint).",
    )
    ap.add_argument(
        "--fetch-estimates",
        action="store_true",
        help="For --watchlist-name: fetch /api/v3/analyst-estimates/{symbol} into analyst_estimates_detail.",
    )
    ap.add_argument(
        "--fetch-surprises",
        action="store_true",
        help="For --watchlist-name: fetch /api/v3/earnings-surprises/{symbol} into earnings_surprises.",
    )
    ap.add_argument(
        "--watchlist-fetch-progress-every",
        type=int,
        default=100,
        help="When running watchlist quant feature flags, log progress every N tickers (default: 100).",
    )
    ap.add_argument(
        "--historical-fundamental-period",
        type=str,
        default="quarter",
        choices=["quarter", "annual"],
        help="Period query param for --fetch-historical-ratios / --fetch-historical-metrics (default: quarter).",
    )
    ap.add_argument(
        "--watchlist-fetch-sleep-s",
        type=float,
        default=0.02,
        help="Optional extra sleep (seconds) after each watchlist ticker in quant feature loops (default: 0.02).",
    )
    ap.add_argument(
        "--ownership-13f-year",
        type=int,
        default=0,
        help="Year for stable institutional-ownership/symbol-positions-summary (0 = auto: prior calendar quarter).",
    )
    ap.add_argument(
        "--ownership-13f-quarter",
        type=int,
        default=0,
        help="Quarter 1-4 for symbol-positions-summary (0 = auto with --ownership-13f-year).",
    )
    ap.add_argument(
        "--analyst-estimates-watchlist-pages",
        type=int,
        default=8,
        help="Max pages per ticker when using --fetch-estimates (stable analyst-estimates). Default: 8.",
    )
    ap.add_argument(
        "--analyst-estimates-watchlist-limit",
        type=int,
        default=50,
        help="Page size for --fetch-estimates (stable), capped at 100. Default: 50.",
    )
    args = ap.parse_args()

    # Optional alias resolution
    if int(getattr(args, "prices_universe_size", 0) or 0) > 0:
        args.prices_batch_size = int(args.prices_universe_size)

    semi_etfs_list = [x.strip().upper() for x in str(args.semi_etfs or "").split(",") if x.strip()]

    lock = None
    if not args.no_single_instance_lock:
        lock = SingleInstanceLock(_singleton_lock_path())
        try:
            lock.acquire()
        except RuntimeError as e:
            print(str(e))
            return

    try:
        if FMP_API_KEY.strip() in ("", "YOUR_API_KEY_HERE"):
            print("FMP_API_KEY is not set. Please edit this file and set FMP_API_KEY before running.")
            return

        print("Connecting to DB...")
        db = init_db(_db_path())
        print(f"DB path: {db}")

        print(f"Initializing FMP client (rate limit: {int(args.calls_per_minute)} calls/min)...")
        client = FmpClient(FMP_API_KEY, calls_per_minute=int(args.calls_per_minute))

        watchlist_name = str(args.watchlist_name or "default").strip() or "default"
        custom_tokens = parse_universe_tokens(str(args.universe_custom or ""))
        strategies = parse_strategy_list(str(args.universe_strategy or ""))
        custom_syms: list[str] = []
        used_watchlist = False
        watchlist_reused = False
        wl_fingerprint: str | None = None

        if strategies and custom_tokens:
            print("ERROR: Use either --universe-strategy OR --universe-custom (not both).")
            return

        adv_cap = int(args.filter_adv_max_candidates or 0)
        adv_cap_opt = None if adv_cap <= 0 else adv_cap

        if strategies:
            if watchlist_name == "default":
                watchlist_name = "union_" + "_".join(strategies)

            wl_fingerprint = build_strategy_universe_fingerprint(
                strategies=strategies,
                ndx_etf=str(args.ndx_etf or "QQQ"),
                russell1000_etf=str(args.russell1000_etf or "IWB"),
                russell3000_etf=str(args.russell3000_etf or "IWV"),
                price_min=float(args.filter_price_min),
                market_cap_min=float(args.filter_market_cap_min),
                adv_min_usd=float(args.filter_adv_min_usd),
                adv_days=int(args.filter_adv_days),
                adv_lookback_days=int(args.filter_adv_lookback_days),
                adv_max_candidates=adv_cap_opt,
            )
            with sqlite3.connect(str(db)) as conn:
                conn.execute("PRAGMA foreign_keys=ON;")
                ensure_fmp_watchlist_schema(conn)
                ok_cached, cached_syms = maybe_load_cached_watchlist(
                    conn,
                    watchlist_name=watchlist_name,
                    fingerprint=wl_fingerprint,
                    max_age_hours=float(args.reuse_watchlist_max_age_hours),
                    force_rebuild=bool(args.force_rebuild_universe),
                )
                conn.commit()
            if ok_cached:
                custom_syms = cached_syms
                watchlist_reused = True
            else:
                print(
                    "Building strategy universe with hygiene filters: "
                    f"price>={float(args.filter_price_min):g}, "
                    f"marketCap>={float(args.filter_market_cap_min):g}, "
                    f"ADV$>={float(args.filter_adv_min_usd):g} ({int(args.filter_adv_days)}d avg)"
                )
                try:
                    custom_syms = build_multi_strategy_universe_symbols(
                        client,
                        strategies=strategies,
                        ndx_etf=str(args.ndx_etf or "QQQ"),
                        russell1000_etf=str(args.russell1000_etf or "IWB"),
                        russell3000_etf=str(args.russell3000_etf or "IWV"),
                        price_min=float(args.filter_price_min),
                        market_cap_min=float(args.filter_market_cap_min),
                        adv_min_usd=float(args.filter_adv_min_usd),
                        adv_days=int(args.filter_adv_days),
                        adv_lookback_days=int(args.filter_adv_lookback_days),
                        adv_max_candidates=adv_cap_opt,
                    )
                except ValueError as e:
                    print(f"ERROR: {e}")
                    return
            used_watchlist = True

        if custom_tokens:
            wl_fingerprint = build_custom_universe_fingerprint(
                tokens=custom_tokens,
                ndx_etf=str(args.ndx_etf or "QQQ"),
                semi_etfs=semi_etfs_list,
            )
            with sqlite3.connect(str(db)) as conn:
                conn.execute("PRAGMA foreign_keys=ON;")
                ensure_fmp_watchlist_schema(conn)
                ok_cached, cached_syms = maybe_load_cached_watchlist(
                    conn,
                    watchlist_name=watchlist_name,
                    fingerprint=wl_fingerprint,
                    max_age_hours=float(args.reuse_watchlist_max_age_hours),
                    force_rebuild=bool(args.force_rebuild_universe),
                )
                conn.commit()
            if ok_cached:
                custom_syms = cached_syms
                watchlist_reused = True
            else:
                print(f"Building custom universe from tokens: {', '.join(custom_tokens)}")
                custom_syms = build_custom_universe_symbols(
                    client,
                    tokens=custom_tokens,
                    ndx_etf=str(args.ndx_etf or "QQQ"),
                    semi_etfs=semi_etfs_list,
                )
                print(f"Custom universe size (unique symbols): {len(custom_syms):,}")
            used_watchlist = True

        if used_watchlist:
            # Persist watchlist for SQL joins during price backfills.
            with sqlite3.connect(str(db)) as conn:
                conn.execute("PRAGMA foreign_keys=ON;")
                ensure_fmp_watchlist_schema(conn)
                if watchlist_reused:
                    print(f"Watchlist '{watchlist_name}' loaded from DB ({len(custom_syms):,} tickers); not rewriting rows.")
                else:
                    n_wl = replace_watchlist(conn, watchlist_name=watchlist_name, tickers=custom_syms)
                    if wl_fingerprint:
                        upsert_watchlist_manifest(
                            conn,
                            watchlist_name=watchlist_name,
                            fingerprint=wl_fingerprint,
                            symbol_count=n_wl,
                        )
                    print(f"Saved fmp_watchlist '{watchlist_name}' rows: {n_wl:,}")
                conn.commit()

            # For watchlist-focused workflows, avoid dragging the full 26k universe into prices-universe SQL.
            if args.prices_universe:
                universe = custom_syms

        # fmp_universe seed list + compatibility variable `universe` (mostly informational).
        if args.prices_universe and used_watchlist:
            universe_full = sorted(set(custom_syms))
            print(f"fmp_universe seed tickers (watchlist scope): {len(universe_full):,} symbols")
        else:
            print("Fetching financial statement symbol universe...")
            universe_full = fetch_financial_statement_symbol_list(client)
            print(f"Universe size (financial statements): {len(universe_full):,} symbols")

        universe = universe_full
        if used_watchlist and args.prices_universe:
            universe = custom_syms

        quarters = last_n_quarters(int(args.quarters))
        print("Target quarters:", ", ".join(str(q) for q in quarters))

        with sqlite3.connect(str(db)) as conn:
            conn.execute("PRAGMA foreign_keys=ON;")
            _ensure_backfill_table(conn)
            _ensure_daily_backfill_table(conn)

            skip_heavy = bool(args.skip_heavy_bulk_on_watchlist and used_watchlist)

            # Ratios are point-in-time datasets (TTM).
            # Profile bulk is optional (often heavily throttled); skip by default in mode A.
            if args.skip_profile:
                print("Skipping company profile (--skip-profile).")
            else:
                ensure_fmp_watchlist_schema(conn)
                profile_syms = load_watchlist_tickers(conn, watchlist_name=watchlist_name)
                if not profile_syms:
                    print(
                        f"WARN: fmp_watchlist has no rows for watchlist_name={watchlist_name!r}; "
                        "skipping v3 profile fetch. Populate the watchlist first (e.g. --universe-strategy ...)."
                    )
                else:
                    print(
                        f"Fetching company profiles (stable /profile, one symbol per call) for watchlist "
                        f"{watchlist_name!r} ({len(profile_syms):,} tickers, log every {int(args.profile_batch_size)})..."
                    )
                    profiles = fetch_profiles_stable_for_watchlist(
                        client,
                        profile_syms,
                        log_every=int(args.profile_batch_size),
                    )
                    n_prof = upsert_profiles(conn, profiles)
                    conn.commit()
                    print(f"Upserted company_profile rows: {n_prof:,}")

            if args.skip_historical_ratios:
                print("Skipping historical financial ratios (--skip-historical-ratios).")
            else:
                ensure_fmp_watchlist_schema(conn)
                ratio_syms = load_watchlist_tickers(conn, watchlist_name=watchlist_name)
                if not ratio_syms:
                    print(
                        f"WARN: fmp_watchlist has no rows for watchlist_name={watchlist_name!r}; "
                        "skipping historical ratios fetch. Populate the watchlist first (e.g. --universe-strategy ...)."
                    )
                else:
                    ratio_period = str(args.historical_fundamental_period or "quarter").strip().lower()
                    if ratio_period not in ("quarter", "annual"):
                        ratio_period = "quarter"
                    _ensure_quant_feature_tables(conn)
                    print(
                        f"Fetching historical financial ratios (v3 then stable fallback, period={ratio_period!r}, "
                        f"limit={int(args.historical_ratios_limit)}) for watchlist "
                        f"{watchlist_name!r} ({len(ratio_syms):,} tickers, log every {int(args.profile_batch_size)})..."
                    )
                    n_ratios = fetch_historical_ratios_for_watchlist(
                        client,
                        conn,
                        ratio_syms,
                        period=ratio_period,
                        limit=int(args.historical_ratios_limit),
                        log_every=int(args.profile_batch_size),
                    )
                    print(f"Upserted historical_financial_ratios rows: {n_ratios:,}")

            if skip_heavy or args.skip_ratios_bulk:
                print("Skipping ratios-ttm-bulk.")
            else:
                print("Fetching ratios-ttm-bulk...")
                ratios = fetch_ratios_ttm_bulk(client)
                n_rat = upsert_ratios_ttm(conn, ratios)
                conn.commit()
                print(f"Upserted financial_ratios rows (as-of today): {n_rat:,}")

            if skip_heavy:
                print("Skipping key-metrics-ttm-bulk (--skip-heavy-bulk-on-watchlist).")
            elif args.skip_key_metrics:
                print("Skipping key-metrics-ttm-bulk.")
            else:
                print("Fetching key-metrics-ttm-bulk...")
                km = fetch_key_metrics_ttm_bulk(client)
                n_km = upsert_key_metrics_ttm(conn, km)
                conn.commit()
                print(f"Upserted key_metrics_ttm rows (as-of today): {n_km:,}")

            # Analyst estimates are non-bulk; keep it intentionally small by default.
            # Run this BEFORE EOD so an EOD plan restriction doesn't abort the whole job.
            if args.analyst_estimates_tech:
                tech_first = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX"]
                print("Fetching analyst estimates (tech-first, non-bulk)...")
                for sym in tech_first:
                    print(f"  - {sym}...")
                    try:
                        rows = fetch_analyst_estimates(
                            client,
                            symbol=sym,
                            period=str(args.analyst_estimates_period or "annual"),
                            limit=int(args.analyst_estimates_limit),
                            page=0,
                        )
                    except requests.HTTPError as e:
                        print(f"    WARN: analyst-estimates failed for {sym}: {e}")
                        time.sleep(0.15)
                        continue
                    n_e = upsert_analyst_estimates(conn, symbol=sym, rows=rows)
                    conn.commit()
                    print(f"    Upserted analyst_estimates rows: {n_e:,}")
                    time.sleep(0.15)

            run_watchlist_quant_features(conn, client, watchlist_name=watchlist_name, args=args)

            # SEC EDGAR supply chain (free alternative to FMP v4/supplier)
            if getattr(args, "fetch_supply_chain_edgar", False):
                try:
                    from edgar_supply_chain import run_edgar_supply_chain  # type: ignore
                except ModuleNotFoundError:
                    try:
                        from scripts.edgar_supply_chain import run_edgar_supply_chain  # type: ignore
                    except ModuleNotFoundError:
                        print("ERROR: scripts/edgar_supply_chain.py not found. Skipping.")
                        run_edgar_supply_chain = None  # type: ignore
                if run_edgar_supply_chain is not None:
                    with sqlite3.connect(str(db)) as esc_conn:
                        esc_conn.execute("PRAGMA journal_mode=WAL;")
                        wl_syms = [
                            r[0]
                            for r in esc_conn.execute(
                                "SELECT DISTINCT ticker FROM fmp_watchlist WHERE watchlist_name=? ORDER BY ticker",
                                (watchlist_name,),
                            ).fetchall()
                        ]
                    print(
                        f"Fetching SEC EDGAR supply chain for watchlist '{watchlist_name}' "
                        f"({len(wl_syms):,} tickers)…"
                    )
                    with sqlite3.connect(str(db)) as esc_conn:
                        esc_conn.execute("PRAGMA journal_mode=WAL;")
                        run_edgar_supply_chain(
                            esc_conn,
                            wl_syms,
                            max_tickers=int(getattr(args, "edgar_max_tickers", 0) or 0),
                            progress_every=int(getattr(args, "watchlist_fetch_progress_every", 25) or 25),
                        )

            # Prices (historical EOD) fallback: per-symbol endpoint that's available on lower plans.
            if args.prices_tech:
                tech_first = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX"]
                p_from = (args.prices_from or "").strip() or None
                p_to = (args.prices_to or "").strip() or None
                print("Fetching historical prices (tech-first, per-symbol)...")
                for sym in tech_first:
                    print(f"  - {sym}...")
                    try:
                        rows = fetch_historical_price_eod_full(client, symbol=sym, date_from=p_from, date_to=p_to)
                    except requests.HTTPError as e:
                        print(f"    WARN: historical-price-eod/full failed for {sym}: {e}")
                        time.sleep(0.15)
                        continue
                    # API returns a list of daily bars, each includes date/symbol/ohlcv.
                    # Insert each row into prices_eod using its own date.
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
                                sym,
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
                    conn.commit()
                    print(f"    Upserted prices_eod rows: {n_ins:,}")
                    time.sleep(0.15)

            # Prices for the full universe (incremental, resumable).
            if args.prices_universe:
                batch_size = max(1, int(args.prices_batch_size))
                window_days = max(5, int(args.prices_window_days))
                default_start = (args.prices_start or "").strip()[:10]
                try:
                    datetime.strptime(default_start, "%Y-%m-%d")
                except Exception:
                    default_start = "2016-01-01"

                end = _yesterday().isoformat()
                print(
                    "Incremental prices backfill (universe): "
                    f"batch={batch_size}, window_days={window_days}, end={end}"
                )

                # Persist universe for efficient selection and fair rotation across runs.
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS fmp_universe (
                      ticker TEXT PRIMARY KEY
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prices_backfill_progress (
                      ticker TEXT PRIMARY KEY,
                      last_price_date TEXT,
                      updated_at TEXT NOT NULL
                    );
                    """
                )
                conn.executemany(
                    "INSERT OR IGNORE INTO fmp_universe (ticker) VALUES (?)",
                    [(s,) for s in universe_full],
                )
                conn.commit()

                cursor_name = str(args.prices_cursor_name or "universe").strip() or "universe"
                last_cursor = get_prices_cursor(conn, cursor_name=cursor_name) if args.prices_sequential else None

                watchlist_filter = ""
                watch_params: tuple[Any, ...] = ()
                if used_watchlist and args.prices_universe:
                    watchlist_filter = "AND EXISTS (SELECT 1 FROM fmp_watchlist w WHERE w.watchlist_name = ? AND w.ticker = u.ticker)"
                    watch_params = (watchlist_name,)

                if args.prices_sequential:
                    # Alphabetical sweep from last cursor, wrapping around at end of universe.
                    head = [
                        str(r[0]).upper()
                        for r in conn.execute(
                            f"""
                            SELECT u.ticker
                            FROM fmp_universe u
                            WHERE u.ticker > ? {watchlist_filter}
                            ORDER BY u.ticker ASC
                            LIMIT ?
                            """,
                            ((last_cursor or ""), *watch_params, batch_size),
                        ).fetchall()
                        if r and r[0]
                    ]
                    need = batch_size - len(head)
                    tail: list[str] = []
                    if need > 0:
                        tail = [
                            str(r[0]).upper()
                            for r in conn.execute(
                                f"""
                                SELECT u.ticker
                                FROM fmp_universe u
                                WHERE 1=1 {watchlist_filter}
                                ORDER BY u.ticker ASC
                                LIMIT ?
                                """,
                                (*watch_params, need),
                            ).fetchall()
                            if r and r[0]
                        ]
                    batch = head + tail
                    next_first = repr(batch[0]) if batch else repr(None)
                    print(f"Sequential cursor: name={cursor_name} last={repr(last_cursor)} -> next_batch_first={next_first}")
                else:
                    # Batch selection rules:
                    # - tickers never processed (NULL last_price_date) first
                    # - then oldest last_price_date
                    # - then oldest updated_at (prevents repeatedly selecting the same names when dates tie)
                    # - stable tie-breaker by ticker
                    batch = [
                        str(r[0]).upper()
                        for r in conn.execute(
                            f"""
                            SELECT u.ticker
                            FROM fmp_universe u
                            LEFT JOIN prices_backfill_progress p
                              ON p.ticker = u.ticker
                            WHERE 1=1 {watchlist_filter}
                            ORDER BY
                              (p.last_price_date IS NOT NULL) ASC,
                              p.last_price_date ASC,
                              p.updated_at ASC,
                              u.ticker ASC
                            LIMIT ?
                            """,
                            (*watch_params, batch_size),
                        ).fetchall()
                        if r and r[0]
                    ]

                for i, sym in enumerate(batch, start=1):
                    last = get_prices_progress(conn, ticker=sym)
                    start = default_start if last is None else _add_days_iso(last, 1)
                    if start > end:
                        # Already caught up through `end`; mark progress so sequential mode doesn't spin here forever.
                        upsert_prices_progress(conn, ticker=sym, last_price_date=end)
                        conn.commit()
                        print(f"  [{i}/{len(batch)}] {sym}: already_up_to_date_through={end}")
                        continue
                    to = min(end, _add_days_iso(start, window_days - 1))

                    print(f"  [{i}/{len(batch)}] {sym}: {start} -> {to}")
                    try:
                        rows = fetch_historical_price_eod_full(client, symbol=sym, date_from=start, date_to=to)
                    except requests.HTTPError as e:
                        print(f"    WARN: price fetch failed for {sym}: {e}")
                        time.sleep(0.10)
                        continue

                    n_ins = 0
                    max_d: str | None = None
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
                                sym,
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
                        if max_d is None or d > max_d:
                            max_d = d

                    # Even if the API returns zero rows for this window, advance progress to `to`
                    # so we don't spin forever on symbols/years with no price history.
                    advance_to = max_d if max_d is not None else to
                    upsert_prices_progress(conn, ticker=sym, last_price_date=advance_to)
                    conn.commit()
                    print(f"    Upserted prices_eod rows: {n_ins:,} | progress={advance_to}")
                    time.sleep(0.10)

                if args.prices_sequential and batch:
                    set_prices_cursor(conn, cursor_name=cursor_name, last_ticker=batch[-1])
                    conn.commit()

            # EOD daily bulk backfill (date-based, huge dataset; track per day).
            if args.skip_eod:
                eod_days = 0
            else:
                eod_days = int(args.eod_days)
            if args.eod_from.strip():
                try:
                    start = datetime.strptime(args.eod_from.strip()[:10], "%Y-%m-%d").date()
                except Exception:
                    start = None
                if start is not None:
                    day = start
                    end = _yesterday()
                    if day > end:
                        print("EOD backfill skipped (eod-from is after yesterday).")
                    else:
                        print(f"EOD backfill: {day.isoformat()} -> {end.isoformat()} (skipping completed days)...")
                        while day <= end:
                            if _is_day_done(conn, "eod", day):
                                day += timedelta(days=1)
                                continue
                            print(f"Fetching eod-bulk for {day.isoformat()}...")
                            try:
                                rows = fetch_eod_bulk(client, day=day)
                            except requests.HTTPError as e:
                                print(f"WARN: eod-bulk failed for {day.isoformat()}: {e}")
                                print("WARN: Your subscription may restrict this endpoint. Use --skip-eod to suppress.")
                                break
                            print(f"Rows received: {len(rows):,}")
                            n_p = upsert_prices_eod(conn, day=day, rows=rows)
                            _mark_day_done(conn, "eod", day, len(rows))
                            conn.commit()
                            print(f"Upserted prices_eod rows: {n_p:,}")
                            day += timedelta(days=1)
            elif eod_days > 0:
                end = _yesterday()
                start = end - timedelta(days=max(0, eod_days - 1))
                print(f"EOD backfill (last {eod_days} days): {start.isoformat()} -> {end.isoformat()} (skipping completed days)...")
                day = start
                while day <= end:
                    if _is_day_done(conn, "eod", day):
                        day += timedelta(days=1)
                        continue
                    print(f"Fetching eod-bulk for {day.isoformat()}...")
                    try:
                        rows = fetch_eod_bulk(client, day=day)
                    except requests.HTTPError as e:
                        print(f"WARN: eod-bulk failed for {day.isoformat()}: {e}")
                        print("WARN: Your subscription may restrict this endpoint. Use --skip-eod to suppress.")
                        break
                    print(f"Rows received: {len(rows):,}")
                    n_p = upsert_prices_eod(conn, day=day, rows=rows)
                    _mark_day_done(conn, "eod", day, len(rows))
                    conn.commit()
                    print(f"Upserted prices_eod rows: {n_p:,}")
                    day += timedelta(days=1)

            # Quarter-based bulk statements (incremental backfill).
            processed = 0
            for q in quarters:
                if processed >= int(args.max_quarters_per_run):
                    print(f"Reached max quarters per run: {int(args.max_quarters_per_run)}. Stopping.")
                    break

                # Skip quarters already completed (no re-fetch).
                if (
                    _is_quarter_done(conn, "income", q)
                    and _is_quarter_done(conn, "cashflow", q)
                    and _is_quarter_done(conn, "balance_sheet", q)
                ):
                    print(f"Quarter already backfilled: {q}. Skipping.")
                    continue

                print(f"Fetching income-statement-bulk for {q}...")
                if _is_quarter_done(conn, "income", q):
                    inc = []
                    print("Rows received: 0 (skipped; already backfilled)")
                else:
                    inc = fetch_income_statement_bulk(client, q)
                    print(f"Rows received: {len(inc):,}")
                    _mark_quarter_done(conn, "income", q, len(inc))

                print(f"Fetching cash-flow-statement-bulk for {q}...")
                if _is_quarter_done(conn, "cashflow", q):
                    cf = []
                    print("Rows received: 0 (skipped; already backfilled)")
                else:
                    cf = fetch_cash_flow_bulk(client, q)
                    print(f"Rows received: {len(cf):,}")
                    _mark_quarter_done(conn, "cashflow", q, len(cf))

                print(f"Fetching balance-sheet-statement-bulk for {q}...")
                if _is_quarter_done(conn, "balance_sheet", q):
                    bs = []
                    print("Rows received: 0 (skipped; already backfilled)")
                else:
                    bs = fetch_balance_sheet_bulk(client, q)
                    print(f"Rows received: {len(bs):,}")
                    _mark_quarter_done(conn, "balance_sheet", q, len(bs))

                print("Upserting fundamental_data (merge income+cashflow)...")
                n_f = upsert_fundamentals_from_bulk(conn, income_rows=inc, cashflow_rows=cf)
                print(f"Upserted fundamental_data rows: {n_f:,}")

                print("Upserting balance_sheet...")
                n_b = upsert_balance_sheet_from_bulk(conn, bs)
                print(f"Upserted balance_sheet rows: {n_b:,}")

                conn.commit()
                processed += 1

        print("Done.")
    finally:
        if lock is not None:
            lock.release()


if __name__ == "__main__":
    main()
