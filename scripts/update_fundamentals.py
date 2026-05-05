"""
update_fundamentals.py

Fetch quarterly fundamentals from Yahoo Finance (yfinance) and upsert into SQLite.

Usage:
  python3 scripts/update_fundamentals.py NVDA
  python3 scripts/update_fundamentals.py NVDA AAPL MSFT
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

# Support both:
# - `streamlit run app.py` (imports from project root)
# - `python3 scripts/update_fundamentals.py ...` (executes with scripts/ on sys.path)
try:
    from scripts.init_db import init_db  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from init_db import init_db  # type: ignore


def _project_root() -> Path:
    # scripts/update_fundamentals.py -> project root is parent of scripts/
    return Path(__file__).resolve().parents[1]


def _db_path() -> Path:
    return _project_root() / "data" / "quant_data.db"


def _is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _c(text: str, code: str) -> str:
    if not _is_tty():
        return text
    return f"\033[{code}m{text}\033[0m"


def ok(text: str) -> str:
    return _c(text, "32")  # green


def warn(text: str) -> str:
    return _c(text, "33")  # yellow


def err(text: str) -> str:
    return _c(text, "31")  # red


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _normalize_quarterly_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    q = df.T.copy()
    q.index = pd.to_datetime(q.index, errors="coerce").tz_localize(None)
    q = q[~q.index.isna()].sort_index()
    for c in q.columns:
        q[c] = pd.to_numeric(q[c], errors="coerce")
    return q


def fetch_quarterly_fundamentals(ticker: str) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by report_date with columns:
      revenue, eps, free_cash_flow
    Missing values remain NaN and will be stored as NULL.
    """
    t = yf.Ticker(ticker)

    fin = _normalize_quarterly_df(getattr(t, "quarterly_financials", None))
    cf = _normalize_quarterly_df(getattr(t, "quarterly_cashflow", None))

    if fin.empty and cf.empty:
        return pd.DataFrame()

    idx = fin.index.union(cf.index).sort_values()
    out = pd.DataFrame(index=idx)

    # Revenue (income stmt)
    if not fin.empty:
        for name in ("Total Revenue", "Operating Revenue", "Revenue"):
            if name in fin.columns:
                out["revenue"] = fin[name]
                break
    if "revenue" not in out.columns:
        out["revenue"] = pd.NA

    # EPS (income stmt)
    if not fin.empty:
        for name in ("Diluted EPS", "Basic EPS"):
            if name in fin.columns:
                out["eps"] = fin[name]
                break
    if "eps" not in out.columns:
        out["eps"] = pd.NA

    # FCF (cashflow preferred)
    if not cf.empty and "Free Cash Flow" in cf.columns:
        out["free_cash_flow"] = cf["Free Cash Flow"]
    else:
        ocf = cf["Operating Cash Flow"] if (not cf.empty and "Operating Cash Flow" in cf.columns) else pd.Series(dtype=float)
        capex = cf["Capital Expenditure"] if (not cf.empty and "Capital Expenditure" in cf.columns) else pd.Series(dtype=float)
        if not ocf.empty or not capex.empty:
            out["free_cash_flow"] = ocf.reindex(idx) + capex.reindex(idx)
        else:
            out["free_cash_flow"] = pd.NA

    out.index.name = "report_date"
    return out


def _get_existing_id(conn: sqlite3.Connection, ticker: str, report_date: str) -> int | None:
    cur = conn.execute(
        "SELECT id FROM fundamental_data WHERE ticker = ? AND report_date = ? LIMIT 1",
        (ticker, report_date),
    )
    row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else None


def upsert_fundamentals(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> int:
    """
    Upsert rows into `fundamental_data`.
    Uses INSERT OR REPLACE, preserving existing `id` when present.
    """
    sym = ticker.strip().upper()
    if df is None or df.empty:
        return 0

    n = 0
    for dt, row in df.iterrows():
        if pd.isna(dt):
            continue
        report_date = pd.Timestamp(dt).date().isoformat()
        existing_id = _get_existing_id(conn, sym, report_date)

        revenue = _safe_float(row.get("revenue"))
        eps = _safe_float(row.get("eps"))
        fcf = _safe_float(row.get("free_cash_flow"))

        conn.execute(
            """
            INSERT OR REPLACE INTO fundamental_data
              (id, ticker, report_date, revenue, eps, free_cash_flow)
            VALUES
              (?,  ?,      ?,          ?,       ?,   ?)
            """,
            (existing_id, sym, report_date, revenue, eps, fcf),
        )
        n += 1

    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch quarterly fundamentals and upsert into data/quant_data.db")
    ap.add_argument("tickers", nargs="+", help="Tickers, e.g. NVDA AAPL MSFT")
    args = ap.parse_args(argv)

    db = init_db(_db_path())

    total_rows = 0
    with sqlite3.connect(str(db)) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        for raw in args.tickers:
            sym = raw.strip().upper()
            if not sym:
                continue
            print(f"⏳ {sym}: 正在抓取季度財報…")
            try:
                df = fetch_quarterly_fundamentals(sym)
            except Exception as e:
                print(warn(f"⚠️ {sym}: 抓取失敗：{e}"))
                continue

            if df.empty:
                print(warn(f"⚠️ {sym}: 沒有取得任何季度資料（可能是 ticker 錯誤或 Yahoo 無資料）。"))
                continue

            try:
                n = upsert_fundamentals(conn, sym, df)
                conn.commit()
                total_rows += n
                print(ok(f"✅ {sym}: 成功寫入 {n} 季資料至資料庫"))
            except Exception as e:
                conn.rollback()
                print(err(f"❌ {sym}: 寫入資料庫失敗：{e}"))

    print(ok(f"🎉 完成：總共寫入 {total_rows} 筆季度資料 → {db}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
