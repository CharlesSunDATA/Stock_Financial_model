"""
update_fundamentals_fmp.py

【專業版】以 FinancialModelingPrep (FMP) API 為唯一資料源，
抓取季度基本面並寫入 SQLite（fundamental_data）。

重點特性：
1) requests 呼叫 FMP API
2) 智慧額度控管（Smart Caching）：若 DB 內該 ticker 最新一季距今 < 90 天則跳過
3) 合併損益表 + 現金流（以 date 欄位對齊）
4) INSERT OR IGNORE（依賴 UNIQUE(ticker, report_date) 防重複）
5) 支援單一 ticker 與批次清單；每檔 sleep 0.5 秒以示負責

用法：
  python3 scripts/update_fundamentals_fmp.py NVDA
  python3 scripts/update_fundamentals_fmp.py AAPL MSFT NVDA TSLA GOOGL
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

try:
    from pages.modules.fmp_key import get_fmp_key as _get_fmp_key
except ImportError:
    from modules.fmp_key import get_fmp_key as _get_fmp_key  # type: ignore

FMP_BASE_URL = "https://financialmodelingprep.com"


def _effective_api_key() -> str:
    return _get_fmp_key()


@dataclass(frozen=True)
class FundamentalRow:
    """單季資料列（以 report_date 對齊）。"""

    ticker: str
    report_date: str  # YYYY-MM-DD
    revenue: float | None
    eps: float | None
    free_cash_flow: float | None


def _project_root() -> Path:
    # scripts/update_fundamentals_fmp.py -> project root is parent of scripts/
    return Path(__file__).resolve().parents[1]


def _db_path() -> Path:
    # 依照你目前專案規格：資料庫放在 data/
    return _project_root() / "data" / "quant_data.db"


def _today() -> date:
    return date.today()

def _ensure_usage_table(conn: sqlite3.Connection) -> None:
    """確保 fmp_api_usage 表存在（避免使用者沒跑 init_db 也能自動補上）。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fmp_api_usage (
          usage_date TEXT PRIMARY KEY,
          calls_used INTEGER NOT NULL
        );
        """
    )


def _get_calls_used_today(conn: sqlite3.Connection) -> int:
    _ensure_usage_table(conn)
    d = _today().isoformat()
    cur = conn.execute("SELECT calls_used FROM fmp_api_usage WHERE usage_date = ?", (d,))
    row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _add_calls_used(conn: sqlite3.Connection, n: int) -> None:
    """將今日 calls_used 累加 n（以交易系統方式做原子 upsert）。"""
    if n <= 0:
        return
    _ensure_usage_table(conn)
    d = _today().isoformat()
    conn.execute(
        """
        INSERT INTO fmp_api_usage (usage_date, calls_used)
        VALUES (?, ?)
        ON CONFLICT(usage_date) DO UPDATE SET calls_used = calls_used + excluded.calls_used
        """,
        (d, int(n)),
    )


def _remaining_quota(conn: sqlite3.Connection, daily_quota: int = 250) -> int:
    used = _get_calls_used_today(conn)
    return max(0, int(daily_quota) - int(used))


def _parse_iso_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception:
        return None


def _safe_float(x: Any) -> float | None:
    """FMP JSON 通常很乾淨；這裡只做最基本的型別轉換與 None 保護。"""
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _get_json(url: str, params: dict[str, Any], timeout_s: int = 20) -> Any:
    r = requests.get(url, params=params, timeout=timeout_s)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # 讓你更快定位是 key/方案/端點問題（FMP 常在 body 回傳原因）
        body = ""
        try:
            body = r.text[:800]
        except Exception:
            body = ""
        raise requests.HTTPError(f"{e} | response_body={body}") from e
    return r.json()


def _income_stmt_url() -> str:
    # Stable endpoint (non-legacy)
    return f"{FMP_BASE_URL}/stable/income-statement"


def _cashflow_url() -> str:
    # Stable endpoint (non-legacy)
    return f"{FMP_BASE_URL}/stable/cash-flow-statement"


def fetch_income_statement_quarterly(ticker: str, limit: int = 4) -> list[dict[str, Any]]:
    """
    FMP Income Statement（quarter）
    欄位：date, revenue, epsdiluted
    """
    url = _income_stmt_url()
    params = {"symbol": ticker, "period": "quarter", "limit": int(limit), "apikey": _effective_api_key()}
    data = _get_json(url, params=params)
    return data if isinstance(data, list) else []


def fetch_cashflow_quarterly(ticker: str, limit: int = 4) -> list[dict[str, Any]]:
    """
    FMP Cash Flow Statement（quarter）
    欄位：date, freeCashFlow
    """
    url = _cashflow_url()
    params = {"symbol": ticker, "period": "quarter", "limit": int(limit), "apikey": _effective_api_key()}
    data = _get_json(url, params=params)
    return data if isinstance(data, list) else []


def merge_fundamentals(
    ticker: str,
    income_rows: list[dict[str, Any]],
    cashflow_rows: list[dict[str, Any]],
) -> list[FundamentalRow]:
    """
    以 date 欄位合併兩份資料，輸出乾淨列表。
    - revenue, epsdiluted 來自 income-statement
    - freeCashFlow 來自 cash-flow-statement
    """
    sym = ticker.strip().upper()
    inc_by_date: dict[str, dict[str, Any]] = {}
    cf_by_date: dict[str, dict[str, Any]] = {}

    for r in income_rows or []:
        d = str(r.get("date") or "").strip()
        if d:
            inc_by_date[d] = r
    for r in cashflow_rows or []:
        d = str(r.get("date") or "").strip()
        if d:
            cf_by_date[d] = r

    all_dates = sorted(set(inc_by_date.keys()) | set(cf_by_date.keys()))
    out: list[FundamentalRow] = []
    for d in all_dates:
        inc = inc_by_date.get(d, {})
        cf = cf_by_date.get(d, {})
        out.append(
            FundamentalRow(
                ticker=sym,
                report_date=d,
                revenue=_safe_float(inc.get("revenue")),
                # FMP stable keys are usually camelCase; keep a couple of fallbacks for safety.
                eps=_safe_float(inc.get("epsdiluted") if "epsdiluted" in inc else inc.get("epsDiluted")),
                free_cash_flow=_safe_float(cf.get("freeCashFlow") if "freeCashFlow" in cf else cf.get("free_cash_flow")),
            )
        )
    # 由舊到新（便於理解 log）
    out.sort(key=lambda x: x.report_date)
    return out


def latest_report_date_in_db(conn: sqlite3.Connection, ticker: str) -> date | None:
    """查詢 DB 中該 ticker 最新一季日期（report_date）。"""
    sym = ticker.strip().upper()
    cur = conn.execute(
        "SELECT MAX(report_date) FROM fundamental_data WHERE ticker = ?",
        (sym,),
    )
    row = cur.fetchone()
    if not row or not row[0]:
        return None
    return _parse_iso_date(str(row[0]))


def should_skip_fetch(conn: sqlite3.Connection, ticker: str, *, lookback_days: int = 90) -> tuple[bool, str]:
    """
    Smart Caching 核心邏輯：
    若 DB 內最新一季距今 < lookback_days，則跳過抓取以節省額度。
    """
    latest = latest_report_date_in_db(conn, ticker)
    if latest is None:
        return False, "資料庫無該 ticker 歷史資料，需抓取"

    age_days = (_today() - latest).days
    if age_days < int(lookback_days):
        return True, f"資料庫已有 {ticker.upper()} 最新資料（{latest.isoformat()}，距今 {age_days} 天），略過以節省額度"
    return False, f"資料庫最新資料已超過 {lookback_days} 天（{latest.isoformat()}，距今 {age_days} 天），允許抓取更新"


def insert_rows_ignore(conn: sqlite3.Connection, rows: list[FundamentalRow]) -> int:
    """
    INSERT OR IGNORE：
    - 若 UNIQUE(ticker, report_date) 已存在，則忽略（不消耗額度重覆寫）
    - 若不存在，則插入
    """
    n = 0
    for r in rows:
        conn.execute(
            """
            INSERT OR IGNORE INTO fundamental_data
              (ticker, report_date, revenue, eps, free_cash_flow)
            VALUES
              (?,      ?,          ?,       ?,   ?)
            """,
            (r.ticker, r.report_date, r.revenue, r.eps, r.free_cash_flow),
        )
        n += 1
    return n


def update_one_ticker(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    daily_quota: int = 250,
    log: callable | None = None,
) -> tuple[int, str]:
    """
    更新單一 ticker。
    回傳 (calls_consumed, message) 供 UI 顯示。
    calls_consumed 以「每次 endpoint 呼叫 1 call」計。
    """
    sym = ticker.strip().upper()
    if not sym:
        return 0, "空白 ticker，略過"

    def _log(s: str) -> None:
        if log is not None:
            try:
                log(s)
                return
            except Exception:
                pass
        print(s)

    skip, msg = should_skip_fetch(conn, sym, lookback_days=90)
    if skip:
        _log(f"🟨 {msg}")
        return 0, msg

    # Quota gate: each ticker may consume up to 2 calls (income + cashflow)
    remain = _remaining_quota(conn, daily_quota=daily_quota)
    if remain <= 0:
        msg2 = f"今日額度已用盡（{daily_quota} calls/day），停止抓取"
        _log(f"🟥 {msg2}")
        return 0, msg2
    if remain < 2:
        msg2 = f"今日剩餘額度不足（剩 {remain} call），避免超額，停止抓取"
        _log(f"🟥 {msg2}")
        return 0, msg2

    _log(f"⏳ 正在從 FMP 抓取 {sym}（損益表 + 現金流）…")

    calls = 0
    income: list[dict[str, Any]] = []
    cashflow: list[dict[str, Any]] = []
    try:
        # Income statement (1 call)
        try:
            income = fetch_income_statement_quarterly(sym, limit=4)
        finally:
            calls += 1
            _add_calls_used(conn, 1)

        # Cashflow (1 call)
        try:
            cashflow = fetch_cashflow_quarterly(sym, limit=4)
        finally:
            calls += 1
            _add_calls_used(conn, 1)
    except Exception as e:
        _log(f"🟥 {sym} 抓取失敗：{e}")
        conn.commit()
        return calls, f"{sym} 抓取失敗"

    rows = merge_fundamentals(sym, income, cashflow)
    if not rows:
        msg3 = f"{sym}：FMP 未回傳任何季度資料（可能 ticker 錯誤或 API 限制）"
        _log(f"🟨 {msg3}")
        conn.commit()
        return calls, msg3

    try:
        inserted_attempts = insert_rows_ignore(conn, rows)
        conn.commit()
        msg4 = f"成功從 FMP 抓取並存入 {sym} 近 {inserted_attempts} 季資料（INSERT OR IGNORE）"
        _log(f"✅ {msg4}")
        return calls, msg4
    except Exception as e:
        conn.rollback()
        _log(f"🟥 {sym}：寫入資料庫失敗：{e}")
        return calls, f"{sym} 寫入失敗"


def bulk_download_tech_first(
    db_path: Path,
    *,
    daily_quota: int = 250,
    sleep_s: float = 0.5,
    log: callable | None = None,
) -> list[str]:
    """
    科技股優先批次下載（避免超過每日額度）。
    會自動套用 Smart Caching（90 天內已更新則略過且不耗額度）。
    """
    tech_first = [
        # Mega-cap / Core tech
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        # Semis / infra
        "AVGO", "AMD", "INTC", "QCOM", "MU", "TXN", "ASML",
        # Software
        "ORCL", "CRM", "ADBE", "NOW", "SNOW", "UBER", "SHOP",
        # Cyber / data infra
        "PANW", "CRWD", "NET", "DDOG", "ZS",
    ]

    logs: list[str] = []

    def _log(s: str) -> None:
        logs.append(s)
        if log is not None:
            try:
                log(s)
            except Exception:
                pass

    with sqlite3.connect(str(db_path)) as conn:
        _ensure_usage_table(conn)
        _log(f"📌 今日已用 calls：{_get_calls_used_today(conn)} / {daily_quota}")
        for i, sym in enumerate(tech_first):
            remain = _remaining_quota(conn, daily_quota=daily_quota)
            if remain < 2:
                _log(f"🟥 今日剩餘額度 {remain} call，不足以抓取下一檔（需要 2 calls），停止。")
                break
            _calls, _ = update_one_ticker(conn, sym, daily_quota=daily_quota, log=_log)
            if i < len(tech_first) - 1:
                time.sleep(float(sleep_s))

        _log(f"📌 今日已用 calls：{_get_calls_used_today(conn)} / {daily_quota}")

    return logs


def main() -> None:
    """
    批次處理入口：
    - 支援 CLI tickers
    - 若不帶參數，則使用下方預設清單（可自行改）
    """
    ap = argparse.ArgumentParser(description="Update quarterly fundamentals using FMP (no yfinance).")
    ap.add_argument("tickers", nargs="*", help="Tickers, e.g. NVDA AAPL MSFT")
    args = ap.parse_args()

    # 你可以在這裡定義你的核心關注清單（未提供 CLI tickers 時使用）
    default_list = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]
    tickers = [t.strip().upper() for t in (args.tickers if args.tickers else default_list) if t.strip()]

    api_key = _effective_api_key()
    if api_key in ("", "你的_API_KEY"):
        print("🟥 找不到有效的 FMP API Key。")
        print("   方案 A：在程式開頭設定 `FMP_API_KEY = \"你的_API_KEY\"`（更換成你自己的 key）")
        print("   方案 B：用環境變數（推薦，避免把 key 寫進 repo）")
        print("          export FMP_API_KEY='你的key'")
        return

    db = _db_path()
    if not db.exists():
        print(f"🟨 找不到資料庫：{db}")
        print("   請先執行：python3 scripts/init_db.py")
        return

    print("—" * 72)
    print(f"📌 DB：{db}")
    print(f"📌 Tickers：{', '.join(tickers)}")
    print("📌 Smart Caching：若最新一季距今 < 90 天則略過（省額度）")
    print("—" * 72)

    with sqlite3.connect(str(db)) as conn:
        for i, sym in enumerate(tickers):
            update_one_ticker(conn, sym)
            if i < len(tickers) - 1:
                time.sleep(0.5)

    print("—" * 72)
    print("🎉 完成：FMP 基本面更新流程結束")
    print("—" * 72)


if __name__ == "__main__":
    main()
