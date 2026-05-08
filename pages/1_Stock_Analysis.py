"""
Stock Analysis — combined Valuation + Quarterly Financials (single page, two tabs).
Sidebar: shared ticker, DCF params, financials source/limit.
"""

from __future__ import annotations

import inspect
import json
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from utils.local_data import (
    connect_readonly,
    default_db_path,
    latest_company_snapshot,
    load_adjusted_close,
    table_exists,
)
from utils.data_loader import fetch_quarterly_metrics

warnings.filterwarnings("ignore", category=FutureWarning)

_PLOTLY_CHART_SUPPORTS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters

# ── shared helpers ─────────────────────────────────────────────────────────────

def _sf(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default

def _db_path() -> Path:
    return default_db_path()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION A — VALUATION
# ══════════════════════════════════════════════════════════════════════════════

PEER_MAP: dict[str, list[str]] = {
    "NVDA": ["AMD", "AVGO", "INTC", "MU", "QCOM"],
    "AMD": ["NVDA", "AVGO", "INTC", "MU", "QCOM"],
    "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
    "MSFT": ["AAPL", "GOOGL", "META", "ORCL"],
    "GOOGL": ["META", "MSFT", "AMZN"],
    "META": ["GOOGL", "SNAP", "PINS"],
    "TSLA": ["F", "GM", "RIVN"],
    "JPM": ["BAC", "WFC", "C", "GS"],
}
SEMIS_PEERS = ["AMD", "AVGO", "INTC", "MU", "QCOM"]
SOFTWARE_PEERS = ["MSFT", "ORCL", "CRM", "ADBE"]


@dataclass
class MarketData:
    price: float | None
    trailing_pe: float | None
    forward_pe: float | None
    peg: float | None
    pb: float | None
    peer_avg_pe: float | None
    peer_symbols_used: tuple[str, ...]


@dataclass
class FundamentalInputs:
    fcf_base: float
    shares: float
    net_debt: float
    used_defaults: list[str]


@dataclass
class DCFResult:
    intrinsic_per_share: float
    band_low: float
    band_high: float
    enterprise_value: float
    pv_fcf_5y: float
    pv_terminal: float


def _valid_positive(value: float | None, *, max_value: float = 1000.0) -> float | None:
    if value is None or not np.isfinite(value) or value <= 0 or value >= max_value:
        return None
    return float(value)


def _json_float(payload: Any, key: str) -> float | None:
    if not payload:
        return None
    try:
        data = json.loads(str(payload))
    except (TypeError, json.JSONDecodeError):
        return None
    return _sf(data.get(key))


def _local_trailing_pe(ticker: str) -> float | None:
    sym = ticker.strip().upper()
    db = _db_path()
    if not sym or not db.exists():
        return None
    try:
        with connect_readonly(db) as conn:
            if table_exists(conn, "key_metrics_ttm"):
                row = conn.execute(
                    "SELECT payload_json FROM key_metrics_ttm WHERE ticker=? ORDER BY as_of_date DESC LIMIT 1",
                    (sym,),
                ).fetchone()
                if row:
                    ey = _json_float(row[0], "earningsYieldTTM")
                    if ey and ey > 0:
                        pe = 1.0 / ey
                        if 0 < pe < 800:
                            return float(pe)
            if table_exists(conn, "historical_financial_ratios"):
                row = conn.execute(
                    "SELECT payload_json FROM historical_financial_ratios WHERE ticker=? ORDER BY report_date DESC LIMIT 1",
                    (sym,),
                ).fetchone()
                if row and row[0]:
                    p = json.loads(row[0])
                    pe = _sf(p.get("priceToEarningsRatio"))
                    if pe and 0 < pe < 800:
                        return pe
    except Exception:
        return None
    return None


def _local_valuation_fields(ticker: str, price: float | None) -> dict[str, Any]:
    sym = ticker.strip().upper()
    db = _db_path()
    out: dict[str, Any] = {}
    if not sym or not db.exists():
        return out
    try:
        with connect_readonly(db) as conn:
            if table_exists(conn, "key_metrics_ttm"):
                row = conn.execute(
                    "SELECT ev_to_sales, payload_json FROM key_metrics_ttm WHERE ticker=? ORDER BY as_of_date DESC LIMIT 1",
                    (sym,),
                ).fetchone()
                if row:
                    ev_to_sales = _sf(row[0])
                    payload = row[1]
                    ey = _json_float(payload, "earningsYieldTTM")
                    if ey and ey > 0:
                        out["trailingPE"] = 1.0 / ey
                    if ev_to_sales and ev_to_sales > 0:
                        out["priceToSalesTrailing12Months"] = ev_to_sales
                    ev_ebitda = _json_float(payload, "evToEBITDATTM")
                    if ev_ebitda and ev_ebitda > 0:
                        out["enterpriseToEbitda"] = ev_ebitda
            if table_exists(conn, "historical_financial_ratios"):
                row = conn.execute(
                    "SELECT payload_json FROM historical_financial_ratios WHERE ticker=? ORDER BY report_date DESC LIMIT 1",
                    (sym,),
                ).fetchone()
                if row and row[0]:
                    p = json.loads(row[0])
                    pe = _sf(p.get("priceToEarningsRatio"))
                    pb = _sf(p.get("priceToBookRatio"))
                    ps = _sf(p.get("priceToSalesRatio"))
                    if pe and out.get("trailingPE") is None:
                        out["trailingPE"] = pe
                    if pb:
                        out["priceToBook"] = pb
                    if ps and out.get("priceToSalesTrailing12Months") is None:
                        out["priceToSalesTrailing12Months"] = ps
            current_date = date.today().isoformat()
            forward_eps = None
            for table, dc, ec in [
                ("analyst_estimates_detail", "estimate_date", "estimated_eps_avg"),
                ("analyst_estimates", "estimated_date", "estimated_eps"),
            ]:
                if not table_exists(conn, table):
                    continue
                row = conn.execute(
                    f"SELECT {ec} FROM {table} WHERE ticker=? AND {dc}>? AND {ec} IS NOT NULL AND {ec}>0 ORDER BY {dc} LIMIT 1",
                    (sym, current_date),
                ).fetchone()
                if row:
                    forward_eps = _sf(row[0])
                    break
            if price and forward_eps and forward_eps > 0:
                out["forwardPE"] = float(price) / float(forward_eps)
            if table_exists(conn, "income_statement"):
                eps_rows = conn.execute(
                    "SELECT report_date, COALESCE(eps_diluted,eps) FROM income_statement WHERE ticker=? AND COALESCE(eps_diluted,eps) IS NOT NULL ORDER BY report_date DESC LIMIT 8",
                    (sym,),
                ).fetchall()
                if len(eps_rows) >= 8:
                    latest_ttm = sum(float(r[1]) for r in eps_rows[:4])
                    prior_ttm = sum(float(r[1]) for r in eps_rows[4:8])
                    if latest_ttm > 0 and prior_ttm > 0:
                        growth_pct = (latest_ttm / prior_ttm - 1.0) * 100.0
                        trailing_pe = _sf(out.get("trailingPE"))
                        if trailing_pe and growth_pct > 0:
                            out["pegRatio"] = trailing_pe / growth_pct
    except Exception:
        return out
    return {k: v for k, v in out.items() if v is not None and np.isfinite(float(v))}


@st.cache_data(ttl=3600, show_spinner=False)
def v_fetch_info(ticker: str) -> dict[str, Any]:
    info = latest_company_snapshot(ticker)
    if info.get("currentPrice") is not None or info.get("marketCap") is not None:
        return info
    t = yf.Ticker(ticker)
    info = t.info
    if not info:
        raise ValueError(f"Could not load fundamentals for {ticker}.")
    return info


@st.cache_data(ttl=1800, show_spinner=False)
def v_fetch_close(ticker: str) -> float | None:
    quote = latest_company_snapshot(ticker)
    px = _sf(quote.get("currentPrice"))
    if px is not None:
        return px
    try:
        hist = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None


def compute_peer_average_pe(ticker: str, info: dict[str, Any]) -> tuple[float | None, tuple[str, ...]]:
    t_up = ticker.upper()
    peers = list(PEER_MAP.get(t_up, []))
    if not peers:
        ind = (info.get("industry") or "").lower()
        if any(x in ind for x in ("semiconductor",)):
            peers = [p for p in SEMIS_PEERS if p.upper() != t_up]
        elif "software" in ind:
            peers = [p for p in SOFTWARE_PEERS if p.upper() != t_up]
        else:
            peers = ["SPY", "QQQ"]

    @st.cache_data(ttl=3600, show_spinner=False)
    def _peer_pe(sym: str) -> float | None:
        local_pe = _local_trailing_pe(sym)
        if local_pe is not None:
            return local_pe
        try:
            inf = yf.Ticker(sym).info
            pe = _sf(inf.get("trailingPE"))
            if pe and 0 < pe < 800:
                return float(pe)
        except Exception:
            return None
        return None

    pes, used = [], []
    for sym in peers[:4]:
        if sym.upper() == t_up:
            continue
        try:
            pe = _peer_pe(sym.upper())
            if pe is not None:
                pes.append(float(pe))
                used.append(sym.upper())
        except Exception:
            continue
    if not pes:
        return None, tuple(peers)
    return float(np.mean(pes)), tuple(used)


def build_market_data(ticker: str, info: dict[str, Any], hist_close: float | None) -> MarketData:
    price = _sf(hist_close if hist_close is not None else info.get("currentPrice") or info.get("regularMarketPrice"))
    merged = dict(info)
    for key, value in _local_valuation_fields(ticker, price).items():
        if _sf(merged.get(key)) is None:
            merged[key] = value
    peer_avg, peer_used = compute_peer_average_pe(ticker, info)
    return MarketData(
        price=price,
        trailing_pe=_sf(merged.get("trailingPE")),
        forward_pe=_sf(merged.get("forwardPE")),
        peg=_sf(merged.get("pegRatio")),
        pb=_sf(merged.get("priceToBook")),
        peer_avg_pe=peer_avg,
        peer_symbols_used=peer_used,
    )


def _local_quarterly_eps_series(ticker: str) -> pd.Series | None:
    sym = ticker.strip().upper()
    db = _db_path()
    if not sym or not db.exists():
        return None
    try:
        with connect_readonly(db) as conn:
            if not table_exists(conn, "income_statement"):
                return None
            df = pd.read_sql_query(
                "SELECT report_date, COALESCE(eps_diluted,eps) AS eps FROM income_statement WHERE ticker=? AND COALESCE(eps_diluted,eps) IS NOT NULL ORDER BY report_date",
                conn, params=(sym,),
            )
    except Exception:
        return None
    if df.empty:
        return None
    s = pd.to_numeric(df["eps"], errors="coerce")
    idx = pd.to_datetime(df["report_date"], errors="coerce")
    out = pd.Series(s.to_numpy(), index=pd.DatetimeIndex(idx).tz_localize(None)).dropna().sort_index()
    return out if not out.empty else None


def _quarterly_eps_series(t: yf.Ticker) -> pd.Series | None:
    local = _local_quarterly_eps_series(str(getattr(t, "ticker", "") or ""))
    if local is not None and not local.empty:
        return local
    q = None
    for attr in ("quarterly_income_stmt", "quarterly_financials"):
        try:
            cand = getattr(t, attr, None)
            if cand is not None and not cand.empty:
                q = cand
                break
        except Exception:
            continue
    if q is None or q.empty:
        return None
    for name in ("Diluted EPS", "Basic EPS", "Normalized EPS"):
        if name in q.index:
            s = pd.to_numeric(q.loc[name], errors="coerce").dropna()
            if s.empty:
                continue
            dti = pd.to_datetime(s.index)
            s.index = pd.DatetimeIndex(dti).tz_localize(None)
            return s.sort_index()
    return None


def _annual_diluted_eps_series(t: yf.Ticker) -> pd.Series | None:
    for attr in ("income_stmt", "financials"):
        try:
            inc = getattr(t, attr, None)
            if inc is None or inc.empty:
                continue
            for name in ("Diluted EPS", "Basic EPS"):
                if name not in inc.index:
                    continue
                s = pd.to_numeric(inc.loc[name], errors="coerce").dropna()
                if s.empty:
                    continue
                dti = pd.to_datetime(s.index)
                s.index = pd.DatetimeIndex(dti).tz_localize(None)
                return s.sort_index()
        except Exception:
            continue
    return None


def _ttm_from_quarterly_eps(eps_q: pd.Series) -> pd.Series:
    eps_q = eps_q.sort_index()
    cols = list(eps_q.index)
    ttm_idx, ttm_val = [], []
    for j in range(len(cols)):
        if j >= 3:
            ttm = float(eps_q.iloc[j - 3: j + 1].sum())
        elif j == 2:
            ttm = float(eps_q.iloc[0:3].sum()) * (4.0 / 3.0)
        elif j == 1:
            ttm = float(eps_q.iloc[0:2].sum()) * 2.0
        else:
            ttm = float(eps_q.iloc[0]) * 4.0
        if ttm > 0:
            ttm_idx.append(pd.Timestamp(cols[j]))
            ttm_val.append(ttm)
    if not ttm_val:
        return pd.Series(dtype=float)
    return pd.Series(ttm_val, index=pd.DatetimeIndex(ttm_idx)).sort_index()


def _price_monthly(t: yf.Ticker, years: int) -> pd.Series | None:
    ticker = getattr(t, "ticker", "")
    if ticker:
        end = date.today()
        start = end - timedelta(days=int(years * 366 + 45))
        local = load_adjusted_close([str(ticker).upper()], start, end, min_rows=6)
        sym = str(ticker).upper()
        if sym in local.columns:
            close_m = local[sym].dropna().resample("ME").last().dropna()
            close_m.index = pd.DatetimeIndex(pd.to_datetime(close_m.index)).tz_localize(None)
            if not close_m.empty:
                return close_m
    hist = t.history(period=f"{years}y", interval="1mo", auto_adjust=True)
    if hist is None or hist.empty:
        return None
    close_m = hist["Close"].copy()
    close_m.index = pd.DatetimeIndex(pd.to_datetime(close_m.index)).tz_localize(None)
    return close_m


def _pe_from_eps(close_m: pd.Series, eps_s: pd.Series, min_periods: int) -> pd.DataFrame | None:
    rows = []
    for dt, px in close_m.items():
        dt_ts = pd.Timestamp(dt)
        eps_v = eps_s.asof(dt_ts)
        if eps_v is None or (isinstance(eps_v, float) and np.isnan(eps_v)):
            continue
        eps_f = float(eps_v)
        if eps_f <= 0 or float(px) <= 0:
            continue
        pe = float(px) / eps_f
        if 0 < pe < 500:
            rows.append({"date": dt_ts, "close": float(px), "eps": eps_f, "pe": pe})
    if len(rows) < min_periods:
        return None
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def build_monthly_pe_history(ticker: str, years: int = 5) -> pd.DataFrame | None:
    t = yf.Ticker(ticker)
    close_m = _price_monthly(t, years)
    if close_m is None:
        return None
    eps_q = _quarterly_eps_series(t)
    if eps_q is not None and len(eps_q) >= 1:
        ttm_s = _ttm_from_quarterly_eps(eps_q)
        if len(ttm_s) > 0:
            out = _pe_from_eps(close_m, ttm_s, min_periods=6)
            if out is not None:
                out.attrs["pe_basis"] = "TTM (quarterly EPS)"
                return out
    eps_a = _annual_diluted_eps_series(t)
    if eps_a is not None and len(eps_a) >= 1:
        out = _pe_from_eps(close_m, eps_a, min_periods=6)
        if out is not None:
            out.attrs["pe_basis"] = "FY diluted EPS"
            return out
    return None


def _local_quarterly_amount_series(ticker: str, table: str, column: str) -> pd.Series | None:
    sym = ticker.strip().upper()
    db = _db_path()
    if not sym or not db.exists():
        return None
    try:
        with connect_readonly(db) as conn:
            if not table_exists(conn, table):
                return None
            df = pd.read_sql_query(
                f"SELECT report_date, {column} AS value FROM {table} WHERE ticker=? AND {column} IS NOT NULL ORDER BY report_date",
                conn, params=(sym,),
            )
    except Exception:
        return None
    if df.empty:
        return None
    values = pd.to_numeric(df["value"], errors="coerce")
    idx = pd.to_datetime(df["report_date"], errors="coerce")
    out = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(idx).tz_localize(None)).dropna().sort_index()
    return out if not out.empty else None


def _ttm_amounts(q: pd.Series) -> pd.Series:
    q = q.sort_index()
    cols = list(q.index)
    idx, val = [], []
    for j in range(len(cols)):
        if j >= 3:
            ttm = float(q.iloc[j - 3: j + 1].sum())
        elif j == 2:
            ttm = float(q.iloc[0:3].sum()) * (4.0 / 3.0)
        elif j == 1:
            ttm = float(q.iloc[0:2].sum()) * 2.0
        else:
            ttm = float(q.iloc[0]) * 4.0
        idx.append(pd.Timestamp(cols[j]))
        val.append(ttm)
    return pd.Series(val, index=pd.DatetimeIndex(idx)).sort_index()


def _local_latest_shares(ticker: str) -> float | None:
    sym = ticker.strip().upper()
    db = _db_path()
    if not sym or not db.exists():
        return None
    try:
        with connect_readonly(db) as conn:
            for tbl, col in [("balance_sheet", "outstanding_shares"), ("income_statement", "weighted_avg_shares_out")]:
                if table_exists(conn, tbl):
                    row = conn.execute(
                        f"SELECT {col} FROM {tbl} WHERE ticker=? AND {col} IS NOT NULL AND {col}>0 ORDER BY report_date DESC LIMIT 1",
                        (sym,),
                    ).fetchone()
                    if row:
                        sh = _sf(row[0])
                        if sh and sh > 0:
                            return sh
    except Exception:
        return None
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def build_monthly_ps_history(ticker: str, years: int = 5) -> pd.DataFrame | None:
    t = yf.Ticker(ticker)
    close_m = _price_monthly(t, years)
    if close_m is None or close_m.empty:
        return None
    rev_q = _local_quarterly_amount_series(ticker, "income_statement", "revenue")
    if rev_q is None or rev_q.empty:
        return None
    rev_ttm = _ttm_amounts(rev_q)
    shares = _local_latest_shares(ticker)
    if shares is None or shares <= 0:
        shares = _sf(t.info.get("sharesOutstanding"))
    if shares is None or shares <= 0:
        return None
    rows = []
    for dt, px in close_m.items():
        dt_ts = pd.Timestamp(dt)
        rev = rev_ttm.asof(dt_ts)
        if rev is None or (isinstance(rev, float) and np.isnan(rev)) or float(rev) <= 0:
            continue
        pxf = float(px)
        if pxf <= 0:
            continue
        ps = (pxf * float(shares)) / float(rev)
        rows.append({"date": dt_ts, "ps": None if not np.isfinite(ps) else ps})
    if len(rows) < 6:
        return None
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _local_fundamental_inputs(ticker: str) -> FundamentalInputs | None:
    sym = ticker.strip().upper()
    db = _db_path()
    if not sym or not db.exists():
        return None
    try:
        with connect_readonly(db) as conn:
            if not (table_exists(conn, "cash_flow_statement") and table_exists(conn, "balance_sheet")):
                return None
            cf_rows = conn.execute(
                "SELECT operating_cash_flow, capital_expenditure, free_cash_flow FROM cash_flow_statement WHERE ticker=? ORDER BY report_date DESC LIMIT 4",
                (sym,),
            ).fetchall()
            bs = conn.execute(
                "SELECT cash_and_equivalents, outstanding_shares, total_debt FROM balance_sheet WHERE ticker=? ORDER BY report_date DESC LIMIT 1",
                (sym,),
            ).fetchone()
            shares_row = None
            if table_exists(conn, "income_statement"):
                shares_row = conn.execute(
                    "SELECT weighted_avg_shares_out FROM income_statement WHERE ticker=? AND weighted_avg_shares_out IS NOT NULL AND weighted_avg_shares_out>0 ORDER BY report_date DESC LIMIT 1",
                    (sym,),
                ).fetchone()
    except Exception:
        return None
    if not cf_rows:
        return None
    fcf_values = []
    for row in cf_rows:
        fcf = _sf(row[2])
        if fcf is None:
            fcf = (_sf(row[0], 0.0) or 0.0) + (_sf(row[1], 0.0) or 0.0)
        if fcf is not None and np.isfinite(fcf):
            fcf_values.append(float(fcf))
    if not fcf_values:
        return None
    fcf_base = sum(fcf_values)
    cash = _sf(bs[0], 0.0) if bs else 0.0
    shares = _sf(bs[1]) if bs else None
    if shares is None or shares <= 0:
        shares = _sf(shares_row[0]) if shares_row else None
    debt = _sf(bs[2], 0.0) if bs else 0.0
    notes = ["Fundamental inputs from local SQLite"]
    if len(fcf_values) >= 4:
        notes.append("FCF = sum of latest 4 quarters (TTM)")
    else:
        notes.append(f"FCF = {len(fcf_values)} quarter(s) available")
    if shares is None or shares <= 0:
        notes.append("Shares outstanding not found; denominator = 1")
        shares = 1.0
    return FundamentalInputs(
        fcf_base=float(fcf_base),
        shares=max(float(shares), 1e-12),
        net_debt=float(debt or 0.0) - float(cash or 0.0),
        used_defaults=notes,
    )


def build_fundamental_inputs(t: yf.Ticker, info: dict[str, Any]) -> FundamentalInputs:
    local = _local_fundamental_inputs(str(getattr(t, "ticker", "") or ""))
    if local is not None:
        return local
    used_defaults: list[str] = []
    cf = None
    try:
        cf = t.cashflow
    except Exception:
        used_defaults.append("Cash flow statement download failed")

    def _av(df, *names):
        if df is None or df.empty:
            return None
        for n in names:
            if n in df.index:
                for col in df.columns:
                    v = df.loc[n, col]
                    if v is not None and not (isinstance(v, float) and pd.isna(v)):
                        return float(v)
        return None

    ocf = _av(cf, "Operating Cash Flow", "Cash From Operating Activities")
    capex = _av(cf, "Capital Expenditure", "Capital Expenditures")
    if ocf is None:
        ocf = 0.0
        used_defaults.append("OCF missing; set to 0")
    if capex is None:
        capex = 0.0
        used_defaults.append("CapEx missing; set to 0")
    fcf_base = float(ocf) + float(capex)
    shares = _sf(info.get("sharesOutstanding"))
    if shares is None or shares <= 0:
        shares = 1.0
        used_defaults.append("Shares outstanding missing; denominator = 1")
    total_debt = _sf(info.get("totalDebt")) or 0.0
    total_cash = _sf(info.get("totalCash")) or 0.0
    return FundamentalInputs(
        fcf_base=fcf_base,
        shares=max(float(shares), 1e-12),
        net_debt=float(total_debt) - float(total_cash),
        used_defaults=used_defaults,
    )


def run_two_stage_dcf(fcf0, growth_5y, wacc, terminal_growth, net_debt, shares) -> DCFResult | None:
    if shares <= 0:
        return None
    if wacc <= terminal_growth:
        raise ValueError("WACC must exceed terminal growth rate.")
    pv_fcf = 0.0
    fcf_prev = fcf0
    for t in range(1, 6):
        fcf_t = fcf_prev * (1.0 + growth_5y)
        pv_fcf += fcf_t / ((1.0 + wacc) ** t)
        fcf_prev = fcf_t
    tv = fcf_prev * (1.0 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = tv / ((1.0 + wacc) ** 5)
    ev = pv_fcf + pv_terminal
    intrinsic = (ev - net_debt) / shares
    return DCFResult(
        intrinsic_per_share=intrinsic,
        band_low=intrinsic * 0.9,
        band_high=intrinsic * 1.1,
        enterprise_value=ev,
        pv_fcf_5y=pv_fcf,
        pv_terminal=pv_terminal,
    )


def _zone_bar(title, subtitle, current, hist: pd.Series, fmt, suffix=""):
    s = hist.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if current is None or not np.isfinite(current) or current <= 0 or len(s) < 6:
        return None
    q25, q50, q75 = (float(x) for x in np.percentile(s, [25, 50, 75]))
    cur = float(current)
    zone_label, zone_color = ("Low ✅", "#2ecc71") if cur < q25 else (("Mid 🟡", "#f1c40f") if cur < q75 else ("High ⚠️", "#e74c3c"))
    x_min = max(0.0, float(s.min())) * 0.90
    x_max = max(float(s.max()), q75) * 1.35
    bar_y0, bar_y1 = 0.30, 0.70
    fig = go.Figure()
    for x0, x1, fill, border in [
        (x_min, q25, "rgba(46,204,113,0.55)", "#2ecc71"),
        (q25, q75, "rgba(241,196,15,0.45)", "#f1c40f"),
        (q75, x_max, "rgba(231,76,60,0.50)", "#e74c3c"),
    ]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=bar_y0, y1=bar_y1, fillcolor=fill, line=dict(color=border, width=1), layer="below")
    fig.add_shape(type="line", x0=cur, x1=cur, y0=bar_y0, y1=bar_y1, line=dict(color="#3498db", width=3))
    fig.add_trace(go.Scatter(
        x=[cur], y=[bar_y1 + 0.22],
        mode="markers+text",
        marker=dict(symbol="diamond", size=12, color="#3498db", line=dict(color="white", width=1.5)),
        text=[f"<b>{format(cur, fmt)}{suffix}</b>"],
        textposition="middle right",
        textfont=dict(size=13, color="#3498db"),
        hovertemplate=f"Current: {format(cur, fmt)}{suffix}<extra></extra>",
        showlegend=False,
    ))
    for xc, col, label in [
        ((x_min + q25) / 2, "#2ecc71", f"Below P25<br><b>{format(q25, fmt)}{suffix}</b>"),
        ((q25 + q75) / 2, "#f1c40f", f"P25–P75<br><b>{format(q25, fmt)}–{format(q75, fmt)}{suffix}</b>"),
        ((q75 + x_max) / 2, "#e74c3c", f"Above P75<br><b>{format(q75, fmt)}{suffix}</b>"),
    ]:
        fig.add_annotation(x=xc, y=bar_y0 - 0.04, text=label, showarrow=False, font=dict(size=11, color=col), yanchor="top", xanchor="center")
    fig.update_layout(
        title=dict(
            text=f"<b>{title}: <span style='color:{zone_color}'>{format(cur, fmt)}{suffix} — {zone_label}</span></b><br><span style='font-size:11px;color:#aaa'>{subtitle}</span>",
            font=dict(size=14, color="white"), x=0, xanchor="left",
        ),
        xaxis=dict(range=[x_min, x_max], showgrid=False, zeroline=False, tickfont=dict(size=11, color="#ccc")),
        yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False),
        height=220, margin=dict(t=65, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
    )
    return fig


def render_valuation_tab(ticker: str, g5: float, g_term: float, wacc: float) -> None:
    try:
        info = v_fetch_info(ticker)
        t = yf.Ticker(ticker)
        close_px = None
        if not (info.get("currentPrice") or info.get("regularMarketPrice")):
            close_px = v_fetch_close(ticker)
        market = build_market_data(ticker, info, close_px)
        fundamentals = build_fundamental_inputs(t, info)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return

    dcf: DCFResult | None = None
    dcf_error: str | None = None
    try:
        dcf = run_two_stage_dcf(fundamentals.fcf_base, g5, wacc, g_term, fundamentals.net_debt, fundamentals.shares)
    except ValueError as e:
        dcf_error = str(e)
    except Exception as e:
        dcf_error = f"DCF error: {e}"

    # ── relative valuation ──
    st.subheader("Relative Valuation")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price", f"${market.price:,.2f}" if market.price else "—")
    c2.metric("Trailing P/E", f"{market.trailing_pe:.1f}" if market.trailing_pe else "—")
    c3.metric("Forward P/E", f"{market.forward_pe:.1f}" if market.forward_pe else "—")
    c4.metric("PEG", f"{market.peg:.2f}" if market.peg else "—")
    c5.metric("P/B", f"{market.pb:.1f}" if market.pb else "—")
    c6.metric("Peer avg P/E", f"{market.peer_avg_pe:.1f}" if market.peer_avg_pe else "—")
    st.caption(f"Peers used: {', '.join(market.peer_symbols_used) if market.peer_symbols_used else 'none'}")

    # ── P/E zone ──
    st.markdown("##### Trailing P/E vs. 5-year historical range")
    try:
        hist_df = build_monthly_pe_history(ticker, years=5)
    except Exception:
        hist_df = None

    if hist_df is not None and market.trailing_pe:
        s = hist_df["pe"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        fig_pe = _zone_bar(
            f"{ticker} Trailing P/E",
            hist_df.attrs.get("pe_basis", "~5y monthly"),
            market.trailing_pe, s, ".1f", "x",
        )
        if fig_pe:
            st.plotly_chart(fig_pe, use_container_width=True)
    else:
        st.info("Insufficient historical P/E data to render zone chart.")

    # ── P/S zone ──
    st.markdown("##### P/S (Price-to-Sales) vs. 5-year historical range")
    try:
        ps_df = build_monthly_ps_history(ticker, years=5)
    except Exception:
        ps_df = None
    cur_ps = _sf(info.get("priceToSalesTrailing12Months"))
    if ps_df is not None and not ps_df["ps"].dropna().empty:
        if cur_ps is None:
            cur_ps = float(ps_df["ps"].dropna().iloc[-1])
        fig_ps = _zone_bar(f"{ticker} P/S", "~5y monthly", cur_ps, ps_df["ps"].dropna(), ".2f", "x")
        if fig_ps:
            st.plotly_chart(fig_ps, use_container_width=True)
    else:
        st.info("Insufficient revenue data to build P/S history.")

    st.divider()

    # ── DCF ──
    st.subheader("Two-Stage DCF Model")
    ca, cb = st.columns(2)
    with ca:
        st.markdown(
            f"| Item | Value |\n|---|---|\n"
            f"| Base FCF (TTM, USD) | {fundamentals.fcf_base:,.0f} |\n"
            f"| Shares outstanding | {fundamentals.shares:,.0f} |\n"
            f"| Net debt (USD) | {fundamentals.net_debt:,.0f} |"
        )
    with cb:
        st.markdown(
            f"| DCF input | Setting |\n|---|---|\n"
            f"| 5-year FCF growth | {g5*100:.1f}% |\n"
            f"| Terminal growth | {g_term*100:.1f}% |\n"
            f"| WACC | {wacc*100:.1f}% |"
        )

    if dcf_error:
        st.error(dcf_error)
    elif dcf is not None:
        if dcf.intrinsic_per_share <= 0:
            st.error("⚠️ Negative intrinsic value — negative FCF or excessive net debt. Use relative multiples instead.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Intrinsic Value / share", f"${dcf.intrinsic_per_share:,.2f}")
            m2.metric("Fair Band Low (−10%)", f"${dcf.band_low:,.2f}")
            m3.metric("Fair Band High (+10%)", f"${dcf.band_high:,.2f}")
            st.caption(f"EV ≈ {dcf.enterprise_value:,.0f} USD  ·  PV(5y FCF): {dcf.pv_fcf_5y:,.0f}  ·  PV(terminal): {dcf.pv_terminal:,.0f}")

            if market.price:
                spot = float(market.price)
                intrinsic = float(dcf.intrinsic_per_share)
                if spot > dcf.band_high:
                    st.error(f"⚠️ Spot ${spot:,.2f} is above fair band (≤ ${dcf.band_high:,.2f})")
                elif spot < dcf.band_low:
                    st.success(f"✅ Spot ${spot:,.2f} is below fair band (≥ ${dcf.band_low:,.2f}) — margin of safety")
                else:
                    st.info(f"🔵 Spot ${spot:,.2f} is within fair band (${dcf.band_low:,.2f} – ${dcf.band_high:,.2f})")

                prem_pct = (spot / intrinsic - 1) * 100
                x_half = max(50.0, abs(prem_pct) * 1.25)
                x_min_p, x_max_p = -x_half, x_half
                band_lo_pct, band_hi_pct = -10.0, 10.0
                bar_y0, bar_y1 = 0.25, 0.75
                fig2 = go.Figure()
                for x0, x1, fill, bord in [
                    (x_min_p, band_lo_pct, "rgba(46,204,113,0.45)", "#2ecc71"),
                    (band_lo_pct, band_hi_pct, "rgba(241,196,15,0.45)", "#f1c40f"),
                    (band_hi_pct, x_max_p, "rgba(231,76,60,0.45)", "#e74c3c"),
                ]:
                    fig2.add_shape(type="rect", x0=x0, x1=x1, y0=bar_y0, y1=bar_y1, fillcolor=fill, line=dict(color=bord, width=1))
                fig2.add_shape(type="line", x0=0, x1=0, y0=bar_y0 - 0.05, y1=bar_y1 + 0.05, line=dict(color="rgba(255,255,255,0.6)", width=2, dash="dot"))
                spot_x = max(x_min_p * 0.98, min(x_max_p * 0.98, prem_pct))
                fig2.add_shape(type="line", x0=spot_x, x1=spot_x, y0=bar_y0, y1=bar_y1, line=dict(color="#3498db", width=3))
                fig2.add_trace(go.Scatter(
                    x=[spot_x], y=[bar_y1 + 0.18],
                    mode="markers+text",
                    marker=dict(symbol="diamond", size=12, color="#3498db", line=dict(color="white", width=1.5)),
                    text=[f"<b>{prem_pct:+.1f}%</b>  (${spot:,.0f})"],
                    textposition="middle right", textfont=dict(size=12, color="#3498db"),
                    hovertemplate=f"Premium: {prem_pct:+.1f}%<extra></extra>", showlegend=False,
                ))
                fig2.add_annotation(x=0, y=bar_y1 + 0.08, text=f"Intrinsic<br>${intrinsic:,.0f}  (0%)", showarrow=False, font=dict(size=10, color="rgba(255,255,255,0.7)"), yanchor="bottom", xanchor="center")
                fig2.update_layout(
                    title=dict(text="Premium / Discount to Intrinsic Value", font=dict(size=13, color="white"), x=0),
                    xaxis=dict(range=[x_min_p, x_max_p], ticksuffix="%", showgrid=False, zeroline=False, tickfont=dict(size=11, color="#ccc")),
                    yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False),
                    height=210, margin=dict(t=40, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
                )
                st.plotly_chart(fig2, use_container_width=True)

    if fundamentals.used_defaults:
        with st.expander("Data notes", expanded=False):
            for line in fundamentals.used_defaults:
                st.caption(line)
    st.caption("For education only — not investment advice.")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION B — QUARTERLY FINANCIALS
# ══════════════════════════════════════════════════════════════════════════════

DISPLAY_COLS: list[tuple[str, str]] = [
    ("period_end", "Period end"), ("fiscal_period", "FY period"),
    ("total_revenue", "Revenue"), ("revenue_qoq_pct", "Rev QoQ %"), ("revenue_yoy_pct", "Rev YoY %"),
    ("gross_margin_pct", "Gross Margin %"), ("operating_margin_pct", "Op. Margin %"), ("net_margin_pct", "Net Margin %"),
    ("ebitda", "EBITDA"), ("diluted_eps", "Diluted EPS"), ("eps_yoy_pct", "EPS YoY %"),
    ("operating_cash_flow", "Operating CF"), ("free_cash_flow", "FCF"), ("fcf_margin_pct", "FCF Margin %"),
    ("capex_pct_revenue", "Capex % Rev"), ("rd_pct_revenue", "R&D % Rev"),
    ("cash_and_equivalents", "Cash"), ("total_debt", "Total Debt"), ("net_debt", "Net Debt"),
    ("debt_to_assets_pct", "Debt/Assets %"),
]


def _payload_float(payload_json: Any, *keys: str) -> float | None:
    if not payload_json:
        return None
    try:
        payload = json.loads(str(payload_json))
    except (TypeError, json.JSONDecodeError):
        return None
    for key in keys:
        if key in payload:
            val = _sf(payload.get(key))
            if val is not None:
                return val
    return None


def _reported_float(payload_json: Any, *keys: str) -> float | None:
    if not payload_json:
        return None
    try:
        payload = json.loads(str(payload_json))
    except (TypeError, json.JSONDecodeError):
        return None
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    data = {str(k).lower(): v for k, v in (data or {}).items()}
    for key in keys:
        val = _sf(data.get(key.lower()))
        if val is not None:
            return val
    return None


def _reported_total_debt(payload_json: Any) -> float | None:
    direct = _reported_float(payload_json, "totalDebt", "debt")
    if direct is not None:
        return direct
    pieces = [
        _reported_float(payload_json, "shortTermDebt", "commercialPaper"),
        _reported_float(payload_json, "longTermDebtCurrent"),
        _reported_float(payload_json, "longTermDebtNonCurrent", "longTermDebt"),
    ]
    vals = [v for v in pieces if v is not None]
    return float(sum(vals)) if vals else None


def _fmt_money(v: Any) -> str:
    x = _sf(v)
    if x is None:
        return "-"
    ax = abs(x)
    if ax >= 1e12: return f"{x/1e12:,.2f}T"
    if ax >= 1e9: return f"{x/1e9:,.2f}B"
    if ax >= 1e6: return f"{x/1e6:,.1f}M"
    return f"{x:,.0f}"

def _fmt_pct(v: Any) -> str:
    x = _sf(v)
    return "-" if x is None else f"{x:,.1f}%"

def _fmt_num(v: Any, digits: int = 2) -> str:
    x = _sf(v)
    return "-" if x is None else f"{x:,.{digits}f}"

def _bar_colors(series: pd.Series) -> list[str]:
    return ["#2ecc71" if (pd.notna(v) and v >= 0) else "#e74c3c" for v in series]

def _metric_delta(current: Any, previous: Any, *, pct: bool = False) -> str | None:
    c, p = _sf(current), _sf(previous)
    if c is None or p is None:
        return None
    diff = c - p
    if pct:
        return f"{diff:+.1f} pp"
    if p == 0:
        return None
    return f"{diff/abs(p)*100:+.1f}% QoQ"


@st.cache_data(ttl=600, show_spinner=False)
def available_db_tickers() -> list[str]:
    db = _db_path()
    if not db.exists():
        return []
    with connect_readonly(db) as conn:
        try:
            df = pd.read_sql_query(
                "SELECT DISTINCT ticker FROM income_statement UNION SELECT DISTINCT ticker FROM cash_flow_statement ORDER BY ticker",
                conn,
            )
        except Exception:
            return []
    return [str(x).upper() for x in df["ticker"].dropna().tolist()]


@st.cache_data(ttl=600, show_spinner=False)
def fetch_db_quarterly_metrics(ticker: str, limit: int = 12) -> pd.DataFrame:
    db = _db_path()
    sym = ticker.strip().upper()
    if not db.exists() or not sym:
        return pd.DataFrame()
    with connect_readonly(db) as conn:
        inc = pd.read_sql_query(
            "SELECT ticker,report_date,period,revenue,gross_profit,operating_income,ebitda,net_income,eps,eps_diluted,weighted_avg_shares_out,payload_json FROM income_statement WHERE ticker=? ORDER BY report_date ASC",
            conn, params=(sym,),
        )
        cf = pd.read_sql_query(
            "SELECT ticker,report_date,period,operating_cash_flow,capital_expenditure,free_cash_flow,investing_cash_flow,financing_cash_flow,net_change_in_cash,payload_json FROM cash_flow_statement WHERE ticker=? ORDER BY report_date ASC",
            conn, params=(sym,),
        )
        bs = pd.read_sql_query(
            "SELECT ticker,report_date,cash_and_equivalents,total_assets,total_liabilities,outstanding_shares,total_debt FROM balance_sheet WHERE ticker=? ORDER BY report_date ASC",
            conn, params=(sym,),
        )
        bs_rep = pd.read_sql_query(
            "SELECT ticker,report_date,payload_json AS balance_sheet_payload FROM balance_sheet_as_reported WHERE ticker=? ORDER BY report_date ASC",
            conn, params=(sym,),
        ) if table_exists(conn, "balance_sheet_as_reported") else pd.DataFrame()

    if inc.empty and cf.empty and bs.empty:
        return pd.DataFrame()

    dates = pd.concat(
        [x[["report_date"]] for x in [inc, cf, bs] if not x.empty],
        ignore_index=True,
    ).drop_duplicates()
    dates["period_end"] = pd.to_datetime(dates["report_date"], errors="coerce")
    dates = dates.dropna(subset=["period_end"]).sort_values("period_end")
    if dates.empty:
        return pd.DataFrame()

    inc = inc.rename(columns={"period":"fiscal_period","revenue":"total_revenue","eps_diluted":"diluted_eps","eps":"basic_eps","weighted_avg_shares_out":"shares_outstanding_income","payload_json":"income_payload"})
    cf = cf.rename(columns={"period":"cf_period","capital_expenditure":"capex","payload_json":"cash_flow_payload"})

    out = dates[["report_date","period_end"]].merge(inc.drop(columns=["ticker"],errors="ignore"), on="report_date", how="left")
    out = out.merge(cf.drop(columns=["ticker"],errors="ignore"), on="report_date", how="left")

    if not bs.empty:
        bs = bs.rename(columns={"report_date":"balance_sheet_date"})
        bs["balance_sheet_date"] = pd.to_datetime(bs["balance_sheet_date"], errors="coerce")
        bs = bs.dropna(subset=["balance_sheet_date"]).sort_values("balance_sheet_date")
        out = pd.merge_asof(out.sort_values("period_end"), bs.drop(columns=["ticker"],errors="ignore"), left_on="period_end", right_on="balance_sheet_date", direction="backward", tolerance=pd.Timedelta(days=140))
    else:
        out["balance_sheet_date"] = pd.NaT

    if not bs_rep.empty:
        bs_rep = bs_rep.rename(columns={"report_date":"bs_rep_date"})
        bs_rep["bs_rep_date"] = pd.to_datetime(bs_rep["bs_rep_date"], errors="coerce")
        bs_rep = bs_rep.dropna(subset=["bs_rep_date"]).sort_values("bs_rep_date")
        out = pd.merge_asof(out.sort_values("period_end"), bs_rep.drop(columns=["ticker"],errors="ignore"), left_on="period_end", right_on="bs_rep_date", direction="backward", tolerance=pd.Timedelta(days=140))
    else:
        out["balance_sheet_payload"] = None

    out["ticker"] = sym
    out["fiscal_period"] = out["fiscal_period"].fillna(out.get("cf_period"))

    # fill from reported BS payload
    for field, keys in [
        ("cash_and_equivalents", ("cashAndCashEquivalentsAtCarryingValue","cashAndCashEquivalents")),
        ("total_assets", ("assets","totalAssets")),
        ("total_liabilities", ("liabilities","totalLiabilities")),
        ("total_debt", ("totalDebt","debt")),
        ("outstanding_shares", ("commonStockSharesOutstanding","commonStockSharesIssued")),
    ]:
        if field in out.columns:
            out[field] = out[field].fillna(out["balance_sheet_payload"].map(lambda p, ks=keys: _reported_float(p, *ks)))

    out["rd_expense"] = out["income_payload"].map(lambda p: _payload_float(p, "researchAndDevelopmentExpenses"))
    out["sga_expense"] = out["income_payload"].map(lambda p: _payload_float(p, "sellingGeneralAndAdministrativeExpenses","generalAndAdministrativeExpenses"))
    out["pretax_income"] = out["income_payload"].map(lambda p: _payload_float(p, "incomeBeforeTax"))
    out["tax_provision"] = out["income_payload"].map(lambda p: _payload_float(p, "incomeTaxExpense"))
    out["stock_based_comp"] = out["cash_flow_payload"].map(lambda p: _payload_float(p, "stockBasedCompensation"))

    numeric = ["total_revenue","gross_profit","operating_income","ebitda","net_income","basic_eps","diluted_eps","shares_outstanding_income","operating_cash_flow","capex","free_cash_flow","investing_cash_flow","financing_cash_flow","net_change_in_cash","cash_and_equivalents","total_assets","total_liabilities","outstanding_shares","total_debt","rd_expense","sga_expense","pretax_income","tax_provision","stock_based_comp"]
    for col in numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["total_equity"] = (out["total_assets"] - out["total_liabilities"])
    out["net_debt"] = out["total_debt"] - out["cash_and_equivalents"]
    out["shares_outstanding"] = out["outstanding_shares"].fillna(out.get("shares_outstanding_income"))

    with np.errstate(divide="ignore", invalid="ignore"):
        out["gross_margin_pct"] = out["gross_profit"] / out["total_revenue"] * 100
        out["operating_margin_pct"] = out["operating_income"] / out["total_revenue"] * 100
        out["net_margin_pct"] = out["net_income"] / out["total_revenue"] * 100
        out["fcf_margin_pct"] = out["free_cash_flow"] / out["total_revenue"] * 100
        out["capex_pct_revenue"] = out["capex"].abs() / out["total_revenue"] * 100
        out["rd_pct_revenue"] = out["rd_expense"] / out["total_revenue"] * 100
        out["debt_to_assets_pct"] = out["total_debt"] / out["total_assets"] * 100

    out = out.replace([np.inf, -np.inf], np.nan).sort_values("period_end").reset_index(drop=True)
    out["revenue_qoq_pct"] = out["total_revenue"].pct_change() * 100
    out["revenue_yoy_pct"] = out["total_revenue"].pct_change(4) * 100
    out["eps_yoy_pct"] = out["diluted_eps"].pct_change(4) * 100
    out["fcf_yoy_pct"] = out["free_cash_flow"].pct_change(4) * 100

    if limit > 0:
        out = out.tail(limit).reset_index(drop=True)
    return out


def _fmt_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "period_end" in out.columns:
        out["period_end"] = pd.to_datetime(out["period_end"]).dt.strftime("%Y-%m-%d")
    money_cols = {"total_revenue","gross_profit","operating_income","net_income","ebitda","rd_expense","sga_expense","operating_cash_flow","capex","free_cash_flow","investing_cash_flow","financing_cash_flow","net_change_in_cash","cash_and_equivalents","total_assets","total_liabilities","total_debt","total_equity","net_debt","pretax_income","stock_based_comp"}
    pct_cols = {"gross_margin_pct","operating_margin_pct","net_margin_pct","fcf_margin_pct","capex_pct_revenue","rd_pct_revenue","revenue_qoq_pct","revenue_yoy_pct","eps_yoy_pct","fcf_yoy_pct","debt_to_assets_pct"}
    for col in out.columns:
        if col in money_cols:
            out[col] = out[col].map(_fmt_money)
        elif col in pct_cols:
            out[col] = out[col].map(_fmt_pct)
        elif col in {"diluted_eps","basic_eps"}:
            out[col] = out[col].map(lambda x: _fmt_num(x, 2))
        elif col == "shares_outstanding":
            out[col] = out[col].map(_fmt_money)
    return out


def _plot_layout(height: int = 300) -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=11), margin=dict(t=60, b=0, l=0, r=0), height=height,
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        legend=dict(orientation="h", y=1.14, x=1, xanchor="right", yanchor="bottom", title_text="", itemwidth=30),
        hovermode="x unified",
    )


def _pc(fig: go.Figure) -> None:
    if _PLOTLY_CHART_SUPPORTS_WIDTH:
        st.plotly_chart(fig, width="stretch", config={})
    else:
        st.plotly_chart(fig, use_container_width=True, config={})


def render_financials_tab(ticker: str, source: str, limit: int) -> None:
    tickers = available_db_tickers()
    df = pd.DataFrame()
    source_used = source
    err = None

    if source in {"Local database", "Auto: DB then Yahoo"}:
        with st.spinner(f"Loading {ticker} from local DB…"):
            try:
                df = fetch_db_quarterly_metrics(ticker, limit=limit)
                source_used = "Local database"
            except Exception as exc:
                err = str(exc)

    if df.empty and source in {"Yahoo live", "Auto: DB then Yahoo"}:
        with st.spinner(f"Downloading {ticker} from Yahoo Finance…"):
            try:
                df = fetch_quarterly_metrics(ticker, num_quarters=limit)
                source_used = "Yahoo live"
            except Exception as exc:
                err = str(exc)

    if df.empty:
        if err:
            st.error(err)
        st.info("No quarterly data found. Try another ticker or switch source.")
        return

    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df = df.dropna(subset=["period_end"]).sort_values("period_end").reset_index(drop=True)
    st.caption(f"Source: {source_used}  ·  {len(df)} quarters  ·  DB tickers: {len(tickers):,}")

    # ── KPI row ──
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else pd.Series(dtype=object)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", _fmt_money(latest.get("total_revenue")), _metric_delta(latest.get("total_revenue"), prev.get("total_revenue")))
    c2.metric("Gross Margin", _fmt_pct(latest.get("gross_margin_pct")), _metric_delta(latest.get("gross_margin_pct"), prev.get("gross_margin_pct"), pct=True))
    c3.metric("Diluted EPS", _fmt_num(latest.get("diluted_eps")), _metric_delta(latest.get("diluted_eps"), prev.get("diluted_eps")))
    c4.metric("Free Cash Flow", _fmt_money(latest.get("free_cash_flow")), _metric_delta(latest.get("free_cash_flow"), prev.get("free_cash_flow")))

    def _ttm(col):
        if col not in df.columns:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.tail(4).sum()) if not s.empty else None

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("TTM Revenue", _fmt_money(_ttm("total_revenue")))
    t2.metric("TTM Net Income", _fmt_money(_ttm("net_income")))
    t3.metric("TTM FCF", _fmt_money(_ttm("free_cash_flow")))
    t4.metric("Net Debt", _fmt_money(latest.get("net_debt")))

    st.divider()

    # ── charts ──
    ch = df.sort_values("period_end").copy()
    x = ch["period_end"].dt.strftime("%Y-%m-%d")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig = go.Figure()
        for col, label, color in [("total_revenue","Revenue","#3498db"),("gross_profit","Gross Profit","#2ecc71"),("operating_income","Op. Income","#f39c12"),("net_income","Net Income","#e74c3c")]:
            if col in ch.columns:
                fig.add_trace(go.Bar(x=x, y=ch[col], name=label, marker_color=color))
        fig.update_layout(**_plot_layout(320), title=dict(text="Income Statement", font=dict(size=13), x=0), barmode="group", yaxis_tickformat=".2s")
        _pc(fig)
    with r1c2:
        fig_m = go.Figure()
        for col, label, color in [("gross_margin_pct","Gross","#2ecc71"),("operating_margin_pct","Operating","#3498db"),("net_margin_pct","Net","#e74c3c"),("fcf_margin_pct","FCF","#9b59b6")]:
            if col in ch.columns:
                fig_m.add_trace(go.Scatter(x=x, y=ch[col], name=label, mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=5)))
        fig_m.update_layout(**_plot_layout(320), title=dict(text="Margins %", font=dict(size=13), x=0))
        fig_m.update_yaxes(ticksuffix="%")
        _pc(fig_m)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig_cf = go.Figure()
        if "operating_cash_flow" in ch.columns:
            fig_cf.add_trace(go.Bar(x=x, y=ch["operating_cash_flow"], name="Operating CF", marker_color="#16a085"))
        if "free_cash_flow" in ch.columns:
            fig_cf.add_trace(go.Bar(x=x, y=ch["free_cash_flow"], name="FCF", marker_color=_bar_colors(ch["free_cash_flow"])))
        if "capex" in ch.columns:
            fig_cf.add_trace(go.Bar(x=x, y=ch["capex"], name="Capex", marker_color="#95a5a6"))
        fig_cf.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
        fig_cf.update_layout(**_plot_layout(300), title=dict(text="Cash Flow", font=dict(size=13), x=0), barmode="group", yaxis_tickformat=".2s")
        _pc(fig_cf)
    with r2c2:
        fig_g = go.Figure()
        for col, label, color in [("revenue_yoy_pct","Revenue YoY","#3498db"),("eps_yoy_pct","EPS YoY","#f39c12"),("fcf_yoy_pct","FCF YoY","#2ecc71")]:
            if col in ch.columns:
                fig_g.add_trace(go.Scatter(x=x, y=ch[col], name=label, mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=5)))
        fig_g.update_layout(**_plot_layout(300), title=dict(text="YoY Growth Rates", font=dict(size=13), x=0))
        fig_g.update_yaxes(ticksuffix="%")
        _pc(fig_g)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig_bs = go.Figure()
        for col, label, color in [("cash_and_equivalents","Cash","#2ecc71"),("total_debt","Debt","#e74c3c"),("net_debt","Net Debt","#f39c12")]:
            if col in ch.columns:
                fig_bs.add_trace(go.Scatter(x=x, y=ch[col], name=label, mode="lines+markers", line=dict(color=color, width=2)))
        fig_bs.update_layout(**_plot_layout(280), title=dict(text="Balance Sheet Liquidity", font=dict(size=13), x=0), yaxis_tickformat=".2s")
        _pc(fig_bs)
    with r3c2:
        fig_eff = go.Figure()
        for col, label, color in [("rd_pct_revenue","R&D % Rev","#1abc9c"),("capex_pct_revenue","Capex % Rev","#95a5a6")]:
            if col in ch.columns:
                fig_eff.add_trace(go.Scatter(x=x, y=ch[col], name=label, mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=5)))
        fig_eff.update_layout(**_plot_layout(280), title=dict(text="Spend Ratios", font=dict(size=13), x=0))
        fig_eff.update_yaxes(ticksuffix="%")
        _pc(fig_eff)

    st.divider()

    # ── tables ──
    st.subheader("Detailed Data")
    newest = df.sort_values("period_end", ascending=False).copy()
    disp = _fmt_display(newest)
    present = [col for col, _ in DISPLAY_COLS if col in disp.columns]
    st.dataframe(disp[present].rename(columns={col: lbl for col, lbl in DISPLAY_COLS}), use_container_width=True, hide_index=True)

    tabs = st.tabs(["Income Statement", "Cash Flow", "Balance Sheet"])
    with tabs[0]:
        cols = ["period_end","fiscal_period","total_revenue","gross_profit","operating_income","ebitda","net_income","basic_eps","diluted_eps","rd_expense","sga_expense"]
        st.dataframe(_fmt_display(newest[[c for c in cols if c in newest.columns]]), use_container_width=True, hide_index=True)
    with tabs[1]:
        cols = ["period_end","operating_cash_flow","capex","free_cash_flow","investing_cash_flow","financing_cash_flow","net_change_in_cash","stock_based_comp"]
        st.dataframe(_fmt_display(newest[[c for c in cols if c in newest.columns]]), use_container_width=True, hide_index=True)
    with tabs[2]:
        cols = ["period_end","cash_and_equivalents","total_assets","total_liabilities","total_equity","total_debt","net_debt","shares_outstanding","debt_to_assets_pct"]
        st.dataframe(_fmt_display(newest[[c for c in cols if c in newest.columns]]), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _latest_report_date(ticker: str) -> str | None:
    try:
        with connect_readonly(_db_path()) as conn:
            row = conn.execute(
                "SELECT MAX(report_date) FROM income_statement WHERE ticker=?",
                (ticker.upper(),),
            ).fetchone()
            return row[0] if row and row[0] else None
    except Exception:
        return None


def main() -> None:
    if "sa_ticker" not in st.session_state:
        st.session_state["sa_ticker"] = "NVDA"

    # ── settings in sidebar (no ticker there — ticker lives on the page) ──
    with st.sidebar:
        st.header("Stock Analysis")
        with st.expander("DCF Settings", expanded=False):
            g5     = st.slider("5-year FCF growth (%)", 0, 100, 20, format="%d%%") / 100.0
            g_term = st.slider("Terminal growth (%)", 1, 5, 3, format="%d%%") / 100.0
            wacc   = st.slider("WACC (%)", 5, 20, 10, format="%d%%") / 100.0
        with st.expander("Financials Settings", expanded=False):
            source = st.radio("Data source", ["Auto: DB then Yahoo", "Local database", "Yahoo live"])
            limit  = st.slider("Quarters to show", 4, 20, 12, step=1)
            if st.button("Clear cache", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

    # ── inline ticker input ───────────────────────────────────────────────
    col_title, col_input = st.columns([3, 1])
    with col_input:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        entered = st.text_input(
            "Ticker",
            value=st.session_state["sa_ticker"],
            key="sa_ticker_input",
            label_visibility="collapsed",
            placeholder="Enter ticker…",
        ).strip().upper()
        if entered and entered != st.session_state["sa_ticker"]:
            st.session_state["sa_ticker"] = entered
            st.rerun()

    ticker = st.session_state["sa_ticker"]
    with col_title:
        st.title(f"Stock Analysis — {ticker}")
        last_report = _latest_report_date(ticker)
        if last_report:
            st.caption(f"Last financial report: **{last_report}**")

    tab_val, tab_fin = st.tabs(["📊 Valuation & DCF", "📋 Quarterly Financials"])

    with tab_val:
        render_valuation_tab(ticker, g5, g_term, wacc)

    with tab_fin:
        render_financials_tab(ticker, source, limit)


main()
