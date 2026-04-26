"""
Market Overview Dashboard — 全球市場即時總覽 (Homepage)

Sections:
  - US major indices (S&P 500, Dow Jones, Nasdaq, Russell 2000)
  - Futures & Volatility (Nasdaq 100 / S&P 500 futures, VIX, VIX futures)
  - Commodities (WTI crude, Brent crude, Gold, Silver)
  - Forex (AUD/USD, EUR/USD, DXY, USD/JPY)
  - US Treasury Yields (3M, 10Y, 30Y)
  - Global Markets (8 major world indices)

Data: Yahoo Finance via yfinance (~15 min delayed).
Cache TTL: 60 s. Auto-refresh via streamlit-autorefresh (optional).
"""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
import streamlit as st
import yfinance as yf

try:
    from zoneinfo import ZoneInfo          # Python 3.9+ built-in
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

# Optional dependency — graceful fallback if not installed
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

# ── Instrument registry ───────────────────────────────────────────────────────
# fmt: Python format spec for the price value
# suffix: appended after the formatted number (e.g. "%" for yields)

_IDX = {"fmt": ",.2f", "suffix": ""}    # indices / futures / commodities
_FX4 = {"fmt": ".4f",  "suffix": ""}    # forex 4 decimal places (AUD/USD, EUR/USD)
_FX2 = {"fmt": ",.2f", "suffix": ""}    # forex 2 dp (USD/JPY)
_DXY = {"fmt": ",.3f", "suffix": ""}    # US Dollar Index
_YLD = {"fmt": ".3f",  "suffix": "%"}   # Treasury yields (already in %)
_SIL = {"fmt": ",.3f", "suffix": ""}    # Silver (3 dp)

US_INDICES: list[dict[str, Any]] = [
    {**_IDX, "sym": "^GSPC",  "name": "S&P 500"},
    {**_IDX, "sym": "^DJI",   "name": "Dow Jones"},
    {**_IDX, "sym": "^IXIC",  "name": "Nasdaq Composite"},
    {**_IDX, "sym": "^RUT",   "name": "Russell 2000"},
]

FUTURES: list[dict[str, Any]] = [
    {**_IDX, "sym": "NQ=F",  "name": "Nasdaq 100 Futures"},
    {**_IDX, "sym": "ES=F",  "name": "S&P 500 Futures"},
    {**_IDX, "sym": "^VIX",  "name": "VIX Index"},
    {**_IDX, "sym": "VX=F",  "name": "VIX Futures"},
]

COMMODITIES: list[dict[str, Any]] = [
    {**_IDX, "sym": "CL=F",  "name": "WTI Crude Oil"},
    {**_IDX, "sym": "BZ=F",  "name": "Brent Crude"},
    {**_IDX, "sym": "GC=F",  "name": "Gold"},
    {**_SIL, "sym": "SI=F",  "name": "Silver"},
]

FOREX: list[dict[str, Any]] = [
    {**_FX4, "sym": "AUDUSD=X", "name": "AUD / USD"},
    {**_FX4, "sym": "EURUSD=X", "name": "EUR / USD"},
    {**_DXY, "sym": "DX-Y.NYB", "name": "USD Index (DXY)"},
    {**_FX2, "sym": "USDJPY=X", "name": "USD / JPY"},
]

BONDS: list[dict[str, Any]] = [
    {**_YLD, "sym": "^IRX", "name": "US 3M T-Bill"},
    {**_YLD, "sym": "^TNX", "name": "US 10Y T-Note"},
    {**_YLD, "sym": "^TYX", "name": "US 30Y T-Bond"},
]

GLOBAL_MKTS: list[dict[str, Any]] = [
    {**_IDX, "sym": "^N225",     "name": "Nikkei 225 🇯🇵"},
    {**_IDX, "sym": "^HSI",      "name": "Hang Seng 🇭🇰"},
    {**_IDX, "sym": "^TWII",     "name": "TAIEX 🇹🇼"},
    {**_IDX, "sym": "000001.SS", "name": "Shanghai 🇨🇳"},
    {**_IDX, "sym": "^FTSE",     "name": "FTSE 100 🇬🇧"},
    {**_IDX, "sym": "^GDAXI",    "name": "DAX 🇩🇪"},
    {**_IDX, "sym": "^KS11",     "name": "KOSPI 🇰🇷"},
    {**_IDX, "sym": "^AXJO",     "name": "ASX 200 🇦🇺"},
]

_ALL_INST = US_INDICES + FUTURES + COMMODITIES + FOREX + BONDS + GLOBAL_MKTS
ALL_SYMS: tuple[str, ...] = tuple(d["sym"] for d in _ALL_INST)


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def fetch_quotes(syms: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """
    Download the last 5 trading-day daily bars for all symbols in one batch call.

    Returns {sym: {"price": float | None, "change": float, "pct": float}}.
      - price  : latest close (today's partial session if market is open)
      - change : price - previous_close
      - pct    : percentage change vs previous_close
    Symbols that fail download will have price=None.
    """
    _blank: dict[str, Any] = {"price": None, "change": 0.0, "pct": 0.0}
    result: dict[str, dict[str, Any]] = {s: _blank.copy() for s in syms}

    try:
        raw = yf.download(
            list(syms),
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        return result

    if raw.empty:
        return result

    # yfinance returns MultiIndex columns when >1 ticker is requested
    try:
        close: pd.DataFrame = (
            raw["Close"]
            if isinstance(raw.columns, pd.MultiIndex)
            else raw[["Close"]].rename(columns={"Close": syms[0]})
        )
    except Exception:
        return result

    for sym in syms:
        try:
            if sym not in close.columns:
                continue
            s = close[sym].dropna()
            if s.empty:
                continue
            price = float(s.iloc[-1])
            prev  = float(s.iloc[-2]) if len(s) >= 2 else price
            chg   = price - prev
            pct   = chg / prev * 100.0 if prev else 0.0
            result[sym] = {"price": price, "change": chg, "pct": pct}
        except Exception:
            continue

    return result


# ── Market clock helpers ──────────────────────────────────────────────────────

_TZ_SA = ZoneInfo("Australia/Adelaide")   # South Australia (ACST/ACDT)
_TZ_ET = ZoneInfo("America/New_York")     # US Eastern (EST/EDT)

# NYSE regular session boundaries (ET)
_MARKET_PRE_OPEN  = datetime.time(4,  0)
_MARKET_OPEN      = datetime.time(9, 30)
_MARKET_CLOSE     = datetime.time(16, 0)
_MARKET_POST_CLOSE = datetime.time(20, 0)


def _market_status(now_et: datetime.datetime) -> tuple[str, str]:
    """Return (badge_label, session_note) for current ET time."""
    t  = now_et.time()
    wd = now_et.weekday()   # 0=Mon … 4=Fri, 5=Sat, 6=Sun
    if wd >= 5:
        return "🔴  Closed", "Weekend"
    if _MARKET_OPEN <= t < _MARKET_CLOSE:
        return "🟢  Open", "Regular  9:30 AM – 4:00 PM ET"
    if _MARKET_PRE_OPEN <= t < _MARKET_OPEN:
        return "🟡  Pre-Market", "4:00 – 9:30 AM ET"
    if _MARKET_CLOSE <= t < _MARKET_POST_CLOSE:
        return "🟡  After Hours", "4:00 – 8:00 PM ET"
    return "🔴  Closed", "After 8:00 PM ET"


def _next_open_et(now_et: datetime.datetime) -> datetime.datetime:
    """Next NYSE regular session open (9:30 AM ET, weekday only)."""
    candidate = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_et >= candidate:
        candidate += datetime.timedelta(days=1)
    while candidate.weekday() >= 5:          # skip Saturday & Sunday
        candidate += datetime.timedelta(days=1)
    return candidate


def _next_close_et(now_et: datetime.datetime) -> datetime.datetime:
    """Today's NYSE close (4:00 PM ET) — only valid when market is open."""
    return now_et.replace(hour=16, minute=0, second=0, microsecond=0)


def _fmt_countdown(delta: datetime.timedelta) -> str:
    total = max(0, int(delta.total_seconds()))
    h, rem = divmod(total, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d} : {m:02d} : {s:02d}"


def _clock_card(label: str, value: str, sub: str, value_color: str = "#e2e8f0") -> str:
    """Return an HTML string for one compact clock card."""
    return f"""
    <div style="background:rgba(255,255,255,0.05);border-radius:10px;
                padding:12px 16px;min-width:0;overflow:hidden;">
      <div style="color:#94a3b8;font-size:0.72rem;
                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                  margin-bottom:4px;">{label}</div>
      <div style="color:{value_color};font-size:1.25rem;font-weight:700;
                  font-family:monospace;white-space:nowrap;overflow:hidden;
                  text-overflow:ellipsis;">{value}</div>
      <div style="color:#64748b;font-size:0.68rem;margin-top:3px;
                  white-space:nowrap;">{sub}</div>
    </div>"""


def _render_market_clock() -> None:
    """Compact HTML banner: Adelaide / New York clocks + NYSE status + countdown."""
    now_et = datetime.datetime.now(tz=_TZ_ET)
    now_sa = datetime.datetime.now(tz=_TZ_SA)

    status_label, status_note = _market_status(now_et)
    is_open = status_label.startswith("🟢")

    if is_open:
        event_et  = _next_close_et(now_et)
        cd_title  = "⏳ Closes in"
        evt_title = "🔔 Close (Adelaide)"
    else:
        event_et  = _next_open_et(now_et)
        cd_title  = "⏳ Opens in"
        evt_title = "🔔 Next Open (Adelaide)"

    event_sa = event_et.astimezone(_TZ_SA)
    delta    = event_et - now_et

    # Status colour
    if is_open:
        s_color = "#22c55e"
    elif "Closed" in status_label:
        s_color = "#ef4444"
    else:
        s_color = "#eab308"

    cards = "".join([
        _clock_card("🇦🇺 Adelaide",   now_sa.strftime("%H:%M:%S"),       now_sa.strftime("%Z")),
        _clock_card("🇺🇸 New York",   now_et.strftime("%H:%M:%S"),       now_et.strftime("%Z")),
        _clock_card("NYSE Status",    status_label.replace("  ", " "),   status_note, value_color=s_color),
        _clock_card(evt_title,        event_sa.strftime("%a %d %b %H:%M"), event_sa.strftime("%Z")),
        _clock_card(cd_title,         _fmt_countdown(delta),             "hh : mm : ss"),
    ])

    st.subheader("🕐  NYSE Market Clock  ·  Adelaide 🇦🇺 vs New York 🇺🇸")
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(5,1fr);'
        f'gap:10px;margin-bottom:0.5rem;">{cards}</div>',
        unsafe_allow_html=True,
    )
    st.divider()


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _price_str(price: float | None, fmt: str, suffix: str) -> str:
    """Format a price value for display; returns '—' when unavailable."""
    return "—" if price is None else f"{price:{fmt}}{suffix}"


def _delta_label(price: float | None, chg: float, pct: float) -> str | None:
    """Return a delta string like '+5.23 (+0.12%)' or None when price is missing."""
    if price is None:
        return None
    sign = "+" if chg >= 0 else ""
    return f"{sign}{chg:.2f}  ({sign}{pct:.2f}%)"


def _render_section(
    title: str,
    instruments: list[dict[str, Any]],
    quotes: dict[str, dict[str, Any]],
    ncols: int = 4,
) -> None:
    """Render a titled group of metric cards in `ncols` columns."""
    st.subheader(title)
    cols = st.columns(ncols)
    for i, inst in enumerate(instruments):
        q     = quotes.get(inst["sym"], {})
        price = q.get("price")
        chg   = float(q.get("change", 0.0) or 0.0)
        pct   = float(q.get("pct",    0.0) or 0.0)
        cols[i % ncols].metric(
            label=inst["name"],
            value=_price_str(price, inst["fmt"], inst["suffix"]),
            delta=_delta_label(price, chg, pct),
        )
    st.markdown("")  # visual breathing room


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Auto-refresh every 60 s — always on, no user control needed.
    if _HAS_AUTOREFRESH:
        st_autorefresh(interval=60_000, key="mkt_autorefresh")

    # ── Header row ────────────────────────────────────────────────────────────
    st.title("📊 Market Overview")
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    st.caption(
        f"🕐 **{now}** (local)  ·  Quotes via Yahoo Finance — ~15 min delayed  "
        f"·  Cache refreshes every 60 s"
    )

    st.divider()

    # ── NYSE market clock & Adelaide countdown ────────────────────────────────
    _render_market_clock()

    # ── Fetch all quotes in one batch ─────────────────────────────────────────
    with st.spinner("Loading market data…"):
        quotes = fetch_quotes(ALL_SYMS)

    # ── US indices ────────────────────────────────────────────────────────────
    _render_section("🇺🇸  US Major Indices", US_INDICES, quotes, ncols=4)

    # ── Futures & volatility ──────────────────────────────────────────────────
    _render_section("📈  Futures & Volatility (VIX)", FUTURES, quotes, ncols=4)

    # ── Commodities | Forex + Yields — side by side ───────────────────────────
    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        _render_section("⚡  Commodities", COMMODITIES, quotes, ncols=2)
    with col_right:
        _render_section("💱  Forex", FOREX, quotes, ncols=2)
        _render_section("🏦  US Treasury Yields", BONDS, quotes, ncols=3)

    # ── Global markets ────────────────────────────────────────────────────────
    _render_section("🌏  Global Markets", GLOBAL_MKTS, quotes, ncols=4)


main()
