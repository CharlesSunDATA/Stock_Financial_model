"""
Earnings Calendar — reporting schedule and estimate monitor.

Data sources:
- Nasdaq public earnings calendar API (no API key required)
- Local SQLite FMP calendar table, when populated
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

from utils.local_data import connect_readonly

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

_TZ_ET = ZoneInfo("America/New_York")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/market-activity/earnings",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _db_path() -> Path:
    return _project_root() / "data" / "quant_data.db"


def _trading_days(start: datetime.date, n: int) -> list[datetime.date]:
    """Return n weekdays starting from start, inclusive when start is a weekday."""
    days: list[datetime.date] = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


def _safe_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text in {"—", "-", "N/A", "n/a"}:
        return None
    text = text.replace(",", "").replace("$", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _surprise(actual: Any, estimate: Any) -> float | None:
    a = _safe_number(actual)
    e = _safe_number(estimate)
    if a is None or e is None or e == 0:
        return None
    return (a - e) / abs(e) * 100.0


def _money_short(value: Any) -> str:
    x = _safe_number(value)
    if x is None:
        return "—"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:,.2f}B"
    if ax >= 1e6:
        return f"{x / 1e6:,.1f}M"
    return f"{x:,.0f}"


def _normalize_when(value: Any) -> str:
    text = str(value or "").strip().lower()
    if any(token in text for token in ("before", "pre", "bmo", "morning")):
        return "Pre-Market"
    if any(token in text for token in ("after", "amc", "close", "evening")):
        return "After Hours"
    return "TBD"


def _when_icon(when: str) -> str:
    if when == "Pre-Market":
        return "🌅 Pre-Market"
    if when == "After Hours":
        return "🌙 After Hours"
    return "❓ TBD"


def _parse_payload_symbol(payload_json: Any) -> str:
    if not payload_json:
        return ""
    try:
        payload = json.loads(str(payload_json))
    except (TypeError, json.JSONDecodeError):
        return ""
    return str(payload.get("symbol") or "").strip().upper()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_nasdaq_earnings(date_str: str) -> list[dict[str, Any]]:
    """Pull Nasdaq earnings calendar for one date (YYYY-MM-DD)."""
    try:
        r = requests.get(
            "https://api.nasdaq.com/api/calendar/earnings",
            headers=_HEADERS,
            params={"date": date_str},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        return (data.get("data") or {}).get("rows") or []
    except Exception:
        return []


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_company_profiles() -> pd.DataFrame:
    db = _db_path()
    if not db.exists():
        return pd.DataFrame(columns=["Symbol", "Company", "Sector", "Industry"])
    with connect_readonly(db) as conn:
        try:
            return pd.read_sql_query(
                """
                SELECT
                  ticker AS Symbol,
                  company_name AS Company,
                  sector AS Sector,
                  industry AS Industry
                FROM company_profile
                """,
                conn,
            )
        except Exception:
            return pd.DataFrame(columns=["Symbol", "Company", "Sector", "Industry"])


@st.cache_data(ttl=60 * 10, show_spinner=False)
def load_watchlists() -> dict[str, set[str]]:
    db = _db_path()
    if not db.exists():
        return {}
    with connect_readonly(db) as conn:
        try:
            df = pd.read_sql_query("SELECT watchlist_name, ticker FROM fmp_watchlist", conn)
        except Exception:
            return {}
    if df.empty:
        return {}
    return {
        str(name): set(group["ticker"].dropna().astype(str).str.upper())
        for name, group in df.groupby("watchlist_name")
    }


@st.cache_data(ttl=60 * 10, show_spinner=False)
def load_local_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    db = _db_path()
    if not db.exists():
        return pd.DataFrame()

    with connect_readonly(db) as conn:
        try:
            raw = pd.read_sql_query(
                """
                SELECT
                  ticker,
                  event_date,
                  eps,
                  eps_estimated,
                  time,
                  revenue,
                  revenue_estimated,
                  fiscal_date_ending,
                  payload_json,
                  updated_at
                FROM earnings_calendar
                WHERE event_date BETWEEN ? AND ?
                ORDER BY event_date ASC, ticker ASC
                """,
                conn,
                params=(start_date, end_date),
            )
        except Exception:
            return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    payload_symbols = raw["payload_json"].map(_parse_payload_symbol)
    raw["Symbol"] = payload_symbols.where(payload_symbols.astype(bool), raw["ticker"].astype(str).str.upper())
    raw = raw.drop_duplicates(["Symbol", "event_date"], keep="last")

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw["event_date"], errors="coerce").dt.date,
            "Symbol": raw["Symbol"].astype(str).str.upper(),
            "When": raw["time"].map(_normalize_when),
            "EPS Est.": pd.to_numeric(raw["eps_estimated"], errors="coerce"),
            "EPS Actual": pd.to_numeric(raw["eps"], errors="coerce"),
            "Revenue Est.": pd.to_numeric(raw["revenue_estimated"], errors="coerce"),
            "Revenue Actual": pd.to_numeric(raw["revenue"], errors="coerce"),
            "Quarter": raw["fiscal_date_ending"].fillna(""),
            "Source": "Local FMP DB",
            "Updated": raw["updated_at"].fillna(""),
        }
    )
    profiles = load_company_profiles()
    if not profiles.empty:
        out = out.merge(profiles, on="Symbol", how="left")
    else:
        out["Company"] = ""
        out["Sector"] = ""
        out["Industry"] = ""
    out["EPS Surprise %"] = [_surprise(a, e) for a, e in zip(out["EPS Actual"], out["EPS Est."])]
    out["Revenue Surprise %"] = [_surprise(a, e) for a, e in zip(out["Revenue Actual"], out["Revenue Est."])]
    return out


def load_nasdaq_calendar(days: list[datetime.date]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    profiles = load_company_profiles()
    profile_map = profiles.set_index("Symbol").to_dict("index") if not profiles.empty else {}

    for day in days:
        for row in fetch_nasdaq_earnings(day.isoformat()):
            symbol = str(row.get("symbol") or "").strip().upper()
            profile = profile_map.get(symbol, {})
            records.append(
                {
                    "Date": day,
                    "Symbol": symbol,
                    "Company": str(row.get("name") or profile.get("Company") or "").strip(),
                    "Sector": profile.get("Sector", ""),
                    "Industry": profile.get("Industry", ""),
                    "When": _normalize_when(row.get("marketTime")),
                    "EPS Est.": _safe_number(row.get("epsForecast")),
                    "EPS Actual": None,
                    "EPS Last Yr": _safe_number(row.get("lastYearEPS")),
                    "Revenue Est.": None,
                    "Revenue Actual": None,
                    "Quarter": row.get("fiscalQuarterEnding") or "",
                    "Source": "Nasdaq Live",
                    "Updated": "",
                }
            )

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).drop_duplicates(["Date", "Symbol"], keep="last")
    df["EPS Surprise %"] = None
    df["Revenue Surprise %"] = None
    return df


def _apply_filters(
    df: pd.DataFrame,
    *,
    query: str,
    watchlist: set[str],
    only_watchlist: bool,
    sectors: list[str],
    show_missing_estimates: bool,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["In Watchlist"] = out["Symbol"].isin(watchlist) if watchlist else False

    q = query.strip().lower()
    if q:
        haystack = (
            out["Symbol"].fillna("").astype(str)
            + " "
            + out["Company"].fillna("").astype(str)
            + " "
            + out["Sector"].fillna("").astype(str)
        ).str.lower()
        out = out[haystack.str.contains(re.escape(q), na=False)]

    if only_watchlist and watchlist:
        out = out[out["In Watchlist"]]
    if sectors:
        out = out[out["Sector"].fillna("").isin(sectors)]
    if not show_missing_estimates and "EPS Est." in out.columns:
        out = out[out["EPS Est."].notna()]

    when_order = {"Pre-Market": 0, "After Hours": 1, "TBD": 2}
    out["_when_order"] = out["When"].map(when_order).fillna(9)
    out["_watch_order"] = (~out["In Watchlist"]).astype(int)
    out = out.sort_values(["Date", "_watch_order", "_when_order", "Symbol"])
    return out.drop(columns=["_when_order", "_watch_order"])


def _display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    show = df.copy()
    show["Date"] = pd.to_datetime(show["Date"]).dt.strftime("%Y-%m-%d")
    show["When"] = show["When"].map(_when_icon)
    show["Watch"] = show["In Watchlist"].map(lambda x: "Yes" if bool(x) else "")
    show["Revenue Est."] = show["Revenue Est."].map(_money_short)
    show["Revenue Actual"] = show["Revenue Actual"].map(_money_short)
    cols = [
        "Date",
        "Watch",
        "Symbol",
        "Company",
        "Sector",
        "When",
        "EPS Est.",
        "EPS Actual",
        "EPS Surprise %",
        "EPS Last Yr",
        "Revenue Est.",
        "Revenue Actual",
        "Revenue Surprise %",
        "Quarter",
        "Source",
    ]
    present = [c for c in cols if c in show.columns]
    return show[present]


def _height(n_rows: int, max_height: int = 520) -> int:
    return min(max(120, 38 + n_rows * 35), max_height)


def _summary(df: pd.DataFrame) -> None:
    total = len(df)
    pre = int((df["When"] == "Pre-Market").sum()) if total else 0
    after = int((df["When"] == "After Hours").sum()) if total else 0
    watch = int(df.get("In Watchlist", pd.Series(dtype=bool)).sum()) if total else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reports", f"{total:,}")
    c2.metric("Pre-Market", f"{pre:,}")
    c3.metric("After Hours", f"{after:,}")
    c4.metric("Watchlist", f"{watch:,}")


def _show_table(df: pd.DataFrame, *, max_height: int = 520) -> None:
    if df.empty:
        st.info("No earnings match the current filters.")
        return
    st.dataframe(
        _display_df(df),
        hide_index=True,
        width="stretch",
        height=_height(len(df), max_height=max_height),
        column_config={
            "EPS Est.": st.column_config.NumberColumn("EPS Est.", format="%.2f"),
            "EPS Actual": st.column_config.NumberColumn("EPS Actual", format="%.2f"),
            "EPS Surprise %": st.column_config.NumberColumn("EPS Surprise %", format="%.1f%%"),
            "EPS Last Yr": st.column_config.NumberColumn("EPS Last Yr", format="%.2f"),
            "Revenue Surprise %": st.column_config.NumberColumn("Revenue Surprise %", format="%.1f%%"),
        },
    )


def _show_day(day: datetime.date, df: pd.DataFrame) -> None:
    day_df = df[df["Date"] == day]
    if day_df.empty:
        st.info(f"No earnings found for {day.strftime('%A, %B %d')}.")
        return

    _summary(day_df)
    st.markdown("")

    pre = day_df[day_df["When"] == "Pre-Market"]
    after = day_df[day_df["When"] == "After Hours"]
    tbd = day_df[day_df["When"] == "TBD"]

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"#### 🌅 Pre-Market ({len(pre)})")
        _show_table(pre, max_height=430)
    with col_r:
        st.markdown(f"#### 🌙 After Hours ({len(after)})")
        _show_table(after, max_height=430)

    if not tbd.empty:
        with st.expander(f"❓ Time Not Confirmed ({len(tbd)})", expanded=False):
            _show_table(tbd)


def main() -> None:
    now_et = datetime.datetime.now(tz=_TZ_ET)
    today = now_et.date()

    st.title("📅 Earnings Calendar")
    st.caption(
        f"{now_et.strftime('%Y-%m-%d %H:%M %Z')} · Track report timing, estimates, and watchlist names."
    )

    watchlists = load_watchlists()
    watchlist_names = sorted(watchlists)

    with st.sidebar:
        st.header("Calendar Controls")
        source = st.radio(
            "Data source",
            ["Auto: local then Nasdaq", "Local FMP database", "Nasdaq live"],
            horizontal=False,
        )
        start_input = st.date_input("Start date", value=today)
        days_count = st.slider("Trading days", min_value=3, max_value=20, value=5, step=1)
        query = st.text_input("Search symbol / company / sector", "")

        selected_watchlist = ""
        watchlist: set[str] = set()
        if watchlist_names:
            selected_watchlist = st.selectbox(
                "Watchlist",
                watchlist_names,
                index=watchlist_names.index("default") if "default" in watchlist_names else 0,
            )
            watchlist = watchlists.get(selected_watchlist, set())
            only_watchlist = st.checkbox("Only show watchlist", value=False)
        else:
            only_watchlist = False
            st.caption("No local watchlist table found.")

        show_missing_estimates = st.checkbox("Show rows without EPS estimate", value=True)
        if st.button("Refresh data", type="primary", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    start_base = start_input if isinstance(start_input, datetime.date) else today
    trading_days = _trading_days(start_base, days_count)
    start_date = trading_days[0]
    end_date = trading_days[-1]

    with st.spinner("Loading earnings calendar..."):
        if source == "Local FMP database":
            calendar = load_local_calendar(start_date.isoformat(), end_date.isoformat())
        elif source == "Nasdaq live":
            calendar = load_nasdaq_calendar(trading_days)
        else:
            calendar = load_local_calendar(start_date.isoformat(), end_date.isoformat())
            if calendar.empty:
                calendar = load_nasdaq_calendar(trading_days)

    sector_options = sorted(s for s in calendar.get("Sector", pd.Series(dtype=str)).dropna().unique() if str(s))
    with st.sidebar:
        sectors = st.multiselect("Sector", sector_options, default=[])
        st.caption(f"Window: {start_date.isoformat()} to {end_date.isoformat()}")
        if selected_watchlist:
            st.caption(f"Watchlist symbols: {len(watchlist):,}")

    filtered = _apply_filters(
        calendar,
        query=query,
        watchlist=watchlist,
        only_watchlist=only_watchlist,
        sectors=sectors,
        show_missing_estimates=show_missing_estimates,
    )

    if calendar.empty:
        st.warning(
            "No earnings data loaded for this window. Try the other data source, refresh the cache, "
            "or update the local FMP calendar data."
        )
        return

    _summary(filtered)

    focus = filtered[filtered.get("In Watchlist", pd.Series(dtype=bool))]
    if not focus.empty:
        with st.expander(f"Watchlist focus ({len(focus)} reports)", expanded=True):
            _show_table(focus, max_height=360)

    st.divider()

    tab_labels = [
        f"{d.strftime('%a')} {d.strftime('%m/%d')}" + (" · Today" if d == today else "")
        for d in trading_days
    ]
    tabs = st.tabs(tab_labels)
    for tab, day in zip(tabs, trading_days):
        with tab:
            _show_day(day, filtered)

    st.divider()
    st.subheader("All Matching Reports")
    _show_table(filtered, max_height=650)


main()
