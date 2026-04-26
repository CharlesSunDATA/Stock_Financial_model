"""
Earnings Calendar — 財報行事曆

Sections:
  - Today: Pre-Market vs After-Hours reporters
  - Upcoming week: one tab per trading day

Data: Nasdaq public earnings calendar API (no API key required).
Cache TTL: 60 min (earnings schedule rarely changes intraday).
"""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st

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


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3_600, show_spinner=False)
def fetch_earnings(date_str: str) -> list[dict[str, Any]]:
    """
    Pull Nasdaq earnings calendar for one date (YYYY-MM-DD).
    Returns a list of row dicts; empty list on any failure.
    """
    try:
        r = requests.get(
            "https://api.nasdaq.com/api/calendar/earnings",
            headers=_HEADERS,
            params={"date": date_str},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        rows = (data.get("data") or {}).get("rows") or []
        return rows
    except Exception:
        return []


def _trading_days(start: datetime.date, n: int) -> list[datetime.date]:
    """Return n trading weekdays (Mon–Fri) starting from start (inclusive)."""
    days: list[datetime.date] = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


# ── Display helpers ───────────────────────────────────────────────────────────

def _when_badge(when: str) -> str:
    if "Before" in when:
        return "🌅 Pre-Mkt"
    if "After" in when:
        return "🌙 After Hrs"
    return "❓ TBD"


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append({
            "Symbol":      (row.get("symbol") or "").strip(),
            "Company":     (row.get("name")   or "").strip(),
            "When":        _when_badge(row.get("marketTime") or ""),
            "EPS Est.":    row.get("epsForecast")  or "—",
            "EPS Last Yr": row.get("lastYearEPS")  or "—",
            "Quarter":     row.get("fiscalQuarterEnding") or "—",
        })
    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["Symbol", "Company", "When", "EPS Est.", "EPS Last Yr", "Quarter"]
    )


def _split(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    pre, post, tbd = [], [], []
    for r in rows:
        t = r.get("marketTime") or ""
        if "Before" in t:
            pre.append(r)
        elif "After" in t:
            post.append(r)
        else:
            tbd.append(r)
    return pre, post, tbd


def _summary_badges(n_pre: int, n_post: int, n_tbd: int) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("🌅 Pre-Market",  n_pre)
    c2.metric("🌙 After Hours", n_post)
    c3.metric("❓ TBD",         n_tbd)


def _show_table(rows: list[dict], height: int = 400) -> None:
    if not rows:
        st.info("No earnings scheduled.")
        return
    df = _rows_to_df(rows)
    st.dataframe(df, hide_index=True, width="stretch", height=height)


def _day_tab(day: datetime.date) -> None:
    """Render content for one day inside a tab."""
    with st.spinner(f"Loading {day.strftime('%b %d')}…"):
        rows = fetch_earnings(day.isoformat())

    if not rows:
        st.info(
            f"No earnings data found for {day.strftime('%A, %B %d')}. "
            "This may be a market holiday or data not yet available."
        )
        return

    pre, post, tbd = _split(rows)
    _summary_badges(len(pre), len(post), len(tbd))
    st.markdown("")

    has_timed = pre or post

    if has_timed:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**🌅 Pre-Market** ({len(pre)})")
            _show_table(pre, height=min(60 + len(pre) * 35, 450))
        with col_r:
            st.markdown(f"**🌙 After Hours** ({len(post)})")
            _show_table(post, height=min(60 + len(post) * 35, 450))
        if tbd:
            with st.expander(f"❓ Time Not Confirmed ({len(tbd)})", expanded=False):
                _show_table(tbd)
        else:
            # All TBD — make it clear these ARE real earnings reports
            st.warning(
                f"**{len(tbd)} companies scheduled to report** — "
                "Nasdaq has not confirmed Pre-Market or After-Hours timing yet."
            )
            st.markdown(f"**📋 All Reports ({len(tbd)}) — Timing Unconfirmed**")
            _show_table(tbd, height=min(60 + len(tbd) * 35, 500))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    now_et = datetime.datetime.now(tz=_TZ_ET)
    today  = now_et.date()

    # ── Header ────────────────────────────────────────────────────────────────
    hcol, bcol = st.columns([5, 1])
    with hcol:
        st.title("📅 Earnings Calendar")
        st.caption(
            f"🕐 **{now_et.strftime('%Y-%m-%d  %H:%M  %Z')}**  ·  "
            "Source: Nasdaq earnings calendar (public)  ·  "
            "Cache refreshes every 60 min"
        )
    with bcol:
        st.markdown("<div style='margin-top:1.6rem'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", width="stretch", type="primary"):
            st.cache_data.clear()
            st.rerun()

    st.divider()

    # ── Today ──────────────────────────────────────────────────────────────────
    st.subheader(f"📌 Today — {today.strftime('%A, %B %d, %Y')}")

    with st.spinner("Loading today's earnings…"):
        today_rows = fetch_earnings(today.isoformat())

    if today_rows:
        pre, post, tbd = _split(today_rows)
        _summary_badges(len(pre), len(post), len(tbd))
        st.markdown("")

        has_timed = pre or post

        if has_timed:
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f"#### 🌅 Pre-Market  ({len(pre)})")
                _show_table(pre, height=min(60 + len(pre) * 35, 500))
            with col_r:
                st.markdown(f"#### 🌙 After Hours  ({len(post)})")
                _show_table(post, height=min(60 + len(post) * 35, 500))
            if tbd:
                with st.expander(f"❓ Time Not Confirmed ({len(tbd)})", expanded=False):
                    _show_table(tbd)
        else:
            # All companies are TBD — make clear these ARE real earnings
            st.warning(
                f"**{len(tbd)} companies are scheduled to report today** — "
                "Nasdaq has not yet tagged Pre-Market / After-Hours timing. "
                "The full list is shown below and will update automatically once confirmed."
            )
            st.markdown(f"#### 📋 Today's Reports ({len(tbd)}) — Timing Unconfirmed")
            _show_table(tbd, height=min(60 + len(tbd) * 35, 600))
    else:
        st.warning(
            "No earnings data for today. "
            "Possible reasons: weekend / market holiday / API temporarily unavailable."
        )

    st.divider()

    # ── Upcoming week ──────────────────────────────────────────────────────────
    st.subheader("📆 Upcoming 5 Trading Days")

    trading_days = _trading_days(today, 5)
    tab_labels   = [
        f"{d.strftime('%a')}  {d.strftime('%m/%d')}"
        + ("  ← Today" if d == today else "")
        for d in trading_days
    ]
    tabs = st.tabs(tab_labels)

    for tab, day in zip(tabs, trading_days):
        with tab:
            _day_tab(day)


main()
