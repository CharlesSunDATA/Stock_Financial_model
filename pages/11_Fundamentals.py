"""
Fundamentals (Streamlit page)

Reads historical fundamentals from SQLite and visualizes:
- Revenue vs Free Cash Flow (grouped bars)
- EPS trend (line with markers)

Notes:
- UI text and comments are in English.
- Designed to fit a dark/professional theme.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scripts.update_fundamentals_fmp import bulk_download_tech_first
from utils.local_data import connect_readonly


def _project_root() -> Path:
    # pages/11_Fundamentals.py -> project root is parent of pages/
    return Path(__file__).resolve().parents[1]


def _db_path() -> Path:
    return _project_root() / "data" / "quant_data.db"


@st.cache_data(ttl=60 * 5, show_spinner=False)
def get_available_tickers() -> list[str]:
    """Return all tickers currently present in the DB."""
    db = _db_path()
    if not db.exists():
        return []

    with connect_readonly(db) as conn:
        df = pd.read_sql_query(
            "SELECT DISTINCT ticker FROM fundamental_data ORDER BY ticker ASC",
            conn,
        )
    if df.empty:
        return []
    return [str(x).upper() for x in df["ticker"].dropna().tolist() if str(x).strip()]


@st.cache_data(ttl=60 * 5, show_spinner=False)
def get_db_data(ticker: str) -> pd.DataFrame:
    """
    Fetch all historical records for a ticker, sorted by report_date asc.
    Uses pandas.read_sql_query as requested.
    """
    db = _db_path()
    if not db.exists():
        return pd.DataFrame()

    sym = (ticker or "").strip().upper()
    if not sym:
        return pd.DataFrame()

    with connect_readonly(db) as conn:
        df = pd.read_sql_query(
            """
            SELECT
              ticker,
              report_date,
              revenue,
              eps,
              free_cash_flow
            FROM fundamental_data
            WHERE ticker = ?
            ORDER BY report_date ASC
            """,
            conn,
            params=(sym,),
        )

    if df.empty:
        return df

    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["report_date"]).sort_values("report_date", ascending=True).reset_index(drop=True)
    for col in ("revenue", "eps", "free_cash_flow"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _abbr_money(v: float | None) -> str:
    """Format large numbers with M/B suffix and thousands separators."""
    if v is None or pd.isna(v):
        return "—"
    x = float(v)
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:,.2f}B"
    if ax >= 1e6:
        return f"{x/1e6:,.1f}M"
    if ax >= 1e3:
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def _abbr_eps(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{float(v):,.2f}"


def _plot_revenue_fcf(df: pd.DataFrame, ticker: str) -> go.Figure:
    x = df["report_date"].dt.strftime("%Y-%m-%d")
    rev = df["revenue"] if "revenue" in df.columns else None
    fcf = df["free_cash_flow"] if "free_cash_flow" in df.columns else None

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=rev,
            name="Revenue",
            marker_color="rgba(52, 152, 219, 0.85)",
            hovertemplate="<b>%{x}</b><br>Revenue: %{customdata}<extra></extra>",
            customdata=[_abbr_money(v) for v in (rev.tolist() if rev is not None else [])],
        )
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=fcf,
            name="Free Cash Flow",
            marker_color="rgba(46, 204, 113, 0.80)",
            hovertemplate="<b>%{x}</b><br>FCF: %{customdata}<extra></extra>",
            customdata=[_abbr_money(v) for v in (fcf.tolist() if fcf is not None else [])],
        )
    )

    fig.update_layout(
        barmode="group",
        height=420,
        title=dict(text=f"{ticker} — Revenue vs Free Cash Flow (Quarterly)", x=0),
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(t=60, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        ),
    )
    return fig


def _plot_eps(df: pd.DataFrame, ticker: str) -> go.Figure:
    x = df["report_date"].dt.strftime("%Y-%m-%d")
    eps = df["eps"] if "eps" in df.columns else None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=eps,
            mode="lines+markers",
            name="EPS",
            line=dict(color="rgba(241, 196, 15, 0.95)", width=2),
            marker=dict(size=7, color="rgba(241, 196, 15, 0.95)", line=dict(color="white", width=1)),
            hovertemplate="<b>%{x}</b><br>EPS: %{customdata}<extra></extra>",
            customdata=[_abbr_eps(v) for v in (eps.tolist() if eps is not None else [])],
        )
    )
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_width=1)
    fig.update_layout(
        height=320,
        title=dict(text=f"{ticker} — EPS trend (Quarterly)", x=0),
        margin=dict(t=60, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False),
    )
    return fig


def main() -> None:
    st.title("📊 Fundamentals")
    st.caption("SQLite → Plotly. All UI text is English by design.")

    with st.sidebar:
        st.header("Data")
        st.caption(f"DB: `{_db_path()}`")
        if st.button("Refresh (clear cache)", type="secondary", use_container_width=True):
            # Clear cached DB queries so newly inserted tickers show up immediately.
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()

        st.divider()
        st.header("Bulk download")
        st.caption("Tech-first batch download with daily quota guard (FMP).")
        daily_quota = st.number_input("Daily quota (calls/day)", min_value=10, max_value=2500, value=250, step=10)
        if st.button("Bulk download (Tech first)", type="primary", use_container_width=True):
            logs: list[str] = []
            with st.spinner("Downloading fundamentals from FMP (tech-first)…"):
                logs = bulk_download_tech_first(_db_path(), daily_quota=int(daily_quota), sleep_s=0.5)
            # Refresh UI immediately after download
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.success("Batch download finished.")
            with st.expander("Run log", expanded=True):
                st.code("\n".join(logs), language="text")
            st.rerun()

    tickers = get_available_tickers()
    if not tickers:
        st.warning("Database is empty. Please run `python3 scripts/update_fundamentals.py <TICKER>` first.")
        st.info(f"Expected DB path: `{_db_path()}`")
        return

    with st.sidebar:
        st.header("Ticker")
        ticker = st.selectbox("Select a ticker", tickers, index=0)

    df = get_db_data(ticker)
    if df.empty:
        st.warning(f"No rows found for `{ticker}`. Try updating fundamentals again.")
        return

    # Quick KPIs
    latest = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Latest revenue", _abbr_money(latest.get("revenue")))
    with c2:
        st.metric("Latest free cash flow", _abbr_money(latest.get("free_cash_flow")))
    with c3:
        st.metric("Latest EPS", _abbr_eps(latest.get("eps")))

    st.divider()

    st.subheader("1) Revenue vs Free Cash Flow")
    st.plotly_chart(_plot_revenue_fcf(df, ticker), use_container_width=True)

    st.subheader("2) EPS trend")
    st.plotly_chart(_plot_eps(df, ticker), use_container_width=True)

    with st.expander("Raw data (from SQLite)", expanded=False):
        show = df.copy()
        show["report_date"] = show["report_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(show, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
