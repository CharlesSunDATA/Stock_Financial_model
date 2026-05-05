"""
Momentum Report (Streamlit page)

Reads prices_eod + company_profile from SQLite and ranks watchlist stocks
by 1/3/6/12-month momentum. Displays top/bottom movers and allows refresh.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.local_data import connect_readonly

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "quant_data.db"
REPORT_SCRIPT = Path(__file__).resolve().parents[1] / "momentum_report.py"


def _connect() -> sqlite3.Connection:
    return connect_readonly(DB_PATH)


@st.cache_data(ttl=60 * 10, show_spinner=False)
def compute_momentum() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    with _connect() as conn:
        latest_date = conn.execute("SELECT MAX(price_date) FROM prices_eod").fetchone()[0]
        if not latest_date:
            return pd.DataFrame()

        latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
        periods = {
            "1M":  (latest_dt - timedelta(days=30)).strftime("%Y-%m-%d"),
            "3M":  (latest_dt - timedelta(days=91)).strftime("%Y-%m-%d"),
            "6M":  (latest_dt - timedelta(days=182)).strftime("%Y-%m-%d"),
            "12M": (latest_dt - timedelta(days=365)).strftime("%Y-%m-%d"),
        }

        try:
            tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM fmp_watchlist").fetchall()]
        except Exception:
            tickers = []
        if not tickers:
            return pd.DataFrame()

        placeholders = ",".join("?" * len(tickers))
        price_df = pd.read_sql(
            f"""
            SELECT ticker, price_date, COALESCE(adj_close, close) AS price
            FROM prices_eod
            WHERE ticker IN ({placeholders})
              AND price_date >= ?
            ORDER BY ticker, price_date
            """,
            conn,
            params=tickers + [periods["12M"]],
        )
        if price_df.empty:
            return pd.DataFrame()

        names = pd.read_sql(
            "SELECT ticker, company_name, sector FROM company_profile",
            conn,
        ).drop_duplicates("ticker")

        try:
            mcap = pd.read_sql(
                """SELECT ticker, market_cap FROM key_metrics_ttm
                   WHERE as_of_date = (SELECT MAX(as_of_date) FROM key_metrics_ttm)""",
                conn,
            ).drop_duplicates("ticker")
        except Exception:
            mcap = pd.DataFrame(columns=["ticker", "market_cap"])

    price_df["price"] = pd.to_numeric(price_df["price"], errors="coerce")

    results = []
    for ticker, grp in price_df.groupby("ticker"):
        grp = grp.sort_values("price_date").dropna(subset=["price"])
        if grp.empty:
            continue
        price_now = grp.iloc[-1]["price"]
        if not price_now or np.isnan(price_now):
            continue
        row: dict = {"ticker": ticker, "price": price_now}
        for period, date_str in periods.items():
            past = grp[grp["price_date"] <= date_str].dropna(subset=["price"])
            if not past.empty and past.iloc[-1]["price"]:
                p0 = float(past.iloc[-1]["price"])
                row[f"ret_{period}"] = (float(price_now) - p0) / p0 * 100
            else:
                row[f"ret_{period}"] = None
        results.append(row)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).dropna(subset=["ret_1M", "ret_3M"])
    df = df.merge(names, on="ticker", how="left")
    df = df.merge(mcap, on="ticker", how="left")

    for col in ["ret_1M", "ret_3M", "ret_6M", "ret_12M"]:
        df[f"rank_{col}"] = df[col].rank(pct=True, na_option="keep")
    rank_cols = [c for c in df.columns if c.startswith("rank_ret_")]
    df["momentum_score"] = df[rank_cols].mean(axis=1) * 100

    df["company_name"] = df["company_name"].apply(
        lambda v: str(v) if pd.notna(v) and v else ""
    )
    return df.sort_values("momentum_score", ascending=False).reset_index(drop=True)


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v):+.1f}%"


def _bar_chart(df: pd.DataFrame, col: str, title: str, top_n: int = 30) -> go.Figure:
    sub = df.dropna(subset=[col]).nlargest(top_n, col)
    sub = sub.iloc[::-1]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sub[col]]
    fig = go.Figure(go.Bar(
        x=sub[col],
        y=sub["ticker"],
        orientation="h",
        marker_color=colors,
        text=[_fmt_pct(v) for v in sub[col]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>" + title + ": %{x:+.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=13, color="white")),
        height=max(300, top_n * 22),
        margin=dict(t=40, b=10, l=70, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=True,
                   zerolinecolor="rgba(255,255,255,0.3)", ticksuffix="%"),
        yaxis=dict(showgrid=False),
    )
    return fig


def main() -> None:
    st.title("Momentum Rankings")
    st.caption("1/3/6/12-month price momentum from SQLite `prices_eod`. Composite score = equal-weight percentile rank.")

    if not DB_PATH.exists():
        st.error(f"Database not found: `{DB_PATH}`")
        return

    with st.sidebar:
        st.header("Controls")
        st.caption(f"DB: `{DB_PATH}`")
        top_n = st.slider("Top/Bottom N", 10, 50, 30, step=5)
        sector_filter = "All"

        st.divider()
        if st.button("Refresh cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        if REPORT_SCRIPT.exists():
            if st.button("Regenerate momentum_report.md", type="secondary", use_container_width=True):
                with st.spinner("Running momentum_report.py…"):
                    res = subprocess.run(
                        [sys.executable, str(REPORT_SCRIPT)],
                        cwd=str(REPORT_SCRIPT.parent),
                        capture_output=True,
                        text=True,
                    )
                if res.returncode == 0:
                    st.success("Report generated.")
                else:
                    st.error("Report generation failed.")
                with st.expander("Output log"):
                    st.code((res.stdout or "") + (res.stderr or ""), language="text")

    with st.spinner("Computing momentum…"):
        df = compute_momentum()

    if df.empty:
        st.warning("No momentum data. Make sure `prices_eod` has data and the `fmp_watchlist` table is populated.")
        return

    sectors = sorted(df["sector"].dropna().unique().tolist())
    with st.sidebar:
        sector_filter = st.selectbox("Sector filter", ["All"] + sectors)

    if sector_filter != "All":
        df = df[df["sector"] == sector_filter]

    latest_date = None
    try:
        with _connect() as conn:
            latest_date = conn.execute("SELECT MAX(price_date) FROM prices_eod").fetchone()[0]
    except Exception:
        pass

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Stocks ranked", f"{len(df):,}")
    with c2:
        st.metric("Data date", latest_date or "—")
    with c3:
        st.metric("Sector", sector_filter)

    st.divider()

    tab_top, tab_bottom, tab_period, tab_table = st.tabs(
        ["Top movers", "Bottom movers", "Period charts", "Full table"]
    )

    with tab_top:
        top_df = df.head(top_n).copy()
        for col in ["ret_1M", "ret_3M", "ret_6M", "ret_12M", "momentum_score"]:
            if col in top_df.columns:
                top_df[col] = top_df[col].apply(_fmt_pct if col != "momentum_score" else lambda v: f"{v:.1f}")
        show_cols = ["ticker", "company_name", "sector", "ret_1M", "ret_3M", "ret_6M", "ret_12M", "momentum_score"]
        show_cols = [c for c in show_cols if c in top_df.columns]
        st.dataframe(top_df[show_cols], use_container_width=True, hide_index=True)

    with tab_bottom:
        bot_df = df.tail(top_n).iloc[::-1].copy()
        for col in ["ret_1M", "ret_3M", "ret_6M", "ret_12M", "momentum_score"]:
            if col in bot_df.columns:
                bot_df[col] = bot_df[col].apply(_fmt_pct if col != "momentum_score" else lambda v: f"{v:.1f}")
        show_cols = ["ticker", "company_name", "sector", "ret_1M", "ret_3M", "ret_6M", "ret_12M", "momentum_score"]
        show_cols = [c for c in show_cols if c in bot_df.columns]
        st.dataframe(bot_df[show_cols], use_container_width=True, hide_index=True)

    with tab_period:
        p1, p2 = st.columns(2)
        with p1:
            st.plotly_chart(
                _bar_chart(df, "ret_1M", "1-Month Return — Top 20", top_n=20),
                use_container_width=True,
            )
            st.plotly_chart(
                _bar_chart(df, "ret_6M", "6-Month Return — Top 20", top_n=20),
                use_container_width=True,
            )
        with p2:
            st.plotly_chart(
                _bar_chart(df, "ret_3M", "3-Month Return — Top 20", top_n=20),
                use_container_width=True,
            )
            st.plotly_chart(
                _bar_chart(df, "ret_12M", "12-Month Return — Top 20", top_n=20),
                use_container_width=True,
            )

    with tab_table:
        st.caption(f"All {len(df):,} stocks sorted by composite momentum score (descending).")
        show = df[["ticker", "company_name", "sector", "ret_1M", "ret_3M", "ret_6M", "ret_12M", "momentum_score", "market_cap"]].copy()
        for col in ["ret_1M", "ret_3M", "ret_6M", "ret_12M"]:
            show[col] = show[col].apply(_fmt_pct)
        show["momentum_score"] = show["momentum_score"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
        show["market_cap"] = show["market_cap"].apply(
            lambda v: f"{v/1e9:,.1f}B" if pd.notna(v) and v else "—"
        )
        st.dataframe(show, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
