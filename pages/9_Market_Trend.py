"""
Market Trend Analysis

Uses local SQLite prices_eod data to evaluate broad market trend, breadth,
new highs/lows, volume expansion, and risk regime.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

from utils.local_data import connect_readonly

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "quant_data.db"
PRICE_LOOKBACK_DAYS = 560
UPDATE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update_prices_eod.py"


@dataclass(frozen=True)
class Regime:
    label: str
    tone: str
    score: int
    details: list[str]


def _connect() -> sqlite3.Connection:
    return connect_readonly(DB_PATH)


def _run_eod_update(days: int) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(UPDATE_SCRIPT),
        "--mode",
        "bulk",
        "--days",
        str(max(1, int(days))),
    ]
    return subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
        check=False,
    )


def _pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{float(value):.{digits}f}%"


def _fmt_num(value: float | int | None) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{float(value):,.0f}"


@st.cache_data(ttl=60 * 10, show_spinner=False)
def _latest_price_date() -> str | None:
    with _connect() as conn:
        return conn.execute("SELECT MAX(price_date) FROM prices_eod").fetchone()[0]


@st.cache_data(ttl=60 * 10, show_spinner=False)
def _load_market_frame(lookback_days: int = PRICE_LOOKBACK_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = _latest_price_date()
    if not latest:
        return pd.DataFrame(), pd.DataFrame()

    start = (pd.to_datetime(latest).date() - timedelta(days=lookback_days)).isoformat()
    with _connect() as conn:
        universe = pd.read_sql(
            """
            WITH latest_rows AS (
                SELECT ticker, MAX(price_date) AS latest_price_date
                FROM prices_eod
                GROUP BY ticker
            ),
            mcap AS (
                SELECT k.ticker, k.market_cap
                FROM key_metrics_ttm k
                JOIN (
                    SELECT ticker, MAX(as_of_date) AS as_of_date
                    FROM key_metrics_ttm
                    GROUP BY ticker
                ) latest_mcap
                  ON latest_mcap.ticker = k.ticker
                 AND latest_mcap.as_of_date = k.as_of_date
            )
            SELECT
                lr.ticker,
                lr.latest_price_date,
                cp.company_name,
                cp.sector,
                mcap.market_cap
            FROM latest_rows lr
            LEFT JOIN company_profile cp ON cp.ticker = lr.ticker
            LEFT JOIN mcap ON mcap.ticker = lr.ticker
            WHERE lr.latest_price_date = ?
              AND cp.company_name IS NOT NULL
              AND cp.sector IS NOT NULL
              AND length(lr.ticker) <= 5
              AND lr.ticker NOT LIKE '%%.%%'
              AND lr.ticker NOT LIKE '%%-%%'
            """,
            conn,
            params=(latest,),
        )

        tickers = universe["ticker"].dropna().astype(str).tolist()
        if not tickers:
            return pd.DataFrame(), pd.DataFrame()

        placeholders = ",".join("?" for _ in tickers)
        prices = pd.read_sql(
            f"""
            SELECT
                ticker,
                price_date,
                COALESCE(adj_close, close) AS price,
                close,
                volume
            FROM prices_eod
            WHERE price_date >= ?
              AND ticker IN ({placeholders})
              AND COALESCE(adj_close, close) IS NOT NULL
              AND COALESCE(adj_close, close) > 0
            ORDER BY ticker, price_date
            """,
            conn,
            params=[start, *tickers],
        )

    if prices.empty:
        return pd.DataFrame(), universe

    prices["price_date"] = pd.to_datetime(prices["price_date"])
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce").fillna(0.0)
    prices = prices.dropna(subset=["price"]).sort_values(["ticker", "price_date"])

    grouped = prices.groupby("ticker", group_keys=False)
    prices["ma20"] = grouped["price"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    prices["ma50"] = grouped["price"].rolling(50, min_periods=50).mean().reset_index(level=0, drop=True)
    prices["ma200"] = grouped["price"].rolling(200, min_periods=200).mean().reset_index(level=0, drop=True)
    prices["vol50"] = grouped["volume"].rolling(50, min_periods=20).mean().reset_index(level=0, drop=True)
    prices["hi252"] = grouped["price"].rolling(252, min_periods=120).max().reset_index(level=0, drop=True)
    prices["lo252"] = grouped["price"].rolling(252, min_periods=120).min().reset_index(level=0, drop=True)
    prices["ret1"] = grouped["price"].pct_change()

    return prices.merge(universe, on="ticker", how="left"), universe


def _latest_rows(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    idx = prices.groupby("ticker")["price_date"].idxmax()
    latest = prices.loc[idx].copy()
    latest = latest.dropna(subset=["ma50", "ma200"])
    latest["above_20ma"] = latest["price"] > latest["ma20"]
    latest["above_50ma"] = latest["price"] > latest["ma50"]
    latest["above_200ma"] = latest["price"] > latest["ma200"]
    latest["new_high"] = latest["price"] >= latest["hi252"]
    latest["new_low"] = latest["price"] <= latest["lo252"]
    latest["volume_expanded"] = (latest["volume"] > latest["vol50"] * 1.5) & (latest["volume"] > 0)
    return latest


def _make_segments(latest: pd.DataFrame) -> dict[str, list[str]]:
    ranked = latest.dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False)
    growth_sectors = {"Technology", "Communication Services", "Consumer Cyclical"}
    return {
        "S&P 500 proxy": ranked.head(500)["ticker"].tolist(),
        "Nasdaq growth proxy": ranked[ranked["sector"].isin(growth_sectors)].head(800)["ticker"].tolist(),
        "Russell proxy": ranked.iloc[1000:3000]["ticker"].tolist(),
    }


def _segment_stats(latest: pd.DataFrame, segments: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for name, tickers in segments.items():
        sub = latest[latest["ticker"].isin(tickers)]
        if sub.empty:
            continue
        rows.append(
            {
                "Market": name,
                "Stocks": len(sub),
                ">20MA": sub["above_20ma"].mean() * 100,
                ">50MA": sub["above_50ma"].mean() * 100,
                ">200MA": sub["above_200ma"].mean() * 100,
                "New highs": int(sub["new_high"].sum()),
                "New lows": int(sub["new_low"].sum()),
                "Volume expanded": sub["volume_expanded"].mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def _history_by_date(prices: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    sub = prices[prices["ticker"].isin(list(tickers))].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["above_50ma"] = sub["price"] > sub["ma50"]
    sub["above_200ma"] = sub["price"] > sub["ma200"]
    sub["new_high"] = sub["price"] >= sub["hi252"]
    sub["new_low"] = sub["price"] <= sub["lo252"]

    hist = (
        sub.groupby("price_date")
        .agg(
            breadth_50=("above_50ma", "mean"),
            breadth_200=("above_200ma", "mean"),
            new_highs=("new_high", "sum"),
            new_lows=("new_low", "sum"),
            avg_return=("ret1", "mean"),
            stocks=("ticker", "nunique"),
        )
        .reset_index()
    )
    hist["breadth_50"] *= 100
    hist["breadth_200"] *= 100
    hist["advance_index"] = (1 + hist["avg_return"].fillna(0)).cumprod() * 100
    return hist


def _classify_market(latest: pd.DataFrame, stats: pd.DataFrame) -> Regime:
    if latest.empty or stats.empty:
        return Regime("Insufficient Data", "info", 0, ["prices_eod does not have enough usable data to classify the market trend."])

    overall_50 = latest["above_50ma"].mean() * 100
    overall_200 = latest["above_200ma"].mean() * 100
    high_low_spread = int(latest["new_high"].sum() - latest["new_low"].sum())
    volume_expanded = latest["volume_expanded"].mean() * 100
    sp_proxy = stats[stats["Market"] == "S&P 500 proxy"]

    trend_score = 0
    details: list[str] = []

    if overall_50 >= 60:
        trend_score += 2
        details.append(f"50MA breadth strong: {_pct(overall_50)} of stocks are above 50MA.")
    elif overall_50 >= 45:
        trend_score += 1
        details.append(f"50MA breadth neutral: {_pct(overall_50)} of stocks are above 50MA.")
    else:
        trend_score -= 2
        details.append(f"50MA breadth weak: only {_pct(overall_50)} of stocks are above 50MA.")

    if overall_200 >= 55:
        trend_score += 2
        details.append(f"Long-term breadth supports the tape: {_pct(overall_200)} above 200MA.")
    elif overall_200 < 40:
        trend_score -= 2
        details.append(f"Long-term breadth is poor: {_pct(overall_200)} above 200MA.")

    if high_low_spread > 0:
        trend_score += 1
        details.append(f"New highs exceed new lows by {_fmt_num(high_low_spread)} stocks.")
    else:
        trend_score -= 1
        details.append(f"New lows exceed/equal new highs; spread is {_fmt_num(high_low_spread)}.")

    if not sp_proxy.empty:
        sp_20 = float(sp_proxy.iloc[0][">20MA"])
        sp_50 = float(sp_proxy.iloc[0][">50MA"])
        sp_200 = float(sp_proxy.iloc[0][">200MA"])
        if sp_20 >= sp_50 >= 50 and sp_200 >= 50:
            trend_score += 1
            details.append("Large-cap proxy has aligned short/intermediate/long-term support.")
        elif sp_20 < 40 and sp_50 < 45:
            trend_score -= 1
            details.append("Large-cap proxy is losing short/intermediate-term support.")

    if volume_expanded >= 18 and overall_50 >= 70:
        return Regime("High-Risk Overheated", "warning", trend_score, details + [f"Volume expansion is broad ({_pct(volume_expanded)}), while breadth is already elevated."])

    if trend_score >= 4:
        return Regime("Bull Trend Continuation", "success", trend_score, details)
    if trend_score >= 1:
        return Regime("Range-Bound / Choppy", "info", trend_score, details)
    if trend_score >= -2:
        return Regime("Weakening", "warning", trend_score, details)
    return Regime("High Risk", "error", trend_score, details)


def _breadth_chart(hist: pd.DataFrame, title: str) -> go.Figure:
    if go is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["price_date"], y=hist["breadth_50"], name="% above 50MA", line=dict(color="#2563eb", width=2)))
    fig.add_trace(go.Scatter(x=hist["price_date"], y=hist["breadth_200"], name="% above 200MA", line=dict(color="#16a34a", width=2)))
    fig.add_hline(y=50, line_color="#94a3b8", line_dash="dash")
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=20, r=20, t=48, b=20),
        yaxis_title="Breadth %",
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def _high_low_chart(hist: pd.DataFrame) -> go.Figure:
    if go is None:
        return None
    recent = hist.tail(90).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=recent["price_date"], y=recent["new_highs"], name="New highs", marker_color="#16a34a"))
    fig.add_trace(go.Bar(x=recent["price_date"], y=-recent["new_lows"], name="New lows", marker_color="#dc2626"))
    fig.update_layout(
        title="52-week new highs vs new lows (latest 90 sessions)",
        height=340,
        margin=dict(l=20, r=20, t=48, b=20),
        barmode="relative",
        yaxis_title="Stocks",
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def _trend_chart(histories: dict[str, pd.DataFrame]) -> go.Figure:
    if go is None:
        return None
    colors = ["#2563eb", "#db2777", "#f59e0b"]
    fig = go.Figure()
    for (name, hist), color in zip(histories.items(), colors):
        if hist.empty:
            continue
        fig.add_trace(go.Scatter(x=hist["price_date"], y=hist["advance_index"], name=name, line=dict(color=color, width=2)))
    fig.update_layout(
        title="Equal-weight trend proxies (base = 100)",
        height=380,
        margin=dict(l=20, r=20, t=48, b=20),
        yaxis_title="Index level",
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def main() -> None:
    st.title("Market Trend Analysis")
    st.caption("Data source: local SQLite `prices_eod`. Index groups are database-derived proxies, not official index constituent lists.")

    if not DB_PATH.exists():
        st.error(f"Database not found: `{DB_PATH}`")
        return

    with st.sidebar:
        st.header("Settings")
        st.caption(f"DB: `{DB_PATH}`")
        st.divider()
        st.header("Daily update")
        update_days = st.number_input("Backfill days", min_value=1, max_value=30, value=7, step=1)
        if st.button("Update prices_eod", type="primary", use_container_width=True):
            with st.spinner("Updating prices_eod from FMP..."):
                res = _run_eod_update(int(update_days))
            log = "\n".join(x for x in [res.stdout, res.stderr] if x.strip())
            if res.returncode == 0:
                st.success("prices_eod update finished.")
                st.cache_data.clear()
            else:
                st.error(f"prices_eod update failed (exit {res.returncode}).")
                st.info("If FMP blocks eod-bulk on your plan, run `scripts/update_prices_eod.py --mode per-symbol` from terminal.")
            with st.expander("Update log", expanded=res.returncode != 0):
                st.code(log or "(no output)", language="text")
        st.divider()
        if st.button("Refresh data cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Loading prices_eod and computing market breadth..."):
        prices, _universe = _load_market_frame()

    if prices.empty:
        st.warning("No usable rows found in `prices_eod`.")
        return

    latest = _latest_rows(prices)
    if latest.empty:
        st.warning("Not enough price history to compute 50MA/200MA breadth.")
        return

    latest_date = latest["price_date"].max().date().isoformat()
    segments = _make_segments(latest)
    stats = _segment_stats(latest, segments)
    regime = _classify_market(latest, stats)

    st.subheader(f"Market regime: {regime.label}")
    getattr(st, regime.tone)(f"Current market assessment: {regime.label} (score {regime.score:+d})")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Data date", latest_date)
    with c2:
        st.metric("Stocks analyzed", f"{len(latest):,}")
    with c3:
        st.metric("Above 50MA", _pct(latest["above_50ma"].mean() * 100))
    with c4:
        st.metric("Above 200MA", _pct(latest["above_200ma"].mean() * 100))
    with c5:
        st.metric("New high / low", f"{int(latest['new_high'].sum()):,} / {int(latest['new_low'].sum()):,}")

    with st.expander("Why this regime?", expanded=True):
        for item in regime.details:
            st.markdown(f"- {item}")

    st.divider()

    st.subheader("S&P 500 / Nasdaq / Russell proxy trend")
    st.dataframe(
        stats.assign(
            **{
                ">20MA": stats[">20MA"].map(lambda x: f"{x:.1f}%"),
                ">50MA": stats[">50MA"].map(lambda x: f"{x:.1f}%"),
                ">200MA": stats[">200MA"].map(lambda x: f"{x:.1f}%"),
                "Volume expanded": stats["Volume expanded"].map(lambda x: f"{x:.1f}%"),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    histories = {name: _history_by_date(prices, tickers) for name, tickers in segments.items()}
    trend_fig = _trend_chart(histories)
    if trend_fig is not None:
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        trend_fallback = pd.concat(
            [
                hist.set_index("price_date")["advance_index"].rename(name)
                for name, hist in histories.items()
                if not hist.empty
            ],
            axis=1,
        )
        st.line_chart(trend_fallback)

    st.divider()

    tab_breadth, tab_high_low, tab_volume, tab_detail = st.tabs(
        ["Market breadth", "New highs/lows", "Volume expansion", "Stock detail"]
    )

    with tab_breadth:
        default_hist = histories.get("S&P 500 proxy", pd.DataFrame())
        breadth_fig = _breadth_chart(default_hist, "Breadth trend: S&P 500 proxy")
        if breadth_fig is not None:
            st.plotly_chart(breadth_fig, use_container_width=True)
        else:
            st.line_chart(default_hist.set_index("price_date")[["breadth_50", "breadth_200"]])

    with tab_high_low:
        default_hist = histories.get("S&P 500 proxy", pd.DataFrame())
        high_low_fig = _high_low_chart(default_hist)
        if high_low_fig is not None:
            st.plotly_chart(high_low_fig, use_container_width=True)
        else:
            fallback = default_hist.tail(90).set_index("price_date")[["new_highs", "new_lows"]]
            st.bar_chart(fallback)

    with tab_volume:
        vol = (
            latest[latest["volume_expanded"]]
            .sort_values(["market_cap", "volume"], ascending=[False, False])
            [["ticker", "company_name", "sector", "price", "volume", "vol50", "above_50ma", "above_200ma"]]
            .head(100)
        )
        st.caption("Volume expanded means latest volume > 1.5x 50-day average volume.")
        st.dataframe(vol, use_container_width=True, hide_index=True)

    with tab_detail:
        filters = st.columns(3)
        with filters[0]:
            sector = st.selectbox("Sector", ["All", *sorted(latest["sector"].dropna().unique().tolist())])
        with filters[1]:
            condition = st.selectbox("Condition", ["All", "Above 50MA", "Below 50MA", "Above 200MA", "Below 200MA", "New high", "New low"])
        with filters[2]:
            sort_by = st.selectbox("Sort by", ["market_cap", "ticker", "sector", "price"])

        detail = latest.copy()
        if sector != "All":
            detail = detail[detail["sector"] == sector]
        if condition == "Above 50MA":
            detail = detail[detail["above_50ma"]]
        elif condition == "Below 50MA":
            detail = detail[~detail["above_50ma"]]
        elif condition == "Above 200MA":
            detail = detail[detail["above_200ma"]]
        elif condition == "Below 200MA":
            detail = detail[~detail["above_200ma"]]
        elif condition == "New high":
            detail = detail[detail["new_high"]]
        elif condition == "New low":
            detail = detail[detail["new_low"]]

        show = detail.sort_values(sort_by, ascending=(sort_by != "market_cap"))[
            [
                "ticker",
                "company_name",
                "sector",
                "price",
                "ma20",
                "ma50",
                "ma200",
                "volume",
                "market_cap",
                "above_50ma",
                "above_200ma",
                "new_high",
                "new_low",
            ]
        ].head(500)
        st.dataframe(show, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
