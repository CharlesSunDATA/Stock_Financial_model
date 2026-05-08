from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from utils.local_data import connect_readonly, default_db_path, latest_company_snapshot, load_adjusted_close, table_exists


@dataclass
class PriceTargets:
    low: float | None
    mean: float | None
    high: float | None
    median: float | None


def _safe_float(x: Any) -> float | None:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


@st.cache_data(ttl=60 * 30, show_spinner=False)
def _fetch_monthly_close(ticker: str, months: int) -> pd.Series:
    """Monthly close series aligned to the selected trend window."""
    end = date.today()
    start = end - timedelta(days=int(months * 31 + 45))
    local = load_adjusted_close([ticker], start, end, min_rows=max(2, months))
    sym = ticker.strip().upper()
    if sym in local.columns:
        close = local[sym].dropna()
        m = close.resample("ME").last()
        m.index = m.index.to_period("M").to_timestamp()
        if not m.dropna().empty:
            return m.dropna()

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        interval="1d",
    )
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    close = df["Close"].dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    m = close.resample("ME").last()
    # align month labels to month start (same as trend index)
    m.index = m.index.to_period("M").to_timestamp()
    return m.dropna()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def _fetch_info_and_price(ticker: str) -> tuple[dict[str, Any], float | None]:
    local_info = latest_company_snapshot(ticker)
    if local_info.get("currentPrice") is not None:
        return local_info, _safe_float(local_info.get("currentPrice"))

    t = yf.Ticker(ticker)
    info = t.info or {}
    px = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    if px is None:
        try:
            hist = t.history(period="5d", auto_adjust=True)
            if hist is not None and not hist.empty:
                px = float(hist["Close"].iloc[-1])
        except Exception:
            px = None
    return info, px


@st.cache_data(ttl=60 * 30, show_spinner=False)
def _fetch_recommendations(ticker: str) -> pd.DataFrame:
    """
    yfinance `Ticker.recommendations` is typically an events table (upgrades/downgrades).
    We'll convert it to monthly counts of Buy/Hold/Sell-style buckets using `To Grade`.
    """
    t = yf.Ticker(ticker)
    df = getattr(t, "recommendations", None)
    if df is None or df.empty:
        # More reliable fallback: upgrades/downgrades events table
        df = getattr(t, "upgrades_downgrades", None)
    if df is None or df.empty:
        # Newer yfinance versions expose getters
        for fn in ("get_recommendations", "get_upgrades_downgrades"):
            try:
                f = getattr(t, fn, None)
                if callable(f):
                    cand = f()
                    if cand is not None and not cand.empty:
                        df = cand
                        break
            except Exception:
                continue
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # normalize datetime index
    try:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    except Exception:
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def _fetch_upgrades_downgrades(ticker: str) -> pd.DataFrame:
    """More reliable revisions event table for many tickers."""
    local = _fetch_local_upgrades_downgrades(ticker)
    if not local.empty:
        return local

    t = yf.Ticker(ticker)
    df = getattr(t, "upgrades_downgrades", None)
    if df is None or df.empty:
        try:
            f = getattr(t, "get_upgrades_downgrades", None)
            if callable(f):
                df = f()
        except Exception:
            df = None
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    try:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    except Exception:
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def _fetch_price_targets(ticker: str) -> PriceTargets:
    local = _fetch_local_price_targets(ticker)
    if local is not None:
        return local

    t = yf.Ticker(ticker)
    # Prefer info fields (often available when analyst_price_target isn't)
    info = getattr(t, "info", None) or {}
    info_pt = {
        "low": info.get("targetLowPrice"),
        "mean": info.get("targetMeanPrice"),
        "high": info.get("targetHighPrice"),
        "median": info.get("targetMedianPrice"),
    }
    pt = getattr(t, "analyst_price_target", None) or {}
    # Newer yfinance getter
    if not pt:
        try:
            f = getattr(t, "get_analyst_price_targets", None)
            if callable(f):
                pt = f() or {}
        except Exception:
            pt = pt or {}
    return PriceTargets(
        low=_safe_float(pt.get("low") or info_pt.get("low")),
        mean=_safe_float(pt.get("mean") or info_pt.get("mean")),
        high=_safe_float(pt.get("high") or info_pt.get("high")),
        median=_safe_float(pt.get("median") or info_pt.get("median")),
    )


def _fetch_local_price_targets(ticker: str) -> PriceTargets | None:
    sym = ticker.strip().upper()
    db = default_db_path()
    if not sym or not db.exists():
        return None
    try:
        with connect_readonly(db) as conn:
            if not table_exists(conn, "price_target_consensus"):
                return None
            row = conn.execute(
                """
                SELECT target_low, target_consensus, target_high, target_median
                FROM price_target_consensus
                WHERE ticker = ?
                """,
                (sym,),
            ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    pt = PriceTargets(
        low=_safe_float(row[0]),
        mean=_safe_float(row[1]),
        high=_safe_float(row[2]),
        median=_safe_float(row[3]),
    )
    return pt if any(v is not None for v in (pt.low, pt.mean, pt.high, pt.median)) else None


def _fetch_local_upgrades_downgrades(ticker: str) -> pd.DataFrame:
    sym = ticker.strip().upper()
    db = default_db_path()
    if not sym or not db.exists():
        return pd.DataFrame()
    try:
        with connect_readonly(db) as conn:
            if not table_exists(conn, "stock_grade"):
                return pd.DataFrame()
            df = pd.read_sql_query(
                """
                SELECT date AS Date, grading_company AS Firm, previous_grade AS "From Grade", new_grade AS "To Grade"
                FROM stock_grade
                WHERE ticker = ?
                ORDER BY date
                """,
                conn,
                params=(sym,),
            )
    except Exception:
        return pd.DataFrame()
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    return df.sort_index()


def _bucket_grade(s: str) -> str | None:
    s = (s or "").strip().lower()
    if not s:
        return None
    # strong buy / buy
    if "strong buy" in s or "overweight" in s or "outperform" in s or "buy" == s:
        return "Buy"
    if "buy" in s:
        return "Buy"
    # hold
    if "hold" in s or "neutral" in s or "market perform" in s or "equal weight" in s:
        return "Hold"
    # sell
    if "sell" in s or "underweight" in s or "underperform" in s:
        return "Sell"
    return None


def _monthly_trend(recs: pd.DataFrame, months: int = 6) -> pd.DataFrame:
    if recs is None or recs.empty:
        return pd.DataFrame()

    df = recs.copy()
    if "To Grade" not in df.columns:
        normalized = {str(c).lower().replace(" ", "").replace("_", ""): c for c in df.columns}
        grade_col = normalized.get("tograde")
        if grade_col is not None:
            df["To Grade"] = df[grade_col]
    if "To Grade" not in df.columns:
        return pd.DataFrame()

    df["bucket"] = df["To Grade"].astype(str).map(_bucket_grade)
    df = df.dropna(subset=["bucket"])

    if df.empty:
        return pd.DataFrame()

    start = pd.Timestamp(date.today()) - pd.DateOffset(months=months)
    df = df[df.index >= start]
    if df.empty:
        return pd.DataFrame()

    m = (
        df.assign(month=df.index.to_period("M").to_timestamp())
        .groupby(["month", "bucket"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    for col in ("Buy", "Hold", "Sell"):
        if col not in m.columns:
            m[col] = 0
    return m[["Buy", "Hold", "Sell"]]


def _targets_ruler(
    ticker: str,
    price: float | None,
    pt: PriceTargets,
) -> go.Figure | None:
    if price is None or price <= 0:
        return None
    vals = [v for v in [pt.low, pt.mean, pt.high, pt.median, price] if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None

    x_min = min(vals) * 0.90
    x_max = max(vals) * 1.10
    y = 0.5

    fig = go.Figure()
    fig.add_shape(type="line", x0=x_min, x1=x_max, y0=y, y1=y, line=dict(color="rgba(255,255,255,0.25)", width=6))

    def add_marker(x: float, label: str, color: str):
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(size=14, color=color, line=dict(color="white", width=1)),
                text=[label],
                textposition="top center",
                textfont=dict(size=12, color=color),
                hovertemplate=f"{label}: USD {x:,.2f}<extra></extra>",
                showlegend=False,
            )
        )

    if pt.low is not None:
        add_marker(pt.low, f"Low\n{pt.low:,.0f}", "#95a5a6")
    if pt.mean is not None:
        add_marker(pt.mean, f"Mean\n{pt.mean:,.0f}", "#f1c40f")
    if pt.high is not None:
        add_marker(pt.high, f"High\n{pt.high:,.0f}", "#e74c3c")
    if price is not None:
        add_marker(price, f"Spot\n{price:,.0f}", "#3498db")

    fig.update_layout(
        title=dict(
            text=f"<b>{ticker} — Analyst price targets vs spot</b>",
            x=0,
            font=dict(size=14, color="white"),
        ),
        xaxis=dict(range=[x_min, x_max], showgrid=False, zeroline=False, tickfont=dict(color="#ccc")),
        yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False),
        height=260,
        margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


def render_analyst_sentiment(ticker: str, months: int) -> None:
    _info, price = _fetch_info_and_price(ticker)
    recs_ud = _fetch_upgrades_downgrades(ticker)
    recs = recs_ud if not recs_ud.empty else _fetch_recommendations(ticker)
    pt = _fetch_price_targets(ticker)

    st.subheader("1. Price target divergence")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Spot price (USD)", f"${price:,.2f}" if price else "—")
    with c2:
        st.metric("Target mean (USD)", f"${pt.mean:,.2f}" if pt.mean else "—")
    with c3:
        gap = None
        if price and pt.mean and price > 0:
            gap = (pt.mean / price - 1) * 100
        st.metric("Mean vs spot", f"{gap:+.1f}%" if gap is not None else "—")
    with c4:
        rng = None
        if pt.low and pt.high and pt.low > 0:
            rng = (pt.high / pt.low - 1) * 100
        st.metric("Target dispersion", f"{rng:.0f}%" if rng is not None else "—")

    fig_ruler = _targets_ruler(ticker, price, pt)
    if fig_ruler is not None:
        st.plotly_chart(fig_ruler, use_container_width=True)
    else:
        st.info("Not enough price target data to render the ruler.")

    st.divider()

    st.subheader("2. Recommendation revision momentum")
    trend = _monthly_trend(recs, months=months)
    if trend.empty:
        st.info("No bucketable recommendation events found in the selected window. Showing raw events below.")

    if not trend.empty:
        px_m = _fetch_monthly_close(ticker, months=months)
        px_m = px_m.reindex(trend.index).ffill()

        if len(trend) >= 2:
            cur_buy = int(trend["Buy"].iloc[-1])
            prev_buy = int(trend["Buy"].iloc[-2])
            if cur_buy >= 3 and (prev_buy == 0 or cur_buy >= int(prev_buy * 1.5)):
                st.success("✅ Positive revision momentum: Buy/Overweight events increased meaningfully in the last month.")

        fig = go.Figure()
        for name, color in [("Buy", "#2ecc71"), ("Hold", "#f1c40f"), ("Sell", "#e74c3c")]:
            fig.add_trace(
                go.Bar(
                    x=trend.index,
                    y=trend[name],
                    name=name,
                    marker_color=color,
                    opacity=0.85,
                    hovertemplate=f"{name}: %{{y}}<extra></extra>",
                )
            )

        if not px_m.empty:
            fig.add_trace(
                go.Scatter(
                    x=trend.index,
                    y=px_m.values,
                    mode="lines+markers",
                    name="Price",
                    line=dict(color="#3498db", width=2),
                    marker=dict(size=5, color="#3498db"),
                    yaxis="y2",
                    hovertemplate="Close: $%{y:,.2f}<extra></extra>",
                )
            )
        fig.update_layout(
            barmode="stack",
            height=360,
            margin=dict(t=40, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title=dict(text="Monthly revision events (bucketed)", x=0, font=dict(size=13)),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False),
            yaxis2=dict(
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False,
                tickprefix="$",
                tickfont=dict(color="#9bbbe6"),
                title=dict(text="Price", font=dict(color="#9bbbe6", size=11)),
            ),
            legend=dict(orientation="h", y=1.1, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw revision events", expanded=False):
        st.dataframe(recs.tail(200), use_container_width=True)


def main() -> None:
    st.title("Analyst Sentiment Dashboard")
    st.caption("Use analyst revisions as a sentiment indicator (local SQLite first, yfinance fallback).")

    with st.sidebar:
        st.header("Ticker")
        ticker = st.text_input("Ticker symbol", value="NVDA").strip().upper() or "NVDA"
        months = st.slider("Trend window (months)", 3, 12, 6)

    render_analyst_sentiment(ticker, months)


if __name__ == "__main__":
    main()
