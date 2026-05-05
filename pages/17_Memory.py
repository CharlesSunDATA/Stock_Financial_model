"""
Memory Industry Sector Tracker

Leading-indicator dashboard for the memory semiconductor industry.
Tracks AI/HBM demand, hyperscaler capex, equipment bookings, and pricing environment
to anticipate DRAM/NAND cycle inflection points.

Sections:
  1. Relative Price Performance — memory stocks vs SPY
  2. Momentum Ranking — 1W / 1M / 3M / 6M returns
  3. AI / HBM Demand Proxy — NVDA + AMD revenue as GPU shipment leading indicator
  4. Hyperscaler Capex Tracker — META / GOOGL / AMZN / MSFT quarterly capex
  5. Memory Equipment Proxy — LRCX / AMAT / KLAC revenue (equipment bookings = supply investment signal)
  6. Memory Pricing Proxy — MU gross margin trend (DRAM pricing environment)
  7. Earnings Call Keyword Scanner — HBM / DDR5 / inventory / pricing signals
  8. Signal Summary — aggregated bull / bear read across all indicators
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from utils.data_loader import fetch_quarterly_metrics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEMORY_STOCKS: dict[str, str] = {
    "MU":   "Micron Technology",
    "WDC":  "Western Digital",
    "SNDK": "SanDisk",
    "STX":  "Seagate",
    "MRVL": "Marvell Technology",
}

AI_GPU_TICKERS: dict[str, str] = {
    "NVDA": "NVIDIA",
    "AMD":  "AMD",
}

HYPERSCALERS: dict[str, str] = {
    "META":  "Meta",
    "GOOGL": "Alphabet",
    "AMZN":  "Amazon",
    "MSFT":  "Microsoft",
}

EQUIPMENT_STOCKS: dict[str, str] = {
    "LRCX": "Lam Research",
    "AMAT": "Applied Materials",
    "KLAC": "KLA Corp",
}

BENCHMARK = "SPY"
PRICING_PROXY_TICKER = "MU"

MOMENTUM_WINDOWS: dict[str, int] = {
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
}

BULLISH_KEYWORDS = [
    "hbm", "hbm3e", "hbm4", "high bandwidth memory",
    "ddr5", "lpddr5", "cxl", "compute express link",
    "design win", "undersupply", "tight supply", "pricing improvement",
    "ai memory", "ramp", "ramping", "backlog", "record revenue",
    "capacity expansion", "data center", "accelerat", "strong demand",
    "double", "triple", "record", "hyperscaler",
]

BEARISH_KEYWORDS = [
    "inventory", "oversupply", "pricing pressure", "weak demand",
    "asp decline", "bit growth", "commodit", "correction",
    "excess inventory", "pushout", "push-out", "delay",
    "softness", "slow", "cautious", "headwind", "decline",
    "disappoint", "cancel", "lower guidance", "customer inventory",
]

SIGNAL_COLORS = {
    "Bullish": "#2ecc71",
    "Neutral": "#f1c40f",
    "Bearish": "#e74c3c",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _load_prices(tickers: tuple[str, ...], days: int) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=days + 30)
    try:
        raw = yf.download(
            list(tickers),
            start=str(start),
            end=str(end),
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else (
        raw[["Close"]] if "Close" in raw.columns else raw
    )
    return closes.dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)
def _load_capex(ticker: str) -> pd.DataFrame:
    df = fetch_quarterly_metrics(ticker, num_quarters=12)
    if df.empty or "capex" not in df.columns:
        return pd.DataFrame()
    out = df[["period_end", "capex", "total_revenue"]].copy()
    out = out.dropna(subset=["capex"])
    out["capex_abs"] = out["capex"].abs()
    out["capex_yoy_pct"] = out["capex_abs"].pct_change(4) * 100
    out["ticker"] = ticker
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _load_revenue(ticker: str, num_quarters: int = 10) -> pd.DataFrame:
    df = fetch_quarterly_metrics(ticker, num_quarters=num_quarters)
    if df.empty or "total_revenue" not in df.columns:
        return pd.DataFrame()
    out = df[["period_end", "total_revenue", "gross_margin_pct", "operating_margin_pct"]].copy()
    out = out.dropna(subset=["total_revenue"])
    out["revenue_yoy_pct"] = out["total_revenue"].pct_change(4) * 100
    return out


def _momentum_table(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        if tk not in prices.columns:
            continue
        s = prices[tk].dropna()
        if len(s) < 2:
            continue
        last = s.iloc[-1]
        row: dict[str, object] = {"Ticker": tk}
        for label, window in MOMENTUM_WINDOWS.items():
            row[label] = (last / s.iloc[-window - 1] - 1) * 100 if len(s) > window else None
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Score"] = df[["1M", "3M", "6M"]].mean(axis=1)
    return df.sort_values("Score", ascending=False).reset_index(drop=True)


def _signal(val: float | None, threshold_bull: float = 5.0, threshold_bear: float = -5.0) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "Neutral"
    if val > threshold_bull:
        return "Bullish"
    if val < threshold_bear:
        return "Bearish"
    return "Neutral"


def _pct_fmt(v) -> str:
    if pd.isna(v):
        return "—"
    return f"+{v:.1f}%" if v > 0 else f"{v:.1f}%"


def _scan_keywords(text: str) -> dict[str, object]:
    lower = text.lower()
    bull_hits = [kw for kw in BULLISH_KEYWORDS if kw in lower]
    bear_hits = [kw for kw in BEARISH_KEYWORDS if kw in lower]
    score = len(bull_hits) - len(bear_hits)
    signal = "Bullish" if score > 2 else ("Bearish" if score < -1 else "Neutral")
    return {
        "bull_count": len(bull_hits),
        "bear_count": len(bear_hits),
        "bull_hits": bull_hits,
        "bear_hits": bear_hits,
        "score": score,
        "signal": signal,
    }


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

_DARK_LAYOUT = dict(
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font_color="#fafafa",
    xaxis=dict(gridcolor="#1e2530"),
    yaxis=dict(gridcolor="#1e2530"),
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
)


def _relative_perf_chart(prices: pd.DataFrame, tickers: list[str], days: int) -> go.Figure:
    cutoff = prices.index[-1] - pd.Timedelta(days=days)
    sub = prices[prices.index >= cutoff]
    fig = go.Figure()
    for tk in tickers + [BENCHMARK]:
        if tk not in sub.columns:
            continue
        s = sub[tk].dropna()
        if s.empty:
            continue
        norm = s / s.iloc[0] * 100
        is_bench = tk == BENCHMARK
        fig.add_trace(go.Scatter(
            x=norm.index,
            y=norm.values,
            name=tk,
            line=dict(
                width=3 if is_bench else 1.5,
                dash="dash" if is_bench else "solid",
                color="#888888" if is_bench else None,
            ),
            hovertemplate=f"<b>{tk}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}<extra></extra>",
        ))
    fig.add_hline(y=100, line_dash="dot", line_color="#444444", line_width=1)
    fig.update_layout(height=380, yaxis_title="Indexed (start = 100)", **_DARK_LAYOUT)
    return fig


def _revenue_chart(df: pd.DataFrame, ticker: str, color: str = "#76b900") -> go.Figure:
    fig = go.Figure()
    x = df["period_end"].dt.to_period("Q").astype(str)
    fig.add_trace(go.Bar(
        x=x,
        y=df["total_revenue"] / 1e9,
        name="Revenue ($B)",
        marker_color=color,
        hovertemplate=f"<b>{ticker} Revenue</b><br>%{{x}}<br>$%{{y:.1f}}B<extra></extra>",
    ))
    valid = df["revenue_yoy_pct"].notna()
    if valid.any():
        fig.add_trace(go.Scatter(
            x=x[valid],
            y=df.loc[valid, "revenue_yoy_pct"],
            name="YoY Growth (%)",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#f1c40f", width=2),
            hovertemplate="YoY: %{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        height=300,
        yaxis=dict(title="Revenue ($B)", gridcolor="#1e2530"),
        yaxis2=dict(title="YoY Growth (%)", overlaying="y", side="right", showgrid=False),
        **{k: v for k, v in _DARK_LAYOUT.items() if k not in ("yaxis",)},
    )
    return fig


def _capex_chart(capex_data: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for tk, df in capex_data.items():
        if df.empty:
            continue
        x = df["period_end"].dt.to_period("Q").astype(str)
        fig.add_trace(go.Bar(
            x=x,
            y=df["capex_abs"] / 1e9,
            name=f"{tk} ({HYPERSCALERS.get(tk, tk)})",
            hovertemplate=f"<b>{tk}</b><br>Quarter: %{{x}}<br>Capex: $%{{y:.1f}}B<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        height=360,
        yaxis_title="Capex (USD Billions)",
        **_DARK_LAYOUT,
    )
    return fig


def _capex_yoy_chart(capex_data: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for tk, df in capex_data.items():
        sub = df.dropna(subset=["capex_yoy_pct"])
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["period_end"].dt.to_period("Q").astype(str),
            y=sub["capex_yoy_pct"],
            name=tk,
            mode="lines+markers",
            hovertemplate=f"<b>{tk}</b><br>%{{x}}<br>YoY: %{{y:.1f}}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="#888888", line_width=1)
    fig.update_layout(height=300, yaxis_title="Capex YoY Growth (%)", **_DARK_LAYOUT)
    return fig


def _margin_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    plot = df.dropna(subset=["gross_margin_pct"]).copy()
    x = plot["period_end"].dt.to_period("Q").astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=plot["gross_margin_pct"],
        name="Gross Margin %",
        mode="lines+markers",
        line=dict(color="#3498db", width=2),
        fill="tozeroy",
        fillcolor="rgba(52,152,219,0.12)",
        hovertemplate=f"<b>{ticker} GM%</b><br>%{{x}}<br>%{{y:.1f}}%<extra></extra>",
    ))
    if "operating_margin_pct" in plot.columns:
        fig.add_trace(go.Scatter(
            x=x,
            y=plot["operating_margin_pct"],
            name="Operating Margin %",
            mode="lines+markers",
            line=dict(color="#e67e22", width=1.5, dash="dot"),
            hovertemplate=f"<b>{ticker} OP%</b><br>%{{x}}<br>%{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        height=300,
        yaxis=dict(title="Margin (%)", gridcolor="#1e2530"),
        **{k: v for k, v in _DARK_LAYOUT.items() if k not in ("yaxis",)},
    )
    return fig


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title("Memory Industry Sector Tracker")
st.caption(
    "Leading-indicator view: AI/HBM demand, hyperscaler capex, equipment bookings, and pricing environment. "
    "Data via yfinance (live)."
)

with st.sidebar:
    st.header("Settings")
    perf_window = st.selectbox(
        "Performance window", [30, 60, 90, 180, 365], index=2,
        format_func=lambda d: f"{d}D",
    )
    custom_raw = st.text_input(
        "Add memory tickers (comma-separated)",
        placeholder="e.g. KIOXIA, SKX",
    )

memory_tickers = list(MEMORY_STOCKS.keys())
if custom_raw.strip():
    for t in custom_raw.upper().split(","):
        t = t.strip()
        if t and t not in memory_tickers:
            memory_tickers.append(t)
            MEMORY_STOCKS[t] = t

all_price_tickers = tuple(sorted(set(memory_tickers + [BENCHMARK])))

with st.spinner("Loading price data..."):
    prices = _load_prices(all_price_tickers, days=perf_window + 30)
prices.index = pd.to_datetime(prices.index)


# ── Section 1: Relative Performance ────────────────────────────────────────
st.subheader("Relative Price Performance")
st.caption(f"Memory stocks vs SPY — last {perf_window} days (indexed to 100 at start)")

if not prices.empty:
    st.plotly_chart(_relative_perf_chart(prices, memory_tickers, perf_window), use_container_width=True)
else:
    st.warning("Price data unavailable. Check network connection.")


# ── Section 2: Momentum Ranking ────────────────────────────────────────────
st.subheader("Momentum Ranking")

mom_df = _momentum_table(prices, memory_tickers)
if not mom_df.empty:
    mom_df.insert(1, "Company", mom_df["Ticker"].map(lambda t: MEMORY_STOCKS.get(t, t)))
    display_df = mom_df.copy()
    for col in ["1W", "1M", "3M", "6M", "Score"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(_pct_fmt)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker":  st.column_config.TextColumn("Ticker", width="small"),
            "Company": st.column_config.TextColumn("Company"),
            "1W":      st.column_config.TextColumn("1W"),
            "1M":      st.column_config.TextColumn("1M"),
            "3M":      st.column_config.TextColumn("3M"),
            "6M":      st.column_config.TextColumn("6M"),
            "Score":   st.column_config.TextColumn("Avg Score (1M-6M)"),
        },
    )
else:
    st.info("No momentum data available.")

st.divider()


# ── Section 3: AI / HBM Demand Proxy ───────────────────────────────────────
st.subheader("AI / HBM Demand Proxy")
st.caption(
    "NVDA and AMD quarterly revenue growth is the primary leading indicator for HBM demand. "
    "Each AI GPU requires HBM — accelerating GPU shipments drive HBM capacity allocation 1–3 quarters ahead. "
    "MU is currently the leading HBM3E supplier."
)

gpu_tickers = tuple(sorted(set(list(AI_GPU_TICKERS.keys()) + [BENCHMARK])))
with st.spinner("Loading NVDA / AMD prices..."):
    gpu_prices = _load_prices(gpu_tickers, days=perf_window + 30)
gpu_prices.index = pd.to_datetime(gpu_prices.index)

col_nvda, col_amd = st.columns(2)

with col_nvda:
    st.markdown("**NVIDIA (NVDA)**")
    with st.spinner("Loading NVDA financials..."):
        nvda_df = _load_revenue("NVDA")
    if not nvda_df.empty:
        st.plotly_chart(_revenue_chart(nvda_df, "NVDA", color="#76b900"), use_container_width=True)
        latest = nvda_df.iloc[-1]
        yoy = latest.get("revenue_yoy_pct")
        sig = _signal(yoy, threshold_bull=30, threshold_bear=0)
        m1, m2, m3 = st.columns(3)
        m1.metric("Revenue", f"${latest['total_revenue'] / 1e9:.1f}B" if pd.notna(latest["total_revenue"]) else "—")
        m2.metric("YoY", f"{yoy:.0f}%" if pd.notna(yoy) else "—")
        m3.markdown(
            f"<div style='padding:5px; border-radius:5px; background:{SIGNAL_COLORS[sig]}22; "
            f"border:1px solid {SIGNAL_COLORS[sig]}; color:{SIGNAL_COLORS[sig]}; "
            f"font-weight:600; text-align:center; margin-top:6px;'>{sig}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("NVDA data unavailable.")

with col_amd:
    st.markdown("**AMD**")
    with st.spinner("Loading AMD financials..."):
        amd_df = _load_revenue("AMD")
    if not amd_df.empty:
        st.plotly_chart(_revenue_chart(amd_df, "AMD", color="#ed1c24"), use_container_width=True)
        latest = amd_df.iloc[-1]
        yoy = latest.get("revenue_yoy_pct")
        sig = _signal(yoy, threshold_bull=20, threshold_bear=0)
        m1, m2, m3 = st.columns(3)
        m1.metric("Revenue", f"${latest['total_revenue'] / 1e9:.1f}B" if pd.notna(latest["total_revenue"]) else "—")
        m2.metric("YoY", f"{yoy:.0f}%" if pd.notna(yoy) else "—")
        m3.markdown(
            f"<div style='padding:5px; border-radius:5px; background:{SIGNAL_COLORS[sig]}22; "
            f"border:1px solid {SIGNAL_COLORS[sig]}; color:{SIGNAL_COLORS[sig]}; "
            f"font-weight:600; text-align:center; margin-top:6px;'>{sig}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("AMD data unavailable.")

st.caption("NVDA/AMD vs SPY performance")
if not gpu_prices.empty:
    fig_gpu = _relative_perf_chart(gpu_prices, list(AI_GPU_TICKERS.keys()), perf_window)
    fig_gpu.update_layout(height=280)
    st.plotly_chart(fig_gpu, use_container_width=True)

st.divider()


# ── Section 4: Hyperscaler Capex Tracker ───────────────────────────────────
st.subheader("Hyperscaler Capex Tracker")
st.caption(
    "Hyperscaler capex drives server DRAM demand and HBM procurement decisions. "
    "Accelerating capex = more AI servers = more DRAM/HBM orders. "
    "Typical lead time: 1–3 quarters for DRAM, 2–4 quarters for HBM design wins."
)

with st.spinner("Loading hyperscaler financials..."):
    capex_data: dict[str, pd.DataFrame] = {tk: _load_capex(tk) for tk in HYPERSCALERS}

valid_capex = {tk: df for tk, df in capex_data.items() if not df.empty}

if valid_capex:
    tab_abs, tab_yoy = st.tabs(["Quarterly Capex (Absolute)", "YoY Growth Rate"])
    with tab_abs:
        st.plotly_chart(_capex_chart(valid_capex), use_container_width=True)
    with tab_yoy:
        st.plotly_chart(_capex_yoy_chart(valid_capex), use_container_width=True)
        st.caption("YoY growth > 20% sustained = strong server DRAM / HBM demand catalyst.")

    rows = []
    for tk, df in valid_capex.items():
        latest = df.iloc[-1]
        yoy = latest.get("capex_yoy_pct")
        yoy_val = yoy if pd.notna(yoy) else None
        rows.append({
            "Company":        f"{tk} ({HYPERSCALERS[tk]})",
            "Latest Quarter": str(latest["period_end"])[:7],
            "Capex ($B)":     f"${latest['capex_abs'] / 1e9:.1f}B",
            "YoY Growth":     f"+{yoy_val:.0f}%" if yoy_val and yoy_val > 0 else (f"{yoy_val:.0f}%" if yoy_val else "—"),
            "Signal":         _signal(yoy_val, threshold_bull=15, threshold_bear=-5),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.warning("Could not load hyperscaler capex data. yfinance may be rate-limited.")

st.divider()


# ── Section 5: Memory Equipment Proxy ──────────────────────────────────────
st.subheader("Memory Equipment Proxy")
st.caption(
    "LRCX, AMAT, and KLAC equipment revenue tracks memory fab investment cycles. "
    "Rising equipment bookings signal that memory makers are expanding capacity — "
    "leading indicator for future oversupply risk (6–12 months)."
)

eq_price_tickers = tuple(sorted(set(list(EQUIPMENT_STOCKS.keys()) + [BENCHMARK])))
with st.spinner("Loading equipment stock prices..."):
    eq_prices = _load_prices(eq_price_tickers, days=perf_window + 30)
eq_prices.index = pd.to_datetime(eq_prices.index)

if not eq_prices.empty:
    fig_eq = _relative_perf_chart(eq_prices, list(EQUIPMENT_STOCKS.keys()), perf_window)
    st.plotly_chart(fig_eq, use_container_width=True)

    eq_mom = _momentum_table(eq_prices, list(EQUIPMENT_STOCKS.keys()))
    if not eq_mom.empty:
        eq_mom.insert(1, "Company", eq_mom["Ticker"].map(lambda t: EQUIPMENT_STOCKS.get(t, t)))
        display_eq = eq_mom.copy()
        for col in ["1W", "1M", "3M", "6M", "Score"]:
            if col in display_eq.columns:
                display_eq[col] = display_eq[col].apply(_pct_fmt)
        st.dataframe(display_eq, use_container_width=True, hide_index=True)
else:
    st.info("Equipment stock price data unavailable.")

st.divider()


# ── Section 6: Memory Pricing Proxy ────────────────────────────────────────
st.subheader("Memory Pricing Proxy")
st.caption(
    "Micron (MU) gross margin is the best public proxy for DRAM spot pricing. "
    "Rising gross margin = DRAM pricing firming / HBM mix improving. "
    "Margin compression = spot price decline or commodity DRAM oversupply."
)

with st.spinner("Loading MU financials..."):
    mu_df = _load_revenue(PRICING_PROXY_TICKER, num_quarters=12)

if not mu_df.empty and "gross_margin_pct" in mu_df.columns:
    st.plotly_chart(_margin_chart(mu_df, "MU"), use_container_width=True)

    plot_mu = mu_df.dropna(subset=["gross_margin_pct"])
    latest_mu = plot_mu.iloc[-1]
    prev_mu = plot_mu.iloc[-5] if len(plot_mu) >= 5 else None
    gm_now = latest_mu["gross_margin_pct"]
    gm_delta = (
        gm_now - prev_mu["gross_margin_pct"]
        if prev_mu is not None and pd.notna(prev_mu["gross_margin_pct"])
        else None
    )
    pricing_sig = _signal(gm_delta, threshold_bull=3.0, threshold_bear=-3.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("MU Gross Margin", f"{gm_now:.1f}%" if pd.notna(gm_now) else "—")
    c2.metric("YoY Change (pp)", f"{gm_delta:+.1f}pp" if gm_delta is not None else "—")
    c3.markdown(
        f"<div style='padding:6px; border-radius:5px; background:{SIGNAL_COLORS[pricing_sig]}22; "
        f"border:1px solid {SIGNAL_COLORS[pricing_sig]}; color:{SIGNAL_COLORS[pricing_sig]}; "
        f"font-weight:600; text-align:center; margin-top:8px;'>{pricing_sig} (Pricing Proxy)</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("MU financial data unavailable.")

st.divider()


# ── Section 7: Earnings Call Keyword Scanner ────────────────────────────────
st.subheader("Earnings Call Keyword Scanner")
st.caption(
    "Paste earnings call transcript or press release text from MU, WDC, LRCX, or hyperscaler calls. "
    "Scanner counts bullish (HBM ramp, DDR5 design wins, pricing improvement) vs "
    "bearish (inventory correction, oversupply, ASP pressure) signals."
)

transcript_text = st.text_area(
    "Paste earnings call / press release text here",
    height=160,
    placeholder="Paste transcript text from MU, WDC, LRCX, AMAT or hyperscaler earnings calls...",
)

if transcript_text.strip():
    result = _scan_keywords(transcript_text)
    sig_color = SIGNAL_COLORS[result["signal"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bullish Keywords", result["bull_count"])
    c2.metric("Bearish Keywords", result["bear_count"])
    c3.metric("Net Score", result["score"])
    c4.markdown(
        f"<div style='padding:6px; border-radius:5px; background:{sig_color}22; "
        f"border:1px solid {sig_color}; color:{sig_color}; font-weight:600; text-align:center; margin-top:8px;'>"
        f"Order Signal: {result['signal']}</div>",
        unsafe_allow_html=True,
    )

    if result["bull_hits"] or result["bear_hits"]:
        with st.expander("Matched keywords"):
            col_b, col_br = st.columns(2)
            with col_b:
                st.markdown("**Bullish hits:**")
                for kw in result["bull_hits"]:
                    st.markdown(f"- `{kw}`")
            with col_br:
                st.markdown("**Bearish hits:**")
                for kw in result["bear_hits"]:
                    st.markdown(f"- `{kw}`")
else:
    st.info("Paste text above to run the keyword scan.")

st.divider()


# ── Section 8: Signal Summary ───────────────────────────────────────────────
st.subheader("Signal Summary")
st.caption("Aggregated read from all leading indicators: momentum, AI/HBM demand, hyperscaler capex, equipment proxy, and memory pricing.")

signal_rows = []

# Memory stock momentum
if not mom_df.empty and "Score" in mom_df.columns:
    for _, r in mom_df.iterrows():
        signal_rows.append({
            "Source": f"Momentum — {r['Ticker']}",
            "Signal": _signal(r.get("Score")),
        })

# AI GPU demand proxy
for tk, df in [("NVDA", nvda_df), ("AMD", amd_df)]:
    if not df.empty:
        yoy = df.iloc[-1].get("revenue_yoy_pct")
        if pd.notna(yoy):
            signal_rows.append({
                "Source": f"{tk} Revenue YoY (AI/HBM Proxy)",
                "Signal": _signal(yoy, threshold_bull=30 if tk == "NVDA" else 20, threshold_bear=0),
            })

# Hyperscaler capex
for tk, df in valid_capex.items():
    yoy = df.iloc[-1].get("capex_yoy_pct")
    if pd.notna(yoy):
        signal_rows.append({
            "Source": f"Hyperscaler Capex YoY — {tk}",
            "Signal": _signal(yoy, threshold_bull=15, threshold_bear=-5),
        })

# Equipment proxy
if not eq_prices.empty:
    eq_avg_score = _momentum_table(eq_prices, list(EQUIPMENT_STOCKS.keys()))
    if not eq_avg_score.empty and "Score" in eq_avg_score.columns:
        avg = eq_avg_score["Score"].mean()
        signal_rows.append({
            "Source": "Memory Equipment Avg Momentum (Capex Proxy)",
            "Signal": _signal(avg, threshold_bull=5, threshold_bear=-5),
        })

# Memory pricing proxy
if not mu_df.empty and "gross_margin_pct" in mu_df.columns:
    gm_data = mu_df.dropna(subset=["gross_margin_pct"])
    if len(gm_data) >= 5:
        gm_delta_sig = gm_data.iloc[-1]["gross_margin_pct"] - gm_data.iloc[-5]["gross_margin_pct"]
        signal_rows.append({
            "Source": "MU Gross Margin YoY (DRAM Pricing Proxy)",
            "Signal": _signal(gm_delta_sig, threshold_bull=3.0, threshold_bear=-3.0),
        })

if signal_rows:
    sig_df = pd.DataFrame(signal_rows)
    bull = (sig_df["Signal"] == "Bullish").sum()
    bear = (sig_df["Signal"] == "Bearish").sum()
    total = len(sig_df)
    neutral = total - bull - bear

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bullish", bull)
    col2.metric("Neutral", neutral)
    col3.metric("Bearish", bear)

    score = (bull - bear) / total * 100 if total > 0 else 0
    if score > 20:
        label, color = "Sector Bullish", SIGNAL_COLORS["Bullish"]
    elif score < -20:
        label, color = "Sector Bearish", SIGNAL_COLORS["Bearish"]
    else:
        label, color = "Sector Neutral", SIGNAL_COLORS["Neutral"]

    col4.markdown(
        f"<div style='padding:8px; border-radius:6px; background:{color}22; "
        f"border:1px solid {color}; color:{color}; font-weight:600; text-align:center;'>"
        f"{label}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("View all signals"):
        st.dataframe(sig_df, use_container_width=True, hide_index=True)
else:
    st.info("No signals available — check that price and financial data loaded correctly.")

st.divider()
st.caption(
    "**Key leading indicators to watch:** "
    "NVDA/AMD revenue acceleration (1–3Q HBM lead) · "
    "Hyperscaler capex growth > 20% YoY (1–3Q DRAM lead) · "
    "MU gross margin inflection (spot price proxy) · "
    "Equipment bookings deceleration (supply cycle risk 6–12M out) · "
    "HBM / DDR5 design win announcements · "
    "Inventory days normalization"
)
