"""
Optical Communications Sector Tracker

Leading-indicator dashboard for the optical communications industry.
Tracks hyperscaler capex (primary demand driver) and sector stock momentum
to anticipate inflection points before they appear in financial results.

Sections:
  1. Relative Price Performance — optical stocks vs SPY
  2. Momentum Ranking — 1W / 1M / 3M / 6M returns
  3. Hyperscaler Capex Tracker — META / GOOGL / AMZN / MSFT quarterly capex
  4. NVIDIA GPU Proxy — NVDA revenue as GPU shipment leading indicator
  5. Data Center Construction Proxy — DC REIT (EQIX / DLR / IRM / AMT) momentum
  6. 800G / 1.6T Order Keyword Scanner — earnings call NLP
  7. Coherent Transceiver Pricing Proxy — COHR gross margin trend
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

OPTICAL_STOCKS: dict[str, str] = {
    "CIEN":  "Ciena",
    "COHR":  "Coherent Corp",
    "LITE":  "Lumentum",
    "AAOI":  "Applied Optoelectronics",
    "VIAV":  "VIAVI Solutions",
    "MACOM": "MACOM Technology",
}

HYPERSCALERS: dict[str, str] = {
    "META":  "Meta",
    "GOOGL": "Alphabet",
    "AMZN":  "Amazon",
    "MSFT":  "Microsoft",
}

BENCHMARK = "SPY"

MOMENTUM_WINDOWS: dict[str, int] = {
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
}

DC_REITS: dict[str, str] = {
    "EQIX": "Equinix",
    "DLR":  "Digital Realty",
    "IRM":  "Iron Mountain",
    "AMT":  "American Tower",
}

NVIDIA_TICKER = "NVDA"

# Keywords for 800G/1.6T order sentiment scanner
BULLISH_ORDER_KEYWORDS = [
    "800g", "1.6t", "800 gig", "1.6 terabit",
    "ai cluster", "hyperscaler", "design win", "record revenue",
    "strong demand", "backlog", "ramping", "ramp", "accelerat",
    "ai interconnect", "datacenter", "data center", "transceiver",
    "double", "triple", "record",
]
BEARISH_ORDER_KEYWORDS = [
    "pushout", "push-out", "inventory", "excess inventory",
    "delay", "weak demand", "softness", "slow", "cautious",
    "pricing pressure", "headwind", "decline", "disappoint",
    "cancel", "lower guidance",
]

SIGNAL_COLORS = {
    "Bullish":  "#2ecc71",
    "Neutral":  "#f1c40f",
    "Bearish":  "#e74c3c",
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
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]] if "Close" in raw.columns else raw
    closes = closes.dropna(how="all")
    return closes


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
            if len(s) > window:
                row[label] = (last / s.iloc[-window - 1] - 1) * 100
            else:
                row[label] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Score"] = df[["1M", "3M", "6M"]].mean(axis=1)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df


def _signal(val: float | None, threshold_bull: float = 5.0, threshold_bear: float = -5.0) -> str:
    if val is None or np.isnan(val):
        return "Neutral"
    if val > threshold_bull:
        return "Bullish"
    if val < threshold_bear:
        return "Bearish"
    return "Neutral"


def _color_momentum(val):
    if pd.isna(val):
        return "color: #888888"
    return f"color: {'#2ecc71' if val > 0 else '#e74c3c'}"


@st.cache_data(ttl=3600, show_spinner=False)
def _load_nvda_quarterly() -> pd.DataFrame:
    df = fetch_quarterly_metrics(NVIDIA_TICKER, num_quarters=10)
    if df.empty:
        return pd.DataFrame()
    cols = ["period_end", "total_revenue", "gross_margin_pct", "operating_margin_pct"]
    out = df[[c for c in cols if c in df.columns]].copy()
    out = out.dropna(subset=["total_revenue"])
    out["revenue_yoy_pct"] = out["total_revenue"].pct_change(4) * 100
    return out


def _scan_keywords(text: str) -> dict[str, object]:
    lower = text.lower()
    bull_hits = [kw for kw in BULLISH_ORDER_KEYWORDS if kw in lower]
    bear_hits = [kw for kw in BEARISH_ORDER_KEYWORDS if kw in lower]
    score = len(bull_hits) - len(bear_hits)
    if score > 2:
        signal = "Bullish"
    elif score < -1:
        signal = "Bearish"
    else:
        signal = "Neutral"
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

def _relative_perf_chart(prices: pd.DataFrame, tickers: list[str], days: int) -> go.Figure:
    cutoff = prices.index[-1] - pd.Timedelta(days=days)
    sub = prices[prices.index >= cutoff]

    fig = go.Figure()
    all_tickers = tickers + [BENCHMARK]
    for tk in all_tickers:
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
    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        yaxis_title="Indexed (start = 100)",
        xaxis_title="",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        xaxis=dict(gridcolor="#1e2530"),
        yaxis=dict(gridcolor="#1e2530"),
    )
    return fig


def _capex_chart(capex_data: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for tk, df in capex_data.items():
        if df.empty:
            continue
        fig.add_trace(go.Bar(
            x=df["period_end"].dt.strftime("%Y-Q%q") if hasattr(df["period_end"].dt, "quarter") else df["period_end"].astype(str),
            y=df["capex_abs"] / 1e9,
            name=f"{tk} ({HYPERSCALERS.get(tk, tk)})",
            hovertemplate=(
                f"<b>{tk}</b><br>"
                "Quarter: %{x}<br>"
                "Capex: $%{y:.1f}B<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        yaxis_title="Capex (USD Billions)",
        xaxis_title="Quarter",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        xaxis=dict(gridcolor="#1e2530"),
        yaxis=dict(gridcolor="#1e2530"),
    )
    return fig


def _nvda_revenue_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x = df["period_end"].dt.to_period("Q").astype(str)
    fig.add_trace(go.Bar(
        x=x,
        y=df["total_revenue"] / 1e9,
        name="Revenue ($B)",
        marker_color="#76b900",
        hovertemplate="<b>NVDA Revenue</b><br>%{x}<br>$%{y:.1f}B<extra></extra>",
    ))
    if "revenue_yoy_pct" in df.columns:
        valid = df["revenue_yoy_pct"].notna()
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
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        yaxis=dict(title="Revenue ($B)", gridcolor="#1e2530"),
        yaxis2=dict(title="YoY Growth (%)", overlaying="y", side="right", showgrid=False),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        xaxis=dict(gridcolor="#1e2530"),
    )
    return fig


def _capex_growth_chart(capex_data: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for tk, df in capex_data.items():
        sub = df.dropna(subset=["capex_yoy_pct"])
        if sub.empty:
            continue
        x_labels = sub["period_end"].dt.to_period("Q").astype(str) if hasattr(sub["period_end"].dt, "quarter") else sub["period_end"].astype(str)
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=sub["capex_yoy_pct"],
            name=tk,
            mode="lines+markers",
            hovertemplate=f"<b>{tk}</b><br>%{{x}}<br>YoY: %{{y:.1f}}%<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="#888888", line_width=1)
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        yaxis_title="Capex YoY Growth (%)",
        xaxis_title="",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        xaxis=dict(gridcolor="#1e2530"),
        yaxis=dict(gridcolor="#1e2530"),
    )
    return fig


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title("Optical Communications Sector Tracker")
st.caption("Leading-indicator view: hyperscaler capex + sector momentum. Data via yfinance (live).")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    perf_window = st.selectbox("Performance window", [30, 60, 90, 180, 365], index=2, format_func=lambda d: f"{d}D")
    custom_tickers_raw = st.text_input(
        "Add optical tickers (comma-separated)",
        placeholder="e.g. IIVI, FNSR",
    )

# Merge custom tickers
optical_tickers = list(OPTICAL_STOCKS.keys())
if custom_tickers_raw.strip():
    for t in custom_tickers_raw.upper().split(","):
        t = t.strip()
        if t and t not in optical_tickers:
            optical_tickers.append(t)
            OPTICAL_STOCKS[t] = t  # use ticker as label

all_price_tickers = tuple(sorted(set(optical_tickers + [BENCHMARK])))

# Load price data
with st.spinner("Loading price data..."):
    prices = _load_prices(all_price_tickers, days=perf_window + 30)

prices.index = pd.to_datetime(prices.index)

# ── Section 1: Relative Performance ────────────────────────────────────────
st.subheader("Relative Price Performance")
st.caption(f"Optical stocks vs SPY — last {perf_window} days (indexed to 100 at start)")

if not prices.empty:
    fig_perf = _relative_perf_chart(prices, optical_tickers, perf_window)
    st.plotly_chart(fig_perf, use_container_width=True)
else:
    st.warning("Price data unavailable. Check network connection.")

# ── Section 2: Momentum Ranking ────────────────────────────────────────────
st.subheader("Momentum Ranking")

mom_df = _momentum_table(prices, optical_tickers)
if not mom_df.empty:
    # Add company names
    mom_df.insert(1, "Company", mom_df["Ticker"].map(lambda t: OPTICAL_STOCKS.get(t, t)))

    def _pct_fmt(v):
        if pd.isna(v):
            return "—"
        return f"+{v:.1f}%" if v > 0 else f"{v:.1f}%"

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

# ── Section 3: Hyperscaler Capex Tracker ───────────────────────────────────
st.subheader("Hyperscaler Capex Tracker")
st.caption(
    "Quarterly capex for META / GOOGL / AMZN / MSFT. "
    "Accelerating hyperscaler capex is the #1 leading indicator for optical transceiver demand "
    "(AI cluster interconnects). Typical lead time: 2–4 quarters."
)

with st.spinner("Loading hyperscaler financials (yfinance)..."):
    capex_data: dict[str, pd.DataFrame] = {}
    for tk in HYPERSCALERS:
        capex_data[tk] = _load_capex(tk)

valid_capex = {tk: df for tk, df in capex_data.items() if not df.empty}

if valid_capex:
    tab_abs, tab_yoy = st.tabs(["Quarterly Capex (Absolute)", "YoY Growth Rate"])

    with tab_abs:
        st.plotly_chart(_capex_chart(valid_capex), use_container_width=True)

    with tab_yoy:
        st.plotly_chart(_capex_growth_chart(valid_capex), use_container_width=True)
        st.caption("YoY growth > 20% sustained = strong optical demand catalyst. Deceleration = risk of order pushouts.")

    # Latest capex summary table
    rows = []
    for tk, df in valid_capex.items():
        if df.empty:
            continue
        latest = df.iloc[-1]
        prev_yr = df.iloc[-5] if len(df) >= 5 else None
        yoy = latest["capex_yoy_pct"] if not pd.isna(latest.get("capex_yoy_pct", float("nan"))) else None
        rows.append({
            "Company":       f"{tk} ({HYPERSCALERS[tk]})",
            "Latest Quarter": str(latest["period_end"])[:7],
            "Capex ($B)":    f"${latest['capex_abs'] / 1e9:.1f}B",
            "YoY Growth":    f"+{yoy:.0f}%" if yoy and yoy > 0 else (f"{yoy:.0f}%" if yoy else "—"),
            "Signal":        _signal(yoy, threshold_bull=15, threshold_bear=-5),
        })

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Signal": st.column_config.TextColumn("Capex Signal"),
            },
        )
else:
    st.warning("Could not load hyperscaler capex data. yfinance may be rate-limited; try again shortly.")

# ── Section 4: NVIDIA GPU Proxy ────────────────────────────────────────────
st.subheader("NVIDIA GPU Shipment Proxy")
st.caption(
    "NVDA quarterly revenue growth is a leading proxy for AI GPU demand. "
    "GPU shipments determine data center interconnect scale — more GPUs = more 800G/1.6T transceivers needed. "
    "Typical optical order lag: 1–3 quarters after NVDA revenue acceleration."
)

with st.spinner("Loading NVDA financials..."):
    nvda_df = _load_nvda_quarterly()

nvda_price_tickers = tuple(sorted({NVIDIA_TICKER, BENCHMARK}))
with st.spinner("Loading NVDA price..."):
    nvda_prices = _load_prices(nvda_price_tickers, days=perf_window + 30)
nvda_prices.index = pd.to_datetime(nvda_prices.index)

col_chart, col_perf = st.columns([3, 2])
with col_chart:
    if not nvda_df.empty:
        st.plotly_chart(_nvda_revenue_chart(nvda_df), use_container_width=True)
    else:
        st.info("NVDA quarterly data unavailable.")

with col_perf:
    st.markdown("**NVDA vs SPY Performance**")
    if not nvda_prices.empty and NVIDIA_TICKER in nvda_prices.columns:
        fig_nvda = _relative_perf_chart(nvda_prices, [NVIDIA_TICKER], perf_window)
        fig_nvda.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_nvda, use_container_width=True)
    else:
        st.info("NVDA price data unavailable.")

# Latest NVDA revenue signal
if not nvda_df.empty:
    latest_nvda = nvda_df.iloc[-1]
    yoy_nvda = latest_nvda.get("revenue_yoy_pct")
    nvda_sig = _signal(yoy_nvda, threshold_bull=30, threshold_bear=0)
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Revenue", f"${latest_nvda['total_revenue'] / 1e9:.1f}B" if pd.notna(latest_nvda['total_revenue']) else "—")
    col2.metric("YoY Growth", f"{yoy_nvda:.0f}%" if pd.notna(yoy_nvda) else "—")
    col3.markdown(
        f"<div style='padding:6px; border-radius:5px; background:{SIGNAL_COLORS[nvda_sig]}22; "
        f"border:1px solid {SIGNAL_COLORS[nvda_sig]}; color:{SIGNAL_COLORS[nvda_sig]}; "
        f"font-weight:600; text-align:center; margin-top:8px;'>{nvda_sig} (GPU Proxy)</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Section 5: Data Center Construction Proxy ──────────────────────────────
st.subheader("Data Center Construction Proxy")
st.caption(
    "Data center REITs (EQIX, DLR, IRM, AMT) signal construction pipeline. "
    "When REITs outperform and raise guidance, new data centers are being built — "
    "optical module procurement follows 6–12 months later."
)

dc_reit_tickers = tuple(sorted(set(list(DC_REITS.keys()) + [BENCHMARK])))
with st.spinner("Loading DC REIT prices..."):
    reit_prices = _load_prices(dc_reit_tickers, days=perf_window + 30)
reit_prices.index = pd.to_datetime(reit_prices.index)

if not reit_prices.empty:
    fig_reit = _relative_perf_chart(reit_prices, list(DC_REITS.keys()), perf_window)
    st.plotly_chart(fig_reit, use_container_width=True)

    reit_mom = _momentum_table(reit_prices, list(DC_REITS.keys()))
    if not reit_mom.empty:
        reit_mom.insert(1, "Company", reit_mom["Ticker"].map(lambda t: DC_REITS.get(t, t)))
        display_reit = reit_mom.copy()
        for col in ["1W", "1M", "3M", "6M", "Score"]:
            if col in display_reit.columns:
                display_reit[col] = display_reit[col].apply(
                    lambda v: "—" if pd.isna(v) else (f"+{v:.1f}%" if v > 0 else f"{v:.1f}%")
                )
        st.dataframe(display_reit, use_container_width=True, hide_index=True)
else:
    st.info("DC REIT price data unavailable.")

st.divider()

# ── Section 6: 800G / 1.6T Order Keyword Scanner ──────────────────────────
st.subheader("800G / 1.6T Order Keyword Scanner")
st.caption(
    "Paste earnings call transcript or press release text. "
    "The scanner counts bullish (demand acceleration, design wins) vs bearish (pushouts, inventory) signals "
    "specific to high-speed optical transceiver orders."
)

transcript_text = st.text_area(
    "Paste earnings call / press release text here",
    height=160,
    placeholder="Paste transcript text from CIEN, COHR, LITE, AAOI or hyperscaler earnings calls...",
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

# ── Section 7: Coherent Transceiver Pricing Proxy ─────────────────────────
st.subheader("Coherent Transceiver Pricing Proxy")
st.caption(
    "COHR (Coherent Corp) gross margin trend is a proxy for transceiver ASP environment. "
    "Rising gross margin = pricing firming / mix shift to higher-speed products (800G+). "
    "Margin compression = pricing pressure or inventory overhang."
)

with st.spinner("Loading COHR financials..."):
    cohr_df = fetch_quarterly_metrics("COHR", num_quarters=10)

if not cohr_df.empty and "gross_margin_pct" in cohr_df.columns:
    cohr_plot = cohr_df.dropna(subset=["gross_margin_pct"]).copy()
    x_cohr = cohr_plot["period_end"].dt.to_period("Q").astype(str)

    fig_cohr = go.Figure()
    fig_cohr.add_trace(go.Scatter(
        x=x_cohr,
        y=cohr_plot["gross_margin_pct"],
        name="COHR Gross Margin %",
        mode="lines+markers",
        line=dict(color="#3498db", width=2),
        fill="tozeroy",
        fillcolor="rgba(52,152,219,0.12)",
        hovertemplate="<b>COHR GM%</b><br>%{x}<br>%{y:.1f}%<extra></extra>",
    ))
    if "operating_margin_pct" in cohr_plot.columns:
        fig_cohr.add_trace(go.Scatter(
            x=x_cohr,
            y=cohr_plot["operating_margin_pct"],
            name="Operating Margin %",
            mode="lines+markers",
            line=dict(color="#e67e22", width=1.5, dash="dot"),
            hovertemplate="<b>COHR OP%</b><br>%{x}<br>%{y:.1f}%<extra></extra>",
        ))
    fig_cohr.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        yaxis=dict(title="Margin (%)", gridcolor="#1e2530"),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        xaxis=dict(gridcolor="#1e2530"),
    )
    st.plotly_chart(fig_cohr, use_container_width=True)

    latest_cohr = cohr_plot.iloc[-1]
    prev_cohr = cohr_plot.iloc[-5] if len(cohr_plot) >= 5 else None
    gm_now = latest_cohr["gross_margin_pct"]
    gm_delta = (gm_now - prev_cohr["gross_margin_pct"]) if prev_cohr is not None and pd.notna(prev_cohr["gross_margin_pct"]) else None
    pricing_sig = _signal(gm_delta, threshold_bull=1.5, threshold_bear=-1.5)

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Gross Margin", f"{gm_now:.1f}%" if pd.notna(gm_now) else "—")
    c2.metric("YoY Change (pp)", f"{gm_delta:+.1f}pp" if gm_delta is not None else "—")
    c3.markdown(
        f"<div style='padding:6px; border-radius:5px; background:{SIGNAL_COLORS[pricing_sig]}22; "
        f"border:1px solid {SIGNAL_COLORS[pricing_sig]}; color:{SIGNAL_COLORS[pricing_sig]}; "
        f"font-weight:600; text-align:center; margin-top:8px;'>{pricing_sig} (Pricing Proxy)</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("COHR financial data unavailable.")

st.divider()

# ── Section 8: Signal Summary ───────────────────────────────────────────────
st.subheader("Signal Summary")
st.caption("Aggregated read from all leading indicators: momentum, hyperscaler capex, GPU proxy, DC REITs, and transceiver pricing.")

signal_rows = []

# Momentum signals from optical stocks
if not mom_df.empty and "Score" in mom_df.columns:
    for _, r in mom_df.iterrows():
        sig = _signal(r.get("Score"))
        signal_rows.append({
            "Source": f"Momentum — {r['Ticker']}",
            "Signal": sig,
        })

# Capex signals
for tk, df in valid_capex.items():
    if df.empty:
        continue
    yoy = df.iloc[-1].get("capex_yoy_pct")
    if pd.notna(yoy):
        sig = _signal(yoy, threshold_bull=15, threshold_bear=-5)
        signal_rows.append({
            "Source": f"Hyperscaler Capex YoY — {tk}",
            "Signal": sig,
        })

# NVDA GPU proxy signal
if not nvda_df.empty:
    yoy_nvda = nvda_df.iloc[-1].get("revenue_yoy_pct")
    if pd.notna(yoy_nvda):
        signal_rows.append({
            "Source": "NVDA Revenue YoY (GPU Proxy)",
            "Signal": _signal(yoy_nvda, threshold_bull=30, threshold_bear=0),
        })

# DC REIT momentum signals
if not reit_prices.empty:
    reit_mom_sig = _momentum_table(reit_prices, list(DC_REITS.keys()))
    if not reit_mom_sig.empty and "Score" in reit_mom_sig.columns:
        avg_reit_score = reit_mom_sig["Score"].mean()
        signal_rows.append({
            "Source": "DC REIT Avg Momentum (Construction Proxy)",
            "Signal": _signal(avg_reit_score, threshold_bull=5, threshold_bear=-5),
        })

# COHR pricing proxy signal
if not cohr_df.empty and "gross_margin_pct" in cohr_df.columns:
    cohr_gm = cohr_df.dropna(subset=["gross_margin_pct"])
    if len(cohr_gm) >= 5:
        gm_delta_sig = cohr_gm.iloc[-1]["gross_margin_pct"] - cohr_gm.iloc[-5]["gross_margin_pct"]
        signal_rows.append({
            "Source": "COHR Gross Margin YoY (Transceiver Pricing Proxy)",
            "Signal": _signal(gm_delta_sig, threshold_bull=1.5, threshold_bear=-1.5),
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

    overall_score = (bull - bear) / total * 100 if total > 0 else 0
    if overall_score > 20:
        overall_label, overall_color = "Sector Bullish", SIGNAL_COLORS["Bullish"]
    elif overall_score < -20:
        overall_label, overall_color = "Sector Bearish", SIGNAL_COLORS["Bearish"]
    else:
        overall_label, overall_color = "Sector Neutral", SIGNAL_COLORS["Neutral"]

    col4.markdown(
        f"<div style='padding:8px; border-radius:6px; background:{overall_color}22; "
        f"border:1px solid {overall_color}; color:{overall_color}; font-weight:600; text-align:center;'>"
        f"{overall_label}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("View all signals"):
        st.dataframe(sig_df, use_container_width=True, hide_index=True)
else:
    st.info("No signals available — check that price and capex data loaded correctly.")

# ── Footer note ─────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Key leading indicators to watch:** "
    "Hyperscaler capex acceleration (2–4Q lead) · "
    "800G/1.6T transceiver order announcements · "
    "NVDA revenue YoY acceleration (1–3Q lead) · "
    "DC REIT outperformance (6–12 month lead) · "
    "COHR gross margin inflection (pricing signal) · "
    "Book-to-bill ratio > 1.0"
)
