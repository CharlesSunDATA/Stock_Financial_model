"""Sector Tracker.

Unified theme and sector dashboard for price momentum, relative strength,
leading proxies, capex, and margin signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from utils.data_loader import fetch_quarterly_metrics


BENCHMARK = "SPY"
MOMENTUM_WINDOWS = {"1W": 5, "1M": 21, "3M": 63, "6M": 126}

HYPERSCALERS = {
    "META": "Meta",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "MSFT": "Microsoft",
}

SIGNAL_COLORS = {
    "Bullish": "#2ecc71",
    "Neutral": "#f1c40f",
    "Bearish": "#e74c3c",
}


@dataclass(frozen=True)
class SectorConfig:
    name: str
    description: str
    tickers: dict[str, str]
    proxies: dict[str, str]


SECTORS: dict[str, SectorConfig] = {
    "Optical Communications": SectorConfig(
        name="Optical Communications",
        description="AI data-center interconnects, optical modules, coherent components, and high-speed networking demand.",
        tickers={
            "CIEN": "Ciena",
            "COHR": "Coherent",
            "LITE": "Lumentum",
            "AAOI": "Applied Optoelectronics",
            "VIAV": "VIAVI Solutions",
            "MTSI": "MACOM Technology",
        },
        proxies={"NVDA": "AI GPU demand", "ANET": "Data-center switching", "EQIX": "Data-center buildout"},
    ),
    "Memory": SectorConfig(
        name="Memory",
        description="DRAM, NAND, HBM, storage, and AI memory-cycle signals.",
        tickers={"MU": "Micron", "WDC": "Western Digital", "STX": "Seagate", "MRVL": "Marvell"},
        proxies={"NVDA": "AI GPU demand", "AMD": "AI accelerator demand", "LRCX": "Memory equipment cycle"},
    ),
    "AI Infrastructure": SectorConfig(
        name="AI Infrastructure",
        description="Accelerators, custom silicon, servers, networking silicon, and AI compute infrastructure.",
        tickers={"NVDA": "NVIDIA", "AMD": "AMD", "AVGO": "Broadcom", "MRVL": "Marvell", "SMCI": "Super Micro", "ARM": "Arm"},
        proxies={"MSFT": "Cloud AI capex", "META": "AI cluster capex", "ANET": "AI networking"},
    ),
    "Semiconductor Equipment": SectorConfig(
        name="Semiconductor Equipment",
        description="Wafer fab equipment and process-control suppliers tied to semiconductor capacity cycles.",
        tickers={"AMAT": "Applied Materials", "LRCX": "Lam Research", "KLAC": "KLA", "ASML": "ASML", "TER": "Teradyne"},
        proxies={"TSM": "Foundry capex", "MU": "Memory capex", "NVDA": "AI demand pull"},
    ),
    "Networking Equipment": SectorConfig(
        name="Networking Equipment",
        description="Switching, routing, observability, test equipment, and enterprise/data-center network infrastructure.",
        tickers={"ANET": "Arista Networks", "CSCO": "Cisco", "JNPR": "Juniper", "KEYS": "Keysight"},
        proxies={"NVDA": "AI server demand", "MSFT": "Cloud buildout", "CIEN": "Optical transport"},
    ),
    "Cloud / Hyperscalers": SectorConfig(
        name="Cloud / Hyperscalers",
        description="Cloud platforms and large AI-capex buyers that drive demand for compute, networking, and storage.",
        tickers={"MSFT": "Microsoft", "AMZN": "Amazon", "GOOGL": "Alphabet", "META": "Meta", "ORCL": "Oracle"},
        proxies={"NVDA": "AI compute supplier", "ANET": "Networking supplier", "MU": "Memory supplier"},
    ),
    "Cybersecurity": SectorConfig(
        name="Cybersecurity",
        description="Security software platforms, cloud security, endpoint, identity, and network security.",
        tickers={"CRWD": "CrowdStrike", "PANW": "Palo Alto Networks", "ZS": "Zscaler", "FTNT": "Fortinet", "OKTA": "Okta"},
        proxies={"MSFT": "Platform security", "NOW": "Enterprise software demand"},
    ),
    "Financials": SectorConfig(
        name="Financials",
        description="Banks, cards, brokers, and capital-market activity sensitive to credit, rates, and liquidity.",
        tickers={"JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs", "MS": "Morgan Stanley", "V": "Visa", "MA": "Mastercard"},
        proxies={"SPY": "Risk appetite", "TLT": "Long rates", "KRE": "Regional-bank stress"},
    ),
}


def _dark_layout(height: int = 360) -> dict:
    return dict(
        height=height,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor="#1e2530"),
        yaxis=dict(gridcolor="#1e2530"),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _load_prices(tickers: tuple[str, ...], days: int) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=days + 35)
    try:
        raw = yf.download(list(tickers), start=str(start), end=str(end), auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    closes = closes.dropna(how="all")
    closes.index = pd.to_datetime(closes.index)
    return closes


@st.cache_data(ttl=3600, show_spinner=False)
def _load_quarterly(ticker: str, quarters: int = 10) -> pd.DataFrame:
    df = fetch_quarterly_metrics(ticker, num_quarters=quarters)
    if df.empty or "total_revenue" not in df.columns:
        return pd.DataFrame()
    cols = ["period_end", "total_revenue", "gross_margin_pct", "operating_margin_pct", "capex"]
    out = df[[c for c in cols if c in df.columns]].copy()
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce")
    out = out.dropna(subset=["period_end"])
    if "total_revenue" in out:
        out["revenue_yoy_pct"] = out["total_revenue"].pct_change(4) * 100
    if "capex" in out:
        out["capex_abs"] = out["capex"].abs()
        out["capex_yoy_pct"] = out["capex_abs"].pct_change(4) * 100
    return out


def _pct(v) -> str:
    if pd.isna(v):
        return "-"
    return f"{float(v):+.1f}%"


def _signal(value, bull: float = 5.0, bear: float = -5.0) -> str:
    if pd.isna(value):
        return "Neutral"
    value = float(value)
    if value > bull:
        return "Bullish"
    if value < bear:
        return "Bearish"
    return "Neutral"


def _momentum_table(prices: pd.DataFrame, tickers: list[str], names: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        series = prices[ticker].dropna()
        if len(series) < 2:
            continue
        latest = float(series.iloc[-1])
        row: dict[str, object] = {"Ticker": ticker, "Company": names.get(ticker, ticker), "Price": latest}
        for label, window in MOMENTUM_WINDOWS.items():
            row[label] = (latest / float(series.iloc[-window - 1]) - 1) * 100 if len(series) > window else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Score"] = df[["1M", "3M", "6M"]].mean(axis=1)
    return df.sort_values("Score", ascending=False, na_position="last").reset_index(drop=True)


def _sector_summary(prices: pd.DataFrame, sectors: dict[str, SectorConfig]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, config in sectors.items():
        tickers = list(config.tickers)
        mom = _momentum_table(prices, tickers, config.tickers)
        if mom.empty:
            rows.append({"Sector": name, "Tickers": 0, "1M": np.nan, "3M": np.nan, "6M": np.nan, "Positive 1M": np.nan, "Leader": "-", "Laggard": "-"})
            continue
        rows.append({
            "Sector": name,
            "Tickers": len(mom),
            "1M": mom["1M"].mean(),
            "3M": mom["3M"].mean(),
            "6M": mom["6M"].mean(),
            "Positive 1M": (mom["1M"] > 0).mean() * 100,
            "Leader": str(mom.iloc[0]["Ticker"]),
            "Laggard": str(mom.iloc[-1]["Ticker"]),
        })
    return pd.DataFrame(rows).sort_values("1M", ascending=False, na_position="last")


def _relative_chart(prices: pd.DataFrame, tickers: list[str], days: int) -> go.Figure:
    fig = go.Figure()
    if prices.empty:
        return fig
    cutoff = prices.index.max() - pd.Timedelta(days=days)
    sub = prices[prices.index >= cutoff]
    for ticker in tickers + [BENCHMARK]:
        if ticker not in sub.columns:
            continue
        series = sub[ticker].dropna()
        if series.empty:
            continue
        norm = series / series.iloc[0] * 100
        is_benchmark = ticker == BENCHMARK
        fig.add_trace(go.Scatter(
            x=norm.index,
            y=norm.values,
            name=ticker,
            line=dict(width=3 if is_benchmark else 1.6, dash="dash" if is_benchmark else "solid", color="#9ca3af" if is_benchmark else None),
            hovertemplate=f"<b>{ticker}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}<extra></extra>",
        ))
    fig.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.35)", line_width=1)
    fig.update_layout(**_dark_layout(380), yaxis_title="Indexed (start = 100)")
    return fig


def _summary_bar(summary: pd.DataFrame) -> go.Figure:
    df = summary.dropna(subset=["1M"]).sort_values("1M")
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["1M"]]
    fig = go.Figure(go.Bar(
        x=df["1M"],
        y=df["Sector"],
        orientation="h",
        marker_color=colors,
        text=[_pct(v) for v in df["1M"]],
        textposition="outside",
        cliponaxis=False,
    ))
    max_x = max(1.0, float(df["1M"].max()) * 1.35) if not df.empty else 1
    min_x = min(0.0, float(df["1M"].min()) * 1.25) if not df.empty else 0
    layout = _dark_layout(max(330, len(df) * 38))
    layout["margin"] = dict(l=10, r=120, t=20, b=0)
    layout["showlegend"] = False
    layout["xaxis"] = dict(range=[min_x, max_x], ticksuffix="%", gridcolor="#1e2530")
    fig.update_layout(**layout)
    return fig


def _revenue_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig
    x = df["period_end"].dt.to_period("Q").astype(str)
    fig.add_trace(go.Bar(
        x=x,
        y=df["total_revenue"] / 1e9,
        name="Revenue ($B)",
        marker_color="#38bdf8",
        hovertemplate=f"<b>{ticker}</b><br>%{{x}}<br>Revenue: $%{{y:.1f}}B<extra></extra>",
    ))
    if "revenue_yoy_pct" in df:
        valid = df["revenue_yoy_pct"].notna()
        fig.add_trace(go.Scatter(
            x=x[valid],
            y=df.loc[valid, "revenue_yoy_pct"],
            name="Revenue YoY %",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#facc15", width=2),
        ))
    layout = _dark_layout(300)
    layout["yaxis"] = dict(title="Revenue ($B)", gridcolor="#1e2530")
    layout["yaxis2"] = dict(title="YoY %", overlaying="y", side="right", showgrid=False)
    fig.update_layout(**layout)
    return fig


def _capex_chart(capex_data: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for ticker, df in capex_data.items():
        if df.empty or "capex_abs" not in df:
            continue
        fig.add_trace(go.Bar(
            x=df["period_end"].dt.to_period("Q").astype(str),
            y=df["capex_abs"] / 1e9,
            name=ticker,
            hovertemplate=f"<b>{ticker}</b><br>%{{x}}<br>Capex: $%{{y:.1f}}B<extra></extra>",
        ))
    fig.update_layout(**_dark_layout(330), barmode="group", yaxis_title="Capex ($B)")
    return fig


def _display_momentum(df: pd.DataFrame) -> None:
    show = df.copy()
    for col in ["1W", "1M", "3M", "6M", "Score"]:
        if col in show:
            show[col] = show[col].map(_pct)
    if "Price" in show:
        show["Price"] = show["Price"].map(lambda v: f"${float(v):,.2f}" if pd.notna(v) else "-")
    st.dataframe(show, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Sector Tracker")
    st.caption("Unified sector and theme tracker for relative strength, momentum, leading proxies, capex, and margin signals.")

    with st.sidebar:
        st.header("Sector Tracker")
        selected_sector = st.selectbox("Sector", list(SECTORS.keys()))
        perf_window = st.selectbox("Performance window", [30, 60, 90, 180, 365], index=2, format_func=lambda d: f"{d}D")
        custom_raw = st.text_input("Add tickers", placeholder="Comma-separated tickers")
        show_all = st.checkbox("Show sector comparison", value=True)

    all_tickers: set[str] = {BENCHMARK}
    for sector in SECTORS.values():
        all_tickers.update(sector.tickers)
        all_tickers.update(sector.proxies)

    config = SECTORS[selected_sector]
    selected_tickers = list(config.tickers)
    custom_names: dict[str, str] = {}
    if custom_raw.strip():
        for ticker in custom_raw.upper().split(","):
            ticker = ticker.strip()
            if ticker and ticker not in selected_tickers:
                selected_tickers.append(ticker)
                custom_names[ticker] = ticker
                all_tickers.add(ticker)

    with st.spinner("Loading price data..."):
        prices = _load_prices(tuple(sorted(all_tickers)), perf_window + 140)

    if prices.empty:
        st.warning("Price data unavailable. Check network connection or try again later.")
        return

    if show_all:
        st.subheader("Sector Comparison")
        summary = _sector_summary(prices, SECTORS)
        left, right = st.columns([1.2, 1])
        with left:
            st.plotly_chart(_summary_bar(summary), use_container_width=True)
        with right:
            display = summary.copy()
            for col in ["1M", "3M", "6M", "Positive 1M"]:
                display[col] = display[col].map(_pct)
            st.dataframe(display, use_container_width=True, hide_index=True)
        st.divider()

    st.subheader(selected_sector)
    st.caption(config.description)

    c1, c2, c3, c4 = st.columns(4)
    sector_momentum = _momentum_table(prices, selected_tickers, {**config.tickers, **custom_names})
    if not sector_momentum.empty:
        c1.metric("Tracked stocks", f"{len(sector_momentum):,}")
        c2.metric("Avg 1M", _pct(sector_momentum["1M"].mean()))
        c3.metric("Avg 3M", _pct(sector_momentum["3M"].mean()))
        c4.metric("Leader", str(sector_momentum.iloc[0]["Ticker"]))

    st.plotly_chart(_relative_chart(prices, selected_tickers, perf_window), use_container_width=True)

    st.subheader("Momentum Ranking")
    if sector_momentum.empty:
        st.info("No momentum data available for this sector.")
    else:
        _display_momentum(sector_momentum)

    st.divider()

    st.subheader("Leading Proxies")
    st.caption("Proxy tickers are sector-specific indicators that often move before reported fundamentals.")
    proxy_tickers = list(config.proxies)
    proxy_momentum = _momentum_table(prices, proxy_tickers, config.proxies)
    if proxy_momentum.empty:
        st.info("No proxy price data available.")
    else:
        _display_momentum(proxy_momentum)

    revenue_cols = st.columns(min(3, max(1, len(proxy_tickers))))
    for idx, ticker in enumerate(proxy_tickers[:3]):
        with revenue_cols[idx % len(revenue_cols)]:
            with st.spinner(f"Loading {ticker} quarterly data..."):
                qdf = _load_quarterly(ticker)
            st.markdown(f"**{ticker}: {config.proxies[ticker]}**")
            if qdf.empty:
                st.info("Quarterly data unavailable.")
            else:
                st.plotly_chart(_revenue_chart(qdf, ticker), use_container_width=True)

    if selected_sector in {"Optical Communications", "Memory", "Cloud / Hyperscalers"}:
        st.subheader("Hyperscaler Capex")
        with st.spinner("Loading hyperscaler capex..."):
            capex_data = {ticker: _load_quarterly(ticker, quarters=12) for ticker in HYPERSCALERS}
        valid_capex = {ticker: df for ticker, df in capex_data.items() if not df.empty and "capex_abs" in df}
        if valid_capex:
            st.plotly_chart(_capex_chart(valid_capex), use_container_width=True)
        else:
            st.info("Hyperscaler capex data unavailable.")


if __name__ == "__main__":
    main()
