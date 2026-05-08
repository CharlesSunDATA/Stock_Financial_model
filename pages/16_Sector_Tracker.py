"""Sector Tracker.

Single-page sector rotation board with sector and ticker rankings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

BENCHMARK = "SPY"
MOMENTUM_WINDOWS = {"1W": 5, "1M": 21, "3M": 63, "6M": 126}


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


def _pct(v) -> str:
    if pd.isna(v):
        return "-"
    return f"{float(v):+.1f}%"


def _score_label(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.1f}"


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
    df["Score"] = (
        df["1M"].fillna(0) * 0.45
        + df["3M"].fillna(0) * 0.35
        + df["6M"].fillna(0) * 0.20
    )
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
            "Score": mom["Score"].mean(),
            "Positive 1M": (mom["1M"] > 0).mean() * 100,
            "Leader": str(mom.iloc[0]["Ticker"]),
            "Laggard": str(mom.iloc[-1]["Ticker"]),
        })
    return pd.DataFrame(rows).sort_values("Score", ascending=False, na_position="last")


def _summary_bar(summary: pd.DataFrame) -> go.Figure:
    df = summary.dropna(subset=["Score"]).sort_values("Score")
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["Score"]]
    fig = go.Figure(go.Bar(
        x=df["Score"],
        y=df["Sector"],
        orientation="h",
        marker_color=colors,
        text=[_score_label(v) for v in df["Score"]],
        textposition="outside",
        cliponaxis=False,
    ))
    max_x = max(1.0, float(df["Score"].max()) * 1.35) if not df.empty else 1
    min_x = min(0.0, float(df["Score"].min()) * 1.25) if not df.empty else 0
    layout = _dark_layout(max(330, len(df) * 38))
    layout["margin"] = dict(l=10, r=120, t=20, b=0)
    layout["showlegend"] = False
    layout["xaxis"] = dict(range=[min_x, max_x], title="Composite score", gridcolor="#1e2530")
    fig.update_layout(**layout)
    return fig


def _all_stock_rankings(prices: pd.DataFrame, sectors: dict[str, SectorConfig]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sector_name, config in sectors.items():
        mom = _momentum_table(prices, list(config.tickers), config.tickers)
        if mom.empty:
            continue
        mom.insert(0, "Sector", sector_name)
        frames.append(mom)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["Rank"] = out["Score"].rank(method="first", ascending=False).astype(int)
    return out.sort_values("Score", ascending=False).reset_index(drop=True)


def _sector_cards(summary: pd.DataFrame) -> None:
    cards = []
    for idx, row in summary.reset_index(drop=True).iterrows():
        score = row.get("Score")
        border = "#22c55e" if pd.notna(score) and float(score) >= 0 else "#ef4444"
        cards.append(
            f"<div class='sector-card' style='border-color:{border};'>"
            f"<div class='sector-rank'>#{idx + 1}</div>"
            f"<div class='sector-name'>{row['Sector']}</div>"
            f"<div class='sector-score'>{_score_label(score)}</div>"
            f"<div class='sector-sub'>1M {_pct(row.get('1M'))} · 3M {_pct(row.get('3M'))}</div>"
            f"<div class='sector-sub'>Leader: <b>{row.get('Leader', '-')}</b></div>"
            f"</div>"
        )
    st.markdown(
        """
        <style>
          .sector-row {
            display: flex;
            gap: 12px;
            overflow-x: auto;
            padding: 4px 2px 14px 2px;
          }
          .sector-card {
            flex: 0 0 230px;
            min-height: 142px;
            border: 1px solid;
            border-radius: 8px;
            padding: 14px;
            background: rgba(255,255,255,0.035);
          }
          .sector-rank {
            color: rgba(255,255,255,0.62);
            font-size: 0.88rem;
            font-weight: 700;
          }
          .sector-name {
            color: white;
            font-size: 1.08rem;
            line-height: 1.2;
            font-weight: 750;
            min-height: 42px;
            margin-top: 5px;
          }
          .sector-score {
            color: white;
            font-size: 2rem;
            line-height: 1.1;
            font-weight: 800;
            margin-top: 8px;
          }
          .sector-sub {
            color: rgba(255,255,255,0.72);
            font-size: 0.84rem;
            margin-top: 6px;
            white-space: nowrap;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='sector-row'>{''.join(cards)}</div>", unsafe_allow_html=True)


def _format_ranking_table(df: pd.DataFrame) -> pd.DataFrame:
    show = df.copy()
    for col in ["1W", "1M", "3M", "6M"]:
        if col in show:
            show[col] = show[col].map(_pct)
    if "Score" in show:
        show["Score"] = show["Score"].map(_score_label)
    if "Price" in show:
        show["Price"] = show["Price"].map(lambda v: f"${float(v):,.2f}" if pd.notna(v) else "-")
    return show[["Rank", "Ticker", "Company", "Sector", "Score", "Price", "1W", "1M", "3M", "6M"]]


def main() -> None:
    st.title("Sector Tracker")
    st.caption("Single-page sector rotation board. Sectors are ranked left to right by recent composite momentum.")

    with st.sidebar:
        st.header("Sector Tracker")
        perf_window = st.selectbox("Performance window", [30, 60, 90, 180, 365], index=2, format_func=lambda d: f"{d}D")
        top_n = st.slider("Top stocks", min_value=10, max_value=80, value=30, step=5)

    all_tickers: set[str] = {BENCHMARK}
    for sector in SECTORS.values():
        all_tickers.update(sector.tickers)
        all_tickers.update(sector.proxies)

    with st.spinner("Loading price data..."):
        prices = _load_prices(tuple(sorted(all_tickers)), perf_window + 140)

    if prices.empty:
        st.warning("Price data unavailable. Check network connection or try again later.")
        return

    summary = _sector_summary(prices, SECTORS)
    rankings = _all_stock_rankings(prices, SECTORS)

    st.subheader("Sector Rotation")
    _sector_cards(summary)

    left, right = st.columns([1.05, 1])
    with left:
        st.plotly_chart(_summary_bar(summary), use_container_width=True)
    with right:
        display = summary.copy()
        for col in ["1M", "3M", "6M", "Positive 1M"]:
            display[col] = display[col].map(_pct)
        display["Score"] = display["Score"].map(_score_label)
        st.dataframe(
            display[["Sector", "Score", "1M", "3M", "6M", "Positive 1M", "Leader", "Laggard", "Tickers"]],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    st.subheader("Recommended Stocks")
    st.caption("Ranked by composite momentum score: 45% 1M, 35% 3M, 20% 6M.")
    if rankings.empty:
        st.info("No stock ranking data available.")
    else:
        st.dataframe(_format_ranking_table(rankings.head(top_n)), use_container_width=True, hide_index=True)

    st.subheader("Top Stocks by Sector")
    for sector_name in summary["Sector"].tolist():
        sector_rows = rankings[rankings["Sector"] == sector_name].head(5)
        if sector_rows.empty:
            continue
        with st.expander(sector_name, expanded=False):
            st.dataframe(_format_ranking_table(sector_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
