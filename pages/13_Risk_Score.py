"""
Stock Risk Score (Streamlit page)

Ranks investment risk from cash flow, leverage, valuation, technical trend,
EPS deterioration, and sector weakness.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.formatters import fmt_money as _fmt_money
from utils.formatters import fmt_pct as _fmt_pct
from utils.formatters import fmt_score as _fmt_score
from utils.risk_score import RISK_LABELS, RISK_WEIGHTS, RiskInputs
from utils.risk_score import available_watchlists, compute_risk_scores


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "quant_data.db"


def _stretch_kwargs(func) -> dict[str, object]:
    if "width" in inspect.signature(func).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


def _plotly_chart(fig: go.Figure) -> None:
    kwargs = _stretch_kwargs(st.plotly_chart)
    st.plotly_chart(fig, config={}, **kwargs)


def _risk_color(score: float) -> str:
    if score <= 30:
        return "#22c55e"
    if score <= 60:
        return "#f59e0b"
    return "#ef4444"


def _score_bar(df: pd.DataFrame, top_n: int) -> go.Figure:
    sub = df.dropna(subset=["risk_score"]).head(top_n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=sub["risk_score"],
            y=sub["ticker"],
            orientation="h",
            marker_color=[_risk_color(float(v)) for v in sub["risk_score"]],
            text=[_fmt_score(v) for v in sub["risk_score"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Risk Score: %{x:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(360, top_n * 28),
        margin=dict(t=20, b=10, l=80, r=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(showgrid=False),
    )
    return fig


def _factor_heatmap(df: pd.DataFrame, top_n: int) -> go.Figure:
    factors = list(RISK_WEIGHTS.keys())
    labels = [RISK_LABELS[f] for f in factors]
    sub = df.dropna(subset=["risk_score"]).head(top_n)
    z = sub[factors].apply(pd.to_numeric, errors="coerce").to_numpy()
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=sub["ticker"],
            colorscale=[
                [0.0, "#16a34a"],
                [0.45, "#facc15"],
                [0.7, "#f97316"],
                [1.0, "#dc2626"],
            ],
            zmin=0,
            zmax=100,
            colorbar=dict(title="Risk", tickmode="array", tickvals=[0, 30, 60, 100]),
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.0f}<extra></extra>",
            xgap=1,
            ygap=1,
        )
    )
    fig.update_layout(
        height=max(360, top_n * 28),
        margin=dict(t=20, b=10, l=80, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


@st.cache_data(ttl=60 * 10, show_spinner=False)
def _cached_scores(db_path: str, watchlist_name: str | None, min_price_rows: int) -> pd.DataFrame:
    return compute_risk_scores(
        Path(db_path),
        RiskInputs(watchlist_name=watchlist_name, min_price_rows=min_price_rows),
    )


def main() -> None:
    st.title("Stock Risk Score")
    st.caption("Investment risk ranking from cash flow, leverage, valuation, trend, EPS deterioration, and sector weakness.")

    if not DB_PATH.exists():
        st.error(f"Database not found: `{DB_PATH}`")
        return

    watchlists = available_watchlists(DB_PATH)

    with st.sidebar:
        st.header("Risk Ranking")
        st.caption(f"DB: `{DB_PATH}`")
        selected_watchlist = st.selectbox("Watchlist", ["All"] + watchlists, index=0)
        top_n = st.slider("Top N", min_value=10, max_value=100, value=30, step=5)
        min_price_rows = st.slider("Minimum price rows", min_value=200, max_value=252, value=200, step=10)
        min_score = st.slider("Minimum risk score", min_value=0, max_value=100, value=0, step=5)

        st.divider()
        if st.button("Refresh cache", **_stretch_kwargs(st.button)):
            st.cache_data.clear()
            st.rerun()

    watchlist_name = None if selected_watchlist == "All" else selected_watchlist
    with st.spinner("Computing risk scores..."):
        df = _cached_scores(str(DB_PATH), watchlist_name, min_price_rows)

    if df.empty:
        st.warning("No risk data. Check `fmp_watchlist`, `prices_eod`, `fundamental_data`, and `key_metrics_ttm`.")
        return

    df = df[df["risk_score"].fillna(-1) >= min_score].copy()
    if df.empty:
        st.warning("No stocks match the selected risk filter.")
        return

    sectors = sorted([x for x in df["sector"].dropna().unique().tolist() if str(x).strip()])
    with st.sidebar:
        sector = st.selectbox("Sector", ["All"] + sectors)
        risk_level = st.selectbox("Risk level", ["All", "Low Risk", "Medium Risk", "High Risk"])
    if sector != "All":
        df = df[df["sector"] == sector].copy()
    if risk_level != "All":
        df = df[df["risk_level"] == risk_level].copy()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Stocks ranked", f"{len(df):,}")
    with c2:
        st.metric("Average risk", _fmt_score(df["risk_score"].mean()))
    with c3:
        st.metric("High risk", f"{int((df['risk_score'] >= 61).sum()):,}")
    with c4:
        latest_date = df["price_date"].dropna().max() if "price_date" in df else None
        st.metric("Price date", latest_date or "-")

    st.divider()

    tab_rank, tab_factors, tab_details, tab_rules = st.tabs(
        ["Rankings", "Risk map", "Details", "Rules"]
    )

    with tab_rank:
        left, right = st.columns([1.1, 1])
        with left:
            _plotly_chart(_score_bar(df, top_n))
        with right:
            show = df.head(top_n)[["ticker", "risk_score", "risk_level", "risk_drivers", "company_name", "sector"]].copy()
            show["risk_score"] = show["risk_score"].apply(_fmt_score)
            show = show.rename(
                columns={
                    "ticker": "Ticker",
                    "risk_score": "Risk Score",
                    "risk_level": "Risk Level",
                    "risk_drivers": "Risk Drivers",
                    "company_name": "Company",
                    "sector": "Sector",
                }
            )
            st.dataframe(show, **_stretch_kwargs(st.dataframe), hide_index=True)

    with tab_factors:
        _plotly_chart(_factor_heatmap(df, min(top_n, 40)))

    with tab_details:
        detail_cols = [
            "ticker",
            "risk_score",
            "risk_level",
            "risk_drivers",
            "company_name",
            "sector",
            "cash_flow_risk",
            "debt_risk",
            "valuation_risk",
            "technical_risk",
            "fundamental_risk",
            "systemic_risk",
            "latest_fcf",
            "net_debt_to_ebitda",
            "ev_to_sales",
            "latest_price",
            "ma200",
            "latest_eps",
            "previous_eps",
            "ret_6m",
            "market_cap",
        ]
        show = df[[c for c in detail_cols if c in df.columns]].copy()
        for col in list(RISK_WEIGHTS) + ["risk_score"]:
            if col in show:
                show[col] = show[col].apply(_fmt_score)
        for col in ["latest_fcf", "market_cap"]:
            if col in show:
                show[col] = show[col].apply(_fmt_money)
        if "ret_6m" in show:
            show["ret_6m"] = show["ret_6m"].apply(_fmt_pct)
        show = show.rename(
            columns={
                "ticker": "Ticker",
                "risk_score": "Risk Score",
                "risk_level": "Risk Level",
                "risk_drivers": "Risk Drivers",
                "company_name": "Company",
                "sector": "Sector",
                "cash_flow_risk": "Cash Flow Risk",
                "debt_risk": "Debt Risk",
                "valuation_risk": "Valuation Risk",
                "technical_risk": "Technical Risk",
                "fundamental_risk": "Fundamental Risk",
                "systemic_risk": "Systemic Risk",
                "latest_fcf": "Latest FCF",
                "net_debt_to_ebitda": "Net Debt/EBITDA",
                "ev_to_sales": "EV/Sales",
                "latest_price": "Latest Price",
                "ma200": "200MA",
                "latest_eps": "Latest EPS",
                "previous_eps": "Previous EPS",
                "ret_6m": "6M Return",
                "market_cap": "Market Cap",
            }
        )
        st.dataframe(show, **_stretch_kwargs(st.dataframe), hide_index=True)
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="stock_risk_score.csv",
            mime="text/csv",
            **_stretch_kwargs(st.download_button),
        )

    with tab_rules:
        rules = pd.DataFrame(
            [
                {"Risk Source": "Negative FCF", "Risk Type": "Cash Flow Risk", "Rule": "100 risk points when latest FCF is negative."},
                {"Risk Source": "Net Debt/EBITDA > 2", "Risk Type": "Debt Risk", "Rule": "Risk scales from 0 at 2.0x to 100 at 5.0x or higher."},
                {"Risk Source": "High EV/Sales", "Risk Type": "Valuation Risk", "Rule": "Higher EV/Sales receives higher percentile-based risk."},
                {"Risk Source": "Price below 200MA", "Risk Type": "Technical Risk", "Rule": "100 risk points when latest price is below 200MA."},
                {"Risk Source": "EPS decline streak", "Risk Type": "Fundamental Risk", "Rule": "100 risk points for three-quarter EPS decline; 50 for latest-quarter decline."},
                {"Risk Source": "Weak sector", "Risk Type": "Systemic Risk", "Rule": "Lower sector six-month momentum receives higher risk."},
                {"Risk Source": "Final score", "Risk Type": "Risk Level", "Rule": "Low Risk: 0-30; Medium Risk: 31-60; High Risk: 61-100."},
            ]
        )
        st.dataframe(rules, **_stretch_kwargs(st.dataframe), hide_index=True)


if __name__ == "__main__":
    main()
