"""Industry ranking dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.formatters import fmt_money, fmt_pct, fmt_score
from utils.industry_screener import SCORE_LABELS, SCORE_WEIGHTS, available_watchlists, write_screening_tables, ScreenerInputs


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "quant_data.db"


def _fmt_score(value) -> str:
    return fmt_score(value, missing="-")


def _load_table(table: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    import sqlite3

    with sqlite3.connect(str(DB_PATH)) as conn:
        try:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)
        except Exception:
            return pd.DataFrame()


@st.cache_data(ttl=60 * 10, show_spinner=False)
def _load_materialized() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        _load_table("companies"),
        _load_table("financials"),
        _load_table("prices"),
        _load_table("scores"),
    )


def _score_bar(df: pd.DataFrame, top_n: int) -> go.Figure:
    sub = df.dropna(subset=["industry_score"]).head(top_n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=sub["industry_score"],
            y=sub["industry"],
            orientation="h",
            marker_color="#2f6f73",
            text=[_fmt_score(v) for v in sub["industry_score"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Industry Score: %{x:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(360, top_n * 30),
        margin=dict(t=20, b=10, l=190, r=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(128,128,128,0.18)"),
        yaxis=dict(showgrid=False),
    )
    return fig


def _stock_scores(companies: pd.DataFrame, financials: pd.DataFrame, prices: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    out = scores.copy()
    for part in (companies, financials, prices):
        if not part.empty:
            out = out.merge(part, on="ticker", how="left", suffixes=("", "_source"))
    return out


def main() -> None:
    st.title("Industry Ranking")
    st.caption("Industry and stock rankings from growth, profitability, balance-sheet safety, valuation, and technical downside risk.")

    if not DB_PATH.exists():
        st.error(f"Database not found: `{DB_PATH}`")
        return

    watchlists = available_watchlists(DB_PATH)
    with st.sidebar:
        st.header("Build")
        selected = st.multiselect(
            "Watchlists",
            watchlists,
            default=[name for name in ["sp500_ndx", "sp500", "ndx"] if name in watchlists] or watchlists[:1],
        )
        min_price_rows = st.slider("Minimum price rows", min_value=20, max_value=252, value=200, step=10)
        if st.button("Rebuild rankings", use_container_width=True):
            with st.spinner("Building industry rankings..."):
                write_screening_tables(
                    DB_PATH,
                    ScreenerInputs(watchlist_names=tuple(selected), min_price_rows=int(min_price_rows)),
                )
            st.cache_data.clear()
            st.rerun()

        st.divider()
        top_n = st.slider("Top industries", min_value=5, max_value=50, value=20, step=5)

    companies, financials, prices, scores = _load_materialized()
    industries = _load_table("industry_rankings").sort_values("industry_score", ascending=False, na_position="last")
    stocks = _stock_scores(companies, financials, prices, scores)

    if industries.empty or stocks.empty:
        st.warning("No materialized rankings found. Use Rebuild rankings after loading the watchlist, fundamentals, and prices.")
        return

    stocks = stocks.sort_values("overall_rank", ascending=True, na_position="last")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Industries", f"{len(industries):,}")
    with c2:
        st.metric("Stocks", f"{len(stocks):,}")
    with c3:
        st.metric("Average score", _fmt_score(stocks["total_score"].mean()))
    with c4:
        st.metric("Low risk", f"{int((stocks['risk_level'] == 'Low').sum()):,}")

    tab_industry, tab_top5, tab_stocks, tab_weights = st.tabs(["Industry Rankings", "Top 5 by Industry", "Stock Details", "Weights"])

    with tab_industry:
        left, right = st.columns([1.15, 1])
        with left:
            st.plotly_chart(_score_bar(industries, top_n), use_container_width=True)
        with right:
            show = industries.head(top_n).copy()
            for col in ["industry_score", "quality_score", "risk_score", "upside_score", "valuation_score"]:
                show[col] = show[col].apply(_fmt_score)
            show = show.rename(
                columns={
                    "industry": "Industry",
                    "sector": "Sector",
                    "industry_score": "Industry Score",
                    "quality_score": "Quality",
                    "risk_score": "Risk Protection",
                    "upside_score": "Upside",
                    "valuation_score": "Valuation",
                    "stock_count": "Stocks",
                    "top_stocks": "Top Stocks",
                }
            )
            st.dataframe(show, use_container_width=True, hide_index=True)

    with tab_top5:
        top5 = (
            stocks[stocks["industry_rank"] <= 5]
            .sort_values(["industry", "industry_rank"], na_position="last")
            .copy()
        )
        show_cols = [
            "industry",
            "industry_rank",
            "ticker",
            "company_name",
            "total_score",
            "upside_factors",
            "downside_risks",
            "valuation_comment",
            "risk_level",
        ]
        show = top5[[col for col in show_cols if col in top5.columns]].copy()
        show["total_score"] = show["total_score"].apply(_fmt_score)
        show = show.rename(
            columns={
                "industry": "Industry",
                "industry_rank": "Industry Rank",
                "ticker": "Ticker",
                "company_name": "Company",
                "total_score": "Total Score",
                "upside_factors": "Upside Factors",
                "downside_risks": "Downside Risks",
                "valuation_comment": "Valuation Comment",
                "risk_level": "Risk Level",
            }
        )
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Top 5 CSV",
            data=top5.to_csv(index=False).encode("utf-8"),
            file_name="industry_top5_stocks.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab_stocks:
        sectors = sorted(x for x in stocks["sector"].dropna().astype(str).unique() if x)
        sector = st.selectbox("Sector", ["All"] + sectors)
        detail = stocks.copy()
        if sector != "All":
            detail = detail[detail["sector"] == sector].copy()
        industries_filter = sorted(x for x in detail["industry"].dropna().astype(str).unique() if x)
        industry = st.selectbox("Industry", ["All"] + industries_filter)
        if industry != "All":
            detail = detail[detail["industry"] == industry].copy()

        detail_cols = [
            "overall_rank",
            "ticker",
            "company_name",
            "sector",
            "industry",
            "total_score",
            "growth_quality",
            "profitability_quality",
            "balance_sheet_safety",
            "valuation_reasonableness",
            "technical_downside_risk",
            "risk_level",
            "upside_factors",
            "downside_risks",
            "valuation_comment",
            "market_cap",
            "revenue_growth",
            "eps_growth",
            "gross_margin",
            "operating_margin",
            "free_cash_flow",
            "debt_to_equity",
            "roe",
            "roic",
            "pe_ratio",
            "forward_pe",
            "ev_ebitda",
            "price",
            "high_52w",
            "low_52w",
            "ma50",
            "ma200",
            "beta",
            "max_drawdown",
            "volatility",
        ]
        show = detail[[col for col in detail_cols if col in detail.columns]].copy()
        for col in list(SCORE_WEIGHTS) + ["total_score"]:
            if col in show:
                show[col] = show[col].apply(_fmt_score)
        for col in ["revenue_growth", "eps_growth", "gross_margin", "operating_margin", "roe", "roic", "max_drawdown", "volatility"]:
            if col in show:
                show[col] = show[col].apply(fmt_pct)
        if "market_cap" in show:
            show["market_cap"] = show["market_cap"].apply(fmt_money)
        show = show.rename(
            columns={
                "overall_rank": "Overall Rank",
                "ticker": "Ticker",
                "company_name": "Company",
                "sector": "Sector",
                "industry": "Industry",
                "total_score": "Total Score",
                "growth_quality": "Growth Quality",
                "profitability_quality": "Profitability Quality",
                "balance_sheet_safety": "Balance Sheet Safety",
                "valuation_reasonableness": "Valuation Reasonableness",
                "technical_downside_risk": "Technical Downside Risk",
                "risk_level": "Risk Level",
                "upside_factors": "Upside Factors",
                "downside_risks": "Downside Risks",
                "valuation_comment": "Valuation Comment",
                "market_cap": "Market Cap",
                "revenue_growth": "Revenue Growth",
                "eps_growth": "EPS Growth",
                "gross_margin": "Gross Margin",
                "operating_margin": "Operating Margin",
                "free_cash_flow": "Free Cash Flow",
                "debt_to_equity": "Debt to Equity",
                "roe": "ROE",
                "roic": "ROIC",
                "pe_ratio": "PE Ratio",
                "forward_pe": "Forward PE",
                "ev_ebitda": "EV/EBITDA",
                "price": "Price",
                "high_52w": "52W High",
                "low_52w": "52W Low",
                "ma50": "50-Day MA",
                "ma200": "200-Day MA",
                "beta": "Beta",
                "max_drawdown": "Max Drawdown",
                "volatility": "Volatility",
            }
        )
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Stock Scores CSV",
            data=detail.to_csv(index=False).encode("utf-8"),
            file_name="stock_scores.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab_weights:
        weights = pd.DataFrame(
            [{"Factor": SCORE_LABELS[key], "Weight": f"{weight:.0%}"} for key, weight in SCORE_WEIGHTS.items()]
        )
        st.dataframe(weights, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
