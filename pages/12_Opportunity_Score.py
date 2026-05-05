"""
Stock Opportunity Score (Streamlit page)

Ranks research candidates with a weighted score:
- Price momentum: 25%
- Revenue growth: 20%
- EPS / FCF improvement: 20%
- Valuation reasonableness: 15%
- Financial safety: 10%
- Industry strength: 10%
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.formatters import fmt_money as _fmt_money
from utils.formatters import fmt_pct as _fmt_pct
from utils.formatters import fmt_score
from utils.opportunity_score import FACTOR_LABELS, FACTOR_WEIGHTS, INDUSTRY_FACTOR_LABELS, INDUSTRY_FACTOR_WEIGHTS, ScoreInputs
from utils.opportunity_score import available_market_themes, available_watchlists, build_theme_rankings, compute_opportunity_scores


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "quant_data.db"


def _fmt_score(v) -> str:
    return fmt_score(v, missing="—")


def _score_bar(df: pd.DataFrame, top_n: int) -> go.Figure:
    sub = df.dropna(subset=["score"]).head(top_n).iloc[::-1]
    colors = [
        "#2ecc71" if v >= 75 else "#f1c40f" if v >= 60 else "#e74c3c"
        for v in sub["score"]
    ]
    fig = go.Figure(
        go.Bar(
            x=sub["score"],
            y=sub["ticker"],
            orientation="h",
            marker_color=colors,
            text=[_fmt_score(v) for v in sub["score"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Score: %{x:.0f}<extra></extra>",
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
    factors = list(FACTOR_WEIGHTS.keys())
    labels = [FACTOR_LABELS[f] for f in factors]
    sub = df.dropna(subset=["score"]).head(top_n)
    z = sub[factors].apply(pd.to_numeric, errors="coerce").to_numpy()
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=sub["ticker"],
            colorscale=[
                [0.0, "#440154"],
                [0.35, "#31688e"],
                [0.7, "#35b779"],
                [1.0, "#fde725"],
            ],
            zmin=0,
            zmax=100,
            colorbar=dict(title="Score", tickmode="array", tickvals=[0, 25, 50, 75, 100]),
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


def _industry_score_bar(df: pd.DataFrame, top_n: int) -> go.Figure:
    sub = (
        df.dropna(subset=["industry_opportunity_score"])
        .sort_values(["industry_group", "industry_rank"])
        .head(top_n)
        .iloc[::-1]
    )
    labels = sub["ticker"] + " · " + sub["industry_group"].astype(str).str.slice(0, 28)
    colors = [
        "#2ecc71" if v >= 80 else "#f1c40f" if v >= 70 else "#95a5a6"
        for v in sub["industry_opportunity_score"]
    ]
    fig = go.Figure(
        go.Bar(
            x=sub["industry_opportunity_score"],
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[_fmt_score(v) for v in sub["industry_opportunity_score"]],
            textposition="outside",
            customdata=sub[["industry_rank", "industry_peer_count", "industry_candidate_type"]],
            hovertemplate=(
                "<b>%{y}</b><br>Industry Score: %{x:.0f}"
                "<br>Industry Rank: %{customdata[0]} of %{customdata[1]}"
                "<br>Type: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=max(420, top_n * 26),
        margin=dict(t=20, b=10, l=170, r=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(showgrid=False),
    )
    return fig


def _industry_factor_heatmap(df: pd.DataFrame, top_n: int) -> go.Figure:
    factors = list(INDUSTRY_FACTOR_WEIGHTS.keys())
    labels = [INDUSTRY_FACTOR_LABELS[f] for f in factors]
    sub = df.dropna(subset=["industry_opportunity_score"]).head(top_n)
    z = sub[factors].apply(pd.to_numeric, errors="coerce").to_numpy()
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=sub["ticker"],
            colorscale=[
                [0.0, "#440154"],
                [0.35, "#31688e"],
                [0.7, "#35b779"],
                [1.0, "#fde725"],
            ],
            zmin=0,
            zmax=100,
            colorbar=dict(title="Score", tickmode="array", tickvals=[0, 25, 50, 75, 100]),
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


def _theme_score_bar(df: pd.DataFrame, top_n: int) -> go.Figure:
    sub = (
        df.dropna(subset=["theme_opportunity_score"])
        .sort_values(["market_theme", "theme_rank"])
        .head(top_n)
        .iloc[::-1]
    )
    labels = sub["ticker"] + " · " + sub["market_theme"].astype(str).str.slice(0, 28)
    colors = [
        "#2ecc71" if v >= 82 else "#f1c40f" if v >= 72 else "#95a5a6"
        for v in sub["theme_opportunity_score"]
    ]
    fig = go.Figure(
        go.Bar(
            x=sub["theme_opportunity_score"],
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[_fmt_score(v) for v in sub["theme_opportunity_score"]],
            textposition="outside",
            customdata=sub[["theme_rank", "theme_peer_count", "theme_candidate_type"]],
            hovertemplate=(
                "<b>%{y}</b><br>Theme Score: %{x:.0f}"
                "<br>Theme Rank: %{customdata[0]} of %{customdata[1]}"
                "<br>Recommendation: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=max(420, top_n * 26),
        margin=dict(t=20, b=10, l=170, r=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(showgrid=False),
    )
    return fig


def _theme_factor_heatmap(df: pd.DataFrame, top_n: int) -> go.Figure:
    factor_cols = [
        "theme_quality_score",
        "theme_growth_score",
        "theme_valuation_score",
        "theme_risk_protection_score",
        "theme_upside_momentum_score",
    ]
    labels = ["Quality", "Growth", "Valuation", "Risk Protection", "Upside Momentum"]
    sub = df.dropna(subset=["theme_opportunity_score"]).head(top_n)
    z = sub[factor_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=sub["ticker"],
            colorscale=[
                [0.0, "#440154"],
                [0.35, "#31688e"],
                [0.7, "#35b779"],
                [1.0, "#fde725"],
            ],
            zmin=0,
            zmax=100,
            colorbar=dict(title="Score", tickmode="array", tickvals=[0, 25, 50, 75, 100]),
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
    return compute_opportunity_scores(
        Path(db_path),
        ScoreInputs(watchlist_name=watchlist_name, min_price_rows=min_price_rows),
    )


def main() -> None:
    st.title("Stock Opportunity Score")
    st.caption("Daily research ranking from price, fundamentals, valuation, balance-sheet risk, and sector strength.")

    if not DB_PATH.exists():
        st.error(f"Database not found: `{DB_PATH}`")
        return

    watchlists = available_watchlists(DB_PATH)

    with st.sidebar:
        st.header("Ranking")
        st.caption(f"DB: `{DB_PATH}`")
        selected_watchlist = st.selectbox("Watchlist", ["All"] + watchlists, index=0)
        top_n = st.slider("Top N", min_value=10, max_value=100, value=30, step=5)
        min_price_rows = st.slider("Minimum price rows", min_value=20, max_value=252, value=60, step=10)
        min_score = st.slider("Minimum score", min_value=0, max_value=100, value=0, step=5)

        st.divider()
        if st.button("Refresh cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    watchlist_name = None if selected_watchlist == "All" else selected_watchlist
    with st.spinner("Computing opportunity scores..."):
        df = _cached_scores(str(DB_PATH), watchlist_name, min_price_rows)

    if df.empty:
        st.warning("No score data. Check `fmp_watchlist`, `prices_eod`, `fundamental_data`, and `key_metrics_ttm`.")
        return

    df = df[df["score"].fillna(-1) >= min_score].copy()
    if df.empty:
        st.warning("No stocks match the selected score filter.")
        return

    sectors = sorted([x for x in df["sector"].dropna().unique().tolist() if str(x).strip()])
    with st.sidebar:
        sector = st.selectbox("Sector", ["All"] + sectors)
    if sector != "All":
        df = df[df["sector"] == sector].copy()

    industry_groups = sorted([x for x in df["industry_group"].dropna().unique().tolist() if str(x).strip()])
    with st.sidebar:
        industry_group = st.selectbox("Industry", ["All"] + industry_groups)
    if industry_group != "All":
        df = df[df["industry_group"] == industry_group].copy()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Stocks ranked", f"{len(df):,}")
    with c2:
        st.metric("Average score", _fmt_score(df["score"].mean()))
    with c3:
        st.metric("Score >= 75", f"{int((df['score'] >= 75).sum()):,}")
    with c4:
        latest_date = df["price_date"].dropna().max() if "price_date" in df else None
        st.metric("Price date", latest_date or "—")

    st.divider()

    tab_rank, tab_theme, tab_industry, tab_factors, tab_details, tab_weights = st.tabs(
        ["Rankings", "Theme rankings", "Industry rankings", "Factor map", "Details", "Weights"]
    )

    with tab_rank:
        left, right = st.columns([1.1, 1])
        with left:
            st.plotly_chart(_score_bar(df, top_n), use_container_width=True)
        with right:
            show = df.head(top_n)[["ticker", "score", "type", "judgment", "company_name", "sector"]].copy()
            show["score"] = show["score"].apply(_fmt_score)
            show = show.rename(
                columns={
                    "ticker": "Ticker",
                    "score": "Score",
                    "type": "Type",
                    "judgment": "Judgment",
                    "company_name": "Company",
                    "sector": "Sector",
                }
            )
            st.dataframe(show, use_container_width=True, hide_index=True)

    with tab_theme:
        theme_df = build_theme_rankings(df)

        if theme_df.empty:
            st.warning("No theme ranking data for the selected filters.")
        else:
            available_themes = [
                theme
                for theme in available_market_themes()
                if theme in set(theme_df["market_theme"].dropna().astype(str))
            ]
            selected_theme = st.selectbox(
                "Market Theme",
                ["All"] + available_themes,
                index=1 if "AI Infrastructure" in available_themes else 0,
                key="theme_rankings_market_theme",
            )
            if selected_theme != "All":
                theme_df = theme_df[theme_df["market_theme"] == selected_theme].copy()

            theme_candidates = theme_df[
                (theme_df["top_theme_candidate"] == "Yes")
                | (theme_df["theme_candidate_type"].isin(["Recommended", "Positive Watchlist"]))
            ].copy()
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Themes covered", f"{theme_df['market_theme'].nunique():,}")
            with c2:
                st.metric("Theme candidates", f"{len(theme_candidates):,}")
            with c3:
                st.metric("Recommended", f"{int((theme_df['theme_candidate_type'] == 'Recommended').sum()):,}")

            left, right = st.columns([1.1, 1])
            ranked_theme_df = theme_candidates if not theme_candidates.empty else theme_df
            with left:
                st.plotly_chart(_theme_score_bar(ranked_theme_df, top_n), use_container_width=True)
            with right:
                show = ranked_theme_df.head(top_n)[
                    [
                        "ticker",
                        "theme_rank",
                        "theme_opportunity_score",
                        "theme_candidate_type",
                        "market_theme",
                        "theme_peer_count",
                        "theme_thesis",
                    ]
                ].copy()
                show["theme_opportunity_score"] = show["theme_opportunity_score"].apply(_fmt_score)
                show = show.rename(
                    columns={
                        "ticker": "Ticker",
                        "theme_rank": "Theme Rank",
                        "theme_opportunity_score": "Theme Score",
                        "theme_candidate_type": "Recommendation",
                        "market_theme": "Market Theme",
                        "theme_peer_count": "Peers",
                        "theme_thesis": "Theme Thesis",
                    }
                )
                st.dataframe(show, use_container_width=True, hide_index=True)

            st.plotly_chart(_theme_factor_heatmap(theme_df.head(min(top_n, 40)), min(top_n, 40)), use_container_width=True)
            st.download_button(
                "Download Theme Rankings CSV",
                data=theme_df.to_csv(index=False).encode("utf-8"),
                file_name="market_theme_rankings.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_industry:
        industry_df = df.sort_values(
            ["industry_group", "industry_rank", "industry_opportunity_score"],
            ascending=[True, True, False],
            na_position="last",
        ).copy()
        top_candidates = industry_df[
            (industry_df["top_industry_candidate"] == "Yes")
            | (industry_df["industry_candidate_type"].isin(["Attractive", "Watchlist"]))
        ].copy()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Industries covered", f"{industry_df['industry_group'].nunique():,}")
        with c2:
            st.metric("Top candidates", f"{len(top_candidates):,}")
        with c3:
            st.metric("Attractive", f"{int((industry_df['industry_candidate_type'] == 'Attractive').sum()):,}")

        left, right = st.columns([1.1, 1])
        with left:
            st.plotly_chart(_industry_score_bar(top_candidates if not top_candidates.empty else industry_df, top_n), use_container_width=True)
        with right:
            show = (top_candidates if not top_candidates.empty else industry_df).head(top_n)[
                [
                    "industry_group",
                    "industry_rank",
                    "industry_peer_count",
                    "ticker",
                    "industry_opportunity_score",
                    "industry_candidate_type",
                    "industry_thesis",
                ]
            ].copy()
            show["industry_opportunity_score"] = show["industry_opportunity_score"].apply(_fmt_score)
            show = show.rename(
                columns={
                    "industry_group": "Industry",
                    "industry_rank": "Industry Rank",
                    "industry_peer_count": "Peers",
                    "ticker": "Ticker",
                    "industry_opportunity_score": "Industry Score",
                    "industry_candidate_type": "Candidate Type",
                    "industry_thesis": "Thesis",
                }
            )
            st.dataframe(show, use_container_width=True, hide_index=True)

        st.plotly_chart(_industry_factor_heatmap(industry_df.head(min(top_n, 40)), min(top_n, 40)), use_container_width=True)

    with tab_factors:
        st.plotly_chart(_factor_heatmap(df, min(top_n, 40)), use_container_width=True)

    with tab_details:
        detail_cols = [
            "ticker",
            "score",
            "type",
            "judgment",
            "market_themes",
            "primary_market_theme",
            "industry_group",
            "industry_rank",
            "industry_peer_count",
            "industry_opportunity_score",
            "industry_candidate_type",
            "industry_thesis",
            "quality_score",
            "growth_score",
            "valuation_score",
            "risk_protection_score",
            "downside_protection_score",
            "upside_momentum_score",
            "company_name",
            "sector",
            "industry",
            "price_momentum",
            "revenue_growth",
            "eps_fcf_improvement",
            "valuation_reasonableness",
            "financial_safety",
            "industry_strength",
            "ret_1m",
            "ret_3m",
            "ret_6m",
            "ret_12m",
            "target_upside_pct",
            "target_low_downside_pct",
            "drawdown_from_52w_high_pct",
            "volatility_6m_pct",
            "revenue_yoy_pct",
            "eps_yoy_pct",
            "fcf_yoy_pct",
            "ev_to_sales",
            "ev_to_free_cash_flow",
            "net_debt_to_ebitda",
            "market_cap",
        ]
        show = df[[c for c in detail_cols if c in df.columns]].copy()
        for col in list(FACTOR_WEIGHTS) + list(INDUSTRY_FACTOR_WEIGHTS) + [
            "score",
            "industry_opportunity_score",
            "downside_protection_score",
        ]:
            if col in show:
                show[col] = show[col].apply(_fmt_score)
        for col in [
            "ret_1m",
            "ret_3m",
            "ret_6m",
            "ret_12m",
            "target_upside_pct",
            "target_low_downside_pct",
            "drawdown_from_52w_high_pct",
            "volatility_6m_pct",
            "revenue_yoy_pct",
            "eps_yoy_pct",
            "fcf_yoy_pct",
        ]:
            if col in show:
                show[col] = show[col].apply(_fmt_pct)
        if "market_cap" in show:
            show["market_cap"] = show["market_cap"].apply(_fmt_money)
        show = show.rename(
            columns={
                "ticker": "Ticker",
                "score": "Score",
                "type": "Type",
                "judgment": "Judgment",
                "market_themes": "Market Themes",
                "primary_market_theme": "Primary Market Theme",
                "industry_group": "Industry Group",
                "industry_rank": "Industry Rank",
                "industry_peer_count": "Industry Peers",
                "industry_opportunity_score": "Industry Score",
                "industry_candidate_type": "Industry Candidate Type",
                "industry_thesis": "Industry Thesis",
                "quality_score": "Quality",
                "growth_score": "Growth",
                "valuation_score": "Valuation",
                "risk_protection_score": "Risk Protection",
                "downside_protection_score": "Downside Protection",
                "upside_momentum_score": "Upside Momentum",
                "company_name": "Company",
                "sector": "Sector",
                "industry": "Industry",
                "price_momentum": "Price Momentum",
                "revenue_growth": "Revenue Growth",
                "eps_fcf_improvement": "EPS/FCF Improvement",
                "valuation_reasonableness": "Valuation Reasonableness",
                "financial_safety": "Financial Safety",
                "industry_strength": "Industry Strength",
                "ret_1m": "1M Return",
                "ret_3m": "3M Return",
                "ret_6m": "6M Return",
                "ret_12m": "12M Return",
                "target_upside_pct": "Target Upside",
                "target_low_downside_pct": "Target Low Downside",
                "drawdown_from_52w_high_pct": "Drawdown from 52W High",
                "volatility_6m_pct": "6M Volatility",
                "revenue_yoy_pct": "Revenue YoY",
                "eps_yoy_pct": "EPS YoY",
                "fcf_yoy_pct": "FCF YoY",
                "ev_to_sales": "EV/Sales",
                "ev_to_free_cash_flow": "EV/FCF",
                "net_debt_to_ebitda": "Net debt/EBITDA",
                "market_cap": "Market Cap",
            }
        )
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="stock_opportunity_score.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab_weights:
        weights = pd.DataFrame(
            [
                {"Factor": FACTOR_LABELS[key], "Weight": f"{weight:.0%}"}
                for key, weight in FACTOR_WEIGHTS.items()
            ]
        )
        st.dataframe(weights, use_container_width=True, hide_index=True)
        industry_weights = pd.DataFrame(
            [
                {"Industry Factor": INDUSTRY_FACTOR_LABELS[key], "Weight": f"{weight:.0%}"}
                for key, weight in INDUSTRY_FACTOR_WEIGHTS.items()
            ]
        )
        st.dataframe(industry_weights, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
