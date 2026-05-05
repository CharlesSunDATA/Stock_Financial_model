"""
Portfolio Watchlist Dashboard (Streamlit page)

Combines opportunity, risk, momentum, analyst targets, and upcoming earnings
into one watchlist cockpit.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.formatters import fmt_money as _fmt_money
from utils.formatters import fmt_pct as _fmt_pct
from utils.formatters import fmt_score as _fmt_score
from utils.local_data import connect_readonly, table_exists
from utils.opportunity_score import ScoreInputs, available_watchlists, compute_opportunity_scores
from utils.risk_score import RiskInputs, compute_risk_scores


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "quant_data.db"


def _risk_color(score: float) -> str:
    if score <= 30:
        return "#22c55e"
    if score <= 60:
        return "#f59e0b"
    return "#ef4444"


def _opportunity_color(score: float) -> str:
    if score >= 75:
        return "#22c55e"
    if score >= 60:
        return "#f59e0b"
    return "#ef4444"


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


@st.cache_data(ttl=60 * 10, show_spinner=False)
def _cached_opportunity_scores(db_path: str, watchlist_name: str | None, min_price_rows: int) -> pd.DataFrame:
    return compute_opportunity_scores(
        Path(db_path),
        ScoreInputs(watchlist_name=watchlist_name, min_price_rows=min_price_rows),
    )


@st.cache_data(ttl=60 * 10, show_spinner=False)
def _cached_risk_scores(db_path: str, watchlist_name: str | None, min_price_rows: int) -> pd.DataFrame:
    return compute_risk_scores(
        Path(db_path),
        RiskInputs(watchlist_name=watchlist_name, min_price_rows=min_price_rows),
    )


@st.cache_data(ttl=60 * 10, show_spinner=False)
def _cached_momentum(db_path: str, tickers: tuple[str, ...]) -> pd.DataFrame:
    db = Path(db_path)
    if not db.exists() or not tickers:
        return pd.DataFrame()

    with connect_readonly(db) as conn:
        if not table_exists(conn, "prices_eod"):
            return pd.DataFrame()
        latest_date = conn.execute("SELECT MAX(price_date) FROM prices_eod").fetchone()[0]
        if not latest_date:
            return pd.DataFrame()

        latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
        periods = {
            "ret_1M": (latest_dt - timedelta(days=30)).strftime("%Y-%m-%d"),
            "ret_3M": (latest_dt - timedelta(days=91)).strftime("%Y-%m-%d"),
            "ret_6M": (latest_dt - timedelta(days=182)).strftime("%Y-%m-%d"),
            "ret_12M": (latest_dt - timedelta(days=365)).strftime("%Y-%m-%d"),
        }

        placeholders = ",".join("?" * len(tickers))
        price_df = pd.read_sql_query(
            f"""
            SELECT ticker, price_date, COALESCE(adj_close, close) AS price
            FROM prices_eod
            WHERE ticker IN ({placeholders})
              AND price_date >= ?
            ORDER BY ticker, price_date
            """,
            conn,
            params=[*tickers, periods["ret_12M"]],
        )

    if price_df.empty:
        return pd.DataFrame()

    price_df["price"] = _safe_num(price_df["price"])
    rows: list[dict[str, object]] = []
    for ticker, grp in price_df.groupby("ticker"):
        grp = grp.sort_values("price_date").dropna(subset=["price"])
        if grp.empty:
            continue
        price_now = float(grp.iloc[-1]["price"])
        row: dict[str, object] = {"ticker": ticker, "latest_price": price_now}
        for col, date_str in periods.items():
            past = grp[grp["price_date"] <= date_str].dropna(subset=["price"])
            if past.empty:
                row[col] = np.nan
                continue
            base = float(past.iloc[-1]["price"])
            row[col] = (price_now - base) / base * 100 if base else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    ret_cols = ["ret_1M", "ret_3M", "ret_6M", "ret_12M"]
    for col in ret_cols:
        df[f"rank_{col}"] = df[col].rank(pct=True, na_option="keep")
    df["momentum_score"] = df[[f"rank_{col}" for col in ret_cols]].mean(axis=1) * 100
    return df[["ticker", "latest_price", *ret_cols, "momentum_score"]]


@st.cache_data(ttl=60 * 10, show_spinner=False)
def _cached_context(db_path: str, tickers: tuple[str, ...], days_ahead: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    db = Path(db_path)
    if not db.exists() or not tickers:
        return pd.DataFrame(), pd.DataFrame()

    placeholders = ",".join("?" * len(tickers))
    today = date.today()
    end = today + timedelta(days=days_ahead)

    with connect_readonly(db) as conn:
        targets = pd.DataFrame()
        if table_exists(conn, "price_target_consensus"):
            targets = pd.read_sql_query(
                f"""
                SELECT ticker, target_low, target_consensus, target_high, target_median, updated_at AS target_updated_at
                FROM price_target_consensus
                WHERE ticker IN ({placeholders})
                """,
                conn,
                params=list(tickers),
            )

        earnings = pd.DataFrame()
        if table_exists(conn, "earnings_calendar"):
            earnings = pd.read_sql_query(
                f"""
                SELECT ticker, event_date, time, eps_estimated, revenue_estimated
                FROM earnings_calendar
                WHERE ticker IN ({placeholders})
                  AND event_date >= ?
                  AND event_date <= ?
                ORDER BY event_date, ticker
                """,
                conn,
                params=[*tickers, today.isoformat(), end.isoformat()],
            )

    return targets, earnings


def _build_dashboard(
    db_path: Path,
    watchlist_name: str | None,
    min_price_rows: int,
    earnings_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    opp = _cached_opportunity_scores(str(db_path), watchlist_name, min_price_rows)
    risk = _cached_risk_scores(str(db_path), watchlist_name, max(min_price_rows, 200))
    if opp.empty:
        return pd.DataFrame(), pd.DataFrame()

    tickers = tuple(opp["ticker"].dropna().astype(str).str.upper().unique().tolist())
    momentum = _cached_momentum(str(db_path), tickers)
    targets, earnings = _cached_context(str(db_path), tickers, earnings_days)

    base_cols = ["ticker", "company_name", "sector", "score", "type", "judgment", "market_cap", "price_date"]
    df = opp[[c for c in base_cols if c in opp.columns]].copy()
    df = df.rename(columns={"score": "opportunity_score", "type": "opportunity_type"})

    risk_cols = ["ticker", "risk_score", "risk_level", "risk_drivers"]
    if not risk.empty:
        df = df.merge(risk[[c for c in risk_cols if c in risk.columns]], on="ticker", how="left")

    if not momentum.empty:
        df = df.merge(momentum, on="ticker", how="left")

    if not targets.empty:
        df = df.merge(targets, on="ticker", how="left")

    if "target_consensus" in df and "latest_price" in df:
        df["target_upside_pct"] = (df["target_consensus"] - df["latest_price"]) / df["latest_price"] * 100

    df["opportunity_score"] = _safe_num(df.get("opportunity_score", pd.Series(dtype=float)))
    df["risk_score"] = _safe_num(df.get("risk_score", pd.Series(dtype=float)))
    df["momentum_score"] = _safe_num(df.get("momentum_score", pd.Series(dtype=float)))
    df["target_upside_pct"] = _safe_num(df.get("target_upside_pct", pd.Series(dtype=float)))
    df["action_score"] = (
        df["opportunity_score"].fillna(50) * 0.45
        + (100 - df["risk_score"]).fillna(50) * 0.30
        + df["momentum_score"].fillna(50) * 0.15
        + df["target_upside_pct"].clip(-50, 50).add(50).fillna(50) * 0.10
    )

    df["dashboard_signal"] = np.select(
        [
            (df["opportunity_score"] >= 75) & (df["risk_score"] <= 45),
            (df["opportunity_score"] >= 65) & (df["target_upside_pct"] >= 15),
            (df["risk_score"] >= 65),
            (df["momentum_score"] <= 25),
        ],
        ["High Priority", "Upside Watch", "Risk Review", "Momentum Weak"],
        default="Monitor",
    )
    df = df.sort_values(["action_score", "opportunity_score"], ascending=False).reset_index(drop=True)
    return df, earnings


def _score_scatter(df: pd.DataFrame) -> go.Figure:
    sub = df.dropna(subset=["opportunity_score", "risk_score"]).copy()
    fig = go.Figure()
    if sub.empty:
        return fig
    fig.add_trace(
        go.Scatter(
            x=sub["risk_score"],
            y=sub["opportunity_score"],
            mode="markers+text",
            text=sub["ticker"],
            textposition="top center",
            marker=dict(
                size=12,
                color=sub["action_score"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Action"),
                line=dict(color="white", width=1),
            ),
            customdata=sub[["company_name", "dashboard_signal"]].fillna("").to_numpy(),
            hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>Signal: %{customdata[1]}<br>Opportunity: %{y:.0f}<br>Risk: %{x:.0f}<extra></extra>",
        )
    )
    fig.add_shape(type="rect", x0=0, x1=45, y0=75, y1=100, fillcolor="rgba(34,197,94,0.08)", line_width=0)
    fig.add_shape(type="rect", x0=65, x1=100, y0=0, y1=100, fillcolor="rgba(239,68,68,0.08)", line_width=0)
    fig.update_layout(
        height=460,
        margin=dict(t=20, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(title="Risk Score", range=[0, 100], gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Opportunity Score", range=[0, 100], gridcolor="rgba(255,255,255,0.08)"),
    )
    return fig


def _top_bar(df: pd.DataFrame, value_col: str, title: str, top_n: int, mode: str) -> go.Figure:
    sub = df.dropna(subset=[value_col]).head(top_n).iloc[::-1]
    if mode == "risk":
        colors = [_risk_color(float(v)) for v in sub[value_col]]
    elif mode == "opportunity":
        colors = [_opportunity_color(float(v)) for v in sub[value_col]]
    else:
        colors = ["#22c55e" if float(v) >= 0 else "#ef4444" for v in sub[value_col]]
    fig = go.Figure(
        go.Bar(
            x=sub[value_col],
            y=sub["ticker"],
            orientation="h",
            marker_color=colors,
            text=[_fmt_score(v) if mode != "pct" else _fmt_pct(v) for v in sub[value_col]],
            textposition="outside",
            hovertemplate=f"<b>%{{y}}</b><br>{title}: %{{x:.1f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=14, color="white")),
        height=max(320, top_n * 30),
        margin=dict(t=45, b=10, l=75, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(showgrid=False),
    )
    return fig


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    show = df.copy()
    for col in ["opportunity_score", "risk_score", "momentum_score", "action_score"]:
        if col in show:
            show[col] = show[col].apply(_fmt_score)
    for col in ["ret_1M", "ret_3M", "ret_6M", "ret_12M", "target_upside_pct"]:
        if col in show:
            show[col] = show[col].apply(_fmt_pct)
    for col in ["latest_price", "target_consensus", "target_low", "target_high", "target_median"]:
        if col in show:
            show[col] = show[col].apply(lambda v: f"${float(v):,.2f}" if pd.notna(v) else "-")
    if "market_cap" in show:
        show["market_cap"] = show["market_cap"].apply(_fmt_money)
    return show.rename(
        columns={
            "ticker": "Ticker",
            "company_name": "Company",
            "sector": "Sector",
            "opportunity_score": "Opportunity",
            "risk_score": "Risk",
            "risk_level": "Risk Level",
            "risk_drivers": "Risk Drivers",
            "momentum_score": "Momentum",
            "action_score": "Action Score",
            "dashboard_signal": "Signal",
            "judgment": "Judgment",
            "latest_price": "Last Price",
            "target_consensus": "Consensus Target",
            "target_upside_pct": "Target Upside",
            "ret_1M": "1M Return",
            "ret_3M": "3M Return",
            "ret_6M": "6M Return",
            "ret_12M": "12M Return",
            "market_cap": "Market Cap",
            "price_date": "Price Date",
        }
    )


def main() -> None:
    st.title("Portfolio Watchlist Dashboard")
    st.caption("One view for opportunity, risk, momentum, analyst targets, and upcoming earnings.")

    if not DB_PATH.exists():
        st.error(f"Database not found: `{DB_PATH}`")
        return

    watchlists = available_watchlists(DB_PATH)

    with st.sidebar:
        st.header("Watchlist")
        selected_watchlist = st.selectbox("Watchlist", ["All"] + watchlists, index=0)
        top_n = st.slider("Top N", min_value=5, max_value=50, value=15, step=5)
        min_price_rows = st.slider("Minimum price rows", min_value=60, max_value=252, value=200, step=10)
        earnings_days = st.slider("Earnings window", min_value=7, max_value=90, value=30, step=7)
        st.divider()
        if st.button("Refresh cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    watchlist_name = None if selected_watchlist == "All" else selected_watchlist
    with st.spinner("Building watchlist dashboard..."):
        df, earnings = _build_dashboard(DB_PATH, watchlist_name, min_price_rows, earnings_days)

    if df.empty:
        st.warning("No dashboard data. Check watchlists, prices, fundamentals, and score inputs.")
        return

    sectors = sorted([x for x in df["sector"].dropna().unique().tolist() if str(x).strip()])
    signals = sorted([x for x in df["dashboard_signal"].dropna().unique().tolist() if str(x).strip()])
    with st.sidebar:
        sector = st.selectbox("Sector", ["All", *sectors])
        signal = st.selectbox("Signal", ["All", *signals])

    if sector != "All":
        df = df[df["sector"] == sector].copy()
    if signal != "All":
        df = df[df["dashboard_signal"] == signal].copy()

    if df.empty:
        st.warning("No stocks match the selected filters.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stocks", f"{len(df):,}")
    c2.metric("Avg Opportunity", _fmt_score(df["opportunity_score"].mean()))
    c3.metric("Avg Risk", _fmt_score(df["risk_score"].mean()))
    c4.metric("High Priority", f"{int((df['dashboard_signal'] == 'High Priority').sum()):,}")
    c5.metric("Earnings Soon", f"{len(earnings):,}")

    st.divider()

    tab_overview, tab_actions, tab_earnings, tab_targets, tab_download = st.tabs(
        ["Overview", "Action List", "Earnings", "Analyst Targets", "Download"]
    )

    with tab_overview:
        left, right = st.columns([1.05, 1])
        with left:
            st.plotly_chart(_score_scatter(df), use_container_width=True)
        with right:
            top = _format_table(
                df.head(top_n)[
                    [
                        "ticker",
                        "company_name",
                        "dashboard_signal",
                        "action_score",
                        "opportunity_score",
                        "risk_score",
                        "momentum_score",
                        "target_upside_pct",
                    ]
                ]
            )
            st.dataframe(top, use_container_width=True, hide_index=True)

    with tab_actions:
        a1, a2 = st.columns(2)
        with a1:
            st.plotly_chart(
                _top_bar(df.sort_values("opportunity_score", ascending=False), "opportunity_score", "Top Opportunity Scores", top_n, "opportunity"),
                use_container_width=True,
            )
        with a2:
            st.plotly_chart(
                _top_bar(df.sort_values("risk_score", ascending=False), "risk_score", "Highest Risk Scores", top_n, "risk"),
                use_container_width=True,
            )

        action_cols = [
            "ticker",
            "company_name",
            "sector",
            "dashboard_signal",
            "judgment",
            "risk_level",
            "risk_drivers",
            "action_score",
            "opportunity_score",
            "risk_score",
            "momentum_score",
            "target_upside_pct",
        ]
        st.dataframe(_format_table(df[[c for c in action_cols if c in df.columns]]), use_container_width=True, hide_index=True)

    with tab_earnings:
        if earnings.empty:
            st.info("No upcoming earnings in the selected window.")
        else:
            view = earnings.copy()
            view["eps_estimated"] = view["eps_estimated"].apply(lambda v: f"{float(v):.2f}" if pd.notna(v) else "-")
            view["revenue_estimated"] = view["revenue_estimated"].apply(_fmt_money)
            view = view.rename(
                columns={
                    "ticker": "Ticker",
                    "event_date": "Event Date",
                    "time": "Time",
                    "eps_estimated": "Estimated EPS",
                    "revenue_estimated": "Estimated Revenue",
                }
            )
            st.dataframe(view, use_container_width=True, hide_index=True)

    with tab_targets:
        target_cols = [
            "ticker",
            "company_name",
            "latest_price",
            "target_low",
            "target_consensus",
            "target_median",
            "target_high",
            "target_upside_pct",
            "target_updated_at",
        ]
        if "target_consensus" not in df.columns:
            target_df = pd.DataFrame()
        else:
            target_df = df[[c for c in target_cols if c in df.columns]].dropna(subset=["target_consensus"], how="all")
        if target_df.empty:
            st.info("No analyst price target data for the selected stocks.")
        else:
            b1, b2 = st.columns([1, 1])
            with b1:
                st.plotly_chart(
                    _top_bar(target_df.sort_values("target_upside_pct", ascending=False), "target_upside_pct", "Largest Target Upside", top_n, "pct"),
                    use_container_width=True,
                )
            with b2:
                st.dataframe(_format_table(target_df), use_container_width=True, hide_index=True)

    with tab_download:
        export_cols = [
            "ticker",
            "company_name",
            "sector",
            "dashboard_signal",
            "action_score",
            "opportunity_score",
            "judgment",
            "risk_score",
            "risk_level",
            "risk_drivers",
            "momentum_score",
            "ret_1M",
            "ret_3M",
            "ret_6M",
            "ret_12M",
            "latest_price",
            "target_consensus",
            "target_upside_pct",
            "market_cap",
            "price_date",
        ]
        export = df[[c for c in export_cols if c in df.columns]].copy()
        st.dataframe(_format_table(export), use_container_width=True, hide_index=True)
        st.download_button(
            "Download Watchlist CSV",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name="portfolio_watchlist_dashboard.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
