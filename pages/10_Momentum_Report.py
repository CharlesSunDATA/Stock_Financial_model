"""
Momentum Report (Streamlit page)

Reads prices_eod + company_profile from SQLite and ranks watchlist stocks
by 1/3/6/12-month momentum. Displays top/bottom movers and allows refresh.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.local_data import connect_readonly

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "quant_data.db"
REPORT_SCRIPT = Path(__file__).resolve().parents[1] / "momentum_report.py"


def _connect() -> sqlite3.Connection:
    return connect_readonly(DB_PATH)


@st.cache_data(ttl=60 * 10, show_spinner=False)
def compute_momentum() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    with _connect() as conn:
        latest_date = conn.execute("SELECT MAX(price_date) FROM prices_eod").fetchone()[0]
        if not latest_date:
            return pd.DataFrame()

        latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
        periods = {
            "1M":  (latest_dt - timedelta(days=30)).strftime("%Y-%m-%d"),
            "3M":  (latest_dt - timedelta(days=91)).strftime("%Y-%m-%d"),
            "6M":  (latest_dt - timedelta(days=182)).strftime("%Y-%m-%d"),
            "12M": (latest_dt - timedelta(days=365)).strftime("%Y-%m-%d"),
        }

        try:
            tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM fmp_watchlist").fetchall()]
        except Exception:
            tickers = []
        if not tickers:
            return pd.DataFrame()

        placeholders = ",".join("?" * len(tickers))
        price_df = pd.read_sql(
            f"""
            SELECT ticker, price_date, COALESCE(adj_close, close) AS price
            FROM prices_eod
            WHERE ticker IN ({placeholders})
              AND price_date >= ?
            ORDER BY ticker, price_date
            """,
            conn,
            params=tickers + [periods["12M"]],
        )
        if price_df.empty:
            return pd.DataFrame()

        names = pd.read_sql(
            "SELECT ticker, company_name, sector FROM company_profile",
            conn,
        ).drop_duplicates("ticker")

        try:
            mcap = pd.read_sql(
                """SELECT ticker, market_cap FROM key_metrics_ttm
                   WHERE as_of_date = (SELECT MAX(as_of_date) FROM key_metrics_ttm)""",
                conn,
            ).drop_duplicates("ticker")
        except Exception:
            mcap = pd.DataFrame(columns=["ticker", "market_cap"])

    price_df["price"] = pd.to_numeric(price_df["price"], errors="coerce")

    results = []
    for ticker, grp in price_df.groupby("ticker"):
        grp = grp.sort_values("price_date").dropna(subset=["price"])
        if grp.empty:
            continue
        price_now = grp.iloc[-1]["price"]
        if not price_now or np.isnan(price_now):
            continue
        row: dict = {"ticker": ticker, "price": price_now}
        for period, date_str in periods.items():
            past = grp[grp["price_date"] <= date_str].dropna(subset=["price"])
            if not past.empty and past.iloc[-1]["price"]:
                p0 = float(past.iloc[-1]["price"])
                row[f"ret_{period}"] = (float(price_now) - p0) / p0 * 100
            else:
                row[f"ret_{period}"] = None
        # max drawdown per period
        for period, date_str in periods.items():
            window = grp[grp["price_date"] >= date_str]["price"]
            if len(window) >= 2:
                roll_max = window.cummax()
                dd = (window - roll_max) / roll_max * 100
                row[f"maxdd_{period}"] = float(dd.min())
            else:
                row[f"maxdd_{period}"] = None

        results.append(row)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).dropna(subset=["ret_1M", "ret_3M"])
    df = df.merge(names, on="ticker", how="left")
    df = df.merge(mcap, on="ticker", how="left")

    for col in ["ret_1M", "ret_3M", "ret_6M", "ret_12M"]:
        df[f"rank_{col}"] = df[col].rank(pct=True, na_option="keep")
    rank_cols = [c for c in df.columns if c.startswith("rank_ret_")]
    df["momentum_score"] = df[rank_cols].mean(axis=1) * 100

    df["company_name"] = df["company_name"].apply(
        lambda v: str(v) if pd.notna(v) and v else ""
    )
    return df.sort_values("momentum_score", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=60 * 10, show_spinner=False)
def compute_selloff_resilience(
    lookback_days: int = 91,
    drop_threshold: float = -1.5,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Identify days in the lookback window where the market (SPY, or median of all
    watchlist stocks if SPY is absent) fell >= drop_threshold %.
    For each watchlist stock rank by:
      - avg_selloff_ret  : average daily return on those market-down days (higher = more resilient)
      - vs_market        : avg_selloff_ret minus market's avg on same days (outperformance)
      - down_beta        : covariance(stock, market) / var(market) on down days only (lower = safer)
      - worst_selloff    : worst single-day return during a selloff day (tail risk)
    Returns (resilience_df, selloff_calendar_df, market_label).
    """
    if not DB_PATH.exists():
        return pd.DataFrame(), pd.DataFrame(), "SPY"

    with _connect() as conn:
        latest_date = conn.execute("SELECT MAX(price_date) FROM prices_eod").fetchone()[0]
        if not latest_date:
            return pd.DataFrame(), pd.DataFrame(), "SPY"

        since = (
            datetime.strptime(latest_date, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        try:
            tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM fmp_watchlist").fetchall()]
        except Exception:
            tickers = []
        if not tickers:
            return pd.DataFrame(), pd.DataFrame(), "SPY"

        all_fetch = list(set(tickers + ["SPY"]))
        placeholders = ",".join("?" * len(all_fetch))
        price_df = pd.read_sql(
            f"""SELECT ticker, price_date, COALESCE(adj_close, close) AS price
                FROM prices_eod
                WHERE ticker IN ({placeholders}) AND price_date >= ?
                ORDER BY ticker, price_date""",
            conn,
            params=all_fetch + [since],
        )
        names = pd.read_sql(
            "SELECT ticker, company_name, sector FROM company_profile", conn
        ).drop_duplicates("ticker")

    price_df["price"] = pd.to_numeric(price_df["price"], errors="coerce")
    pivot = price_df.pivot(index="price_date", columns="ticker", values="price").sort_index()
    daily_ret = pivot.pct_change() * 100  # % daily returns

    # Market proxy: SPY if available, else median of watchlist
    if "SPY" in daily_ret.columns and daily_ret["SPY"].notna().sum() > 10:
        market_ret = daily_ret["SPY"].dropna()
        market_label = "SPY"
    else:
        watchlist_in = [t for t in tickers if t in daily_ret.columns]
        market_ret = daily_ret[watchlist_in].median(axis=1).dropna()
        market_label = "Watchlist median"

    # Selloff days: market daily return <= threshold
    selloff_idx = market_ret[market_ret <= drop_threshold].index
    if len(selloff_idx) == 0:
        return pd.DataFrame(), pd.DataFrame(), market_label

    selloff_calendar = pd.DataFrame({
        "date": selloff_idx,
        "market_return_%": market_ret[selloff_idx].values,
    }).sort_values("market_return_%").reset_index(drop=True)

    results = []
    for ticker in tickers:
        if ticker not in daily_ret.columns:
            continue
        s = daily_ret[ticker].dropna()
        if s.empty:
            continue

        common_selloff = selloff_idx.intersection(s.index)
        if len(common_selloff) < 2:
            continue

        s_down = s[common_selloff]
        m_down = market_ret[common_selloff]

        avg_ret = float(s_down.mean())
        mkt_avg = float(m_down.mean())
        vs_market = avg_ret - mkt_avg
        worst_day = float(s_down.min())

        # downside beta: cov(stock, market) / var(market) on selloff days
        var_m = float(m_down.var())
        down_beta = float(np.cov(s_down, m_down)[0, 1] / var_m) if var_m > 0 else None

        # positive days count during selloffs (held up / went up)
        positive_days = int((s_down >= 0).sum())
        pct_positive = positive_days / len(common_selloff) * 100

        results.append({
            "ticker": ticker,
            "avg_selloff_ret_%": avg_ret,
            "vs_market_%": vs_market,
            "down_beta": down_beta,
            "worst_single_day_%": worst_day,
            "pct_days_positive_%": pct_positive,
            "selloff_days_n": len(common_selloff),
        })

    if not results:
        return pd.DataFrame(), selloff_calendar, market_label

    res = pd.DataFrame(results)
    res = res.merge(names, on="ticker", how="left")
    res["company_name"] = res["company_name"].apply(
        lambda v: str(v) if pd.notna(v) and v else ""
    )
    # resilience rank: higher avg_selloff_ret = more resilient
    res = res.sort_values("avg_selloff_ret_%", ascending=False).reset_index(drop=True)
    return res, selloff_calendar, market_label


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v):+.1f}%"


def _bar_chart(df: pd.DataFrame, col: str, title: str, top_n: int = 30) -> go.Figure:
    sub = df.dropna(subset=[col]).nlargest(top_n, col)
    sub = sub.iloc[::-1]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sub[col]]
    fig = go.Figure(go.Bar(
        x=sub[col],
        y=sub["ticker"],
        orientation="h",
        marker_color=colors,
        text=[_fmt_pct(v) for v in sub[col]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>" + title + ": %{x:+.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=13, color="white")),
        height=max(300, top_n * 22),
        margin=dict(t=40, b=10, l=70, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=True,
                   zerolinecolor="rgba(255,255,255,0.3)", ticksuffix="%"),
        yaxis=dict(showgrid=False),
    )
    return fig


def _resilience_chart(res_df: pd.DataFrame, col: str, label: str, top_n: int, ascending: bool = False) -> go.Figure:
    sub = res_df.dropna(subset=[col])
    sub = sub.nsmallest(top_n, col) if ascending else sub.nlargest(top_n, col)
    sub = sub.iloc[::-1]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sub[col]]
    fig = go.Figure(go.Bar(
        x=sub[col],
        y=sub["ticker"],
        orientation="h",
        marker_color=colors,
        text=[_fmt_pct(v) if "%" in col else f"{v:.2f}" for v in sub[col]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>" + label + ": %{x:+.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=label, x=0, font=dict(size=13, color="white")),
        height=max(300, top_n * 22),
        margin=dict(t=40, b=10, l=70, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=True,
                   zerolinecolor="rgba(255,255,255,0.3)",
                   ticksuffix="%" if "%" in col else ""),
        yaxis=dict(showgrid=False),
    )
    return fig


def main() -> None:
    st.title("Momentum Rankings")
    st.caption("1/3/6/12-month price momentum from SQLite `prices_eod`. Composite score = equal-weight percentile rank.")

    if not DB_PATH.exists():
        st.error(f"Database not found: `{DB_PATH}`")
        return

    with st.sidebar:
        st.header("Controls")
        st.caption(f"DB: `{DB_PATH}`")
        top_n = st.slider("Top/Bottom N", 10, 50, 30, step=5)
        sector_filter = "All"

        st.divider()
        if st.button("Refresh cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        if REPORT_SCRIPT.exists():
            if st.button("Regenerate momentum_report.md", type="secondary", use_container_width=True):
                with st.spinner("Running momentum_report.py…"):
                    res = subprocess.run(
                        [sys.executable, str(REPORT_SCRIPT)],
                        cwd=str(REPORT_SCRIPT.parent),
                        capture_output=True,
                        text=True,
                    )
                if res.returncode == 0:
                    st.success("Report generated.")
                else:
                    st.error("Report generation failed.")
                with st.expander("Output log"):
                    st.code((res.stdout or "") + (res.stderr or ""), language="text")

    with st.spinner("Computing momentum…"):
        df = compute_momentum()

    if df.empty:
        st.warning("No momentum data. Make sure `prices_eod` has data and the `fmp_watchlist` table is populated.")
        return

    sectors = sorted(df["sector"].dropna().unique().tolist())
    with st.sidebar:
        sector_filter = st.selectbox("Sector filter", ["All"] + sectors)

    if sector_filter != "All":
        df = df[df["sector"] == sector_filter]

    latest_date = None
    try:
        with _connect() as conn:
            latest_date = conn.execute("SELECT MAX(price_date) FROM prices_eod").fetchone()[0]
    except Exception:
        pass

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Stocks ranked", f"{len(df):,}")
    with c2:
        st.metric("Data date", latest_date or "—")
    with c3:
        st.metric("Sector", sector_filter)

    st.divider()

    tab_top, tab_bottom, tab_period, tab_dd, tab_resilience, tab_table = st.tabs(
        ["Top movers", "Bottom movers", "Period charts", "Max Drawdown", "Selloff Resilience", "Full table"]
    )

    with tab_top:
        top_df = df.head(top_n).copy()
        for col in ["ret_1M", "ret_3M", "ret_6M", "ret_12M", "momentum_score"]:
            if col in top_df.columns:
                top_df[col] = top_df[col].apply(_fmt_pct if col != "momentum_score" else lambda v: f"{v:.1f}")
        show_cols = ["ticker", "company_name", "sector", "ret_1M", "ret_3M", "ret_6M", "ret_12M", "momentum_score"]
        show_cols = [c for c in show_cols if c in top_df.columns]
        st.dataframe(top_df[show_cols], use_container_width=True, hide_index=True)

    with tab_bottom:
        bot_df = df.tail(top_n).iloc[::-1].copy()
        for col in ["ret_1M", "ret_3M", "ret_6M", "ret_12M", "momentum_score"]:
            if col in bot_df.columns:
                bot_df[col] = bot_df[col].apply(_fmt_pct if col != "momentum_score" else lambda v: f"{v:.1f}")
        show_cols = ["ticker", "company_name", "sector", "ret_1M", "ret_3M", "ret_6M", "ret_12M", "momentum_score"]
        show_cols = [c for c in show_cols if c in bot_df.columns]
        st.dataframe(bot_df[show_cols], use_container_width=True, hide_index=True)

    with tab_period:
        p1, p2 = st.columns(2)
        with p1:
            st.plotly_chart(
                _bar_chart(df, "ret_1M", "1-Month Return — Top 20", top_n=20),
                use_container_width=True,
            )
            st.plotly_chart(
                _bar_chart(df, "ret_6M", "6-Month Return — Top 20", top_n=20),
                use_container_width=True,
            )
        with p2:
            st.plotly_chart(
                _bar_chart(df, "ret_3M", "3-Month Return — Top 20", top_n=20),
                use_container_width=True,
            )
            st.plotly_chart(
                _bar_chart(df, "ret_12M", "12-Month Return — Top 20", top_n=20),
                use_container_width=True,
            )

    with tab_dd:
        dd_period = st.selectbox(
            "Period", ["1M", "3M", "6M", "12M"], index=1, key="dd_period"
        )
        dd_col = f"maxdd_{dd_period}"
        dd_df = df.dropna(subset=[dd_col]).nsmallest(top_n, dd_col).copy()
        dd_df["ret_col"] = dd_df[f"ret_{dd_period}"]

        colors = ["#e74c3c" if v <= 0 else "#2ecc71" for v in dd_df[dd_col]]
        fig_dd = go.Figure(go.Bar(
            x=dd_df[dd_col],
            y=dd_df["ticker"],
            orientation="h",
            marker_color=colors,
            text=[_fmt_pct(v) for v in dd_df[dd_col]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Max Drawdown: %{x:+.1f}%<extra></extra>",
        ))
        fig_dd.update_layout(
            title=dict(text=f"{dd_period} Max Drawdown — Worst {top_n}", x=0, font=dict(size=13, color="white")),
            height=max(300, top_n * 22),
            margin=dict(t=40, b=10, l=70, r=60),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=True,
                       zerolinecolor="rgba(255,255,255,0.3)", ticksuffix="%"),
            yaxis=dict(showgrid=False, autorange="reversed"),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        show_dd = dd_df[["ticker", "company_name", "sector", f"ret_{dd_period}", dd_col]].copy()
        show_dd[f"ret_{dd_period}"] = show_dd[f"ret_{dd_period}"].apply(_fmt_pct)
        show_dd[dd_col] = show_dd[dd_col].apply(_fmt_pct)
        show_dd = show_dd.rename(columns={dd_col: f"max_drawdown_{dd_period}"})
        st.dataframe(show_dd, use_container_width=True, hide_index=True)

    with tab_resilience:
        st.markdown(
            "**Which stocks held up (or went up) when the market sold off hard?**  \n"
            "Every day where the market benchmark dropped past the threshold counts as a "
            "\"selloff day\". Each stock is ranked by how it behaved on those exact days."
        )

        rc1, rc2 = st.columns([1, 1])
        with rc1:
            lookback = st.selectbox("Lookback window", ["1M (30d)", "3M (91d)", "6M (182d)", "12M (365d)"],
                                    index=1, key="res_lookback")
        with rc2:
            threshold = st.slider(
                "Market drop threshold (%)",
                min_value=-5.0, max_value=-0.5, value=-1.5, step=0.25,
                help="Days where market fell by at least this % are counted as selloff days.",
                key="res_threshold",
            )

        lookback_map = {"1M (30d)": 30, "3M (91d)": 91, "6M (182d)": 182, "12M (365d)": 365}
        ldays = lookback_map[lookback]

        with st.spinner("Analysing selloff days…"):
            res_df, selloff_cal, mkt_label = compute_selloff_resilience(
                lookback_days=ldays, drop_threshold=threshold
            )

        if res_df.empty:
            st.info(
                f"No market selloff days found where {mkt_label} dropped ≤ {threshold:.1f}% "
                f"in the past {ldays} days. Try loosening the threshold."
            )
        else:
            # Apply sector filter if active
            if sector_filter != "All" and "sector" in res_df.columns:
                res_df = res_df[res_df["sector"] == sector_filter]

            n_days = int(res_df["selloff_days_n"].iloc[0]) if not res_df.empty else 0
            sm1, sm2, sm3, sm4 = st.columns(4)
            with sm1:
                st.metric("Selloff days identified", len(selloff_cal))
            with sm2:
                st.metric("Market benchmark", mkt_label)
            with sm3:
                st.metric("Threshold", f"≤ {threshold:.1f}%")
            with sm4:
                worst = float(selloff_cal["market_return_%"].min()) if not selloff_cal.empty else 0
                st.metric("Worst day", _fmt_pct(worst))

            with st.expander(f"Selloff day calendar ({len(selloff_cal)} days)"):
                cal_show = selloff_cal.copy()
                cal_show["market_return_%"] = cal_show["market_return_%"].apply(_fmt_pct)
                st.dataframe(cal_show, use_container_width=True, hide_index=True)

            st.divider()

            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(
                    _resilience_chart(
                        res_df, "avg_selloff_ret_%",
                        f"Avg return on selloff days — Top {top_n} most resilient",
                        top_n,
                    ),
                    use_container_width=True,
                )
            with ch2:
                st.plotly_chart(
                    _resilience_chart(
                        res_df, "vs_market_%",
                        f"Outperformance vs {mkt_label} on selloff days — Top {top_n}",
                        top_n,
                    ),
                    use_container_width=True,
                )

            ch3, ch4 = st.columns(2)
            with ch3:
                st.plotly_chart(
                    _resilience_chart(
                        res_df.dropna(subset=["down_beta"]), "down_beta",
                        f"Downside Beta (lowest = safest) — Top {top_n}",
                        top_n, ascending=True,
                    ),
                    use_container_width=True,
                )
            with ch4:
                st.plotly_chart(
                    _resilience_chart(
                        res_df, "pct_days_positive_%",
                        f"% of selloff days stock was flat/up — Top {top_n}",
                        top_n,
                    ),
                    use_container_width=True,
                )

            st.divider()
            st.subheader("Full resilience table")
            st.caption(
                "avg_selloff_ret: avg daily return on market selloff days.  "
                "vs_market: outperformance vs benchmark on those days.  "
                "down_beta: sensitivity to market drops (< 1 = defensive, < 0 = inverse).  "
                "worst_single_day: tail risk — worst single day during a selloff.  "
                "pct_days_positive: how often the stock held flat or gained on selloff days."
            )
            table_res = res_df[["ticker", "company_name", "sector",
                                "avg_selloff_ret_%", "vs_market_%", "down_beta",
                                "worst_single_day_%", "pct_days_positive_%", "selloff_days_n"]].copy()
            for col in ["avg_selloff_ret_%", "vs_market_%", "worst_single_day_%"]:
                table_res[col] = table_res[col].apply(_fmt_pct)
            table_res["down_beta"] = table_res["down_beta"].apply(
                lambda v: f"{v:.2f}" if pd.notna(v) else "—"
            )
            table_res["pct_days_positive_%"] = table_res["pct_days_positive_%"].apply(
                lambda v: f"{v:.0f}%" if pd.notna(v) else "—"
            )
            st.dataframe(table_res, use_container_width=True, hide_index=True)

    with tab_table:
        st.caption(f"All {len(df):,} stocks sorted by composite momentum score (descending).")
        full_cols = ["ticker", "company_name", "sector", "ret_1M", "ret_3M", "ret_6M", "ret_12M",
                     "maxdd_1M", "maxdd_3M", "maxdd_6M", "maxdd_12M", "momentum_score", "market_cap"]
        show = df[[c for c in full_cols if c in df.columns]].copy()
        for col in ["ret_1M", "ret_3M", "ret_6M", "ret_12M", "maxdd_1M", "maxdd_3M", "maxdd_6M", "maxdd_12M"]:
            if col in show.columns:
                show[col] = show[col].apply(_fmt_pct)
        show["momentum_score"] = show["momentum_score"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
        show["market_cap"] = show["market_cap"].apply(
            lambda v: f"{v/1e9:,.1f}B" if pd.notna(v) and v else "—"
        )
        st.dataframe(show, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
