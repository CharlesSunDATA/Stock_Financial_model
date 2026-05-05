"""Smart Money — insider trading & institutional ownership signals."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.local_data import connect_readonly, table_exists


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "quant_data.db"

_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
)
_GRID = dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)")


# ── data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60 * 15, show_spinner=False)
def _load_insider_purchases(db_path: str, days: int, min_value: float) -> pd.DataFrame:
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with connect_readonly(Path(db_path)) as conn:
        if not table_exists(conn, "insider_trading"):
            return pd.DataFrame()
        df = pd.read_sql_query(
            """
            SELECT ticker, filing_date, insider_name, transaction_type,
                   shares, price, payload_json
            FROM insider_trading
            WHERE transaction_type = 'P-Purchase'
              AND filing_date >= ?
              AND shares IS NOT NULL
              AND price IS NOT NULL
            ORDER BY filing_date DESC
            """,
            conn,
            params=(cutoff,),
        )
    if df.empty:
        return df

    df["value"] = df["shares"] * df["price"]
    df = df[df["value"] >= min_value].copy()

    owner_types: list[str] = []
    txn_dates: list[str] = []
    for p_str in df["payload_json"]:
        if p_str:
            try:
                p = json.loads(p_str)
                owner_types.append(str(p.get("typeOfOwner", "")))
                txn_dates.append(str(p.get("transactionDate", "")))
                continue
            except Exception:
                pass
        owner_types.append("")
        txn_dates.append("")

    df["owner_type"] = owner_types
    df["transaction_date"] = txn_dates
    df = df.drop(columns=["payload_json"])
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    return df.sort_values("value", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=60 * 15, show_spinner=False)
def _load_buy_sell_ratio(db_path: str, days: int) -> pd.DataFrame:
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with connect_readonly(Path(db_path)) as conn:
        if not table_exists(conn, "insider_trading"):
            return pd.DataFrame()
        df = pd.read_sql_query(
            """
            SELECT ticker, transaction_type, shares, price
            FROM insider_trading
            WHERE transaction_type IN ('P-Purchase', 'S-Sale')
              AND filing_date >= ?
              AND shares IS NOT NULL
              AND price IS NOT NULL
            """,
            conn,
            params=(cutoff,),
        )
    if df.empty:
        return df

    df["value"] = df["shares"] * df["price"]
    buys = df[df["transaction_type"] == "P-Purchase"].groupby("ticker")["value"].sum().rename("buy_value")
    sells = df[df["transaction_type"] == "S-Sale"].groupby("ticker")["value"].sum().rename("sell_value")
    buy_cnt = df[df["transaction_type"] == "P-Purchase"].groupby("ticker").size().rename("buy_count")
    sell_cnt = df[df["transaction_type"] == "S-Sale"].groupby("ticker").size().rename("sell_count")

    agg = pd.concat([buys, sells, buy_cnt, sell_cnt], axis=1).fillna(0).reset_index()
    agg["total_value"] = agg["buy_value"] + agg["sell_value"]
    agg["buy_ratio"] = agg.apply(
        lambda r: r["buy_value"] / r["total_value"] * 100 if r["total_value"] > 0 else None,
        axis=1,
    )
    return agg[agg["total_value"] > 0].sort_values("buy_value", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=60 * 15, show_spinner=False)
def _load_institutional_changes(db_path: str) -> pd.DataFrame:
    """Latest-period aggregate institutional ownership per ticker (from 13-F payload)."""
    with connect_readonly(Path(db_path)) as conn:
        if not table_exists(conn, "institutional_ownership"):
            return pd.DataFrame()
        df = pd.read_sql_query(
            """
            SELECT ticker, as_of_date, payload_json
            FROM institutional_ownership
            """,
            conn,
        )
    if df.empty:
        return df

    records: list[dict] = []
    for _, row in df.iterrows():
        rec: dict = {"ticker": row["ticker"], "as_of_date": row["as_of_date"]}
        if row["payload_json"]:
            try:
                p = json.loads(row["payload_json"])
                rec["ownership_pct"] = p.get("ownershipPercent")
                rec["prev_ownership_pct"] = p.get("lastOwnershipPercent")
                rec["ownership_pct_change"] = p.get("ownershipPercentChange")
                rec["investors_holding"] = p.get("investorsHolding")
                rec["investors_change"] = p.get("investorsHoldingChange")
                rec["new_positions"] = p.get("newPositions")
                rec["increased_positions"] = p.get("increasedPositions")
                rec["reduced_positions"] = p.get("reducedPositions")
                rec["closed_positions"] = p.get("closedPositions")
                rec["total_invested"] = p.get("totalInvested")
                rec["total_invested_change"] = p.get("totalInvestedChange")
            except Exception:
                pass
        records.append(rec)

    out = pd.DataFrame(records)
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
    # One row per ticker: keep latest filing date
    out = (
        out.sort_values("as_of_date", ascending=False)
        .drop_duplicates(subset=["ticker"])
        .reset_index(drop=True)
    )
    return out


# ── charts ────────────────────────────────────────────────────────────────────

def _bar_purchases(df: pd.DataFrame, top_n: int) -> go.Figure:
    sub = df.head(top_n).iloc[::-1].copy()
    sub["label"] = sub["ticker"] + "  " + sub["insider_name"].str.split().str[0].fillna("")
    fig = go.Figure(
        go.Bar(
            x=sub["value"],
            y=sub["label"],
            orientation="h",
            marker_color="#2ecc71",
            text=sub["value"].apply(_fmt_val),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Value: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(360, top_n * 30),
        margin=dict(t=10, b=10, l=210, r=90),
        **_DARK,
        xaxis=dict(**_GRID),
        yaxis=dict(showgrid=False),
    )
    return fig


def _bar_buy_ratio(df: pd.DataFrame, top_n: int) -> go.Figure:
    sub = df.dropna(subset=["buy_ratio"]).head(top_n).iloc[::-1].copy()
    colors = ["#2ecc71" if v >= 80 else "#f1c40f" if v >= 50 else "#e74c3c" for v in sub["buy_ratio"]]
    fig = go.Figure(
        go.Bar(
            x=sub["buy_ratio"],
            y=sub["ticker"],
            orientation="h",
            marker_color=colors,
            text=sub["buy_ratio"].apply(lambda v: f"{v:.0f}%"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Buy ratio: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(360, top_n * 30),
        margin=dict(t=10, b=10, l=80, r=70),
        **_DARK,
        xaxis=dict(range=[0, 115], **_GRID),
        yaxis=dict(showgrid=False),
    )
    return fig


def _bar_inst(df: pd.DataFrame, top_n: int, ascending: bool) -> go.Figure:
    col = "ownership_pct_change"
    sub = df.dropna(subset=[col]).sort_values(col, ascending=ascending).head(top_n).iloc[::-1].copy()
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in sub[col]]
    fig = go.Figure(
        go.Bar(
            x=sub[col],
            y=sub["ticker"],
            orientation="h",
            marker_color=colors,
            text=sub[col].apply(lambda v: f"{v:+.2f}%"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Δ Own %: %{x:+.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(360, top_n * 30),
        margin=dict(t=10, b=10, l=80, r=90),
        **_DARK,
        xaxis=dict(**_GRID),
        yaxis=dict(showgrid=False),
    )
    return fig


# ── formatters ────────────────────────────────────────────────────────────────

def _fmt_val(v) -> str:
    if pd.isna(v):
        return "—"
    x = float(v)
    ax = abs(x)
    if ax >= 1e9:
        return f"${x / 1e9:.1f}B"
    if ax >= 1e6:
        return f"${x / 1e6:.1f}M"
    if ax >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:,.0f}"


def _fmt_pct(v) -> str:
    if pd.isna(v):
        return "—"
    return f"{float(v):+.2f}%"


# ── page ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("Smart Money")
    st.caption(
        "Insider open-market purchases (SEC Form 4) and institutional 13-F ownership signals."
    )

    if not DB_PATH.exists():
        st.error(f"Database not found: `{DB_PATH}`")
        return

    with st.sidebar:
        st.header("Filters")
        days = st.selectbox(
            "Lookback window",
            [30, 60, 90, 180],
            index=2,
            format_func=lambda d: f"{d} days",
        )
        min_value = st.selectbox(
            "Min transaction value",
            [0, 10_000, 50_000, 100_000, 500_000],
            index=2,
            format_func=lambda v: "Any" if v == 0 else f"${v:,}",
        )
        top_n = st.slider("Top N", 10, 60, 25, step=5)
        min_inst_change = st.selectbox(
            "Min institutional ownership change",
            [0, 1, 5, 10, 25],
            index=1,
            format_func=lambda v: "Any" if v == 0 else f"{v}%+",
        )
        st.divider()
        if st.button("Refresh cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    tab_buys, tab_ratio, tab_inst = st.tabs(
        ["Insider Purchases", "Buy / Sell Ratio", "Institutional Flows"]
    )

    # ── Tab 1: Insider Purchases ──────────────────────────────────────────────
    with tab_buys:
        df_buys = _load_insider_purchases(str(DB_PATH), days, min_value)

        if df_buys.empty:
            st.info("No insider purchase data. Run `scripts/update_compliance_fmp.py` to backfill.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Transactions", f"{len(df_buys):,}")
            c2.metric("Unique tickers", f"{df_buys['ticker'].nunique():,}")
            c3.metric("Total value", _fmt_val(df_buys["value"].sum()))

            left, right = st.columns([1.3, 1])
            with left:
                st.plotly_chart(_bar_purchases(df_buys, top_n), use_container_width=True)
            with right:
                show = df_buys.head(top_n)[
                    ["ticker", "filing_date", "insider_name", "owner_type", "shares", "price", "value"]
                ].copy()
                show["filing_date"] = show["filing_date"].dt.strftime("%Y-%m-%d")
                show["value"] = show["value"].apply(_fmt_val)
                show["shares"] = show["shares"].apply(
                    lambda v: f"{int(v):,}" if pd.notna(v) else "—"
                )
                show["price"] = show["price"].apply(
                    lambda v: f"${v:.2f}" if pd.notna(v) else "—"
                )
                show = show.rename(
                    columns={
                        "ticker": "Ticker",
                        "filing_date": "Filed",
                        "insider_name": "Insider",
                        "owner_type": "Role",
                        "shares": "Shares",
                        "price": "Price",
                        "value": "Value",
                    }
                )
                st.dataframe(show, use_container_width=True, hide_index=True)

            st.download_button(
                "Download CSV",
                data=df_buys.to_csv(index=False).encode("utf-8"),
                file_name="insider_purchases.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── Tab 2: Buy / Sell Ratio ───────────────────────────────────────────────
    with tab_ratio:
        df_ratio = _load_buy_sell_ratio(str(DB_PATH), days)

        if df_ratio.empty:
            st.info("No insider transaction data for this window.")
        else:
            st.caption(
                "Buy ratio = open-market purchases ÷ (purchases + sales) by dollar value. "
                "Stock awards, option exercises, and tax withholding are excluded."
            )

            # Split into pure buyers vs mixed
            pure = df_ratio[df_ratio["sell_value"] == 0].sort_values("buy_value", ascending=False)
            mixed = df_ratio[df_ratio["sell_value"] > 0].sort_values("buy_ratio", ascending=False)

            left, right = st.columns(2)
            with left:
                st.subheader("Mixed (buys + sells)")
                if mixed.empty:
                    st.info("No tickers with both sides in this window.")
                else:
                    st.plotly_chart(_bar_buy_ratio(mixed, top_n), use_container_width=True)
                    show = mixed.head(top_n)[
                        ["ticker", "buy_ratio", "buy_value", "sell_value", "buy_count", "sell_count"]
                    ].copy()
                    show["buy_ratio"] = show["buy_ratio"].apply(
                        lambda v: f"{v:.1f}%" if pd.notna(v) else "—"
                    )
                    show["buy_value"] = show["buy_value"].apply(_fmt_val)
                    show["sell_value"] = show["sell_value"].apply(_fmt_val)
                    show["buy_count"] = show["buy_count"].astype(int)
                    show["sell_count"] = show["sell_count"].astype(int)
                    show = show.rename(
                        columns={
                            "ticker": "Ticker",
                            "buy_ratio": "Buy %",
                            "buy_value": "Buy $",
                            "sell_value": "Sell $",
                            "buy_count": "# Buys",
                            "sell_count": "# Sells",
                        }
                    )
                    st.dataframe(show, use_container_width=True, hide_index=True)

            with right:
                st.subheader("Buy-only tickers")
                if pure.empty:
                    st.info("No buy-only tickers in this window.")
                else:
                    show_pure = pure.head(top_n)[["ticker", "buy_value", "buy_count"]].copy()
                    show_pure["buy_value"] = show_pure["buy_value"].apply(_fmt_val)
                    show_pure["buy_count"] = show_pure["buy_count"].astype(int)
                    show_pure = show_pure.rename(
                        columns={
                            "ticker": "Ticker",
                            "buy_value": "Total Buy $",
                            "buy_count": "Transactions",
                        }
                    )
                    st.dataframe(show_pure, use_container_width=True, hide_index=True)

            st.download_button(
                "Download CSV",
                data=df_ratio.to_csv(index=False).encode("utf-8"),
                file_name="insider_buy_sell_ratio.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── Tab 3: Institutional Flows ────────────────────────────────────────────
    with tab_inst:
        df_inst = _load_institutional_changes(str(DB_PATH))

        if df_inst.empty:
            st.info(
                "No institutional ownership data. Run `scripts/update_compliance_fmp.py` to backfill."
            )
        else:
            st.caption(
                "Source: FMP 13-F institutional ownership. "
                "Δ Own % compares current filing period vs prior period."
            )
            if min_inst_change > 0:
                df_inst = df_inst[
                    df_inst["ownership_pct_change"].abs() >= float(min_inst_change)
                ].copy()

            if df_inst.empty:
                st.info("No institutional ownership changes match the selected threshold.")
                return

            m1, m2, m3 = st.columns(3)
            m1.metric("Tracked tickers", f"{df_inst['ticker'].nunique():,}")
            m2.metric("Ownership increases", f"{int((df_inst['ownership_pct_change'] > 0).sum()):,}")
            m3.metric("Ownership decreases", f"{int((df_inst['ownership_pct_change'] < 0).sum()):,}")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Biggest increases")
                st.plotly_chart(
                    _bar_inst(df_inst, min(top_n, 30), ascending=False),
                    use_container_width=True,
                )
            with c2:
                st.subheader("Biggest decreases")
                st.plotly_chart(
                    _bar_inst(df_inst, min(top_n, 30), ascending=True),
                    use_container_width=True,
                )

            st.divider()

            show = (
                df_inst.dropna(subset=["ownership_pct_change"])
                .sort_values("ownership_pct_change", ascending=False)
                .head(top_n * 2)
                .copy()
            )
            show["as_of_date"] = show["as_of_date"].dt.strftime("%Y-%m-%d")
            for col in ["ownership_pct", "prev_ownership_pct", "ownership_pct_change"]:
                show[col] = show[col].apply(_fmt_pct)
            show["total_invested"] = show["total_invested"].apply(_fmt_val)
            show["total_invested_change"] = show["total_invested_change"].apply(_fmt_val)

            show = show[
                [
                    "ticker", "as_of_date",
                    "ownership_pct", "prev_ownership_pct", "ownership_pct_change",
                    "investors_holding", "investors_change",
                    "new_positions", "increased_positions",
                    "reduced_positions", "closed_positions",
                    "total_invested", "total_invested_change",
                ]
            ].rename(
                columns={
                    "ticker": "Ticker",
                    "as_of_date": "As-of Date",
                    "ownership_pct": "Own %",
                    "prev_ownership_pct": "Prev Own %",
                    "ownership_pct_change": "Δ Own %",
                    "investors_holding": "# Holders",
                    "investors_change": "Holders Δ",
                    "new_positions": "New",
                    "increased_positions": "Increased",
                    "reduced_positions": "Reduced",
                    "closed_positions": "Closed",
                    "total_invested": "Total Invested",
                    "total_invested_change": "Invested Δ",
                }
            )
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.download_button(
                "Download CSV",
                data=df_inst.to_csv(index=False).encode("utf-8"),
                file_name="institutional_flows.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
