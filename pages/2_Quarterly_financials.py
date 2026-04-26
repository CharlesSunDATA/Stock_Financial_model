"""
Quarterly financials: any ticker, live yfinance pull only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import fetch_quarterly_metrics


def _fmt_billions(v: float) -> str:
    """Human-readable axis label: 35B, 1.2B, 450M …"""
    if pd.isna(v):
        return "—"
    abs_v = abs(v)
    if abs_v >= 1e9:
        return f"{v/1e9:.1f}B"
    if abs_v >= 1e6:
        return f"{v/1e6:.0f}M"
    return f"{v:,.0f}"


def _bar_colors(series: pd.Series, pos: str = "#2ecc71", neg: str = "#e74c3c") -> list[str]:
    return [pos if v >= 0 else neg for v in series.fillna(0)]

DISPLAY_COLS: list[tuple[str, str]] = [
    ("period_end", "Period end"),
    ("total_revenue", "Revenue (USD)"),
    ("revenue_qoq_pct", "Rev. QoQ %"),
    ("revenue_yoy_pct", "Rev. YoY %"),
    ("gross_margin_pct", "Gross margin %"),
    ("operating_margin_pct", "Operating margin %"),
    ("net_margin_pct", "Net margin %"),
    ("ebitda", "EBITDA (USD)"),
    ("diluted_eps", "Diluted EPS"),
    ("eps_yoy_pct", "EPS YoY %"),
    ("operating_cash_flow", "Operating CF (USD)"),
    ("free_cash_flow", "FCF (USD)"),
    ("fcf_margin_pct", "FCF margin %"),
    ("rd_pct_revenue", "R&D % rev."),
    ("sga_pct_revenue", "SG&A % rev."),
    ("total_debt", "Total debt (USD)"),
    ("cash_and_equivalents", "Cash (USD)"),
    ("net_debt", "Net debt (USD)"),
    ("debt_to_equity", "Debt / equity"),
    ("effective_tax_rate_pct", "Eff. tax rate %"),
]


def _fmt_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "period_end" in out.columns:
        out["period_end"] = pd.to_datetime(out["period_end"]).dt.strftime("%Y-%m-%d")
    money_cols = {
        "total_revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "ebitda",
        "rd_expense",
        "sga_expense",
        "operating_cash_flow",
        "capex",
        "free_cash_flow",
        "total_debt",
        "cash_and_equivalents",
        "total_equity",
        "net_debt",
        "pretax_income",
        "interest_expense",
        "tax_provision",
    }
    pct_cols = {
        "gross_margin_pct",
        "operating_margin_pct",
        "net_margin_pct",
        "fcf_margin_pct",
        "rd_pct_revenue",
        "sga_pct_revenue",
        "revenue_qoq_pct",
        "revenue_yoy_pct",
        "eps_yoy_pct",
        "effective_tax_rate_pct",
    }
    for c in out.columns:
        if c in money_cols:
            out[c] = out[c].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
        elif c in pct_cols:
            out[c] = out[c].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        elif c == "diluted_eps":
            out[c] = out[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        elif c == "debt_to_equity":
            out[c] = out[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    return out


def main() -> None:
    st.title("Quarterly financials")
    st.caption(
        "Live data from Yahoo Finance (yfinance). Use the sidebar to set ticker and refresh. "
        "We show **every quarter-end** Yahoo returns across income, cash flow, and balance sheet (often "
        "only a handful of periods); older rows may lack revenue while leverage or other fields still "
        "appear. Statement line names are normalized so both spaced and CamelCase Yahoo labels map correctly. "
        "Margins, growth, and leverage are analyst-style shortcuts — not investment advice."
    )

    with st.sidebar:
        st.header("Ticker & refresh")
        ticker = st.text_input("Ticker symbol", value="NVDA").strip().upper() or "NVDA"
        if st.button("Refresh from Yahoo", type="primary", width="stretch"):
            st.session_state["fin_pull"] = True

    do_pull = st.session_state.pop("fin_pull", False)
    if "fin_df" not in st.session_state:
        do_pull = True

    df_raw = pd.DataFrame()
    err: str | None = None
    if do_pull:
        with st.spinner(f"Downloading quarterly filings for {ticker}…"):
            try:
                df_raw = fetch_quarterly_metrics(ticker)
                if df_raw.empty:
                    err = f"No quarterly financials returned for {ticker}."
                else:
                    st.session_state["fin_df"] = df_raw
                    st.session_state["fin_ticker"] = ticker
            except Exception as e:
                err = str(e)
    else:
        df_raw = st.session_state.get("fin_df", pd.DataFrame())
        if st.session_state.get("fin_ticker") != ticker:
            st.warning("Ticker changed — click **Refresh from Yahoo** to load the new symbol.")

    if err:
        st.error(err)
        return

    if df_raw.empty:
        st.info("Click **Refresh from Yahoo** in the sidebar to load quarterly data.")
        return

    st.subheader(f"{ticker} — key quarterly metrics (newest first)")
    show_tbl = df_raw.sort_values("period_end", ascending=False).copy()
    if "period_end" in show_tbl.columns:
        show_tbl["period_end"] = pd.to_datetime(show_tbl["period_end"])

    disp = _fmt_display(show_tbl)
    present = [c for c, _ in DISPLAY_COLS if c in disp.columns]
    disp = disp[present]

    rename = {c: lab for c, lab in DISPLAY_COLS if c in disp.columns}
    disp = disp.rename(columns=rename)
    st.dataframe(disp, width="stretch", hide_index=True)

    st.subheader("Charts")
    ch  = df_raw.sort_values("period_end", ascending=True).copy()
    x   = ch["period_end"].dt.strftime("%Y-%m-%d").tolist()

    _LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=11),
        margin=dict(t=36, b=0, l=0, r=0),
        height=280,
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified",
    )

    c1, c2 = st.columns(2)

    # ── Revenue bar (colour by QoQ) ──────────────────────────────────────────
    with c1:
        rev  = ch["total_revenue"]
        qoq  = ch.get("revenue_qoq_pct", pd.Series([np.nan] * len(ch)))
        cols = _bar_colors(rev, pos="#3498db", neg="#e74c3c")
        fig_rev = go.Figure(go.Bar(
            x=x, y=rev,
            marker_color=cols,
            text=[_fmt_billions(v) for v in rev],
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate="<b>%{x}</b><br>Revenue: %{text}<extra></extra>",
            name="Revenue",
        ))
        fig_rev.update_layout(**_LAYOUT,
            title=dict(text="Revenue (USD)", font=dict(size=13), x=0),
            yaxis_tickformat=".2s",
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    # ── Margins area-line chart ───────────────────────────────────────────────
    with c2:
        fig_mg = go.Figure()
        for col, color, fill, name in [
            ("gross_margin_pct",     "#2ecc71", "rgba(46,204,113,0.12)",  "Gross"),
            ("operating_margin_pct", "#3498db", "rgba(52,152,219,0.12)",  "Operating"),
            ("net_margin_pct",       "#e74c3c", "rgba(231,76,60,0.12)",   "Net"),
        ]:
            y = ch[col] if col in ch.columns else [None] * len(ch)
            fig_mg.add_trace(go.Scatter(
                x=x, y=y, name=name,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor=fill,
                hovertemplate=f"<b>{name}</b>: %{{y:.1f}}%<extra></extra>",
            ))
        fig_mg.update_layout(**_LAYOUT,
            title=dict(text="Margins (% of revenue)", font=dict(size=13), x=0),
            yaxis_ticksuffix="%",
        )
        st.plotly_chart(fig_mg, use_container_width=True)

    # ── Free cash flow bars (green/red) ──────────────────────────────────────
    fcf = ch["free_cash_flow"] if "free_cash_flow" in ch.columns else pd.Series([np.nan] * len(ch))
    fig_fcf = go.Figure(go.Bar(
        x=x, y=fcf,
        marker_color=_bar_colors(fcf),
        text=[_fmt_billions(v) for v in fcf],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{x}</b><br>FCF: %{text}<extra></extra>",
    ))
    fig_fcf.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    fig_fcf.update_layout(**_LAYOUT,
        title=dict(text="Free Cash Flow (USD)", font=dict(size=13), x=0),
        yaxis_tickformat=".2s",
    )
    st.plotly_chart(fig_fcf, use_container_width=True)

    # ── EPS trend line ───────────────────────────────────────────────────────
    if "diluted_eps" in ch.columns:
        eps = ch["diluted_eps"]
        fig_eps = go.Figure(go.Scatter(
            x=x, y=eps,
            mode="lines+markers+text",
            line=dict(color="#f39c12", width=2),
            marker=dict(size=7, color=_bar_colors(eps, "#f39c12", "#e74c3c"),
                        line=dict(color="white", width=1)),
            text=[f"{v:.2f}" if pd.notna(v) else "" for v in eps],
            textposition="top center",
            textfont=dict(size=10),
            fill="tozeroy",
            fillcolor="rgba(243,156,18,0.12)",
            hovertemplate="<b>%{x}</b><br>Diluted EPS: $%{y:.2f}<extra></extra>",
        ))
        fig_eps.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
        fig_eps.update_layout(**{**_LAYOUT, "height": 220},
            title=dict(text="Diluted EPS (USD)", font=dict(size=13), x=0),
            yaxis_tickprefix="$",
        )
        st.plotly_chart(fig_eps, use_container_width=True)

    st.caption(
        "🟢 Green bars = positive  ·  🔴 Red bars = negative  ·  "
        "Margins chart: filled area lines for Gross / Operating / Net.  "
        "Some periods may be blank if Yahoo does not provide the line items."
    )


main()

