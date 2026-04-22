"""
Quarterly financials: any ticker, live yfinance pull only.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pipeline import fetch_quarterly_metrics

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
    ch = df_raw.sort_values("period_end", ascending=True)
    x = ch["period_end"].dt.strftime("%Y-%m-%d")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Revenue (USD)**")
        rdf = pd.DataFrame({"Period": x.values, "Revenue": ch["total_revenue"].values})
        st.bar_chart(rdf.set_index("Period"))
    with c2:
        st.markdown("**Margins (% of revenue)**")
        mdf = pd.DataFrame(
            {
                "Period": x.values,
                "Gross": ch["gross_margin_pct"].values,
                "Operating": ch["operating_margin_pct"].values,
                "Net": ch["net_margin_pct"].values,
            }
        )
        st.line_chart(mdf.set_index("Period"))

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Operating margin (% of revenue)**")
        omdf = pd.DataFrame({"Period": x.values, "Operating margin %": ch["operating_margin_pct"].values})
        st.bar_chart(omdf.set_index("Period"))
    with c4:
        st.markdown("**Free cash flow (USD)**")
        fdf = pd.DataFrame({"Period": x.values, "FCF": ch["free_cash_flow"].values})
        st.bar_chart(fdf.set_index("Period"))

    st.caption(
        "Operating margin is operating income as a percent of revenue. "
        "Some periods may be blank if Yahoo does not provide the line items for that quarter."
    )


main()

