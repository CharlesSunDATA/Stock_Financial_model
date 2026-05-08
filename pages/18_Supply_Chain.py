"""
Supply Chain Explorer (Streamlit page)

Interactive tool for exploring tech company supply chains:
- Enter any ticker to see its upstream suppliers, downstream customers, and peers
- Click any company card to navigate to that company's profile
- Financial summary pulled from local SQLite DB
- Supply chain data from data/supply_chain.json (curated) + DB suppliers table
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.local_data import connect_readonly

# ── paths ──────────────────────────────────────────────────────────────────────

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _db_path() -> Path:
    return _project_root() / "data" / "quant_data.db"

def _sc_json_path() -> Path:
    return _project_root() / "data" / "supply_chain.json"

# ── data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_supply_chain_map() -> dict:
    path = _sc_json_path()
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    data.pop("_meta", None)
    return data


@st.cache_data(ttl=300, show_spinner=False)
def load_company_profile(ticker: str) -> dict:
    db = _db_path()
    if not db.exists():
        return {}
    sym = ticker.strip().upper()
    try:
        with connect_readonly(db) as conn:
            row = conn.execute(
                "SELECT company_name, sector, industry, full_time_employees, image_url "
                "FROM company_profile WHERE ticker = ? LIMIT 1",
                (sym,),
            ).fetchone()
        if row:
            return {
                "company_name": row[0] or sym,
                "sector": row[1] or "—",
                "industry": row[2] or "—",
                "employees": row[3],
                "image_url": row[4],
            }
    except Exception:
        pass
    return {"company_name": sym, "sector": "—", "industry": "—", "employees": None, "image_url": None}


@st.cache_data(ttl=300, show_spinner=False)
def load_latest_price(ticker: str) -> float | None:
    db = _db_path()
    if not db.exists():
        return None
    sym = ticker.strip().upper()
    try:
        with connect_readonly(db) as conn:
            row = conn.execute(
                "SELECT close FROM prices_eod WHERE ticker = ? ORDER BY price_date DESC LIMIT 1",
                (sym,),
            ).fetchone()
        return float(row[0]) if row else None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def load_market_cap(ticker: str) -> float | None:
    db = _db_path()
    if not db.exists():
        return None
    sym = ticker.strip().upper()
    try:
        with connect_readonly(db) as conn:
            row = conn.execute(
                "SELECT market_cap FROM key_metrics_ttm WHERE ticker = ? ORDER BY as_of_date DESC LIMIT 1",
                (sym,),
            ).fetchone()
        return float(row[0]) if row else None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def load_income_history(ticker: str, periods: int = 8) -> pd.DataFrame:
    db = _db_path()
    if not db.exists():
        return pd.DataFrame()
    sym = ticker.strip().upper()
    try:
        with connect_readonly(db) as conn:
            df = pd.read_sql_query(
                """
                SELECT report_date, period, revenue, gross_profit, operating_income,
                       net_income, eps_diluted
                FROM income_statement
                WHERE ticker = ?
                ORDER BY report_date DESC
                LIMIT ?
                """,
                conn,
                params=(sym, periods),
            )
        return df.iloc[::-1].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_valuation_metrics(ticker: str) -> dict:
    db = _db_path()
    if not db.exists():
        return {}
    sym = ticker.strip().upper()
    try:
        with connect_readonly(db) as conn:
            row = conn.execute(
                """SELECT market_cap, enterprise_value, ev_to_sales, ev_to_free_cash_flow
                   FROM key_metrics_ttm WHERE ticker = ? ORDER BY as_of_date DESC LIMIT 1""",
                (sym,),
            ).fetchone()
            # P/E, P/S, P/B from historical_financial_ratios payload
            ratio_row = conn.execute(
                "SELECT payload_json FROM historical_financial_ratios WHERE ticker = ? ORDER BY report_date DESC LIMIT 1",
                (sym,),
            ).fetchone()
        result = {
            "market_cap": row[0] if row else None,
            "ev":         row[1] if row else None,
            "ev_sales":   row[2] if row else None,
            "ev_fcf":     row[3] if row else None,
            "pe_ratio": None, "ps_ratio": None, "pb_ratio": None,
        }
        if ratio_row and ratio_row[0]:
            try:
                p = json.loads(ratio_row[0])
                result["pe_ratio"] = p.get("priceToEarningsRatio")
                result["ps_ratio"] = p.get("priceToSalesRatio")
                result["pb_ratio"] = p.get("priceToBookRatio")
            except Exception:
                pass
        return result
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_db_customers(ticker: str) -> list[dict]:
    """Load ticker's customers from the suppliers table (ticker is the customer)."""
    db = _db_path()
    if not db.exists():
        return []
    sym = ticker.strip().upper()
    try:
        with connect_readonly(db) as conn:
            rows = conn.execute(
                """SELECT ticker, supplier_name, payload_json
                   FROM suppliers WHERE supplier_symbol = ?
                   ORDER BY ticker LIMIT 20""",
                (sym,),
            ).fetchall()
        results = []
        for r in rows:
            cust = r[0]
            if len(cust) > 6 or not cust.isalpha():
                continue
            pct = None
            try:
                p = json.loads(r[2] or "{}")
                pct = p.get("pct_of_revenue")
            except Exception:
                pass
            results.append({"ticker": cust, "name": r[1] or cust, "pct_revenue": pct})
        return results
    except Exception:
        return []


# ── formatting helpers ─────────────────────────────────────────────────────────

def _fmt_number(val: float | None, prefix: str = "", suffix: str = "", decimals: int = 1) -> str:
    if val is None or (isinstance(val, float) and (val != val)):
        return "N/A"
    if abs(val) >= 1e12:
        return f"{prefix}{val/1e12:.{decimals}f}T{suffix}"
    if abs(val) >= 1e9:
        return f"{prefix}{val/1e9:.{decimals}f}B{suffix}"
    if abs(val) >= 1e6:
        return f"{prefix}{val/1e6:.{decimals}f}M{suffix}"
    if abs(val) >= 1e3:
        return f"{prefix}{int(val):,}{suffix}"
    return f"{prefix}{val:.{decimals}f}{suffix}"

def _fmt_ratio(val: Any, decimals: int = 1) -> str:
    try:
        return f"{float(val):.{decimals}f}x"
    except (TypeError, ValueError):
        return "N/A"

def _navigate(ticker: str) -> None:
    st.session_state["sc_ticker"] = ticker.strip().upper()
    st.session_state["sc_history"].append(ticker.strip().upper())

# ── chart helpers ──────────────────────────────────────────────────────────────

def _revenue_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        x=df["report_date"],
        y=df["revenue"] / 1e9,
        name="Revenue",
        marker_color="#4C8BF5",
    )
    if "net_income" in df.columns:
        fig.add_bar(
            x=df["report_date"],
            y=df["net_income"] / 1e9,
            name="Net Income",
            marker_color="#34A853",
        )
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="USD Billions",
        xaxis_title="",
        height=260,
    )
    return fig


def _margin_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    rev = df["revenue"].replace(0, float("nan"))
    if "gross_profit" in df.columns:
        fig.add_scatter(
            x=df["report_date"],
            y=(df["gross_profit"] / rev * 100).round(1),
            mode="lines+markers",
            name="Gross Margin %",
            line=dict(color="#4C8BF5"),
        )
    if "operating_income" in df.columns:
        fig.add_scatter(
            x=df["report_date"],
            y=(df["operating_income"] / rev * 100).round(1),
            mode="lines+markers",
            name="Op. Margin %",
            line=dict(color="#FBBC04"),
        )
    if "net_income" in df.columns:
        fig.add_scatter(
            x=df["report_date"],
            y=(df["net_income"] / rev * 100).round(1),
            mode="lines+markers",
            name="Net Margin %",
            line=dict(color="#34A853"),
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="Margin %",
        xaxis_title="",
        height=260,
    )
    return fig

# ── company card (clickable button) ───────────────────────────────────────────

def _company_card(
    ticker: str,
    name: str,
    relationship: str,
    segment: str,
    card_color: str = "#1E2A3A",
    key_prefix: str = "",
) -> None:
    """Render a compact company card with a clickable navigate button."""
    profile = load_company_profile(ticker)
    price = load_latest_price(ticker)
    display_name = profile.get("company_name") or name

    with st.container():
        st.markdown(
            f"""
            <div style="
                background:{card_color};
                border-radius:8px;
                padding:10px 12px;
                margin-bottom:6px;
                border-left:3px solid #4C8BF5;
            ">
                <div style="font-size:0.95em;font-weight:700;color:#FFFFFF;">{ticker}</div>
                <div style="font-size:0.78em;color:#B0BEC5;margin-bottom:2px;">{display_name}</div>
                <div style="font-size:0.72em;color:#90A4AE;">{segment}</div>
                <div style="font-size:0.72em;color:#78909C;font-style:italic;margin-top:2px;">{relationship}</div>
                {'<div style="font-size:0.75em;color:#80DEEA;margin-top:3px;">$' + f'{price:,.2f}' + '</div>' if price else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            f"→ Explore {ticker}",
            key=f"{key_prefix}_{ticker}_{relationship[:10]}",
            on_click=_navigate,
            args=(ticker,),
            use_container_width=True,
        )


# ── company dashboard ──────────────────────────────────────────────────────────

def _render_company(ticker: str, sc_map: dict) -> None:
    sym = ticker.strip().upper()
    profile = load_company_profile(sym)
    price = load_latest_price(sym)
    valuation = load_valuation_metrics(sym)
    income_df = load_income_history(sym)

    # ── header ──
    company_name = profile.get("company_name") or sym
    st.markdown(f"## {sym} — {company_name}")
    st.caption(
        f"**Sector:** {profile.get('sector','—')}  |  "
        f"**Industry:** {profile.get('industry','—')}  |  "
        f"**Employees:** {_fmt_number(profile.get('employees'))}"
    )

    price_str    = f"${price:,.2f}" if price else "N/A"
    mktcap_str   = _fmt_number(valuation.get("market_cap"), prefix="$")
    ev_str       = _fmt_number(valuation.get("ev"), prefix="$")
    pe           = valuation.get("pe_ratio")
    pe_str       = _fmt_ratio(pe) if pe else "N/A"
    ps_str       = _fmt_ratio(valuation.get("ps_ratio"))
    ev_sales_str = _fmt_ratio(valuation.get("ev_sales"))
    ev_fcf_str   = _fmt_ratio(valuation.get("ev_fcf"))

    st.markdown(
        f"""
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin:8px 0 12px 0;">
            <div style="background:#1A237E;border-radius:8px;padding:10px 20px;min-width:100px;text-align:center;">
                <div style="font-size:0.72em;color:#90CAF9;font-weight:600;letter-spacing:0.05em;">PRICE</div>
                <div style="font-size:1.15em;font-weight:700;color:#E3F2FD;">{price_str}</div>
            </div>
            <div style="background:#1B3A2B;border-radius:8px;padding:10px 20px;min-width:100px;text-align:center;">
                <div style="font-size:0.72em;color:#A5D6A7;font-weight:600;letter-spacing:0.05em;">MKT CAP</div>
                <div style="font-size:1.15em;font-weight:700;color:#E8F5E9;">{mktcap_str}</div>
            </div>
            <div style="background:#1B3A2B;border-radius:8px;padding:10px 20px;min-width:100px;text-align:center;">
                <div style="font-size:0.72em;color:#A5D6A7;font-weight:600;letter-spacing:0.05em;">EV</div>
                <div style="font-size:1.15em;font-weight:700;color:#E8F5E9;">{ev_str}</div>
            </div>
            <div style="background:#2D1B4E;border-radius:8px;padding:10px 20px;min-width:100px;text-align:center;">
                <div style="font-size:0.72em;color:#CE93D8;font-weight:600;letter-spacing:0.05em;">P/E TTM</div>
                <div style="font-size:1.15em;font-weight:700;color:#F3E5F5;">{pe_str}</div>
            </div>
            <div style="background:#2D1B4E;border-radius:8px;padding:10px 20px;min-width:100px;text-align:center;">
                <div style="font-size:0.72em;color:#CE93D8;font-weight:600;letter-spacing:0.05em;">P/S TTM</div>
                <div style="font-size:1.15em;font-weight:700;color:#F3E5F5;">{ps_str}</div>
            </div>
            <div style="background:#2D2B1B;border-radius:8px;padding:10px 20px;min-width:100px;text-align:center;">
                <div style="font-size:0.72em;color:#FFE082;font-weight:600;letter-spacing:0.05em;">EV/SALES</div>
                <div style="font-size:1.15em;font-weight:700;color:#FFF9C4;">{ev_sales_str}</div>
            </div>
            <div style="background:#2D2B1B;border-radius:8px;padding:10px 20px;min-width:100px;text-align:center;">
                <div style="font-size:0.72em;color:#FFE082;font-weight:600;letter-spacing:0.05em;">EV/FCF</div>
                <div style="font-size:1.15em;font-weight:700;color:#FFF9C4;">{ev_fcf_str}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── supply chain section ──
    sc_data = sc_map.get(sym, {})
    upstream = sc_data.get("upstream", [])
    downstream = sc_data.get("downstream", [])
    peers = sc_data.get("peers", [])

    # Supplement downstream with DB customer data
    db_customers = load_db_customers(sym)
    known_downstream_tickers = {d["ticker"] for d in downstream}
    for c in db_customers:
        if c["ticker"] not in known_downstream_tickers:
            downstream.append({
                "ticker": c["ticker"],
                "name": c["name"],
                "relationship": f"10-K reported customer ({c.get('pct_revenue','?')}% of revenue)" if c.get("pct_revenue") else "10-K reported customer",
                "segment": "Customer (10-K)",
            })
            known_downstream_tickers.add(c["ticker"])

    st.subheader("Supply Chain Map")
    col_up, col_center, col_down = st.columns([1, 1, 1])

    with col_up:
        st.markdown(
            "<div style='text-align:center;font-size:0.85em;font-weight:700;"
            "color:#90CAF9;letter-spacing:0.05em;margin-bottom:8px;'>⬆ UPSTREAM</div>",
            unsafe_allow_html=True,
        )
        st.caption("Suppliers / Equipment / Components")
        if upstream:
            for i, co in enumerate(upstream):
                _company_card(
                    co["ticker"], co["name"], co["relationship"], co.get("segment", ""),
                    card_color="#0D1B2A", key_prefix=f"up{i}"
                )
        else:
            st.info("No upstream data available", icon="ℹ️")

    with col_center:
        st.markdown(
            f"<div style='text-align:center;background:#1A237E;border-radius:12px;"
            f"padding:18px 8px;margin-bottom:16px;'>"
            f"<div style='font-size:1.6em;font-weight:900;color:#E3F2FD;'>{sym}</div>"
            f"<div style='font-size:0.78em;color:#90CAF9;'>{profile.get('company_name','')}</div>"
            f"<div style='font-size:0.72em;color:#64B5F6;margin-top:4px;'>{profile.get('sector','')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if peers:
            st.markdown(
                "<div style='text-align:center;font-size:0.82em;font-weight:700;"
                "color:#CE93D8;margin-bottom:6px;'>⇌ PEERS</div>",
                unsafe_allow_html=True,
            )
            for p in peers:
                st.button(
                    p,
                    key=f"peer_{sym}_{p}",
                    on_click=_navigate,
                    args=(p,),
                    use_container_width=True,
                )

    with col_down:
        st.markdown(
            "<div style='text-align:center;font-size:0.85em;font-weight:700;"
            "color:#A5D6A7;letter-spacing:0.05em;margin-bottom:8px;'>⬇ DOWNSTREAM</div>",
            unsafe_allow_html=True,
        )
        st.caption("Customers / OEMs / End Markets")
        if downstream:
            for i, co in enumerate(downstream):
                _company_card(
                    co["ticker"], co["name"], co["relationship"], co.get("segment", ""),
                    card_color="#0A1F0A", key_prefix=f"dn{i}"
                )
        else:
            st.info("No downstream data available", icon="ℹ️")

    st.divider()

    # ── financial charts ──
    if not income_df.empty:
        st.subheader("Financial History")
        ch1, ch2 = st.columns(2)
        with ch1:
            st.caption("Revenue & Net Income (USD Billions)")
            st.plotly_chart(_revenue_chart(income_df), use_container_width=True, key=f"rc_{sym}")
        with ch2:
            st.caption("Margin Trends (%)")
            st.plotly_chart(_margin_chart(income_df), use_container_width=True, key=f"mc_{sym}")

        with st.expander("Raw financials table"):
            disp = income_df.copy()
            for col in ["revenue", "gross_profit", "operating_income", "net_income"]:
                if col in disp.columns:
                    disp[col] = disp[col].apply(lambda x: _fmt_number(x, prefix="$") if pd.notna(x) else "—")
            st.dataframe(disp, use_container_width=True)
    else:
        st.info("No financial history found in local DB for this ticker.", icon="ℹ️")


# ── main page ──────────────────────────────────────────────────────────────────

def main() -> None:
    # session state init
    if "sc_ticker" not in st.session_state:
        st.session_state["sc_ticker"] = ""
    if "sc_history" not in st.session_state:
        st.session_state["sc_history"] = []

    sc_map = load_supply_chain_map()
    known_tickers = sorted(sc_map.keys())

    # ── page header ──
    st.markdown("# Supply Chain Explorer")
    st.caption(
        "Search any ticker to explore its upstream suppliers, downstream customers, and peers. "
        "Click any company card to navigate to that company."
    )

    # ── search bar ──
    search_col, btn_col, clear_col = st.columns([3, 1, 1])
    with search_col:
        input_ticker = st.text_input(
            "Enter ticker",
            value=st.session_state["sc_ticker"],
            placeholder="e.g. NVDA, COHR, ANET, MU …",
            label_visibility="collapsed",
        ).strip().upper()
    with btn_col:
        if st.button("Search", type="primary", use_container_width=True):
            if input_ticker:
                _navigate(input_ticker)
    with clear_col:
        if st.button("Clear", use_container_width=True):
            st.session_state["sc_ticker"] = ""
            st.session_state["sc_history"] = []
            st.rerun()

    # quick-access buttons for known tickers
    with st.expander("Quick access — curated companies", expanded=False):
        cols = st.columns(8)
        for i, t in enumerate(known_tickers):
            cols[i % 8].button(t, key=f"qa_{t}", on_click=_navigate, args=(t,), use_container_width=True)

    # ── breadcrumb / back navigation ──
    history = st.session_state.get("sc_history", [])
    if len(history) > 1:
        bc_cols = st.columns([6, 1])
        with bc_cols[0]:
            trail = " › ".join(history[-5:])
            st.caption(f"Navigation: {trail}")
        with bc_cols[1]:
            if st.button("← Back"):
                st.session_state["sc_history"].pop()
                if st.session_state["sc_history"]:
                    st.session_state["sc_ticker"] = st.session_state["sc_history"][-1]
                else:
                    st.session_state["sc_ticker"] = ""
                st.rerun()

    st.divider()

    # ── render selected company ──
    active = st.session_state.get("sc_ticker", "") or input_ticker
    if active:
        if active not in sc_map:
            st.info(
                f"**{active}** is not in the curated supply chain map yet. "
                "Showing financial data from local DB. "
                "You can still navigate to this company's profile.",
                icon="ℹ️",
            )
        _render_company(active, sc_map)
    else:
        # landing state
        st.markdown(
            """
            <div style="text-align:center;padding:60px 20px;color:#607D8B;">
                <div style="font-size:3em;">🔗</div>
                <div style="font-size:1.2em;font-weight:600;margin-top:12px;">Enter a ticker above to explore its supply chain</div>
                <div style="font-size:0.9em;margin-top:8px;">
                    Curated data for: NVDA · AMD · INTC · AVGO · MRVL · TSM · ASML · AMAT · LRCX · KLAC ·
                    MU · WDC · COHR · LITE · AAOI · FN · CSCO · ANET · CIEN · JNPR ·
                    MSFT · GOOGL · AMZN · META · AAPL · QCOM · SNPS · DELL · HPE · SMCI · ORCL · TSLA
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


main()
