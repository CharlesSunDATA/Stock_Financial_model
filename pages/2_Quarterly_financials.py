"""
Quarterly financials dashboard.

Primary source: local SQLite database populated from FMP.
Fallback source: yfinance live pull.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.local_data import connect_readonly
from utils.data_loader import fetch_quarterly_metrics


_PLOTLY_CHART_SUPPORTS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


DISPLAY_COLS: list[tuple[str, str]] = [
    ("period_end", "Period end"),
    ("fiscal_period", "Fiscal period"),
    ("balance_sheet_date", "BS date"),
    ("total_revenue", "Revenue"),
    ("revenue_qoq_pct", "Rev. QoQ %"),
    ("revenue_yoy_pct", "Rev. YoY %"),
    ("gross_margin_pct", "Gross margin %"),
    ("operating_margin_pct", "Operating margin %"),
    ("net_margin_pct", "Net margin %"),
    ("ebitda", "EBITDA"),
    ("diluted_eps", "Diluted EPS"),
    ("eps_yoy_pct", "EPS YoY %"),
    ("operating_cash_flow", "Operating CF"),
    ("free_cash_flow", "FCF"),
    ("fcf_margin_pct", "FCF margin %"),
    ("fcf_conversion_pct", "FCF / Net income %"),
    ("capex_pct_revenue", "Capex % rev."),
    ("rd_pct_revenue", "R&D % rev."),
    ("sga_pct_revenue", "SG&A % rev."),
    ("cash_and_equivalents", "Cash"),
    ("total_debt", "Total debt"),
    ("net_debt", "Net debt"),
    ("debt_to_assets_pct", "Debt / assets %"),
    ("effective_tax_rate_pct", "Eff. tax rate %"),
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _db_path() -> Path:
    return _project_root() / "data" / "quant_data.db"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _payload_float(payload_json: Any, *keys: str) -> float | None:
    if not payload_json:
        return None
    try:
        payload = json.loads(str(payload_json))
    except (TypeError, json.JSONDecodeError):
        return None
    for key in keys:
        if key in payload:
            val = _safe_float(payload.get(key))
            if val is not None:
                return val
    return None


def _reported_payload_data(payload_json: Any) -> dict[str, Any]:
    if not payload_json:
        return {}
    try:
        payload = json.loads(str(payload_json))
    except (TypeError, json.JSONDecodeError):
        return {}
    data = payload.get("data")
    if isinstance(data, dict):
        return {str(k).lower(): v for k, v in data.items()}
    if isinstance(payload, dict):
        return {str(k).lower(): v for k, v in payload.items()}
    return {}


def _reported_float(payload_json: Any, *keys: str) -> float | None:
    data = _reported_payload_data(payload_json)
    for key in keys:
        val = _safe_float(data.get(key.lower()))
        if val is not None:
            return val
    return None


def _reported_total_debt(payload_json: Any) -> float | None:
    direct = _reported_float(payload_json, "totalDebt", "debt")
    if direct is not None:
        return direct
    pieces = [
        _reported_float(payload_json, "shortTermDebt", "shorttermdebt", "commercialPaper"),
        _reported_float(payload_json, "longTermDebtCurrent", "currentLongTermDebt"),
        _reported_float(payload_json, "longTermDebtNonCurrent", "longTermDebt"),
        _reported_float(payload_json, "capitalLeaseObligationsCurrent"),
        _reported_float(payload_json, "capitalLeaseObligationsNonCurrent"),
    ]
    vals = [v for v in pieces if v is not None]
    return float(sum(vals)) if vals else None


def _fmt_money(v: Any) -> str:
    x = _safe_float(v)
    if x is None:
        return "-"
    ax = abs(x)
    if ax >= 1e12:
        return f"{x / 1e12:,.2f}T"
    if ax >= 1e9:
        return f"{x / 1e9:,.2f}B"
    if ax >= 1e6:
        return f"{x / 1e6:,.1f}M"
    return f"{x:,.0f}"


def _fmt_pct(v: Any) -> str:
    x = _safe_float(v)
    return "-" if x is None else f"{x:,.1f}%"


def _fmt_num(v: Any, digits: int = 2) -> str:
    x = _safe_float(v)
    return "-" if x is None else f"{x:,.{digits}f}"


def _bar_colors(series: pd.Series, pos: str = "#2ecc71", neg: str = "#e74c3c") -> list[str]:
    return [pos if (pd.notna(v) and v >= 0) else neg for v in series]


def _metric_delta(current: Any, previous: Any, *, pct: bool = False) -> str | None:
    c = _safe_float(current)
    p = _safe_float(previous)
    if c is None or p is None:
        return None
    diff = c - p
    if pct:
        return f"{diff:+.1f} pp"
    if p == 0:
        return None
    return f"{diff / abs(p) * 100:+.1f}% QoQ"


@st.cache_data(ttl=60 * 10, show_spinner=False)
def available_db_tickers() -> list[str]:
    db = _db_path()
    if not db.exists():
        return []
    with connect_readonly(db) as conn:
        try:
            df = pd.read_sql_query(
                """
                SELECT DISTINCT ticker FROM income_statement
                UNION
                SELECT DISTINCT ticker FROM cash_flow_statement
                UNION
                SELECT DISTINCT ticker FROM balance_sheet
                ORDER BY ticker
                """,
                conn,
            )
        except Exception:
            return []
    return [str(x).upper() for x in df["ticker"].dropna().tolist()]


@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_company_profile(ticker: str) -> dict[str, Any]:
    db = _db_path()
    sym = ticker.strip().upper()
    if not db.exists() or not sym:
        return {}
    with connect_readonly(db) as conn:
        try:
            row = conn.execute(
                """
                SELECT company_name, sector, industry, full_time_employees
                FROM company_profile
                WHERE ticker = ?
                """,
                (sym,),
            ).fetchone()
        except Exception:
            return {}
    if not row:
        return {}
    return {
        "company_name": row[0],
        "sector": row[1],
        "industry": row[2],
        "full_time_employees": row[3],
    }


@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_db_quarterly_metrics(ticker: str, limit: int = 16) -> pd.DataFrame:
    """Merge quarterly income statement, cash flow, and balance sheet rows from SQLite."""
    db = _db_path()
    sym = ticker.strip().upper()
    if not db.exists() or not sym:
        return pd.DataFrame()

    with connect_readonly(db) as conn:
        inc = pd.read_sql_query(
            """
            SELECT
              ticker,
              report_date,
              period,
              revenue,
              gross_profit,
              operating_income,
              ebitda,
              net_income,
              eps,
              eps_diluted,
              weighted_avg_shares_out,
              payload_json
            FROM income_statement
            WHERE ticker = ?
            ORDER BY report_date ASC
            """,
            conn,
            params=(sym,),
        )
        cf = pd.read_sql_query(
            """
            SELECT
              ticker,
              report_date,
              period,
              operating_cash_flow,
              capital_expenditure,
              free_cash_flow,
              investing_cash_flow,
              financing_cash_flow,
              net_change_in_cash,
              payload_json
            FROM cash_flow_statement
            WHERE ticker = ?
            ORDER BY report_date ASC
            """,
            conn,
            params=(sym,),
        )
        bs = pd.read_sql_query(
            """
            SELECT
              ticker,
              report_date,
              cash_and_equivalents,
              total_assets,
              total_liabilities,
              outstanding_shares,
              total_debt
            FROM balance_sheet
            WHERE ticker = ?
            ORDER BY report_date ASC
            """,
            conn,
            params=(sym,),
        )
        bs_reported = pd.read_sql_query(
            """
            SELECT
              ticker,
              report_date,
              payload_json AS balance_sheet_payload
            FROM balance_sheet_as_reported
            WHERE ticker = ?
            ORDER BY report_date ASC
            """,
            conn,
            params=(sym,),
        )

    if inc.empty and cf.empty and bs.empty:
        return pd.DataFrame()

    dates = pd.concat(
        [
            inc[["report_date"]] if not inc.empty else pd.DataFrame(columns=["report_date"]),
            cf[["report_date"]] if not cf.empty else pd.DataFrame(columns=["report_date"]),
            bs[["report_date"]] if not bs.empty else pd.DataFrame(columns=["report_date"]),
        ],
        ignore_index=True,
    ).drop_duplicates()
    dates["period_end"] = pd.to_datetime(dates["report_date"], errors="coerce")
    dates = dates.dropna(subset=["period_end"]).sort_values("period_end")
    if dates.empty:
        return pd.DataFrame()

    inc = inc.rename(
        columns={
            "period": "fiscal_period",
            "revenue": "total_revenue",
            "eps_diluted": "diluted_eps",
            "eps": "basic_eps",
            "weighted_avg_shares_out": "shares_outstanding_income",
            "payload_json": "income_payload",
        }
    )
    cf = cf.rename(
        columns={
            "period": "cash_flow_period",
            "capital_expenditure": "capex",
            "payload_json": "cash_flow_payload",
        }
    )

    out = dates[["report_date", "period_end"]].merge(
        inc.drop(columns=["ticker"], errors="ignore"), on="report_date", how="left"
    )
    out = out.merge(cf.drop(columns=["ticker"], errors="ignore"), on="report_date", how="left")

    if not bs.empty:
        bs = bs.rename(columns={"report_date": "balance_sheet_date"})
        bs["balance_sheet_date"] = pd.to_datetime(bs["balance_sheet_date"], errors="coerce")
        bs = bs.dropna(subset=["balance_sheet_date"]).sort_values("balance_sheet_date")
        out = pd.merge_asof(
            out.sort_values("period_end"),
            bs.drop(columns=["ticker"], errors="ignore"),
            left_on="period_end",
            right_on="balance_sheet_date",
            direction="backward",
            tolerance=pd.Timedelta(days=140),
        )
    else:
        out["balance_sheet_date"] = pd.NaT

    if not bs_reported.empty:
        bs_reported = bs_reported.rename(columns={"report_date": "reported_balance_sheet_date"})
        bs_reported["reported_balance_sheet_date"] = pd.to_datetime(
            bs_reported["reported_balance_sheet_date"], errors="coerce"
        )
        bs_reported = bs_reported.dropna(subset=["reported_balance_sheet_date"]).sort_values(
            "reported_balance_sheet_date"
        )
        out = pd.merge_asof(
            out.sort_values("period_end"),
            bs_reported.drop(columns=["ticker"], errors="ignore"),
            left_on="period_end",
            right_on="reported_balance_sheet_date",
            direction="backward",
            tolerance=pd.Timedelta(days=140),
        )
    else:
        out["reported_balance_sheet_date"] = pd.NaT
        out["balance_sheet_payload"] = None

    out["ticker"] = sym
    out["fiscal_period"] = out["fiscal_period"].fillna(out.get("cash_flow_period"))

    reported_cash = out["balance_sheet_payload"].map(
        lambda p: _reported_float(p, "cashAndCashEquivalentsAtCarryingValue", "cashAndCashEquivalents")
    )
    reported_assets = out["balance_sheet_payload"].map(lambda p: _reported_float(p, "assets", "totalAssets"))
    reported_liabilities = out["balance_sheet_payload"].map(
        lambda p: _reported_float(p, "liabilities", "totalLiabilities")
    )
    reported_equity = out["balance_sheet_payload"].map(
        lambda p: _reported_float(p, "stockholdersEquity", "totalStockholdersEquity", "totalEquity")
    )
    reported_shares = out["balance_sheet_payload"].map(
        lambda p: _reported_float(p, "commonStockSharesOutstanding", "commonStockSharesIssued")
    )
    reported_debt = out["balance_sheet_payload"].map(_reported_total_debt)

    out["cash_and_equivalents"] = out["cash_and_equivalents"].fillna(reported_cash)
    out["total_assets"] = out["total_assets"].fillna(reported_assets)
    out["total_liabilities"] = out["total_liabilities"].fillna(reported_liabilities)
    out["total_debt"] = out["total_debt"].fillna(reported_debt)
    out["outstanding_shares"] = out["outstanding_shares"].fillna(reported_shares)

    out["rd_expense"] = out["income_payload"].map(
        lambda p: _payload_float(p, "researchAndDevelopmentExpenses")
    )
    out["sga_expense"] = out["income_payload"].map(
        lambda p: _payload_float(
            p,
            "sellingGeneralAndAdministrativeExpenses",
            "generalAndAdministrativeExpenses",
            "sellingAndMarketingExpenses",
        )
    )
    out["pretax_income"] = out["income_payload"].map(lambda p: _payload_float(p, "incomeBeforeTax"))
    out["tax_provision"] = out["income_payload"].map(lambda p: _payload_float(p, "incomeTaxExpense"))
    out["interest_expense"] = out["income_payload"].map(lambda p: _payload_float(p, "interestExpense"))
    out["depreciation_amortization"] = out["income_payload"].map(
        lambda p: _payload_float(p, "depreciationAndAmortization")
    )
    out["stock_based_comp"] = out["cash_flow_payload"].map(
        lambda p: _payload_float(p, "stockBasedCompensation")
    )
    out["working_capital_change"] = out["cash_flow_payload"].map(
        lambda p: _payload_float(p, "changeInWorkingCapital")
    )

    numeric_cols = [
        "total_revenue",
        "gross_profit",
        "operating_income",
        "ebitda",
        "net_income",
        "basic_eps",
        "diluted_eps",
        "shares_outstanding_income",
        "operating_cash_flow",
        "capex",
        "free_cash_flow",
        "investing_cash_flow",
        "financing_cash_flow",
        "net_change_in_cash",
        "cash_and_equivalents",
        "total_assets",
        "total_liabilities",
        "outstanding_shares",
        "total_debt",
        "rd_expense",
        "sga_expense",
        "pretax_income",
        "tax_provision",
        "interest_expense",
        "depreciation_amortization",
        "stock_based_comp",
        "working_capital_change",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    calculated_equity = out["total_assets"] - out["total_liabilities"]
    out["total_equity"] = calculated_equity.fillna(reported_equity)
    out["net_debt"] = out["total_debt"] - out["cash_and_equivalents"]
    out["shares_outstanding"] = out["outstanding_shares"].fillna(out["shares_outstanding_income"])

    with np.errstate(divide="ignore", invalid="ignore"):
        out["gross_margin_pct"] = out["gross_profit"] / out["total_revenue"] * 100.0
        out["operating_margin_pct"] = out["operating_income"] / out["total_revenue"] * 100.0
        out["net_margin_pct"] = out["net_income"] / out["total_revenue"] * 100.0
        out["fcf_margin_pct"] = out["free_cash_flow"] / out["total_revenue"] * 100.0
        out["fcf_conversion_pct"] = out["free_cash_flow"] / out["net_income"] * 100.0
        out["capex_pct_revenue"] = out["capex"].abs() / out["total_revenue"] * 100.0
        out["rd_pct_revenue"] = out["rd_expense"] / out["total_revenue"] * 100.0
        out["sga_pct_revenue"] = out["sga_expense"] / out["total_revenue"] * 100.0
        out["debt_to_equity"] = out["total_debt"] / out["total_equity"]
        out["debt_to_assets_pct"] = out["total_debt"] / out["total_assets"] * 100.0
        out["effective_tax_rate_pct"] = out["tax_provision"].abs() / out["pretax_income"].abs() * 100.0

    out = out.replace([np.inf, -np.inf], np.nan).sort_values("period_end").reset_index(drop=True)
    out["revenue_qoq_pct"] = out["total_revenue"].pct_change() * 100.0
    out["revenue_yoy_pct"] = out["total_revenue"].pct_change(4) * 100.0
    out["eps_qoq_pct"] = out["diluted_eps"].pct_change() * 100.0
    out["eps_yoy_pct"] = out["diluted_eps"].pct_change(4) * 100.0
    out["fcf_yoy_pct"] = out["free_cash_flow"].pct_change(4) * 100.0

    if limit > 0:
        out = out.tail(limit).reset_index(drop=True)
    return out


def _fmt_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "period_end" in out.columns:
        out["period_end"] = pd.to_datetime(out["period_end"]).dt.strftime("%Y-%m-%d")
    if "balance_sheet_date" in out.columns:
        out["balance_sheet_date"] = pd.to_datetime(out["balance_sheet_date"], errors="coerce").dt.strftime("%Y-%m-%d")
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
        "investing_cash_flow",
        "financing_cash_flow",
        "net_change_in_cash",
        "cash_and_equivalents",
        "total_assets",
        "total_liabilities",
        "total_debt",
        "total_equity",
        "net_debt",
        "pretax_income",
        "interest_expense",
        "tax_provision",
        "depreciation_amortization",
        "stock_based_comp",
        "working_capital_change",
    }
    pct_cols = {
        "gross_margin_pct",
        "operating_margin_pct",
        "net_margin_pct",
        "fcf_margin_pct",
        "fcf_conversion_pct",
        "capex_pct_revenue",
        "rd_pct_revenue",
        "sga_pct_revenue",
        "revenue_qoq_pct",
        "revenue_yoy_pct",
        "eps_qoq_pct",
        "eps_yoy_pct",
        "fcf_yoy_pct",
        "debt_to_assets_pct",
        "effective_tax_rate_pct",
    }
    for col in out.columns:
        if col in money_cols:
            out[col] = out[col].map(_fmt_money)
        elif col in pct_cols:
            out[col] = out[col].map(_fmt_pct)
        elif col in {"diluted_eps", "basic_eps", "debt_to_equity"}:
            out[col] = out[col].map(lambda x: _fmt_num(x, 2))
        elif col == "shares_outstanding":
            out[col] = out[col].map(_fmt_money)
    return out


def _latest_ttm(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.tail(4).sum())


def _profile_caption(profile: dict[str, Any], source: str, db_rows: int) -> str:
    parts = [f"Source: {source}"]
    if profile.get("company_name"):
        parts.append(str(profile["company_name"]))
    if profile.get("sector"):
        parts.append(str(profile["sector"]))
    if profile.get("industry"):
        parts.append(str(profile["industry"]))
    if source.startswith("Local"):
        parts.append(f"{db_rows} quarters loaded")
    return " · ".join(parts)


def _plot_layout(height: int = 300) -> dict[str, Any]:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=11),
        margin=dict(t=70, b=0, l=0, r=0),
        height=height,
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        legend=dict(
            orientation="h",
            y=1.14,
            x=1,
            xanchor="right",
            yanchor="bottom",
            title_text="",
            itemwidth=30,
        ),
        hovermode="x unified",
    )


def _line_chart(ch: pd.DataFrame, cols: list[tuple[str, str, str]], title: str, suffix: str = "") -> go.Figure:
    x = ch["period_end"].dt.strftime("%Y-%m-%d")
    fig = go.Figure()
    for col, label, color in cols:
        if col not in ch.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ch[col],
                name=label,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate=f"{label}: %{{y:,.1f}}{suffix}<extra></extra>",
            )
        )
    fig.update_layout(**_plot_layout(), title=dict(text=title, font=dict(size=13), x=0))
    if suffix:
        fig.update_yaxes(ticksuffix=suffix)
    return fig


def _plotly_chart(fig: go.Figure) -> None:
    if _PLOTLY_CHART_SUPPORTS_WIDTH:
        st.plotly_chart(fig, width="stretch", config={})
    else:
        st.plotly_chart(fig, use_container_width=True, config={})


def _render_kpis(df: pd.DataFrame) -> None:
    latest = df.dropna(subset=["period_end"]).iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else pd.Series(dtype=object)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", _fmt_money(latest.get("total_revenue")), _metric_delta(latest.get("total_revenue"), prev.get("total_revenue")))
    c2.metric("Gross Margin", _fmt_pct(latest.get("gross_margin_pct")), _metric_delta(latest.get("gross_margin_pct"), prev.get("gross_margin_pct"), pct=True))
    c3.metric("Diluted EPS", _fmt_num(latest.get("diluted_eps")), _metric_delta(latest.get("diluted_eps"), prev.get("diluted_eps")))
    c4.metric("Free Cash Flow", _fmt_money(latest.get("free_cash_flow")), _metric_delta(latest.get("free_cash_flow"), prev.get("free_cash_flow")))

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("TTM Revenue", _fmt_money(_latest_ttm(df, "total_revenue")))
    t2.metric("TTM Net Income", _fmt_money(_latest_ttm(df, "net_income")))
    t3.metric("TTM FCF", _fmt_money(_latest_ttm(df, "free_cash_flow")))
    t4.metric("Net Debt", _fmt_money(latest.get("net_debt")))


def _render_charts(df: pd.DataFrame) -> None:
    ch = df.sort_values("period_end").copy()
    x = ch["period_end"].dt.strftime("%Y-%m-%d")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for col, label, color in [
            ("total_revenue", "Revenue", "#3498db"),
            ("gross_profit", "Gross Profit", "#2ecc71"),
            ("operating_income", "Operating Income", "#f39c12"),
            ("net_income", "Net Income", "#e74c3c"),
        ]:
            if col in ch.columns:
                fig.add_trace(go.Bar(x=x, y=ch[col], name=label, marker_color=color))
        fig.update_layout(
            **_plot_layout(330),
            title=dict(text="Income Statement Stack", font=dict(size=13), x=0),
            barmode="group",
            yaxis_tickformat=".2s",
        )
        _plotly_chart(fig)

    with c2:
        _plotly_chart(
            _line_chart(
                ch,
                [
                    ("gross_margin_pct", "Gross", "#2ecc71"),
                    ("operating_margin_pct", "Operating", "#3498db"),
                    ("net_margin_pct", "Net", "#e74c3c"),
                    ("fcf_margin_pct", "FCF", "#9b59b6"),
                ],
                "Margins",
                "%",
            )
        )

    c3, c4 = st.columns(2)
    with c3:
        fig_cf = go.Figure()
        if "operating_cash_flow" in ch.columns:
            fig_cf.add_trace(go.Bar(x=x, y=ch["operating_cash_flow"], name="Operating CF", marker_color="#16a085"))
        if "free_cash_flow" in ch.columns:
            fig_cf.add_trace(
                go.Bar(
                    x=x,
                    y=ch["free_cash_flow"],
                    name="FCF",
                    marker_color=_bar_colors(ch["free_cash_flow"]),
                )
            )
        if "capex" in ch.columns:
            fig_cf.add_trace(go.Bar(x=x, y=ch["capex"], name="Capex", marker_color="#95a5a6"))
        fig_cf.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_width=1)
        fig_cf.update_layout(
            **_plot_layout(300),
            title=dict(text="Cash Flow", font=dict(size=13), x=0),
            barmode="group",
            yaxis_tickformat=".2s",
        )
        _plotly_chart(fig_cf)

    with c4:
        _plotly_chart(
            _line_chart(
                ch,
                [
                    ("revenue_yoy_pct", "Revenue YoY", "#3498db"),
                    ("eps_yoy_pct", "EPS YoY", "#f39c12"),
                    ("fcf_yoy_pct", "FCF YoY", "#2ecc71"),
                ],
                "Growth Rates",
                "%",
            )
        )

    c5, c6 = st.columns(2)
    with c5:
        fig_bs = go.Figure()
        for col, label, color in [
            ("cash_and_equivalents", "Cash", "#2ecc71"),
            ("total_debt", "Debt", "#e74c3c"),
            ("net_debt", "Net Debt", "#f39c12"),
        ]:
            if col in ch.columns:
                fig_bs.add_trace(go.Scatter(x=x, y=ch[col], mode="lines+markers", name=label, line=dict(color=color, width=2)))
        fig_bs.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_width=1)
        fig_bs.update_layout(
            **_plot_layout(280),
            title=dict(text="Balance Sheet Liquidity", font=dict(size=13), x=0),
            yaxis_tickformat=".2s",
        )
        _plotly_chart(fig_bs)

    with c6:
        _plotly_chart(
            _line_chart(
                ch,
                [
                    ("fcf_conversion_pct", "FCF / Net income", "#9b59b6"),
                    ("capex_pct_revenue", "Capex / revenue", "#95a5a6"),
                    ("rd_pct_revenue", "R&D / revenue", "#1abc9c"),
                    ("sga_pct_revenue", "SG&A / revenue", "#e67e22"),
                ],
                "Efficiency Ratios",
                "%",
            )
        )


def _render_tables(df: pd.DataFrame) -> None:
    newest = df.sort_values("period_end", ascending=False).copy()
    disp = _fmt_display(newest)
    present = [col for col, _label in DISPLAY_COLS if col in disp.columns]
    st.dataframe(
        disp[present].rename(columns={col: label for col, label in DISPLAY_COLS}),
        width="stretch",
        hide_index=True,
    )

    tabs = st.tabs(["Income statement", "Cash flow", "Balance sheet", "Raw merged data"])
    with tabs[0]:
        cols = [
            "period_end",
            "fiscal_period",
            "total_revenue",
            "gross_profit",
            "operating_income",
            "ebitda",
            "net_income",
            "basic_eps",
            "diluted_eps",
            "rd_expense",
            "sga_expense",
            "interest_expense",
            "tax_provision",
        ]
        show = _fmt_display(newest[[c for c in cols if c in newest.columns]])
        st.dataframe(show, width="stretch", hide_index=True)
    with tabs[1]:
        cols = [
            "period_end",
            "operating_cash_flow",
            "capex",
            "free_cash_flow",
            "investing_cash_flow",
            "financing_cash_flow",
            "net_change_in_cash",
            "stock_based_comp",
            "working_capital_change",
        ]
        show = _fmt_display(newest[[c for c in cols if c in newest.columns]])
        st.dataframe(show, width="stretch", hide_index=True)
    with tabs[2]:
        cols = [
            "period_end",
            "balance_sheet_date",
            "cash_and_equivalents",
            "total_assets",
            "total_liabilities",
            "total_equity",
            "total_debt",
            "net_debt",
            "shares_outstanding",
            "debt_to_equity",
            "debt_to_assets_pct",
        ]
        show = _fmt_display(newest[[c for c in cols if c in newest.columns]])
        st.dataframe(show, width="stretch", hide_index=True)
    with tabs[3]:
        raw = newest.drop(
            columns=["income_payload", "cash_flow_payload", "balance_sheet_payload"],
            errors="ignore",
        )
        st.dataframe(raw, width="stretch", hide_index=True)


def main() -> None:
    st.title("Quarterly Financials")
    st.caption(
        "Local SQLite first, Yahoo fallback when needed. This view merges income statement, cash flow, "
        "balance sheet, growth, margin, liquidity, and cash-conversion metrics."
    )

    tickers = available_db_tickers()

    with st.sidebar:
        st.header("Data")
        default_ticker = "NVDA" if "NVDA" in tickers else (tickers[0] if tickers else "NVDA")
        ticker = st.text_input("Ticker symbol", value=default_ticker).strip().upper() or default_ticker
        source = st.radio("Source", ["Local database", "Yahoo live", "Auto: database then Yahoo"])
        limit = st.slider("Quarters to show", min_value=4, max_value=20, value=12, step=1)
        if tickers:
            st.caption(f"Database tickers: {len(tickers):,}")
        st.caption(f"DB: `{_db_path()}`")
        if st.button("Refresh data", type="primary", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    profile = fetch_company_profile(ticker)

    df = pd.DataFrame()
    source_used = source
    err: str | None = None

    if source in {"Local database", "Auto: database then Yahoo"}:
        with st.spinner(f"Loading {ticker} from local SQLite..."):
            try:
                df = fetch_db_quarterly_metrics(ticker, limit=limit)
                source_used = "Local database"
            except Exception as exc:
                err = str(exc)
                df = pd.DataFrame()

    if df.empty and source in {"Yahoo live", "Auto: database then Yahoo"}:
        with st.spinner(f"Downloading {ticker} from Yahoo Finance..."):
            try:
                df = fetch_quarterly_metrics(ticker, num_quarters=limit)
                source_used = "Yahoo live"
            except Exception as exc:
                err = str(exc)
                df = pd.DataFrame()

    if df.empty:
        if err:
            st.error(err)
        st.info("No quarterly data found. Try another ticker, refresh the local loaders, or switch source.")
        return

    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df = df.dropna(subset=["period_end"]).sort_values("period_end").reset_index(drop=True)

    st.subheader(f"{ticker} — Quarterly Financials")
    st.caption(_profile_caption(profile, source_used, len(df)))

    _render_kpis(df)
    st.divider()
    _render_charts(df)
    st.divider()
    st.subheader("Detailed Tables")
    _render_tables(df)


main()
