"""
Quarterly financials from yfinance (live pull; no local persistence).
"""

from __future__ import annotations

import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# yfinance row labels vary by ticker and API version
REVENUE_NAMES = ("Total Revenue", "Total Revenues", "Revenue")
GROSS_NAMES = ("Gross Profit",)
OP_INCOME_NAMES = ("Operating Income", "EBIT", "Operating Income Loss")
NET_INCOME_NAMES = (
    "Net Income",
    "Net Income Common Stockholders",
    "Net Income Including Non-Controlling Interests",
    "Net Income Continuous Operations",
)
EBITDA_NAMES = ("EBITDA", "Normalized EBITDA")
RD_NAMES = ("Research And Development", "Research Development")
SGA_NAMES = ("Selling General And Administration", "Selling General And Administrative")
EPS_DILUTED_NAMES = ("Diluted EPS", "Normalized EPS")
EPS_BASIC_NAMES = ("Basic EPS",)
INTEREST_NAMES = ("Interest Expense", "Interest Expense Non Operating")
TAX_NAMES = ("Tax Provision", "Income Tax")
PRETAX_NAMES = ("Pretax Income", "Income Before Tax", "Earnings Before Tax")

OCF_NAMES = ("Operating Cash Flow", "Cash Flow From Operations", "Cash from Operations", "Operating Cash Flow")
CAPEX_NAMES = ("Capital Expenditure", "Capital Expenditures")
FCF_NAMES = ("Free Cash Flow",)

DEBT_NAMES = ("Total Debt", "Long Term Debt And Capital Lease Obligation")
CASH_NAMES = ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
EQUITY_NAMES = (
    "Stockholders Equity",
    "Common Stock Equity",
    "Total Equity Gross Minority Interest",
    "Total Stockholder Equity",
)
SHARES_NAMES = ("Ordinary Shares Number", "Share Issued", "Diluted Average Shares", "Basic Average Shares")


def _norm_stmt_label(label: str) -> str:
    """Compare Yahoo labels without spaces/punct: e.g. Gross Profit vs GrossProfit."""
    return "".join(ch.lower() for ch in str(label) if ch.isalnum())


def _pick_row(df: pd.DataFrame | None, candidates: tuple[str, ...]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    idx = [str(x) for x in df.index]
    norms = [_norm_stmt_label(x) for x in idx]

    for name in candidates:
        if name in df.index:
            return df.loc[name]
        n_cand = _norm_stmt_label(name)
        for i, _label in enumerate(idx):
            if n_cand == norms[i]:
                return df.iloc[i]

    for i, label in enumerate(idx):
        low = label.lower()
        for name in candidates:
            if name.lower() in low:
                return df.iloc[i]

    for name in candidates:
        n_cand = _norm_stmt_label(name)
        if len(n_cand) < 8:
            continue
        for i, _label in enumerate(idx):
            if n_cand in norms[i]:
                return df.iloc[i]
    return None


def _value_at_col(series: pd.Series | None, col) -> float | None:
    if series is None:
        return None
    if col not in series.index:
        return None
    v = series[col]
    if pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _exact_column(df: pd.DataFrame | None, period_end) -> object | None:
    """Column whose date matches this quarter end (no fuzzy merge)."""
    if df is None or df.empty:
        return None
    pe = pd.Timestamp(period_end).normalize()
    for c in df.columns:
        if pd.Timestamp(c).normalize() == pe:
            return c
    return None


def _union_quarter_ends(
    inc: pd.DataFrame | None,
    cf: pd.DataFrame | None,
    bs: pd.DataFrame | None,
    num_quarters: int | None = None,
) -> list[pd.Timestamp]:
    """
    Yahoo often returns fewer income-statement columns than CF/BS. Union all quarter-end dates
    so we can show extra rows with partial metrics (e.g. BS-only).
    """
    ts: set[pd.Timestamp] = set()
    for df in (inc, cf, bs):
        if df is None or df.empty:
            continue
        for c in df.columns:
            ts.add(pd.Timestamp(c).normalize())
    if not ts:
        return []
    newest_first = sorted(ts, reverse=True)
    if num_quarters is not None:
        newest_first = newest_first[:num_quarters]
    return sorted(newest_first)


def _get_quarterly_frames(t: yf.Ticker) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Prefer get_* APIs (often more columns); fall back to legacy properties."""
    inc: pd.DataFrame | None = None
    cf: pd.DataFrame | None = None
    bs: pd.DataFrame | None = None
    try:
        inc = t.get_income_stmt(freq="quarterly")
    except Exception:
        pass
    try:
        cf = t.get_cash_flow(freq="quarterly")
    except Exception:
        pass
    try:
        bs = t.get_balance_sheet(freq="quarterly")
    except Exception:
        pass
    if inc is None or inc.empty:
        inc = getattr(t, "quarterly_income_stmt", None) or t.quarterly_financials
    if cf is None or cf.empty:
        cf = getattr(t, "quarterly_cashflow", None) or t.quarterly_cash_flow
    if bs is None or bs.empty:
        bs = getattr(t, "quarterly_balance_sheet", None) or t.quarterly_balancesheet
    return inc, cf, bs


def fetch_quarterly_metrics(ticker: str, num_quarters: int | None = None) -> pd.DataFrame:
    """
    Pull fiscal quarters from Yahoo via yfinance (default: every quarter Yahoo exposes).

    Income statement history is often shorter (~5 quarters) than cash flow or balance sheet; we merge
    **all quarter-end dates** from the three statements so extra periods appear with whatever line
    items exist (revenue may be blank).
    """
    sym = ticker.strip().upper()
    if not sym:
        return pd.DataFrame()

    t = yf.Ticker(sym)
    inc, cf, bs = _get_quarterly_frames(t)

    if (inc is None or inc.empty) and (cf is None or cf.empty) and (bs is None or bs.empty):
        return pd.DataFrame()

    period_list = _union_quarter_ends(inc, cf, bs, num_quarters)
    if not period_list:
        return pd.DataFrame()

    inc_ok = inc is not None and not inc.empty
    rev_row = _pick_row(inc, REVENUE_NAMES) if inc_ok else None
    gross_row = _pick_row(inc, GROSS_NAMES) if inc_ok else None
    op_row = _pick_row(inc, OP_INCOME_NAMES) if inc_ok else None
    net_row = _pick_row(inc, NET_INCOME_NAMES) if inc_ok else None
    ebitda_row = _pick_row(inc, EBITDA_NAMES) if inc_ok else None
    rd_row = _pick_row(inc, RD_NAMES) if inc_ok else None
    sga_row = _pick_row(inc, SGA_NAMES) if inc_ok else None
    eps_d_row = _pick_row(inc, EPS_DILUTED_NAMES) if inc_ok else None
    int_row = _pick_row(inc, INTEREST_NAMES) if inc_ok else None
    tax_row = _pick_row(inc, TAX_NAMES) if inc_ok else None
    pretax_row = _pick_row(inc, PRETAX_NAMES) if inc_ok else None

    ocf_row = _pick_row(cf, OCF_NAMES) if cf is not None else None
    capex_row = _pick_row(cf, CAPEX_NAMES) if cf is not None else None
    fcf_row = _pick_row(cf, FCF_NAMES) if cf is not None else None

    debt_row = _pick_row(bs, DEBT_NAMES) if bs is not None else None
    cash_row = _pick_row(bs, CASH_NAMES) if bs is not None else None
    eq_row = _pick_row(bs, EQUITY_NAMES) if bs is not None else None
    sh_row = _pick_row(bs, SHARES_NAMES) if bs is not None else None

    rows: list[dict[str, object]] = []
    for pe in period_list:
        pe = pd.Timestamp(pe).normalize()
        inc_c = _exact_column(inc, pe)
        cf_c = _exact_column(cf, pe)
        bs_c = _exact_column(bs, pe)

        rev = _value_at_col(rev_row, inc_c) if inc_c is not None else None
        gross = _value_at_col(gross_row, inc_c) if inc_c is not None else None
        op_inc = _value_at_col(op_row, inc_c) if inc_c is not None else None
        net = _value_at_col(net_row, inc_c) if inc_c is not None else None
        ebitda = _value_at_col(ebitda_row, inc_c) if inc_c is not None else None
        rd = _value_at_col(rd_row, inc_c) if inc_c is not None else None
        sga = _value_at_col(sga_row, inc_c) if inc_c is not None else None
        eps_d = _value_at_col(eps_d_row, inc_c) if inc_c is not None else None

        ocf = _value_at_col(ocf_row, cf_c) if cf_c is not None else None
        capex = _value_at_col(capex_row, cf_c) if cf_c is not None else None
        fcf_direct = _value_at_col(fcf_row, cf_c) if cf_c is not None else None

        if fcf_direct is not None:
            fcf_calc = fcf_direct
        elif ocf is not None and capex is not None:
            fcf_calc = ocf + capex
        else:
            fcf_calc = None

        debt = _value_at_col(debt_row, bs_c) if bs_c is not None else None
        cash = _value_at_col(cash_row, bs_c) if bs_c is not None else None
        equity = _value_at_col(eq_row, bs_c) if bs_c is not None else None
        shares = _value_at_col(sh_row, bs_c) if bs_c is not None else None

        gm = (gross / rev * 100.0) if rev and gross is not None and rev else None
        om = (op_inc / rev * 100.0) if rev and op_inc is not None and rev else None
        nm = (net / rev * 100.0) if rev and net is not None and rev else None
        fcf_m = (fcf_calc / rev * 100.0) if rev and fcf_calc is not None and rev else None
        rd_pct = (rd / rev * 100.0) if rev and rd is not None and rev else None
        sga_pct = (sga / rev * 100.0) if rev and sga is not None and rev else None

        net_debt = None
        if debt is not None and cash is not None:
            net_debt = debt - cash

        de_ratio = (debt / equity) if debt is not None and equity and equity != 0 else None

        pretax = _value_at_col(pretax_row, inc_c) if inc_c is not None and pretax_row is not None else None

        eff_tax = None
        if tax_row is not None and pretax is not None and pretax != 0 and inc_c is not None:
            tax_v = _value_at_col(tax_row, inc_c)
            if tax_v is not None:
                eff_tax = abs(tax_v) / abs(pretax) * 100.0

        rows.append(
            {
                "ticker": sym,
                "period_end": pe.strftime("%Y-%m-%d"),
                "total_revenue": rev,
                "gross_profit": gross,
                "gross_margin_pct": gm,
                "operating_income": op_inc,
                "operating_margin_pct": om,
                "ebitda": ebitda,
                "rd_expense": rd,
                "rd_pct_revenue": rd_pct,
                "sga_expense": sga,
                "sga_pct_revenue": sga_pct,
                "net_income": net,
                "net_margin_pct": nm,
                "diluted_eps": eps_d,
                "operating_cash_flow": ocf,
                "capex": capex,
                "free_cash_flow": fcf_calc,
                "fcf_margin_pct": fcf_m,
                "total_debt": debt,
                "cash_and_equivalents": cash,
                "total_equity": equity,
                "net_debt": net_debt,
                "debt_to_equity": de_ratio,
                "shares_outstanding": shares,
                "interest_expense": _value_at_col(int_row, inc_c) if int_row is not None and inc_c is not None else None,
                "tax_provision": _value_at_col(tax_row, inc_c) if tax_row is not None and inc_c is not None else None,
                "pretax_income": pretax,
                "effective_tax_rate_pct": eff_tax,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["period_end"] = pd.to_datetime(out["period_end"])
    out = out.sort_values("period_end", ascending=True).reset_index(drop=True)

    rev_s = out["total_revenue"]
    out["revenue_qoq_pct"] = rev_s.pct_change() * 100.0
    out["revenue_yoy_pct"] = rev_s.pct_change(4) * 100.0
    eps_s = out["diluted_eps"]
    out["eps_yoy_pct"] = eps_s.pct_change(4) * 100.0

    return out

