"""
Stock valuation — DCF + P/E zone (Streamlit page module).
Do not call st.set_page_config here — the entrypoint app.py sets it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# Comparable tickers for peer P/E (expand as needed). Used when no industry match.
PEER_MAP: dict[str, list[str]] = {
    "NVDA": ["AMD", "AVGO", "INTC", "MU", "QCOM"],
    "AMD": ["NVDA", "AVGO", "INTC", "MU", "QCOM"],
    "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
    "MSFT": ["AAPL", "GOOGL", "META", "ORCL"],
    "GOOGL": ["META", "MSFT", "AMZN"],
    "META": ["GOOGL", "SNAP", "PINS"],
    "TSLA": ["F", "GM", "RIVN"],
    "JPM": ["BAC", "WFC", "C", "GS"],
}
SEMIS_PEERS = ["AMD", "AVGO", "INTC", "MU", "QCOM"]
SOFTWARE_PEERS = ["MSFT", "ORCL", "CRM", "ADBE"]


@dataclass
class MarketData:
    """Spot price and relative valuation multiples."""

    price: float | None
    trailing_pe: float | None
    forward_pe: float | None
    peg: float | None
    pb: float | None
    peer_avg_pe: float | None  # Mean trailing P/E of comparable names (industry proxy)
    peer_symbols_used: tuple[str, ...]  # Peers actually averaged


@dataclass
class FundamentalInputs:
    """Fundamental inputs for DCF (with fallbacks applied)."""

    fcf_base: float  # Implied base-year FCF (USD)
    shares: float  # Shares outstanding
    net_debt: float  # Total debt minus cash (USD); can be negative (net cash)
    used_defaults: list[str]  # Human-readable notes when defaults were used


@dataclass
class DCFResult:
    """Two-stage DCF outputs."""

    intrinsic_per_share: float
    band_low: float  # intrinsic -10%
    band_high: float  # intrinsic +10%
    enterprise_value: float
    pv_fcf_5y: float
    pv_terminal: float


def _safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def fetch_ticker_info(ticker: str) -> tuple[dict[str, Any], Any]:
    """Return (info dict, Ticker). Raises if fundamentals are unavailable."""
    t = yf.Ticker(ticker)
    info = t.info
    if not info:
        raise ValueError(f"Could not load fundamentals for {ticker} (empty info).")
    return info, t


def compute_peer_average_pe(ticker: str, info: dict[str, Any]) -> tuple[float | None, tuple[str, ...]]:
    """
    Mean trailing P/E of a small peer set (industry proxy). Falls back to SPY/QQQ if unknown sector.
    """
    t_up = ticker.upper()
    peers = list(PEER_MAP.get(t_up, []))
    if not peers:
        ind = (info.get("industry") or "").lower()
        if any(x in ind for x in ("semiconductor", "semiconductors")):
            peers = [p for p in SEMIS_PEERS if p.upper() != t_up]
        elif "software" in ind:
            peers = [p for p in SOFTWARE_PEERS if p.upper() != t_up]
        else:
            peers = ["SPY", "QQQ"]

    pes: list[float] = []
    used: list[str] = []
    for sym in peers:
        if sym.upper() == t_up:
            continue
        try:
            inf = yf.Ticker(sym).info
            pe = inf.get("trailingPE")
            pe = _safe_float(pe)
            if pe is not None and 0 < pe < 800:
                pes.append(pe)
                used.append(sym.upper())
        except Exception:
            continue

    if not pes:
        return None, tuple(peers)
    return float(np.mean(pes)), tuple(used)


def build_market_data(ticker: str, info: dict[str, Any], hist_close: float | None) -> MarketData:
    """Price, multiples, and peer-average P/E."""
    price = _safe_float(
        hist_close if hist_close is not None else info.get("currentPrice") or info.get("regularMarketPrice")
    )
    peer_avg, peer_used = compute_peer_average_pe(ticker, info)
    return MarketData(
        price=price,
        trailing_pe=_safe_float(info.get("trailingPE")),
        forward_pe=_safe_float(info.get("forwardPE")),
        peg=_safe_float(info.get("pegRatio")),
        pb=_safe_float(info.get("priceToBook")),
        peer_avg_pe=peer_avg,
        peer_symbols_used=peer_used,
    )


def _quarterly_eps_series(t: yf.Ticker) -> pd.Series | None:
    q = None
    for attr in ("quarterly_income_stmt", "quarterly_financials"):
        try:
            cand = getattr(t, attr, None)
            if cand is not None and not cand.empty:
                q = cand
                break
        except Exception:
            continue
    if q is None or q.empty:
        return None
    for name in ("Diluted EPS", "Basic EPS", "Normalized EPS"):
        if name in q.index:
            s = q.loc[name]
            s = pd.to_numeric(s, errors="coerce").dropna()
            if s.empty:
                continue
            dti = pd.to_datetime(s.index)
            s.index = pd.DatetimeIndex(dti).tz_localize(None)
            return s.sort_index()
    return None


def _annual_diluted_eps_series(t: yf.Ticker) -> pd.Series | None:
    """Fiscal-year diluted EPS (annual statements); index = fiscal period ends."""
    for attr in ("income_stmt", "financials"):
        try:
            inc = getattr(t, attr, None)
            if inc is None or inc.empty:
                continue
            for name in ("Diluted EPS", "Basic EPS"):
                if name not in inc.index:
                    continue
                s = pd.to_numeric(inc.loc[name], errors="coerce").dropna()
                if s.empty:
                    continue
                dti = pd.to_datetime(s.index)
                s.index = pd.DatetimeIndex(dti).tz_localize(None)
                return s.sort_index()
        except Exception:
            continue
    return None


def _ttm_points_from_quarterly(eps_q: pd.Series) -> pd.Series:
    """
    One TTM value per quarter-end date. Uses full 4Q sum when possible; otherwise
    annualizes partial-year sums (1Q–3Q) so early months are not all NaN.
    """
    eps_q = eps_q.sort_index()
    cols = list(eps_q.index)
    ttm_idx: list[pd.Timestamp] = []
    ttm_val: list[float] = []
    for j in range(len(cols)):
        if j >= 3:
            ttm = float(eps_q.iloc[j - 3 : j + 1].sum())
        elif j == 2:
            ttm = float(eps_q.iloc[0:3].sum()) * (4.0 / 3.0)
        elif j == 1:
            ttm = float(eps_q.iloc[0:2].sum()) * 2.0
        else:
            ttm = float(eps_q.iloc[0]) * 4.0
        if ttm > 0:
            ttm_idx.append(pd.Timestamp(cols[j]))
            ttm_val.append(ttm)
    if not ttm_val:
        return pd.Series(dtype=float)
    return pd.Series(ttm_val, index=pd.DatetimeIndex(ttm_idx)).sort_index()


def _price_monthly(t: yf.Ticker, years: int) -> pd.Series | None:
    hist = t.history(period=f"{years}y", interval="1mo", auto_adjust=True)
    if hist is None or hist.empty:
        return None
    close_m = hist["Close"].copy()
    close_m.index = pd.DatetimeIndex(pd.to_datetime(close_m.index)).tz_localize(None)
    return close_m


def _pe_from_eps_series(close_m: pd.Series, eps_s: pd.Series, min_periods: int) -> pd.DataFrame | None:
    """Monthly P/E = price / eps_s.asof(month) where eps_s is step function (TTM or FY EPS)."""
    rows: list[dict[str, Any]] = []
    for dt, px in close_m.items():
        dt_ts = pd.Timestamp(dt)
        eps_v = eps_s.asof(dt_ts)
        if eps_v is None or (isinstance(eps_v, float) and np.isnan(eps_v)):
            continue
        eps_f = float(eps_v)
        if eps_f <= 0:
            continue
        pxf = float(px)
        if pxf <= 0:
            continue
        pe = pxf / eps_f
        if pe > 500 or pe < 0:
            continue
        rows.append({"date": dt_ts, "close": pxf, "eps": eps_f, "pe": pe})
    if len(rows) < min_periods:
        return None
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def build_monthly_pe_history(ticker: str, years: int = 5) -> pd.DataFrame | None:
    """
    Monthly trailing P/E series for percentile / zone comparison.
    Prefer TTM (quarterly EPS with partial annualization); else FY diluted EPS.
    """
    t = yf.Ticker(ticker)
    close_m = _price_monthly(t, years)
    if close_m is None:
        return None

    eps_q = _quarterly_eps_series(t)
    if eps_q is not None and len(eps_q) >= 1:
        ttm_s = _ttm_points_from_quarterly(eps_q)
        if len(ttm_s) > 0:
            out = _pe_from_eps_series(close_m, ttm_s, min_periods=6)
            if out is not None:
                out.attrs["pe_basis"] = "TTM (quarterly EPS, partial-year annualized when <4Q)"
                return out

    eps_a = _annual_diluted_eps_series(t)
    if eps_a is not None and len(eps_a) >= 1:
        out = _pe_from_eps_series(close_m, eps_a, min_periods=6)
        if out is not None:
            out.attrs["pe_basis"] = "FY diluted EPS (annual reports)"
            return out

    return None


def pe_zone_gauge(
    ticker: str,
    current_pe: float | None,
    hist_df: pd.DataFrame | None,
    lookback_years: int,
):
    """
    Horizontal zone bar showing where the current trailing P/E sits vs history.
    Returns None when data is insufficient (caller falls back to text).
    """
    import plotly.graph_objects as go  # noqa: PLC0415

    sym = ticker.upper()
    if current_pe is None or current_pe <= 0 or not np.isfinite(current_pe):
        return None
    if hist_df is None or hist_df.empty or "pe" not in hist_df.columns:
        return None
    s = hist_df["pe"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 6:
        return None

    q25, q50, q75 = (float(x) for x in np.percentile(s, [25, 50, 75]))
    cur = float(current_pe)

    if cur < q25:
        zone_label, zone_color = "Undervalued  ✅", "#2ecc71"
    elif cur < q75:
        zone_label, zone_color = "Fair Value  🟡", "#f1c40f"
    else:
        zone_label, zone_color = "Overvalued  ⚠️", "#e74c3c"

    basis = hist_df.attrs.get("pe_basis", "")
    subtitle = f"~{lookback_years}y monthly · {basis}" if basis else f"~{lookback_years}y monthly"

    delta_val = cur - q50
    delta_sign = "▲" if delta_val >= 0 else "▼"
    delta_color = "#e74c3c" if delta_val >= 0 else "#2ecc71"

    # Axis range: give slight padding on each side
    x_min = max(0.0, float(s.min())) * 0.90
    x_max = q75 * 1.40

    bar_y0, bar_y1 = 0.3, 0.7   # bar occupies middle 40 % of y

    fig = go.Figure()

    # ── Coloured zone rectangles ────────────────────────────────────────────
    zones = [
        (x_min, q25,   "rgba(46,204,113,0.55)",  "#2ecc71",  f"Below P25\n(Cheap)"),
        (q25,   q75,   "rgba(241,196,15,0.45)",  "#f1c40f",  f"P25 – P75\n(Fair Value)"),
        (q75,   x_max, "rgba(231,76,60,0.50)",   "#e74c3c",  f"Above P75\n(Expensive)"),
    ]
    for x0, x1, fill, border, _ in zones:
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=bar_y0, y1=bar_y1,
            fillcolor=fill, line=dict(color=border, width=1),
            layer="below",
        )

    # ── Current P/E marker (blue solid line inside bar + label above) ────────
    fig.add_shape(
        type="line", x0=cur, x1=cur, y0=bar_y0, y1=bar_y1,
        line=dict(color="#3498db", width=3),
    )
    fig.add_trace(go.Scatter(
        x=[cur], y=[bar_y1 + 0.22],
        mode="markers+text",
        marker=dict(symbol="diamond", size=12, color="#3498db",
                    line=dict(color="white", width=1.5)),
        text=[f"<b>{cur:.2f}x</b>"],
        textposition="middle right",
        textfont=dict(size=13, color="#3498db"),
        hovertemplate=f"Current P/E: {cur:.2f}x<extra></extra>",
        showlegend=False,
    ))

    # ── Annotations: zone labels below bar, P50 label above ─────────────────
    # zone centre x positions
    zone_centres = [
        ((x_min + q25) / 2,  "#2ecc71", f"Below P25<br><b>{q25:.1f}x</b>",    "bottom"),
        ((q25  + q75) / 2,   "#f1c40f", f"P25 – P75<br><b>{q25:.1f}–{q75:.1f}x</b>", "bottom"),
        ((q75  + x_max) / 2, "#e74c3c", f"Above P75<br><b>{q75:.1f}x</b>",    "bottom"),
    ]
    for xc, col, label, _ in zone_centres:
        fig.add_annotation(
            x=xc, y=bar_y0 - 0.04, xref="x", yref="y",
            text=label, showarrow=False,
            font=dict(size=11, color=col),
            yanchor="top", xanchor="center",
        )

    # ── Layout ──────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{sym}  Trailing P/E: <span style='color:{zone_color}'>"
                f"{cur:.2f}x  —  {zone_label}</span></b><br>"
                f"<span style='font-size:11px;color:#aaa'>"
                f"{delta_sign} {abs(delta_val):.2f}x vs P50  ·  {subtitle}</span>"
            ),
            font=dict(size=14, color="white"),
            x=0, xanchor="left",
        ),
        xaxis=dict(
            range=[x_min, x_max],
            showgrid=False, zeroline=False,
            tickfont=dict(size=11, color="#ccc"),
            tickformat=".1f",
            showline=False,
        ),
        yaxis=dict(
            range=[0, 1],
            showgrid=False, zeroline=False,
            showticklabels=False, showline=False,
        ),
        height=220,
        margin=dict(t=65, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


def pe_zone_fallback_text(
    ticker: str,
    current_pe: float | None,
    hist_df: pd.DataFrame | None,
    lookback_years: int,
) -> str:
    """Plain-text fallback when gauge cannot be rendered."""
    sym = ticker.upper()
    if current_pe is None or current_pe <= 0 or not np.isfinite(current_pe):
        return f"**{sym}** — No valid **trailing P/E** is available; cannot classify vs. history."
    if hist_df is None or hist_df.empty or "pe" not in hist_df.columns:
        return f"**{sym}** — Not enough historical monthly P/E to classify the current multiple."
    s = hist_df["pe"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 6:
        return f"**{sym}** — Too few monthly P/E observations to estimate stable percentiles."
    q25, q50, q75 = (float(x) for x in np.percentile(s, [25, 50, 75]))
    cur = float(current_pe)
    if cur < q25:
        zone_label, zone_note = "Low", "Below P25"
    elif cur < q50:
        zone_label, zone_note = "Below median", "P25–P50"
    elif cur < q75:
        zone_label, zone_note = "Above median", "P50–P75"
    else:
        zone_label, zone_note = "High", "Above P75"
    basis = hist_df.attrs.get("pe_basis", "")
    basis_line = f"Historical P/E basis: {basis}." if basis else ""
    return (
        f"**{sym} — Trailing P/E vs. historical range**\n\n"
        f"- **Current trailing P/E:** {cur:.2f}\n"
        f"- **Band:** **{zone_label}** ({zone_note})\n"
        f"- **Reference percentiles (~{lookback_years}y monthly sample):** "
        f"P25 = {q25:.2f}, P50 = {q50:.2f}, P75 = {q75:.2f}\n"
        f"- {basis_line}"
    )


def _latest_annual_value(cf: pd.DataFrame, row_names: tuple[str, ...]) -> float | None:
    """Latest annual value from cash flow statement (iterate columns for first non-null)."""
    if cf is None or cf.empty:
        return None
    for name in row_names:
        if name in cf.index:
            row = cf.loc[name]
            for col in cf.columns:
                v = row[col]
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    return float(v)
    return None


def _latest_bs_value(bs: pd.DataFrame, row_names: tuple[str, ...]) -> float | None:
    """Latest value from balance sheet."""
    if bs is None or bs.empty:
        return None
    for name in row_names:
        if name in bs.index:
            row = bs.loc[name]
            for col in bs.columns:
                v = row[col]
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    return float(v)
    return None


def build_fundamental_inputs(t: yf.Ticker, info: dict[str, Any]) -> FundamentalInputs:
    """
    FCF = Operating cash flow + Capital expenditure (CapEx is usually negative).
    Shares and net debt: prefer info, else balance sheet.
    """
    used_defaults: list[str] = []
    cf = None
    bs = None
    try:
        cf = t.cashflow
    except Exception:
        used_defaults.append("Cash flow statement download failed")

    ocf = _latest_annual_value(
        cf if cf is not None else pd.DataFrame(),
        ("Operating Cash Flow", "Cash From Operating Activities", "Cash Flow From Operations"),
    )
    capex = _latest_annual_value(
        cf if cf is not None else pd.DataFrame(),
        ("Capital Expenditure", "Capital Expenditures"),
    )

    if ocf is None:
        ocf = 0.0
        used_defaults.append("Operating cash flow missing; OCF in FCF set to 0")
    if capex is None:
        capex = 0.0
        used_defaults.append("CapEx missing; CapEx in FCF set to 0")

    fcf_base = float(ocf) + float(capex)

    shares = _safe_float(info.get("sharesOutstanding"))
    if shares is None or shares <= 0:
        try:
            bs = t.balance_sheet
            sh_row = _latest_bs_value(bs, ("Ordinary Shares Number", "Share Issued", "Common Stock"))
            if sh_row is not None and sh_row > 0:
                shares = float(sh_row)
            else:
                shares = 1.0
                used_defaults.append(
                    "Shares outstanding missing; denominator set to 1 (illustrative only, not for trading)"
                )
        except Exception:
            shares = 1.0
            used_defaults.append("Shares outstanding missing; denominator set to 1")

    total_debt = _safe_float(info.get("totalDebt"))
    total_cash = _safe_float(info.get("totalCash"))
    if total_debt is None:
        try:
            if bs is None:
                bs = t.balance_sheet
            td = _latest_bs_value(bs, ("Total Debt", "Long Term Debt"))
            total_debt = td if td is not None else 0.0
            if td is None:
                used_defaults.append("Total debt defaulted to 0")
        except Exception:
            total_debt = 0.0
            used_defaults.append("Total debt missing; set to 0")
    if total_cash is None:
        try:
            if bs is None:
                bs = t.balance_sheet
            tc = _latest_bs_value(
                bs,
                ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"),
            )
            total_cash = tc if tc is not None else 0.0
            if tc is None:
                used_defaults.append("Cash defaulted to 0")
        except Exception:
            total_cash = 0.0
            used_defaults.append("Cash missing; set to 0")

    net_debt = float(total_debt) - float(total_cash)

    return FundamentalInputs(
        fcf_base=fcf_base,
        shares=max(float(shares), 1e-12),
        net_debt=net_debt,
        used_defaults=used_defaults,
    )


def run_two_stage_dcf(
    fcf0: float,
    growth_5y: float,
    wacc: float,
    terminal_growth: float,
    net_debt: float,
    shares: float,
) -> DCFResult | None:
    """
    Stage 1: FCF_t = FCF_{t-1} * (1+g), t=1..5, discount each year.
    Stage 2: TV = FCF_5*(1+g_term)/(WACC-g_term), discount to today.
    """
    if shares <= 0:
        return None
    if wacc <= terminal_growth:
        raise ValueError("WACC must exceed terminal growth (terminal value formula requires it).")

    pv_fcf = 0.0
    fcf_prev = fcf0
    for t in range(1, 6):
        fcf_t = fcf_prev * (1.0 + growth_5y)
        pv_fcf += fcf_t / ((1.0 + wacc) ** t)
        fcf_prev = fcf_t

    fcf5 = fcf_prev
    tv = fcf5 * (1.0 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = tv / ((1.0 + wacc) ** 5)

    ev = pv_fcf + pv_terminal
    equity = ev - net_debt
    intrinsic = equity / shares

    band_low = intrinsic * 0.9
    band_high = intrinsic * 1.1

    return DCFResult(
        intrinsic_per_share=intrinsic,
        band_low=band_low,
        band_high=band_high,
        enterprise_value=ev,
        pv_fcf_5y=pv_fcf,
        pv_terminal=pv_terminal,
    )


def classify_valuation(price: float | None, low: float, high: float) -> str:
    """Map spot price vs [low, high] fair band to a short English label."""
    if price is None or price <= 0:
        return "Unable to classify (no price)"
    if price < low:
        return "Materially undervalued (below fair band)"
    if price <= high:
        return "Fair (within valuation band)"
    return "Overvalued (above fair band)"


def main() -> None:
    st.title("Stock Valuation Dashboard")
    st.caption("Data: yfinance — simplified two-stage DCF for education/research only, not investment advice.")

    with st.sidebar:
        st.header("Ticker & DCF inputs")
        ticker = st.text_input("Ticker symbol", value="NVDA").strip().upper() or "NVDA"
        g5 = st.slider("5-year FCF growth rate (annual, %)", 0, 100, 20, format="%d%%") / 100.0
        g_term = st.slider("Terminal growth rate (%)", 1, 5, 3, format="%d%%") / 100.0
        wacc = st.slider("Discount rate WACC (%)", 5, 20, 10, format="%d%%") / 100.0

    try:
        info, t = fetch_ticker_info(ticker)
        hist = t.history(period="5d")
        close_px = float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else None
        market = build_market_data(ticker, info, close_px)
        fundamentals = build_fundamental_inputs(t, info)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.info("Check the ticker, network, or try again later.")
        return

    dcf: DCFResult | None = None
    dcf_error: str | None = None
    try:
        dcf = run_two_stage_dcf(
            fcf0=fundamentals.fcf_base,
            growth_5y=g5,
            wacc=wacc,
            terminal_growth=g_term,
            net_debt=fundamentals.net_debt,
            shares=fundamentals.shares,
        )
    except ValueError as e:
        dcf_error = str(e)
    except Exception as e:
        dcf_error = f"DCF error: {e}"

    st.subheader("1. Fundamentals & relative valuation")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Last price (USD)", f"${market.price:,.2f}" if market.price else "—")
    with c2:
        st.metric("Trailing P/E", f"{market.trailing_pe:.2f}" if market.trailing_pe else "—")
    with c3:
        st.metric("Forward P/E", f"{market.forward_pe:.2f}" if market.forward_pe else "—")
    with c4:
        st.metric("PEG", f"{market.peg:.2f}" if market.peg else "—")
    with c5:
        st.metric("P/B", f"{market.pb:.2f}" if market.pb else "—")
    with c6:
        st.metric("Peer avg P/E", f"{market.peer_avg_pe:.2f}" if market.peer_avg_pe else "—")
    st.caption(
        "**Peer avg P/E (industry proxy):** mean trailing P/E of comparable tickers — "
        f"{', '.join(market.peer_symbols_used) if market.peer_symbols_used else 'none resolved'}."
    )

    st.markdown("##### Trailing P/E vs. historical range (~5y monthly)")
    _lookback_y = 5
    try:
        hist_df = build_monthly_pe_history(ticker, years=_lookback_y)
    except Exception:
        hist_df = None
    try:
        gauge_fig = pe_zone_gauge(ticker, market.trailing_pe, hist_df, _lookback_y)
    except Exception:
        gauge_fig = None
    if gauge_fig is not None:
        st.plotly_chart(gauge_fig, use_container_width=True)
    else:
        st.markdown(pe_zone_fallback_text(ticker, market.trailing_pe, hist_df, _lookback_y))
    st.caption(
        "🟢 Green = below P25 (historically cheap)  ·  "
        "🟡 Yellow = P25–P75 (fair value range)  ·  "
        "🔴 Red = above P75 (historically expensive)  ·  "
        "White line = P50 median.  "
        "Bands use monthly P/E distribution; Yahoo trailing P/E as current value. "
        "For education only; not investment advice."
    )

    if fundamentals.used_defaults:
        with st.expander("Data gaps (defaults or substitutes applied)", expanded=False):
            for line in fundamentals.used_defaults:
                st.warning(line)

    st.divider()

    st.subheader("2. Absolute valuation — two-stage DCF")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
| Item | Value |
|------|-------|
| Base-year FCF (implied) | {fundamentals.fcf_base:,.0f} USD |
| Shares outstanding | {fundamentals.shares:,.0f} |
| Net debt (debt − cash) | {fundamentals.net_debt:,.0f} USD |
"""
        )
    with col_b:
        st.markdown(
            f"""
| DCF input | Setting |
|-----------|---------|
| 5-year FCF growth | {g5*100:.1f}% |
| Terminal growth | {g_term*100:.1f}% |
| WACC | {wacc*100:.1f}% |
"""
        )

    if dcf_error:
        st.error(dcf_error)
    elif dcf is not None:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Intrinsic value / share (est.)", f"${dcf.intrinsic_per_share:,.2f}")
        with m2:
            st.metric("Fair band low (−10%)", f"${dcf.band_low:,.2f}")
        with m3:
            st.metric("Fair band high (+10%)", f"${dcf.band_high:,.2f}")
        st.caption(
            f"EV ≈ {dcf.enterprise_value:,.0f} USD — PV(5y FCF) {dcf.pv_fcf_5y:,.0f} — PV(terminal) {dcf.pv_terminal:,.0f}"
        )
    else:
        st.warning("DCF could not be completed — check inputs and data.")

    st.divider()

    st.subheader("3. Margin of safety vs spot price")
    if dcf is not None and market.price:
        import plotly.graph_objects as go  # noqa: PLC0415

        spot       = float(market.price)
        intrinsic  = float(dcf.intrinsic_per_share)
        band_low   = float(dcf.band_low)
        band_high  = float(dcf.band_high)

        # Guard: negative intrinsic value breaks % math and inverts band order
        if intrinsic <= 0:
            st.error(
                "⚠️ **DCF estimate invalid:** The model produced a negative intrinsic value, "
                "likely because the company has negative free cash flow or excessive net debt. "
                "The traditional margin-of-safety calculation does not apply here. "
                "Consider relative valuation methods such as **P/S** or **EV/Sales** instead."
            )
            return  # skip all chart rendering below

        # Dynamic valuation alert
        if spot > band_high:
            st.error(f"⚠️ **Warning:** Spot price (USD {spot:,.2f}) is significantly above the fair value band (≤ USD {band_high:,.2f}).")
        elif spot < band_low:
            st.success(f"✅ **Note:** Spot price (USD {spot:,.2f}) is below the fair value band (≥ USD {band_low:,.2f}) — margin of safety present.")
        else:
            st.info(f"🔵 **Note:** Spot price (USD {spot:,.2f}) is within the fair value band (USD {band_low:,.2f} – USD {band_high:,.2f}).")

        # ── Key price comparison cards ───────────────────────────────────────
        ka, kb, kc, kd = st.columns(4)
        with ka:
            st.metric("🔵 Intrinsic Value", f"${intrinsic:,.2f}")
        with kb:
            st.metric("🟢 Fair Band Low", f"${band_low:,.2f}")
        with kc:
            st.metric("🟢 Fair Band High", f"${band_high:,.2f}")
        with kd:
            prem_pct = (spot / intrinsic - 1) * 100
            prem_label = f"+{prem_pct:.1f}%" if prem_pct >= 0 else f"{prem_pct:.1f}%"
            prem_color = "inverse" if prem_pct > 0 else "normal"
            st.metric("🟠 Current Spot", f"${spot:,.2f}",
                      delta=prem_label, delta_color=prem_color,
                      help="Delta = (Spot / Intrinsic − 1) × 100")

        # ── Premium / Discount % bar ─────────────────────────────────────────
        # Works at any scale — always shows % vs intrinsic value
        prem_pct   = (spot / intrinsic - 1) * 100
        band_lo_pct = (band_low  / intrinsic - 1) * 100   # always -10
        band_hi_pct = (band_high / intrinsic - 1) * 100   # always +10

        # x-axis: symmetric around 0, wide enough to show current spot
        x_half = max(50.0, abs(prem_pct) * 1.25)
        x_min_p, x_max_p = -x_half, x_half

        bar_y0, bar_y1 = 0.25, 0.75

        fig2 = go.Figure()

        # Background zones (% scale)
        fig2.add_shape(type="rect",
            x0=x_min_p, x1=band_lo_pct, y0=bar_y0, y1=bar_y1,
            fillcolor="rgba(46,204,113,0.45)", line=dict(color="#2ecc71", width=1))
        fig2.add_shape(type="rect",
            x0=band_lo_pct, x1=band_hi_pct, y0=bar_y0, y1=bar_y1,
            fillcolor="rgba(241,196,15,0.45)", line=dict(color="#f1c40f", width=1))
        fig2.add_shape(type="rect",
            x0=band_hi_pct, x1=x_max_p, y0=bar_y0, y1=bar_y1,
            fillcolor="rgba(231,76,60,0.45)", line=dict(color="#e74c3c", width=1))

        # Zero line (= intrinsic value)
        fig2.add_shape(type="line",
            x0=0, x1=0, y0=bar_y0 - 0.05, y1=bar_y1 + 0.05,
            line=dict(color="rgba(255,255,255,0.6)", width=2, dash="dot"))

        # Current spot marker (blue, clamped to axis)
        spot_x = max(x_min_p * 0.98, min(x_max_p * 0.98, prem_pct))
        fig2.add_shape(type="line",
            x0=spot_x, x1=spot_x, y0=bar_y0, y1=bar_y1,
            line=dict(color="#3498db", width=3))
        fig2.add_trace(go.Scatter(
            x=[spot_x], y=[bar_y1 + 0.18],
            mode="markers+text",
            marker=dict(symbol="diamond", size=12, color="#3498db",
                        line=dict(color="white", width=1.5)),
            text=[f"<b>{prem_pct:+.1f}%</b>  (${spot:,.0f})"],
            textposition="middle right",
            textfont=dict(size=12, color="#3498db"),
            hovertemplate=f"Spot premium: {prem_pct:+.1f}%<extra></extra>",
            showlegend=False,
        ))

        # Zone labels below bar
        for xc, col, lbl in [
            ((x_min_p + band_lo_pct) / 2, "#2ecc71", f"Undervalued<br>below −10%"),
            ((band_lo_pct + band_hi_pct) / 2, "#f1c40f", f"Fair Value<br>−10% to +10%"),
            ((band_hi_pct + x_max_p) / 2, "#e74c3c", f"Overvalued<br>above +10%"),
        ]:
            fig2.add_annotation(x=xc, y=bar_y0 - 0.05, xref="x", yref="y",
                text=lbl, showarrow=False,
                font=dict(size=10, color=col), yanchor="top", xanchor="center")

        # Intrinsic = 0% label
        fig2.add_annotation(x=0, y=bar_y1 + 0.08, xref="x", yref="y",
            text=f"Intrinsic<br>${intrinsic:,.0f}  (0%)",
            showarrow=False, font=dict(size=10, color="rgba(255,255,255,0.7)"),
            yanchor="bottom", xanchor="center")

        fig2.update_layout(
            title=dict(
                text="Premium / Discount to Intrinsic Value  "
                     "<span style='font-size:12px;color:#aaa'>"
                     "(% scale — works at any price ratio)</span>",
                font=dict(size=13, color="white"), x=0,
            ),
            xaxis=dict(
                range=[x_min_p, x_max_p],
                ticksuffix="%", showgrid=False, zeroline=False,
                tickfont=dict(size=11, color="#ccc"),
            ),
            yaxis=dict(range=[0, 1], showgrid=False, zeroline=False,
                       showticklabels=False),
            height=210,
            margin=dict(t=40, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("For education only; not investment advice.")
    else:
        st.info("Need both spot price and DCF output to show margin-of-safety chart.")


main()

