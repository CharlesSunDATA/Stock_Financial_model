"""
Pullback Analyzer — three quantitative lenses on "how much rally before a pullback":
  1. Price Bias Ratio  (distance from 200-day SMA)
  2. Run-up before Drawdown statistics
  3. RSI Overbought Duration analysis
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ── Helper: RSI ──────────────────────────────────────────────────────────────
def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("RSI")


# ── Helper: find run-up / pullback cycles ────────────────────────────────────
def _find_runup_pullback(
    close: pd.Series, pullback_pct: float = 0.05
) -> pd.DataFrame:
    """
    Walk through price history and tag each pullback event.
    A pullback is triggered when price drops ≥ pullback_pct from a recent peak.
    Run-up = (peak - prior cycle start) / prior cycle start.
    """
    prices = close.values
    dates  = close.index
    n      = len(prices)

    events: list[dict] = []
    in_pullback    = False
    peak_i         = 0
    trough_i       = 0
    cycle_start_i  = 0   # trough of the previous pullback → start of current run-up

    for i in range(1, n):
        if not in_pullback:
            if prices[i] > prices[peak_i]:
                peak_i = i
            dd = (prices[i] - prices[peak_i]) / prices[peak_i]
            if dd <= -pullback_pct:
                run_up_pct = (prices[peak_i] - prices[cycle_start_i]) / prices[cycle_start_i] * 100
                days_rally  = (dates[peak_i] - dates[cycle_start_i]).days
                events.append({
                    "rally_start":       dates[cycle_start_i],
                    "rally_start_price": round(float(prices[cycle_start_i]), 2),
                    "peak_date":         dates[peak_i],
                    "peak_price":        round(float(prices[peak_i]), 2),
                    "run_up_pct":        round(run_up_pct, 2),
                    "rally_days":        days_rally,
                    "dd_trigger_date":   dates[i],
                    "dd_at_trigger_pct": round(float(dd) * 100, 2),
                })
                in_pullback = True
                trough_i    = i
        else:
            if prices[i] < prices[trough_i]:
                trough_i = i
            recovery = (prices[i] - prices[trough_i]) / prices[trough_i]
            if recovery >= pullback_pct:
                in_pullback   = False
                cycle_start_i = trough_i
                peak_i        = i

    return pd.DataFrame(events)


# ── Helper: RSI overbought streaks ───────────────────────────────────────────
def _find_ob_streaks(
    close: pd.Series, rsi: pd.Series, ob_level: float = 70
) -> pd.DataFrame:
    """Find contiguous RSI > ob_level streaks and compute forward returns."""
    rows: list[dict] = []
    in_streak   = False
    streak_start = 0
    streak_len   = 0

    for i in range(len(rsi)):
        if pd.isna(rsi.iloc[i]):
            continue
        if rsi.iloc[i] >= ob_level:
            if not in_streak:
                in_streak    = True
                streak_start = i
                streak_len   = 1
            else:
                streak_len  += 1
        else:
            if in_streak and streak_len >= 3:
                end_i = i
                fwd: dict[str, float | None] = {}
                for days in [5, 10, 21]:
                    if end_i + days < len(close):
                        fwd[f"fwd_{days}d_pct"] = round(
                            (close.iloc[end_i + days] - close.iloc[end_i])
                            / close.iloc[end_i] * 100, 2
                        )
                    else:
                        fwd[f"fwd_{days}d_pct"] = None
                rows.append({
                    "streak_start":  rsi.index[streak_start],
                    "streak_end":    rsi.index[end_i - 1],
                    "duration_days": streak_len,
                    "peak_rsi":      round(float(rsi.iloc[streak_start:end_i].max()), 1),
                    **fwd,
                })
            in_streak  = False
            streak_len = 0

    return pd.DataFrame(rows)


# ── Risk index helpers ────────────────────────────────────────────────────────
def _percentile_rank(series: pd.Series, value: float) -> float:
    """Fraction of historical values ≤ value → 0-100 scale."""
    clean = series.dropna()
    if clean.empty:
        return 50.0
    return float((clean <= value).mean() * 100)


def _risk_color(score: float) -> str:
    if score < 30:  return "#2ecc71"   # green — low
    if score < 55:  return "#f1c40f"   # yellow — moderate
    if score < 75:  return "#e67e22"   # orange — high
    return "#e74c3c"                   # red — extreme


def _risk_label(score: float) -> str:
    if score < 30:  return "Low"
    if score < 55:  return "Moderate"
    if score < 75:  return "High"
    return "Extreme"


def _risk_badge(title: str, score: float) -> str:
    """HTML badge showing section title + colour-coded risk score."""
    col  = _risk_color(score)
    lbl  = _risk_label(score)
    return (
        f"<span style='font-size:1.25rem;font-weight:700'>{title}</span>"
        f"&nbsp;&nbsp;"
        f"<span style='background:{col};color:#111;font-weight:700;"
        f"padding:3px 10px;border-radius:12px;font-size:0.9rem'>"
        f"Risk {score:.0f}/100 — {lbl}</span>"
    )


def _mini_gauge(score: float, title: str) -> go.Figure:
    """Compact 160-px horizontal gauge for a 0-100 risk score."""
    col = _risk_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 22, "color": col}},
        gauge={
            "axis": {"range": [0, 100], "showticklabels": False, "ticks": ""},
            "bar": {"color": col, "thickness": 0.30},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(46,204,113,0.20)"},
                {"range": [30, 55], "color": "rgba(241,196,15,0.20)"},
                {"range": [55, 75], "color": "rgba(230,126,34,0.20)"},
                {"range": [75,100], "color": "rgba(231,76,60,0.20)"},
            ],
        },
    ))
    fig.update_layout(
        height=130, margin=dict(t=28, b=0, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
        title=dict(text=f"<b>{title}</b>", font=dict(size=11, color="#aaa"), x=0.5),
    )
    return fig


def calc_bias_risk(cur_bias: float | None, bias_series: pd.Series) -> float:
    """Bias ratio risk: percentile rank of cur_bias in historical distribution."""
    if cur_bias is None or np.isnan(cur_bias):
        return 50.0
    # below SMA (negative bias) → capped at 15
    raw = _percentile_rank(bias_series, cur_bias)
    return round(max(0.0, min(100.0, raw)), 1)


def calc_runup_risk(current_runup: float, historical_runups: pd.Series) -> float:
    """Run-up risk: percentile rank of current rally vs past pre-pullback rallies."""
    if historical_runups.empty or current_runup <= 0:
        return 20.0
    raw = _percentile_rank(historical_runups, current_runup)
    return round(max(0.0, min(100.0, raw)), 1)


def calc_rsi_risk(cur_rsi: float | None, cur_streak: int, ob_level: float) -> float:
    """
    RSI risk = RSI-level component (0-70) + streak penalty (0-30).
    RSI component: linear from ob_level-10 → 0  to  100 → 70.
    Streak: each day above ob_level adds ~2 pts, capped at 30.
    """
    if cur_rsi is None or np.isnan(cur_rsi):
        return 20.0
    rsi_floor = max(ob_level - 20, 40)
    rsi_component = max(0.0, min(70.0, (cur_rsi - rsi_floor) / (100 - rsi_floor) * 70))
    streak_component = min(30.0, cur_streak * 2.5)
    return round(min(100.0, rsi_component + streak_component), 1)


# ── Adaptive pullback baseline ────────────────────────────────────────────────
def calculate_dynamic_pullback(
    close: pd.Series,
    noise_filter: float = 0.03,
    fallback: float = 0.05,
) -> tuple[float, list[float]]:
    """
    Return the **median** of all discrete trough→peak rally legs, using a
    state-machine to detect completed swing legs.

    Algorithm (state machine)
    -------------------------
    State IN_RALLY:
      - Keep extending the peak while price rises.
      - When price falls ≥ noise_filter from the peak → rally leg is complete;
        record it, set new trough, switch to IN_DECLINE.

    State IN_DECLINE:
      - Keep extending the trough while price falls.
      - When price rises ≥ noise_filter from the trough → switch to IN_RALLY.

    Only discrete completed rally legs are stored, so each trough→peak cycle
    is counted exactly once, and sub-3% wiggles are ignored entirely.

    Parameters
    ----------
    close        : daily closing price series
    noise_filter : minimum move to count as a real swing leg (e.g. 0.03 = 3 %)
    fallback     : returned when no qualifying rally is found (default 5 %)

    Returns
    -------
    (median_rally_fraction, all_rally_fractions)
    """
    if close.empty or len(close) < 20:
        return fallback, []

    prices = close.values.astype(float)
    rallies: list[float] = []

    trough    = prices[0]
    peak      = prices[0]
    in_rally  = True          # assume we start in an upswing

    for price in prices[1:]:
        if in_rally:
            if price > peak:
                peak = price                          # extend the rally
            elif (peak - price) / peak >= noise_filter:
                # Reversal large enough → record completed rally leg
                leg = (peak - trough) / trough
                if leg >= noise_filter:
                    rallies.append(leg)
                trough   = price
                peak     = price
                in_rally = False
        else:
            if price < trough:
                trough = price                        # extend the decline
            elif (price - trough) / trough >= noise_filter:
                # Reversal upward → start a new rally leg
                peak     = price
                in_rally = True

    # Capture the final open rally leg
    if in_rally:
        leg = (peak - trough) / trough
        if leg >= noise_filter:
            rallies.append(leg)

    if not rallies:
        return fallback, []

    return float(np.median(rallies)), rallies


# ── Stock adaptive pullback baseline (median swing drawdown) ──────────────────
def calculate_stock_pullback_baseline(
    close: pd.Series,
    noise_filter: float = 0.03,
    fallback: float = 0.05,
) -> tuple[float, list[float]]:
    """
    For single stocks: compute a pullback threshold as the **median** depth of
    discrete peak→trough drawdowns, ignoring anything smaller than `noise_filter`.

    This is a ZigZag-style swing detector:
    - Track peaks while rising.
    - When price drops ≥ noise_filter from the peak, enter drawdown tracking.
    - Track trough while falling.
    - When price rebounds ≥ noise_filter from the trough, the drawdown is
      considered complete and recorded exactly once.

    Returns (median_drawdown_fraction, all_drawdowns).
    """
    if close.empty or len(close) < 20:
        return fallback, []

    prices = close.values.astype(float)
    drawdowns: list[float] = []

    peak = prices[0]
    trough = prices[0]
    in_drawdown = False

    for price in prices[1:]:
        if not in_drawdown:
            if price > peak:
                peak = price
            elif (peak - price) / peak >= noise_filter:
                # start drawdown
                trough = price
                in_drawdown = True
        else:
            if price < trough:
                trough = price
            elif (price - trough) / trough >= noise_filter:
                # drawdown completed
                dd = (peak - trough) / peak
                if dd >= noise_filter:
                    drawdowns.append(dd)
                peak = price
                trough = price
                in_drawdown = False

    # capture an open drawdown at the end
    if in_drawdown:
        dd = (peak - trough) / peak
        if dd >= noise_filter:
            drawdowns.append(dd)

    if not drawdowns:
        return fallback, []

    return float(np.median(drawdowns)), drawdowns


# ── Shared layout defaults ────────────────────────────────────────────────────
_L = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", size=11),
    margin=dict(t=40, b=0, l=0, r=0),
    hovermode="x unified",
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False),
)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("📉 Pullback Analyzer")
    st.caption(
        "Three quantitative lenses: **Bias Ratio** (distance from 200-day SMA), "
        "**Run-up before Drawdown** statistics, and **RSI Overbought Duration** analysis. "
        "Data via yfinance — for research only, not investment advice."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Parameters")
        ticker     = st.text_input("Ticker", value="QQQ").strip().upper() or "QQQ"
        years      = st.slider("Lookback (years)", 3, 15, 10)
        auto_stock_pb = st.checkbox(
            "Auto-calculate pullback threshold (stocks only)",
            value=True,
            help=(
                "For single stocks, automatically compute the median swing drawdown "
                "depth over the lookback period, ignoring noise below the filter. "
                "For index ETFs, the manual slider is used (default 5%)."
            ),
        )
        stock_noise_filter_pct = st.slider(
            "Noise filter for stock auto-calc (%)",
            min_value=1, max_value=10, value=3,
            help="Swing drawdowns smaller than this are ignored as noise.",
            disabled=not auto_stock_pb,
        ) / 100

        pullback_pct = st.slider("Pullback threshold (%)", 3, 20, 5) / 100

        ob_level   = st.slider("RSI overbought level", 65, 80, 70)
        sma_period = st.selectbox("Bias ratio SMA", [50, 100, 200], index=2)
        run_btn    = st.button("Run Analysis", type="primary", use_container_width=True)

    if "pa_df" not in st.session_state:
        run_btn = True  # auto-run on first load

    if run_btn:
        end   = date.today()
        start = end - timedelta(days=365 * years + 30)
        with st.spinner(f"Downloading {ticker} ({years}y)…"):
            try:
                raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
                if raw.empty:
                    st.error(f"No data for {ticker}.")
                    return
                close = raw["Close"].squeeze().dropna()
                close.index = pd.to_datetime(close.index)
                st.session_state.update({
                    "pa_close":  close,
                    "pa_ticker": ticker,
                    "pa_sma":    sma_period,
                    "pa_ob":     ob_level,
                    "pa_df":     True,
                })
            except Exception as e:
                st.error(str(e))
                return
    elif st.session_state.get("pa_ticker") != ticker:
        st.warning("Ticker changed — click **Run Analysis** to reload.")

    if "pa_close" not in st.session_state:
        return

    close      = st.session_state["pa_close"]
    sma_period = st.session_state["pa_sma"]
    ob_level   = st.session_state["pa_ob"]
    sym        = st.session_state["pa_ticker"]
    x          = close.index

    # ── Resolve pullback threshold: index ETFs fixed/manual, stocks auto ──────
    INDEX_ETFS = {"QQQ", "SPY", "DIA", "IWM", "VTI", "VOO"}
    if auto_stock_pb and sym not in INDEX_ETFS:
        auto_pb, dds = calculate_stock_pullback_baseline(
            close, noise_filter=stock_noise_filter_pct, fallback=pullback_pct
        )
        pullback_pct = auto_pb
        st.sidebar.info(
            f"📐 **{sym}** stock auto pullback: **{pullback_pct*100:.1f}%**\n\n"
            f"Median of **{len(dds)}** swing drawdowns (≥ {stock_noise_filter_pct*100:.0f}%) "
            f"over {years}y"
        )
    elif sym in INDEX_ETFS:
        st.sidebar.caption(f"Index/ETF preset: using manual threshold (**{pullback_pct*100:.0f}%**).")

    sma         = close.rolling(sma_period).mean()
    bias_pct    = (close - sma) / sma * 100
    rsi         = _calc_rsi(close)

    # ── Current snapshot ─────────────────────────────────────────────────────
    cur_price  = float(close.iloc[-1])
    cur_sma    = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None
    cur_bias   = float(bias_pct.iloc[-1]) if cur_sma else None
    cur_rsi    = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

    # ── Current RSI overbought streak length ──────────────────────────────────
    cur_ob_streak = 0
    for v in reversed(rsi.dropna().values):
        if v >= ob_level:
            cur_ob_streak += 1
        else:
            break

    # ── Pre-compute risk indices ──────────────────────────────────────────────
    events_pre   = _find_runup_pullback(close, pullback_pct)
    if not events_pre.empty:
        last_peak_row  = events_pre.iloc[-1]
        last_cycle_end = pd.to_datetime(last_peak_row["dd_trigger_date"])
        recent_slice   = close[close.index > last_cycle_end]
        cur_runup = (
            (recent_slice.max() - recent_slice.min()) / recent_slice.min() * 100
            if not recent_slice.empty else 0.0
        )
        hist_runups = events_pre["run_up_pct"]
    else:
        cur_runup   = 0.0
        hist_runups = pd.Series([], dtype=float)

    r_bias   = calc_bias_risk(cur_bias, bias_pct)
    r_runup  = calc_runup_risk(cur_runup, hist_runups)
    r_rsi    = calc_rsi_risk(cur_rsi, cur_ob_streak, ob_level)
    r_total  = round((r_bias + r_runup + r_rsi) / 3, 1)

    # ── Overall risk dashboard ────────────────────────────────────────────────
    st.markdown("### Overall Risk Dashboard")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.plotly_chart(_mini_gauge(r_total,  "⚡ Overall Risk"), use_container_width=True)
    with g2:
        st.plotly_chart(_mini_gauge(r_bias,   f"1. Bias Ratio (SMA{sma_period})"), use_container_width=True)
    with g3:
        st.plotly_chart(_mini_gauge(r_runup,  f"2. Run-up (vs ≥{pullback_pct*100:.0f}% DD)"), use_container_width=True)
    with g4:
        st.plotly_chart(_mini_gauge(r_rsi,    f"3. RSI Overbought (>{ob_level})"), use_container_width=True)

    oc = _risk_color(r_total)
    st.markdown(
        f"<div style='background:rgba(255,255,255,0.05);border-left:4px solid {oc};"
        f"padding:10px 16px;border-radius:6px;margin-bottom:8px'>"
        f"<b style='color:{oc}'>Combined Risk: {r_total:.0f}/100 — {_risk_label(r_total)}</b>  "
        f"&nbsp;·&nbsp; Bias {r_bias:.0f} / Run-up {r_runup:.0f} / RSI {r_rsi:.0f}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    ca, cb, cc, cd = st.columns(4)
    with ca:
        st.metric("Current Price", f"${cur_price:,.2f}")
    with cb:
        sma_lbl = f"${cur_sma:,.2f}" if cur_sma else "—"
        st.metric(f"SMA{sma_period}", sma_lbl)
    with cc:
        bias_lbl = f"{cur_bias:+.1f}%" if cur_bias is not None else "—"
        bias_delta_color = "inverse" if (cur_bias or 0) > 15 else "normal"
        st.metric("Bias Ratio", bias_lbl, delta=bias_lbl, delta_color=bias_delta_color)
    with cd:
        rsi_lbl = f"{cur_rsi:.1f}" if cur_rsi else "—"
        st.metric("RSI(14)", rsi_lbl,
                  delta="Overbought" if (cur_rsi or 0) >= ob_level else "Normal",
                  delta_color="inverse" if (cur_rsi or 0) >= ob_level else "normal")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Bias Ratio
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(
        _risk_badge(f"1. Price Bias Ratio — distance from SMA{sma_period}", r_bias),
        unsafe_allow_html=True,
    )
    st.markdown("")

    fig_bias = go.Figure()
    fig_bias.add_trace(go.Scatter(
        x=x, y=close, name="Price",
        line=dict(color="#3498db", width=1.5),
        hovertemplate="%{y:,.2f}<extra>Price</extra>",
    ))
    fig_bias.add_trace(go.Scatter(
        x=x, y=sma, name=f"SMA{sma_period}",
        line=dict(color="#f39c12", width=1.5, dash="dot"),
        hovertemplate="%{y:,.2f}<extra>SMA</extra>",
    ))
    fig_bias.update_layout(**_L,
        height=260,
        title=dict(text=f"{sym} Price vs SMA{sma_period}", font=dict(size=13), x=0),
        yaxis_tickprefix="$",
        legend=dict(orientation="h", y=1.12, x=0),
    )
    st.plotly_chart(fig_bias, use_container_width=True)

    # Bias % over time
    bias_colors = ["rgba(231,76,60,0.7)" if v > 15
                   else "rgba(241,196,15,0.7)" if v > 8
                   else "rgba(46,204,113,0.7)"
                   for v in bias_pct.fillna(0)]
    fig_bpct = go.Figure()
    fig_bpct.add_trace(go.Bar(
        x=x, y=bias_pct,
        marker_color=bias_colors,
        name="Bias %",
        hovertemplate="%{y:+.1f}%<extra>Bias</extra>",
    ))
    fig_bpct.add_hline(y=15, line_color="#e74c3c", line_dash="dash", line_width=1,
                        annotation_text="⚠ +15% caution", annotation_position="top left",
                        annotation_font_color="#e74c3c")
    fig_bpct.add_hline(y=20, line_color="#c0392b", line_dash="dash", line_width=1.5,
                        annotation_text="🚨 +20% extreme", annotation_position="top left",
                        annotation_font_color="#c0392b")
    fig_bpct.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    fig_bpct.update_layout(**_L,
        height=220,
        title=dict(text=f"Bias Ratio (%) — how far above SMA{sma_period}", font=dict(size=13), x=0),
        yaxis_ticksuffix="%",
    )
    st.plotly_chart(fig_bpct, use_container_width=True)

    # Historical context
    p75_bias = float(np.nanpercentile(bias_pct, 75))
    p90_bias = float(np.nanpercentile(bias_pct, 90))
    if cur_bias is not None:
        _bias_col = _risk_color(r_bias)
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.05);border-left:4px solid {_bias_col};"
            f"padding:8px 14px;border-radius:6px;font-size:0.9rem'>"
            f"📊 <b>Current bias: {cur_bias:+.1f}%</b> — "
            f"Historical {years}y: P75 = {p75_bias:+.1f}%, P90 = {p90_bias:+.1f}%"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Run-up before Drawdown
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(
        _risk_badge(f"2. Run-up statistics before ≥{pullback_pct*100:.0f}% pullback", r_runup),
        unsafe_allow_html=True,
    )
    st.markdown("")

    events = events_pre  # reuse already-computed result

    if events.empty:
        st.info(f"No pullbacks ≥ {pullback_pct*100:.0f}% found in this period.")
    else:
        n_events   = len(events)
        avg_runup  = events["run_up_pct"].mean()
        med_runup  = events["run_up_pct"].median()
        p25_runup  = events["run_up_pct"].quantile(0.25)
        p75_runup  = events["run_up_pct"].quantile(0.75)

        # Current run-up (from last cycle bottom)
        last_peak_row   = events.iloc[-1]
        last_cycle_end  = pd.to_datetime(last_peak_row["dd_trigger_date"])
        recent_slice    = close[close.index > last_cycle_end]
        if not recent_slice.empty:
            recent_low      = recent_slice.min()
            recent_high     = recent_slice.max()
            current_runup   = (recent_high - recent_low) / recent_low * 100
        else:
            current_runup   = 0.0

        ka, kb, kc, kd, ke = st.columns(5)
        with ka:
            danger = current_runup >= med_runup
            st.metric(
                "Current run-up (est.)", f"{current_runup:.1f}%",
                delta="⚠ Near median threshold" if danger else "Below median",
                delta_color="inverse" if danger else "normal",
            )
        with kb:
            st.metric("Avg run-up before pullback", f"{avg_runup:.1f}%")
        with kc:
            st.metric("Median run-up", f"{med_runup:.1f}%")
        with kd:
            st.metric("P25–P75 range", f"{p25_runup:.1f}%–{p75_runup:.1f}%")
        with ke:
            st.metric("Pullback events", n_events)

        _runup_col = _risk_color(r_runup)
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.05);border-left:4px solid {_runup_col};"
            f"padding:8px 14px;border-radius:6px;font-size:0.9rem;margin-bottom:8px'>"
            f"📈 <b>Current rally since last pullback: {cur_runup:.1f}%</b> — "
            f"Median pre-pullback run-up: {med_runup:.1f}%, P75: {p75_runup:.1f}%</div>",
            unsafe_allow_html=True,
        )

        # Histogram of run-ups
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=events["run_up_pct"], nbinsx=20,
            marker_color="rgba(52,152,219,0.7)",
            marker_line=dict(color="rgba(52,152,219,1)", width=1),
            name="Run-up distribution",
            hovertemplate="Run-up: %{x:.1f}%<br>Count: %{y}<extra></extra>",
        ))
        fig_hist.add_vline(x=med_runup, line_color="#f1c40f", line_dash="dash",
                           annotation_text=f"Median {med_runup:.1f}%",
                           annotation_font_color="#f1c40f",
                           annotation_position="top right")
        if current_runup > 0:
            fig_hist.add_vline(x=current_runup, line_color="#e74c3c", line_width=2,
                               annotation_text=f"Now {current_runup:.1f}%",
                               annotation_font_color="#e74c3c",
                               annotation_position="top left")
        fig_hist.update_layout(**_L,
            height=280,
            title=dict(text=f"Distribution of run-ups before ≥{pullback_pct*100:.0f}% pullbacks", font=dict(size=13), x=0),
            xaxis_ticksuffix="%",
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Scatter: run-up vs drawdown severity
        fig_sc = go.Figure(go.Scatter(
            x=events["run_up_pct"], y=events["dd_at_trigger_pct"].abs(),
            mode="markers+text",
            marker=dict(
                size=10, color=events["run_up_pct"],
                colorscale="RdYlGn_r", showscale=True,
                colorbar=dict(title="Run-up %", thickness=12),
            ),
            text=[str(d.year) for d in pd.to_datetime(events["peak_date"])],
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Run-up: %{x:.1f}%<br>"
                "Pullback depth: -%{y:.1f}%<extra></extra>"
            ),
        ))
        fig_sc.update_layout(**{
            **_L,
            "height": 300,
            "title": dict(text="Run-up vs Pullback depth (each dot = one event)", font=dict(size=13), x=0),
            "xaxis": dict(**_L["xaxis"], title="Run-up before pullback (%)", ticksuffix="%"),
            "yaxis": dict(**_L["yaxis"], title="Pullback depth (%)", ticksuffix="%"),
        })
        st.plotly_chart(fig_sc, use_container_width=True)

        # Detailed table
        with st.expander("All pullback events (detail table)", expanded=False):
            disp = events[["rally_start", "peak_date", "run_up_pct", "rally_days",
                            "dd_trigger_date", "dd_at_trigger_pct"]].copy()
            disp.columns = ["Rally start", "Peak date", "Run-up %", "Rally days",
                             "Pullback triggered", "DD at trigger %"]
            disp["Run-up %"]         = disp["Run-up %"].map("{:.1f}%".format)
            disp["DD at trigger %"]  = disp["DD at trigger %"].map("{:.1f}%".format)
            st.dataframe(disp, hide_index=True, use_container_width=True)

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. RSI Overbought Duration
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(
        _risk_badge(f"3. RSI Overbought Duration (RSI > {ob_level})", r_rsi),
        unsafe_allow_html=True,
    )
    st.markdown("")

    # RSI chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=x, y=rsi, name="RSI(14)",
        line=dict(color="#9b59b6", width=1.5),
        hovertemplate="RSI: %{y:.1f}<extra></extra>",
    ))
    ob_fill = rsi.where(rsi >= ob_level)
    fig_rsi.add_trace(go.Scatter(
        x=x, y=ob_fill, name=f"Overbought (>{ob_level})",
        fill="tozeroy",
        fillcolor="rgba(231,76,60,0.20)",
        line=dict(color="rgba(231,76,60,0.6)", width=0.5),
        hoverinfo="skip",
    ))
    fig_rsi.add_hline(y=ob_level, line_color="#e74c3c", line_dash="dash", line_width=1)
    fig_rsi.add_hline(y=30, line_color="#2ecc71", line_dash="dash", line_width=1)
    fig_rsi.update_layout(**{
        **_L,
        "height": 220,
        "title": dict(text="RSI(14) — overbought zones highlighted", font=dict(size=13), x=0),
        "yaxis": dict(**_L["yaxis"], range=[0, 100]),
    })
    st.plotly_chart(fig_rsi, use_container_width=True)

    # Current streak info
    _rsi_col = _risk_color(r_rsi)
    if cur_ob_streak > 0:
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.05);border-left:4px solid {_rsi_col};"
            f"padding:8px 14px;border-radius:6px;font-size:0.9rem;margin-bottom:8px'>"
            f"⏱ <b>Current RSI overbought streak:</b> {cur_ob_streak} day(s) "
            f"(RSI = {cur_rsi:.1f})"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        _rsi_txt = f"{cur_rsi:.1f}" if cur_rsi is not None else "—"
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.04);border-left:4px solid #2ecc71;"
            f"padding:8px 14px;border-radius:6px;font-size:0.9rem;margin-bottom:8px'>"
            f"✅ <b>Not currently in overbought territory</b> (RSI = {_rsi_txt})"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Overbought streaks analysis
    streaks = _find_ob_streaks(close, rsi, ob_level)

    if streaks.empty:
        st.info(f"No RSI > {ob_level} streaks of ≥ 3 days found.")
    else:
        avg_dur = streaks["duration_days"].mean()
        fwd5  = streaks["fwd_5d_pct"].dropna().mean()
        fwd10 = streaks["fwd_10d_pct"].dropna().mean()
        fwd21 = streaks["fwd_21d_pct"].dropna().mean()

        sa, sb, sc, sd, se = st.columns(5)
        with sa:
            st.metric("OB streaks found", len(streaks))
        with sb:
            st.metric("Avg streak length", f"{avg_dur:.1f} days")
        with sc:
            st.metric("Avg return 5d after", f"{fwd5:+.1f}%" if not np.isnan(fwd5) else "—",
                      delta_color="inverse" if fwd5 < 0 else "normal")
        with sd:
            st.metric("Avg return 10d after", f"{fwd10:+.1f}%" if not np.isnan(fwd10) else "—",
                      delta_color="inverse" if fwd10 < 0 else "normal")
        with se:
            st.metric("Avg return 21d after", f"{fwd21:+.1f}%" if not np.isnan(fwd21) else "—",
                      delta_color="inverse" if fwd21 < 0 else "normal")

        # Forward returns by streak length (scatter)
        long_streaks = streaks[streaks["duration_days"] >= 5]
        short_streaks = streaks[streaks["duration_days"] < 5]

        fig_fwd = go.Figure()
        for grp, col, name in [
            (short_streaks, "#f39c12", "Short OB streak (<5d)"),
            (long_streaks,  "#e74c3c", "Extended OB streak (≥5d)"),
        ]:
            if grp.empty:
                continue
            fig_fwd.add_trace(go.Scatter(
                x=grp["duration_days"], y=grp["fwd_21d_pct"],
                mode="markers",
                marker=dict(size=9, color=col, opacity=0.8,
                            line=dict(color="white", width=0.5)),
                name=name,
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    "Streak: %{x} days<br>"
                    "21d fwd return: %{y:+.1f}%<extra></extra>"
                ),
                customdata=[str(d.date()) for d in pd.to_datetime(grp["streak_end"])],
            ))
        fig_fwd.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
        fig_fwd.update_layout(**{
            **_L,
            "height": 280,
            "title": dict(text="OB streak duration vs 21-day forward return", font=dict(size=13), x=0),
            "xaxis": dict(**_L["xaxis"], title="Streak length (days)", ticksuffix="d"),
            "yaxis": dict(**_L["yaxis"], title="21d fwd return (%)", ticksuffix="%"),
            "legend": dict(orientation="h", y=1.12, x=0),
        })
        st.plotly_chart(fig_fwd, use_container_width=True)

        with st.expander("OB streak detail table", expanded=False):
            disp_s = streaks.copy()
            for c in ["fwd_5d_pct", "fwd_10d_pct", "fwd_21d_pct"]:
                disp_s[c] = disp_s[c].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
            disp_s.columns = ["Start", "End", "Days", "Peak RSI",
                               "Fwd 5d %", "Fwd 10d %", "Fwd 21d %"]
            st.dataframe(disp_s, hide_index=True, use_container_width=True)

    st.divider()
    st.caption(
        "**Methodology:** Bias ratio = (Price − SMAₙ) / SMAₙ.  "
        "Run-up cycles: price rises from prior trough until a ≥X% drawdown is triggered.  "
        "RSI streaks: contiguous days with RSI ≥ threshold (min 3 days).  "
        "For research only — not investment advice."
    )


main()
