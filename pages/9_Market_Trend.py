"""
Market Trend Analysis — Theme Rotation & Market Radar

Uses local SQLite prices_eod to evaluate:
- Market regime and breadth
- Theme momentum and rotation across sectors
- Leading and lagging tickers with return data
- Actionable watchlist implications
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

from utils.local_data import connect_readonly

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "quant_data.db"
PRICE_LOOKBACK_DAYS = 560
UPDATE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update_prices_eod.py"

# ── Theme map: theme name → representative tickers ────────────────────────────
THEMES: dict[str, list[str]] = {
    "AI Infrastructure":      ["NVDA", "AMD", "AVGO", "MRVL", "SMCI", "TSM", "ARM"],
    "Semiconductors":         ["INTC", "MU", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "ASML", "ON", "MCHP"],
    "Optical Communications": ["LITE", "COHR", "CIEN", "AAOI", "VIAV"],
    "Networking Equipment":   ["ANET", "CSCO", "JNPR", "KEYS"],
    "Cloud / Hyperscalers":   ["MSFT", "AMZN", "GOOGL", "META", "ORCL", "CRM"],
    "Memory":                 ["MU", "WDC", "STX"],
    "Software":               ["ADBE", "NOW", "SNOW", "CRWD", "ZS", "PANW"],
    "Financials":             ["JPM", "BAC", "GS", "V", "MA"],
    "Energy":                 ["XOM", "CVX", "COP", "SLB"],
}

# Reverse map: ticker → theme (first listed theme wins for shared tickers like MU)
TICKER_THEME: dict[str, str] = {}
for _theme, _tickers in THEMES.items():
    for _t in _tickers:
        if _t not in TICKER_THEME:
            TICKER_THEME[_t] = _theme


@dataclass(frozen=True)
class Regime:
    label: str
    tone: str
    score: int
    details: list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    return connect_readonly(DB_PATH)


def _run_eod_update(days: int) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(UPDATE_SCRIPT), "--mode", "bulk", "--days", str(max(1, int(days)))]
    return subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]),
                          text=True, capture_output=True, check=False)


def _pct(value, digits: int = 1) -> str:
    try:
        f = float(value)
        return f"{f:.{digits}f}%" if np.isfinite(f) else "-"
    except (TypeError, ValueError):
        return "-"


def _fmt_num(value) -> str:
    try:
        f = float(value)
        return f"{f:,.0f}" if np.isfinite(f) else "-"
    except (TypeError, ValueError):
        return "-"


def _fmt_ret(v) -> str:
    try:
        f = float(v)
        return f"{f:+.1f}%" if np.isfinite(f) else "-"
    except (TypeError, ValueError):
        return "-"


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60 * 10, show_spinner=False)
def _latest_price_date() -> str | None:
    with _connect() as conn:
        return conn.execute("SELECT MAX(price_date) FROM prices_eod").fetchone()[0]


@st.cache_data(ttl=60 * 10, show_spinner=False)
def _load_market_frame(lookback_days: int = PRICE_LOOKBACK_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = _latest_price_date()
    if not latest:
        return pd.DataFrame(), pd.DataFrame()

    start = (pd.to_datetime(latest).date() - timedelta(days=lookback_days)).isoformat()
    with _connect() as conn:
        universe = pd.read_sql(
            """
            WITH latest_rows AS (
                SELECT ticker, MAX(price_date) AS latest_price_date
                FROM prices_eod
                GROUP BY ticker
            ),
            mcap AS (
                SELECT k.ticker, k.market_cap
                FROM key_metrics_ttm k
                JOIN (
                    SELECT ticker, MAX(as_of_date) AS as_of_date
                    FROM key_metrics_ttm
                    GROUP BY ticker
                ) lm ON lm.ticker = k.ticker AND lm.as_of_date = k.as_of_date
            )
            SELECT lr.ticker, lr.latest_price_date, cp.company_name, cp.sector, mcap.market_cap
            FROM latest_rows lr
            LEFT JOIN company_profile cp ON cp.ticker = lr.ticker
            LEFT JOIN mcap ON mcap.ticker = lr.ticker
            WHERE lr.latest_price_date = ?
              AND cp.company_name IS NOT NULL
              AND cp.sector IS NOT NULL
              AND length(lr.ticker) <= 5
              AND lr.ticker NOT LIKE '%.%'
              AND lr.ticker NOT LIKE '%-%'
            """,
            conn, params=(latest,),
        )

        tickers = universe["ticker"].dropna().astype(str).tolist()
        if not tickers:
            return pd.DataFrame(), pd.DataFrame()

        placeholders = ",".join("?" for _ in tickers)
        prices = pd.read_sql(
            f"""
            SELECT ticker, price_date,
                   COALESCE(adj_close, close) AS price, close, volume
            FROM prices_eod
            WHERE price_date >= ?
              AND ticker IN ({placeholders})
              AND COALESCE(adj_close, close) IS NOT NULL
              AND COALESCE(adj_close, close) > 0
            ORDER BY ticker, price_date
            """,
            conn, params=[start, *tickers],
        )

    if prices.empty:
        return pd.DataFrame(), universe

    prices["price_date"] = pd.to_datetime(prices["price_date"])
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce").fillna(0.0)
    prices = prices.dropna(subset=["price"]).sort_values(["ticker", "price_date"])

    grp = prices.groupby("ticker", group_keys=False)
    prices["ma20"]  = grp["price"].rolling(20,  min_periods=15).mean().reset_index(level=0, drop=True)
    prices["ma50"]  = grp["price"].rolling(50,  min_periods=40).mean().reset_index(level=0, drop=True)
    prices["ma150"] = grp["price"].rolling(150, min_periods=100).mean().reset_index(level=0, drop=True)
    prices["ma200"] = grp["price"].rolling(200, min_periods=150).mean().reset_index(level=0, drop=True)
    prices["vol50"] = grp["volume"].rolling(50, min_periods=20).mean().reset_index(level=0, drop=True)
    prices["hi252"] = grp["price"].rolling(252, min_periods=120).max().reset_index(level=0, drop=True)
    prices["lo252"] = grp["price"].rolling(252, min_periods=120).min().reset_index(level=0, drop=True)
    prices["ret1"]  = grp["price"].pct_change()

    return prices.merge(universe, on="ticker", how="left"), universe


def _latest_rows(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    idx = prices.groupby("ticker")["price_date"].idxmax()
    latest = prices.loc[idx].copy()
    latest["above_20ma"]  = latest["price"] > latest["ma20"]
    latest["above_50ma"]  = latest["price"] > latest["ma50"]
    latest["above_150ma"] = latest["price"] > latest["ma150"]
    latest["above_200ma"] = latest["price"] > latest["ma200"]
    latest["new_high"]     = latest["price"] >= latest["hi252"]
    latest["new_low"]      = latest["price"] <= latest["lo252"]
    latest["volume_expanded"] = (latest["volume"] > latest["vol50"] * 1.5) & (latest["volume"] > 0)
    return latest


def _compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Multi-period returns per ticker (requires at least min_periods rows)."""
    if prices.empty:
        return pd.DataFrame()
    rows = []
    for ticker, grp in prices.groupby("ticker"):
        grp = grp.sort_values("price_date").reset_index(drop=True)
        px = grp["price"]
        latest_px = float(px.iloc[-1])

        def _r(n: int) -> float | None:
            if len(px) <= n:
                return None
            past = float(px.iloc[-n - 1])
            return (latest_px / past - 1) * 100 if past > 0 else None

        rows.append({
            "ticker":   ticker,
            "ret_5d":   _r(5),
            "ret_21d":  _r(21),
            "ret_63d":  _r(63),
            "ret_126d": _r(126),
        })
    return pd.DataFrame(rows)


def _make_segments(latest: pd.DataFrame) -> dict[str, list[str]]:
    ranked = latest.dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False)
    growth_sectors = {"Technology", "Communication Services", "Consumer Cyclical"}
    return {
        "Large-cap proxy":  ranked.head(500)["ticker"].tolist(),
        "Growth proxy":     ranked[ranked["sector"].isin(growth_sectors)].head(800)["ticker"].tolist(),
        "Small-cap proxy":  ranked.iloc[1000:3000]["ticker"].tolist(),
    }


def _segment_stats(latest: pd.DataFrame, segments: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for name, tickers in segments.items():
        sub = latest[latest["ticker"].isin(tickers)]
        if sub.empty:
            continue
        rows.append({
            "Market":      name,
            "Stocks":      len(sub),
            ">20MA %":     sub["above_20ma"].mean() * 100 if "above_20ma" in sub.columns else None,
            ">50MA %":     sub["above_50ma"].mean() * 100,
            ">150MA %":    sub["above_150ma"].mean() * 100 if "above_150ma" in sub.columns else None,
            ">200MA %":    sub["above_200ma"].mean() * 100,
            "New highs":   int(sub["new_high"].sum()),
            "New lows":    int(sub["new_low"].sum()),
        })
    return pd.DataFrame(rows)


def _history_by_date(prices: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    sub = prices[prices["ticker"].isin(list(tickers))].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["above_50ma"]  = sub["price"] > sub["ma50"]
    sub["above_200ma"] = sub["price"] > sub["ma200"]
    sub["new_high"]    = sub["price"] >= sub["hi252"]
    sub["new_low"]     = sub["price"] <= sub["lo252"]
    hist = (
        sub.groupby("price_date")
        .agg(
            breadth_50=("above_50ma", "mean"),
            breadth_200=("above_200ma", "mean"),
            new_highs=("new_high", "sum"),
            new_lows=("new_low", "sum"),
            avg_return=("ret1", "mean"),
            stocks=("ticker", "nunique"),
        )
        .reset_index()
    )
    hist["breadth_50"]  *= 100
    hist["breadth_200"] *= 100
    hist["advance_index"] = (1 + hist["avg_return"].fillna(0)).cumprod() * 100
    return hist


def _classify_market(latest: pd.DataFrame, stats: pd.DataFrame) -> Regime:
    if latest.empty or stats.empty:
        return Regime("Insufficient Data", "info", 0,
                      ["prices_eod does not have enough usable data to classify the market trend."])

    overall_50  = latest["above_50ma"].mean() * 100
    overall_200 = latest["above_200ma"].mean() * 100
    hi_lo_spread = int(latest["new_high"].sum() - latest["new_low"].sum())
    volume_exp   = latest["volume_expanded"].mean() * 100
    sp_proxy     = stats[stats["Market"] == "Large-cap proxy"]

    score = 0
    details: list[str] = []

    if overall_50 >= 60:
        score += 2
        details.append(f"50MA breadth strong: {_pct(overall_50)} of stocks are above 50MA.")
    elif overall_50 >= 45:
        score += 1
        details.append(f"50MA breadth neutral: {_pct(overall_50)} above 50MA.")
    else:
        score -= 2
        details.append(f"50MA breadth weak: only {_pct(overall_50)} above 50MA.")

    if overall_200 >= 55:
        score += 2
        details.append(f"Long-term breadth supportive: {_pct(overall_200)} above 200MA.")
    elif overall_200 < 40:
        score -= 2
        details.append(f"Long-term breadth poor: only {_pct(overall_200)} above 200MA.")

    if hi_lo_spread > 0:
        score += 1
        details.append(f"New highs exceed new lows by {hi_lo_spread} stocks.")
    else:
        score -= 1
        details.append(f"New lows match or exceed new highs (spread {hi_lo_spread:+d}).")

    if not sp_proxy.empty:
        row = sp_proxy.iloc[0]
        s20  = float(row.get(">20MA %",  0) or 0)
        s50  = float(row.get(">50MA %",  0) or 0)
        s200 = float(row.get(">200MA %", 0) or 0)
        if s20 >= s50 >= 50 and s200 >= 50:
            score += 1
            details.append("Large-cap proxy shows aligned short/intermediate/long-term support.")
        elif s20 < 40 and s50 < 45:
            score -= 1
            details.append("Large-cap proxy losing short and intermediate-term support.")

    if volume_exp >= 18 and overall_50 >= 70:
        return Regime("Overheated / Overbought", "warning", score,
                      details + [f"Volume expansion broad ({_pct(volume_exp)}) while breadth already elevated."])
    if score >= 4:
        return Regime("Bull Trend Continuation", "success", score, details)
    if score >= 1:
        return Regime("Range-Bound / Choppy", "info", score, details)
    if score >= -2:
        return Regime("Weakening Trend", "warning", score, details)
    return Regime("High Risk / Bear", "error", score, details)


def _theme_rotation(latest: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    df = latest.merge(returns[["ticker", "ret_5d", "ret_21d", "ret_63d"]], on="ticker", how="left")
    rows = []
    for theme, tickers in THEMES.items():
        sub = df[df["ticker"].isin(tickers)]
        in_db = len(sub)
        rows.append({
            "Theme":   theme,
            "In DB":   in_db,
            "Tickers": ", ".join(sorted(sub["ticker"].tolist())) if in_db else "—",
            "1W %":    sub["ret_5d"].mean()  if in_db and sub["ret_5d"].notna().any()  else None,
            "1M %":    sub["ret_21d"].mean() if in_db and sub["ret_21d"].notna().any() else None,
            "3M %":    sub["ret_63d"].mean() if in_db and sub["ret_63d"].notna().any() else None,
            ">50MA":   sub["above_50ma"].mean() * 100  if in_db else None,
            ">200MA":  sub["above_200ma"].mean() * 100 if in_db else None,
        })
    theme_df = pd.DataFrame(rows).sort_values("1M %", ascending=False, na_position="last").reset_index(drop=True)
    theme_df.insert(0, "Rank", range(1, len(theme_df) + 1))
    return theme_df


def _leading_lagging(latest: pd.DataFrame, returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = latest.merge(returns[["ticker", "ret_5d", "ret_21d", "ret_63d"]], on="ticker", how="left")
    df["theme"] = df["ticker"].map(TICKER_THEME).fillna("Other")
    has_ret = df.dropna(subset=["ret_21d"])
    if has_ret.empty:
        return pd.DataFrame(), pd.DataFrame()
    cols = [c for c in ["ticker", "company_name", "theme", "price", "ret_5d", "ret_21d", "ret_63d",
                         "above_50ma", "above_200ma", "new_high", "new_low"] if c in has_ret.columns]
    ranked = has_ret.sort_values("ret_21d", ascending=False)
    return ranked.head(10)[cols].copy(), ranked.tail(10).sort_values("ret_21d")[cols].copy()


def _investor_takeaway(regime: Regime, theme_df: pd.DataFrame) -> str:
    env = {
        "Bull Trend Continuation":  "Risk-on — market internals broadly supportive of new exposure.",
        "Range-Bound / Choppy":     "Mixed — be selective, favor leaders, reduce exposure to laggards.",
        "Weakening Trend":          "Caution — reduce risk, focus on names holding 50MA.",
        "High Risk / Bear":         "Risk-off — protect capital, avoid new longs except the strongest themes.",
        "Overheated / Overbought":  "Overbought conditions — trim extended positions, manage position sizes.",
        "Insufficient Data":        "Not enough data for a reliable market read.",
    }.get(regime.label, f"Current regime: {regime.label}.")

    parts = [env]
    top3 = theme_df.dropna(subset=["1M %"]).head(3)
    bot2 = theme_df.dropna(subset=["1M %"]).tail(2)
    if not top3.empty:
        parts.append(f"Strongest themes (1M): {', '.join(top3['Theme'].tolist())}.")
    if not bot2.empty:
        parts.append(f"Weakest themes: {', '.join(bot2['Theme'].tolist())}.")
    return " ".join(parts)


# ── Charts ────────────────────────────────────────────────────────────────────

def _plot_base(height: int = 340) -> dict:
    return dict(
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )


def _breadth_chart(hist: pd.DataFrame, title: str):
    if go is None or hist.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["price_date"], y=hist["breadth_50"],
                             name="% above 50MA", line=dict(color="#3b82f6", width=2)))
    fig.add_trace(go.Scatter(x=hist["price_date"], y=hist["breadth_200"],
                             name="% above 200MA", line=dict(color="#22c55e", width=2)))
    fig.add_hline(y=50, line_color="rgba(255,255,255,0.25)", line_dash="dash",
                  annotation_text="50%", annotation_font_color="rgba(255,255,255,0.5)")
    fig.update_layout(**_plot_base(340),
                      title=dict(text=title, font=dict(size=13), x=0),
                      yaxis=dict(range=[0, 100], title="Breadth %",
                                 gridcolor="rgba(255,255,255,0.08)"))
    return fig


def _high_low_chart(hist: pd.DataFrame):
    if go is None or hist.empty:
        return None
    recent = hist.tail(90)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=recent["price_date"], y=recent["new_highs"],
                         name="New highs", marker_color="#22c55e"))
    fig.add_trace(go.Bar(x=recent["price_date"], y=-recent["new_lows"],
                         name="New lows", marker_color="#ef4444"))
    fig.update_layout(**_plot_base(320),
                      title=dict(text="52-week new highs vs new lows (last 90 sessions)", font=dict(size=13), x=0),
                      barmode="relative",
                      yaxis=dict(title="Stocks", gridcolor="rgba(255,255,255,0.08)"))
    return fig


def _theme_bar_chart(theme_df: pd.DataFrame):
    if go is None or theme_df.empty:
        return None
    df = theme_df.dropna(subset=["1M %"]).sort_values("1M %")
    if df.empty:
        return None
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df["1M %"]]
    fig = go.Figure(go.Bar(
        x=df["1M %"], y=df["Theme"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in df["1M %"]],
        textposition="outside",
        hovertemplate="%{y}: %{x:+.1f}%<extra></extra>",
    ))
    fig.update_layout(**_plot_base(max(280, len(df) * 38)),
                      title=dict(text="Theme momentum — avg 1-month return", font=dict(size=13), x=0),
                      margin=dict(l=10, r=70, t=48, b=10),
                      xaxis=dict(ticksuffix="%", zeroline=True,
                                 zerolinecolor="rgba(255,255,255,0.3)",
                                 gridcolor="rgba(255,255,255,0.08)"),
                      yaxis=dict(tickfont=dict(size=12)),
                      legend=dict(visible=False))
    return fig


def _trend_chart(histories: dict[str, pd.DataFrame]):
    if go is None:
        return None
    colors = ["#3b82f6", "#db2777", "#f59e0b"]
    fig = go.Figure()
    for (name, hist), color in zip(histories.items(), colors):
        if hist.empty:
            continue
        fig.add_trace(go.Scatter(x=hist["price_date"], y=hist["advance_index"],
                                 name=name, line=dict(color=color, width=2)))
    fig.update_layout(**_plot_base(340),
                      title=dict(text="Equal-weight trend proxies (base = 100)", font=dict(size=13), x=0),
                      yaxis=dict(title="Index level", gridcolor="rgba(255,255,255,0.08)"))
    return fig


# ── Stock table formatter ─────────────────────────────────────────────────────

def _fmt_stock_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["ret_5d", "ret_21d", "ret_63d"]:
        if col in out.columns:
            out[col] = out[col].map(_fmt_ret)
    if "price" in out.columns:
        out["price"] = out["price"].map(lambda x: f"${float(x):,.2f}" if x else "-")
    for src, dst in [("above_50ma", ">50MA"), ("above_200ma", ">200MA"),
                     ("new_high", "New High"), ("new_low", "New Low")]:
        if src in out.columns:
            out[dst] = out[src].map(lambda x: "✓" if x else "")
            out.drop(columns=[src], inplace=True)
    out = out.rename(columns={"ticker": "Ticker", "company_name": "Company", "theme": "Theme",
                               "ret_5d": "1W", "ret_21d": "1M", "ret_63d": "3M"})
    keep = ["Ticker", "Company", "Theme", "price", "1W", "1M", "3M", ">50MA", ">200MA", "New High", "New Low"]
    return out[[c for c in keep if c in out.columns]]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("Market Trend Analysis")
    st.caption(
        "Data source: local SQLite `prices_eod`. "
        "Index groups are database-derived proxies, not official constituent lists. "
        "Theme rotation uses a hard-coded ticker list — only tickers present in the local DB are included."
    )

    if not DB_PATH.exists():
        st.error(f"Database not found: `{DB_PATH}`")
        return

    with st.sidebar:
        st.header("Settings")
        st.caption(f"DB: `{DB_PATH.name}`")
        st.divider()
        st.subheader("Daily update")
        update_days = st.number_input("Backfill days", min_value=1, max_value=30, value=7, step=1)
        if st.button("Update prices_eod", type="primary", use_container_width=True):
            with st.spinner("Updating prices_eod from FMP..."):
                res = _run_eod_update(int(update_days))
            log = "\n".join(x for x in [res.stdout, res.stderr] if x.strip())
            if res.returncode == 0:
                st.success("Update finished.")
                st.cache_data.clear()
            else:
                st.error(f"Update failed (exit {res.returncode}).")
                st.info("If FMP blocks bulk mode, run `update_prices_eod.py --mode per-symbol` from terminal.")
            with st.expander("Update log", expanded=res.returncode != 0):
                st.code(log or "(no output)", language="text")
        st.divider()
        if st.button("Refresh data cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Loading prices and computing breadth..."):
        prices, _universe = _load_market_frame()

    if prices.empty:
        st.warning("No usable rows found in `prices_eod`.")
        return

    latest = _latest_rows(prices)
    if latest.empty:
        st.warning("Not enough price history to compute moving-average breadth.")
        return

    latest_date = latest["price_date"].max().date().isoformat()
    segments    = _make_segments(latest)
    stats       = _segment_stats(latest, segments)
    regime      = _classify_market(latest, stats)
    returns     = _compute_returns(prices)
    theme_df    = _theme_rotation(latest, returns)
    leaders, laggards = _leading_lagging(latest, returns)

    # ── SECTION 1: MARKET REGIME ──────────────────────────────────────────────
    score_clr = "#22c55e" if regime.score >= 3 else ("#f59e0b" if regime.score >= 0 else "#ef4444")
    col_h, col_badge = st.columns([4, 1])
    with col_h:
        st.subheader(f"Market regime: {regime.label}")
    with col_badge:
        st.markdown(
            f"<div style='padding:8px 12px;border-radius:8px;"
            f"background:{score_clr}22;border:1px solid {score_clr};"
            f"text-align:center;margin-top:6px'>"
            f"<span style='color:{score_clr};font-weight:700;font-size:1.1rem'>Score {regime.score:+d}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    getattr(st, regime.tone)(f"**{regime.label}** — {regime.score:+d} points")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Data date",       latest_date)
    c2.metric("Stocks analyzed", f"{len(latest):,}")
    c3.metric("Above 20MA",      _pct(latest["above_20ma"].mean() * 100) if "above_20ma" in latest.columns else "—")
    c4.metric("Above 50MA",      _pct(latest["above_50ma"].mean() * 100))
    c5.metric("Above 200MA",     _pct(latest["above_200ma"].mean() * 100))
    c6.metric("New hi / lo",     f"{int(latest['new_high'].sum())} / {int(latest['new_low'].sum())}")

    with st.expander("Why this regime?", expanded=True):
        for item in regime.details:
            st.markdown(f"- {item}")

    takeaway = _investor_takeaway(regime, theme_df)
    st.info(f"**Investor takeaway:** {takeaway}")

    st.divider()

    # ── SECTION 2: BREADTH DASHBOARD ─────────────────────────────────────────
    st.subheader("Breadth Dashboard")
    histories = {name: _history_by_date(prices, tickers) for name, tickers in segments.items()}

    tab_b, tab_hl, tab_seg, tab_trend = st.tabs(
        ["Breadth Trend", "New Highs / Lows", "Segment Table", "Trend Proxies"]
    )

    with tab_b:
        hist0 = histories.get("Large-cap proxy", pd.DataFrame())
        fig = _breadth_chart(hist0, "Breadth trend: large-cap proxy (% above 50MA / 200MA)")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        elif not hist0.empty:
            st.line_chart(hist0.set_index("price_date")[["breadth_50", "breadth_200"]])
        else:
            st.info("Not enough data for breadth trend chart.")

    with tab_hl:
        hist0 = histories.get("Large-cap proxy", pd.DataFrame())
        fig = _high_low_chart(hist0)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        elif not hist0.empty:
            st.bar_chart(hist0.tail(90).set_index("price_date")[["new_highs", "new_lows"]])
        else:
            st.info("Not enough data for highs/lows chart.")

    with tab_seg:
        if not stats.empty:
            pct_cols = [">20MA %", ">50MA %", ">150MA %", ">200MA %"]
            disp = stats.copy()
            for col in pct_cols:
                if col in disp.columns:
                    disp[col] = disp[col].map(lambda x: _pct(x) if x is not None else "—")
            st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data to build segment table.")

    with tab_trend:
        fig = _trend_chart(histories)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for trend proxies.")

    st.divider()

    # ── SECTION 3: THEME ROTATION ─────────────────────────────────────────────
    st.subheader("Theme Rotation Radar")

    ranked_themes = theme_df.dropna(subset=["1M %"])
    if not ranked_themes.empty:
        col_lead, col_lag = st.columns(2)
        with col_lead:
            top = ranked_themes.iloc[0]
            n = f" ({int(top['In DB'])} in DB)" if top["In DB"] > 0 else ""
            st.success(f"**Leading:** {top['Theme']} — 1M avg {top['1M %']:+.1f}%{n}")
        with col_lag:
            bot = ranked_themes.iloc[-1]
            n = f" ({int(bot['In DB'])} in DB)" if bot["In DB"] > 0 else ""
            st.error(f"**Lagging:** {bot['Theme']} — 1M avg {bot['1M %']:+.1f}%{n}")

    col_c, col_t = st.columns([2, 3])
    with col_c:
        fig = _theme_bar_chart(theme_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough return data for theme chart.")
    with col_t:
        disp = theme_df.copy()
        for col in ["1W %", "1M %", "3M %"]:
            disp[col] = disp[col].map(_fmt_ret)
        for col in [">50MA", ">200MA"]:
            disp[col] = disp[col].map(lambda x: _pct(x, 0) if x is not None and np.isfinite(float(x)) else "—")
        st.dataframe(
            disp[["Rank", "Theme", "In DB", "Tickers", "1W %", "1M %", "3M %", ">50MA", ">200MA"]],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ── SECTION 4: LEADING & LAGGING STOCKS ──────────────────────────────────
    st.subheader("Leading & Lagging Stocks")
    st.caption("Ranked by 1-month return among tickers with sufficient price history in the local DB.")

    col_lead, col_lag = st.columns(2)
    with col_lead:
        st.markdown("##### Strongest Leaders")
        if not leaders.empty:
            st.dataframe(_fmt_stock_table(leaders), use_container_width=True, hide_index=True)
        else:
            st.info("Not enough return data to identify leaders.")
    with col_lag:
        st.markdown("##### Weakest / Laggards")
        if not laggards.empty:
            st.dataframe(_fmt_stock_table(laggards), use_container_width=True, hide_index=True)
        else:
            st.info("Not enough return data to identify laggards.")

    st.divider()

    # ── SECTION 5: WATCHLIST IMPLICATIONS ────────────────────────────────────
    st.subheader("Watchlist Implications")

    advice = {
        "Bull Trend Continuation":  "Market internals healthy. Favor momentum plays in leading themes; consider adding on pullbacks to key moving averages.",
        "Range-Bound / Choppy":     "Be selective. Overweight quality leaders, avoid chasing. Wait for breakouts on volume confirmation.",
        "Weakening Trend":          "Reduce risk. Focus on names holding 50MA. Avoid new longs in lagging themes. Tighten stops on extended winners.",
        "High Risk / Bear":         "Defensive stance. Protect capital. Monitor for market bottom signals before adding exposure.",
        "Overheated / Overbought":  "Breadth extended. Consider trimming winners that have moved far from moving averages.",
        "Insufficient Data":        "Load more price history to generate reliable signals.",
    }
    st.markdown(f"**Market environment:** {advice.get(regime.label, '')}")

    top3 = ranked_themes.head(3) if not ranked_themes.empty else pd.DataFrame()
    bot2 = ranked_themes.tail(2).sort_values("1M %") if not ranked_themes.empty else pd.DataFrame()

    if not top3.empty:
        st.markdown(f"**What is working:** {', '.join(top3['Theme'].tolist())} "
                    f"— prioritize research and watchlist additions in these themes.")
    if not bot2.empty:
        st.markdown(f"**Losing strength:** {', '.join(bot2['Theme'].tolist())} "
                    f"— reduce exposure or tighten stops in these areas.")

    monitor: list[str] = []
    if not top3.empty:
        monitor.append(f"Breakout and momentum setups in **{top3.iloc[0]['Theme']}** (leading theme)")
    if regime.score >= 3:
        monitor.append("New 52-week highs on expanding volume as trend confirmation")
        monitor.append("Rotation within leaders — which names in the leading theme are accelerating?")
    if regime.score <= 0:
        monitor.append("Breadth deterioration — watch for >50MA % falling below 40%")
        monitor.append("Potential sector rotation from growth to defensive / value")
        monitor.append("Support tests at 200MA as key risk management trigger")
    if monitor:
        st.markdown("**What to monitor next:**")
        for item in monitor:
            st.markdown(f"- {item}")

    st.divider()

    # ── SECTION 6: STOCK DETAIL ───────────────────────────────────────────────
    with st.expander("All stocks — detailed view", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            sector = st.selectbox("Sector", ["All", *sorted(latest["sector"].dropna().unique().tolist())])
        with f2:
            condition = st.selectbox(
                "Condition",
                ["All", "Above 50MA", "Below 50MA", "Above 200MA", "Below 200MA", "New high", "New low"],
            )
        with f3:
            sort_by = st.selectbox("Sort by", ["market_cap", "ticker", "sector", "price"])

        detail = latest.copy()
        if sector != "All":
            detail = detail[detail["sector"] == sector]
        cond_map = {
            "Above 50MA":  detail["above_50ma"],
            "Below 50MA":  ~detail["above_50ma"],
            "Above 200MA": detail["above_200ma"],
            "Below 200MA": ~detail["above_200ma"],
            "New high":    detail["new_high"],
            "New low":     detail["new_low"],
        }
        if condition in cond_map:
            detail = detail[cond_map[condition]]

        show_cols = ["ticker", "company_name", "sector", "price", "ma20", "ma50", "ma200",
                     "volume", "market_cap", "above_50ma", "above_200ma", "new_high", "new_low"]
        show = detail.sort_values(sort_by, ascending=(sort_by != "market_cap"))[
            [c for c in show_cols if c in detail.columns]
        ].head(500)
        st.dataframe(show, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
