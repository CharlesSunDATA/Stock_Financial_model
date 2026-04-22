"""
Technical Strategy Backtester — Streamlit page module.
All logic is self-contained in this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    df: pd.DataFrame
    equity_strategy: pd.Series
    equity_buyhold: pd.Series
    trades: pd.DataFrame  # entry_date, exit_date, entry_px, exit_px, pnl, pnl_pct, exit_reason


def _pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:.2f}%"


def _money(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"${x:,.0f}"


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can return MultiIndex columns like ('Open', 'SPY').
        df.columns = [c[0] for c in df.columns]

    # Normalize columns to standard OHLCV names (defensive: ensure string keys)
    df = df.rename(columns={str(c): str(c).title() for c in df.columns})

    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()

    df = df[needed].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna()
    return df


def add_indicators(
    df: pd.DataFrame,
    rsi_len: int,
    atr_len: int,
    sma_lens: tuple[int, ...],
) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out["Close"].pct_change().fillna(0.0)
    for L in sorted(set(int(x) for x in sma_lens if int(x) > 0)):
        out[f"SMA_{L}"] = ta.sma(out["Close"], length=L)
    out[f"RSI_{rsi_len}"] = ta.rsi(out["Close"], length=rsi_len)
    out[f"ATR_{atr_len}"] = ta.atr(out["High"], out["Low"], out["Close"], length=atr_len)
    return out


@st.cache_data(show_spinner=False, ttl=60 * 30)
def prepare_indicators(
    raw: pd.DataFrame,
    rsi_len: int,
    atr_len: int,
    sma_lens: tuple[int, ...],
) -> pd.DataFrame:
    df = add_indicators(raw, rsi_len=rsi_len, atr_len=atr_len, sma_lens=sma_lens)
    return df.dropna().copy()


def backtest_simple_vectorized(
    df: pd.DataFrame,
    sma_len: int,
    rsi_len: int,
    rsi_buy: float,
    rsi_sell: float,
    initial_capital: float,
) -> BacktestResult:
    d = df.copy()
    sma_col = f"SMA_{sma_len}"
    rsi_col = f"RSI_{rsi_len}"

    buy = (d["Close"] > d[sma_col]) & (d[rsi_col] < rsi_buy)
    sell = d[rsi_col] > rsi_sell

    sig = pd.Series(np.nan, index=d.index, dtype=float)
    sig.loc[buy] = 1.0
    sig.loc[sell] = 0.0
    pos = sig.ffill().fillna(0.0).astype(int)

    strat_ret = pos.shift(1).fillna(0).astype(float) * d["ret"]
    equity = float(initial_capital) * (1.0 + strat_ret).cumprod()
    buyhold = float(initial_capital) * (d["Close"] / float(d["Close"].iloc[0]))

    d["position"] = pos
    d["strategy_ret"] = strat_ret
    d["equity_strategy"] = equity
    d["equity_buyhold"] = buyhold

    chg = d["position"].diff().fillna(0)
    entries = d.index[chg == 1]
    exits = d.index[chg == -1]
    if len(exits) and len(entries) and exits[0] < entries[0]:
        exits = exits[1:]
    n = min(len(entries), len(exits))
    entries = entries[:n]
    exits = exits[:n]

    trades = []
    for en, ex in zip(entries, exits):
        ep = float(d.loc[en, "Close"])
        xp = float(d.loc[ex, "Close"])
        trades.append(
            {
                "entry_date": en,
                "exit_date": ex,
                "entry_px": ep,
                "exit_px": xp,
                "pnl": (xp / ep - 1.0) * float(initial_capital),
                "pnl_pct": xp / ep - 1.0,
                "exit_reason": "RSI Sell",
            }
        )

    return BacktestResult(
        df=d,
        equity_strategy=equity,
        equity_buyhold=buyhold,
        trades=pd.DataFrame(trades),
    )


def backtest_advanced_atr_stop_sma20(
    df: pd.DataFrame,
    rsi_len: int,
    rsi_buy: float,
    sma_200_len: int,
    sma_20_len: int,
    atr_len: int,
    atr_mult: float,
    initial_capital: float,
) -> BacktestResult:
    d = df.copy()
    rsi_col = f"RSI_{rsi_len}"
    sma200_col = f"SMA_{sma_200_len}"
    sma20_col = f"SMA_{sma_20_len}"
    atr_col = f"ATR_{atr_len}"

    buy_sig = (d["Close"] > d[sma200_col]) & (d[rsi_col] < rsi_buy)

    pos = np.zeros(len(d), dtype=int)
    entry_px = np.full(len(d), np.nan, dtype=float)
    stop_px = np.full(len(d), np.nan, dtype=float)
    exit_reason = np.array([None] * len(d), dtype=object)
    buy_mark = np.zeros(len(d), dtype=bool)
    sell_mark = np.zeros(len(d), dtype=bool)

    in_pos = False
    entry_price = math.nan
    stop_price = math.nan
    pending_exit_reason: str | None = None

    for i in range(len(d)):
        # 1) Execute pending exit at today's open (we go flat today)
        if pending_exit_reason is not None:
            in_pos = False
            sell_mark[i] = True
            exit_reason[i] = pending_exit_reason
            pending_exit_reason = None
            entry_price = math.nan
            stop_price = math.nan

        # 2) Entry at today's close (position becomes 1 from today close onwards)
        if (not in_pos) and bool(buy_sig.iloc[i]):
            in_pos = True
            buy_mark[i] = True
            entry_price = float(d["Close"].iloc[i])
            atr = float(d[atr_col].iloc[i]) if pd.notna(d[atr_col].iloc[i]) else math.nan
            stop_price = entry_price - float(atr_mult) * atr if np.isfinite(atr) else math.nan

        # 3) Evaluate exit triggers at today's close; exit happens tomorrow open
        if in_pos:
            low = float(d["Low"].iloc[i])
            close = float(d["Close"].iloc[i])
            sma20 = float(d[sma20_col].iloc[i]) if pd.notna(d[sma20_col].iloc[i]) else math.nan

            hit_stop = np.isfinite(stop_price) and (low < stop_price)
            hit_sma20 = np.isfinite(sma20) and (close < sma20)
            if hit_stop or hit_sma20:
                pending_exit_reason = "Stop Loss" if hit_stop else "SMA20 Trailing Exit"

        # 4) Record end-of-day position and diagnostics (position is "held after close")
        pos[i] = 1 if in_pos else 0
        entry_px[i] = entry_price
        stop_px[i] = stop_price

    d["position"] = pos
    d["entry_px"] = entry_px
    d["stop_px"] = stop_px
    d["buy_mark"] = buy_mark
    d["sell_mark"] = sell_mark
    d["exit_reason"] = exit_reason

    strat_ret = d["position"].shift(1).fillna(0).astype(float) * d["ret"]
    equity = float(initial_capital) * (1.0 + strat_ret).cumprod()
    buyhold = float(initial_capital) * (d["Close"] / float(d["Close"].iloc[0]))

    d["strategy_ret"] = strat_ret
    d["equity_strategy"] = equity
    d["equity_buyhold"] = buyhold

    entry_idx = list(d.index[d["buy_mark"]])
    exit_idx = list(d.index[d["sell_mark"]])
    trades = []
    j = 0
    for en in entry_idx:
        while j < len(exit_idx) and exit_idx[j] <= en:
            j += 1
        if j >= len(exit_idx):
            break
        ex = exit_idx[j]
        ep = float(d.loc[en, "Close"])
        xp = float(d.loc[ex, "Close"])
        trades.append(
            {
                "entry_date": en,
                "exit_date": ex,
                "entry_px": ep,
                "exit_px": xp,
                "pnl": (xp / ep - 1.0) * float(initial_capital),
                "pnl_pct": xp / ep - 1.0,
                "exit_reason": str(d.loc[ex, "exit_reason"]) if d.loc[ex, "exit_reason"] else "Exit",
            }
        )
        j += 1

    return BacktestResult(
        df=d,
        equity_strategy=equity,
        equity_buyhold=buyhold,
        trades=pd.DataFrame(trades),
    )


def perf_metrics(res: BacktestResult, initial_capital: float) -> dict[str, float]:
    eq = res.equity_strategy
    bh = res.equity_buyhold
    if eq.empty:
        return {}
    total_ret = float(eq.iloc[-1] / float(initial_capital) - 1.0)
    n_days = max(1, int(eq.shape[0]))
    ann_ret = float((1.0 + total_ret) ** (TRADING_DAYS / n_days) - 1.0)
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min()) if not dd.empty else 0.0
    bh_ret = float(bh.iloc[-1] / float(initial_capital) - 1.0)

    trades = res.trades
    win_rate = np.nan
    if trades is not None and not trades.empty and "pnl_pct" in trades.columns:
        win_rate = float((trades["pnl_pct"] > 0).mean())

    # Risk metrics
    r = res.df.get("strategy_ret", pd.Series(dtype=float)).astype(float)
    if len(r) > 1 and float(r.std()) > 1e-12:
        sharpe = float(np.sqrt(TRADING_DAYS) * (r.mean() / r.std()))
    else:
        sharpe = np.nan
    calmar = float(ann_ret / abs(max_dd)) if max_dd < -1e-12 else np.nan
    excess_total = float(total_ret - bh_ret)

    return {
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "buy_hold_return": bh_ret,
        "sharpe": sharpe,
        "calmar": calmar,
        "excess_total": excess_total,
    }


def _year_folds(index: pd.DatetimeIndex, train_years: int, test_years: int, rolling: bool) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Returns list of (train_end, test_end) cut points.
    Train is (start .. train_end], test is (train_end .. test_end]
    """
    if len(index) == 0:
        return []
    start = pd.Timestamp(index.min()).normalize()
    end = pd.Timestamp(index.max()).normalize()
    train_years = int(train_years)
    test_years = int(test_years)
    if train_years <= 0 or test_years <= 0:
        return []

    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    train_end = start + pd.DateOffset(years=train_years)
    while True:
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > end:
            break
        folds.append((train_end, test_end))
        if not rolling:
            break
        train_end = train_end + pd.DateOffset(years=1)
    return folds


def _slice_by_dates(df: pd.DataFrame, train_end: pd.Timestamp, test_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = df.index
    train = df.loc[idx <= train_end].copy()
    test = df.loc[(idx > train_end) & (idx <= test_end)].copy()
    return train, test


def _objective_from_metrics(
    m: dict[str, float],
    objective: str,
    dd_cap: float | None,
    min_trades: int | None,
    n_trades: int,
) -> float:
    if objective == "Excess Return vs Buy&Hold":
        return float(m.get("excess_total", np.nan))
    if objective == "Total Return":
        return float(m.get("total_return", np.nan))
    if objective == "Annualized Return":
        return float(m.get("annualized_return", np.nan))
    if objective == "Sharpe":
        return float(m.get("sharpe", np.nan))
    if objective == "Calmar (AnnRet/MaxDD)":
        return float(m.get("calmar", np.nan))
    # Custom constrained excess return
    if dd_cap is not None and float(m.get("max_drawdown", 0.0)) < -abs(float(dd_cap)):
        return -np.inf
    if min_trades is not None and int(n_trades) < int(min_trades):
        return -np.inf
    return float(m.get("excess_total", np.nan))


def plot_equity(res: BacktestResult) -> go.Figure:
    d = res.df
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["equity_buyhold"], mode="lines", name="Buy & Hold"))
    fig.add_trace(go.Scatter(x=d.index, y=d["equity_strategy"], mode="lines", name="Strategy"))
    fig.update_layout(
        title="Equity Curve (Strategy vs Buy & Hold)",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        height=420,
    )
    return fig


def plot_price_signals(res: BacktestResult, mode: str) -> go.Figure:
    d = res.df
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], mode="lines", name="Close", line=dict(width=2)))

    if mode == "Advanced":
        buy_idx = d.index[d["buy_mark"].fillna(False)]
        fig.add_trace(
            go.Scatter(
                x=buy_idx,
                y=d.loc[buy_idx, "Close"],
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=12, color="#00C853"),
                hovertemplate="Buy<br>%{x|%Y-%m-%d}<br>Close=%{y:.2f}<extra></extra>",
            )
        )

        stop_mask = d["exit_reason"].astype(str).eq("Stop Loss")
        sma_mask = d["exit_reason"].astype(str).eq("SMA20 Trailing Exit")
        stop_idx = d.index[d["sell_mark"].fillna(False) & stop_mask]
        sma_idx = d.index[d["sell_mark"].fillna(False) & sma_mask]
        other_idx = d.index[d["sell_mark"].fillna(False) & ~(stop_mask | sma_mask)]

        if len(stop_idx):
            fig.add_trace(
                go.Scatter(
                    x=stop_idx,
                    y=d.loc[stop_idx, "Close"],
                    mode="markers",
                    name="Sell (Stop Loss)",
                    marker=dict(symbol="triangle-down", size=12, color="#FF5252"),
                    hovertemplate="Stop Loss Exit<br>%{x|%Y-%m-%d}<br>Close=%{y:.2f}<extra></extra>",
                )
            )
        if len(sma_idx):
            fig.add_trace(
                go.Scatter(
                    x=sma_idx,
                    y=d.loc[sma_idx, "Close"],
                    mode="markers",
                    name="Sell (SMA20 Exit)",
                    marker=dict(symbol="triangle-down", size=12, color="#FFB300"),
                    hovertemplate="SMA20 Exit<br>%{x|%Y-%m-%d}<br>Close=%{y:.2f}<extra></extra>",
                )
            )
        if len(other_idx):
            fig.add_trace(
                go.Scatter(
                    x=other_idx,
                    y=d.loc[other_idx, "Close"],
                    mode="markers",
                    name="Sell",
                    marker=dict(symbol="triangle-down", size=12, color="#FF5252"),
                )
            )
    else:
        chg = d["position"].diff().fillna(0)
        buy_idx = d.index[chg == 1]
        sell_idx = d.index[chg == -1]
        fig.add_trace(
            go.Scatter(
                x=buy_idx,
                y=d.loc[buy_idx, "Close"],
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=12, color="#00C853"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sell_idx,
                y=d.loc[sell_idx, "Close"],
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", size=12, color="#FF5252"),
            )
        )

    fig.update_layout(
        title="Price with Buy/Sell Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        height=460,
    )
    return fig


def main() -> None:
    st.title("Technical Strategy Backtester")
    st.caption("Data: yfinance • Indicators: pandas_ta • Charts: plotly • For research/education only.")

    # ---- Sidebar: Backtest + Optimizer controls (always visible) ----
    with st.sidebar:
        st.header("Backtest")
        ticker = st.text_input("Ticker", value=st.session_state.get("bt_ticker", "SPY")).strip().upper() or "SPY"
        st.session_state["bt_ticker"] = ticker

        end_d = date.today()
        start_d = st.date_input("Start date", value=end_d - timedelta(days=365 * 10), key="bt_start")
        end_d_in = st.date_input("End date", value=end_d, key="bt_end")
        if start_d >= end_d_in:
            st.error("Start date must be before end date.")
            st.stop()

        initial_capital = st.number_input(
            "Initial capital ($)",
            min_value=100.0,
            max_value=10_000_000.0,
            value=float(st.session_state.get("bt_capital", 10_000.0)),
            step=500.0,
            format="%.0f",
            key="bt_capital",
        )

        mode = st.selectbox("Strategy mode", ["Advanced", "Simple (vectorized)"], index=0, key="bt_mode")

        st.subheader("Indicator settings")
        rsi_len = st.slider("RSI length", 5, 50, 14, 1, key="bt_rsi_len")
        sma_len = st.slider("SMA length (Simple mode)", 20, 300, 200, 5, key="bt_sma_len")

        st.subheader("Simple thresholds")
        rsi_buy = st.slider("RSI buy threshold (RSI < x)", 1, 60, 30, 1, key="bt_rsi_buy")
        rsi_sell = st.slider("RSI sell threshold (RSI > x)", 40, 99, 70, 1, key="bt_rsi_sell")

        st.subheader("Advanced settings")
        sma_200_len = st.slider("Trend SMA length", 100, 300, 200, 10, key="bt_sma200")
        sma_exit_len = st.slider("Trailing exit SMA length", 10, 200, 20, 5, key="bt_smaexit")
        atr_len = st.slider("ATR length", 5, 50, 14, 1, key="bt_atr_len")
        atr_mult = st.slider("Stop loss ATR multiple", 0.5, 5.0, 2.0, 0.25, key="bt_atr_mult")

        run_backtest = st.button("Run backtest", type="primary", width="stretch", key="bt_run")
        if run_backtest:
            st.session_state["bt_run_request"] = True

        st.divider()
        with st.expander("Optimizer (optional)", expanded=False):
            st.caption("Run optimizer first, then optionally apply best params and auto-run a backtest.")
            preset = st.selectbox(
                "Preset",
                ["Control drawdown & beat Buy&Hold", "Maximize return"],
                index=0,
                key="opt_preset",
            )
            opt_mode = st.selectbox("Optimize which strategy?", ["Advanced", "Simple (vectorized)"], index=0, key="opt_mode")
            objective = st.selectbox(
                "Objective",
                [
                    "Excess Return vs Buy&Hold",
                    "Total Return",
                    "Annualized Return",
                    "Sharpe",
                    "Calmar (AnnRet/MaxDD)",
                    "Custom (constraints + Excess Return)",
                ],
                index=0,
                key="opt_obj",
            )
            rolling = st.checkbox("Rolling walk-forward (step 1y)", value=True, key="opt_roll")
            c1, c2 = st.columns(2)
            with c1:
                train_years = st.number_input("Train years", min_value=2, max_value=20, value=7, step=1, key="opt_train")
            with c2:
                test_years = st.number_input("Test years", min_value=1, max_value=10, value=3, step=1, key="opt_test")

            dd_cap = None
            min_trades = None
            if objective.startswith("Custom"):
                c3, c4 = st.columns(2)
                with c3:
                    dd_cap = st.slider("Max drawdown cap (absolute %)", 5, 60, 25, 1, key="opt_dd") / 100.0
                with c4:
                    min_trades = st.number_input("Min trades (per test fold)", min_value=0, max_value=200, value=4, step=1, key="opt_tr")

            if st.button("Apply preset defaults", key="opt_apply_preset"):
                if preset == "Control drawdown & beat Buy&Hold":
                    st.session_state["opt_obj"] = "Custom (constraints + Excess Return)"
                    st.session_state["opt_dd"] = 25
                    st.session_state["opt_tr"] = 4
                    st.session_state["opt_roll"] = True
                    st.session_state["opt_train"] = 7
                    st.session_state["opt_test"] = 3
                    if opt_mode == "Advanced":
                        st.session_state["g_rsi_buy"] = (30, 45)
                        st.session_state["g_atr"] = (2.5, 4.5)
                        st.session_state["g_exit"] = (30, 120)
                        st.session_state["g_trend"] = (150, 250)
                    else:
                        st.session_state["g_s_rsi_buy"] = (30, 45)
                        st.session_state["g_s_rsi_sell"] = (60, 80)
                        st.session_state["g_s_sma"] = (100, 250)
                else:
                    st.session_state["opt_obj"] = "Total Return"
                    st.session_state["opt_roll"] = True
                    st.session_state["opt_train"] = 7
                    st.session_state["opt_test"] = 3
                    if opt_mode == "Advanced":
                        st.session_state["g_rsi_buy"] = (20, 55)
                        st.session_state["g_atr"] = (2.0, 5.0)
                        st.session_state["g_exit"] = (20, 120)
                        st.session_state["g_trend"] = (120, 280)
                    else:
                        st.session_state["g_s_rsi_buy"] = (20, 55)
                        st.session_state["g_s_rsi_sell"] = (55, 90)
                        st.session_state["g_s_sma"] = (50, 280)
                st.success("Preset applied.")

            st.markdown("#### Grid")
            if opt_mode == "Advanced":
                rsi_buy_min, rsi_buy_max = st.slider("RSI buy threshold range", 5, 60, (25, 45), 1, key="g_rsi_buy")
                atr_mult_min, atr_mult_max = st.slider("ATR multiple range", 1.0, 6.0, (2.0, 4.0), 0.25, key="g_atr")
                sma_exit_min, sma_exit_max = st.slider("Exit SMA length range", 10, 200, (20, 80), 5, key="g_exit")
                sma_trend_min, sma_trend_max = st.slider("Trend SMA length range", 100, 300, (150, 250), 10, key="g_trend")
            else:
                rsi_buy_min, rsi_buy_max = st.slider("RSI buy threshold range", 5, 60, (25, 45), 1, key="g_s_rsi_buy")
                rsi_sell_min, rsi_sell_max = st.slider("RSI sell threshold range", 40, 99, (60, 80), 1, key="g_s_rsi_sell")
                sma_min, sma_max = st.slider("SMA length range", 20, 300, (100, 250), 5, key="g_s_sma")

            max_combos = st.number_input("Max combinations", min_value=20, max_value=2000, value=300, step=20, key="opt_maxc")
            run_opt = st.button("Run optimizer", type="primary", key="opt_run")
            if run_opt:
                st.session_state["opt_run_request"] = True

    # ---- Run backtest and/or optimizer depending on requests ----
    run_bt = bool(st.session_state.pop("bt_run_request", False))
    run_opt_req = bool(st.session_state.pop("opt_run_request", False))

    if not (run_bt or run_opt_req) and "last_backtest" not in st.session_state and "opt_results" not in st.session_state:
        st.info("Use the left sidebar to run a backtest and/or run the optimizer.")

    raw: pd.DataFrame | None = None
    if run_bt or run_opt_req:
        with st.spinner("Downloading data…"):
            raw = load_ohlcv(ticker, start_d, end_d_in)
        if raw.empty or len(raw) < 250:
            st.error("Not enough data returned. Try a longer period or a different ticker.")
            return

    # Backtest execution
    if run_bt:
        sma_lens_needed = (int(sma_len), int(sma_200_len), int(sma_exit_len))
        with st.spinner("Computing indicators…"):
            df_bt = prepare_indicators(raw, rsi_len=int(rsi_len), atr_len=int(atr_len), sma_lens=sma_lens_needed)
        if df_bt.empty:
            st.error("Indicators produced no usable rows (too few periods).")
            return
        with st.spinner("Running strategy…"):
            if mode == "Advanced":
                res = backtest_advanced_atr_stop_sma20(
                    df=df_bt,
                    rsi_len=rsi_len,
                    rsi_buy=float(rsi_buy),
                    sma_200_len=sma_200_len,
                    sma_20_len=sma_exit_len,
                    atr_len=atr_len,
                    atr_mult=float(atr_mult),
                    initial_capital=float(initial_capital),
                )
            else:
                res = backtest_simple_vectorized(
                    df=df_bt,
                    sma_len=sma_len,
                    rsi_len=rsi_len,
                    rsi_buy=float(rsi_buy),
                    rsi_sell=float(rsi_sell),
                    initial_capital=float(initial_capital),
                )
        st.session_state["last_backtest"] = {"res": res, "capital": float(initial_capital)}

    # Optimizer execution (independent from backtest)
    if run_opt_req:
        try:
            # Build folds over the available date range
            # Recreate a df_all with all SMA lengths needed for the grid
            folds = _year_folds(raw.index, int(st.session_state.get("opt_train", 7)), int(st.session_state.get("opt_test", 3)), bool(st.session_state.get("opt_roll", True)))
            if not folds:
                st.error("Not enough data to build the requested walk-forward folds. Increase history or reduce years.")
            else:
                # Build parameter grid + required SMA lengths
                opt_mode_cur = st.session_state.get("opt_mode", "Advanced")
                objective_cur = st.session_state.get("opt_obj", "Excess Return vs Buy&Hold")
                dd_cap = (st.session_state.get("opt_dd", None) / 100.0) if str(objective_cur).startswith("Custom") else None
                min_trades = int(st.session_state.get("opt_tr", 0)) if str(objective_cur).startswith("Custom") else None
                max_combos = int(st.session_state.get("opt_maxc", 300))

                grid: list[dict[str, float]] = []
                required_sma: set[int] = set()
                if opt_mode_cur == "Advanced":
                    rsi_buy_min, rsi_buy_max = st.session_state.get("g_rsi_buy", (25, 45))
                    atr_mult_min, atr_mult_max = st.session_state.get("g_atr", (2.0, 4.0))
                    sma_exit_min, sma_exit_max = st.session_state.get("g_exit", (20, 80))
                    sma_trend_min, sma_trend_max = st.session_state.get("g_trend", (150, 250))
                    rsi_vals = list(range(int(rsi_buy_min), int(rsi_buy_max) + 1, 1))
                    atr_vals = list(np.arange(float(atr_mult_min), float(atr_mult_max) + 1e-9, 0.25))
                    exit_vals = list(range(int(sma_exit_min), int(sma_exit_max) + 1, 5))
                    trend_vals = list(range(int(sma_trend_min), int(sma_trend_max) + 1, 10))
                    required_sma.update(exit_vals)
                    required_sma.update(trend_vals)
                    for rb in rsi_vals:
                        for am in atr_vals:
                            for exl in exit_vals:
                                for trl in trend_vals:
                                    grid.append({"rsi_buy": float(rb), "atr_mult": float(am), "sma_exit": float(exl), "sma_trend": float(trl)})
                else:
                    rsi_buy_min, rsi_buy_max = st.session_state.get("g_s_rsi_buy", (25, 45))
                    rsi_sell_min, rsi_sell_max = st.session_state.get("g_s_rsi_sell", (60, 80))
                    sma_min, sma_max = st.session_state.get("g_s_sma", (100, 250))
                    rsi_b_vals = list(range(int(rsi_buy_min), int(rsi_buy_max) + 1, 1))
                    rsi_s_vals = list(range(int(rsi_sell_min), int(rsi_sell_max) + 1, 1))
                    sma_vals = list(range(int(sma_min), int(sma_max) + 1, 5))
                    required_sma.update(sma_vals)
                    for rb in rsi_b_vals:
                        for rs in rsi_s_vals:
                            for sl in sma_vals:
                                grid.append({"rsi_buy": float(rb), "rsi_sell": float(rs), "sma": float(sl)})

                if len(grid) > max_combos:
                    rng = np.random.default_rng(42)
                    pick = rng.choice(len(grid), size=max_combos, replace=False)
                    grid = [grid[int(i)] for i in pick]

                base = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                df_all = prepare_indicators(base, int(rsi_len), int(atr_len), tuple(sorted(required_sma)))
                rows: list[dict[str, object]] = []
                with st.spinner(f"Optimizer running… combos={len(grid):,}, folds={len(folds):,}"):
                    prog = st.progress(0.0)
                    for gi, params in enumerate(grid):
                        fold_scores = []
                        fold_tests = []
                        for train_end, test_end in folds:
                            train_i, test_i = _slice_by_dates(df_all, train_end, test_end)
                            if len(train_i) < 200 or len(test_i) < 50:
                                continue
                            if opt_mode_cur == "Advanced":
                                res_tr = backtest_advanced_atr_stop_sma20(
                                    train_i, int(rsi_len), float(params["rsi_buy"]), int(params["sma_trend"]), int(params["sma_exit"]), int(atr_len), float(params["atr_mult"]), float(initial_capital)
                                )
                                m_tr = perf_metrics(res_tr, float(initial_capital))
                                res_te = backtest_advanced_atr_stop_sma20(
                                    test_i, int(rsi_len), float(params["rsi_buy"]), int(params["sma_trend"]), int(params["sma_exit"]), int(atr_len), float(params["atr_mult"]), float(initial_capital)
                                )
                                m_te = perf_metrics(res_te, float(initial_capital))
                                ntr = 0 if res_te.trades is None else len(res_te.trades)
                            else:
                                res_tr = backtest_simple_vectorized(train_i, int(params["sma"]), int(rsi_len), float(params["rsi_buy"]), float(params["rsi_sell"]), float(initial_capital))
                                m_tr = perf_metrics(res_tr, float(initial_capital))
                                res_te = backtest_simple_vectorized(test_i, int(params["sma"]), int(rsi_len), float(params["rsi_buy"]), float(params["rsi_sell"]), float(initial_capital))
                                m_te = perf_metrics(res_te, float(initial_capital))
                                ntr = 0 if res_te.trades is None else len(res_te.trades)

                            score = _objective_from_metrics(
                                m_tr,
                                "Custom (constraints + Excess Return)" if str(objective_cur).startswith("Custom") else str(objective_cur),
                                dd_cap=dd_cap,
                                min_trades=min_trades,
                                n_trades=ntr,
                            )
                            if not np.isfinite(score):
                                continue
                            fold_scores.append(float(score))
                            fold_tests.append(m_te)
                        if fold_scores and fold_tests:
                            avg = {k: float(np.nanmean([ft.get(k, np.nan) for ft in fold_tests])) for k in fold_tests[0].keys()}
                            rows.append({**{f"p_{k}": v for k, v in params.items()}, "train_score": float(np.mean(fold_scores)), **{f"test_{k}": v for k, v in avg.items()}})
                        prog.progress((gi + 1) / max(1, len(grid)))
                    prog.empty()
                if rows:
                    st.session_state["opt_results"] = pd.DataFrame(rows)
                else:
                    st.warning("Optimizer finished, but no valid rows were produced. Try widening ranges.")
        except Exception as e:
            st.error("Optimizer failed.")
            st.exception(e)

    # ---- Render results (if available) ----
    if "last_backtest" in st.session_state:
        res = st.session_state["last_backtest"]["res"]
        cap = float(st.session_state["last_backtest"]["capital"])
        m = perf_metrics(res, cap)
        st.subheader("Backtest results")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Return", _pct(m.get("total_return", np.nan)))
        with c2:
            st.metric("Annualized Return", _pct(m.get("annualized_return", np.nan)))
        with c3:
            st.metric("Max Drawdown", _pct(m.get("max_drawdown", np.nan)))
        with c4:
            st.metric("Win Rate", _pct(m.get("win_rate", np.nan)))
        with c5:
            st.metric("Buy & Hold Return", _pct(m.get("buy_hold_return", np.nan)))
        st.caption(f"Initial capital: **{_money(cap)}** • Rows used: **{len(res.df):,}** • Trades: **{0 if res.trades is None else len(res.trades):,}**")
        st.plotly_chart(plot_equity(res), use_container_width=True)
        st.plotly_chart(plot_price_signals(res, "Advanced" if st.session_state.get("bt_mode") == "Advanced" else "Simple"), use_container_width=True)

        with st.expander("Trades (entry/exit)", expanded=False):
            if res.trades is None or res.trades.empty:
                st.info("No completed trades.")
            else:
                tdf = res.trades.copy()
                tdf["entry_date"] = pd.to_datetime(tdf["entry_date"]).dt.strftime("%Y-%m-%d")
                tdf["exit_date"] = pd.to_datetime(tdf["exit_date"]).dt.strftime("%Y-%m-%d")
                tdf["pnl_pct"] = (tdf["pnl_pct"] * 100.0).round(2)
                tdf = tdf.rename(
                    columns={"entry_date": "Entry", "exit_date": "Exit", "entry_px": "Entry Px", "exit_px": "Exit Px", "pnl_pct": "PnL %", "exit_reason": "Exit reason"}
                )
                st.dataframe(tdf, use_container_width=True, hide_index=True)

    if "opt_results" in st.session_state:
        out = st.session_state["opt_results"].copy()
        objective_cur = str(st.session_state.get("opt_obj", "Excess Return vs Buy&Hold"))
        sort_key = "test_excess_total"
        sort_label = "test excess return"
        if objective_cur == "Total Return":
            sort_key, sort_label = "test_total_return", "test total return"
        elif objective_cur == "Annualized Return":
            sort_key, sort_label = "test_annualized_return", "test annualized return"
        elif objective_cur == "Sharpe":
            sort_key, sort_label = "test_sharpe", "test Sharpe"
        elif objective_cur == "Calmar (AnnRet/MaxDD)":
            sort_key, sort_label = "test_calmar", "test Calmar"
        if sort_key in out.columns:
            out = out.sort_values(sort_key, ascending=False).reset_index(drop=True)
        st.subheader(f"Optimizer results (sorted by {sort_label})")
        show_cols = [c for c in out.columns if c.startswith("p_")] + [
            "train_score",
            "test_total_return",
            "test_annualized_return",
            "test_buy_hold_return",
            "test_excess_total",
            "test_sharpe",
            "test_calmar",
            "test_max_drawdown",
            "test_win_rate",
        ]
        present = [c for c in show_cols if c in out.columns]
        st.dataframe(out[present].head(30), use_container_width=True, hide_index=True)

        if st.button("Apply best parameters → run backtest", key="opt_apply_and_run"):
            best = out.iloc[0].to_dict()
            opt_mode_cur = st.session_state.get("opt_mode", "Advanced")
            st.session_state["bt_mode"] = "Advanced" if opt_mode_cur == "Advanced" else "Simple (vectorized)"
            if opt_mode_cur == "Advanced":
                st.session_state["bt_rsi_buy"] = int(best["p_rsi_buy"])
                st.session_state["bt_atr_mult"] = float(best["p_atr_mult"])
                st.session_state["bt_sma200"] = int(best["p_sma_trend"])
                st.session_state["bt_smaexit"] = int(best["p_sma_exit"])
            else:
                st.session_state["bt_rsi_buy"] = int(best["p_rsi_buy"])
                st.session_state["bt_rsi_sell"] = int(best["p_rsi_sell"])
                st.session_state["bt_sma_len"] = int(best["p_sma"])
            st.session_state["bt_run_request"] = True
            st.rerun()


main()

