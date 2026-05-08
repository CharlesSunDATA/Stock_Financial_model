"""
Technical Analyser — combines:
  1. Markowitz Efficient Frontier & Portfolio Optimization
  2. Technical Strategy Backtester
  3. Pullback Analyzer
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import Bounds, LinearConstraint, minimize

from utils.local_data import latest_quotes, load_adjusted_close
from utils.local_data import load_ohlcv as load_local_ohlcv

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────────────
TRADING_DAYS = 252


# ═════════════════════════════════════════════════════════════════════════════
# MARKOWITZ helpers
# ═════════════════════════════════════════════════════════════════════════════
WEIGHT_MIN = 0.05
WEIGHT_MAX = 0.20
MIN_ASSETS_FOR_WEIGHT_CAP = 5
MAX_ASSETS_FOR_WEIGHT_FLOOR = 20
N_MONTE_CARLO = 10_000
OPTIMIZER_BUILD_ID = "min5-max20-per-asset-bounds-2026-04"


def _weight_bounds(n: int) -> tuple[tuple[float, float], ...]:
    return tuple((WEIGHT_MIN, WEIGHT_MAX) for _ in range(n))


def _feasible_weight_cap(n: int) -> bool:
    return (
        n >= MIN_ASSETS_FOR_WEIGHT_CAP
        and n <= MAX_ASSETS_FOR_WEIGHT_FLOOR
        and n * WEIGHT_MAX >= 1.0 - 1e-12
        and n * WEIGHT_MIN <= 1.0 + 1e-12
    )


def _budget_constraint(n: int) -> LinearConstraint:
    return LinearConstraint(np.ones((1, n)), 1.0, 1.0)


def _feasible_start(n: int) -> np.ndarray:
    w = np.full(n, WEIGHT_MIN, dtype=float)
    remaining = 1.0 - float(n) * WEIGHT_MIN
    if remaining <= 0:
        return np.full(n, 1.0 / n, dtype=float)
    w += remaining / n
    w = np.clip(w, WEIGHT_MIN, WEIGHT_MAX)
    return w / w.sum()


def _parse_tickers(raw: list[str]) -> list[str]:
    out: list[str] = []
    for s in raw:
        t = (s or "").strip().upper()
        if t and t not in out:
            out.append(t)
    return out


def mk_fetch_adj_close(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    if not tickers:
        raise ValueError("At least one ticker is required.")
    prices = load_adjusted_close(tickers, start, end, min_rows=30)
    missing = [t for t in tickers if t.upper() not in prices.columns]
    if missing:
        data = yf.download(
            missing,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                if "Close" in data.columns.get_level_values(0):
                    y_prices = data["Close"].copy()
                else:
                    y_prices = data.xs("Close", axis=1, level=0, drop_level=True)
            else:
                y_prices = data[["Close"]].copy() if "Close" in data.columns else data.copy()
                if len(missing) == 1:
                    y_prices.columns = [missing[0]]
            prices = prices.join(y_prices, how="outer") if not prices.empty else y_prices
    if prices.empty:
        raise ValueError("No price data returned. Check tickers and date range.")
    prices = prices.dropna(how="all").ffill().dropna()
    if prices.empty or len(prices) < 30:
        raise ValueError("Insufficient overlapping history. Try a wider date range.")
    return prices.astype(float)


def mk_ann_stats(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu_ann = returns.mean().values * TRADING_DAYS
    cov_ann = returns.cov().values * TRADING_DAYS
    return mu_ann, cov_ann


def mk_portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(np.dot(w, mu))


def mk_portfolio_volatility(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(w @ cov @ w, 0.0)))


def mk_neg_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    vol = mk_portfolio_volatility(w, cov)
    if vol < 1e-12:
        return 1e12
    return -(mk_portfolio_return(w, mu) - rf) / vol


def _weights_feasible(w: np.ndarray) -> bool:
    s = float(w.sum())
    if abs(s - 1.0) > 5e-4:
        return False
    if float(w.max()) > WEIGHT_MAX + 1e-5 or float(w.min()) < WEIGHT_MIN - 1e-5:
        return False
    return True


def mk_optimize_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    n = len(mu)
    x0 = _feasible_start(n)
    bounds = Bounds(WEIGHT_MIN, WEIGHT_MAX, keep_feasible=True)
    lc = _budget_constraint(n)
    res = minimize(mk_neg_sharpe, x0, args=(mu, cov, rf), method="trust-constr",
                   bounds=bounds, constraints=[lc], options={"maxiter": 3000, "gtol": 1e-8})
    w = np.asarray(res.x, dtype=float)
    if not res.success:
        st.warning(f"Max-Sharpe (trust-constr): {res.message}")
    if not _weights_feasible(w):
        res2 = minimize(mk_neg_sharpe, _feasible_start(n), args=(mu, cov, rf), method="SLSQP",
                        bounds=_weight_bounds(n),
                        constraints={"type": "eq", "fun": lambda ww: float(np.sum(ww)) - 1.0},
                        options={"maxiter": 3000, "ftol": 1e-11})
        w = np.asarray(res2.x, dtype=float)
        if not res2.success:
            st.warning(f"Max-Sharpe (SLSQP fallback): {res2.message}")
    return np.clip(w, WEIGHT_MIN, WEIGHT_MAX)


def mk_optimize_min_volatility(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    x0 = _feasible_start(n)
    bounds = Bounds(WEIGHT_MIN, WEIGHT_MAX, keep_feasible=True)
    lc = _budget_constraint(n)

    def obj(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    res = minimize(obj, x0, method="trust-constr", bounds=bounds, constraints=[lc],
                   options={"maxiter": 3000, "gtol": 1e-8})
    w = np.asarray(res.x, dtype=float)
    if not res.success:
        st.warning(f"Min-vol (trust-constr): {res.message}")
    if not _weights_feasible(w):
        res2 = minimize(obj, _feasible_start(n), method="SLSQP", bounds=_weight_bounds(n),
                        constraints={"type": "eq", "fun": lambda ww: float(np.sum(ww)) - 1.0},
                        options={"maxiter": 3000, "ftol": 1e-11})
        w = np.asarray(res2.x, dtype=float)
        if not res2.success:
            st.warning(f"Min-vol (SLSQP fallback): {res2.message}")
    return np.clip(w, WEIGHT_MIN, WEIGHT_MAX)


def _sample_capped_simplex_weights(n: int, rng: np.random.Generator, n_sims: int) -> np.ndarray:
    if not _feasible_weight_cap(n):
        raise ValueError("Infeasible weight bounds for this n.")
    remaining_mass = 1.0 - float(n) * WEIGHT_MIN
    if abs(remaining_mass) < 1e-12:
        return np.tile(np.full(n, WEIGHT_MIN, dtype=float), (n_sims, 1))
    u_cap = (WEIGHT_MAX - WEIGHT_MIN) / remaining_mass
    out: list[np.ndarray] = []
    alpha = max(12.0, float(n) * 2.0)
    attempts = 0
    max_attempts = max(500_000, n_sims * 500)
    while len(out) < n_sims and attempts < max_attempts:
        batch = min(65536, (n_sims - len(out)) * 4 + 1024)
        W = rng.dirichlet(np.ones(n) * alpha, size=batch)
        attempts += batch
        for u in W:
            if float(u.max()) <= u_cap + 1e-12:
                w = WEIGHT_MIN + remaining_mass * np.asarray(u, dtype=np.float64)
                out.append(w)
                if len(out) >= n_sims:
                    break
        if len(out) < max(1, n_sims // 100) and attempts >= n_sims * 20:
            alpha *= 1.35
    while len(out) < n_sims:
        out.append(_feasible_start(n))
    return np.stack(out[:n_sims], axis=0)


def mk_monte_carlo_portfolios(mu: np.ndarray, cov: np.ndarray, rf: float,
                               n_sims: int, rng: np.random.Generator) -> pd.DataFrame:
    n = len(mu)
    W = _sample_capped_simplex_weights(n, rng, n_sims)
    ann_returns = W @ mu
    ann_vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    sharpe = np.divide(ann_returns - rf, ann_vols,
                       out=np.full(n_sims, np.nan, dtype=float), where=ann_vols > 1e-12)
    return pd.DataFrame({"sim_id": np.arange(n_sims), "ann_return": ann_returns,
                         "ann_volatility": ann_vols, "sharpe_ratio": sharpe})


# ═════════════════════════════════════════════════════════════════════════════
# BACKTESTER helpers
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class BacktestResult:
    df: pd.DataFrame
    equity_strategy: pd.Series
    equity_buyhold: pd.Series
    trades: pd.DataFrame


def _pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:.2f}%"


def _money(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"${x:,.0f}"


@st.cache_data(show_spinner=False, ttl=60 * 30)
def bt_load_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = load_local_ohlcv(ticker, start, end, min_rows=30)
    if df is None or df.empty:
        df = yf.download(ticker, start=start.isoformat(),
                         end=(end + timedelta(days=1)).isoformat(),
                         auto_adjust=False, progress=False, threads=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns={str(c): str(c).title() for c in df.columns})
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()


def bt_add_indicators(df: pd.DataFrame, rsi_len: int, atr_len: int,
                       sma_lens: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out["Close"].pct_change().fillna(0.0)
    for L in sorted(set(int(x) for x in sma_lens if int(x) > 0)):
        out[f"SMA_{L}"] = out["Close"].rolling(window=L, min_periods=L).mean()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / float(rsi_len)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=rsi_len).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=rsi_len).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out[f"RSI_{rsi_len}"] = 100.0 - (100.0 / (1.0 + rs))
    prev_close = out["Close"].shift(1)
    tr = pd.concat([(out["High"] - out["Low"]).abs(),
                    (out["High"] - prev_close).abs(),
                    (out["Low"] - prev_close).abs()], axis=1).max(axis=1)
    out[f"ATR_{atr_len}"] = tr.ewm(alpha=1.0 / float(atr_len), adjust=False,
                                    min_periods=atr_len).mean()
    return out


@st.cache_data(show_spinner=False, ttl=60 * 30)
def bt_prepare_indicators(raw: pd.DataFrame, rsi_len: int, atr_len: int,
                           sma_lens: tuple[int, ...]) -> pd.DataFrame:
    df = bt_add_indicators(raw, rsi_len=rsi_len, atr_len=atr_len, sma_lens=sma_lens)
    return df.dropna().copy()


def bt_backtest_simple(df: pd.DataFrame, sma_len: int, rsi_len: int, rsi_buy: float,
                        rsi_sell: float, initial_capital: float) -> BacktestResult:
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
        trades.append({"entry_date": en, "exit_date": ex, "entry_px": ep, "exit_px": xp,
                       "pnl": (xp / ep - 1.0) * float(initial_capital),
                       "pnl_pct": xp / ep - 1.0, "exit_reason": "RSI Sell"})
    return BacktestResult(df=d, equity_strategy=equity, equity_buyhold=buyhold,
                          trades=pd.DataFrame(trades))


def bt_backtest_advanced(df: pd.DataFrame, rsi_len: int, rsi_buy: float, sma_200_len: int,
                          sma_20_len: int, atr_len: int, atr_mult: float,
                          initial_capital: float) -> BacktestResult:
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
        if pending_exit_reason is not None:
            in_pos = False
            sell_mark[i] = True
            exit_reason[i] = pending_exit_reason
            pending_exit_reason = None
            entry_price = math.nan
            stop_price = math.nan
        if (not in_pos) and bool(buy_sig.iloc[i]):
            in_pos = True
            buy_mark[i] = True
            entry_price = float(d["Close"].iloc[i])
            atr = float(d[atr_col].iloc[i]) if pd.notna(d[atr_col].iloc[i]) else math.nan
            stop_price = entry_price - float(atr_mult) * atr if np.isfinite(atr) else math.nan
        if in_pos:
            low = float(d["Low"].iloc[i])
            close = float(d["Close"].iloc[i])
            sma20 = float(d[sma20_col].iloc[i]) if pd.notna(d[sma20_col].iloc[i]) else math.nan
            hit_stop = np.isfinite(stop_price) and (low < stop_price)
            hit_sma20 = np.isfinite(sma20) and (close < sma20)
            if hit_stop or hit_sma20:
                pending_exit_reason = "Stop Loss" if hit_stop else "SMA20 Trailing Exit"
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
        trades.append({"entry_date": en, "exit_date": ex, "entry_px": ep, "exit_px": xp,
                       "pnl": (xp / ep - 1.0) * float(initial_capital),
                       "pnl_pct": xp / ep - 1.0,
                       "exit_reason": str(d.loc[ex, "exit_reason"]) if d.loc[ex, "exit_reason"] else "Exit"})
        j += 1
    return BacktestResult(df=d, equity_strategy=equity, equity_buyhold=buyhold,
                          trades=pd.DataFrame(trades))


def bt_perf_metrics(res: BacktestResult, initial_capital: float) -> dict[str, float]:
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
    r = res.df.get("strategy_ret", pd.Series(dtype=float)).astype(float)
    if len(r) > 1 and float(r.std()) > 1e-12:
        sharpe = float(np.sqrt(TRADING_DAYS) * (r.mean() / r.std()))
    else:
        sharpe = np.nan
    calmar = float(ann_ret / abs(max_dd)) if max_dd < -1e-12 else np.nan
    excess_total = float(total_ret - bh_ret)
    return {"total_return": total_ret, "annualized_return": ann_ret, "max_drawdown": max_dd,
            "win_rate": win_rate, "buy_hold_return": bh_ret, "sharpe": sharpe,
            "calmar": calmar, "excess_total": excess_total}


def bt_plot_equity(res: BacktestResult) -> go.Figure:
    d = res.df
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["equity_buyhold"], mode="lines", name="Buy & Hold"))
    fig.add_trace(go.Scatter(x=d.index, y=d["equity_strategy"], mode="lines", name="Strategy"))
    fig.update_layout(title="Equity Curve (Strategy vs Buy & Hold)", xaxis_title="Date",
                      yaxis_title="Equity ($)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
                      height=420)
    return fig


def bt_plot_signals(res: BacktestResult, mode: str) -> go.Figure:
    d = res.df
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], mode="lines", name="Close",
                              line=dict(width=2)))
    if mode == "Advanced":
        buy_idx = d.index[d["buy_mark"].fillna(False)]
        fig.add_trace(go.Scatter(x=buy_idx, y=d.loc[buy_idx, "Close"], mode="markers",
                                  name="Buy", marker=dict(symbol="triangle-up", size=12, color="#00C853"),
                                  hovertemplate="Buy<br>%{x|%Y-%m-%d}<br>Close=%{y:.2f}<extra></extra>"))
        stop_mask = d["exit_reason"].astype(str).eq("Stop Loss")
        sma_mask = d["exit_reason"].astype(str).eq("SMA20 Trailing Exit")
        stop_idx = d.index[d["sell_mark"].fillna(False) & stop_mask]
        sma_idx = d.index[d["sell_mark"].fillna(False) & sma_mask]
        other_idx = d.index[d["sell_mark"].fillna(False) & ~(stop_mask | sma_mask)]
        if len(stop_idx):
            fig.add_trace(go.Scatter(x=stop_idx, y=d.loc[stop_idx, "Close"], mode="markers",
                                      name="Sell (Stop Loss)",
                                      marker=dict(symbol="triangle-down", size=12, color="#FF5252"),
                                      hovertemplate="Stop Loss Exit<br>%{x|%Y-%m-%d}<br>Close=%{y:.2f}<extra></extra>"))
        if len(sma_idx):
            fig.add_trace(go.Scatter(x=sma_idx, y=d.loc[sma_idx, "Close"], mode="markers",
                                      name="Sell (SMA20 Exit)",
                                      marker=dict(symbol="triangle-down", size=12, color="#FFB300"),
                                      hovertemplate="SMA20 Exit<br>%{x|%Y-%m-%d}<br>Close=%{y:.2f}<extra></extra>"))
        if len(other_idx):
            fig.add_trace(go.Scatter(x=other_idx, y=d.loc[other_idx, "Close"], mode="markers",
                                      name="Sell", marker=dict(symbol="triangle-down", size=12, color="#FF5252")))
    else:
        chg = d["position"].diff().fillna(0)
        buy_idx = d.index[chg == 1]
        sell_idx = d.index[chg == -1]
        fig.add_trace(go.Scatter(x=buy_idx, y=d.loc[buy_idx, "Close"], mode="markers", name="Buy",
                                  marker=dict(symbol="triangle-up", size=12, color="#00C853")))
        fig.add_trace(go.Scatter(x=sell_idx, y=d.loc[sell_idx, "Close"], mode="markers", name="Sell",
                                  marker=dict(symbol="triangle-down", size=12, color="#FF5252")))
    fig.update_layout(title="Price with Buy/Sell Signals", xaxis_title="Date", yaxis_title="Price",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
                      height=460)
    return fig


def bt_year_folds(index: pd.DatetimeIndex, train_years: int, test_years: int,
                   rolling: bool) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if len(index) == 0:
        return []
    start = pd.Timestamp(index.min()).normalize()
    end = pd.Timestamp(index.max()).normalize()
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


def bt_slice_by_dates(df: pd.DataFrame, train_end: pd.Timestamp,
                       test_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = df.index
    train = df.loc[idx <= train_end].copy()
    test = df.loc[(idx > train_end) & (idx <= test_end)].copy()
    return train, test


def bt_objective(m: dict[str, float], objective: str, dd_cap: float | None,
                  min_trades: int | None, n_trades: int) -> float:
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
    if dd_cap is not None and float(m.get("max_drawdown", 0.0)) < -abs(float(dd_cap)):
        return -np.inf
    if min_trades is not None and int(n_trades) < int(min_trades):
        return -np.inf
    return float(m.get("excess_total", np.nan))


# ═════════════════════════════════════════════════════════════════════════════
# PULLBACK ANALYZER helpers
# ═════════════════════════════════════════════════════════════════════════════
def pa_calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("RSI")


def pa_find_runup_pullback(close: pd.Series, pullback_pct: float = 0.05) -> pd.DataFrame:
    prices = close.values
    dates = close.index
    n = len(prices)
    events: list[dict] = []
    in_pullback = False
    peak_i = 0
    trough_i = 0
    cycle_start_i = 0
    for i in range(1, n):
        if not in_pullback:
            if prices[i] > prices[peak_i]:
                peak_i = i
            dd = (prices[i] - prices[peak_i]) / prices[peak_i]
            if dd <= -pullback_pct:
                run_up_pct = (prices[peak_i] - prices[cycle_start_i]) / prices[cycle_start_i] * 100
                days_rally = (dates[peak_i] - dates[cycle_start_i]).days
                events.append({"rally_start": dates[cycle_start_i],
                               "rally_start_price": round(float(prices[cycle_start_i]), 2),
                               "peak_date": dates[peak_i],
                               "peak_price": round(float(prices[peak_i]), 2),
                               "run_up_pct": round(run_up_pct, 2),
                               "rally_days": days_rally,
                               "dd_trigger_date": dates[i],
                               "dd_at_trigger_pct": round(float(dd) * 100, 2)})
                in_pullback = True
                trough_i = i
        else:
            if prices[i] < prices[trough_i]:
                trough_i = i
            recovery = (prices[i] - prices[trough_i]) / prices[trough_i]
            if recovery >= pullback_pct:
                in_pullback = False
                cycle_start_i = trough_i
                peak_i = i
    return pd.DataFrame(events)


def pa_find_ob_streaks(close: pd.Series, rsi: pd.Series, ob_level: float = 70) -> pd.DataFrame:
    rows: list[dict] = []
    in_streak = False
    streak_start = 0
    streak_len = 0
    for i in range(len(rsi)):
        if pd.isna(rsi.iloc[i]):
            continue
        if rsi.iloc[i] >= ob_level:
            if not in_streak:
                in_streak = True
                streak_start = i
                streak_len = 1
            else:
                streak_len += 1
        else:
            if in_streak and streak_len >= 3:
                end_i = i
                fwd: dict[str, float | None] = {}
                for days in [5, 10, 21]:
                    if end_i + days < len(close):
                        fwd[f"fwd_{days}d_pct"] = round(
                            (close.iloc[end_i + days] - close.iloc[end_i]) / close.iloc[end_i] * 100, 2)
                    else:
                        fwd[f"fwd_{days}d_pct"] = None
                rows.append({"streak_start": rsi.index[streak_start],
                             "streak_end": rsi.index[end_i - 1],
                             "duration_days": streak_len,
                             "peak_rsi": round(float(rsi.iloc[streak_start:end_i].max()), 1),
                             **fwd})
            in_streak = False
            streak_len = 0
    return pd.DataFrame(rows)


def pa_percentile_rank(series: pd.Series, value: float) -> float:
    clean = series.dropna()
    if clean.empty:
        return 50.0
    return float((clean <= value).mean() * 100)


def pa_risk_color(score: float) -> str:
    if score < 30: return "#2ecc71"
    if score < 55: return "#f1c40f"
    if score < 75: return "#e67e22"
    return "#e74c3c"


def pa_risk_label(score: float) -> str:
    if score < 30: return "Low"
    if score < 55: return "Moderate"
    if score < 75: return "High"
    return "Extreme"


def pa_risk_badge(title: str, score: float) -> str:
    col = pa_risk_color(score)
    lbl = pa_risk_label(score)
    return (f"<span style='font-size:1.25rem;font-weight:700'>{title}</span>&nbsp;&nbsp;"
            f"<span style='background:{col};color:#111;font-weight:700;"
            f"padding:3px 10px;border-radius:12px;font-size:0.9rem'>"
            f"Risk {score:.0f}/100 — {lbl}</span>")


def pa_mini_gauge(score: float, title: str) -> go.Figure:
    col = pa_risk_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": "/100", "font": {"size": 22, "color": col}},
        gauge={"axis": {"range": [0, 100], "showticklabels": False, "ticks": ""},
               "bar": {"color": col, "thickness": 0.30},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, 30], "color": "rgba(46,204,113,0.20)"},
                          {"range": [30, 55], "color": "rgba(241,196,15,0.20)"},
                          {"range": [55, 75], "color": "rgba(230,126,34,0.20)"},
                          {"range": [75, 100], "color": "rgba(231,76,60,0.20)"}]}))
    fig.update_layout(height=130, margin=dict(t=28, b=0, l=10, r=10),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
                      title=dict(text=f"<b>{title}</b>", font=dict(size=11, color="#aaa"), x=0.5))
    return fig


def pa_calc_bias_risk(cur_bias: float | None, bias_series: pd.Series) -> float:
    if cur_bias is None or np.isnan(cur_bias):
        return 50.0
    raw = pa_percentile_rank(bias_series, cur_bias)
    return round(max(0.0, min(100.0, raw)), 1)


def pa_calc_runup_risk(current_runup: float, historical_runups: pd.Series) -> float:
    if historical_runups.empty or current_runup <= 0:
        return 20.0
    raw = pa_percentile_rank(historical_runups, current_runup)
    return round(max(0.0, min(100.0, raw)), 1)


def pa_calc_rsi_risk(cur_rsi: float | None, cur_streak: int, ob_level: float) -> float:
    if cur_rsi is None or np.isnan(cur_rsi):
        return 20.0
    rsi_floor = max(ob_level - 20, 40)
    rsi_component = max(0.0, min(70.0, (cur_rsi - rsi_floor) / (100 - rsi_floor) * 70))
    streak_component = min(30.0, cur_streak * 2.5)
    return round(min(100.0, rsi_component + streak_component), 1)


def pa_calc_stock_pullback_baseline(close: pd.Series, noise_filter: float = 0.03,
                                    fallback: float = 0.05) -> tuple[float, list[float]]:
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
                trough = price
                in_drawdown = True
        else:
            if price < trough:
                trough = price
            elif (price - trough) / trough >= noise_filter:
                dd = (peak - trough) / peak
                if dd >= noise_filter:
                    drawdowns.append(dd)
                peak = price
                trough = price
                in_drawdown = False
    if in_drawdown:
        dd = (peak - trough) / peak
        if dd >= noise_filter:
            drawdowns.append(dd)
    if not drawdowns:
        return fallback, []
    return float(np.median(drawdowns)), drawdowns


_PA_L = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
             font=dict(color="white", size=11), margin=dict(t=40, b=0, l=0, r=0),
             hovermode="x unified", xaxis=dict(showgrid=False),
             yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False))


# ═════════════════════════════════════════════════════════════════════════════
# PAGE RENDERERS
# ═════════════════════════════════════════════════════════════════════════════
def render_markowitz() -> None:
    st.title("Markowitz Efficient Frontier & Portfolio Optimization")
    st.caption(
        f"**Build `{OPTIMIZER_BUILD_ID}`** — Mean–variance with Monte Carlo + SciPy **trust-constr** "
        f"(explicit per-asset **Bounds**). Max-Sharpe and min-vol: **{WEIGHT_MIN:.0%}–{WEIGHT_MAX:.0%}** per name. "
        f"Requires **{MIN_ASSETS_FOR_WEIGHT_CAP}–{MAX_ASSETS_FOR_WEIGHT_FLOOR}** tickers. "
        "Education/research only — not investment advice."
    )

    default_end = date.today()
    default_start = default_end - timedelta(days=365 * 5)
    defaults = ["NVDA", "AAPL", "MSFT", "AMZN", "JPM", "JNJ", "XOM", "CAT", "TLT", "GLD"]

    st.subheader("Inputs")
    tickers_in: list[str] = []
    cols = st.columns(5)
    for i in range(10):
        with cols[i % 5]:
            v = st.text_input(f"Ticker {i + 1}", value=defaults[i], key=f"mk_t{i}")
            tickers_in.append(v)

    c1, c2, c3 = st.columns(3)
    with c1:
        start_d = st.date_input("Start date", value=default_start, key="mk_start")
    with c2:
        end_d = st.date_input("End date", value=default_end, key="mk_end")
    with c3:
        @st.cache_data(ttl=3600)
        def _fetch_rf() -> float:
            quote = latest_quotes(["^IRX"]).get("^IRX", {})
            val = quote.get("price")
            if val and val > 0:
                return round(float(val) / 100, 4)
            try:
                irx = yf.Ticker("^IRX").fast_info
                val = getattr(irx, "last_price", None)
                if val and val > 0:
                    return round(float(val) / 100, 4)
            except Exception:
                pass
            return 0.04

        _auto_rf = _fetch_rf()
        rf = st.number_input("Risk-free rate (annual, decimal)", min_value=0.0, max_value=0.5,
                              value=_auto_rf, step=0.001, format="%.4f",
                              help="US 13-week T-bill (^IRX). Edit manually if needed.",
                              key="mk_rf")
        st.caption(f"^IRX estimate: {_auto_rf * 100:.2f}%")

    run = st.button("Run optimization", type="primary", key="mk_run")

    tickers = _parse_tickers(tickers_in)
    if not tickers:
        st.info("Enter at least one ticker above.")
        return
    if start_d >= end_d:
        st.error("Start date must be before end date.")
        return
    if not run:
        st.info("Set tickers and dates, then click **Run optimization**.")
        return

    try:
        with st.spinner("Loading prices and building returns…"):
            prices = mk_fetch_adj_close(tickers, start_d, end_d)
            valid_cols = [c for c in tickers if c in prices.columns]
            if len(valid_cols) < len(tickers):
                missing = set(tickers) - set(valid_cols)
                st.warning(f"No data for: {', '.join(sorted(missing))}. Using: {', '.join(valid_cols)}")
            if len(valid_cols) < 2:
                st.error("Need at least two assets with data.")
                return
            if not _feasible_weight_cap(len(valid_cols)):
                st.error(f"With per-asset bounds **{WEIGHT_MIN:.0%}–{WEIGHT_MAX:.0%}**, "
                         f"need **{MIN_ASSETS_FOR_WEIGHT_CAP}–{MAX_ASSETS_FOR_WEIGHT_FLOOR}** tickers.")
                return
            prices = prices[valid_cols]
            daily_ret = prices.pct_change().dropna()
            mu_ann, cov_ann = mk_ann_stats(daily_ret)
            corr = daily_ret.corr()

        rng = np.random.default_rng(42)
        with st.spinner(f"Monte Carlo: {N_MONTE_CARLO:,} portfolios…"):
            mc = mk_monte_carlo_portfolios(mu_ann, cov_ann, rf, N_MONTE_CARLO, rng)

        with st.spinner("SciPy: max Sharpe & min volatility…"):
            w_max_sharpe = mk_optimize_max_sharpe(mu_ann, cov_ann, rf)
            w_min_vol = mk_optimize_min_volatility(cov_ann)

        r_ms = mk_portfolio_return(w_max_sharpe, mu_ann)
        v_ms = mk_portfolio_volatility(w_max_sharpe, cov_ann)
        sr_ms = (r_ms - rf) / v_ms if v_ms > 1e-12 else np.nan
        r_mv = mk_portfolio_return(w_min_vol, mu_ann)
        v_mv = mk_portfolio_volatility(w_min_vol, cov_ann)
        sr_mv = (r_mv - rf) / v_mv if v_mv > 1e-12 else np.nan

    except Exception as e:
        st.error(str(e))
        return

    st.subheader("1. Efficient Frontier (Monte Carlo + optima)")
    fig = px.scatter(mc, x="ann_volatility", y="ann_return", color="sharpe_ratio",
                     color_continuous_scale="Viridis", opacity=0.35,
                     labels={"ann_volatility": "Annualized volatility (σ)",
                              "ann_return": "Annualized expected return (μ)",
                              "sharpe_ratio": "Sharpe ratio"},
                     title=f"{N_MONTE_CARLO:,} random long-only portfolios, each weight ∈ [{WEIGHT_MIN:.0%}, {WEIGHT_MAX:.0%}]",
                     height=520)
    fig.add_trace(go.Scatter(x=[v_ms], y=[r_ms], mode="markers",
                              marker=dict(symbol="star", size=22, color="gold", line=dict(width=1, color="white")),
                              name="Max Sharpe (SciPy)",
                              hovertemplate="Max Sharpe<br>σ=%{x:.4f}<br>μ=%{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[v_mv], y=[r_mv], mode="markers",
                              marker=dict(symbol="star", size=22, color="cyan", line=dict(width=1, color="white")),
                              name="Min volatility (SciPy)",
                              hovertemplate="Min vol<br>σ=%{x:.4f}<br>μ=%{y:.4f}<extra></extra>"))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2. Optimal Weights (pie charts + tables)")
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(f"**Max-Sharpe portfolio** (Sharpe ≈ {sr_ms:.3f})")
        df_ms = pd.DataFrame({"Asset": valid_cols, "Weight %": w_max_sharpe * 100.0}).sort_values("Weight %", ascending=False)
        fig_p1 = px.pie(df_ms, names="Asset", values="Weight %", hole=0.35, title="Weights — max Sharpe")
        fig_p1.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_p1, use_container_width=True)
        df_ms["Weight %"] = df_ms["Weight %"].round(2)
        st.dataframe(df_ms, use_container_width=True, hide_index=True)
    with c_right:
        st.markdown(f"**Min-volatility portfolio** (Sharpe ≈ {sr_mv:.3f})")
        df_mv = pd.DataFrame({"Asset": valid_cols, "Weight %": w_min_vol * 100.0}).sort_values("Weight %", ascending=False)
        fig_p2 = px.pie(df_mv, names="Asset", values="Weight %", hole=0.35, title="Weights — min volatility")
        fig_p2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_p2, use_container_width=True)
        df_mv["Weight %"] = df_mv["Weight %"].round(2)
        st.dataframe(df_mv, use_container_width=True, hide_index=True)

    st.subheader("3. Historical Correlation Heatmap")
    fig_h = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                      zmin=-1, zmax=1, title="Correlation matrix of daily returns")
    fig_h.update_layout(height=480)
    st.plotly_chart(fig_h, use_container_width=True)

    with st.expander("Methodology"):
        st.markdown(
            f"- **Returns:** simple daily returns; annualized mean `× 252`, covariance `× 252`.\n"
            f"- **Long-only with bounds:** each `w_i ∈ [{WEIGHT_MIN}, {WEIGHT_MAX}]`, `∑w = 1`.\n"
            f"- **Max Sharpe:** trust-constr + SLSQP fallback.\n"
            f"- **Min variance:** same solvers.\n"
            f"- **Monte Carlo:** Dirichlet rejection sampling on the same per-asset cap."
        )


def render_backtester() -> None:
    st.title("Technical Strategy Backtester")
    st.caption("Data: local SQLite first, yfinance fallback · Indicators: pandas/numpy · For research/education only.")

    with st.sidebar:
        st.header("Backtest")
        ticker = st.text_input("Ticker", value=st.session_state.get("bt_ticker", "SPY"),
                                key="bt_ticker_in").strip().upper() or "SPY"
        st.session_state["bt_ticker"] = ticker
        end_d = date.today()
        start_d = st.date_input("Start date", value=end_d - timedelta(days=365 * 10), key="bt_start")
        end_d_in = st.date_input("End date", value=end_d, key="bt_end")
        if start_d >= end_d_in:
            st.error("Start date must be before end date.")
            st.stop()
        initial_capital = st.number_input("Initial capital ($)", min_value=100.0, max_value=10_000_000.0,
                                           value=float(st.session_state.get("bt_capital", 10_000.0)),
                                           step=500.0, format="%.0f", key="bt_capital")
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
        run_backtest = st.button("Run backtest", type="primary", use_container_width=True, key="bt_run")
        if run_backtest:
            st.session_state["bt_run_request"] = True
        st.divider()
        with st.expander("Optimizer (optional)", expanded=False):
            st.caption("Run optimizer first, then optionally apply best params.")
            preset = st.selectbox("Preset", ["Control drawdown & beat Buy&Hold", "Maximize return"],
                                  index=0, key="opt_preset")
            opt_mode = st.selectbox("Optimize which strategy?", ["Advanced", "Simple (vectorized)"],
                                    index=0, key="opt_mode")
            objective = st.selectbox("Objective",
                                     ["Excess Return vs Buy&Hold", "Total Return", "Annualized Return",
                                      "Sharpe", "Calmar (AnnRet/MaxDD)",
                                      "Custom (constraints + Excess Return)"],
                                     index=0, key="opt_obj")
            rolling = st.checkbox("Rolling walk-forward (step 1y)", value=True, key="opt_roll")
            oc1, oc2 = st.columns(2)
            with oc1:
                train_years = st.number_input("Train years", min_value=2, max_value=20, value=7, step=1, key="opt_train")
            with oc2:
                test_years = st.number_input("Test years", min_value=1, max_value=10, value=3, step=1, key="opt_test")
            dd_cap = None
            min_trades = None
            if objective.startswith("Custom"):
                oc3, oc4 = st.columns(2)
                with oc3:
                    dd_cap = st.slider("Max drawdown cap (%)", 5, 60, 25, 1, key="opt_dd") / 100.0
                with oc4:
                    min_trades = st.number_input("Min trades (per fold)", min_value=0, max_value=200,
                                                 value=4, step=1, key="opt_tr")
            if st.button("Apply preset defaults", key="opt_apply_preset"):
                if preset == "Control drawdown & beat Buy&Hold":
                    st.session_state["opt_obj"] = "Custom (constraints + Excess Return)"
                    st.session_state["opt_dd"] = 25
                    st.session_state["opt_tr"] = 4
                    st.session_state["opt_roll"] = True
                    st.session_state["opt_train"] = 7
                    st.session_state["opt_test"] = 3
                else:
                    st.session_state["opt_obj"] = "Total Return"
                    st.session_state["opt_roll"] = True
                    st.session_state["opt_train"] = 7
                    st.session_state["opt_test"] = 3
                st.success("Preset applied.")
            st.markdown("#### Grid")
            if opt_mode == "Advanced":
                rsi_buy_min, rsi_buy_max = st.slider("RSI buy threshold range", 5, 60, (25, 45), 1, key="g_rsi_buy")
                atr_mult_min, atr_mult_max = st.slider("ATR multiple range", 1.0, 6.0, (2.0, 4.0), 0.25, key="g_atr")
                sma_exit_min, sma_exit_max = st.slider("Exit SMA range", 10, 200, (20, 80), 5, key="g_exit")
                sma_trend_min, sma_trend_max = st.slider("Trend SMA range", 100, 300, (150, 250), 10, key="g_trend")
            else:
                rsi_buy_min, rsi_buy_max = st.slider("RSI buy threshold range", 5, 60, (25, 45), 1, key="g_s_rsi_buy")
                rsi_sell_min, rsi_sell_max = st.slider("RSI sell threshold range", 40, 99, (60, 80), 1, key="g_s_rsi_sell")
                sma_min, sma_max = st.slider("SMA length range", 20, 300, (100, 250), 5, key="g_s_sma")
            max_combos = st.number_input("Max combinations", min_value=20, max_value=2000, value=300, step=20, key="opt_maxc")
            run_opt = st.button("Run optimizer", type="primary", key="opt_run")
            if run_opt:
                st.session_state["opt_run_request"] = True

    run_bt = bool(st.session_state.pop("bt_run_request", False))
    run_opt_req = bool(st.session_state.pop("opt_run_request", False))

    if not (run_bt or run_opt_req) and "last_backtest" not in st.session_state and "opt_results" not in st.session_state:
        st.info("Use the sidebar to configure and run a backtest or optimizer.")

    raw: pd.DataFrame | None = None
    if run_bt or run_opt_req:
        with st.spinner("Loading data…"):
            raw = bt_load_ohlcv(ticker, start_d, end_d_in)
        if raw.empty or len(raw) < 250:
            st.error("Not enough data. Try a longer period or different ticker.")
            return

    if run_bt:
        sma_lens_needed = (int(sma_len), int(sma_200_len), int(sma_exit_len))
        with st.spinner("Computing indicators…"):
            df_bt = bt_prepare_indicators(raw, rsi_len=int(rsi_len), atr_len=int(atr_len),
                                           sma_lens=sma_lens_needed)
        if df_bt.empty:
            st.error("Indicators produced no usable rows.")
            return
        with st.spinner("Running strategy…"):
            if mode == "Advanced":
                res = bt_backtest_advanced(df=df_bt, rsi_len=rsi_len, rsi_buy=float(rsi_buy),
                                           sma_200_len=sma_200_len, sma_20_len=sma_exit_len,
                                           atr_len=atr_len, atr_mult=float(atr_mult),
                                           initial_capital=float(initial_capital))
            else:
                res = bt_backtest_simple(df=df_bt, sma_len=sma_len, rsi_len=rsi_len,
                                         rsi_buy=float(rsi_buy), rsi_sell=float(rsi_sell),
                                         initial_capital=float(initial_capital))
        st.session_state["last_backtest"] = {"res": res, "capital": float(initial_capital)}

    if run_opt_req:
        try:
            folds = bt_year_folds(raw.index, int(st.session_state.get("opt_train", 7)),
                                  int(st.session_state.get("opt_test", 3)),
                                  bool(st.session_state.get("opt_roll", True)))
            if not folds:
                st.error("Not enough data for walk-forward folds.")
            else:
                opt_mode_cur = st.session_state.get("opt_mode", "Advanced")
                objective_cur = st.session_state.get("opt_obj", "Excess Return vs Buy&Hold")
                _dd_cap = (st.session_state.get("opt_dd", None) / 100.0) if str(objective_cur).startswith("Custom") else None
                _min_trades = int(st.session_state.get("opt_tr", 0)) if str(objective_cur).startswith("Custom") else None
                _max_combos = int(st.session_state.get("opt_maxc", 300))
                grid: list[dict[str, float]] = []
                required_sma: set[int] = set()
                if opt_mode_cur == "Advanced":
                    rb_min, rb_max = st.session_state.get("g_rsi_buy", (25, 45))
                    am_min, am_max = st.session_state.get("g_atr", (2.0, 4.0))
                    ex_min, ex_max = st.session_state.get("g_exit", (20, 80))
                    tr_min, tr_max = st.session_state.get("g_trend", (150, 250))
                    rsi_vals = list(range(int(rb_min), int(rb_max) + 1, 1))
                    atr_vals = list(np.arange(float(am_min), float(am_max) + 1e-9, 0.25))
                    exit_vals = list(range(int(ex_min), int(ex_max) + 1, 5))
                    trend_vals = list(range(int(tr_min), int(tr_max) + 1, 10))
                    required_sma.update(exit_vals)
                    required_sma.update(trend_vals)
                    for rb in rsi_vals:
                        for am in atr_vals:
                            for exl in exit_vals:
                                for trl in trend_vals:
                                    grid.append({"rsi_buy": float(rb), "atr_mult": float(am),
                                                 "sma_exit": float(exl), "sma_trend": float(trl)})
                else:
                    rb_min, rb_max = st.session_state.get("g_s_rsi_buy", (25, 45))
                    rs_min, rs_max = st.session_state.get("g_s_rsi_sell", (60, 80))
                    sm_min, sm_max = st.session_state.get("g_s_sma", (100, 250))
                    rsi_b_vals = list(range(int(rb_min), int(rb_max) + 1, 1))
                    rsi_s_vals = list(range(int(rs_min), int(rs_max) + 1, 1))
                    sma_vals = list(range(int(sm_min), int(sm_max) + 1, 5))
                    required_sma.update(sma_vals)
                    for rb in rsi_b_vals:
                        for rs in rsi_s_vals:
                            for sl in sma_vals:
                                grid.append({"rsi_buy": float(rb), "rsi_sell": float(rs), "sma": float(sl)})
                if len(grid) > _max_combos:
                    rng = np.random.default_rng(42)
                    pick = rng.choice(len(grid), size=_max_combos, replace=False)
                    grid = [grid[int(i)] for i in pick]
                base = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                df_all = bt_prepare_indicators(base, int(rsi_len), int(atr_len), tuple(sorted(required_sma)))
                rows: list[dict[str, object]] = []
                with st.spinner(f"Optimizer running… combos={len(grid):,}, folds={len(folds):,}"):
                    prog = st.progress(0.0)
                    for gi, params in enumerate(grid):
                        fold_scores = []
                        fold_tests = []
                        for train_end, test_end in folds:
                            train_i, test_i = bt_slice_by_dates(df_all, train_end, test_end)
                            if len(train_i) < 200 or len(test_i) < 50:
                                continue
                            if opt_mode_cur == "Advanced":
                                res_tr = bt_backtest_advanced(train_i, int(rsi_len), float(params["rsi_buy"]),
                                                              int(params["sma_trend"]), int(params["sma_exit"]),
                                                              int(atr_len), float(params["atr_mult"]), float(initial_capital))
                                m_tr = bt_perf_metrics(res_tr, float(initial_capital))
                                res_te = bt_backtest_advanced(test_i, int(rsi_len), float(params["rsi_buy"]),
                                                              int(params["sma_trend"]), int(params["sma_exit"]),
                                                              int(atr_len), float(params["atr_mult"]), float(initial_capital))
                                m_te = bt_perf_metrics(res_te, float(initial_capital))
                                ntr = 0 if res_te.trades is None else len(res_te.trades)
                            else:
                                res_tr = bt_backtest_simple(train_i, int(params["sma"]), int(rsi_len),
                                                            float(params["rsi_buy"]), float(params["rsi_sell"]), float(initial_capital))
                                m_tr = bt_perf_metrics(res_tr, float(initial_capital))
                                res_te = bt_backtest_simple(test_i, int(params["sma"]), int(rsi_len),
                                                            float(params["rsi_buy"]), float(params["rsi_sell"]), float(initial_capital))
                                m_te = bt_perf_metrics(res_te, float(initial_capital))
                                ntr = 0 if res_te.trades is None else len(res_te.trades)
                            score = bt_objective(m_tr,
                                                 "Custom (constraints + Excess Return)" if str(objective_cur).startswith("Custom") else str(objective_cur),
                                                 dd_cap=_dd_cap, min_trades=_min_trades, n_trades=ntr)
                            if not np.isfinite(score):
                                continue
                            fold_scores.append(float(score))
                            fold_tests.append(m_te)
                        if fold_scores and fold_tests:
                            avg = {k: float(np.nanmean([ft.get(k, np.nan) for ft in fold_tests])) for k in fold_tests[0].keys()}
                            rows.append({**{f"p_{k}": v for k, v in params.items()},
                                         "train_score": float(np.mean(fold_scores)),
                                         **{f"test_{k}": v for k, v in avg.items()}})
                        prog.progress((gi + 1) / max(1, len(grid)))
                    prog.empty()
                if rows:
                    st.session_state["opt_results"] = pd.DataFrame(rows)
                else:
                    st.warning("Optimizer finished, but no valid rows. Try widening ranges.")
        except Exception as e:
            st.error("Optimizer failed.")
            st.exception(e)

    if "last_backtest" in st.session_state:
        res = st.session_state["last_backtest"]["res"]
        cap = float(st.session_state["last_backtest"]["capital"])
        m = bt_perf_metrics(res, cap)
        st.subheader("Backtest Results")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Total Return", _pct(m.get("total_return", np.nan)))
        with c2: st.metric("Annualized Return", _pct(m.get("annualized_return", np.nan)))
        with c3: st.metric("Max Drawdown", _pct(m.get("max_drawdown", np.nan)))
        with c4: st.metric("Win Rate", _pct(m.get("win_rate", np.nan)))
        with c5: st.metric("Buy & Hold Return", _pct(m.get("buy_hold_return", np.nan)))
        st.caption(f"Initial capital: **{_money(cap)}** · Rows: **{len(res.df):,}** · Trades: **{0 if res.trades is None else len(res.trades):,}**")
        st.plotly_chart(bt_plot_equity(res), use_container_width=True)
        st.plotly_chart(bt_plot_signals(res, "Advanced" if st.session_state.get("bt_mode") == "Advanced" else "Simple"),
                        use_container_width=True)
        with st.expander("Trades (entry/exit)", expanded=False):
            if res.trades is None or res.trades.empty:
                st.info("No completed trades.")
            else:
                tdf = res.trades.copy()
                tdf["entry_date"] = pd.to_datetime(tdf["entry_date"]).dt.strftime("%Y-%m-%d")
                tdf["exit_date"] = pd.to_datetime(tdf["exit_date"]).dt.strftime("%Y-%m-%d")
                tdf["pnl_pct"] = (tdf["pnl_pct"] * 100.0).round(2)
                tdf = tdf.rename(columns={"entry_date": "Entry", "exit_date": "Exit",
                                          "entry_px": "Entry Px", "exit_px": "Exit Px",
                                          "pnl_pct": "PnL %", "exit_reason": "Exit reason"})
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
        st.subheader(f"Optimizer Results (sorted by {sort_label})")
        show_cols = [c for c in out.columns if c.startswith("p_")] + [
            "train_score", "test_total_return", "test_annualized_return",
            "test_buy_hold_return", "test_excess_total", "test_sharpe",
            "test_calmar", "test_max_drawdown", "test_win_rate"]
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


def render_pullback() -> None:
    st.title("Pullback Analyzer")
    st.caption(
        "Three quantitative lenses: **Bias Ratio** (distance from SMA), "
        "**Run-up before Drawdown** statistics, and **RSI Overbought Duration** analysis. "
        "For research only — not investment advice."
    )

    with st.sidebar:
        st.header("Parameters")
        ticker = st.text_input("Ticker", value="QQQ", key="pa_ticker_in").strip().upper() or "QQQ"
        years = st.slider("Lookback (years)", 3, 15, 10, key="pa_years")
        auto_stock_pb = st.checkbox("Auto-calculate pullback threshold (stocks only)", value=True,
                                    key="pa_auto_pb",
                                    help="Auto-compute median swing drawdown; for index ETFs, manual slider is used.")
        stock_noise_filter_pct = st.slider("Noise filter for stock auto-calc (%)", min_value=1, max_value=10,
                                           value=3, key="pa_noise",
                                           help="Swing drawdowns smaller than this are ignored.",
                                           disabled=not auto_stock_pb) / 100
        pullback_pct = st.slider("Pullback threshold (%)", 3, 20, 5, key="pa_pb") / 100
        ob_level = st.slider("RSI overbought level", 65, 80, 70, key="pa_ob")
        sma_period = st.selectbox("Bias ratio SMA", [50, 100, 200], index=2, key="pa_sma")
        run_btn = st.button("Run Analysis", type="primary", use_container_width=True, key="pa_run")

    if "pa_df" not in st.session_state:
        run_btn = True

    if run_btn:
        end = date.today()
        start = end - timedelta(days=365 * years + 30)
        with st.spinner(f"Loading {ticker} ({years}y)…"):
            try:
                local = load_adjusted_close([ticker], start, end, min_rows=200)
                if ticker in local.columns:
                    close = local[ticker].dropna()
                else:
                    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
                    if raw.empty:
                        st.error(f"No data for {ticker}.")
                        return
                    close = raw["Close"].squeeze().dropna()
                if close.empty:
                    st.error(f"No data for {ticker}.")
                    return
                close.index = pd.to_datetime(close.index)
                st.session_state.update({"pa_close": close, "pa_ticker": ticker,
                                         "pa_sma_val": sma_period, "pa_ob_val": ob_level, "pa_df": True})
            except Exception as e:
                st.error(str(e))
                return
    elif st.session_state.get("pa_ticker") != ticker:
        st.warning("Ticker changed — click **Run Analysis** to reload.")

    if "pa_close" not in st.session_state:
        return

    close = st.session_state["pa_close"]
    sma_period = st.session_state.get("pa_sma_val", sma_period)
    ob_level = st.session_state.get("pa_ob_val", ob_level)
    sym = st.session_state["pa_ticker"]
    x = close.index

    INDEX_ETFS = {"QQQ", "SPY", "DIA", "IWM", "VTI", "VOO"}
    if auto_stock_pb and sym not in INDEX_ETFS:
        auto_pb, dds = pa_calc_stock_pullback_baseline(close, noise_filter=stock_noise_filter_pct,
                                                       fallback=pullback_pct)
        pullback_pct = auto_pb
        st.sidebar.info(f"📐 **{sym}** auto pullback: **{pullback_pct * 100:.1f}%**\n\n"
                        f"Median of **{len(dds)}** swing drawdowns (≥ {stock_noise_filter_pct * 100:.0f}%) over {years}y")
    elif sym in INDEX_ETFS:
        st.sidebar.caption(f"Index/ETF: using manual threshold (**{pullback_pct * 100:.0f}%**).")

    sma = close.rolling(sma_period).mean()
    bias_pct = (close - sma) / sma * 100
    rsi = pa_calc_rsi(close)

    cur_price = float(close.iloc[-1])
    cur_sma = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None
    cur_bias = float(bias_pct.iloc[-1]) if cur_sma else None
    cur_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

    cur_ob_streak = 0
    for v in reversed(rsi.dropna().values):
        if v >= ob_level:
            cur_ob_streak += 1
        else:
            break

    events_pre = pa_find_runup_pullback(close, pullback_pct)
    if not events_pre.empty:
        last_peak_row = events_pre.iloc[-1]
        last_cycle_end = pd.to_datetime(last_peak_row["dd_trigger_date"])
        recent_slice = close[close.index > last_cycle_end]
        cur_runup = ((recent_slice.max() - recent_slice.min()) / recent_slice.min() * 100
                     if not recent_slice.empty else 0.0)
        hist_runups = events_pre["run_up_pct"]
    else:
        cur_runup = 0.0
        hist_runups = pd.Series([], dtype=float)

    r_bias = pa_calc_bias_risk(cur_bias, bias_pct)
    r_runup = pa_calc_runup_risk(cur_runup, hist_runups)
    r_rsi = pa_calc_rsi_risk(cur_rsi, cur_ob_streak, ob_level)
    r_total = round((r_bias + r_runup + r_rsi) / 3, 1)

    st.markdown("### Overall Risk Dashboard")
    g1, g2, g3, g4 = st.columns(4)
    with g1: st.plotly_chart(pa_mini_gauge(r_total, "⚡ Overall Risk"), use_container_width=True)
    with g2: st.plotly_chart(pa_mini_gauge(r_bias, f"1. Bias Ratio (SMA{sma_period})"), use_container_width=True)
    with g3: st.plotly_chart(pa_mini_gauge(r_runup, f"2. Run-up (vs ≥{pullback_pct * 100:.0f}% DD)"), use_container_width=True)
    with g4: st.plotly_chart(pa_mini_gauge(r_rsi, f"3. RSI OB (>{ob_level})"), use_container_width=True)

    oc = pa_risk_color(r_total)
    st.markdown(
        f"<div style='background:rgba(255,255,255,0.05);border-left:4px solid {oc};"
        f"padding:10px 16px;border-radius:6px;margin-bottom:8px'>"
        f"<b style='color:{oc}'>Combined Risk: {r_total:.0f}/100 — {pa_risk_label(r_total)}</b>"
        f"&nbsp;·&nbsp; Bias {r_bias:.0f} / Run-up {r_runup:.0f} / RSI {r_rsi:.0f}</div>",
        unsafe_allow_html=True)
    st.divider()

    ca, cb, cc, cd = st.columns(4)
    with ca: st.metric("Current Price", f"${cur_price:,.2f}")
    with cb: st.metric(f"SMA{sma_period}", f"${cur_sma:,.2f}" if cur_sma else "—")
    with cc:
        bias_lbl = f"{cur_bias:+.1f}%" if cur_bias is not None else "—"
        st.metric("Bias Ratio", bias_lbl, delta=bias_lbl,
                  delta_color="inverse" if (cur_bias or 0) > 15 else "normal")
    with cd:
        rsi_lbl = f"{cur_rsi:.1f}" if cur_rsi else "—"
        st.metric("RSI(14)", rsi_lbl,
                  delta="Overbought" if (cur_rsi or 0) >= ob_level else "Normal",
                  delta_color="inverse" if (cur_rsi or 0) >= ob_level else "normal")
    st.divider()

    # 1. Bias Ratio
    st.markdown(pa_risk_badge(f"1. Price Bias Ratio — distance from SMA{sma_period}", r_bias),
                unsafe_allow_html=True)
    st.markdown("")
    fig_bias = go.Figure()
    fig_bias.add_trace(go.Scatter(x=x, y=close, name="Price",
                                   line=dict(color="#3498db", width=1.5),
                                   hovertemplate="%{y:,.2f}<extra>Price</extra>"))
    fig_bias.add_trace(go.Scatter(x=x, y=sma, name=f"SMA{sma_period}",
                                   line=dict(color="#f39c12", width=1.5, dash="dot"),
                                   hovertemplate="%{y:,.2f}<extra>SMA</extra>"))
    fig_bias.update_layout(**_PA_L, height=260,
                            title=dict(text=f"{sym} Price vs SMA{sma_period}", font=dict(size=13), x=0),
                            yaxis_tickprefix="$", legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig_bias, use_container_width=True)

    bias_colors = ["rgba(231,76,60,0.7)" if v > 15 else "rgba(241,196,15,0.7)" if v > 8
                   else "rgba(46,204,113,0.7)" for v in bias_pct.fillna(0)]
    fig_bpct = go.Figure()
    fig_bpct.add_trace(go.Bar(x=x, y=bias_pct, marker_color=bias_colors, name="Bias %",
                               hovertemplate="%{y:+.1f}%<extra>Bias</extra>"))
    fig_bpct.add_hline(y=15, line_color="#e74c3c", line_dash="dash", line_width=1,
                        annotation_text="⚠ +15% caution", annotation_position="top left",
                        annotation_font_color="#e74c3c")
    fig_bpct.add_hline(y=20, line_color="#c0392b", line_dash="dash", line_width=1.5,
                        annotation_text="🚨 +20% extreme", annotation_position="top left",
                        annotation_font_color="#c0392b")
    fig_bpct.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    fig_bpct.update_layout(**_PA_L, height=220,
                            title=dict(text=f"Bias Ratio (%) — how far above SMA{sma_period}", font=dict(size=13), x=0),
                            yaxis_ticksuffix="%")
    st.plotly_chart(fig_bpct, use_container_width=True)

    p75_bias = float(np.nanpercentile(bias_pct, 75))
    p90_bias = float(np.nanpercentile(bias_pct, 90))
    if cur_bias is not None:
        _bc = pa_risk_color(r_bias)
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.05);border-left:4px solid {_bc};"
            f"padding:8px 14px;border-radius:6px;font-size:0.9rem'>"
            f"📊 <b>Current bias: {cur_bias:+.1f}%</b> — "
            f"Historical {years}y: P75 = {p75_bias:+.1f}%, P90 = {p90_bias:+.1f}%</div>",
            unsafe_allow_html=True)
    st.divider()

    # 2. Run-up before Drawdown
    st.markdown(pa_risk_badge(f"2. Run-up statistics before ≥{pullback_pct * 100:.0f}% pullback", r_runup),
                unsafe_allow_html=True)
    st.markdown("")
    events = events_pre
    if events.empty:
        st.info(f"No pullbacks ≥ {pullback_pct * 100:.0f}% found.")
    else:
        avg_runup = events["run_up_pct"].mean()
        med_runup = events["run_up_pct"].median()
        p25_runup = events["run_up_pct"].quantile(0.25)
        p75_runup = events["run_up_pct"].quantile(0.75)
        last_peak_row = events.iloc[-1]
        last_cycle_end = pd.to_datetime(last_peak_row["dd_trigger_date"])
        recent_slice = close[close.index > last_cycle_end]
        current_runup = ((recent_slice.max() - recent_slice.min()) / recent_slice.min() * 100
                         if not recent_slice.empty else 0.0)
        ka, kb, kc, kd, ke = st.columns(5)
        danger = current_runup >= med_runup
        with ka: st.metric("Current run-up (est.)", f"{current_runup:.1f}%",
                            delta="⚠ Near median threshold" if danger else "Below median",
                            delta_color="inverse" if danger else "normal")
        with kb: st.metric("Avg run-up before pullback", f"{avg_runup:.1f}%")
        with kc: st.metric("Median run-up", f"{med_runup:.1f}%")
        with kd: st.metric("P25–P75 range", f"{p25_runup:.1f}%–{p75_runup:.1f}%")
        with ke: st.metric("Pullback events", len(events))
        _rc = pa_risk_color(r_runup)
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.05);border-left:4px solid {_rc};"
            f"padding:8px 14px;border-radius:6px;font-size:0.9rem;margin-bottom:8px'>"
            f"📈 <b>Current rally since last pullback: {cur_runup:.1f}%</b> — "
            f"Median pre-pullback run-up: {med_runup:.1f}%, P75: {p75_runup:.1f}%</div>",
            unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=events["run_up_pct"], nbinsx=20,
                                         marker_color="rgba(52,152,219,0.7)",
                                         marker_line=dict(color="rgba(52,152,219,1)", width=1),
                                         name="Run-up distribution",
                                         hovertemplate="Run-up: %{x:.1f}%<br>Count: %{y}<extra></extra>"))
        fig_hist.add_vline(x=med_runup, line_color="#f1c40f", line_dash="dash",
                           annotation_text=f"Median {med_runup:.1f}%",
                           annotation_font_color="#f1c40f", annotation_position="top right")
        if current_runup > 0:
            fig_hist.add_vline(x=current_runup, line_color="#e74c3c", line_width=2,
                               annotation_text=f"Now {current_runup:.1f}%",
                               annotation_font_color="#e74c3c", annotation_position="top left")
        fig_hist.update_layout(**_PA_L, height=280,
                                title=dict(text=f"Distribution of run-ups before ≥{pullback_pct * 100:.0f}% pullbacks",
                                           font=dict(size=13), x=0),
                                xaxis_ticksuffix="%", showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
        fig_sc = go.Figure(go.Scatter(
            x=events["run_up_pct"], y=events["dd_at_trigger_pct"].abs(),
            mode="markers+text",
            marker=dict(size=10, color=events["run_up_pct"], colorscale="RdYlGn_r", showscale=True,
                        colorbar=dict(title="Run-up %", thickness=12)),
            text=[str(d.year) for d in pd.to_datetime(events["peak_date"])],
            textposition="top center", textfont=dict(size=9),
            hovertemplate="<b>%{text}</b><br>Run-up: %{x:.1f}%<br>Pullback depth: -%{y:.1f}%<extra></extra>"))
        fig_sc.update_layout(**{**_PA_L, "height": 300,
                                "title": dict(text="Run-up vs Pullback depth", font=dict(size=13), x=0),
                                "xaxis": dict(**_PA_L["xaxis"], title="Run-up before pullback (%)", ticksuffix="%"),
                                "yaxis": dict(**_PA_L["yaxis"], title="Pullback depth (%)", ticksuffix="%")})
        st.plotly_chart(fig_sc, use_container_width=True)
        with st.expander("All pullback events (detail table)", expanded=False):
            disp = events[["rally_start", "peak_date", "run_up_pct", "rally_days",
                            "dd_trigger_date", "dd_at_trigger_pct"]].copy()
            disp.columns = ["Rally start", "Peak date", "Run-up %", "Rally days",
                             "Pullback triggered", "DD at trigger %"]
            disp["Run-up %"] = disp["Run-up %"].map("{:.1f}%".format)
            disp["DD at trigger %"] = disp["DD at trigger %"].map("{:.1f}%".format)
            st.dataframe(disp, hide_index=True, use_container_width=True)
    st.divider()

    # 3. RSI Overbought Duration
    st.markdown(pa_risk_badge(f"3. RSI Overbought Duration (RSI > {ob_level})", r_rsi),
                unsafe_allow_html=True)
    st.markdown("")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=x, y=rsi, name="RSI(14)",
                                  line=dict(color="#9b59b6", width=1.5),
                                  hovertemplate="RSI: %{y:.1f}<extra></extra>"))
    ob_fill = rsi.where(rsi >= ob_level)
    fig_rsi.add_trace(go.Scatter(x=x, y=ob_fill, name=f"Overbought (>{ob_level})",
                                  fill="tozeroy", fillcolor="rgba(231,76,60,0.20)",
                                  line=dict(color="rgba(231,76,60,0.6)", width=0.5),
                                  hoverinfo="skip"))
    fig_rsi.add_hline(y=ob_level, line_color="#e74c3c", line_dash="dash", line_width=1)
    fig_rsi.add_hline(y=30, line_color="#2ecc71", line_dash="dash", line_width=1)
    fig_rsi.update_layout(**{**_PA_L, "height": 220,
                             "title": dict(text="RSI(14) — overbought zones highlighted", font=dict(size=13), x=0),
                             "yaxis": dict(**_PA_L["yaxis"], range=[0, 100])})
    st.plotly_chart(fig_rsi, use_container_width=True)

    _rsi_col = pa_risk_color(r_rsi)
    if cur_ob_streak > 0:
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.05);border-left:4px solid {_rsi_col};"
            f"padding:8px 14px;border-radius:6px;font-size:0.9rem;margin-bottom:8px'>"
            f"⏱ <b>Current RSI overbought streak:</b> {cur_ob_streak} day(s) (RSI = {cur_rsi:.1f})</div>",
            unsafe_allow_html=True)
    else:
        _rsi_txt = f"{cur_rsi:.1f}" if cur_rsi is not None else "—"
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.04);border-left:4px solid #2ecc71;"
            f"padding:8px 14px;border-radius:6px;font-size:0.9rem;margin-bottom:8px'>"
            f"✅ <b>Not currently overbought</b> (RSI = {_rsi_txt})</div>",
            unsafe_allow_html=True)

    streaks = pa_find_ob_streaks(close, rsi, ob_level)
    if streaks.empty:
        st.info(f"No RSI > {ob_level} streaks of ≥ 3 days found.")
    else:
        avg_dur = streaks["duration_days"].mean()
        fwd5 = streaks["fwd_5d_pct"].dropna().mean()
        fwd10 = streaks["fwd_10d_pct"].dropna().mean()
        fwd21 = streaks["fwd_21d_pct"].dropna().mean()
        sa, sb, sc, sd, se = st.columns(5)
        with sa: st.metric("OB streaks found", len(streaks))
        with sb: st.metric("Avg streak length", f"{avg_dur:.1f} days")
        with sc: st.metric("Avg return 5d after", f"{fwd5:+.1f}%" if not np.isnan(fwd5) else "—",
                           delta_color="inverse" if fwd5 < 0 else "normal")
        with sd: st.metric("Avg return 10d after", f"{fwd10:+.1f}%" if not np.isnan(fwd10) else "—",
                           delta_color="inverse" if fwd10 < 0 else "normal")
        with se: st.metric("Avg return 21d after", f"{fwd21:+.1f}%" if not np.isnan(fwd21) else "—",
                           delta_color="inverse" if fwd21 < 0 else "normal")
        long_streaks = streaks[streaks["duration_days"] >= 5]
        short_streaks = streaks[streaks["duration_days"] < 5]
        fig_fwd = go.Figure()
        for grp, col, name in [(short_streaks, "#f39c12", "Short OB streak (<5d)"),
                                (long_streaks, "#e74c3c", "Extended OB streak (≥5d)")]:
            if grp.empty:
                continue
            fig_fwd.add_trace(go.Scatter(
                x=grp["duration_days"], y=grp["fwd_21d_pct"], mode="markers",
                marker=dict(size=9, color=col, opacity=0.8, line=dict(color="white", width=0.5)),
                name=name,
                hovertemplate="<b>%{customdata}</b><br>Streak: %{x} days<br>21d fwd return: %{y:+.1f}%<extra></extra>",
                customdata=[str(d.date()) for d in pd.to_datetime(grp["streak_end"])]))
        fig_fwd.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
        fig_fwd.update_layout(**{**_PA_L, "height": 280,
                                 "title": dict(text="OB streak duration vs 21-day forward return", font=dict(size=13), x=0),
                                 "xaxis": dict(**_PA_L["xaxis"], title="Streak length (days)", ticksuffix="d"),
                                 "yaxis": dict(**_PA_L["yaxis"], title="21d fwd return (%)", ticksuffix="%"),
                                 "legend": dict(orientation="h", y=1.12, x=0)})
        st.plotly_chart(fig_fwd, use_container_width=True)
        with st.expander("OB streak detail table", expanded=False):
            disp_s = streaks.copy()
            for c in ["fwd_5d_pct", "fwd_10d_pct", "fwd_21d_pct"]:
                disp_s[c] = disp_s[c].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
            disp_s.columns = ["Start", "End", "Days", "Peak RSI", "Fwd 5d %", "Fwd 10d %", "Fwd 21d %"]
            st.dataframe(disp_s, hide_index=True, use_container_width=True)

    st.divider()
    st.caption("**Methodology:** Bias ratio = (Price − SMAₙ) / SMAₙ. Run-up cycles: price rises from prior trough until ≥X% drawdown. RSI streaks: contiguous days with RSI ≥ threshold (min 3 days). For research only — not investment advice.")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
st.title("Technical Analyser")

tool = st.sidebar.radio(
    "Select tool",
    ["Markowitz Optimization", "Strategy Backtester", "Pullback Analyzer"],
    key="ta_tool",
)

st.sidebar.divider()

if tool == "Markowitz Optimization":
    render_markowitz()
elif tool == "Strategy Backtester":
    render_backtester()
else:
    render_pullback()
