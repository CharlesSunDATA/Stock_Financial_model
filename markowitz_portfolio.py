"""
Markowitz efficient frontier & portfolio optimization (Streamlit page module).
Do not call st.set_page_config here — the entrypoint app.py sets it.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import Bounds, LinearConstraint, minimize

warnings.filterwarnings("ignore", category=FutureWarning)

TRADING_DAYS = 252
N_MONTE_CARLO = 10_000

# Diversification: avoid corner solutions (100% in one asset). Long-only, per-asset cap.
WEIGHT_MIN = 0.05
WEIGHT_MAX = 0.2  # 20% max per name → need at least ceil(1/0.2) = 5 names to sum to 100%
MIN_ASSETS_FOR_WEIGHT_CAP = 5
MAX_ASSETS_FOR_WEIGHT_FLOOR = 20  # floor 5% → at most floor(1/0.05)=20 names can fit

# Shown in UI so you can confirm Streamlit Cloud deployed the capped-optimizer code.
OPTIMIZER_BUILD_ID = "min5-max20-per-asset-bounds-2026-04"


def _weight_bounds(n: int) -> tuple[tuple[float, float], ...]:
    """Box constraints for scipy.optimize: each weight in [WEIGHT_MIN, WEIGHT_MAX]."""
    return tuple((WEIGHT_MIN, WEIGHT_MAX) for _ in range(n))


def _feasible_weight_cap(n: int) -> bool:
    """Feasible iff we can satisfy sum(w)=1 with WEIGHT_MIN ≤ w_i ≤ WEIGHT_MAX."""
    return (
        n >= MIN_ASSETS_FOR_WEIGHT_CAP
        and n <= MAX_ASSETS_FOR_WEIGHT_FLOOR
        and n * WEIGHT_MAX >= 1.0 - 1e-12
        and n * WEIGHT_MIN <= 1.0 + 1e-12
    )


def _budget_constraint(n: int) -> LinearConstraint:
    """Linear equality: 1'w = 1."""
    return LinearConstraint(np.ones((1, n)), 1.0, 1.0)


def _feasible_start(n: int) -> np.ndarray:
    # Start from the minimum allocation and distribute remaining budget evenly.
    w = np.full(n, WEIGHT_MIN, dtype=float)
    remaining = 1.0 - float(n) * WEIGHT_MIN
    if remaining <= 0:
        return np.full(n, 1.0 / n, dtype=float)
    w += remaining / n
    # With feasibility constraints above (n >= 5), this equalized start should be ≤ WEIGHT_MAX,
    # but keep a safe clip + renormalize guard for numerical stability.
    w = np.clip(w, WEIGHT_MIN, WEIGHT_MAX)
    return w / w.sum()


def _parse_tickers(raw: list[str]) -> list[str]:
    out: list[str] = []
    for s in raw:
        t = (s or "").strip().upper()
        if t and t not in out:
            out.append(t)
    return out


def fetch_adj_close(
    tickers: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    if not tickers:
        raise ValueError("At least one ticker is required.")
    data = yf.download(
        tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data.empty:
        raise ValueError("No price data returned. Check tickers and date range.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            prices = data.xs("Close", axis=1, level=0, drop_level=True)
    else:
        prices = data[["Close"]].copy() if "Close" in data.columns else data.copy()
        if len(tickers) == 1:
            prices.columns = [tickers[0]]

    prices = prices.dropna(how="all").ffill().dropna()
    if prices.empty or len(prices) < 30:
        raise ValueError("Insufficient overlapping history after cleaning. Try a wider date range.")
    return prices.astype(float)


def ann_stats(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu_daily = returns.mean().values
    cov_daily = returns.cov().values
    mu_ann = mu_daily * TRADING_DAYS
    cov_ann = cov_daily * TRADING_DAYS
    return mu_ann, cov_ann


def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(np.dot(w, mu))


def portfolio_volatility(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(w @ cov @ w, 0.0)))


def neg_sharpe(
    w: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
) -> float:
    vol = portfolio_volatility(w, cov)
    if vol < 1e-12:
        return 1e12
    return -(portfolio_return(w, mu) - rf) / vol


def _weights_feasible(w: np.ndarray) -> bool:
    s = float(w.sum())
    if abs(s - 1.0) > 5e-4:
        return False
    if float(w.max()) > WEIGHT_MAX + 1e-5 or float(w.min()) < WEIGHT_MIN - 1e-5:
        return False
    return True


def optimize_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    """
    Maximize Sharpe ratio subject to ∑w=1 and WEIGHT_MIN ≤ w_i ≤ WEIGHT_MAX (default 0–20%).
    Uses trust-constr + explicit Bounds; SLSQP fallback if needed.
    """
    n = len(mu)
    x0 = _feasible_start(n)
    bounds = Bounds(WEIGHT_MIN, WEIGHT_MAX, keep_feasible=True)
    lc = _budget_constraint(n)
    res = minimize(
        neg_sharpe,
        x0,
        args=(mu, cov, rf),
        method="trust-constr",
        bounds=bounds,
        constraints=[lc],
        options={"maxiter": 3000, "gtol": 1e-8},
    )
    w = np.asarray(res.x, dtype=float)
    if not res.success:
        st.warning(f"Max-Sharpe (trust-constr): {res.message}")
    if not _weights_feasible(w):
        res2 = minimize(
            neg_sharpe,
            _feasible_start(n),
            args=(mu, cov, rf),
            method="SLSQP",
            bounds=_weight_bounds(n),
            constraints={"type": "eq", "fun": lambda ww: float(np.sum(ww)) - 1.0},
            options={"maxiter": 3000, "ftol": 1e-11},
        )
        w = np.asarray(res2.x, dtype=float)
        if not res2.success:
            st.warning(f"Max-Sharpe (SLSQP fallback): {res2.message}")
    return np.clip(w, WEIGHT_MIN, WEIGHT_MAX)


def optimize_min_volatility(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    x0 = _feasible_start(n)
    bounds = Bounds(WEIGHT_MIN, WEIGHT_MAX, keep_feasible=True)
    lc = _budget_constraint(n)

    def obj(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    res = minimize(
        obj,
        x0,
        method="trust-constr",
        bounds=bounds,
        constraints=[lc],
        options={"maxiter": 3000, "gtol": 1e-8},
    )
    w = np.asarray(res.x, dtype=float)
    if not res.success:
        st.warning(f"Min-vol (trust-constr): {res.message}")
    if not _weights_feasible(w):
        cons = {"type": "eq", "fun": lambda ww: float(np.sum(ww)) - 1.0}
        res2 = minimize(
            obj,
            _feasible_start(n),
            method="SLSQP",
            bounds=_weight_bounds(n),
            constraints=cons,
            options={"maxiter": 3000, "ftol": 1e-11},
        )
        w = np.asarray(res2.x, dtype=float)
        if not res2.success:
            st.warning(f"Min-vol (SLSQP fallback): {res2.message}")
    return np.clip(w, WEIGHT_MIN, WEIGHT_MAX)


def _sample_capped_simplex_weights(
    n: int,
    rng: np.random.Generator,
    n_sims: int,
) -> np.ndarray:
    """
    Random portfolios on {w : sum w = 1, WEIGHT_MIN <= w_i <= WEIGHT_MAX}.
    Rejection sampling on Dirichlet(α); α is raised if acceptance is too low.
    When 1 - n×WEIGHT_MIN = 0, the feasible set is a single point (all weights = WEIGHT_MIN).
    """
    if not _feasible_weight_cap(n):
        raise ValueError("Infeasible weight bounds for this n.")
    remaining_mass = 1.0 - float(n) * WEIGHT_MIN
    if abs(remaining_mass) < 1e-12:
        return np.tile(np.full(n, WEIGHT_MIN, dtype=float), (n_sims, 1))

    # Sample u on the simplex (sum u = 1, u_i >= 0) with an upper cap so that
    # w_i = WEIGHT_MIN + remaining_mass * u_i ≤ WEIGHT_MAX.
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


def monte_carlo_portfolios(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
    n_sims: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(mu)
    W = _sample_capped_simplex_weights(n, rng, n_sims)
    rows: list[dict] = []
    for i in range(n_sims):
        w = W[i]
        ret = portfolio_return(w, mu)
        vol = portfolio_volatility(w, cov)
        sharpe = (ret - rf) / vol if vol > 1e-12 else np.nan
        rows.append(
            {
                "sim_id": i,
                "ann_return": ret,
                "ann_volatility": vol,
                "sharpe_ratio": sharpe,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.title("Markowitz efficient frontier & portfolio optimization")
    st.caption(
        f"**Build `{OPTIMIZER_BUILD_ID}`** — Mean–variance with Monte Carlo + SciPy **trust-constr** "
        f"(explicit per-asset **Bounds**). Max-Sharpe and min-vol: **{WEIGHT_MIN:.0%}–{WEIGHT_MAX:.0%}** per name, "
        f"all names get weight (no 0%). Requires **{MIN_ASSETS_FOR_WEIGHT_CAP}–{MAX_ASSETS_FOR_WEIGHT_FLOOR}** tickers with data. "
        "Education/research only — not investment advice."
    )

    default_end = date.today()
    default_start = default_end - timedelta(days=365 * 5)
    defaults = ["NVDA", "AAPL", "MSFT", "TLT", "GLD", "", "", "", "", ""]

    with st.sidebar:
        st.header("Inputs")
        tickers_in: list[str] = []
        for i in range(10):
            lab = f"Ticker {i + 1}"
            v = st.text_input(lab, value=defaults[i], key=f"mk_t{i}")
            tickers_in.append(v)

        c1, c2 = st.columns(2)
        with c1:
            start_d = st.date_input("Start date", value=default_start, key="mk_start")
        with c2:
            end_d = st.date_input("End date", value=default_end, key="mk_end")

        rf = st.number_input(
            "Risk-free rate (annual, decimal)",
            min_value=0.0,
            max_value=0.5,
            value=0.02,
            step=0.005,
            format="%.4f",
            help="e.g. 0.02 = 2%",
            key="mk_rf",
        )

        run = st.button("Run optimization", type="primary", width="stretch", key="mk_run")

    tickers = _parse_tickers(tickers_in)
    if not tickers:
        st.info("Enter at least one ticker in the sidebar.")
        return

    if start_d >= end_d:
        st.error("Start date must be before end date.")
        return

    if not run:
        st.info("Set tickers and dates, then click **Run optimization**.")
        return

    try:
        with st.spinner("Downloading prices and building returns…"):
            prices = fetch_adj_close(tickers, start_d, end_d)
            cols = [c for c in tickers if c in prices.columns]
            if len(cols) < len(tickers):
                missing = set(tickers) - set(cols)
                st.warning(f"No data for: {', '.join(sorted(missing))}. Using: {', '.join(cols)}")
            if len(cols) < 2:
                st.error("Need at least two assets with data to build a covariance matrix.")
                return
            if not _feasible_weight_cap(len(cols)):
                st.error(
                    f"With per-asset bounds **{WEIGHT_MIN:.0%}–{WEIGHT_MAX:.0%}**, you need **{MIN_ASSETS_FOR_WEIGHT_CAP}–"
                    f"{MAX_ASSETS_FOR_WEIGHT_FLOOR}** tickers with data so the portfolio can sum to 100%. "
                    "Adjust the symbols in the sidebar."
                )
                return
            prices = prices[cols]
            daily_ret = prices.pct_change().dropna()
            mu_ann, cov_ann = ann_stats(daily_ret)
            corr = daily_ret.corr()

        rng = np.random.default_rng(42)
        with st.spinner(f"Monte Carlo: {N_MONTE_CARLO:,} portfolios…"):
            mc = monte_carlo_portfolios(mu_ann, cov_ann, rf, N_MONTE_CARLO, rng)

        with st.spinner("SciPy: max Sharpe & min volatility…"):
            w_max_sharpe = optimize_max_sharpe(mu_ann, cov_ann, rf)
            w_min_vol = optimize_min_volatility(cov_ann)

        r_ms = portfolio_return(w_max_sharpe, mu_ann)
        v_ms = portfolio_volatility(w_max_sharpe, cov_ann)
        sr_ms = (r_ms - rf) / v_ms if v_ms > 1e-12 else np.nan

        r_mv = portfolio_return(w_min_vol, mu_ann)
        v_mv = portfolio_volatility(w_min_vol, cov_ann)
        sr_mv = (r_mv - rf) / v_mv if v_mv > 1e-12 else np.nan

    except Exception as e:
        st.error(str(e))
        return

    st.subheader("1. Efficient frontier (Monte Carlo + optima)")
    fig = px.scatter(
        mc,
        x="ann_volatility",
        y="ann_return",
        color="sharpe_ratio",
        color_continuous_scale="Viridis",
        opacity=0.35,
        labels={
            "ann_volatility": "Annualized volatility (σ)",
            "ann_return": "Annualized expected return (μ)",
            "sharpe_ratio": "Sharpe ratio",
        },
        title=f"{N_MONTE_CARLO:,} random long-only portfolios, each weight ∈ [{WEIGHT_MIN:.0%}, {WEIGHT_MAX:.0%}] (color = Sharpe)",
        width=None,
        height=520,
    )
    fig.add_trace(
        go.Scatter(
            x=[v_ms],
            y=[r_ms],
            mode="markers",
            marker=dict(symbol="star", size=22, color="gold", line=dict(width=1, color="white")),
            name="Max Sharpe (SciPy)",
            hovertemplate="Max Sharpe<br>σ=%{x:.4f}<br>μ=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[v_mv],
            y=[r_mv],
            mode="markers",
            marker=dict(symbol="star", size=22, color="cyan", line=dict(width=1, color="white")),
            name="Min volatility (SciPy)",
            hovertemplate="Min vol<br>σ=%{x:.4f}<br>μ=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig, width="stretch")

    st.subheader("2. Optimal weights (pie charts + tables)")
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Max-Sharpe portfolio** " f"(Sharpe ≈ {sr_ms:.3f})")
        df_ms = pd.DataFrame({"Asset": cols, "Weight %": w_max_sharpe * 100.0}).sort_values(
            "Weight %", ascending=False
        )
        fig_p1 = px.pie(
            df_ms,
            names="Asset",
            values="Weight %",
            hole=0.35,
            title="Weights — max Sharpe",
        )
        fig_p1.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_p1, width="stretch")
        df_ms["Weight %"] = df_ms["Weight %"].round(2)
        st.dataframe(df_ms, width="stretch", hide_index=True)

    with c_right:
        st.markdown("**Min-volatility portfolio** " f"(Sharpe ≈ {sr_mv:.3f})")
        df_mv = pd.DataFrame({"Asset": cols, "Weight %": w_min_vol * 100.0}).sort_values(
            "Weight %", ascending=False
        )
        fig_p2 = px.pie(
            df_mv,
            names="Asset",
            values="Weight %",
            hole=0.35,
            title="Weights — min volatility",
        )
        fig_p2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_p2, width="stretch")
        df_mv["Weight %"] = df_mv["Weight %"].round(2)
        st.dataframe(df_mv, width="stretch", hide_index=True)

    st.subheader("3. Historical correlation heatmap (daily returns)")
    fig_h = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation matrix of daily returns",
    )
    fig_h.update_layout(height=480)
    st.plotly_chart(fig_h, width="stretch")

    with st.expander("Methodology"):
        st.markdown(
            f"""
- **Returns:** simple daily returns; annualized mean `× 252`, covariance `× 252`.
- **Long-only with bounds:** each `w_i ∈ [{WEIGHT_MIN}, {WEIGHT_MAX}]`, `∑w = 1`. This forces diversification: every name has at least **{WEIGHT_MIN:.0%}**, and no name exceeds **{WEIGHT_MAX:.0%}**.
- **Max Sharpe:** maximize `(μ'w − rf) / √(w'Σw)` via `scipy.optimize.minimize` (**trust-constr** + `Bounds` + `LinearConstraint`; SLSQP as fallback).
- **Min variance:** minimize `w'Σw` under the same bounds and budget constraint (same solvers).
- **Monte Carlo:** Dirichlet rejection sampling on the **same** per-asset cap (matches SciPy feasible set; when `n × cap = 1` the cloud collapses to the unique equal-weight portfolio).
            """
        )


main()

