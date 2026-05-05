# Stock Financial Model

A Streamlit multipage dashboard for equity research, portfolio analysis, and quantitative screening — powered by [FMP](https://financialmodelingprep.com/) (primary data) and yfinance (fallback).

## Pages

| # | Page | Description |
|---|------|-------------|
| 0 | Dashboard | Market overview and homepage |
| 1 | Stock Valuation | Two-stage DCF + trailing P/E zone vs historical range |
| 2 | Quarterly Financials | Margins, growth, and leverage from yfinance |
| 3 | Markowitz Optimization | Max Sharpe + min volatility with per-asset weight cap (0–20%) |
| 4 | Technical Backtester | Signal-based strategy backtesting |
| 5 | Earnings Call NLP | Earnings call sentiment analysis (VADER) |
| 6 | Earnings Calendar | Upcoming earnings from local DB |
| 7 | Pullback Analyzer | Bias ratio, run-up stats, RSI overbought duration |
| 8 | Analyst Sentiment | Analyst ratings and price targets |
| 9 | Market Trend | Breadth, moving averages, new highs/lows |
| 10 | Momentum Report | 1/3/6/12-month composite momentum rankings |
| 11 | Fundamentals | Fundamental data viewer from local DB |
| 12 | Opportunity Score | Weighted multi-factor opportunity ranking |
| 13 | Risk Score | Investment risk ranking |
| 14 | Smart Money | Institutional and insider activity |
| 15 | Portfolio Watchlist | Personal watchlist with live metrics |

## Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialize local SQLite database
python3 scripts/init_db.py

# 4. Run the app
./run.sh
```

## Data Backfill (FMP)

Set your FMP API key first (any one of these):

```bash
export FMP_API_KEY=your_key_here
# or add to .streamlit/secrets.toml: FMP_API_KEY = "your_key_here"
```

Then run the relevant backfill scripts:

```bash
python3 scripts/update_fmp.py AAPL MSFT NVDA          # fundamentals, ratios, estimates
python3 scripts/update_fundamentals_fmp.py AAPL        # income / cash flow / balance sheet
python3 scripts/update_prices_eod.py AAPL              # EOD price history
python3 scripts/update_valuation_analyst_fmp.py AAPL   # DCF, EV, price targets
python3 scripts/update_calendars_fmp.py                # earnings + economic calendar
python3 scripts/update_news_fmp.py AAPL                # stock news
python3 scripts/update_compliance_fmp.py AAPL          # executives, insider trades
```

## Architecture

```
app.py                  # entry point — registers all pages
pages/                  # one file per page (N_Name.py)
  modules/              # shared utilities (fmp_key, analyst_sentiment)
utils/                  # data helpers (local_data, opportunity_score, risk_score, …)
scripts/                # one-off backfill and DB init scripts
data/                   # local SQLite DB (quant_data.db, excluded from git)
.streamlit/             # Streamlit config (secrets.toml excluded from git)
```

Data read priority: **local SQLite** → **yfinance (live)** → FMP API (backfill scripts only).

## Deploy to Streamlit Community Cloud

1. Fork or push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo
3. Set **Main file path** to `app.py`
4. Add `FMP_API_KEY` in the app's **Secrets** settings
