# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the app:**
```bash
./run.sh
# or explicitly:
.venv/bin/python -m streamlit run app.py
```

**Setup (first time):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/init_db.py
```

**Initialize / migrate the local SQLite database:**
```bash
python3 scripts/init_db.py
```

**Backfill data from FMP (Financial Modeling Prep):**
```bash
python3 scripts/update_fmp.py AAPL MSFT NVDA          # fundamentals, ratios, estimates
python3 scripts/update_fundamentals_fmp.py AAPL        # income/CF/BS statements
python3 scripts/update_prices_eod.py AAPL              # EOD price history
python3 scripts/update_valuation_analyst_fmp.py AAPL   # DCF, EV, price targets
python3 scripts/update_calendars_fmp.py                # earnings + economic calendar
python3 scripts/update_news_fmp.py AAPL                # stock news
python3 scripts/update_compliance_fmp.py AAPL          # executives, insider trades
```

**FMP API key:** Set `FMP_API_KEY` as an env var, in `.streamlit/secrets.toml`, or in Streamlit Cloud secrets. Resolution order is: env var → `st.secrets` → direct TOML parse (see `pages/modules/fmp_key.py`).

## Architecture

This is a **Streamlit multipage app** (`app.py` registers all pages; each `pages/N_Name.py` is a standalone page). There is no routing framework — Streamlit handles navigation.

### Data layers (read order of preference)

1. **Local SQLite** (`data/quant_data.db`) — primary store for all FMP-sourced data (prices, fundamentals, statements, analyst data, calendars). Pages should read from here first via `utils/local_data.py`.
2. **yfinance (live)** — fallback for quarterly financials when local data is absent. `utils/data_loader.py` handles label normalization across yfinance API versions.
3. **FMP API (live)** — only used during backfill scripts, never directly in pages.

### Key modules

- **`utils/local_data.py`** — all read helpers for `quant_data.db`: `load_price_history()`, `load_ohlcv()`, `load_adjusted_close()`, `latest_quotes()`, `latest_company_snapshot()`. Always opens in read-only mode (`?mode=ro&immutable=1`).
- **`utils/data_loader.py`** — `fetch_quarterly_metrics()` pulls from yfinance with multi-name label resolution (`_pick_row()`) to handle Yahoo's inconsistent row labels across tickers/versions.
- **`utils/opportunity_score.py`** — weighted factor scoring model (momentum 25%, revenue growth 20%, EPS/FCF 20%, valuation 15%, safety 10%, industry 10%). All labels must stay in English (see AGENTS.md).
- **`utils/risk_score.py`** — companion risk ranking model.
- **`pages/modules/fmp_key.py`** — single source of truth for FMP API key resolution.
- **`pages/modules/analyst_sentiment.py`** — analyst sentiment utilities shared across pages.

### Database schema (SQLite, `data/quant_data.db`)

All tables use `UNIQUE(ticker, date_col)` constraints for safe upserts. Key tables:
- `prices_eod` — daily OHLCV + adj_close
- `fundamental_data`, `balance_sheet`, `financial_ratios` — legacy FMP fundamentals
- `income_statement`, `cash_flow_statement` — normalized statement rows + `payload_json` for full data
- `key_metrics_ttm`, `historical_key_metrics`, `historical_financial_ratios` — TTM and historical metrics
- `analyst_estimates`, `analyst_estimates_detail`, `price_target_consensus` — analyst data
- `earnings_calendar`, `economic_calendar` — upcoming events
- `fmp_universe`, `fmp_watchlist` — ticker universe management
- `fmp_bulk_backfill`, `fmp_daily_backfill`, `prices_backfill_progress` — incremental backfill tracking

Schema migrations use `_add_column_if_missing()` in `scripts/init_db.py` — re-run `init_db.py` after schema changes.

### Pages overview

| Page | What it does |
|------|-------------|
| `0_Dashboard.py` | Market overview / homepage |
| `1_Stock_valuation.py` | Two-stage DCF + trailing P/E zone vs historical range |
| `2_Quarterly_financials.py` | yfinance quarterly pull with margins/growth/leverage |
| `3_Markowitz_opt.py` | Max Sharpe + min volatility with per-asset weight cap (0–20%) |
| `4_Technical_backtester.py` | Technical strategy backtesting |
| `5_Earnings_Call_NLP.py` | Earnings call sentiment (VADER NLP) |
| `6_Earnings_Calendar.py` | Upcoming earnings from local DB |
| `7_Pullback_Analyzer.py` | Bias ratio, run-up stats, RSI overbought duration |
| `8_Analyst_Sentiment.py` | Analyst ratings + price targets |
| `9_Market_Trend.py` | Breadth, moving averages, new highs/lows |
| `10_Momentum_Report.py` | 1/3/6/12-month composite momentum rankings |
| `11_Fundamentals.py` | Fundamental data viewer from local DB |
| `12_Opportunity_Score.py` | Weighted multi-factor opportunity ranking |
| `13_Risk_Score.py` | Investment risk ranking |

### Conventions

- All UI labels, table headers, chart labels, comments, and generated output must be in **English** (see `AGENTS.md`).
- Pages use `st.set_page_config` only in `app.py` (not in individual pages).
- The app applies full-width layout globally via CSS in `app.py`; pages should not override this.
- `legacy/` contains old single-file versions of pages; do not modify them.
