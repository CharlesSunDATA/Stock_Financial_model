"""
init_db.py

Initialize the local SQLite database for quantitative analysis.

What this script does
- Connects to a local SQLite database named `quant_data.db` (stored under `data/` by default).
- Creates required tables using CREATE TABLE IF NOT EXISTS:
  - fundamental_data
  - balance_sheet
  - financial_ratios
  - analyst_estimates
  - analyst_estimates_detail (v3 analyst-estimates; full rows + period)
    --fetch-estimates also mirrors mapped columns into analyst_estimates (legacy UNIQUE ticker+date)
  - suppliers, institutional_ownership, insider_trading
  - historical_financial_ratios, historical_key_metrics
  - company_profile

Notes
- All dates are stored as ISO strings (YYYY-MM-DD) for SQLite compatibility.
- Each time-series table has a UNIQUE(ticker, report_date) (or UNIQUE(ticker, estimated_date)) constraint.

Run
  python3 scripts/init_db.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def project_root() -> Path:
    # scripts/init_db.py -> project root is parent of scripts/
    return Path(__file__).resolve().parents[1]

def default_db_path() -> Path:
    return project_root() / "data" / "quant_data.db"

def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols: set[str] = set()
    for row in cur.fetchall():
        # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
        if len(row) >= 2 and row[1]:
            cols.add(str(row[1]))
    return cols


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    cols = _table_columns(conn, table)
    if column in cols:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def init_db(db_path: Path | None = None) -> Path:
    if db_path is None:
        db_path = default_db_path()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fundamental_data (
              ticker TEXT NOT NULL,
              report_date TEXT NOT NULL,
              revenue REAL,
              eps REAL,
              free_cash_flow REAL,
              UNIQUE(ticker, report_date)
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS balance_sheet (
              ticker TEXT NOT NULL,
              report_date TEXT NOT NULL,
              cash_and_equivalents REAL,
              total_assets REAL,
              total_liabilities REAL,
              total_debt REAL,
              outstanding_shares REAL,
              UNIQUE(ticker, report_date)
            );
            """
        )

        # Migration: add columns if the DB already exists.
        _add_column_if_missing(conn, "balance_sheet", "total_debt", "REAL")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS financial_ratios (
              ticker TEXT NOT NULL,
              report_date TEXT NOT NULL,
              roe REAL,
              roic REAL,
              pe_ratio REAL,
              pb_ratio REAL,
              ev_to_ebitda REAL,
              UNIQUE(ticker, report_date)
            );
            """
        )

        # Latest TTM key metrics (bulk). Store as-of date + a few commonly used fields,
        # plus a JSON payload for forward compatibility.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS key_metrics_ttm (
              ticker TEXT NOT NULL,
              as_of_date TEXT NOT NULL,
              market_cap REAL,
              enterprise_value REAL,
              ev_to_sales REAL,
              ev_to_operating_cash_flow REAL,
              ev_to_free_cash_flow REAL,
              net_debt_to_ebitda REAL,
              payload_json TEXT,
              UNIQUE(ticker, as_of_date)
            );
            """
        )

        # EOD prices (bulk by date). Designed for daily backfills.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prices_eod (
              ticker TEXT NOT NULL,
              price_date TEXT NOT NULL,
              open REAL,
              high REAL,
              low REAL,
              close REAL,
              adj_close REAL,
              volume REAL,
              UNIQUE(ticker, price_date)
            );
            """
        )

        # Per-ticker incremental backfill progress for prices (per-symbol endpoints).
        # Tracks the last successfully backfilled trading day for each ticker.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prices_backfill_progress (
              ticker TEXT PRIMARY KEY,
              last_price_date TEXT,
              updated_at TEXT NOT NULL
            );
            """
        )

        # Persisted universe table to support efficient batch selection
        # without scanning Python lists or doing per-ticker lookups.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fmp_universe (
              ticker TEXT PRIMARY KEY
            );
            """
        )

        # Optional curated universe lists (e.g., SP500 + Nasdaq + semis ETF holdings).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fmp_watchlist (
              watchlist_name TEXT NOT NULL,
              ticker TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (watchlist_name, ticker)
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyst_estimates (
              ticker TEXT NOT NULL,
              estimated_date TEXT NOT NULL,
              estimated_revenue REAL,
              estimated_eps REAL,
              UNIQUE(ticker, estimated_date)
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyst_estimates_detail (
              ticker TEXT NOT NULL,
              estimate_date TEXT NOT NULL,
              period_name TEXT NOT NULL DEFAULT '',
              estimated_revenue_avg REAL,
              estimated_eps_avg REAL,
              payload_json TEXT,
              updated_at TEXT NOT NULL,
              UNIQUE(ticker, estimate_date, period_name)
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS suppliers (
              ticker TEXT NOT NULL,
              supplier_symbol TEXT NOT NULL,
              supplier_name TEXT,
              relationship TEXT,
              payload_json TEXT,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (ticker, supplier_symbol)
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS institutional_ownership (
              ticker TEXT NOT NULL,
              holder_name TEXT NOT NULL,
              as_of_date TEXT NOT NULL DEFAULT '',
              ownership_percent REAL,
              shares REAL,
              payload_json TEXT,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (ticker, holder_name, as_of_date)
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS insider_trading (
              ticker TEXT NOT NULL,
              row_key TEXT NOT NULL,
              filing_date TEXT,
              insider_name TEXT,
              transaction_type TEXT,
              shares REAL,
              price REAL,
              payload_json TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (ticker, row_key)
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_financial_ratios (
              ticker TEXT NOT NULL,
              period TEXT NOT NULL,
              report_date TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(ticker, period, report_date)
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_key_metrics (
              ticker TEXT NOT NULL,
              period TEXT NOT NULL,
              report_date TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(ticker, period, report_date)
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS company_profile (
              ticker TEXT PRIMARY KEY,
              company_name TEXT,
              sector TEXT,
              industry TEXT,
              full_time_employees INTEGER,
              image_url TEXT
            );
            """
        )

        # Backfill progress tracker for bulk jobs (one row per dataset/quarter).
        # This allows incremental runs without re-fetching already completed quarters.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fmp_bulk_backfill (
              dataset TEXT NOT NULL,
              year INTEGER NOT NULL,
              period TEXT NOT NULL,
              completed_at TEXT NOT NULL,
              rows_received INTEGER,
              PRIMARY KEY (dataset, year, period)
            );
            """
        )

        # Daily FMP quota tracker (calls/day). One row per date.
        # Not required by schema, but useful for enforcing daily API limits.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fmp_api_usage (
              usage_date TEXT PRIMARY KEY,
              calls_used INTEGER NOT NULL
            );
            """
        )

        # Daily backfill tracker for date-based bulk datasets (e.g., eod-bulk).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fmp_daily_backfill (
              dataset TEXT NOT NULL,
              as_of_date TEXT NOT NULL,
              completed_at TEXT NOT NULL,
              rows_received INTEGER,
              PRIMARY KEY (dataset, as_of_date)
            );
            """
        )

        # Helpful indexes for typical queries
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_fundamental_ticker_date
            ON fundamental_data(ticker, report_date);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_balance_sheet_ticker_date
            ON balance_sheet(ticker, report_date);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ratios_ticker_date
            ON financial_ratios(ticker, report_date);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_key_metrics_ticker_date
            ON key_metrics_ttm(ticker, as_of_date);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
            ON prices_eod(ticker, price_date);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_prices_eod_ticker_date
            ON prices_eod(ticker, price_date);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_prices_progress_last_date
            ON prices_backfill_progress(last_price_date);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_universe_ticker
            ON fmp_universe(ticker);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_watchlist_name_ticker
            ON fmp_watchlist(watchlist_name, ticker);
            """
        )

        # Sequential cursor for long-running per-batch ingestion (optional mode).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fmp_prices_cursor (
              cursor_name TEXT PRIMARY KEY,
              last_ticker TEXT,
              updated_at TEXT NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_estimates_ticker_date
            ON analyst_estimates(ticker, estimated_date);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_analyst_estimates_detail_ticker
            ON analyst_estimates_detail(ticker, estimate_date);
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_suppliers_ticker
            ON suppliers(ticker);
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_inst_own_ticker
            ON institutional_ownership(ticker);
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_insider_ticker
            ON insider_trading(ticker, filing_date);
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_hist_ratios_ticker
            ON historical_financial_ratios(ticker, report_date);
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_hist_km_ticker
            ON historical_key_metrics(ticker, report_date);
            """
        )

        # ── Income Statement ────────────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS income_statement (
              ticker TEXT NOT NULL,
              report_date TEXT NOT NULL,
              period TEXT NOT NULL DEFAULT '',
              revenue REAL,
              gross_profit REAL,
              operating_income REAL,
              ebitda REAL,
              net_income REAL,
              eps REAL,
              eps_diluted REAL,
              weighted_avg_shares_out REAL,
              payload_json TEXT,
              updated_at TEXT NOT NULL,
              UNIQUE(ticker, report_date, period)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_income_stmt_ticker_date
            ON income_statement(ticker, report_date);
            """
        )

        # ── Cash Flow Statement ─────────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cash_flow_statement (
              ticker TEXT NOT NULL,
              report_date TEXT NOT NULL,
              period TEXT NOT NULL DEFAULT '',
              operating_cash_flow REAL,
              capital_expenditure REAL,
              free_cash_flow REAL,
              investing_cash_flow REAL,
              financing_cash_flow REAL,
              net_change_in_cash REAL,
              payload_json TEXT,
              updated_at TEXT NOT NULL,
              UNIQUE(ticker, report_date, period)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cash_flow_ticker_date
            ON cash_flow_statement(ticker, report_date);
            """
        )

        # ── Growth tables (JSON payload; too many columns to enumerate) ─────
        for _tbl, _idx in (
            ("income_statement_growth",    "idx_is_growth_ticker_date"),
            ("cash_flow_statement_growth", "idx_cf_growth_ticker_date"),
            ("balance_sheet_growth",       "idx_bs_growth_ticker_date"),
        ):
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_tbl} (
                  ticker TEXT NOT NULL,
                  report_date TEXT NOT NULL,
                  period TEXT NOT NULL DEFAULT '',
                  payload_json TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  UNIQUE(ticker, report_date, period)
                );
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {_idx}
                ON {_tbl}(ticker, report_date);
                """
            )

        # ── As-reported (SEC raw filing) ────────────────────────────────────
        for _tbl, _idx in (
            ("income_statement_as_reported",    "idx_is_reported_ticker_date"),
            ("cash_flow_statement_as_reported", "idx_cf_reported_ticker_date"),
            ("balance_sheet_as_reported",       "idx_bs_reported_ticker_date"),
        ):
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_tbl} (
                  ticker TEXT NOT NULL,
                  report_date TEXT NOT NULL,
                  period TEXT NOT NULL DEFAULT '',
                  payload_json TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  UNIQUE(ticker, report_date, period)
                );
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {_idx}
                ON {_tbl}(ticker, report_date);
                """
            )

        # ── Valuation tables ────────────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS enterprise_values (
              ticker TEXT NOT NULL,
              report_date TEXT NOT NULL,
              stock_price REAL,
              number_of_shares REAL,
              market_cap REAL,
              minus_cash REAL,
              add_total_debt REAL,
              enterprise_value REAL,
              payload_json TEXT,
              updated_at TEXT NOT NULL,
              UNIQUE(ticker, report_date)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ev_ticker_date
            ON enterprise_values(ticker, report_date);
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_market_cap (
              ticker TEXT NOT NULL,
              price_date TEXT NOT NULL,
              market_cap REAL,
              updated_at TEXT NOT NULL,
              UNIQUE(ticker, price_date)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_hist_mktcap_ticker_date
            ON historical_market_cap(ticker, price_date);
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dcf (
              ticker TEXT NOT NULL,
              as_of_date TEXT NOT NULL,
              dcf_value REAL,
              stock_price REAL,
              payload_json TEXT,
              updated_at TEXT NOT NULL,
              UNIQUE(ticker, as_of_date)
            );
            """
        )

        # ── Analyst / sentiment tables ───────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS price_target_consensus (
              ticker TEXT PRIMARY KEY,
              target_high REAL,
              target_low REAL,
              target_consensus REAL,
              target_median REAL,
              payload_json TEXT,
              updated_at TEXT NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS price_target_summary (
              ticker TEXT PRIMARY KEY,
              last_month REAL,
              last_quarter REAL,
              last_year REAL,
              all_time REAL,
              publishers TEXT,
              payload_json TEXT,
              updated_at TEXT NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_peers (
              ticker TEXT NOT NULL,
              peer_ticker TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (ticker, peer_ticker)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stock_peers_ticker
            ON stock_peers(ticker);
            """
        )

        # ── Calendar & Event tables ──────────────────────────────────────────

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS earnings_calendar (
              ticker TEXT NOT NULL,
              event_date TEXT NOT NULL,
              eps REAL,
              eps_estimated REAL,
              time TEXT,
              revenue REAL,
              revenue_estimated REAL,
              fiscal_date_ending TEXT,
              payload_json TEXT,
              updated_at TEXT NOT NULL,
              UNIQUE(ticker, event_date)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_earnings_cal_ticker_date
            ON earnings_calendar(ticker, event_date);
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_dividends (
              ticker TEXT NOT NULL,
              ex_dividend_date TEXT NOT NULL,
              label TEXT,
              adj_dividend REAL,
              dividend REAL,
              record_date TEXT,
              payment_date TEXT,
              declaration_date TEXT,
              payload_json TEXT,
              updated_at TEXT NOT NULL,
              UNIQUE(ticker, ex_dividend_date)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_hist_div_ticker_date
            ON historical_dividends(ticker, ex_dividend_date);
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_splits (
              ticker TEXT NOT NULL,
              split_date TEXT NOT NULL,
              label TEXT,
              numerator REAL,
              denominator REAL,
              payload_json TEXT,
              updated_at TEXT NOT NULL,
              UNIQUE(ticker, split_date)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_hist_splits_ticker_date
            ON historical_splits(ticker, split_date);
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS economic_calendar (
              event_date TEXT NOT NULL,
              event_name TEXT NOT NULL DEFAULT '',
              country TEXT NOT NULL DEFAULT '',
              currency TEXT,
              previous REAL,
              estimate REAL,
              actual REAL,
              impact TEXT,
              payload_json TEXT,
              updated_at TEXT NOT NULL,
              UNIQUE(event_date, event_name, country)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_econ_cal_date
            ON economic_calendar(event_date);
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mergers_acquisitions (
              transaction_url TEXT NOT NULL DEFAULT '',
              event_date TEXT,
              title TEXT,
              acquirer_ticker TEXT,
              acquirer_name TEXT,
              target_ticker TEXT,
              target_name TEXT,
              payload_json TEXT,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (transaction_url)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ma_date
            ON mergers_acquisitions(event_date);
            """
        )

        # ── Compliance / Disclosure tables ──────────────────────────────────

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS company_executives (
              ticker TEXT NOT NULL,
              name TEXT NOT NULL DEFAULT '',
              title TEXT,
              pay REAL,
              currency_pay TEXT,
              gender TEXT,
              year_born INTEGER,
              title_since TEXT,
              payload_json TEXT,
              updated_at TEXT NOT NULL,
              UNIQUE(ticker, name)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_executives_ticker
            ON company_executives(ticker);
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS employee_count (
              ticker TEXT NOT NULL,
              period_of_report TEXT NOT NULL,
              form_type TEXT,
              filing_date TEXT,
              employee_count INTEGER,
              payload_json TEXT,
              updated_at TEXT NOT NULL,
              UNIQUE(ticker, period_of_report)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_employee_count_ticker
            ON employee_count(ticker, period_of_report);
            """
        )

        # ── News table ───────────────────────────────────────────────────────
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_news (
              ticker TEXT NOT NULL,
              published_date TEXT NOT NULL,
              publisher TEXT,
              title TEXT NOT NULL DEFAULT '',
              site TEXT,
              text TEXT,
              url TEXT,
              image_url TEXT,
              payload_json TEXT,
              updated_at TEXT NOT NULL,
              UNIQUE(ticker, published_date, url)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stock_news_ticker_date
            ON stock_news(ticker, published_date);
            """
        )

        conn.commit()

    return db_path


def main() -> None:
    db = init_db()
    print("Connecting to DB...")
    print(f"Initialized SQLite DB at: {db}")
    print("Schema is ready.")


if __name__ == "__main__":
    main()
