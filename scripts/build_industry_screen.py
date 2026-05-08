#!/usr/bin/env python3
"""Build the stock screening and industry ranking tables."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.init_db import default_db_path, init_db
    from scripts.market_loader_fmp import (
        FMP_API_KEY,
        FmpClient,
        fetch_etf_equity_holdings_symbols,
        fetch_sp500_constituents,
        replace_watchlist,
    )
except ModuleNotFoundError:  # pragma: no cover
    from init_db import default_db_path, init_db  # type: ignore
    from market_loader_fmp import (  # type: ignore
        FMP_API_KEY,
        FmpClient,
        fetch_etf_equity_holdings_symbols,
        fetch_sp500_constituents,
        replace_watchlist,
    )

from utils.industry_screener import ScreenerInputs, export_screening_results, write_screening_tables


def _refresh_core_universe(db_path: Path, calls_per_minute: int) -> None:
    if FMP_API_KEY.strip() in ("", "YOUR_API_KEY_HERE"):
        raise SystemExit("FMP_API_KEY is not set. Configure .streamlit/secrets.toml or environment first.")

    client = FmpClient(FMP_API_KEY, calls_per_minute=int(calls_per_minute))
    sp500 = fetch_sp500_constituents(client)
    ndx = fetch_etf_equity_holdings_symbols(client, "QQQ")
    combined = sorted(set(sp500).union(ndx))

    with sqlite3.connect(str(db_path)) as conn:
        replace_watchlist(conn, watchlist_name="sp500", tickers=sp500)
        replace_watchlist(conn, watchlist_name="ndx", tickers=ndx)
        replace_watchlist(conn, watchlist_name="sp500_ndx", tickers=combined)
        conn.commit()

    print(f"Saved sp500 watchlist: {len(sp500):,} tickers")
    print(f"Saved ndx watchlist: {len(ndx):,} tickers")
    print(f"Saved sp500_ndx watchlist: {len(combined):,} tickers")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stock screening and industry ranking outputs.")
    parser.add_argument("--db", default=str(default_db_path()), help="SQLite database path.")
    parser.add_argument(
        "--watchlists",
        default="sp500_ndx,sp500,ndx,nasdaq100,QQQ",
        help="Comma-separated fmp_watchlist names to include, in priority order.",
    )
    parser.add_argument("--min-price-rows", type=int, default=200)
    parser.add_argument("--refresh-universe", action="store_true", help="Refresh S&P 500 and Nasdaq 100 watchlists from FMP.")
    parser.add_argument("--calls-per-minute", type=int, default=280)
    parser.add_argument("--export", action="store_true", help="Write CSV and HTML dashboard files.")
    parser.add_argument(
        "--output-dir",
        default="industry_screen_exports",
        help="Directory for CSV and HTML exports. Relative paths are resolved from the project root.",
    )
    args = parser.parse_args()

    db_path = init_db(Path(args.db).expanduser())
    if args.refresh_universe:
        _refresh_core_universe(db_path, int(args.calls_per_minute))

    watchlists = tuple(x.strip() for x in str(args.watchlists).split(",") if x.strip())
    inputs = ScreenerInputs(watchlist_names=watchlists, min_price_rows=max(20, int(args.min_price_rows)))

    if args.export:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = Path(__file__).resolve().parents[1] / output_dir
        paths = export_screening_results(db_path, output_dir, inputs)
        if not paths:
            raise SystemExit("No screening rows were generated. Check watchlists and source data.")
        for label, path in paths.items():
            print(f"{label}: {path}")
        return

    stock_df, industry_df = write_screening_tables(db_path, inputs)
    if stock_df.empty:
        raise SystemExit("No screening rows were generated. Check watchlists and source data.")
    print(f"Stocks ranked: {len(stock_df):,}")
    print(f"Industries ranked: {len(industry_df):,}")
    print(f"Best industry: {industry_df.iloc[0]['industry']} ({industry_df.iloc[0]['industry_score']:.0f})")
    print(f"Best stock: {stock_df.iloc[0]['ticker']} ({stock_df.iloc[0]['total_score']:.0f})")


if __name__ == "__main__":
    main()
