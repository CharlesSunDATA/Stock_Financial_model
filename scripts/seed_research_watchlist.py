"""
Seed a compact research watchlist for CI and scheduled delta jobs.

GitHub Actions starts from a fresh SQLite cache when no prior cache is
available. This script ensures event and news jobs have ticker targets.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.init_db import init_db
except ModuleNotFoundError:
    from init_db import init_db  # type: ignore


RESEARCH_CATEGORIES: dict[str, list[str]] = {
    "GPU / CPU / ASIC": ["NVDA", "AMD", "INTC", "AVGO", "MRVL", "TSM"],
    "Memory / HBM / Storage": ["MU", "WDC", "STX", "SSNLF", "HXSCL"],
    "Optical Communications / Photonics": ["LITE", "COHR", "FN", "AAOI", "CIEN", "GLW", "NOK"],
    "Connector / Interconnect": ["APH", "TEL", "CRDO", "ALAB"],
    "AI Server / ODM / EMS": ["DELL", "SMCI", "HPE", "CLS", "JBL", "FLEX", "QUCCF", "WICOF", "HNHPF"],
    "Networking": ["ANET", "CSCO", "AVGO", "MRVL", "CIEN"],
    "Power / Cooling / Data Center Equipment": [
        "VRT",
        "ETN",
        "GEV",
        "PWR",
        "EME",
        "FIX",
        "TT",
        "CARR",
        "JCI",
        "NVT",
        "BE",
        "GNRC",
        "CMI",
        "CAT",
    ],
    "Semiconductor Equipment / Packaging": ["ASML", "AMAT", "LRCX", "KLAC", "TER", "AEHR"],
    "Cloud Demand Drivers": ["MSFT", "AMZN", "GOOGL", "META", "ORCL"],
}


NAME_TO_TICKER = {
    "SAMSUNG": "SSNLF",
    "SAMSUNG ELECTRONICS": "SSNLF",
    "SK HYNIX": "HXSCL",
    "QUANTA": "QUCCF",
    "QUANTA COMPUTER": "QUCCF",
    "WISTRON": "WICOF",
    "FOXCONN": "HNHPF",
    "HON HAI": "HNHPF",
}


def default_tickers() -> list[str]:
    tickers: list[str] = []
    for symbols in RESEARCH_CATEGORIES.values():
        for symbol in symbols:
            if symbol not in tickers:
                tickers.append(symbol)
    return tickers


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_tickers(raw: str) -> list[str]:
    tickers = []
    for token in raw.replace("\n", ",").split(","):
        ticker = NAME_TO_TICKER.get(token.strip().upper(), token.strip().upper())
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def seed_watchlist(watchlist_name: str, tickers: list[str], *, replace: bool) -> int:
    db_path = init_db()
    now = utc_now_iso()
    rows = [(watchlist_name, ticker, now) for ticker in tickers]

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        if replace:
            conn.execute("DELETE FROM fmp_watchlist WHERE watchlist_name = ?", (watchlist_name,))
        conn.executemany(
            """
            INSERT OR REPLACE INTO fmp_watchlist (watchlist_name, ticker, updated_at)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed fmp_watchlist with research tickers.")
    parser.add_argument("--watchlist", default="default")
    parser.add_argument(
        "--tickers",
        default=",".join(default_tickers()),
        help="Comma-separated tickers to seed.",
    )
    parser.add_argument("--append", action="store_true", help="Append instead of replacing the watchlist.")
    args = parser.parse_args()

    tickers = parse_tickers(args.tickers)
    count = seed_watchlist(args.watchlist, tickers, replace=not args.append)
    print(f"Seeded watchlist '{args.watchlist}' with {count} tickers.")


if __name__ == "__main__":
    main()
