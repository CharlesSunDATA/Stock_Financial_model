#!/bin/bash
# Weekly fundamental update — runs via launchd every Sunday at 08:00

set -euo pipefail

REPO="/Users/charlessun/Desktop/Google Drive/Python/Stock_Financial_model"
PYTHON="$REPO/.venv/bin/python3"
DB="$REPO/data/quant_data.db"
LOG="$REPO/logs/weekly_update.log"

mkdir -p "$REPO/logs"

exec >> "$LOG" 2>&1
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

cd "$REPO"

# Read watchlist tickers from DB
TICKERS=$(sqlite3 "$DB" "SELECT ticker FROM fmp_watchlist ORDER BY ticker;" | tr '\n' ' ')

if [ -z "$TICKERS" ]; then
    echo "No tickers in fmp_watchlist. Exiting."
    exit 1
fi

TICKER_COUNT=$(echo "$TICKERS" | wc -w | tr -d ' ')
echo "Watchlist: $TICKER_COUNT tickers"

echo "[1/3] Updating fundamentals, ratios, estimates..."
"$PYTHON" -m scripts.update_fmp $TICKERS

echo "[2/3] Updating income/CF/BS statements..."
"$PYTHON" -m scripts.update_fundamentals_fmp $TICKERS

echo "[3/3] Updating valuation + analyst data..."
"$PYTHON" -m scripts.update_valuation_analyst_fmp --watchlist default

echo "Done."
