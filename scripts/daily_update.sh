#!/bin/bash
# Daily database update script — runs via launchd each morning

set -euo pipefail

REPO="/Users/charlessun/Desktop/Google Drive/Python/Stock_Financial_model"
PYTHON="$REPO/.venv/bin/python3"
LOG="$REPO/logs/daily_update.log"

mkdir -p "$REPO/logs"

exec >> "$LOG" 2>&1
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

cd "$REPO"

echo "[1/3] Updating EOD prices..."
"$PYTHON" -m scripts.update_prices_eod --mode per-symbol --universe profile --calls-per-minute 240 --max-retries 3 --fallback-source alpha-vantage --alpha-calls-per-minute 5

echo "[2/3] Updating calendars..."
"$PYTHON" -m scripts.update_calendars_fmp

echo "[3/3] Updating news..."
"$PYTHON" -m scripts.update_news_fmp

echo "Done."
