#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "Starting Streamlit with: $(pwd)/app.py" >&2
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m streamlit run app.py "$@"
fi
exec python3 -m streamlit run app.py "$@"

